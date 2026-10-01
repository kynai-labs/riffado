/**
 * Connect / disconnect flow for the ChatGPT plan-usage provider, plus
 * the single entry point enhancement code calls to run a completion.
 */

import { and, eq } from "drizzle-orm";
import { db } from "@/db";
import { apiCredentials } from "@/db/schema";
import { decrypt, encrypt } from "@/lib/encryption";
import { AppError, ErrorCode } from "@/lib/errors";
import {
    applyTokenResponse,
    type ChatGptCredential,
    getChatGptAccessToken,
    parseCredential,
    serializeCredential,
} from "./credentials";
import {
    type ChatGptModel,
    createChatGptResponse,
    listChatGptModels,
} from "./inference";
import {
    assertPlanUsageGranted,
    buildAuthorizeUrl,
    ChatGptOAuthError,
    exchangeAuthorizationCode,
    generateHostId,
    generatePkce,
    parseCallbackUrl,
    randomToken,
    revokeRefreshToken,
    verifyIdToken,
} from "./oauth";
import { CHATGPT_BASE_URL, CHATGPT_PROVIDER_NAME } from "./shared";

/** How long a started sign-in stays valid. */
export const PENDING_SIGN_IN_TTL_MS = 10 * 60 * 1000;

/** Server-side state between "start" and "complete", sealed in a cookie. */
export interface PendingSignIn {
    userId: string;
    state: string;
    nonce: string;
    codeVerifier: string;
    hostId: string;
    /** Issued client id when re-authorizing an existing connection. */
    clientId: string | null;
    createdAt: number;
}

export function sealPendingSignIn(pending: PendingSignIn): string {
    return encrypt(JSON.stringify(pending));
}

export function unsealPendingSignIn(
    sealed: string | undefined | null,
    userId: string,
    now = Date.now(),
): PendingSignIn {
    const expired = new AppError(
        ErrorCode.INVALID_INPUT,
        "This ChatGPT sign-in has expired. Click “Sign in with ChatGPT” again.",
        400,
    );
    if (!sealed) throw expired;
    let pending: PendingSignIn;
    try {
        pending = JSON.parse(decrypt(sealed)) as PendingSignIn;
    } catch {
        throw expired;
    }
    if (
        pending.userId !== userId ||
        typeof pending.createdAt !== "number" ||
        now - pending.createdAt > PENDING_SIGN_IN_TTL_MS
    ) {
        throw expired;
    }
    return pending;
}

async function findChatGptRow(userId: string) {
    const [row] = await db
        .select()
        .from(apiCredentials)
        .where(
            and(
                eq(apiCredentials.userId, userId),
                eq(apiCredentials.provider, CHATGPT_PROVIDER_NAME),
            ),
        )
        .limit(1);
    return row;
}

function tryParse(encrypted: string): ChatGptCredential | null {
    try {
        return parseCredential(encrypted);
    } catch {
        return null;
    }
}

/**
 * Begin a sign-in. Re-uses the issued client id and host id of an
 * existing connection, as OpenAI asks, so reconnecting doesn't register
 * a new client every time.
 */
export async function startChatGptSignIn(userId: string): Promise<{
    authorizeUrl: string;
    pending: PendingSignIn;
}> {
    const existingRow = await findChatGptRow(userId);
    const existing = existingRow ? tryParse(existingRow.apiKey) : null;

    const { verifier, challenge } = generatePkce();
    const pending: PendingSignIn = {
        userId,
        state: randomToken(),
        nonce: randomToken(),
        codeVerifier: verifier,
        hostId: existing?.hostId ?? generateHostId(),
        clientId: existing?.clientId ?? null,
        createdAt: Date.now(),
    };

    const authorizeUrl = buildAuthorizeUrl({
        clientId: pending.clientId,
        hostId: pending.hostId,
        state: pending.state,
        nonce: pending.nonce,
        codeChallenge: challenge,
        idTokenHint: existing?.idToken ?? null,
        loginHint: existing?.email ?? null,
    });

    return { authorizeUrl, pending };
}

function toAppError(error: unknown): AppError {
    if (error instanceof AppError) return error;
    if (error instanceof ChatGptOAuthError) {
        return new AppError(
            ErrorCode.INVALID_INPUT,
            error.code === "insufficient_scope"
                ? error.message
                : `ChatGPT sign-in failed: ${error.message}`,
            400,
            { provider: CHATGPT_PROVIDER_NAME, upstreamCode: error.code },
        );
    }
    return new AppError(
        ErrorCode.AI_PROVIDER_API_ERROR,
        "ChatGPT sign-in failed. Try again.",
        502,
        { provider: CHATGPT_PROVIDER_NAME },
    );
}

export interface ConnectResult {
    id: string;
    email: string | null;
    defaultModel: string | null;
    models: ChatGptModel[];
}

/**
 * Finish a sign-in from the loopback URL the user pasted back: verify
 * state, exchange the code, validate the ID token and granted scopes,
 * then create or update the user's ChatGPT provider row.
 */
export async function completeChatGptSignIn(args: {
    userId: string;
    pending: PendingSignIn;
    callbackUrl: string;
}): Promise<ConnectResult> {
    const { userId, pending } = args;

    const parsed = parseCallbackUrl(args.callbackUrl);
    if (!parsed.ok) {
        throw new AppError(ErrorCode.INVALID_INPUT, parsed.message, 400, {
            field: "callbackUrl",
        });
    }
    if (parsed.state !== pending.state) {
        throw new AppError(
            ErrorCode.INVALID_INPUT,
            "That URL belongs to a different sign-in attempt. Click “Sign in with ChatGPT” again and paste the new URL.",
            400,
            { field: "callbackUrl" },
        );
    }

    // A new registration returns the issued client id on the callback.
    const clientId = parsed.clientId ?? pending.clientId;
    if (!clientId) {
        throw new AppError(
            ErrorCode.INVALID_INPUT,
            "The URL is missing the client_id OpenAI issued. Copy the full address, including everything after the ?.",
            400,
            { field: "callbackUrl" },
        );
    }

    let credential: ChatGptCredential;
    let models: ChatGptModel[];
    try {
        const tokens = await exchangeAuthorizationCode({
            clientId,
            code: parsed.code,
            codeVerifier: pending.codeVerifier,
        });
        assertPlanUsageGranted(tokens.scope);
        if (!tokens.id_token) {
            throw new ChatGptOAuthError(
                "No ID token was returned",
                "invalid_id_token",
                400,
            );
        }
        const claims = await verifyIdToken(tokens.id_token, {
            clientId,
            nonce: pending.nonce,
        });
        credential = applyTokenResponse(
            {
                v: 1,
                clientId,
                hostId: pending.hostId,
                subject: claims.sub,
                email: claims.email,
                idToken: tokens.id_token,
            },
            tokens,
        );
        models = await listChatGptModels(credential.accessToken);
    } catch (error) {
        throw toAppError(error);
    }

    const existing = await findChatGptRow(userId);
    const keepModel =
        existing?.defaultModel &&
        models.some((m) => m.slug === existing.defaultModel)
            ? existing.defaultModel
            : null;
    const defaultModel = keepModel ?? models[0]?.slug ?? null;
    const sealed = serializeCredential(credential);

    const id = await db.transaction(async (tx) => {
        if (existing) {
            await tx
                .update(apiCredentials)
                .set({
                    apiKey: sealed,
                    baseUrl: CHATGPT_BASE_URL,
                    defaultModel,
                    isDefaultTranscription: false,
                    updatedAt: new Date(),
                })
                .where(
                    and(
                        eq(apiCredentials.id, existing.id),
                        eq(apiCredentials.userId, userId),
                    ),
                );
            return existing.id;
        }

        // First connection: become the enhancement default unless the
        // user already picked one.
        const [currentDefault] = await tx
            .select({ id: apiCredentials.id })
            .from(apiCredentials)
            .where(
                and(
                    eq(apiCredentials.userId, userId),
                    eq(apiCredentials.isDefaultEnhancement, true),
                ),
            )
            .limit(1);

        const [inserted] = await tx
            .insert(apiCredentials)
            .values({
                userId,
                provider: CHATGPT_PROVIDER_NAME,
                apiKey: sealed,
                baseUrl: CHATGPT_BASE_URL,
                defaultModel,
                isDefaultTranscription: false,
                isDefaultEnhancement: !currentDefault,
            })
            .returning({ id: apiCredentials.id });
        return inserted.id;
    });

    return { id, email: credential.email, defaultModel, models };
}

/** Revoke the stored refresh token before a ChatGPT row is deleted. */
export async function revokeChatGptCredential(
    encrypted: string,
): Promise<void> {
    const credential = tryParse(encrypted);
    if (!credential?.refreshToken) return;
    await revokeRefreshToken({
        clientId: credential.clientId,
        refreshToken: credential.refreshToken,
    });
}

/** Signed-in account email for display; null if unreadable. */
export function chatGptAccountEmail(encrypted: string): string | null {
    return tryParse(encrypted)?.email ?? null;
}

export async function listModelsForCredential(row: {
    id: string;
    userId: string;
    apiKey: string;
}): Promise<ChatGptModel[]> {
    const accessToken = await getChatGptAccessToken(row);
    return listChatGptModels(accessToken);
}

/**
 * Run a summary/title completion on the user's ChatGPT plan. Used by
 * the enhancement call sites in place of `chat.completions.create`.
 */
export async function runChatGptCompletion(
    row: {
        id: string;
        userId: string;
        apiKey: string;
        defaultModel: string | null;
    },
    args: { model?: string | null; instructions: string; input: string },
): Promise<{ text: string; model: string }> {
    const accessToken = await getChatGptAccessToken(row);
    let model = args.model || row.defaultModel;
    if (!model) {
        const models = await listChatGptModels(accessToken);
        model = models[0]?.slug ?? null;
    }
    if (!model) {
        throw new AppError(
            ErrorCode.AI_PROVIDER_NOT_CONFIGURED,
            "No ChatGPT models are available for this account.",
            400,
            { provider: CHATGPT_PROVIDER_NAME },
        );
    }
    const text = await createChatGptResponse({
        accessToken,
        model,
        instructions: args.instructions,
        input: args.input,
    });
    return { text, model };
}
