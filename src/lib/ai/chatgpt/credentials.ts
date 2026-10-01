/**
 * Storage and refresh for "Sign in with ChatGPT" credentials.
 *
 * A ChatGPT connection is an ordinary `api_credentials` row with
 * `provider = "ChatGPT"`. Instead of an API key, the encrypted `api_key`
 * column holds a JSON document with the issued OAuth client id, the
 * stable host id, the validated identity and the token set. Reusing the
 * row (and its AES-256-GCM envelope) keeps provider selection, defaults
 * and deletion working unchanged, and needs no schema migration.
 */

import { and, eq } from "drizzle-orm";
import { db } from "@/db";
import { apiCredentials } from "@/db/schema";
import { decrypt, encrypt } from "@/lib/encryption";
import { AppError, ErrorCode } from "@/lib/errors";
import {
    isReauthRequiredError,
    refreshAccessToken,
    type TokenResponse,
} from "./oauth";
import { CHATGPT_PROVIDER_NAME, reconnectError } from "./shared";

export interface ChatGptCredential {
    v: 1;
    /** Issued `oaiapp_...` client id, reused for every later sign-in. */
    clientId: string;
    /** Stable `ext_agent_host_id` for this Riffado instance + user. */
    hostId: string;
    /** Validated ID token subject. */
    subject: string;
    email: string | null;
    /** Kept for `id_token_hint` on re-authorization. */
    idToken: string | null;
    accessToken: string;
    refreshToken: string | null;
    /** Access token expiry, epoch ms. */
    expiresAt: number;
    /** Don't refresh before this (epoch ms) unless the token expired. */
    earliestRefreshAt: number | null;
    scopes: string[];
}

export function serializeCredential(credential: ChatGptCredential): string {
    return encrypt(JSON.stringify(credential));
}

export function parseCredential(encrypted: string): ChatGptCredential {
    let parsed: unknown;
    try {
        parsed = JSON.parse(decrypt(encrypted));
    } catch {
        throw reconnectError();
    }
    const c = parsed as Partial<ChatGptCredential> | null;
    if (
        !c ||
        c.v !== 1 ||
        typeof c.clientId !== "string" ||
        typeof c.hostId !== "string" ||
        typeof c.accessToken !== "string" ||
        typeof c.expiresAt !== "number"
    ) {
        throw reconnectError();
    }
    return c as ChatGptCredential;
}

/** `earliest_refresh_at` unit isn't documented; accept seconds or ms. */
function toEpochMs(value: number | undefined): number | null {
    if (typeof value !== "number" || !Number.isFinite(value)) return null;
    return value > 1e12 ? value : value * 1000;
}

/** Merge a token endpoint response into a stored credential. */
export function applyTokenResponse(
    base: Omit<
        ChatGptCredential,
        | "accessToken"
        | "refreshToken"
        | "expiresAt"
        | "earliestRefreshAt"
        | "scopes"
    > &
        Partial<Pick<ChatGptCredential, "refreshToken" | "scopes">>,
    tokens: TokenResponse,
    now = Date.now(),
): ChatGptCredential {
    const expiresIn =
        typeof tokens.expires_in === "number" && tokens.expires_in > 0
            ? tokens.expires_in
            : 3600;
    return {
        ...base,
        v: 1,
        idToken: tokens.id_token ?? base.idToken,
        accessToken: tokens.access_token,
        // Refresh tokens rotate; keep the old one only if none came back.
        refreshToken: tokens.refresh_token ?? base.refreshToken ?? null,
        expiresAt: now + expiresIn * 1000,
        earliestRefreshAt: toEpochMs(tokens.earliest_refresh_at),
        scopes: tokens.scope
            ? tokens.scope.split(/\s+/).filter(Boolean)
            : (base.scopes ?? []),
    };
}

/** Refresh when less than this much lifetime is left. */
const REFRESH_MARGIN_MS = 2 * 60 * 1000;

export function needsRefresh(
    credential: ChatGptCredential,
    now = Date.now(),
): boolean {
    if (credential.expiresAt - now > REFRESH_MARGIN_MS) return false;
    const expired = credential.expiresAt <= now;
    if (
        !expired &&
        credential.earliestRefreshAt !== null &&
        now < credential.earliestRefreshAt
    ) {
        return false;
    }
    return true;
}

async function persistCredential(
    row: { id: string; userId: string },
    credential: ChatGptCredential,
): Promise<void> {
    await db
        .update(apiCredentials)
        .set({ apiKey: serializeCredential(credential), updatedAt: new Date() })
        .where(
            and(
                eq(apiCredentials.id, row.id),
                eq(apiCredentials.userId, row.userId),
            ),
        );
}

/**
 * Refresh tokens rotate and OpenAI rejects a reused refresh token, so two
 * concurrent callers (auto-title and auto-summary right after a
 * transcription) must not both refresh. Serialize per credential row.
 */
const inflightRefresh = new Map<string, Promise<ChatGptCredential>>();

async function loadStoredCredential(row: {
    id: string;
    userId: string;
}): Promise<ChatGptCredential | null> {
    const [stored] = await db
        .select({ apiKey: apiCredentials.apiKey })
        .from(apiCredentials)
        .where(
            and(
                eq(apiCredentials.id, row.id),
                eq(apiCredentials.userId, row.userId),
            ),
        )
        .limit(1);
    return stored ? parseCredential(stored.apiKey) : null;
}

async function refreshAndPersist(
    row: { id: string; userId: string },
    staleCredential: ChatGptCredential,
): Promise<ChatGptCredential> {
    // The caller's row may predate a refresh another request already
    // persisted. Refreshing with that rotated-out token would trip
    // `refresh_token_reused` and kill the session, so re-read first.
    const credential = (await loadStoredCredential(row)) ?? staleCredential;
    if (!needsRefresh(credential)) return credential;
    if (!credential.refreshToken) throw reconnectError();

    let tokens: TokenResponse;
    try {
        tokens = await refreshAccessToken({
            clientId: credential.clientId,
            refreshToken: credential.refreshToken,
        });
    } catch (error) {
        if (isReauthRequiredError(error)) throw reconnectError();
        throw new AppError(
            ErrorCode.AI_PROVIDER_API_ERROR,
            "Couldn't refresh the ChatGPT sign-in. Try again in a moment.",
            502,
            { provider: CHATGPT_PROVIDER_NAME },
        );
    }

    const next = applyTokenResponse(credential, tokens);
    await persistCredential(row, next);
    return next;
}

/**
 * Return a usable access token for a ChatGPT credential row, refreshing
 * (and persisting the rotated tokens) when it's close to expiry.
 */
export async function getChatGptAccessToken(row: {
    id: string;
    userId: string;
    apiKey: string;
}): Promise<string> {
    const credential = parseCredential(row.apiKey);
    if (!needsRefresh(credential)) return credential.accessToken;

    let pending = inflightRefresh.get(row.id);
    if (!pending) {
        pending = refreshAndPersist(row, credential).finally(() => {
            inflightRefresh.delete(row.id);
        });
        inflightRefresh.set(row.id, pending);
    }
    const refreshed = await pending;
    return refreshed.accessToken;
}
