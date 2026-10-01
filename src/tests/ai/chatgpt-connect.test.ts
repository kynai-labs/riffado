/**
 * ChatGPT sign-in completion: the pending sign-in must belong to the
 * caller and be fresh, and the pasted URL must match the started attempt.
 */

import { describe, expect, it, vi } from "vitest";

vi.mock("@/lib/env", () => ({
    env: {
        ENCRYPTION_KEY:
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    },
}));

vi.mock("@/db", () => ({
    db: { select: vi.fn(), update: vi.fn(), transaction: vi.fn() },
}));

import {
    completeChatGptSignIn,
    PENDING_SIGN_IN_TTL_MS,
    type PendingSignIn,
    sealPendingSignIn,
    unsealPendingSignIn,
} from "@/lib/ai/chatgpt/connect";
import { ErrorCode } from "@/lib/errors";

function pending(overrides: Partial<PendingSignIn> = {}): PendingSignIn {
    return {
        userId: "user-1",
        state: "state-1",
        nonce: "nonce-1",
        codeVerifier: "verifier",
        hostId: "urn:uuid:h",
        clientId: null,
        createdAt: Date.now(),
        ...overrides,
    };
}

describe("pending sign-in cookie", () => {
    it("round-trips for the same user", () => {
        const p = pending();
        expect(unsealPendingSignIn(sealPendingSignIn(p), "user-1")).toEqual(p);
    });

    it("rejects another user's sign-in", () => {
        expect(() =>
            unsealPendingSignIn(sealPendingSignIn(pending()), "user-2"),
        ).toThrow(/expired/);
    });

    it("rejects an expired sign-in", () => {
        const p = pending({
            createdAt: Date.now() - PENDING_SIGN_IN_TTL_MS - 1,
        });
        expect(() =>
            unsealPendingSignIn(sealPendingSignIn(p), "user-1"),
        ).toThrow(/expired/);
    });

    it("rejects a missing or tampered cookie", () => {
        expect(() => unsealPendingSignIn(null, "user-1")).toThrow();
        expect(() => unsealPendingSignIn("aa:bb:cc", "user-1")).toThrow();
    });
});

describe("completeChatGptSignIn", () => {
    it("rejects a URL from a different sign-in attempt", async () => {
        await expect(
            completeChatGptSignIn({
                userId: "user-1",
                pending: pending(),
                callbackUrl:
                    "http://127.0.0.1:1455/auth/callback?code=c&state=other&client_id=oaiapp_1",
            }),
        ).rejects.toMatchObject({ code: ErrorCode.INVALID_INPUT });
    });

    it("requires the issued client id on a first registration", async () => {
        await expect(
            completeChatGptSignIn({
                userId: "user-1",
                pending: pending(),
                callbackUrl:
                    "http://127.0.0.1:1455/auth/callback?code=c&state=state-1",
            }),
        ).rejects.toThrow(/client_id/);
    });

    it("rejects a malformed paste before calling OpenAI", async () => {
        const fetchSpy = vi.spyOn(globalThis, "fetch");
        await expect(
            completeChatGptSignIn({
                userId: "user-1",
                pending: pending(),
                callbackUrl: "hello",
            }),
        ).rejects.toMatchObject({ code: ErrorCode.INVALID_INPUT });
        expect(fetchSpy).not.toHaveBeenCalled();
        fetchSpy.mockRestore();
    });
});
