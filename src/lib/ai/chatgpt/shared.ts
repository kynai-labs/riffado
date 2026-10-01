/**
 * Dependency-light constants and helpers for the ChatGPT plan-usage
 * provider, safe to import from anywhere (no DB, no encryption).
 */

import { AppError, ErrorCode } from "@/lib/errors";

export const CHATGPT_PROVIDER_NAME = "ChatGPT";
export const CHATGPT_BASE_URL = "https://api.openai.com/v1";

export function isChatGptProvider(provider: string): boolean {
    return provider === CHATGPT_PROVIDER_NAME;
}

export function reconnectError(): AppError {
    return new AppError(
        ErrorCode.AI_PROVIDER_NOT_CONFIGURED,
        "Your ChatGPT sign-in has expired. Reconnect ChatGPT in Settings → Providers.",
        400,
        { provider: CHATGPT_PROVIDER_NAME, reconnect: true },
    );
}
