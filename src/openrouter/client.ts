import { requestUrl, RequestUrlResponse } from "obsidian";
import type { OpenRouterModel, OpenRouterMessage, ModelListResponse } from "./types";

// NOTE: OpenRouter does not send CORS headers that permit Obsidian's requestUrl
// to stream Server-Sent Events, so the streaming path intentionally uses fetch
// (which is what SSE streaming requires here).

const MODELS_URL = "https://openrouter.ai/api/v1/models";
const CHAT_URL = "https://openrouter.ai/api/v1/chat/completions";
const MODELS_CACHE_TTL_MS = 60 * 60 * 1000;

let cachedModels: { apiKey: string; data: OpenRouterModel[]; fetchedAt: number } | null = null;

export async function fetchModels(apiKey: string): Promise<OpenRouterModel[]> {
  const now = Date.now();
  if (cachedModels && cachedModels.apiKey === apiKey && (now - cachedModels.fetchedAt) < MODELS_CACHE_TTL_MS) {
    return cachedModels.data;
  }
  const response: RequestUrlResponse = await requestUrl({
    url: MODELS_URL,
    method: "GET",
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
  });

  const json = typeof response.json === "object" && response.json !== null
    ? response.json
    : JSON.parse(response.text);

  if (json && typeof json === "object" && "error" in json) {
    const err = (json as Record<string, unknown>).error as Record<string, unknown> | undefined;
    throw new Error(err?.message as string ?? JSON.stringify(err));
  }

  const body = json as ModelListResponse;
  const models = (body.data ?? []).map((m) => ({
    id: m.id,
    name: m.name,
    description: m.description,
    pricing: {
      prompt: parseFloat(m.pricing.prompt),
      completion: parseFloat(m.pricing.completion),
    },
    context_length: m.context_length,
    isFree: parseFloat(m.pricing.prompt) === 0 && parseFloat(m.pricing.completion) === 0,
  }));
  cachedModels = { apiKey, data: models, fetchedAt: now };
  return models;
}

export async function sendChatStream(
  apiKey: string,
  model: string,
  messages: OpenRouterMessage[],
  onToken: (token: string) => void,
  signal?: AbortSignal,
): Promise<string> {
  // Internal controller so we can abort on idle timeout while still honoring a
  // caller-provided signal (user-triggered abort keeps working).
  const internalController = new AbortController();
  const onUserAbort = () => internalController.abort();
  if (signal) {
    if (signal.aborted) internalController.abort();
    else signal.addEventListener("abort", onUserAbort, { once: true });
  }

  // Idle watchdog: if no chunk arrives for IDLE_TIMEOUT_MS, abort the request so a
  // hung connection cannot block forever.
  const IDLE_TIMEOUT_MS = 60_000;
  let lastActivity = Date.now();
  const idleTimer = setInterval(() => {
    if (Date.now() - lastActivity > IDLE_TIMEOUT_MS) {
      internalController.abort();
    }
  }, 5_000);

  // Generous cap on the unterminated-line buffer to avoid unbounded growth on a
  // newline-free response (defensive).
  const MAX_BUFFER_BYTES = 10 * 1024 * 1024;

  const cleanup = () => {
    clearInterval(idleTimer);
    if (signal) signal.removeEventListener("abort", onUserAbort);
  };

  let response: Response | undefined;
  try {
    response = await fetch(CHAT_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        model,
        messages,
        stream: true,
        temperature: 0.3,
        max_tokens: 2048,
      }),
      signal: internalController.signal,
    });

    if (!response.ok) {
      let errorMsg = `OpenRouter returned HTTP ${response.status}`;
      try {
        const errBody = await response.json();
        if (errBody?.error?.message) errorMsg = errBody.error.message;
        else if (errBody?.error) errorMsg = JSON.stringify(errBody.error);
      } catch { /* ignore */ }
      throw Object.assign(new Error(errorMsg), { status: response.status });
    }

    if (!response.body) {
      throw new Error("OpenRouter did not return a response body.");
    }

    let fullContent = "";
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const emit = (data: string) => {
      if (data === "[DONE]") return;
      try {
        const chunk = JSON.parse(data);
        const token = chunk.choices?.[0]?.delta?.content ?? "";
        if (token) {
          fullContent += token;
          onToken(token);
        }
      } catch {
        // skip malformed chunks
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      lastActivity = Date.now();
      buffer += decoder.decode(value, { stream: true });

      if (buffer.length > MAX_BUFFER_BYTES) {
        console.warn(
          `OpenRouter stream buffer exceeded ${MAX_BUFFER_BYTES} bytes; ` +
          `dropping excess to bound memory usage.`,
        );
        buffer = buffer.slice(-MAX_BUFFER_BYTES);
      }

      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        emit(line.slice(6).trim());
      }
    }

    // Trailing buffer flush: emit any unterminated final line left in the buffer.
    if (buffer.trim()) {
      const trailing = buffer.trim();
      if (trailing.startsWith("data: ")) emit(trailing.slice(6).trim());
    }

    cleanup();
    return fullContent;
  } catch (err) {
    cleanup();
    const userAborted = !!signal?.aborted;
    const aborted = internalController.signal.aborted;
    if (aborted && !userAborted) {
      // Aborted by our idle watchdog (or a fetch-level abort), not by the user.
      const e = new Error(
        `OpenRouter stream timed out: no data received for ${IDLE_TIMEOUT_MS / 1000}s.`,
      );
      (e as Error & { code?: string }).code = "ETIMEDOUT";
      throw e;
    }
    if (err instanceof Error) {
      const status = (err as Error & { status?: number }).status;
      const code = (err as Error & { code?: string }).code;
      if (status !== undefined || code !== undefined) {
        const detail = [code, status !== undefined ? `HTTP ${status}` : undefined]
          .filter(Boolean)
          .join(" ");
        const e = new Error(`${err.message} (${detail})`);
        if (status !== undefined) (e as Error & { status?: number }).status = status;
        if (code !== undefined) (e as Error & { code?: string }).code = code;
        throw e;
      }
      throw err;
    }
    throw new Error(`OpenRouter stream failed: ${String(err)}`);
  }
}
