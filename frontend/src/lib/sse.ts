// Reads the backend's Server-Sent Events stream from a POST request.
// EventSource only supports GET, so we parse the stream from fetch() ourselves.

import { API_URL, ApiError, readError, type ChatMode, type Citation, type HistoryMessage } from "./api";

export type ChatEvent =
  | {
      type: "step";
      content: string;
      node?: string;
      route?: string;
      chunks_found?: number;
      relevant?: number;
      grounded?: boolean;
      rewritten_query?: string;
      cache_hit?: boolean;
    }
  | { type: "token"; content: string }
  | { type: "citations"; citations: Citation[] }
  | {
      type: "done";
      latency_ms: number;
      total_tokens?: number;
      prompt_tokens?: number;
      completion_tokens?: number;
      cache_hit?: boolean;
    }
  | { type: "error"; content: string };

export interface StreamChatOptions {
  path: "/chat" | "/chat/demo";
  query: string;
  mode: ChatMode;
  history: HistoryMessage[];
  token?: string | null;
  signal?: AbortSignal;
  onEvent: (event: ChatEvent) => void;
}

export async function streamChat(opts: StreamChatOptions): Promise<void> {
  let resp: Response;
  try {
    resp = await fetch(`${API_URL}${opts.path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(opts.token ? { Authorization: `Bearer ${opts.token}` } : {}),
      },
      body: JSON.stringify({ query: opts.query, mode: opts.mode, history: opts.history }),
      signal: opts.signal,
    });
  } catch (err) {
    if ((err as Error).name === "AbortError") return;
    throw new ApiError(0, "Can't reach the server. It may be waking up (free tier) — try again in ~30s.");
  }
  if (!resp.ok) throw await readError(resp);
  if (!resp.body) throw new ApiError(0, "Empty response stream");

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE messages are separated by a blank line.
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const message = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of message.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") return;
        try {
          opts.onEvent(JSON.parse(data) as ChatEvent);
        } catch {
          // ignore malformed event
        }
      }
    }
  }
}
