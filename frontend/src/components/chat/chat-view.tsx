"use client";

import { useEffect, useRef, useState } from "react";
import { ArrowUp, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { ChatMode, Citation, HistoryMessage } from "@/lib/api";
import { streamChat, type ChatEvent } from "@/lib/sse";
import { CitationPanel } from "./citation-panel";
import { MessageBubble } from "./message-bubble";
import type { ChatMessage } from "./types";

const MODES: { value: ChatMode; label: string }[] = [
  { value: "agentic", label: "Agentic" },
  { value: "hybrid_rerank", label: "Hybrid + rerank" },
  { value: "hybrid", label: "Hybrid" },
  { value: "vector", label: "Vector" },
];

const HISTORY_TURNS = 6;

let idCounter = 0;
const newId = () => `m${Date.now()}-${idCounter++}`;

/** Apply one SSE event to the assistant message being streamed. */
function applyEvent(msg: ChatMessage, event: ChatEvent): ChatMessage {
  switch (event.type) {
    case "step":
      return { ...msg, steps: [...msg.steps, event] };
    case "token":
      return { ...msg, content: msg.content + event.content };
    case "citations":
      return { ...msg, citations: event.citations };
    case "done": {
      const tokens =
        event.total_tokens ?? (event.prompt_tokens ?? 0) + (event.completion_tokens ?? 0);
      const cacheHit = event.cache_hit ?? msg.steps.some((s) => s.cache_hit);
      return { ...msg, meta: { latencyMs: event.latency_ms, tokens, cacheHit } };
    }
    case "error":
      return { ...msg, error: event.content };
  }
}

export function ChatView({
  path,
  token,
  suggestions,
  emptyTitle,
  emptyText,
}: {
  path: "/chat" | "/chat/demo";
  token?: string | null;
  suggestions: string[];
  emptyTitle: string;
  emptyText: string;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<ChatMode>("agentic");
  const [busy, setBusy] = useState(false);
  const [openCitation, setOpenCitation] = useState<Citation | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // Stop an in-flight stream if the view unmounts.
  useEffect(() => () => abortRef.current?.abort(), []);

  async function send(text: string) {
    const query = text.trim();
    if (!query || busy) return;

    const history: HistoryMessage[] = messages
      .filter((m) => m.content && !m.error)
      .slice(-HISTORY_TURNS)
      .map((m) => ({ role: m.role, content: m.content }));

    const assistantId = newId();
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: "user", content: query, steps: [], citations: [] },
      { id: assistantId, role: "assistant", content: "", steps: [], citations: [], streaming: true },
    ]);
    setInput("");
    setBusy(true);

    const update = (fn: (m: ChatMessage) => ChatMessage) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? fn(m) : m)));

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat({
        path,
        query,
        mode,
        history,
        token,
        signal: controller.signal,
        onEvent: (event) => update((m) => applyEvent(m, event)),
      });
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        update((m) => ({ ...m, error: (err as Error).message }));
      }
    } finally {
      update((m) => ({ ...m, streaming: false }));
      setBusy(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4">
      <div className="flex-1 space-y-5 py-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center pt-10 text-center sm:pt-16">
            <h1 className="text-2xl font-semibold tracking-tight">{emptyTitle}</h1>
            <p className="mt-2 max-w-md text-sm text-muted-foreground">{emptyText}</p>
            <div className="mt-6 flex max-w-xl flex-wrap justify-center gap-2">
              {suggestions.map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => send(s)}
                  className="rounded-full border px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m) => <MessageBubble key={m.id} message={m} onCite={setOpenCitation} />)
        )}
        <div ref={bottomRef} />
      </div>

      <form
        className="sticky bottom-0 bg-background pb-4 pt-2"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <div className="rounded-2xl border bg-card p-2 shadow-sm focus-within:ring-2 focus-within:ring-ring/40">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
            placeholder="Ask a question about the documents…"
            rows={1}
            className="max-h-40 min-h-10 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0 dark:bg-transparent"
          />
          <div className="mt-1 flex items-center justify-between gap-2">
            <label className="flex items-center gap-2 text-xs text-muted-foreground">
              Mode
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as ChatMode)}
                className="rounded-md border bg-background px-2 py-1 text-xs text-foreground"
              >
                {MODES.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            {busy ? (
              <Button type="button" size="icon" variant="secondary" aria-label="Stop" onClick={() => abortRef.current?.abort()}>
                <Square className="size-3.5" />
              </Button>
            ) : (
              <Button type="submit" size="icon" aria-label="Send" disabled={!input.trim()}>
                <ArrowUp />
              </Button>
            )}
          </div>
        </div>
        <p className="mt-2 text-center text-[0.7rem] text-muted-foreground">
          Answers come only from the uploaded documents. First request may take ~30s on the free tier.
        </p>
      </form>

      <CitationPanel citation={openCitation} onClose={() => setOpenCitation(null)} />
    </div>
  );
}
