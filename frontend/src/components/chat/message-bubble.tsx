"use client";

import { Fragment } from "react";
import { AlertCircle, Clock, Coins, Zap } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { Citation } from "@/lib/api";
import { cn } from "@/lib/utils";
import { AgentSteps } from "./agent-steps";
import type { ChatMessage } from "./types";

// Citations appear inline as [doc_name:page], e.g. [leave_policy:2].
const CITATION_RE = /\[([^\]:\s]+):(\d+)\]/g;

function stem(filename: string) {
  return filename.replace(/\.[^.]+$/, "");
}

function findCitation(citations: Citation[], doc: string, page: number): Citation {
  return (
    citations.find((c) => stem(c.filename) === doc && c.page === page) ??
    citations.find((c) => stem(c.filename) === doc) ?? { filename: doc, page, snippet: "" }
  );
}

/** Render **bold** spans (the model uses them for key facts). */
function withBold(text: string, keyPrefix: string): React.ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={`${keyPrefix}-${i}`}>{part.slice(2, -2)}</strong>
    ) : (
      part
    ),
  );
}

/** Render answer text with inline citations turned into clickable chips. */
function AnswerText({
  text,
  citations,
  onCite,
}: {
  text: string;
  citations: Citation[];
  onCite: (c: Citation) => void;
}) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(CITATION_RE)) {
    const idx = m.index ?? 0;
    parts.push(...withBold(text.slice(last, idx), `t${idx}`));
    const doc = m[1];
    const page = Number(m[2]);
    parts.push(
      <button
        key={idx}
        type="button"
        onClick={() => onCite(findCitation(citations, doc, page))}
        className="mx-0.5 inline-flex items-center rounded-md border bg-muted px-1.5 py-px align-baseline font-mono text-[0.7rem] text-foreground hover:bg-accent"
      >
        {doc}:{page}
      </button>,
    );
    last = idx + m[0].length;
  }
  parts.push(...withBold(text.slice(last), "end"));
  return (
    <p className="whitespace-pre-wrap leading-relaxed">
      {parts.map((p, i) => (
        <Fragment key={i}>{p}</Fragment>
      ))}
    </p>
  );
}

export function MessageBubble({
  message,
  onCite,
}: {
  message: ChatMessage;
  onCite: (c: Citation) => void;
}) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-sm text-primary-foreground">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  const waiting = message.streaming && !message.content;

  return (
    <div className="flex flex-col gap-2">
      <div className="max-w-[92%] rounded-2xl rounded-bl-sm border bg-card px-4 py-3 text-sm">
        <AgentSteps steps={message.steps} streaming={message.streaming} />
        {waiting && (
          <div className="mt-2 flex items-center gap-1" aria-label="Thinking">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="size-1.5 animate-bounce rounded-full bg-muted-foreground"
                style={{ animationDelay: `${i * 150}ms` }}
              />
            ))}
          </div>
        )}
        {message.content && (
          <div className={cn(message.steps.length > 0 && "mt-2")}>
            <AnswerText text={message.content} citations={message.citations} onCite={onCite} />
          </div>
        )}
        {message.error && (
          <p className="mt-2 flex items-center gap-1.5 text-destructive">
            <AlertCircle className="size-4" /> {message.error}
          </p>
        )}
        {message.citations.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5 border-t pt-2.5">
            <span className="text-xs text-muted-foreground">Sources:</span>
            {message.citations.map((c) => (
              <button
                key={`${c.filename}:${c.page}`}
                type="button"
                onClick={() => onCite(c)}
                className="rounded-md border px-1.5 py-0.5 text-xs hover:bg-muted"
              >
                {c.filename} · p.{c.page}
              </button>
            ))}
          </div>
        )}
      </div>
      {message.meta && (
        <div className="flex flex-wrap items-center gap-2 pl-1 text-xs text-muted-foreground">
          {message.meta.cacheHit && (
            <Badge variant="secondary" className="gap-1">
              <Zap className="size-3" /> Cache hit
            </Badge>
          )}
          <span className="flex items-center gap-1">
            <Clock className="size-3" /> {(message.meta.latencyMs / 1000).toFixed(2)}s
          </span>
          {message.meta.tokens > 0 && (
            <span className="flex items-center gap-1">
              <Coins className="size-3" /> {message.meta.tokens} tokens
            </span>
          )}
        </div>
      )}
    </div>
  );
}
