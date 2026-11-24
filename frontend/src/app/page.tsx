import Link from "next/link";
import { BadgeCheck, Layers, Quote, ShieldAlert } from "lucide-react";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const FEATURES = [
  {
    icon: Layers,
    title: "Hybrid retrieval + reranking",
    text: "Vector search and PostgreSQL full-text search fused with RRF, then reranked by a cross-encoder.",
  },
  {
    icon: BadgeCheck,
    title: "Self-checking agent",
    text: "A LangGraph agent rewrites the query, grades retrieved chunks and verifies the answer is grounded.",
  },
  {
    icon: Quote,
    title: "Citations for every claim",
    text: "Each answer cites document and page. Click a citation to see the exact source snippet.",
  },
  {
    icon: ShieldAlert,
    title: "Guardrails",
    text: "Prompt-injection detection, PII redaction, and a refusal when the answer isn't in the documents.",
  },
];

export default function Home() {
  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col items-center px-4 py-16 sm:py-24">
      <span className="rounded-full border px-3 py-1 text-xs text-muted-foreground">Agentic RAG knowledge copilot</span>
      <h1 className="mt-5 max-w-2xl text-center text-4xl font-semibold tracking-tight sm:text-5xl">
        Answers from your documents. Nothing made up.
      </h1>
      <p className="mt-4 max-w-xl text-center text-muted-foreground">
        Veritas answers questions over internal documents with citations, checks its own answers for
        groundedness, and says &ldquo;I couldn&apos;t find this&rdquo; instead of guessing.
      </p>
      <div className="mt-8 flex flex-wrap justify-center gap-3">
        <Link href="/demo" className={cn(buttonVariants({ size: "lg" }), "px-4")}>
          Try the demo
        </Link>
        <Link href="/evals" className={cn(buttonVariants({ size: "lg", variant: "outline" }), "px-4")}>
          See eval results
        </Link>
      </div>

      <div className="mt-16 grid w-full gap-4 sm:grid-cols-2">
        {FEATURES.map((f) => (
          <div key={f.title} className="rounded-xl border bg-card p-5">
            <f.icon className="size-5 text-primary" />
            <h2 className="mt-3 font-medium">{f.title}</h2>
            <p className="mt-1 text-sm text-muted-foreground">{f.text}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
