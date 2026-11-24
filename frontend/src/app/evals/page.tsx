"use client";

import { useCallback, useEffect, useState } from "react";
import { FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, type EvalReport } from "@/lib/api";

const CONFIG_LABELS: Record<string, string> = {
  vector: "Vector",
  hybrid: "Hybrid (BM25 + vector)",
  hybrid_rerank: "Hybrid + rerank",
  agentic: "Agentic (LangGraph)",
};

// Rate metrics are 0–1 and shown as percentages and bars.
const RATE_METRICS: { key: string; label: string }[] = [
  { key: "faithfulness", label: "Faithfulness" },
  { key: "answer_correctness", label: "Answer correctness" },
  { key: "context_recall", label: "Context recall" },
  { key: "citation_accuracy", label: "Citation accuracy" },
  { key: "refusal_accuracy", label: "Refusal accuracy" },
];

const pct = (v: number | undefined) => (v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
const ms = (v: number | undefined) => (v === undefined ? "—" : `${Math.round(v)} ms`);

function formatTimestamp(ts: string) {
  // YYYYMMDD_HHMMSS
  const m = ts.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/);
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : ts;
}

function MetricBars({ reports, metric }: { reports: EvalReport[]; metric: { key: string; label: string } }) {
  const rows = reports.filter((r) => r.aggregated[metric.key] !== undefined);
  if (rows.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">{metric.label}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2.5">
        {rows.map((r) => {
          const v = r.aggregated[metric.key];
          return (
            <div key={r.config} className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">{CONFIG_LABELS[r.config] ?? r.config}</span>
                <span className="font-medium tabular-nums">{pct(v)}</span>
              </div>
              <div className="h-2 rounded-full bg-muted">
                <div className="h-2 rounded-full bg-primary" style={{ width: `${Math.max(0, Math.min(1, v)) * 100}%` }} />
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

export default function EvalsPage() {
  const [reports, setReports] = useState<EvalReport[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  // setState only runs in promise callbacks, so this is safe to call from an effect.
  const load = useCallback(() => {
    api
      .latestEvals()
      .then((r) => {
        setReports(r);
        setError(null);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(load, [load]);

  return (
    <div className="mx-auto w-full max-w-5xl space-y-6 px-4 py-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Evaluation results</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Measured on a golden set of questions (answerable, multi-hop, unanswerable and adversarial) for each
          retrieval configuration. Numbers come straight from the latest report files.
        </p>
      </div>

      {error && (
        <div className="flex items-center justify-between gap-2 rounded-lg border border-destructive/40 p-3 text-sm text-destructive">
          <span>{error}</span>
          <Button size="sm" variant="outline" onClick={load}>
            Retry
          </Button>
        </div>
      )}

      {!reports && !error && <div className="h-48 animate-pulse rounded-xl bg-muted" />}

      {reports && reports.length === 0 && (
        <Card>
          <CardContent className="flex flex-col items-center gap-2 py-12 text-center">
            <FlaskConical className="size-8 text-muted-foreground" />
            <p className="font-medium">No eval reports yet</p>
            <p className="max-w-md text-sm text-muted-foreground">
              Run <code className="rounded bg-muted px-1">make evals-all</code> in the backend to generate reports for
              all four configurations. They will appear here automatically.
            </p>
          </CardContent>
        </Card>
      )}

      {reports && reports.length > 0 && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Comparison</CardTitle>
              <CardDescription>Latest report per configuration</CardDescription>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Config</TableHead>
                    <TableHead className="text-right">Questions</TableHead>
                    {RATE_METRICS.map((m) => (
                      <TableHead key={m.key} className="text-right">
                        {m.label}
                      </TableHead>
                    ))}
                    <TableHead className="text-right">p50</TableHead>
                    <TableHead className="text-right">p95</TableHead>
                    <TableHead>Run</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {reports.map((r) => (
                    <TableRow key={r.config}>
                      <TableCell className="font-medium">{CONFIG_LABELS[r.config] ?? r.config}</TableCell>
                      <TableCell className="text-right tabular-nums">{r.aggregated.n ?? "—"}</TableCell>
                      {RATE_METRICS.map((m) => (
                        <TableCell key={m.key} className="text-right tabular-nums">
                          {pct(r.aggregated[m.key])}
                        </TableCell>
                      ))}
                      <TableCell className="text-right tabular-nums">{ms(r.aggregated.p50_ms)}</TableCell>
                      <TableCell className="text-right tabular-nums">{ms(r.aggregated.p95_ms)}</TableCell>
                      <TableCell className="text-xs text-muted-foreground">{formatTimestamp(r.timestamp)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {RATE_METRICS.map((m) => (
              <MetricBars key={m.key} reports={reports} metric={m} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
