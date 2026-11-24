"use client";

import { CheckCircle2, ChevronRight, Loader2, XCircle } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { StepEvent } from "./types";

/** Extra detail the backend attaches to some step events. */
function stepDetail(step: StepEvent): string | null {
  if (step.rewritten_query) return `→ "${step.rewritten_query}"`;
  if (step.route) return `route: ${step.route}`;
  if (step.chunks_found !== undefined) return `${step.chunks_found} chunks`;
  if (step.relevant !== undefined) return `${step.relevant} relevant`;
  if (step.grounded !== undefined) return step.grounded ? "grounded" : "not grounded";
  return null;
}

export function AgentSteps({ steps, streaming }: { steps: StepEvent[]; streaming?: boolean }) {
  if (steps.length === 0 && !streaming) return null;

  return (
    <Collapsible defaultOpen={false} className="text-xs">
      <CollapsibleTrigger className="group flex items-center gap-1 text-muted-foreground hover:text-foreground">
        <ChevronRight className="size-3.5 transition-transform group-data-[panel-open]:rotate-90" />
        Agent steps ({steps.length})
        {streaming && <Loader2 className="ml-1 size-3 animate-spin" />}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ol className="mt-2 space-y-1.5 border-l pl-3">
          {steps.map((step, i) => {
            const detail = stepDetail(step);
            const failed = step.grounded === false;
            return (
              <li key={i} className="flex items-start gap-2">
                {failed ? (
                  <XCircle className="mt-0.5 size-3.5 shrink-0 text-amber-500" />
                ) : (
                  <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-emerald-500" />
                )}
                <span>
                  {step.content}
                  {detail && <span className="ml-1.5 text-muted-foreground">{detail}</span>}
                </span>
              </li>
            );
          })}
        </ol>
      </CollapsibleContent>
    </Collapsible>
  );
}
