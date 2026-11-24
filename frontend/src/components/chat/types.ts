import type { Citation } from "@/lib/api";
import type { ChatEvent } from "@/lib/sse";

export type StepEvent = Extract<ChatEvent, { type: "step" }>;

export interface AssistantMeta {
  latencyMs: number;
  tokens: number;
  cacheHit: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  steps: StepEvent[];
  citations: Citation[];
  meta?: AssistantMeta;
  error?: string;
  streaming?: boolean;
}
