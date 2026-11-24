"use client";

import { ChatView } from "@/components/chat/chat-view";

// Public demo: no login, queries the demo workspace (NimbusWorks sample docs).
// The backend rate-limits /chat/demo per IP.
export default function DemoPage() {
  return (
    <ChatView
      path="/chat/demo"
      emptyTitle="Try Veritas on NimbusWorks"
      emptyText="A demo knowledge base of 15 internal policy documents for a fictional company. Ask anything — if the answer isn't in the documents, Veritas says so instead of guessing."
      suggestions={[
        "How many days of annual leave do employees get?",
        "What is the hotel limit for business travel?",
        "How often must passwords be changed?",
        "What is the CEO's favourite food?",
      ]}
    />
  );
}
