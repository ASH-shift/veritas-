"use client";

import { ChatView } from "@/components/chat/chat-view";
import { RequireAuth } from "@/components/require-auth";

export default function ChatPage() {
  return (
    <RequireAuth>
      {(token) => (
        <ChatView
          path="/chat"
          token={token}
          emptyTitle="Ask your documents"
          emptyText="Answers come only from documents in your workspace, with a citation for every claim. Upload files on the Documents page first."
          suggestions={["Summarise the key points of my documents", "What policies are covered?"]}
        />
      )}
    </RequireAuth>
  );
}
