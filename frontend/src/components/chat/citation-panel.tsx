"use client";

import { FileText } from "lucide-react";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import type { Citation } from "@/lib/api";

export function CitationPanel({
  citation,
  onClose,
}: {
  citation: Citation | null;
  onClose: () => void;
}) {
  return (
    <Sheet open={citation !== null} onOpenChange={(open) => !open && onClose()}>
      <SheetContent side="right" className="w-full sm:max-w-md">
        {citation && (
          <>
            <SheetHeader>
              <SheetTitle className="flex items-center gap-2">
                <FileText className="size-4" />
                {citation.filename}
              </SheetTitle>
              <SheetDescription>Page {citation.page}</SheetDescription>
            </SheetHeader>
            <div className="px-4 pb-4">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Source snippet
              </p>
              <blockquote className="mt-2 whitespace-pre-wrap rounded-lg border bg-muted/40 p-3 text-sm leading-relaxed">
                {citation.snippet || "No snippet available."}
              </blockquote>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
