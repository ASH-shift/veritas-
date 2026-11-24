"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, FileText, Loader2, Trash2, Upload } from "lucide-react";
import { toast } from "sonner";
import { RequireAuth } from "@/components/require-auth";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { api, uploadDocument, type DocumentOut } from "@/lib/api";
import { cn } from "@/lib/utils";

const ACCEPT = ".pdf,.md,.txt";
const MAX_BYTES = 10 * 1024 * 1024;
const POLL_MS = 2000;

interface UploadState {
  name: string;
  percent: number;
}

function StatusBadge({ doc }: { doc: DocumentOut }) {
  if (doc.status === "ready")
    return (
      <Badge variant="secondary" className="gap-1">
        <CheckCircle2 className="size-3 text-emerald-500" /> Ready
      </Badge>
    );
  if (doc.status === "failed")
    return (
      <Badge variant="destructive" className="gap-1" title={doc.error ?? undefined}>
        <AlertCircle className="size-3" /> Failed
      </Badge>
    );
  return (
    <Badge variant="outline" className="gap-1">
      <Loader2 className="size-3 animate-spin" /> {doc.status === "pending" ? "Queued" : "Processing"}
    </Badge>
  );
}

function DocumentsManager({ token }: { token: string }) {
  const [docs, setDocs] = useState<DocumentOut[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [uploads, setUploads] = useState<UploadState[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // setState only runs in promise callbacks, so this is safe to call from an effect.
  const refresh = useCallback(() => {
    api
      .listDocuments(token)
      .then((d) => {
        setDocs(d);
        setLoadError(null);
      })
      .catch((err: Error) => setLoadError(err.message));
  }, [token]);

  useEffect(refresh, [refresh]);

  // Poll while any document is still being ingested.
  const inProgress = docs?.some((d) => d.status === "pending" || d.status === "processing");
  useEffect(() => {
    if (!inProgress) return;
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [inProgress, refresh]);

  async function handleFiles(files: FileList | null) {
    if (!files) return;
    for (const file of Array.from(files)) {
      if (file.size > MAX_BYTES) {
        toast.error(`${file.name} is larger than 10 MB`);
        continue;
      }
      setUploads((u) => [...u, { name: file.name, percent: 0 }]);
      try {
        await uploadDocument(token, file, (percent) =>
          setUploads((u) => u.map((x) => (x.name === file.name ? { ...x, percent } : x))),
        );
        toast.success(`${file.name} uploaded — indexing now`);
      } catch (err) {
        toast.error(`${file.name}: ${(err as Error).message}`);
      } finally {
        setUploads((u) => u.filter((x) => x.name !== file.name));
        refresh();
      }
    }
  }

  async function handleDelete(doc: DocumentOut) {
    if (!window.confirm(`Delete ${doc.filename}? This removes its chunks and cannot be undone.`)) return;
    try {
      await api.deleteDocument(token, doc.id);
      toast.success(`Deleted ${doc.filename}`);
      refresh();
    } catch (err) {
      toast.error((err as Error).message);
    }
  }

  return (
    <div className="mx-auto w-full max-w-4xl space-y-6 px-4 py-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Upload PDF, Markdown or TXT files (max 10 MB). They are chunked, embedded and indexed for search.
        </p>
      </div>

      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={cn(
          "flex w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-10 text-sm text-muted-foreground transition-colors hover:bg-muted/50",
          dragging && "border-primary bg-muted/50",
        )}
      >
        <Upload className="size-6" />
        <span>
          <span className="font-medium text-foreground">Click to upload</span> or drag files here
        </span>
        <span className="text-xs">.pdf · .md · .txt</span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        multiple
        hidden
        onChange={(e) => {
          handleFiles(e.target.files);
          e.target.value = "";
        }}
      />

      {uploads.map((u) => (
        <div key={u.name} className="space-y-1.5">
          <div className="flex justify-between text-xs">
            <span>{u.name}</span>
            <span className="text-muted-foreground">{u.percent}%</span>
          </div>
          <Progress value={u.percent} />
        </div>
      ))}

      <Card>
        <CardHeader>
          <CardTitle>Your workspace</CardTitle>
          <CardDescription>
            {docs ? `${docs.length} document${docs.length === 1 ? "" : "s"}` : "Loading…"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loadError && (
            <div className="flex items-center justify-between gap-2 rounded-lg border border-destructive/40 p-3 text-sm text-destructive">
              <span>{loadError}</span>
              <Button size="sm" variant="outline" onClick={refresh}>
                Retry
              </Button>
            </div>
          )}
          {!docs && !loadError && (
            <div className="space-y-2">
              {[0, 1, 2].map((i) => (
                <div key={i} className="h-10 animate-pulse rounded-md bg-muted" />
              ))}
            </div>
          )}
          {docs && docs.length === 0 && (
            <p className="py-6 text-center text-sm text-muted-foreground">
              No documents yet. Upload one above to start asking questions.
            </p>
          )}
          {docs && docs.length > 0 && (
            <ul className="divide-y">
              {docs.map((doc) => (
                <li key={doc.id} className="flex items-center gap-3 py-2.5">
                  <FileText className="size-4 shrink-0 text-muted-foreground" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm">{doc.filename}</p>
                    <p className="text-xs text-muted-foreground">
                      {doc.mime_type}
                      {doc.num_pages ? ` · ${doc.num_pages} pages` : ""}
                    </p>
                  </div>
                  <StatusBadge doc={doc} />
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    aria-label={`Delete ${doc.filename}`}
                    onClick={() => handleDelete(doc)}
                  >
                    <Trash2 />
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function DocumentsPage() {
  return <RequireAuth>{(token) => <DocumentsManager token={token} />}</RequireAuth>;
}
