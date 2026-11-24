// Thin client for the Veritas FastAPI backend.
// The base URL is inlined at build time (NEXT_PUBLIC_ vars are frozen by `next build`).

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

// ── Types (mirror backend Pydantic schemas) ──────────────────────────────────

export type ChatMode = "vector" | "hybrid" | "hybrid_rerank" | "agentic";

export interface Citation {
  filename: string;
  page: number;
  snippet: string;
}

export interface DocumentOut {
  id: number;
  filename: string;
  mime_type: string;
  num_pages: number | null;
  status: "pending" | "processing" | "ready" | "failed";
  error: string | null;
}

export interface EvalReport {
  config: ChatMode;
  timestamp: string;
  aggregated: Record<string, number>;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  workspace_id: number;
}

// One message in the conversation history sent to the backend.
export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

// ── Errors ───────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

/** Turn a FastAPI error response into a readable message. */
export async function readError(resp: Response): Promise<ApiError> {
  let message = `Request failed (${resp.status})`;
  try {
    const body = await resp.json();
    if (typeof body.detail === "string") message = body.detail;
    else if (Array.isArray(body.detail) && body.detail[0]?.msg) message = body.detail[0].msg;
  } catch {
    // non-JSON body: keep the generic message
  }
  if (resp.status === 429) message = "Rate limit reached — please wait a bit and try again.";
  return new ApiError(resp.status, message);
}

// ── Request helper ───────────────────────────────────────────────────────────

export async function apiFetch<T>(
  path: string,
  options: RequestInit & { token?: string | null } = {},
): Promise<T> {
  const { token, headers, ...rest } = options;
  let resp: Response;
  try {
    resp = await fetch(`${API_URL}${path}`, {
      ...rest,
      headers: {
        ...(rest.body && !(rest.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...headers,
      },
    });
  } catch {
    // Network failure, or a server crash (500s from unhandled errors carry no CORS headers).
    throw new ApiError(
      0,
      "Can't reach the server or it hit an error. On the free tier it may be waking up — try again in ~30s.",
    );
  }
  if (!resp.ok) throw await readError(resp);
  if (resp.status === 204) return undefined as T;
  return (await resp.json()) as T;
}

// ── Endpoints ────────────────────────────────────────────────────────────────

export const api = {
  login: (email: string, password: string) =>
    apiFetch<TokenResponse>("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),

  register: (email: string, password: string) =>
    apiFetch<{ id: number; email: string; workspace_id: number }>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  listDocuments: (token: string) => apiFetch<DocumentOut[]>("/documents", { token }),

  deleteDocument: (token: string, id: number) =>
    apiFetch<void>(`/documents/${id}`, { method: "DELETE", token }),

  latestEvals: () => apiFetch<EvalReport[]>("/evals/latest"),
};

/**
 * Upload a document with progress events. fetch() has no upload progress,
 * so this uses XMLHttpRequest.
 */
export function uploadDocument(
  token: string,
  file: File,
  onProgress: (percent: number) => void,
): Promise<DocumentOut> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_URL}/documents`);
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      let body: { detail?: string } & Partial<DocumentOut> = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        // ignore
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as DocumentOut);
      else reject(new ApiError(xhr.status, body.detail ?? `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () => reject(new ApiError(0, "Network error during upload"));
    const form = new FormData();
    form.append("file", file);
    xhr.send(form);
  });
}
