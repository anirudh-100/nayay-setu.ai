import type { AskResponse, AnalyzeResponse, RTIDraftRequest, RTIDraftResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.status = status;
  }
}

/** POST /ask — send a question, get a structured, cited answer. */
export async function askQuestion(query: string, language: string): Promise<AskResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, language }),
    });
  } catch {
    // Network/connection failure (backend not running, CORS, etc.)
    throw new ApiError("network");
  }

  if (!res.ok) {
    // Surface the backend's friendly detail when present.
    let detail = "";
    try {
      detail = (await res.json())?.detail ?? "";
    } catch {
      /* ignore */
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }

  return (await res.json()) as AskResponse;
}

/** POST /analyze/file — upload a document (PDF/.txt/.md), get a structured analysis. */
export async function analyzeFile(
  file: File,
  language: string,
  question?: string
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("language", language);
  if (question) form.append("question", question);

  let res: Response;
  try {
    // Note: don't set Content-Type — the browser adds the multipart boundary itself.
    res = await fetch(`${API_URL}/analyze/file`, { method: "POST", body: form });
  } catch {
    throw new ApiError("network");
  }

  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json())?.detail ?? "";
    } catch {
      /* ignore */
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }

  return (await res.json()) as AnalyzeResponse;
}

/** POST /draft/rti — draft a Right to Information application from a plain-language request. */
export async function draftRti(req: RTIDraftRequest): Promise<RTIDraftResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/draft/rti`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  } catch {
    throw new ApiError("network");
  }

  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json())?.detail ?? "";
    } catch {
      /* ignore */
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }

  return (await res.json()) as RTIDraftResponse;
}

export const appConfig = {
  name: process.env.NEXT_PUBLIC_APP_NAME ?? "NyaySetu",
  tagline: process.env.NEXT_PUBLIC_APP_TAGLINE ?? "Aapka Kanoon, Aapke Haath",
};
