// Mirrors the backend's app/schemas/ask.py response contract.

export type Confidence = "high" | "medium" | "low";
export type SourceType = "statute" | "judgment" | "qa" | "guide";
export type CodeStatus = "current" | "repealed" | "unknown";
// How trustworthy the source text is — drives the per-citation trust badge.
export type Verification = "official" | "curated" | "unverified";

export interface Citation {
  label: string;
  source_type: SourceType;
  snippet: string;
  url?: string | null;
  code_status: CodeStatus;
  verification: Verification;
  source_authority?: string | null;
}

export interface AskResponse {
  answer: string;
  law_reference: string;
  action: string;
  confidence: Confidence;
  reasoning: string;
  citations: Citation[];
  abstained: boolean;
  escalation?: string | null;
  current_law_note?: string | null;
  citation_verified: boolean;
  disclaimer: string;
  response_time_ms: number;
}

// A single turn in the conversation UI.
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  // user messages carry text; assistant messages carry the structured response
  text?: string;
  response?: AskResponse;
  error?: string;
  pending?: boolean;
}
