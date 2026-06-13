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

// Document understanding (/analyze) — mirrors backend app/schemas/analyze.py.
export interface AnalyzeResponse {
  document_type: string;
  summary: string;
  key_points: string[];
  deadlines: string[];
  action: string;
  confidence: Confidence;
  citations: Citation[];
  current_law_note?: string | null;
  citation_verified: boolean;
  abstained: boolean;
  escalation?: string | null;
  disclaimer: string;
  response_time_ms: number;
}

// RTI drafting (/draft/rti) — mirrors backend app/schemas/rti.py.
export type GovLevel = "central" | "state";

export interface RTIDraftRequest {
  subject: string;
  public_authority?: string;
  level: GovLevel;
  applicant_name?: string;
  applicant_address?: string;
  is_bpl: boolean;
  language: string;
}

export interface FilingInfo {
  where_to_file: string;
  fee: string;
  is_bpl_exempt: boolean;
  portal?: string | null;
}

export interface RTIDraftResponse {
  application_text: string;
  subject_line: string;
  questions: string[];
  public_authority: string;
  filing: FilingInfo;
  timeline: string[];
  appeals: string[];
  tips: string[];
  confidence: Confidence;
  citations: Citation[];
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
  // assistant messages for an uploaded document carry an analysis instead
  analysis?: AnalyzeResponse;
  // true while a document upload is being analysed (shows a doc-specific loading label)
  analysisPending?: boolean;
  // assistant messages for an RTI draft carry the drafted application
  rti?: RTIDraftResponse;
  // true while an RTI application is being drafted (shows an RTI-specific loading label)
  rtiPending?: boolean;
  error?: string;
  pending?: boolean;
}
