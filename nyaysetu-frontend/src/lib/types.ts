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

// Generic drafting engine (/draft) — mirrors backend app/schemas/draft.py.
export type FieldKind = "text" | "textarea" | "select" | "checkbox";
export type SectionTone = "default" | "info" | "warn";

export interface FieldOption {
  value: string;
  label: string;
}

export interface FieldSpec {
  key: string;
  label: string;
  kind: FieldKind;
  required: boolean;
  placeholder?: string | null;
  help?: string | null;
  options: FieldOption[];
  default?: string | null;
}

export interface JourneyInfo {
  id: string;
  title: string;
  description: string;
  doc_title: string;
  icon?: string | null;
  fields: FieldSpec[];
  note?: string | null;
}

export interface DraftSection {
  label: string;
  items: string[];
  tone: SectionTone;
}

export interface DraftRequest {
  journey: string;
  fields: Record<string, string | boolean>;
  applicant_name?: string;
  applicant_address?: string;
  language: string;
}

export interface DraftResponse {
  journey: string;
  doc_title: string;
  document_text: string;
  subject_line?: string | null;
  key_points: string[];
  sections: DraftSection[];
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
  // assistant messages for a drafted document carry the draft
  draft?: DraftResponse;
  // true while a document is being drafted (shows a drafting-specific loading label)
  draftPending?: boolean;
  error?: string;
  pending?: boolean;
}
