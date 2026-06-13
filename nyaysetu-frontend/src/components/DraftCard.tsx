"use client";

import { useState } from "react";
import { FileText, Copy, Check } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/cn";
import type { DraftResponse, DraftSection } from "@/lib/types";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { CitationCard } from "./CitationCard";

function CopyButton({ text }: { text: string }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* clipboard unavailable */
    }
  };
  return (
    <button
      onClick={copy}
      className="inline-flex items-center gap-1 rounded-lg border border-border bg-bg px-2.5 py-1 text-xs font-medium text-ink transition-colors hover:border-primary/40 hover:bg-primary-soft/30"
    >
      {copied ? <Check size={13} className="text-success" aria-hidden /> : <Copy size={13} aria-hidden />}
      {copied ? t("draftCopied") : t("draftCopy")}
    </button>
  );
}

function Section({ section }: { section: DraftSection }) {
  if (section.items.length === 0) return null;
  const tone =
    section.tone === "warn"
      ? { box: "border-warning/30 bg-warning/10", label: "text-warning" }
      : section.tone === "info"
      ? { box: "border-primary/20 bg-primary-soft/40", label: "text-primary" }
      : { box: "border-border bg-bg", label: "text-muted" };
  return (
    <div className={cn("mt-3 rounded-xl border p-3", tone.box)}>
      <p className={cn("text-xs font-semibold uppercase tracking-wide", tone.label)}>{section.label}</p>
      <ul className="mt-1.5 space-y-1.5">
        {section.items.map((it, i) => (
          <li key={i} className="flex gap-2 text-sm leading-relaxed text-ink">
            <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" aria-hidden />
            {it}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function DraftCard({ draft }: { draft: DraftResponse }) {
  const { t } = useI18n();

  return (
    <div className="animate-fade-up rounded-2xl border border-border bg-surface p-4 shadow-sm sm:p-5">
      {/* Meta row */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <ConfidenceBadge level={draft.confidence} />
        <span className="inline-flex items-center gap-1 rounded-full bg-primary-soft/60 px-2 py-0.5 text-xs font-medium text-primary">
          <FileText size={12} aria-hidden />
          {draft.doc_title}
        </span>
        <span className="ml-auto text-[11px] text-muted">
          {t("responseTime", { ms: draft.response_time_ms })}
        </span>
      </div>

      {/* The drafted document — the hero, with a copy button */}
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted">{t("draftDocLabel")}</p>
        <CopyButton text={draft.document_text} />
      </div>
      <pre className="mt-1.5 max-h-96 overflow-auto whitespace-pre-wrap rounded-xl border border-border bg-bg p-3 font-sans text-[13.5px] leading-relaxed text-ink">
        {draft.document_text}
      </pre>

      {/* Procedural sections (how to file, timeline, escalation, tips) */}
      {draft.sections.map((s, i) => (
        <Section key={i} section={s} />
      ))}

      {/* Based on (sources) */}
      {draft.citations.length > 0 && (
        <div className="mt-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
            {t("draftBasedOn")} · {draft.citations.length}
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {draft.citations.map((c, i) => (
              <CitationCard key={`${c.label}-${i}`} citation={c} />
            ))}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <p className="mt-4 border-t border-border pt-3 text-xs leading-relaxed text-muted">
        <span className="font-medium">{t("disclaimerLabel")}:</span> {draft.disclaimer}
      </p>
    </div>
  );
}
