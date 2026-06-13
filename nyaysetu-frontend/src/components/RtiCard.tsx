"use client";

import { useState } from "react";
import {
  ScrollText,
  Copy,
  Check,
  Send,
  CalendarClock,
  Scale,
  Lightbulb,
  ExternalLink,
} from "lucide-react";
import { useI18n } from "@/lib/i18n";
import type { RTIDraftResponse } from "@/lib/types";
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
      /* clipboard unavailable — no-op */
    }
  };

  return (
    <button
      onClick={copy}
      className="inline-flex items-center gap-1 rounded-lg border border-border bg-bg px-2.5 py-1 text-xs font-medium text-ink transition-colors hover:border-primary/40 hover:bg-primary-soft/30"
    >
      {copied ? <Check size={13} className="text-success" aria-hidden /> : <Copy size={13} aria-hidden />}
      {copied ? t("rtiCopied") : t("rtiCopy")}
    </button>
  );
}

function ListSection({
  icon,
  label,
  items,
  tone = "default",
}: {
  icon: React.ReactNode;
  label: string;
  items: string[];
  tone?: "default" | "warn";
}) {
  if (items.length === 0) return null;
  const box =
    tone === "warn"
      ? "border-warning/30 bg-warning/10"
      : "border-border bg-bg";
  return (
    <div className={`mt-3 rounded-xl border p-3 ${box}`}>
      <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted">
        {icon}
        {label}
      </p>
      <ul className="mt-1.5 space-y-1.5">
        {items.map((it, i) => (
          <li key={i} className="flex gap-2 text-sm leading-relaxed text-ink">
            <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" aria-hidden />
            {it}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function RtiCard({ rti }: { rti: RTIDraftResponse }) {
  const { t } = useI18n();

  return (
    <div className="animate-fade-up rounded-2xl border border-border bg-surface p-4 shadow-sm sm:p-5">
      {/* Meta row */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <ConfidenceBadge level={rti.confidence} />
        <span className="inline-flex items-center gap-1 rounded-full bg-primary-soft/60 px-2 py-0.5 text-xs font-medium text-primary">
          <ScrollText size={12} aria-hidden />
          {t("rtiCardTitle")}
        </span>
        <span className="ml-auto text-[11px] text-muted">
          {t("responseTime", { ms: rti.response_time_ms })}
        </span>
      </div>

      {/* The draft application — the hero, with a copy button */}
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-muted">{t("rtiLetterLabel")}</p>
        <CopyButton text={rti.application_text} />
      </div>
      <pre className="mt-1.5 max-h-96 overflow-auto whitespace-pre-wrap rounded-xl border border-border bg-bg p-3 font-sans text-[13.5px] leading-relaxed text-ink">
        {rti.application_text}
      </pre>

      {/* How to file */}
      <div className="mt-3 rounded-xl border border-primary/20 bg-primary-soft/40 p-3 text-sm text-ink">
        <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-primary">
          <Send size={14} aria-hidden />
          {t("rtiFilingLabel")}
        </p>
        <p className="mt-1.5 leading-relaxed">{rti.filing.where_to_file}</p>
        <p className="mt-1.5 leading-relaxed">{rti.filing.fee}</p>
        {rti.filing.portal && (
          <a
            href={rti.filing.portal}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-primary"
          >
            {t("rtiOpenPortal")}
            <ExternalLink size={12} aria-hidden />
          </a>
        )}
      </div>

      {/* Timeline */}
      <ListSection
        icon={<CalendarClock size={14} aria-hidden />}
        label={t("rtiTimelineLabel")}
        items={rti.timeline}
      />

      {/* Appeals — what to do if ignored */}
      <ListSection icon={<Scale size={14} aria-hidden />} label={t("rtiAppealsLabel")} items={rti.appeals} />

      {/* Tips */}
      <ListSection icon={<Lightbulb size={14} aria-hidden />} label={t("rtiTipsLabel")} items={rti.tips} />

      {/* Based on (sources) */}
      {rti.citations.length > 0 && (
        <div className="mt-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
            {t("rtiBasedOnLabel")} · {rti.citations.length}
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {rti.citations.map((c, i) => (
              <CitationCard key={`${c.label}-${i}`} citation={c} />
            ))}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <p className="mt-4 border-t border-border pt-3 text-xs leading-relaxed text-muted">
        <span className="font-medium">{t("disclaimerLabel")}:</span> {rti.disclaimer}
      </p>
    </div>
  );
}
