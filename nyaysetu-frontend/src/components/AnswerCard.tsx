"use client";

import { Info, AlertTriangle, ArrowRight, LifeBuoy, BookMarked } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import type { AskResponse } from "@/lib/types";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { CitationCard } from "./CitationCard";

export function AnswerCard({ response }: { response: AskResponse }) {
  const { t } = useI18n();

  return (
    <div className="animate-fade-up rounded-2xl border border-border bg-surface p-4 shadow-sm sm:p-5">
      {/* Meta row */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <ConfidenceBadge level={response.confidence} />
        {response.law_reference && (
          <span className="inline-flex items-center gap-1 rounded-full bg-primary-soft/60 px-2 py-0.5 text-xs font-medium text-primary">
            <BookMarked size={12} aria-hidden />
            {response.law_reference}
          </span>
        )}
        <span className="ml-auto text-[11px] text-muted">
          {t("responseTime", { ms: response.response_time_ms })}
        </span>
      </div>

      {/* Answer */}
      <p className="whitespace-pre-line text-[15px] leading-relaxed text-ink">{response.answer}</p>

      {/* Current-law bridge (IPC -> BNS) */}
      {response.current_law_note && (
        <div className="mt-3 flex gap-2 rounded-xl border border-primary/20 bg-primary-soft/40 p-3 text-sm text-ink">
          <Info size={16} className="mt-0.5 shrink-0 text-primary" aria-hidden />
          <div>
            <p className="font-medium text-primary">{t("currentLawLabel")}</p>
            <p className="mt-0.5 leading-relaxed">{response.current_law_note}</p>
          </div>
        </div>
      )}

      {/* Unverified-citation caution */}
      {!response.citation_verified && (
        <div className="mt-3 flex gap-2 rounded-xl border border-warning/30 bg-warning/10 p-3 text-sm text-ink">
          <AlertTriangle size={16} className="mt-0.5 shrink-0 text-warning" aria-hidden />
          <p className="leading-relaxed">{t("verifyWarn")}</p>
        </div>
      )}

      {/* Action */}
      {response.action && (
        <div className="mt-3 flex gap-2 rounded-xl bg-bg p-3">
          <ArrowRight size={16} className="mt-0.5 shrink-0 text-primary" aria-hidden />
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-muted">{t("actionLabel")}</p>
            <p className="mt-0.5 text-sm leading-relaxed text-ink">{response.action}</p>
          </div>
        </div>
      )}

      {/* Escalation to legal aid */}
      {response.escalation && (
        <div className="mt-3 flex gap-2 rounded-xl border border-border bg-bg p-3 text-sm">
          <LifeBuoy size={16} className="mt-0.5 shrink-0 text-primary" aria-hidden />
          <div>
            <p className="font-medium text-ink">{t("escalationLabel")}</p>
            <p className="mt-0.5 leading-relaxed text-muted">{response.escalation}</p>
          </div>
        </div>
      )}

      {/* Sources */}
      {response.citations.length > 0 && (
        <div className="mt-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
            {t("sourcesLabel")} · {response.citations.length}
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {response.citations.map((c, i) => (
              <CitationCard key={`${c.label}-${i}`} citation={c} />
            ))}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <p className="mt-4 border-t border-border pt-3 text-xs leading-relaxed text-muted">
        <span className="font-medium">{t("disclaimerLabel")}:</span> {response.disclaimer}
      </p>
    </div>
  );
}
