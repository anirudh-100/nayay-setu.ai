"use client";

import { FileText, Info, AlertTriangle, ArrowRight, LifeBuoy, CalendarClock, ListChecks } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import type { AnalyzeResponse } from "@/lib/types";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { CitationCard } from "./CitationCard";

export function DocumentCard({ analysis }: { analysis: AnalyzeResponse }) {
  const { t } = useI18n();

  return (
    <div className="animate-fade-up rounded-2xl border border-border bg-surface p-4 shadow-sm sm:p-5">
      {/* Meta row */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <ConfidenceBadge level={analysis.confidence} />
        <span className="inline-flex items-center gap-1 rounded-full bg-primary-soft/60 px-2 py-0.5 text-xs font-medium text-primary">
          <FileText size={12} aria-hidden />
          {analysis.document_type}
        </span>
        <span className="ml-auto text-[11px] text-muted">
          {t("responseTime", { ms: analysis.response_time_ms })}
        </span>
      </div>

      {/* Summary */}
      <p className="text-xs font-semibold uppercase tracking-wide text-muted">{t("summaryLabel")}</p>
      <p className="mt-1 whitespace-pre-line text-[15px] leading-relaxed text-ink">{analysis.summary}</p>

      {/* Key points */}
      {analysis.key_points.length > 0 && (
        <div className="mt-3">
          <p className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted">
            <ListChecks size={14} aria-hidden />
            {t("keyPointsLabel")}
          </p>
          <ul className="mt-1.5 space-y-1.5">
            {analysis.key_points.map((point, i) => (
              <li key={i} className="flex gap-2 text-sm leading-relaxed text-ink">
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" aria-hidden />
                {point}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Deadlines — visually prominent, these are time-critical */}
      {analysis.deadlines.length > 0 && (
        <div className="mt-3 flex gap-2 rounded-xl border border-warning/30 bg-warning/10 p-3 text-sm text-ink">
          <CalendarClock size={16} className="mt-0.5 shrink-0 text-warning" aria-hidden />
          <div>
            <p className="font-medium text-warning">{t("deadlinesLabel")}</p>
            <ul className="mt-0.5 space-y-0.5">
              {analysis.deadlines.map((d, i) => (
                <li key={i} className="leading-relaxed">{d}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Current-law bridge (IPC/CrPC/IEA -> BNS/BNSS/BSA) */}
      {analysis.current_law_note && (
        <div className="mt-3 flex gap-2 rounded-xl border border-primary/20 bg-primary-soft/40 p-3 text-sm text-ink">
          <Info size={16} className="mt-0.5 shrink-0 text-primary" aria-hidden />
          <div>
            <p className="font-medium text-primary">{t("currentLawLabel")}</p>
            <p className="mt-0.5 leading-relaxed">{analysis.current_law_note}</p>
          </div>
        </div>
      )}

      {/* Unverified-citation caution */}
      {!analysis.citation_verified && (
        <div className="mt-3 flex gap-2 rounded-xl border border-warning/30 bg-warning/10 p-3 text-sm text-ink">
          <AlertTriangle size={16} className="mt-0.5 shrink-0 text-warning" aria-hidden />
          <p className="leading-relaxed">{t("verifyWarn")}</p>
        </div>
      )}

      {/* Action */}
      {analysis.action && (
        <div className="mt-3 flex gap-2 rounded-xl bg-bg p-3">
          <ArrowRight size={16} className="mt-0.5 shrink-0 text-primary" aria-hidden />
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-muted">{t("actionLabel")}</p>
            <p className="mt-0.5 text-sm leading-relaxed text-ink">{analysis.action}</p>
          </div>
        </div>
      )}

      {/* Escalation to legal aid */}
      {analysis.escalation && (
        <div className="mt-3 flex gap-2 rounded-xl border border-border bg-bg p-3 text-sm">
          <LifeBuoy size={16} className="mt-0.5 shrink-0 text-primary" aria-hidden />
          <div>
            <p className="font-medium text-ink">{t("escalationLabel")}</p>
            <p className="mt-0.5 leading-relaxed text-muted">{analysis.escalation}</p>
          </div>
        </div>
      )}

      {/* Sources */}
      {analysis.citations.length > 0 && (
        <div className="mt-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted">
            {t("sourcesLabel")} · {analysis.citations.length}
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {analysis.citations.map((c, i) => (
              <CitationCard key={`${c.label}-${i}`} citation={c} />
            ))}
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <p className="mt-4 border-t border-border pt-3 text-xs leading-relaxed text-muted">
        <span className="font-medium">{t("disclaimerLabel")}:</span> {analysis.disclaimer}
      </p>
    </div>
  );
}
