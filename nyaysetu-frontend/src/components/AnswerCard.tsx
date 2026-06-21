"use client";

import { useState } from "react";
import { Info, AlertTriangle, ArrowRight, LifeBuoy, BookMarked, Scale, ThumbsUp, ThumbsDown } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { sendFeedback } from "@/lib/api";
import type { AskResponse } from "@/lib/types";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { CitationCard } from "./CitationCard";

/** One case-analysis section: a tinted box with a label and a bulleted (or numbered) list.
 *  Renders nothing when empty, so suppressed/ungrounded sections simply disappear. */
function AnalysisList({
  label,
  items,
  tone = "default",
  ordered = false,
  caveat,
}: {
  label: string;
  items?: string[];
  tone?: "default" | "info" | "warn";
  ordered?: boolean;
  caveat?: string;
}) {
  if (!items || items.length === 0) return null;
  const box =
    tone === "warn"
      ? "border-warning/30 bg-warning/10"
      : tone === "info"
      ? "border-primary/20 bg-primary-soft/40"
      : "border-border bg-bg";
  const labelColor = tone === "warn" ? "text-warning" : tone === "info" ? "text-primary" : "text-muted";
  return (
    <div className={`rounded-xl border p-3 ${box}`}>
      <p className={`text-xs font-semibold uppercase tracking-wide ${labelColor}`}>{label}</p>
      {ordered ? (
        <ol className="mt-1.5 list-decimal space-y-1.5 pl-5 text-sm leading-relaxed text-ink">
          {items.map((it, i) => (
            <li key={i}>{it}</li>
          ))}
        </ol>
      ) : (
        <ul className="mt-1.5 space-y-1.5">
          {items.map((it, i) => (
            <li key={i} className="flex gap-2 text-sm leading-relaxed text-ink">
              <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" aria-hidden />
              <span>{it}</span>
            </li>
          ))}
        </ul>
      )}
      {caveat && <p className="mt-2 text-[11px] italic leading-relaxed text-muted">{caveat}</p>}
    </div>
  );
}

export function AnswerCard({ response, question }: { response: AskResponse; question?: string }) {
  const { t, locale } = useI18n();
  const [voted, setVoted] = useState<null | "up" | "down">(null);

  const vote = (verdict: "up" | "down") => {
    if (voted) return;
    setVoted(verdict);
    sendFeedback({ verdict, query: question, law_reference: response.law_reference, language: locale });
  };

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

      {/* Case analysis — present only on a strong, verified answer (else null) */}
      {response.analysis && (
        <div className="mt-3 space-y-3">
          {/* Outcome framing — always shown, so "not a prediction" is unmissable */}
          <div className="flex gap-2 text-xs italic leading-relaxed text-muted">
            <Info size={14} className="mt-0.5 shrink-0 text-primary" aria-hidden />
            <p>{response.analysis.outcome_framing || t("outcomeFramingNote")}</p>
          </div>
          <AnalysisList label={t("situationLabel")} items={response.analysis.situation} />
          <AnalysisList label={t("doNowLabel")} items={response.analysis.do_now} tone="info" />
          <AnalysisList label={t("applicableLawLabel")} items={response.analysis.applicable_law} />
          {/* Grounded offence classification (BNSS First Schedule) — only when unambiguous */}
          {response.analysis.classification && (
            <div className="flex gap-2 rounded-xl border border-border bg-bg p-3 text-sm text-ink">
              <Scale size={16} className="mt-0.5 shrink-0 text-primary" aria-hidden />
              <div>
                <p className="text-xs font-semibold uppercase tracking-wide text-muted">
                  {t("classificationLabel")}
                </p>
                <p className="mt-0.5 leading-relaxed">{response.analysis.classification}</p>
              </div>
            </div>
          )}
          <AnalysisList
            label={t("whatHappensNextLabel")}
            items={response.analysis.what_happens_next}
            ordered
          />
          <AnalysisList label={t("alsoPossibleLabel")} items={response.analysis.also_possible} tone="info" />
          <AnalysisList
            label={t("advocateLabel")}
            items={response.analysis.for_your_advocate}
            tone="warn"
            caveat={t("advocateCaveat")}
          />
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

      {/* Feedback */}
      <div className="mt-4 flex items-center gap-2 border-t border-border pt-3">
        {voted ? (
          <span className="text-xs text-muted">{t("feedbackThanks")}</span>
        ) : (
          <>
            <span className="text-xs text-muted">{t("feedbackPrompt")}</span>
            <button
              onClick={() => vote("up")}
              aria-label={t("feedbackUp")}
              className="rounded-full border border-border p-1.5 text-muted transition-colors hover:text-primary"
            >
              <ThumbsUp size={14} aria-hidden />
            </button>
            <button
              onClick={() => vote("down")}
              aria-label={t("feedbackDown")}
              className="rounded-full border border-border p-1.5 text-muted transition-colors hover:text-danger"
            >
              <ThumbsDown size={14} aria-hidden />
            </button>
          </>
        )}
      </div>

      {/* Disclaimer */}
      <p className="mt-3 border-t border-border pt-3 text-xs leading-relaxed text-muted">
        <span className="font-medium">{t("disclaimerLabel")}:</span> {response.disclaimer}
      </p>
    </div>
  );
}
