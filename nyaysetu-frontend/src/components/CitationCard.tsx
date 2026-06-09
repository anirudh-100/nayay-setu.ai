"use client";

import { BookText, Gavel, MessagesSquare, FileText, ExternalLink } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/cn";
import type { Citation, SourceType, CodeStatus } from "@/lib/types";

const SOURCE_ICON: Record<SourceType, typeof BookText> = {
  statute: BookText,
  judgment: Gavel,
  qa: MessagesSquare,
  guide: FileText,
};

function StatusChip({ status }: { status: CodeStatus }) {
  const { t } = useI18n();
  if (status === "unknown") return null;
  const label = t(`codeStatus.${status}`);
  if (!label) return null;
  return (
    <span
      className={cn(
        "rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
        status === "current" ? "bg-success/10 text-success" : "bg-muted/15 text-muted"
      )}
    >
      {label}
    </span>
  );
}

export function CitationCard({ citation }: { citation: Citation }) {
  const { t } = useI18n();
  const Icon = SOURCE_ICON[citation.source_type] ?? FileText;
  const clickable = Boolean(citation.url);

  const inner = (
    <>
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-1.5 text-sm font-medium text-ink">
          <Icon size={14} className="shrink-0 text-primary" aria-hidden />
          <span>{citation.label}</span>
        </div>
        <StatusChip status={citation.code_status} />
      </div>
      <p className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-muted">{citation.snippet}</p>
      {clickable && (
        <span className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-primary">
          {t("openSource")}
          <ExternalLink size={12} aria-hidden />
        </span>
      )}
    </>
  );

  const base = "block rounded-xl border p-3 transition-colors";

  if (clickable) {
    return (
      <a
        href={citation.url!}
        target="_blank"
        rel="noopener noreferrer"
        className={cn(base, "border-border bg-bg/50 hover:border-primary/40 hover:bg-primary-soft/20")}
      >
        {inner}
      </a>
    );
  }

  return <div className={cn(base, "border-border bg-bg/50")}>{inner}</div>;
}
