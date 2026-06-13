"use client";

import {
  BookText,
  Gavel,
  MessagesSquare,
  FileText,
  ExternalLink,
  ShieldCheck,
  ShieldAlert,
  ShieldQuestion,
} from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/cn";
import type { Citation, SourceType, CodeStatus, Verification } from "@/lib/types";

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

const VERIFICATION_STYLE: Record<Verification, { cls: string; Icon: typeof ShieldCheck }> = {
  official: { cls: "bg-success/10 text-success", Icon: ShieldCheck },
  curated: { cls: "bg-warning/10 text-warning", Icon: ShieldQuestion },
  unverified: { cls: "bg-muted/15 text-muted", Icon: ShieldAlert },
};

/** The trust signal — tells the user, per source, whether to rely on it or confirm it. */
function VerificationBadge({ level, authority }: { level: Verification; authority?: string | null }) {
  const { t } = useI18n();
  const { cls, Icon } = VERIFICATION_STYLE[level] ?? VERIFICATION_STYLE.unverified;
  const label = t(`verification.${level}`);
  // Show the authority on hover (e.g. "India Code (indiacode.nic.in)") for full provenance.
  const title = authority ? `${label} · ${authority}` : label;
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
        cls
      )}
    >
      <Icon size={11} aria-hidden />
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
        <div className="flex shrink-0 items-center gap-1">
          <StatusChip status={citation.code_status} />
          <VerificationBadge level={citation.verification} authority={citation.source_authority} />
        </div>
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
