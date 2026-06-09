"use client";

import { ShieldCheck, ShieldAlert, ShieldQuestion } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/cn";
import type { Confidence } from "@/lib/types";

const STYLES: Record<Confidence, { cls: string; Icon: typeof ShieldCheck }> = {
  high: { cls: "bg-success/10 text-success", Icon: ShieldCheck },
  medium: { cls: "bg-warning/10 text-warning", Icon: ShieldQuestion },
  low: { cls: "bg-danger/10 text-danger", Icon: ShieldAlert },
};

export function ConfidenceBadge({ level }: { level: Confidence }) {
  const { t } = useI18n();
  const { cls, Icon } = STYLES[level];
  return (
    <span
      className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium", cls)}
      title={t("confidenceLabel")}
    >
      <Icon size={13} aria-hidden />
      {t(`confidence.${level}`)}
    </span>
  );
}
