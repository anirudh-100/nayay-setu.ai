"use client";

import { Scale } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { appConfig } from "@/lib/api";
import { LanguageToggle } from "./LanguageToggle";

export function Header({ onReset, showReset }: { onReset: () => void; showReset: boolean }) {
  const { t } = useI18n();
  return (
    <header className="sticky top-0 z-10 border-b border-border bg-bg/80 backdrop-blur">
      <div className="mx-auto flex max-w-3xl items-center justify-between gap-3 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary text-primary-fg">
            <Scale size={18} aria-hidden />
          </span>
          <div className="leading-tight">
            <p className="font-semibold tracking-tight">{appConfig.name}</p>
            <p className="text-xs text-muted">{t("tagline")}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {showReset && (
            <button
              onClick={onReset}
              className="rounded-full border border-border px-3 py-1.5 text-sm text-muted transition-colors hover:text-ink"
            >
              {t("newChat")}
            </button>
          )}
          <LanguageToggle />
        </div>
      </div>
    </header>
  );
}
