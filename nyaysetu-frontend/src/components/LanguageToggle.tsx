"use client";

import { useI18n, type Locale } from "@/lib/i18n";
import { cn } from "@/lib/cn";

const OPTIONS: { value: Locale; label: string }[] = [
  { value: "en", label: "EN" },
  { value: "hi", label: "हिं" },
];

export function LanguageToggle() {
  const { locale, setLocale } = useI18n();
  return (
    <div
      role="group"
      aria-label="Language"
      className="inline-flex items-center rounded-full border border-border bg-surface p-0.5"
    >
      {OPTIONS.map((opt) => (
        <button
          key={opt.value}
          onClick={() => setLocale(opt.value)}
          aria-pressed={locale === opt.value}
          className={cn(
            "min-w-[2.25rem] rounded-full px-2.5 py-1 text-sm font-medium transition-colors",
            locale === opt.value
              ? "bg-primary text-primary-fg"
              : "text-muted hover:text-ink"
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}
