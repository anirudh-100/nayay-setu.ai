"use client";

import { useRef } from "react";
import { Search, FileText, ScrollText } from "lucide-react";
import { useI18n } from "@/lib/i18n";

interface HomeProps {
  onPickExample: (text: string) => void;
  onAttach: (file: File) => void;
  onOpenDraft: () => void;
  onFocusAsk: () => void;
}

// Same set the /analyze/file endpoint accepts.
const ACCEPT = ".pdf,.txt,.md,.markdown,.text";

/** The empty-state home: surfaces the three things a citizen can actually do here
 * (ask, understand a document, draft a letter) so the capabilities aren't hidden behind
 * a bare chat box, plus example questions that hint at the breadth (statutes, procedure,
 * case law). */
export function Home({ onPickExample, onAttach, onOpenDraft, onFocusAsk }: HomeProps) {
  const { t, dict } = useI18n();
  const fileRef = useRef<HTMLInputElement>(null);

  const cards = [
    { key: "ask", Icon: Search, onClick: onFocusAsk },
    { key: "understand", Icon: FileText, onClick: () => fileRef.current?.click() },
    { key: "draft", Icon: ScrollText, onClick: onOpenDraft },
  ] as const;

  return (
    <div className="flex flex-1 flex-col justify-center py-8">
      <h1 className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl">
        {t("welcomeTitle")}
      </h1>
      <p className="mt-2 max-w-xl text-pretty leading-relaxed text-muted">{t("welcomeBody")}</p>

      {/* Capability cards — the activation unlock */}
      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        {cards.map(({ key, Icon, onClick }) => (
          <button
            key={key}
            onClick={onClick}
            className="group flex flex-col items-start gap-2 rounded-2xl border border-border bg-surface p-4 text-left shadow-sm transition-colors hover:border-primary/40 hover:bg-primary-soft/20"
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary-soft/60 text-primary transition-transform group-hover:scale-105">
              <Icon size={18} aria-hidden />
            </span>
            <span className="text-sm font-semibold text-ink">{t(`home.${key}.title`)}</span>
            <span className="text-xs leading-relaxed text-muted">{t(`home.${key}.body`)}</span>
          </button>
        ))}
      </div>

      {/* Hidden input opened by the "understand a document" card */}
      <input
        ref={fileRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onAttach(file);
          e.target.value = "";
        }}
      />

      {/* Example questions */}
      <p className="mb-1.5 mt-7 text-xs font-medium text-muted">{t("examplesLabel")}</p>
      <div className="flex flex-wrap gap-2">
        {dict.examples.map((ex) => (
          <button
            key={ex}
            onClick={() => onPickExample(ex)}
            className="rounded-full border border-border bg-surface px-3 py-1.5 text-left text-sm text-ink transition-colors hover:border-primary/40 hover:bg-primary-soft/30"
          >
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}
