"use client";

import { useRef, useState } from "react";
import { ArrowUp } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/cn";

interface ComposerProps {
  onSend: (text: string) => void;
  loading: boolean;
  showExamples: boolean;
}

export function Composer({ onSend, loading, showExamples }: ComposerProps) {
  const { t, dict } = useI18n();
  const [value, setValue] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);

  const submit = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    onSend(trimmed);
    setValue("");
    if (taRef.current) taRef.current.style.height = "auto";
  };

  const grow = () => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  };

  return (
    <div className="space-y-3">
      {showExamples && (
        <div>
          <p className="mb-1.5 text-xs font-medium text-muted">{t("examplesLabel")}</p>
          <div className="flex flex-wrap gap-2">
            {dict.examples.map((ex) => (
              <button
                key={ex}
                onClick={() => submit(ex)}
                disabled={loading}
                className="rounded-full border border-border bg-surface px-3 py-1.5 text-left text-sm text-ink transition-colors hover:border-primary/40 hover:bg-primary-soft/30 disabled:opacity-50"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex items-end gap-2 rounded-2xl border border-border bg-surface p-2 shadow-sm focus-within:border-primary/50">
        <textarea
          ref={taRef}
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            grow();
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit(value);
            }
          }}
          rows={1}
          placeholder={t("placeholder")}
          aria-label={t("placeholder")}
          className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-relaxed outline-none placeholder:text-muted"
        />
        <button
          onClick={() => submit(value)}
          disabled={loading || !value.trim()}
          aria-label={t("send")}
          className={cn(
            "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl transition-colors",
            loading || !value.trim()
              ? "bg-border text-muted"
              : "bg-primary text-primary-fg hover:opacity-90"
          )}
        >
          <ArrowUp size={18} aria-hidden />
        </button>
      </div>
    </div>
  );
}
