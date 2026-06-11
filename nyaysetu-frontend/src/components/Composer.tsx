"use client";

import { useRef, useState } from "react";
import { ArrowUp, Paperclip } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/cn";

interface ComposerProps {
  onSend: (text: string) => void;
  onAttach: (file: File) => void;
  loading: boolean;
  showExamples: boolean;
}

// File types the backend's /analyze/file endpoint can extract text from.
const ACCEPT = ".pdf,.txt,.md,.markdown,.text";

export function Composer({ onSend, onAttach, loading, showExamples }: ComposerProps) {
  const { t, dict } = useI18n();
  const [value, setValue] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

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
        {/* Hidden file input — opened by the paperclip; analyses a document on select. */}
        <input
          ref={fileRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onAttach(file);
            e.target.value = ""; // allow re-selecting the same file
          }}
        />
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          disabled={loading}
          aria-label={t("attachDocument")}
          title={t("attachDocument")}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-muted transition-colors hover:bg-primary-soft/30 hover:text-primary disabled:opacity-50"
        >
          <Paperclip size={18} aria-hidden />
        </button>
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
