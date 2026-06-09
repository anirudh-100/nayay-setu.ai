"use client";

import { AlertCircle } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import type { ChatMessage } from "@/lib/types";
import { AnswerCard } from "./AnswerCard";

function Thinking() {
  const { t } = useI18n();
  return (
    <div className="flex items-center gap-2 rounded-2xl border border-border bg-surface px-4 py-3 text-sm text-muted">
      <span className="flex gap-1">
        <span className="h-1.5 w-1.5 animate-blink rounded-full bg-primary" />
        <span className="h-1.5 w-1.5 animate-blink rounded-full bg-primary [animation-delay:200ms]" />
        <span className="h-1.5 w-1.5 animate-blink rounded-full bg-primary [animation-delay:400ms]" />
      </span>
      {t("thinking")}
    </div>
  );
}

function ErrorBubble({ message }: { message: string }) {
  const { t } = useI18n();
  const body = message === "network" ? t("errorNetwork") : message || t("errorGeneric");
  return (
    <div className="flex gap-2 rounded-2xl border border-danger/30 bg-danger/5 p-4 text-sm">
      <AlertCircle size={18} className="mt-0.5 shrink-0 text-danger" aria-hidden />
      <div>
        <p className="font-medium text-ink">{t("errorTitle")}</p>
        <p className="mt-0.5 leading-relaxed text-muted">{body}</p>
      </div>
    </div>
  );
}

export function MessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-[15px] leading-relaxed text-primary-fg shadow-sm">
          {message.text}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="w-full max-w-full">
        {message.pending ? (
          <Thinking />
        ) : message.error ? (
          <ErrorBubble message={message.error} />
        ) : message.response ? (
          <AnswerCard response={message.response} />
        ) : null}
      </div>
    </div>
  );
}
