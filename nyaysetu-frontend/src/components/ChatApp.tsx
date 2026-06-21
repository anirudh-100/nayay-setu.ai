"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useI18n } from "@/lib/i18n";
import { askQuestion, analyzeFile, draftDocument, ApiError } from "@/lib/api";
import type { ChatMessage } from "@/lib/types";
import { Header } from "./Header";
import { MessageBubble } from "./MessageBubble";
import { Composer } from "./Composer";
import { DraftModal } from "./DraftModal";
import { Home } from "./Home";

let counter = 0;
const nextId = () => `m${++counter}`;

export function ChatApp() {
  const { t } = useI18n();
  const { locale } = useI18n();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [draftOpen, setDraftOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (text: string) => {
    const userMsg: ChatMessage = { id: nextId(), role: "user", text };
    const pendingId = nextId();
    setMessages((prev) => [...prev, userMsg, { id: pendingId, role: "assistant", pending: true }]);
    setLoading(true);

    try {
      const response = await askQuestion(text, locale);
      setMessages((prev) =>
        prev.map((m) => (m.id === pendingId ? { ...m, pending: false, response } : m))
      );
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "generic";
      setMessages((prev) =>
        prev.map((m) => (m.id === pendingId ? { ...m, pending: false, error: message } : m))
      );
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeFile = async (file: File) => {
    const userMsg: ChatMessage = { id: nextId(), role: "user", text: `📄 ${file.name}` };
    const pendingId = nextId();
    setMessages((prev) => [
      ...prev,
      userMsg,
      { id: pendingId, role: "assistant", pending: true, analysisPending: true },
    ]);
    setLoading(true);

    try {
      const analysis = await analyzeFile(file, locale);
      setMessages((prev) =>
        prev.map((m) => (m.id === pendingId ? { ...m, pending: false, analysis } : m))
      );
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "generic";
      setMessages((prev) =>
        prev.map((m) => (m.id === pendingId ? { ...m, pending: false, error: message } : m))
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDraft = async (
    journeyId: string,
    fields: Record<string, string | boolean>,
    applicantName?: string,
    applicantAddress?: string
  ) => {
    setDraftOpen(false);
    const userMsg: ChatMessage = { id: nextId(), role: "user", text: `📝 ${t("draftLaunch")}` };
    const pendingId = nextId();
    setMessages((prev) => [
      ...prev,
      userMsg,
      { id: pendingId, role: "assistant", pending: true, draftPending: true },
    ]);
    setLoading(true);

    try {
      const draft = await draftDocument({
        journey: journeyId,
        fields,
        applicant_name: applicantName,
        applicant_address: applicantAddress,
        language: locale,
      });
      setMessages((prev) =>
        prev.map((m) => (m.id === pendingId ? { ...m, pending: false, draft } : m))
      );
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "generic";
      setMessages((prev) =>
        prev.map((m) => (m.id === pendingId ? { ...m, pending: false, error: message } : m))
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setMessages([]);
    setLoading(false);
  };

  const empty = messages.length === 0;

  return (
    <div className="flex min-h-dvh flex-col">
      <Header onReset={reset} showReset={!empty} />

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4">
        {empty ? (
          <Home
            onPickExample={handleSend}
            onAttach={handleAnalyzeFile}
            onOpenDraft={() => setDraftOpen(true)}
            onFocusAsk={() => composerRef.current?.focus()}
          />
        ) : (
          <div className="flex-1 space-y-4 py-5">
            {messages.map((m) => (
              <MessageBubble key={m.id} message={m} />
            ))}
            <div ref={bottomRef} />
          </div>
        )}

        {/* Composer sticks to the bottom; examples show only on the empty screen. */}
        <div className="sticky bottom-0 bg-gradient-to-t from-bg via-bg to-transparent pb-4 pt-2">
          <Composer
            onSend={handleSend}
            onAttach={handleAnalyzeFile}
            onOpenDraft={() => setDraftOpen(true)}
            inputRef={composerRef}
            loading={loading}
            showExamples={false}
          />
          <p className="mt-2 text-center text-[11px] text-muted">
            {t("footerNote")}{" · "}
            <Link href="/about" className="underline underline-offset-2 hover:opacity-80">
              {t("aboutLink")}
            </Link>
          </p>
        </div>
      </main>

      {draftOpen && <DraftModal onSubmit={handleDraft} onClose={() => setDraftOpen(false)} />}
    </div>
  );
}
