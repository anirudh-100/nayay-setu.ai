"use client";

import { useEffect, useRef, useState } from "react";
import { useI18n } from "@/lib/i18n";
import { askQuestion, analyzeFile, draftRti, ApiError } from "@/lib/api";
import type { ChatMessage, RTIDraftRequest } from "@/lib/types";
import { Header } from "./Header";
import { MessageBubble } from "./MessageBubble";
import { Composer } from "./Composer";
import { RtiForm } from "./RtiForm";

let counter = 0;
const nextId = () => `m${++counter}`;

export function ChatApp() {
  const { t } = useI18n();
  const { locale } = useI18n();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [rtiOpen, setRtiOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

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

  const handleDraftRti = async (req: Omit<RTIDraftRequest, "language">) => {
    setRtiOpen(false);
    const userMsg: ChatMessage = { id: nextId(), role: "user", text: `📝 ${req.subject}` };
    const pendingId = nextId();
    setMessages((prev) => [
      ...prev,
      userMsg,
      { id: pendingId, role: "assistant", pending: true, rtiPending: true },
    ]);
    setLoading(true);

    try {
      const rti = await draftRti({ ...req, language: locale });
      setMessages((prev) =>
        prev.map((m) => (m.id === pendingId ? { ...m, pending: false, rti } : m))
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
          <WelcomeView />
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
            onOpenRti={() => setRtiOpen(true)}
            loading={loading}
            showExamples={empty}
          />
          <p className="mt-2 text-center text-[11px] text-muted">{t("footerNote")}</p>
        </div>
      </main>

      {rtiOpen && <RtiForm onSubmit={handleDraftRti} onClose={() => setRtiOpen(false)} />}
    </div>
  );
}

function WelcomeView() {
  const { t } = useI18n();
  return (
    <div className="flex flex-1 flex-col justify-center py-8">
      <h1 className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl">
        {t("welcomeTitle")}
      </h1>
      <p className="mt-2 max-w-xl text-pretty leading-relaxed text-muted">{t("welcomeBody")}</p>
    </div>
  );
}
