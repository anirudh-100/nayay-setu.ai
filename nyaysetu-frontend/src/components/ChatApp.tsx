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
// Unique across sessions (timestamp) and within a session (counter) — so restored
// messages never collide with new ones.
const nextId = () => `m${Date.now().toString(36)}${++counter}`;

const CHAT_KEY = "nyaysetu.chat";

/** Recent settled turns, sent so the backend can resolve follow-ups ("what about that?"). */
function buildHistory(messages: ChatMessage[]): { role: string; content: string }[] {
  const turns: { role: string; content: string }[] = [];
  for (const m of messages) {
    if (m.pending || m.error) continue;
    if (m.role === "user" && m.text) turns.push({ role: "user", content: m.text });
    else if (m.role === "assistant" && m.response) turns.push({ role: "assistant", content: m.response.answer });
  }
  return turns.slice(-6);
}

export function ChatApp() {
  const { t } = useI18n();
  const { locale } = useI18n();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [draftOpen, setDraftOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);

  // Restore the conversation on mount so a refresh doesn't wipe it.
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(CHAT_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as ChatMessage[];
        if (Array.isArray(parsed) && parsed.length) setMessages(parsed);
      }
    } catch {
      /* ignore corrupt storage */
    }
  }, []);

  // Persist settled messages (never removes here — only reset() clears, avoiding a mount race).
  useEffect(() => {
    const settled = messages.filter((m) => !m.pending);
    if (settled.length) {
      try {
        window.localStorage.setItem(CHAT_KEY, JSON.stringify(settled.slice(-30)));
      } catch {
        /* ignore quota errors */
      }
    }
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (text: string) => {
    const history = buildHistory(messages); // capture context before adding the new turn
    const userMsg: ChatMessage = { id: nextId(), role: "user", text };
    const pendingId = nextId();
    setMessages((prev) => [...prev, userMsg, { id: pendingId, role: "assistant", pending: true }]);
    setLoading(true);

    try {
      const response = await askQuestion(text, locale, history);
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
    try {
      window.localStorage.removeItem(CHAT_KEY);
    } catch {
      /* ignore */
    }
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
            {messages.map((m, i) => (
              <MessageBubble
                key={m.id}
                message={m}
                question={messages[i - 1]?.role === "user" ? messages[i - 1].text : undefined}
              />
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
