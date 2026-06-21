"use client";

import Link from "next/link";
import { Scale } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { appConfig } from "@/lib/api";
import { LanguageToggle } from "@/components/LanguageToggle";

export default function AboutPage() {
  const { dict } = useI18n();
  const a = dict.about;

  return (
    <div className="min-h-dvh">
      <header className="sticky top-0 z-10 border-b border-border bg-bg/80 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between gap-3 px-4 py-3">
          <Link href="/" className="flex items-center gap-2.5">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary text-primary-fg">
              <Scale size={18} aria-hidden />
            </span>
            <p className="font-semibold tracking-tight">{appConfig.name}</p>
          </Link>
          <LanguageToggle />
        </div>
      </header>

      <main className="mx-auto max-w-3xl space-y-8 px-4 py-8">
        <h1 className="text-2xl font-semibold tracking-tight">{a.title}</h1>

        <section className="space-y-2">
          <h2 className="font-semibold">{a.whatTitle}</h2>
          <p className="text-sm leading-relaxed text-muted">{a.whatBody}</p>
        </section>

        <section className="space-y-2">
          <h2 className="font-semibold">{a.howTitle}</h2>
          <p className="text-sm leading-relaxed text-muted">{a.howBody}</p>
        </section>

        {/* The trust-critical bit: make the limits impossible to miss. */}
        <section className="space-y-3 rounded-2xl border border-amber-300 bg-amber-50 p-4">
          <h2 className="font-semibold text-amber-900">{a.limitsTitle}</h2>
          <ul className="list-disc space-y-1.5 pl-5 text-sm leading-relaxed text-amber-900">
            {a.limits.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </section>

        <section className="space-y-2">
          <h2 className="font-semibold">{a.privacyTitle}</h2>
          <p className="text-sm leading-relaxed text-muted">{a.privacyBody}</p>
        </section>

        <section className="space-y-2">
          <h2 className="font-semibold">{a.helpTitle}</h2>
          <p className="text-sm leading-relaxed text-muted">{a.helpBody}</p>
        </section>

        <p className="text-xs text-muted">{a.reviewNote}</p>

        <Link
          href="/"
          className="inline-block rounded-full bg-primary px-4 py-2 text-sm font-medium text-primary-fg"
        >
          {a.back}
        </Link>
      </main>
    </div>
  );
}
