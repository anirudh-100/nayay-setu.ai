"use client";

import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import en from "@/i18n/en.json";
import hi from "@/i18n/hi.json";

export type Locale = "en" | "hi";
type Dict = typeof en;

const DICTS: Record<Locale, Dict> = { en, hi };
const STORAGE_KEY = "nyaysetu.locale";

interface I18nContextValue {
  locale: Locale;
  setLocale: (l: Locale) => void;
  dict: Dict;
  /** Translate a dotted key, e.g. t("confidence.high"). Supports {var} interpolation. */
  t: (key: string, vars?: Record<string, string | number>) => string;
}

const I18nContext = createContext<I18nContextValue | null>(null);

function resolve(dict: Dict, key: string): string {
  const value = key.split(".").reduce<unknown>((acc, part) => {
    if (acc && typeof acc === "object" && part in acc) return (acc as Record<string, unknown>)[part];
    return undefined;
  }, dict);
  return typeof value === "string" ? value : key;
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("en");

  // Restore saved choice on mount (client-only, avoids hydration mismatch).
  useEffect(() => {
    const saved = window.localStorage.getItem(STORAGE_KEY) as Locale | null;
    if (saved === "en" || saved === "hi") setLocaleState(saved);
  }, []);

  const setLocale = (l: Locale) => {
    setLocaleState(l);
    window.localStorage.setItem(STORAGE_KEY, l);
    document.documentElement.lang = l;
  };

  const value = useMemo<I18nContextValue>(() => {
    const dict = DICTS[locale];
    return {
      locale,
      setLocale,
      dict,
      t: (key, vars) => {
        let out = resolve(dict, key);
        if (vars) for (const [k, v] of Object.entries(vars)) out = out.replace(`{${k}}`, String(v));
        return out;
      },
    };
  }, [locale]);

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18nContextValue {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n must be used within I18nProvider");
  return ctx;
}
