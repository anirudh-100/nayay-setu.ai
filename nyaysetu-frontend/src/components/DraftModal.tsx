"use client";

import { useEffect, useState } from "react";
import {
  X,
  ArrowLeft,
  ChevronRight,
  FileText,
  ScrollText,
  ShoppingBag,
  Shield,
  AlertCircle,
} from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/cn";
import { listJourneys, ApiError } from "@/lib/api";
import type { JourneyInfo, FieldSpec } from "@/lib/types";

interface DraftModalProps {
  onSubmit: (
    journeyId: string,
    fields: Record<string, string | boolean>,
    applicantName?: string,
    applicantAddress?: string
  ) => void;
  onClose: () => void;
}

const ICONS: Record<string, typeof FileText> = {
  "scroll-text": ScrollText,
  "shopping-bag": ShoppingBag,
  shield: Shield,
  "file-text": FileText,
};

function initialValues(j: JourneyInfo): Record<string, string | boolean> {
  const v: Record<string, string | boolean> = {};
  for (const f of j.fields) v[f.key] = f.kind === "checkbox" ? false : f.default ?? "";
  return v;
}

export function DraftModal({ onSubmit, onClose }: DraftModalProps) {
  const { t } = useI18n();
  const [journeys, setJourneys] = useState<JourneyInfo[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<JourneyInfo | null>(null);
  const [values, setValues] = useState<Record<string, string | boolean>>({});
  const [name, setName] = useState("");
  const [address, setAddress] = useState("");

  useEffect(() => {
    let active = true;
    listJourneys()
      .then((js) => active && setJourneys(js))
      .catch((e) => active && setLoadError(e instanceof ApiError ? e.message : "generic"));
    return () => {
      active = false;
    };
  }, []);

  const pick = (j: JourneyInfo) => {
    setSelected(j);
    setValues(initialValues(j));
  };

  const canSubmit =
    selected != null &&
    selected.fields.every((f) => !f.required || String(values[f.key] ?? "").trim().length > 0);

  const submit = () => {
    if (!selected || !canSubmit) return;
    onSubmit(selected.id, values, name.trim() || undefined, address.trim() || undefined);
  };

  const inputCls =
    "w-full rounded-xl border border-border bg-bg px-3 py-2 text-[15px] text-ink outline-none placeholder:text-muted focus:border-primary/50";
  const labelCls = "mb-1 block text-xs font-semibold uppercase tracking-wide text-muted";

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-black/40 p-0 sm:items-center sm:p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={t("draftMenuTitle")}
    >
      <div
        className="max-h-[92dvh] w-full max-w-lg overflow-auto rounded-t-2xl border border-border bg-surface p-5 shadow-xl sm:rounded-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="mb-4 flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            {selected && (
              <button
                onClick={() => setSelected(null)}
                aria-label={t("draftBack")}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-muted hover:bg-bg hover:text-ink"
              >
                <ArrowLeft size={18} aria-hidden />
              </button>
            )}
            <h2 className="text-lg font-semibold tracking-tight text-ink">
              {selected ? selected.title : t("draftMenuTitle")}
            </h2>
          </div>
          <button
            onClick={onClose}
            aria-label={t("draftCancel")}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-muted hover:bg-bg hover:text-ink"
          >
            <X size={18} aria-hidden />
          </button>
        </div>

        {/* Step 1: pick a journey */}
        {!selected && (
          <div className="space-y-2">
            {loadError && (
              <div className="flex gap-2 rounded-xl border border-danger/30 bg-danger/5 p-3 text-sm">
                <AlertCircle size={18} className="mt-0.5 shrink-0 text-danger" aria-hidden />
                <p className="text-muted">
                  {loadError === "network" ? t("errorNetwork") : t("errorGeneric")}
                </p>
              </div>
            )}
            {!journeys && !loadError && <p className="text-sm text-muted">{t("draftLoading")}</p>}
            {journeys?.map((j) => {
              const Icon = ICONS[j.icon ?? "file-text"] ?? FileText;
              return (
                <button
                  key={j.id}
                  onClick={() => pick(j)}
                  className="flex w-full items-center gap-3 rounded-xl border border-border bg-bg p-3 text-left transition-colors hover:border-primary/40 hover:bg-primary-soft/20"
                >
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary-soft/60 text-primary">
                    <Icon size={18} aria-hidden />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-medium text-ink">{j.title}</span>
                    <span className="block text-xs leading-relaxed text-muted">{j.description}</span>
                  </span>
                  <ChevronRight size={18} className="shrink-0 text-muted" aria-hidden />
                </button>
              );
            })}
          </div>
        )}

        {/* Step 2: dynamic form */}
        {selected && (
          <div className="space-y-4">
            {selected.note && (
              <p className="rounded-xl border border-border bg-bg px-3 py-2 text-xs leading-relaxed text-muted">
                {selected.note}
              </p>
            )}

            {selected.fields.map((f) => (
              <FieldInput
                key={f.key}
                field={f}
                value={values[f.key]}
                onChange={(v) => setValues((prev) => ({ ...prev, [f.key]: v }))}
                inputCls={inputCls}
                labelCls={labelCls}
              />
            ))}

            {/* Shared applicant details */}
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label className={labelCls} htmlFor="draft-name">
                  {t("draftNameLabel")}
                </label>
                <input
                  id="draft-name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder={t("draftNamePlaceholder")}
                  className={inputCls}
                />
              </div>
              <div>
                <label className={labelCls} htmlFor="draft-address">
                  {t("draftAddressLabel")}
                </label>
                <input
                  id="draft-address"
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  placeholder={t("draftAddressPlaceholder")}
                  className={inputCls}
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-1">
              <button
                onClick={() => setSelected(null)}
                className="rounded-xl border border-border bg-bg px-4 py-2 text-sm font-medium text-ink hover:bg-surface"
              >
                {t("draftBack")}
              </button>
              <button
                onClick={submit}
                disabled={!canSubmit}
                className={cn(
                  "rounded-xl px-4 py-2 text-sm font-semibold transition-colors",
                  canSubmit ? "bg-primary text-primary-fg hover:opacity-90" : "bg-border text-muted"
                )}
              >
                {t("draftSubmit")}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function FieldInput({
  field,
  value,
  onChange,
  inputCls,
  labelCls,
}: {
  field: FieldSpec;
  value: string | boolean | undefined;
  onChange: (v: string | boolean) => void;
  inputCls: string;
  labelCls: string;
}) {
  if (field.kind === "checkbox") {
    return (
      <label className="flex cursor-pointer items-start gap-2 rounded-xl border border-border bg-bg p-3 text-sm text-ink">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          className="mt-0.5 h-4 w-4 accent-primary"
        />
        <span>{field.label}</span>
      </label>
    );
  }

  return (
    <div>
      <label className={labelCls} htmlFor={`draft-${field.key}`}>
        {field.label}
        {field.required && <span className="text-danger"> *</span>}
      </label>
      {field.kind === "select" ? (
        <select
          id={`draft-${field.key}`}
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          className={inputCls}
        >
          {field.options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      ) : field.kind === "textarea" ? (
        <textarea
          id={`draft-${field.key}`}
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          rows={3}
          placeholder={field.placeholder ?? ""}
          className={cn(inputCls, "resize-none")}
        />
      ) : (
        <input
          id={`draft-${field.key}`}
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          placeholder={field.placeholder ?? ""}
          className={inputCls}
        />
      )}
      {field.help && <p className="mt-1 text-xs text-muted">{field.help}</p>}
    </div>
  );
}
