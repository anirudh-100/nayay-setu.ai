"use client";

import { useState } from "react";
import { X, ScrollText } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/lib/cn";
import type { GovLevel, RTIDraftRequest } from "@/lib/types";

interface RtiFormProps {
  onSubmit: (req: Omit<RTIDraftRequest, "language">) => void;
  onClose: () => void;
}

export function RtiForm({ onSubmit, onClose }: RtiFormProps) {
  const { t } = useI18n();
  const [subject, setSubject] = useState("");
  const [authority, setAuthority] = useState("");
  const [level, setLevel] = useState<GovLevel>("central");
  const [name, setName] = useState("");
  const [address, setAddress] = useState("");
  const [isBpl, setIsBpl] = useState(false);

  const canSubmit = subject.trim().length >= 5;

  const submit = () => {
    if (!canSubmit) return;
    onSubmit({
      subject: subject.trim(),
      public_authority: authority.trim() || undefined,
      level,
      applicant_name: name.trim() || undefined,
      applicant_address: address.trim() || undefined,
      is_bpl: isBpl,
    });
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
      aria-label={t("rtiTitle")}
    >
      <div
        className="max-h-[92dvh] w-full max-w-lg overflow-auto rounded-t-2xl border border-border bg-surface p-5 shadow-xl sm:rounded-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="mb-1 flex items-start justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary-soft/60 text-primary">
              <ScrollText size={17} aria-hidden />
            </span>
            <h2 className="text-lg font-semibold tracking-tight text-ink">{t("rtiTitle")}</h2>
          </div>
          <button
            onClick={onClose}
            aria-label={t("rtiCancel")}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-muted hover:bg-bg hover:text-ink"
          >
            <X size={18} aria-hidden />
          </button>
        </div>
        <p className="mb-4 text-sm leading-relaxed text-muted">{t("rtiSubtitle")}</p>

        <div className="space-y-4">
          {/* Subject (required) */}
          <div>
            <label className={labelCls} htmlFor="rti-subject">
              {t("rtiSubjectLabel")}
            </label>
            <textarea
              id="rti-subject"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              rows={3}
              placeholder={t("rtiSubjectPlaceholder")}
              className={cn(inputCls, "resize-none")}
              autoFocus
            />
          </div>

          {/* Public authority (optional) */}
          <div>
            <label className={labelCls} htmlFor="rti-authority">
              {t("rtiAuthorityLabel")}
            </label>
            <input
              id="rti-authority"
              value={authority}
              onChange={(e) => setAuthority(e.target.value)}
              placeholder={t("rtiAuthorityPlaceholder")}
              className={inputCls}
            />
          </div>

          {/* Government level */}
          <div>
            <span className={labelCls}>{t("rtiLevelLabel")}</span>
            <div className="flex gap-2">
              {(["central", "state"] as GovLevel[]).map((lv) => (
                <button
                  key={lv}
                  type="button"
                  onClick={() => setLevel(lv)}
                  className={cn(
                    "flex-1 rounded-xl border px-3 py-2 text-sm font-medium transition-colors",
                    level === lv
                      ? "border-primary bg-primary-soft/40 text-primary"
                      : "border-border bg-bg text-muted hover:border-primary/40"
                  )}
                >
                  {t(lv === "central" ? "rtiLevelCentral" : "rtiLevelState")}
                </button>
              ))}
            </div>
          </div>

          {/* Name + address (optional) */}
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className={labelCls} htmlFor="rti-name">
                {t("rtiNameLabel")}
              </label>
              <input
                id="rti-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder={t("rtiNamePlaceholder")}
                className={inputCls}
              />
            </div>
            <div>
              <label className={labelCls} htmlFor="rti-address">
                {t("rtiAddressLabel")}
              </label>
              <input
                id="rti-address"
                value={address}
                onChange={(e) => setAddress(e.target.value)}
                placeholder={t("rtiAddressPlaceholder")}
                className={inputCls}
              />
            </div>
          </div>

          {/* BPL exemption */}
          <label className="flex cursor-pointer items-start gap-2 rounded-xl border border-border bg-bg p-3 text-sm text-ink">
            <input
              type="checkbox"
              checked={isBpl}
              onChange={(e) => setIsBpl(e.target.checked)}
              className="mt-0.5 h-4 w-4 accent-primary"
            />
            <span>{t("rtiBplLabel")}</span>
          </label>
        </div>

        {/* Actions */}
        <div className="mt-5 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="rounded-xl border border-border bg-bg px-4 py-2 text-sm font-medium text-ink hover:bg-surface"
          >
            {t("rtiCancel")}
          </button>
          <button
            onClick={submit}
            disabled={!canSubmit}
            className={cn(
              "rounded-xl px-4 py-2 text-sm font-semibold transition-colors",
              canSubmit ? "bg-primary text-primary-fg hover:opacity-90" : "bg-border text-muted"
            )}
          >
            {t("rtiSubmit")}
          </button>
        </div>
      </div>
    </div>
  );
}
