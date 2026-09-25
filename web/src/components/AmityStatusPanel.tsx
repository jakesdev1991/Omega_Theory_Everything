// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"use client";

import { useEffect, useState } from "react";

type OperatorFileStatus = {
  label: string;
  configured: boolean;
  exists: boolean;
};

type AmityStatusPayload = {
  ok: true;
  amity: {
    stage: string;
    wiredIntoUnlockFlow: boolean;
    holderVerificationReady: boolean;
    scaffoldConfigured: boolean;
    operatorFilesReady: boolean;
    network: string;
    manifestPath: string;
    manifestPresent: boolean;
    manifestValid: boolean;
    localEnvPath: string;
    localEnvPresent: boolean;
    configSource: string;
    assetId: string | null;
    assetName: string | null;
    unitName: string | null;
    universeUrl: string | null;
    litdRpcConfigured: boolean;
    tapdRpcConfigured: boolean;
    litdRpcStatus: string;
    tapdRpcStatus: string;
    operatorFilesPresentCount: number;
    operatorFilesRequiredCount: number;
    operatorFiles: OperatorFileStatus[];
    message: string;
    warnings: string[];
  };
};

type State =
  | { status: "loading" }
  | { status: "error"; error: string }
  | { status: "ready"; payload: AmityStatusPayload };

function Pill({ label, tone }: { label: string; tone: "success" | "warning" | "muted" }) {
  const styles =
    tone === "success"
      ? {
          border: "1px solid var(--color-unlock-soft)",
          background: "rgba(52,211,153,0.08)",
          color: "var(--color-unlock)",
        }
      : tone === "warning"
        ? {
            border: "1px solid var(--color-care)",
            background: "rgba(244,114,182,0.08)",
            color: "var(--color-care)",
          }
        : {
            border: "1px solid var(--color-border-strong)",
            background: "rgba(255,255,255,0.04)",
            color: "var(--color-muted-strong)",
          };

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "4px 10px",
        borderRadius: "999px",
        fontSize: "11px",
        fontWeight: 700,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        fontFamily: "ui-monospace, monospace",
        ...styles,
      }}
    >
      {label}
    </span>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "160px 1fr",
        gap: "12px",
        padding: "8px 0",
        borderTop: "1px solid var(--color-border)",
        alignItems: "start",
      }}
    >
      <div
        style={{
          fontFamily: "ui-monospace, monospace",
          fontSize: "11px",
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "var(--color-muted)",
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: "13px", color: "var(--color-muted-strong)", lineHeight: 1.6, wordBreak: "break-word" }}>
        {value}
      </div>
    </div>
  );
}

export function AmityStatusPanel() {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    fetch("/api/amity/status", { cache: "no-store" })
      .then((response) => response.json())
      .then((data) => {
        if (!data.ok) {
          setState({ status: "error", error: data.error || "Unable to load AMITY scaffold status." });
          return;
        }
        setState({ status: "ready", payload: data });
      })
      .catch((error) => {
        setState({ status: "error", error: error instanceof Error ? error.message : "Unable to load AMITY scaffold status." });
      });
  }, []);

  if (state.status === "loading") {
    return <div className="gate-panel">Loading AMITY scaffold status…</div>;
  }

  if (state.status === "error") {
    return (
      <div className="gate-panel" style={{ borderColor: "var(--color-care)" }}>
        <div style={{ fontWeight: 600, marginBottom: "8px", color: "var(--color-care)" }}>
          AMITY scaffold status unavailable
        </div>
        <div style={{ fontSize: "14px", lineHeight: 1.6 }}>{state.error}</div>
      </div>
    );
  }

  const { amity } = state.payload;
  const headlineTone = amity.scaffoldConfigured ? (amity.operatorFilesReady ? "success" : "muted") : "warning";

  return (
    <div
      style={{
        padding: "24px",
        border: "1px solid var(--color-border-strong)",
        borderRadius: "16px",
        background: "rgba(255,255,255,0.015)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: "12px", alignItems: "center", flexWrap: "wrap", marginBottom: "14px" }}>
        <div>
          <div style={{ fontFamily: "ui-serif, Georgia, Cambria, serif", fontSize: "24px", fontWeight: 500, color: "var(--color-foreground)", marginBottom: "4px" }}>
            AMITY testnet scaffold
          </div>
          <div style={{ fontFamily: "ui-monospace, monospace", fontSize: "12px", color: "var(--color-accent)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            {amity.network} · separate from the two live unlock rails
          </div>
        </div>
        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", justifyContent: "flex-end" }}>
          <Pill label={amity.scaffoldConfigured ? "scaffold configured" : "setup missing"} tone={headlineTone} />
          <Pill label={amity.wiredIntoUnlockFlow ? "wired" : "not wired"} tone="muted" />
        </div>
      </div>

      <p style={{ margin: "0 0 14px", color: "var(--color-muted-strong)", fontSize: "14px", lineHeight: 1.6 }}>
        {amity.message}
      </p>

      <div>
        <DetailRow label="stage" value={amity.stage} />
        <DetailRow label="config source" value={amity.configSource} />
        <DetailRow label="manifest" value={amity.manifestPresent ? (amity.manifestValid ? "present and valid" : "present but invalid") : "missing"} />
        <DetailRow label="manifest path" value={amity.manifestPath} />
        <DetailRow label="local .env" value={amity.localEnvPresent ? `present (${amity.localEnvPath})` : `missing (${amity.localEnvPath})`} />
        <DetailRow label="asset id" value={amity.assetId ?? "not set"} />
        <DetailRow label="asset label" value={amity.assetName && amity.unitName ? `${amity.assetName} (${amity.unitName})` : "not set"} />
        <DetailRow label="universe" value={amity.universeUrl ?? "not set"} />
        <DetailRow label="litd endpoint" value={amity.litdRpcConfigured ? "configured" : "missing"} />
        <DetailRow label="tapd endpoint" value={amity.tapdRpcConfigured ? "configured" : "missing"} />
        <DetailRow label="operator files" value={`${amity.operatorFilesPresentCount}/${amity.operatorFilesRequiredCount} present`} />
        <DetailRow label="unlock status" value="AMITY remains outside the live wallet/web unlock flow; only $OMEGA and TWC are accepted today." />
      </div>

      <div style={{ marginTop: "16px", display: "grid", gap: "10px" }}>
        {amity.operatorFiles.map((entry) => (
          <div
            key={entry.label}
            style={{
              padding: "12px 14px",
              border: "1px solid var(--color-border)",
              borderRadius: "12px",
              background: "rgba(255,255,255,0.01)",
              display: "flex",
              justifyContent: "space-between",
              gap: "12px",
              flexWrap: "wrap",
            }}
          >
            <div style={{ color: "var(--color-foreground)", fontSize: "14px", fontWeight: 500 }}>{entry.label}</div>
            <div style={{ color: "var(--color-muted-strong)", fontSize: "13px" }}>
              {entry.configured ? (entry.exists ? "present" : "configured but missing") : "not configured yet"}
            </div>
          </div>
        ))}
      </div>

      {amity.warnings.length > 0 ? (
        <div
          style={{
            marginTop: "16px",
            padding: "14px 16px",
            border: "1px solid var(--color-care)",
            borderRadius: "12px",
            background: "rgba(244,114,182,0.06)",
          }}
        >
          <div style={{ fontWeight: 600, color: "var(--color-care)", marginBottom: "8px" }}>Warnings</div>
          <ul style={{ margin: 0, paddingLeft: "18px", color: "var(--color-muted-strong)", fontSize: "14px", lineHeight: 1.6 }}>
            {amity.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
