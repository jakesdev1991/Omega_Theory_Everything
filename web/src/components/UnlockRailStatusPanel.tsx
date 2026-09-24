"use client";

import { useEffect, useState } from "react";

type OmegaRail = {
  configured: boolean;
  network: string;
  rpcUrl: string;
  manifestPath: string;
  manifestPresent: boolean;
  gateAddress: string | null;
  tokenAddress: string | null;
  message: string;
};

type TwcRail = {
  configured: boolean;
  network: string;
  rpcUrl: string;
  manifestPath: string;
  manifestPresent: boolean;
  mintAddress: string | null;
  message: string;
};

type StatusPayload = {
  ok: true;
  ready: boolean;
  rails: {
    omega: OmegaRail;
    twc: TwcRail;
  };
};

type State =
  | { status: "loading" }
  | { status: "error"; error: string }
  | { status: "ready"; payload: StatusPayload };

function StatusPill({ ok }: { ok: boolean }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "6px",
        padding: "4px 10px",
        borderRadius: "999px",
        border: `1px solid ${ok ? "var(--color-unlock-soft)" : "var(--color-care)"}`,
        background: ok ? "rgba(52,211,153,0.08)" : "rgba(244,114,182,0.08)",
        color: ok ? "var(--color-unlock)" : "var(--color-care)",
        fontSize: "11px",
        fontWeight: 700,
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        fontFamily: "ui-monospace, monospace",
      }}
    >
      {ok ? "configured" : "missing setup"}
    </span>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "150px 1fr",
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
      <div
        style={{
          fontSize: "13px",
          color: "var(--color-muted-strong)",
          lineHeight: 1.6,
          wordBreak: "break-word",
          fontFamily: label.toLowerCase().includes("path") || label.toLowerCase().includes("address") ? "ui-monospace, monospace" : undefined,
        }}
      >
        {value}
      </div>
    </div>
  );
}

function RailCard({
  title,
  color,
  rail,
  lines,
}: {
  title: string;
  color: string;
  rail: OmegaRail | TwcRail;
  lines: Array<{ label: string; value: string }>;
}) {
  return (
    <div
      style={{
        padding: "24px",
        border: "1px solid var(--color-border-strong)",
        borderRadius: "16px",
        background: "rgba(255,255,255,0.015)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: "12px", alignItems: "center", marginBottom: "12px", flexWrap: "wrap" }}>
        <div>
          <div
            style={{
              fontFamily: "ui-serif, Georgia, Cambria, serif",
              fontSize: "24px",
              fontWeight: 500,
              color: "var(--color-foreground)",
              marginBottom: "4px",
            }}
          >
            {title}
          </div>
          <div
            style={{
              fontFamily: "ui-monospace, monospace",
              fontSize: "12px",
              color: color,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            {rail.network}
          </div>
        </div>
        <StatusPill ok={rail.configured} />
      </div>

      <p style={{ margin: "0 0 12px", color: "var(--color-muted-strong)", fontSize: "14px", lineHeight: 1.6 }}>
        {rail.message}
      </p>

      <div>
        <DetailRow label="manifest" value={rail.manifestPresent ? "present" : "missing"} />
        {lines.map((line) => (
          <DetailRow key={line.label} label={line.label} value={line.value} />
        ))}
      </div>
    </div>
  );
}

export function UnlockRailStatusPanel() {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    fetch("/api/unlock/status", { cache: "no-store" })
      .then((response) => response.json())
      .then((data) => {
        if (!data.ok) {
          setState({ status: "error", error: data.error || "Unable to load operator status." });
          return;
        }
        setState({ status: "ready", payload: data });
      })
      .catch((error) => {
        setState({ status: "error", error: error instanceof Error ? error.message : "Unable to load operator status." });
      });
  }, []);

  if (state.status === "loading") {
    return (
      <div className="gate-panel">Loading operator rail status…</div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="gate-panel" style={{ borderColor: "var(--color-care)" }}>
        <div style={{ fontWeight: 600, marginBottom: "8px", color: "var(--color-care)" }}>
          Operator status unavailable
        </div>
        <div style={{ fontSize: "14px", lineHeight: 1.6 }}>{state.error}</div>
      </div>
    );
  }

  const { payload } = state;

  return (
    <div style={{ display: "grid", gap: "16px" }}>
      <div
        style={{
          padding: "18px 20px",
          border: `1px solid ${payload.ready ? "var(--color-unlock-soft)" : "var(--color-care)"}`,
          borderRadius: "14px",
          background: payload.ready ? "rgba(52,211,153,0.06)" : "rgba(244,114,182,0.06)",
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: "8px", color: payload.ready ? "var(--color-unlock)" : "var(--color-care)" }}>
          {payload.ready ? "Both wired rails are locally configured" : "Local unlock activation is still blocked"}
        </div>
        <p style={{ margin: 0, fontSize: "14px", lineHeight: 1.6, color: "var(--color-muted-strong)" }}>
          The web reader requires both a valid wallet proof and an independent on-chain check. This panel shows whether the current sandbox has enough local pilot metadata to perform those checks.
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: "16px",
        }}
      >
        <RailCard
          title="$OMEGA rail"
          color="var(--color-omega)"
          rail={payload.rails.omega}
          lines={[
            { label: "rpc", value: payload.rails.omega.rpcUrl },
            { label: "manifest path", value: payload.rails.omega.manifestPath },
            { label: "gate address", value: payload.rails.omega.gateAddress ?? "not set" },
            { label: "token address", value: payload.rails.omega.tokenAddress ?? "not set" },
          ]}
        />

        <RailCard
          title="TWC rail"
          color="var(--color-twc)"
          rail={payload.rails.twc}
          lines={[
            { label: "rpc", value: payload.rails.twc.rpcUrl },
            { label: "manifest path", value: payload.rails.twc.manifestPath },
            { label: "mint address", value: payload.rails.twc.mintAddress ?? "not set" },
          ]}
        />
      </div>
    </div>
  );
}
