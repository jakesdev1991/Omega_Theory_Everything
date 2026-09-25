// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"use client";

import { useEffect, useState } from "react";

type RailState = {
  configured: boolean;
  network: string;
  message: string;
};

type StatusState =
  | { status: "checking" }
  | { status: "hidden" }
  | { status: "error"; error: string }
  | {
      status: "show";
      ready: boolean;
      rails: {
        omega: RailState;
        twc: RailState;
      };
    };

export function UnlockRailBanner() {
  const [state, setState] = useState<StatusState>({ status: "checking" });

  useEffect(() => {
    fetch("/api/unlock/status", { cache: "no-store" })
      .then((response) => response.json())
      .then((data) => {
        if (!data.ok) {
          setState({ status: "error", error: data.error || "Unable to read unlock-rail status." });
          return;
        }

        if (data.ready) {
          setState({ status: "hidden" });
          return;
        }

        setState({
          status: "show",
          ready: !!data.ready,
          rails: data.rails,
        });
      })
      .catch((error) => {
        setState({ status: "error", error: error instanceof Error ? error.message : "Unable to read unlock-rail status." });
      });
  }, []);

  if (state.status === "checking" || state.status === "hidden") {
    return null;
  }

  if (state.status === "error") {
    return (
      <div
        style={{
          margin: "24px 0 0",
          padding: "16px 18px",
          border: "1px solid var(--color-care)",
          borderRadius: "12px",
          background: "rgba(244,114,182,0.06)",
          color: "var(--color-muted-strong)",
        }}
      >
        <div style={{ fontWeight: 600, color: "var(--color-care)", marginBottom: "6px" }}>
          Unlock rail status unavailable
        </div>
        <div style={{ fontSize: "14px", lineHeight: 1.55 }}>{state.error}</div>
      </div>
    );
  }

  return (
    <div
      style={{
        margin: "24px 0 0",
        padding: "18px 20px",
        border: "1px solid var(--color-care)",
        borderRadius: "12px",
        background: "rgba(244,114,182,0.06)",
        color: "var(--color-muted-strong)",
      }}
    >
      <div style={{ fontWeight: 600, color: "var(--color-care)", marginBottom: "8px" }}>
        Operator setup still required
      </div>
      <p style={{ margin: "0 0 12px", fontSize: "14px", lineHeight: 1.6 }}>
        This reader now fails closed unless the web app has local pilot deployment metadata or explicit
        environment variables for the two wired currencies. At least one unlock rail is still unconfigured.
      </p>
      <ul style={{ margin: 0, paddingLeft: "18px", fontSize: "14px", lineHeight: 1.6 }}>
        <li>
          <strong>$OMEGA</strong> ({state.rails.omega.network}) — {state.rails.omega.message}
        </li>
        <li>
          <strong>TWC</strong> ({state.rails.twc.network}) — {state.rails.twc.message}
        </li>
      </ul>
    </div>
  );
}
