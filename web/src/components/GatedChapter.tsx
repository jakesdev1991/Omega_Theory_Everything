// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const STORAGE_KEY = "omega.unlock.proof";

type State =
  | { status: "checking" }
  | { status: "locked" }
  | { status: "error"; error: string }
  | { status: "unlocked"; address: string; currency: string; network: string; unlocked: number }
  | { status: "denied"; address: string; currency?: string; network?: string; unlocked: number; error: string };

function readStagedProof(): {
  message: string;
  signature: string;
} | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    // Consume the tab-scoped proof immediately so a reload cannot silently
    // replay it. A failed verification requires the user to sign again.
    sessionStorage.removeItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && parsed.message && parsed.signature) return parsed;
    return null;
  } catch {
    return null;
  }
}

export function GatedChapter({ slug }: { slug: string }) {
  const [state, setState] = useState<State>({ status: "checking" });
  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    const staged = readStagedProof();
    if (!staged) {
      setState({ status: "locked" });
      return;
    }
    fetch(`/api/chapter/${slug}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(staged),
    })
      .then((r) => r.json())
      .then((data) => {
        if (data.ok) {
          setHtml(data.html);
          setState({
            status: "unlocked",
            address: data.address,
            currency: data.currency,
            network: data.network,
            unlocked: data.unlocked,
          });
        } else if (data.address && typeof data.unlocked === "number") {
          setState({
            status: "denied",
            address: data.address,
            currency: data.currency,
            network: data.network,
            unlocked: data.unlocked,
            error: data.error,
          });
        } else if (data.error) {
          setState({ status: "error", error: data.error });
        } else {
          setState({ status: "locked" });
        }
      })
      .catch((error) =>
        setState({
          status: "error",
          error: error instanceof Error ? error.message : "Unable to contact the optional reader-proof verifier.",
        }),
      );
  }, [slug]);

  if (state.status === "checking") {
    return (
      <div className="gate-panel">
        Checking the optional $OMEGA / TWC reader-proof demonstration…
      </div>
    );
  }

  if (state.status === "unlocked" && html) {
    return (
      <>
        <div
          style={{
            padding: "14px 18px",
            marginBottom: "32px",
            border: "1px solid var(--color-unlock-soft)",
            borderRadius: "10px",
            background: "rgba(52,211,153,0.05)",
            fontSize: "13px",
            color: "var(--color-unlock)",
            fontFamily: "ui-monospace, monospace",
          }}
        >
          OPTIONAL READER PROOF VERIFIED · {state.currency} · {state.network} · signer {state.address} · {state.unlocked}/16 chapters
        </div>
        <div
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            lineHeight: 1.75,
            fontSize: "clamp(16px, 1.5vw, 18px)",
            color: "var(--color-foreground)",
          }}
          dangerouslySetInnerHTML={{ __html: html }}
        />
      </>
    );
  }

  if (state.status === "denied") {
    return (
      <div className="gate-panel" style={{ borderColor: "var(--color-care)" }}>
        <div style={{ fontWeight: 600, marginBottom: "8px", color: "var(--color-care)" }}>
          This optional proof did not open the chapter reader
        </div>
        <p style={{ margin: "0 0 12px", fontSize: "14px" }}>
          {state.error} Your address ({state.address}) has {state.unlocked} of 16
          chapters available through this optional reader proof. This does not
          affect the public CC BY 4.0 manuscript, which needs no token or wallet.
        </p>
        <Link href="/book/full-book.md" className="link-soft" style={{ color: "var(--color-accent)", fontSize: "14px" }}>
          Read or download the complete public manuscript
        </Link>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="gate-panel" style={{ borderColor: "var(--color-care)" }}>
        <div style={{ fontWeight: 600, marginBottom: "8px", color: "var(--color-care)" }}>
          Optional reader-proof verification could not complete
        </div>
        <p style={{ margin: "0 0 12px", fontSize: "14px" }}>
          {state.error} This affects only the optional proof-reader route; the complete
          CC BY 4.0 manuscript remains publicly available without proof.
        </p>
        <Link href="/book/full-book.md" className="link-soft" style={{ color: "var(--color-accent)", fontSize: "14px" }}>
          Read or download the complete public manuscript
        </Link>
      </div>
    );
  }

  return (
    <div className="gate-panel">
      <div style={{ fontWeight: 600, marginBottom: "8px", color: "var(--color-foreground)" }}>
        Optional proof-reader route
      </div>
      <p style={{ margin: "0 0 12px", fontSize: "14px", color: "var(--color-muted-strong)" }}>
        This chapter-view route can demonstrate signed $OMEGA or TWC proofs and on-chain
        verification. It is optional: the complete text is public under CC BY 4.0 and
        reading or reusing it does not require a token or wallet.
      </p>
      <Link href="/book/full-book.md" className="link-soft" style={{ color: "var(--color-accent)", fontSize: "14px" }}>
        Read or download the complete public manuscript
      </Link>
    </div>
  );
}
