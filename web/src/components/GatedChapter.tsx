"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const STORAGE_KEY = "amity.unlock.proof";

type State =
  | { status: "checking" }
  | { status: "locked" }
  | { status: "unlocked"; address: string; tier: string; unlocked: number }
  | { status: "denied"; address: string; unlocked: number; error: string };

function readStagedProof(): {
  message: string;
  signature: string;
} | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && parsed.message && parsed.signature) return parsed;
    return null;
  } catch {
    return null;
  }
}

export function GatedChapter({ slug, number }: { slug: string; number: number }) {
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
            tier: data.tier,
            unlocked: data.unlocked,
          });
        } else {
          setState({
            status: "denied",
            address: data.address,
            unlocked: data.unlocked,
            error: data.error,
          });
        }
      })
      .catch(() => setState({ status: "locked" }));
  }, [slug]);

  if (state.status === "checking") {
    return (
      <div className="gate-panel">
        Verifying wallet signature against the ledger…
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
          UNLOCKED · signer {state.address} · {state.tier} · {state.unlocked}/16 chapters
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
          Verified signer, chapter above tier
        </div>
        <p style={{ margin: "0 0 12px", fontSize: "14px" }}>
          {state.error} Your address ({state.address}) has {state.unlocked} of 16
          chapters open. Raise your participation tier in the Amity wallet and sign
          again to read further.
        </p>
        <Link href="/novel" className="link-soft" style={{ color: "var(--color-accent)", fontSize: "14px" }}>
          Back to chapter list
        </Link>
      </div>
    );
  }

  return (
    <div className="gate-panel">
      <div style={{ fontWeight: 600, marginBottom: "8px", color: "var(--color-foreground)" }}>
        This chapter is locked
      </div>
      <p style={{ margin: "0 0 12px", fontSize: "14px", color: "var(--color-muted-strong)" }}>
        Chapter {number} of <em>Genesis Block: The Satoshi Protocol</em> is released to
        verified participants of the Omega tri-token economy. Open the Amity wallet,
        sign the release-day unlock statement, and return here — the chapter prose is
        served only after your signature verifies on the server.
      </p>
      <Link href="/novel" className="link-soft" style={{ color: "var(--color-accent)", fontSize: "14px" }}>
        Back to chapter list
      </Link>
    </div>
  );
}
