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
          error: error instanceof Error ? error.message : "Unable to contact the unlock verifier.",
        }),
      );
  }, [slug]);

  if (state.status === "checking") {
    return (
      <div className="gate-panel">
        Verifying $OMEGA / TWC wallet proof and on-chain eligibility…
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
          UNLOCKED · {state.currency} · {state.network} · signer {state.address} · {state.unlocked}/16 chapters
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
          Verified signer, proof did not unlock this chapter
        </div>
        <p style={{ margin: "0 0 12px", fontSize: "14px" }}>
          {state.error} Your address ({state.address}) has {state.unlocked} of 16
          chapters open for this proof. Refresh your signed $OMEGA or TWC proof and
          try again.
        </p>
        <Link href="/novel" className="link-soft" style={{ color: "var(--color-accent)", fontSize: "14px" }}>
          Back to chapter list
        </Link>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="gate-panel" style={{ borderColor: "var(--color-care)" }}>
        <div style={{ fontWeight: 600, marginBottom: "8px", color: "var(--color-care)" }}>
          Verification could not complete
        </div>
        <p style={{ margin: "0 0 12px", fontSize: "14px" }}>
          {state.error}
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
        Chapter {number} of <em>Crucible: The Satoshi Protocol</em> is released to
        verified $OMEGA and TWC participants. Open the wallet, sign a release-day
        unlock statement for one of the two wired currencies, and return here — the
        chapter prose is served only after your signature verifies and the server
        independently confirms the configured on-chain rail.
      </p>
      <Link href="/novel" className="link-soft" style={{ color: "var(--color-accent)", fontSize: "14px" }}>
        Back to chapter list
      </Link>
    </div>
  );
}
