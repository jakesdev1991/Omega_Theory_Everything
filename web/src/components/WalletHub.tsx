// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { CallToAction } from "./CallToAction";

interface WalletManifestFile {
  path: string;
  bytes: number;
  sha256: string;
  generated: boolean;
}

interface WalletManifest {
  ok: boolean;
  name?: string;
  version?: string;
  channel?: string;
  notice?: string;
  mountPath?: string;
  synced?: boolean;
  generatedAt?: string | null;
  launchUrl?: string;
  files?: WalletManifestFile[];
  webApp?: { installable: boolean; manifestUrl: string; launchUrl: string };
  bundle?: { fileName: string; bytes: number; sha256: string; downloadUrl: string };
  error?: string;
}

const FEATURES = [
  {
    icon: "🔑",
    title: "Real keys, real cryptography",
    body: "BIP-39 mnemonic generation and import, BIP-44 derivation (m/44'/60'/0'/0/0), and a scrypt-encrypted V3 keystore stored only in your browser.",
  },
  {
    icon: "🦊",
    title: "Browser wallet bridges",
    body: "MetaMask for the $OMEGA EVM rail and Phantom for the TWC Solana rail. The GUI signs statements; it never sends transactions.",
  },
  {
    icon: "✍️",
    title: "Auditable unlock proofs",
    body: "Release-day statements are signed locally and verified server-side by this website before any gated chapter is served.",
  },
  {
    icon: "📡",
    title: "Live balance reads",
    body: "On-chain ETH balance reads through public RPCs with automatic failover across five endpoints.",
  },
  {
    icon: "🖥️",
    title: "Installable and offline",
    body: "The GUI is a PWA: install it from your browser, or download the offline bundle and run it on your own machine with no server at all.",
  },
  {
    icon: "🔒",
    title: "Local-first privacy",
    body: "Keystores, proofs, and settings live in local storage. Nothing is uploaded, and the prototype rails are valueless by design.",
  },
];

export function WalletHub() {
  const [manifest, setManifest] = useState<WalletManifest | null>(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/wallet/manifest")
      .then((response) => response.json())
      .then((data: WalletManifest) => {
        if (!cancelled) setManifest(data);
      })
      .catch(() => {
        if (!cancelled) setManifest({ ok: false, error: "Wallet manifest unavailable." });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  async function copy(value: string, label: string) {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(label);
      window.setTimeout(() => setCopied(null), 1600);
    } catch {
      setCopied(null);
    }
  }

  const launchUrl = manifest?.launchUrl ?? "/omega-wallet/index.html";

  return (
    <div className="wallet-hub">
      <section className="wallet-hero">
        <p className="eyebrow">Wallet · GUI on this site</p>
        <h1>
          The wallet has a graphical interface.
          <br />
          <span className="accent">It runs right here.</span>
        </h1>
        <p className="lede">
          The Omega release wallet is a real client-side wallet GUI: BIP-39 keys, an encrypted
          keystore, MetaMask and Phantom bridges, and signed release-day proofs for the two wired
          rails — $OMEGA on EVM and TWC on Solana. Launch it in your browser, install it as an app,
          or take the offline bundle with you.
        </p>
        <div className="cta-row">
          <CallToAction label="Launch the wallet GUI ↗" href={launchUrl} />
          <CallToAction label="Go to downloads" href="/wallet/download" variant="ghost" />
        </div>
        <div className="meta-row">
          <span>
            version <b>{manifest?.version ?? "…"}</b>
          </span>
          <span>
            channel <b>{manifest?.channel ?? "…"}</b>
          </span>
          <span>
            mount <b>{manifest?.mountPath ?? "/omega-wallet"}</b>
          </span>
          <span>
            synced <b>{manifest?.synced ? "yes" : "no"}</b>
          </span>
          {manifest?.generatedAt ? <span>built {new Date(manifest.generatedAt).toLocaleString()}</span> : null}
        </div>
        {!manifest?.ok ? (
          <p className="warning">
            {manifest?.error ?? "Wallet manifest not loaded."} Run <code>npm run sync:wallet</code> in
            <code>web/</code> to publish the GUI into this site.
          </p>
        ) : null}
      </section>

      <section className="preview-block">
        <div className="preview-head">
          <div>
            <h2>Live preview</h2>
            <p>The same GUI that the launch button opens, embedded below for inspection.</p>
          </div>
          <button type="button" className="toggle-btn" onClick={() => setPreviewOpen((open) => !open)}>
            {previewOpen ? "Hide preview" : "Show preview"}
          </button>
        </div>
        {previewOpen ? (
          <iframe
            title="Omega Wallet GUI preview"
            src={launchUrl}
            className="preview-frame"
            sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
          />
        ) : (
          <div className="preview-placeholder">
            <p>
              The preview loads the wallet GUI (~500&nbsp;KB of vendored ethers) inside a sandboxed
              frame. Keys created in the preview stay in this browser profile, exactly as they would
              in the standalone app.
            </p>
            <button type="button" className="toggle-btn" onClick={() => setPreviewOpen(true)}>
              Load preview
            </button>
          </div>
        )}
      </section>

      <section className="feature-grid">
        {FEATURES.map((feature) => (
          <article key={feature.title} className="feature-card">
            <span className="feature-icon">{feature.icon}</span>
            <h3>{feature.title}</h3>
            <p>{feature.body}</p>
          </article>
        ))}
      </section>

      <section className="hash-block">
        <div className="preview-head">
          <div>
            <h2>What you are running, byte for byte</h2>
            <p>
              SHA-256 of every file served at <code>{manifest?.mountPath ?? "/omega-wallet"}</code>. The
              offline bundle publishes the same hashes inside <code>CHECKSUMS.txt</code>.
            </p>
          </div>
          <Link href="/wallet/download" className="toggle-btn">
            Downloads &amp; verification
          </Link>
        </div>
        <div className="hash-table-wrapper">
          <table className="hash-table">
            <thead>
              <tr>
                <th>file</th>
                <th>bytes</th>
                <th>sha256</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {(manifest?.files ?? []).map((file) => (
                <tr key={file.path}>
                  <td>
                    <code>{file.path}</code>
                    {file.generated ? <span className="gen-tag">generated</span> : null}
                  </td>
                  <td>{file.bytes.toLocaleString()}</td>
                  <td>
                    <code title={file.sha256}>{file.sha256.slice(0, 16)}…</code>
                  </td>
                  <td>
                    <button type="button" className="copy-btn" onClick={() => copy(file.sha256, file.path)}>
                      {copied === file.path ? "copied" : "copy"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <style>{`
        .wallet-hub { max-width: 1100px; margin: 0 auto; padding: 140px 24px 96px; }
        .wallet-hero .eyebrow { color: var(--color-accent); font: 600 12px ui-monospace, monospace; letter-spacing: .2em; text-transform: uppercase; margin: 0 0 22px; }
        .wallet-hero h1 { font-family: ui-serif, Georgia, serif; font-size: clamp(38px, 6vw, 72px); line-height: 1.05; font-weight: 500; letter-spacing: -0.025em; margin: 0 0 22px; max-width: 900px; }
        .wallet-hero .accent { color: var(--color-accent); }
        .wallet-hero .lede { color: var(--color-muted-strong); font-size: clamp(16px, 1.5vw, 19px); line-height: 1.7; max-width: 720px; margin: 0 0 34px; }
        .cta-row { display: flex; gap: 14px; flex-wrap: wrap; }
        .meta-row { display: flex; gap: 22px; flex-wrap: wrap; margin-top: 34px; padding-top: 22px; border-top: 1px solid var(--color-border); color: var(--color-muted); font: 12px ui-monospace, monospace; }
        .meta-row b { color: var(--color-muted-strong); font-weight: 600; }
        .warning { margin-top: 18px; color: var(--color-care); font-size: 14px; }
        .warning code { background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 4px; }

        .preview-block, .hash-block { margin-top: 56px; }
        .preview-head { display: flex; justify-content: space-between; gap: 20px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 18px; }
        .preview-head h2 { font-family: ui-serif, Georgia, serif; font-size: clamp(22px, 3vw, 32px); font-weight: 500; margin: 0 0 6px; }
        .preview-head p { color: var(--color-muted); font-size: 14px; margin: 0; max-width: 560px; }
        .toggle-btn { border: 1px solid var(--color-border-strong); background: transparent; color: var(--color-foreground); border-radius: 8px; padding: 9px 16px; font: 600 12px ui-monospace, monospace; cursor: pointer; text-decoration: none; }
        .toggle-btn:hover { border-color: var(--color-accent); color: var(--color-accent-soft); }
        .preview-frame { width: 100%; height: 720px; border: 1px solid var(--color-border-strong); border-radius: 14px; background: #0b0d13; }
        .preview-placeholder { border: 1px dashed var(--color-border-strong); border-radius: 14px; padding: 28px; display: flex; flex-direction: column; gap: 14px; align-items: flex-start; }
        .preview-placeholder p { color: var(--color-muted-strong); font-size: 14px; line-height: 1.6; margin: 0; }

        .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 56px; }
        .feature-card { border: 1px solid var(--color-border); border-radius: 14px; padding: 22px; background: rgba(255,255,255,0.015); }
        .feature-card .feature-icon { font-size: 20px; }
        .feature-card h3 { font-size: 16px; font-weight: 600; margin: 12px 0 8px; }
        .feature-card p { color: var(--color-muted-strong); font-size: 13.5px; line-height: 1.6; margin: 0; }

        .hash-table-wrapper { overflow-x: auto; border: 1px solid var(--color-border); border-radius: 12px; }
        .hash-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
        .hash-table th { text-align: left; padding: 10px 12px; color: var(--color-muted); font: 600 11px ui-monospace, monospace; text-transform: uppercase; letter-spacing: .08em; border-bottom: 1px solid var(--color-border); }
        .hash-table td { padding: 9px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); color: var(--color-muted-strong); }
        .hash-table tr:last-child td { border-bottom: none; }
        .gen-tag { margin-left: 8px; padding: 1px 6px; border-radius: 4px; background: rgba(96,165,250,0.14); color: var(--color-use); font: 600 10px ui-monospace, monospace; }
        .copy-btn { border: 1px solid var(--color-border-strong); background: transparent; color: var(--color-muted-strong); border-radius: 6px; padding: 3px 8px; font: 600 10px ui-monospace, monospace; cursor: pointer; }
        .copy-btn:hover { color: var(--color-accent-soft); border-color: var(--color-accent); }

        @media (max-width: 720px) { .preview-frame { height: 560px; } }
      `}</style>
    </div>
  );
}
