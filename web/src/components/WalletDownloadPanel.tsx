// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
"use client";

import { useEffect, useState } from "react";

interface DesktopPlatform {
  os: string;
  arch: string;
  status: "published" | "not_published";
  artifact?: string;
  url?: string;
  sha256?: string;
}

interface WalletManifest {
  ok: boolean;
  name?: string;
  version?: string;
  channel?: string;
  license?: string;
  notice?: string;
  launchUrl?: string;
  synced?: boolean;
  webApp?: { installable: boolean; manifestUrl: string; launchUrl: string };
  bundle?: { fileName: string; bytes: number; sha256: string; downloadUrl: string; contents: string[] };
  desktop?: {
    status: "published" | "not_published";
    platforms: DesktopPlatform[];
    buildWorkflow: string;
    buildInstructions: string[];
  };
  error?: string;
}

function humanBytes(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

export function WalletDownloadPanel() {
  const [manifest, setManifest] = useState<WalletManifest | null>(null);
  const [copied, setCopied] = useState(false);

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

  async function copySha() {
    if (!manifest?.bundle?.sha256) return;
    try {
      await navigator.clipboard.writeText(manifest.bundle.sha256);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  }

  const bundle = manifest?.bundle;
  const desktop = manifest?.desktop;

  return (
    <div className="download-panel">
      <section className="download-hero">
        <p className="eyebrow">Downloads · {manifest?.channel ?? "…"}</p>
        <h1>
          Take the wallet with you.
          <br />
          <span className="accent">Verify every byte.</span>
        </h1>
        <p className="lede">
          {manifest?.notice ??
            "Omega Wallet prototype. Keys never leave the device. Prototype rails only."}{" "}
          Four ways to run the same GUI: in this browser, installed as an app, offline from a
          downloadable bundle, or as a native desktop build once the release workflow publishes one.
        </p>
        <div className="version-strip">
          <span>
            {manifest?.name ?? "Omega Wallet"} <b>v{manifest?.version ?? "…"}</b>
          </span>
          <span>
            license <b>{manifest?.license ?? "…"}</b>
          </span>
        </div>
      </section>

      <section className="option-grid">
        <article className="option-card highlight">
          <div className="option-tag">01 · recommended</div>
          <h2>Offline bundle</h2>
          <p>
            The complete GUI plus local launchers for macOS, Linux, and Windows, integrity
            checksums, and the product license. Runs on <code>127.0.0.1</code> with no internet
            access required after download.
          </p>
          {bundle ? (
            <>
              <div className="bundle-meta">
                <code>{bundle.fileName}</code>
                <span>{humanBytes(bundle.bytes)}</span>
                <span>{bundle.contents.length} files</span>
              </div>
              <div className="sha-row">
                <code title={bundle.sha256}>{bundle.sha256}</code>
                <button type="button" onClick={copySha} className="copy-btn">
                  {copied ? "copied ✓" : "copy sha256"}
                </button>
              </div>
              <a className="download-btn" href={bundle.downloadUrl} download={bundle.fileName}>
                ⬇ Download {bundle.fileName}
              </a>
              <div className="verify-block">
                <p>Verify after download:</p>
                <pre>{`sha256sum ${bundle.fileName}
# or: shasum -a 256 ${bundle.fileName}
# or: certutil -hashfile ${bundle.fileName} SHA256`}</pre>
              </div>
            </>
          ) : (
            <p className="loading">Loading bundle manifest…</p>
          )}
        </article>

        <article className="option-card">
          <div className="option-tag">02</div>
          <h2>Web app, right here</h2>
          <p>
            Launch the GUI from this site. No download, no account. Keys live in this browser
            profile&apos;s local storage.
          </p>
          <a className="download-btn ghost" href={manifest?.launchUrl ?? "/omega-wallet/index.html"}>
            Launch {manifest?.launchUrl ?? "/omega-wallet/"} ↗
          </a>
          <ul className="mini-list">
            <li>Served with per-file SHA-256 published at <code>/api/wallet/manifest</code></li>
            <li>Works with MetaMask and Phantom when installed</li>
          </ul>
        </article>

        <article className="option-card">
          <div className="option-tag">03</div>
          <h2>Install as an app</h2>
          <p>
            The GUI ships a web manifest and an offline service worker, so Chrome, Edge, and
            Android can install it like a native app.
          </p>
          <ol className="mini-list ordered">
            <li>Open the web app above.</li>
            <li>Use the “Install app” button in its top bar, or your browser&apos;s install menu.</li>
            <li>Launch it from your desktop or home screen; it works offline after install.</li>
          </ol>
          <code className="manifest-path">{manifest?.webApp?.manifestUrl ?? "/omega-wallet/manifest.webmanifest"}</code>
        </article>

        <article className="option-card">
          <div className="option-tag">04</div>
          <h2>Native desktop builds</h2>
          <p>
            A Tauri wrapper around this exact GUI produces signed installers. Status per platform:
          </p>
          <ul className="platform-list">
            {(desktop?.platforms ?? []).map((platform) => (
              <li key={`${platform.os}-${platform.arch}`}>
                <span className="platform-name">
                  {platform.os} · {platform.arch}
                </span>
                {platform.status === "published" ? (
                  <a href={platform.url} className="status-pill published">
                    {platform.artifact}
                  </a>
                ) : (
                  <span className="status-pill pending">not published yet</span>
                )}
              </li>
            ))}
          </ul>
          <div className="verify-block">
            <p>Build from source:</p>
            <pre>{(desktop?.buildInstructions ?? []).join("\n")}</pre>
            <p className="fine">
              Release workflow: <code>{desktop?.buildWorkflow ?? ".github/workflows/wallet-desktop.yml"}</code>
            </p>
          </div>
        </article>
      </section>

      <section className="terms-block">
        <h2>Before you install</h2>
        <ul>
          <li>
            <b>Prototype rails only.</b> $OMEGA on Ethereum Sepolia and TWC on Solana Devnet. The
            tokens are valueless test assets; nothing here is a financial product.
          </li>
          <li>
            <b>Your keys, your risk.</b> The keystore is encrypted with your password and stored
            locally. Losing the password or the mnemonic means losing the wallet; there is no
            recovery service.
          </li>
          <li>
            <b>License.</b> The wallet GUI is product material under{" "}
            <code>{manifest?.license ?? "LicenseRef-Omega-Product-Proprietary"}</code>. The bundle
            includes <code>LICENSE.txt</code>; redistribution or commercial use requires a signed
            agreement with the rights holder.
          </li>
          <li>
            <b>Integrity.</b> Compare the published SHA-256 with the checksum in the bundle&apos;s{" "}
            <code>CHECKSUMS.txt</code> before running anything.
          </li>
        </ul>
      </section>

      <style>{`
        .download-panel { max-width: 1100px; margin: 0 auto; padding: 140px 24px 96px; }
        .download-hero .eyebrow { color: var(--color-use); font: 600 12px ui-monospace, monospace; letter-spacing: .2em; text-transform: uppercase; margin: 0 0 22px; }
        .download-hero h1 { font-family: ui-serif, Georgia, serif; font-size: clamp(38px, 6vw, 68px); line-height: 1.05; font-weight: 500; letter-spacing: -0.025em; margin: 0 0 22px; max-width: 860px; }
        .download-hero .accent { color: var(--color-accent); }
        .download-hero .lede { color: var(--color-muted-strong); font-size: clamp(16px, 1.5vw, 19px); line-height: 1.7; max-width: 720px; margin: 0 0 26px; }
        .version-strip { display: flex; gap: 24px; flex-wrap: wrap; color: var(--color-muted); font: 12px ui-monospace, monospace; padding-top: 18px; border-top: 1px solid var(--color-border); }
        .version-strip b { color: var(--color-muted-strong); }

        .option-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-top: 48px; }
        .option-card { border: 1px solid var(--color-border); border-radius: 16px; padding: 24px; background: rgba(255,255,255,0.015); display: flex; flex-direction: column; gap: 12px; }
        .option-card.highlight { border-color: rgba(192,132,87,0.5); background: rgba(192,132,87,0.05); }
        .option-tag { color: var(--color-muted); font: 600 10px ui-monospace, monospace; letter-spacing: .16em; text-transform: uppercase; }
        .option-card h2 { font-family: ui-serif, Georgia, serif; font-size: 24px; font-weight: 500; margin: 0; }
        .option-card p { color: var(--color-muted-strong); font-size: 13.5px; line-height: 1.6; margin: 0; }
        .option-card code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
        .bundle-meta { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; color: var(--color-muted); font: 12px ui-monospace, monospace; }
        .sha-row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .sha-row code { font-size: 11px; color: var(--color-unlock-soft); word-break: break-all; }
        .copy-btn { border: 1px solid var(--color-border-strong); background: transparent; color: var(--color-muted-strong); border-radius: 6px; padding: 4px 10px; font: 600 10px ui-monospace, monospace; cursor: pointer; }
        .download-btn { display: inline-flex; align-items: center; justify-content: center; gap: 8px; padding: 12px 20px; border-radius: 10px; background: linear-gradient(135deg, var(--color-accent), #a86a36); color: #07080c; font-weight: 700; font-size: 13.5px; text-decoration: none; }
        .download-btn.ghost { background: transparent; border: 1px solid var(--color-border-strong); color: var(--color-foreground); font-weight: 600; }
        .verify-block { border-top: 1px dashed var(--color-border-strong); padding-top: 10px; }
        .verify-block p { color: var(--color-muted); font: 600 10px ui-monospace, monospace; letter-spacing: .12em; text-transform: uppercase; margin: 0 0 6px; }
        .verify-block pre { margin: 0; padding: 10px 12px; background: rgba(0,0,0,0.35); border: 1px solid var(--color-border); border-radius: 8px; color: var(--color-unlock-soft); font: 12px ui-monospace, monospace; overflow-x: auto; }
        .verify-block .fine { text-transform: none; letter-spacing: 0; font-weight: 400; margin-top: 8px; }
        .mini-list { margin: 0; padding-left: 18px; color: var(--color-muted-strong); font-size: 12.5px; line-height: 1.7; }
        .mini-list.ordered { padding-left: 20px; }
        .manifest-path { display: block; font-size: 11px; color: var(--color-muted); }
        .platform-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }
        .platform-list li { display: flex; justify-content: space-between; align-items: center; gap: 10px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px; }
        .platform-name { color: var(--color-muted-strong); font: 600 12px ui-monospace, monospace; }
        .status-pill { font: 600 10px ui-monospace, monospace; padding: 3px 8px; border-radius: 999px; text-decoration: none; }
        .status-pill.published { background: rgba(52,211,153,0.14); color: var(--color-unlock); }
        .status-pill.pending { background: rgba(251,191,36,0.12); color: var(--color-omega); }
        .loading { color: var(--color-muted); font-size: 13px; }

        .terms-block { margin-top: 56px; border: 1px solid var(--color-border); border-radius: 16px; padding: 26px; background: rgba(255,255,255,0.015); }
        .terms-block h2 { font-family: ui-serif, Georgia, serif; font-size: 24px; font-weight: 500; margin: 0 0 14px; }
        .terms-block ul { margin: 0; padding-left: 20px; color: var(--color-muted-strong); font-size: 13.5px; line-height: 1.75; display: grid; gap: 8px; }
        .terms-block b { color: var(--color-foreground); }
        .terms-block code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
      `}</style>
    </div>
  );
}
