// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import type { Metadata } from "next";
import Link from "next/link";

import { loadLegalDocument } from "@/lib/legal";
import { PUBLISHER_AGREEMENT_ID, STORE_TERMS_ID } from "@/lib/store-license";

export const metadata: Metadata = {
  title: "App Store Terms & Licensing",
  description:
    "Terms of Use & End-User License for the Omega App Store, the Publisher Agreement template, and how Nostr-based app licenses work.",
};

export default function StoreTermsPage() {
  const terms = loadLegalDocument("store-terms.md");
  const publisher = loadLegalDocument("publisher-agreement.md");

  return (
    <div className="legal">
      <p className="legal-eyebrow">
        <Link href="/store">← App Store</Link> · Licensing
      </p>
      <h1>Terms &amp; licensing</h1>
      <div className="legal-summary">
        <p>
          <strong>Plain-language summary</strong> (not a substitute for the full text):
        </p>
        <ul>
          <li>Apps and store software are proprietary. A license lets you <em>run</em> an app through the store; it does not give you the code.</li>
          <li>Your license is a signed Nostr event (kind 31335) bound to your npub. Whoever holds your nsec can use it, so keep it safe.</li>
          <li>Licenses and job parameters are <strong>public</strong> on relays. Never submit personal or confidential data.</li>
          <li>Results come as-is from a mobile node that may be offline. Check anything before relying on it.</li>
          <li>Licenses can be revoked for breach or security reasons; revocation is a newer signed record.</li>
        </ul>
        <p className="legal-ids">
          Current terms: <code>{STORE_TERMS_ID}</code> · Publisher template: <code>{PUBLISHER_AGREEMENT_ID}</code> · Raw text:{" "}
          <a href="/legal/store-terms.md">store-terms.md</a>, <a href="/legal/publisher-agreement.md">publisher-agreement.md</a>
        </p>
      </div>

      <article className="legal-doc" dangerouslySetInnerHTML={{ __html: terms.html }} />

      <hr id="publisher" />

      <article className="legal-doc" dangerouslySetInnerHTML={{ __html: publisher.html }} />

      <style>{`
        .legal { max-width: 860px; margin: 0 auto; padding: 120px 24px 96px; }
        .legal-eyebrow { color: var(--color-amity); font: 600 12px ui-monospace, monospace; letter-spacing: .2em; text-transform: uppercase; margin: 0 0 20px; }
        .legal-eyebrow a { color: inherit; }
        .legal h1 { font-family: ui-serif, Georgia, serif; font-size: clamp(34px, 5vw, 54px); font-weight: 500; letter-spacing: -0.02em; margin: 0 0 24px; }
        .legal-summary { border: 1px solid var(--color-border); border-radius: 14px; padding: 18px 22px; background: rgba(167,139,250,0.05); color: var(--color-muted-strong); font-size: 14px; line-height: 1.7; margin-bottom: 40px; }
        .legal-summary p { margin: 0 0 8px; }
        .legal-summary ul { margin: 0 0 10px; padding-left: 20px; }
        .legal-ids { font-size: 12.5px; color: var(--color-muted); }
        .legal code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
        .legal a { color: var(--color-amity); }
        .legal hr { border: 0; border-top: 1px solid var(--color-border); margin: 56px 0; }
        .legal-doc { color: var(--color-muted-strong); font-size: 14.5px; line-height: 1.75; }
        .legal-doc h1 { font-size: clamp(26px, 3.4vw, 36px); margin: 0 0 12px; }
        .legal-doc h2 { font-family: ui-serif, Georgia, serif; font-weight: 500; font-size: 22px; color: var(--color-foreground); margin: 34px 0 10px; }
        .legal-doc h3 { font-size: 16px; color: var(--color-foreground); margin: 24px 0 8px; }
        .legal-doc blockquote { margin: 16px 0; padding: 12px 16px; border-left: 3px solid var(--color-omega); background: rgba(251,191,36,0.06); border-radius: 0 8px 8px 0; }
        .legal-doc blockquote p { margin: 0; }
        .legal-doc ol, .legal-doc ul { padding-left: 22px; }
        .legal-doc li { margin: 4px 0; }
        .legal-doc table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 13px; }
        .legal-doc th, .legal-doc td { border: 1px solid var(--color-border); padding: 8px 10px; text-align: left; vertical-align: top; }
        .legal-doc th { color: var(--color-foreground); background: rgba(255,255,255,0.03); }
      `}</style>
    </div>
  );
}
