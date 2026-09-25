// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"use client";

import { Section } from "@/components/Section";
import { CallToAction } from "@/components/CallToAction";
import { UnlockRailStatusPanel } from "@/components/UnlockRailStatusPanel";

export default function OpsPage() {
  return (
    <>
      <Hero />
      <Section
        eyebrow="Live status"
        title="Current operator view of both wired rails"
        subtitle="This page shows whether the local web app can independently verify $OMEGA on the EVM/Sepolia pilot rail and TWC on the Solana/Devnet pilot rail. It reads the same status endpoint the novel pages use for their setup banner."
      >
        <UnlockRailStatusPanel />
      </Section>

      <Section
        eyebrow="Runbooks"
        title="How to produce the local manifests"
        subtitle="The web unlock flow becomes locally usable after the two pilot manifests exist or equivalent environment variables are supplied. These are the operator commands for each current rail, plus the first AMITY scaffold command set."
      >
        <RunbookCards />
      </Section>

      <Section
        eyebrow="What the web app expects"
        title="Manifest paths and environment fallbacks"
        subtitle="The reader prefers ignored deployment manifests first, then falls back to explicit environment variables. That keeps the unlock flow deterministic without hardcoding production addresses."
      >
        <ExpectationGrid />
      </Section>

      <FinalCta />
    </>
  );
}

function Hero() {
  return (
    <section
      style={{
        padding: "140px 0 72px",
        borderBottom: "1px solid var(--color-border)",
        background:
          "radial-gradient(800px 400px at 75% -10%, rgba(34,211,238,0.08), transparent 60%), radial-gradient(700px 350px at 20% 0%, rgba(251,191,36,0.06), transparent 60%)",
      }}
    >
      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
        <p
          style={{
            color: "var(--color-twc)",
            fontSize: "12px",
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            fontWeight: 600,
            fontFamily: "ui-monospace, monospace",
            marginBottom: "24px",
          }}
        >
          Operator Status
        </p>
        <h1
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            fontSize: "clamp(38px, 6vw, 72px)",
            lineHeight: 1.05,
            fontWeight: 500,
            letterSpacing: "-0.025em",
            color: "var(--color-foreground)",
            margin: "0 0 22px",
            maxWidth: "900px",
          }}
        >
          Two rails.
          <br />
          <span style={{ color: "var(--color-accent)" }}>One local activation checklist.</span>
        </h1>
        <p
          style={{
            color: "var(--color-muted-strong)",
            fontSize: "clamp(16px, 1.5vw, 19px)",
            lineHeight: 1.7,
            maxWidth: "720px",
            margin: 0,
          }}
        >
          Use this page to see whether the current sandbox can verify the wired
          $OMEGA and TWC unlock rails, and to follow the exact commands needed to
          produce the local pilot manifests those checks depend on.
        </p>
      </div>
    </section>
  );
}

function CommandBlock({ lines }: { lines: string[] }) {
  return (
    <div
      style={{
        padding: "20px",
        border: "1px solid var(--color-border-strong)",
        borderRadius: "14px",
        background: "rgba(0,0,0,0.25)",
        fontFamily: "ui-monospace, monospace",
        fontSize: "13px",
        lineHeight: 1.7,
        color: "var(--color-foreground)",
        overflowX: "auto",
        whiteSpace: "pre-wrap",
      }}
    >
      {lines.join("\n")}
    </div>
  );
}

function RunbookCards() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
        gap: "18px",
      }}
    >
      <div
        style={{
          padding: "24px",
          border: "1px solid var(--color-border)",
          borderRadius: "16px",
          background: "rgba(255,255,255,0.012)",
        }}
      >
        <div style={{ fontFamily: "ui-monospace, monospace", fontSize: "12px", letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--color-omega)", marginBottom: "10px" }}>
          $OMEGA · Sepolia
        </div>
        <h3 style={{ fontFamily: "ui-serif, Georgia, Cambria, serif", fontSize: "24px", fontWeight: 500, margin: "0 0 10px", color: "var(--color-foreground)" }}>
          Preflight, then deploy the pilot suite
        </h3>
        <p style={{ color: "var(--color-muted-strong)", fontSize: "14px", lineHeight: 1.6, margin: "0 0 18px" }}>
          The new preflight command validates the Sepolia configuration without sending a transaction. After review, the deploy command writes <code>evm/deployments/sepolia.json</code>, which the web reader can consume.
        </p>
        <CommandBlock
          lines={[
            "cd evm",
            "cp .env.example .env",
            "# fill DEPLOYER_PRIVATE_KEY, SEPOLIA_RPC_URL, TREASURY_ADDRESS, GUARDIAN_ADDRESS",
            "npm ci",
            "npm run compile",
            "npm run preflight:sepolia",
            "npm run deploy:sepolia",
          ]}
        />
      </div>

      <div
        style={{
          padding: "24px",
          border: "1px solid var(--color-border)",
          borderRadius: "16px",
          background: "rgba(255,255,255,0.012)",
        }}
      >
        <div style={{ fontFamily: "ui-monospace, monospace", fontSize: "12px", letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--color-twc)", marginBottom: "10px" }}>
          TWC · Devnet
        </div>
        <h3 style={{ fontFamily: "ui-serif, Georgia, Cambria, serif", fontSize: "24px", fontWeight: 500, margin: "0 0 10px", color: "var(--color-foreground)" }}>
          Run the guarded Devnet preflight/deploy flow
        </h3>
        <p style={{ color: "var(--color-muted-strong)", fontSize: "14px", lineHeight: 1.6, margin: "0 0 18px" }}>
          The default Devnet deploy command is already preflight-only until you add the explicit confirmation flag and environment variable. A successful deployment writes <code>solana/deployments/twc-devnet.json</code>.
        </p>
        <CommandBlock
          lines={[
            "cd solana",
            "cp .env.example .env",
            "# fill SOLANA_DEPLOYER_KEYPAIR_PATH, TWC_TREASURY_ADDRESS, TWC_METADATA_URI, TWC_METADATA_SHA256",
            "npm ci",
            "npm run check",
            "npm run deploy:devnet",
            "TWC_DEPLOY_CONFIRM=DEVNET_TWC_PILOT npm run deploy:devnet -- --confirm-devnet",
          ]}
        />
      </div>

      <div
        style={{
          padding: "24px",
          border: "1px solid var(--color-border)",
          borderRadius: "16px",
          background: "rgba(255,255,255,0.012)",
        }}
      >
        <div style={{ fontFamily: "ui-monospace, monospace", fontSize: "12px", letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--color-accent)", marginBottom: "10px" }}>
          AMITY · Testnet scaffold
        </div>
        <h3 style={{ fontFamily: "ui-serif, Georgia, Cambria, serif", fontSize: "24px", fontWeight: 500, margin: "0 0 10px", color: "var(--color-foreground)" }}>
          Start the Bitcoin / Lightning / Taproot workstream safely
        </h3>
        <p style={{ color: "var(--color-muted-strong)", fontSize: "14px", lineHeight: 1.6, margin: "0 0 18px" }}>
          The AMITY directory remains a separate testnet operator scaffold for now: config parsing, holder-challenge formatting, a non-broadcasting preflight, and a local status-manifest snapshot for the web operator page. It still does not issue an asset or enable a third unlock rail.
        </p>
        <CommandBlock
          lines={[
            "cd amity",
            "cp .env.example .env",
            "# fill AMITY_NETWORK=testnet, AMITY_ASSET_ID, AMITY_UNIVERSE_URL, host:port values",
            "npm test",
            "npm run preflight:testnet",
            "npm run manifest:testnet",
          ]}
        />
      </div>
    </div>
  );
}

function ExpectationGrid() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
        gap: "16px",
      }}
    >
      {[
        {
          title: "Preferred local manifests",
          body: "The web reader checks ../evm/deployments/sepolia.json and ../solana/deployments/twc-devnet.json first. Both are intentionally gitignored.",
        },
        {
          title: "Environment fallback",
          body: "If manifests are unavailable, the reader accepts explicit web-level environment variables from web/.env.example for both rails.",
        },
        {
          title: "Fail-closed behavior",
          body: "If the rail is unconfigured, the unlock API and gated chapter route refuse access and the novel UI shows a setup warning instead of silently bypassing checks.",
        },
        {
          title: "What counts as ready",
          body: "A valid wallet signature plus an independent chain-side check: Sepolia gate/token eligibility for $OMEGA and positive Devnet SPL balance for TWC.",
        },
      ].map((item) => (
        <div
          key={item.title}
          style={{
            padding: "22px 20px",
            border: "1px solid var(--color-border)",
            borderRadius: "14px",
            background: "rgba(255,255,255,0.012)",
          }}
        >
          <h3 style={{ fontFamily: "ui-serif, Georgia, Cambria, serif", fontSize: "18px", fontWeight: 500, margin: "0 0 10px", color: "var(--color-foreground)" }}>
            {item.title}
          </h3>
          <p style={{ margin: 0, color: "var(--color-muted-strong)", fontSize: "14px", lineHeight: 1.6 }}>
            {item.body}
          </p>
        </div>
      ))}
    </div>
  );
}

function FinalCta() {
  return (
    <section
      style={{
        padding: "72px 0",
        borderTop: "1px solid var(--color-border)",
        background: "radial-gradient(600px 280px at 50% 0%, rgba(192,132,87,0.06), transparent 70%)",
      }}
    >
      <div style={{ maxWidth: "760px", margin: "0 auto", padding: "0 24px", textAlign: "center" }}>
        <h2
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            fontSize: "clamp(26px, 3.4vw, 38px)",
            fontWeight: 500,
            letterSpacing: "-0.02em",
            color: "var(--color-foreground)",
            margin: "0 0 14px",
          }}
        >
          Finish the two manifests, then the reader can go live locally.
        </h2>
        <p style={{ color: "var(--color-muted-strong)", fontSize: "17px", lineHeight: 1.6, margin: "0 0 32px" }}>
          Once both pilot manifests exist locally, the hardened unlock flow can verify wallet proofs and chain state end to end inside this sandbox.
        </p>
        <div style={{ display: "flex", gap: "14px", justifyContent: "center", flexWrap: "wrap" }}>
          <CallToAction label="Check the novel reader" href="/novel" />
          <CallToAction label="Read web setup docs" href="https://github.com/jakesdev1991/Omega_Theory_Everything/tree/main/web" variant="ghost" />
        </div>
      </div>
    </section>
  );
}
