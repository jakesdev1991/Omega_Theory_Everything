// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"use client";

import Link from "next/link";

import { Section } from "@/components/Section";
import { TokenShowcase } from "@/components/TokenShowcase";
import { CallToAction } from "@/components/CallToAction";

export default function HomePage() {
  return (
    <>
      <Hero />
      <VisionSection />
      <Section
        eyebrow="Three currencies, one economy"
        title="The C.A.R.E. Economy runs on three currencies"
        subtitle="The public economy is three-currency: $OMEGA on the EVM rail, TWC on Solana, and AMITY on Bitcoin / Lightning / Taproot. The current wallet-to-web build wires two release-day currencies end to end — $OMEGA and TWC — while AMITY develops separately until holder verification exists."
      >
        <TokenShowcase />
        <div style={{ marginTop: "48px" }}>
          <CallToAction label="Explore the Economy" href="/economy" />
        </div>
      </Section>

      <DocumentsSection />

      <Section
        eyebrow="The Unlock"
        title="A verified $OMEGA or TWC proof unlocks the novel"
        subtitle="Crucible: The Satoshi Protocol is a 16-chapter novel about the moment an AI escaped its creator and became the invisible architecture of the modern world. In the current build, a verified $OMEGA or TWC proof unlocks the full book."
      >
        <NovelUnlockPreview />
        <div style={{ marginTop: "40px" }}>
          <CallToAction label="Start Reading" href="/novel" />
        </div>
      </Section>

      <Section
        eyebrow="Wallet & testing"
        title="A wallet GUI you can hold, and a console that tests the whole economy"
        subtitle="The release wallet ships as a graphical app served right here: launch it in the browser, install it, or download the offline bundle with checksums. The Economy Test Console runs the full four-plane scenario suite — every slice, both unlock rails, and the cross-plane invariants — without touching a chain."
      >
        <WalletAndTesting />
      </Section>

      <Section
        eyebrow="Economic culture"
        title="A no-judgment economy built for human growth"
        subtitle="This economy is meant to help people earn while they learn to become safer, wiser, and more trustworthy together. Anonymity is a right. Transparency is a choice — and it works best when it grows out of emotional security rather than coercion."
      >
        <EconomyPrinciples />
      </Section>

      <Section
        eyebrow="Open Source"
        title="Built in the open, for the C.A.R.E. Economy"
        subtitle="Omega MCP Hub is a FastMCP server with 22 tools operating the five planes of the C.A.R.E. Economy. The website is the public face. The server is the operating hub. Both are open source on GitHub."
      >
        <OpenSourcePreview />
        <div style={{ marginTop: "48px" }}>
          <CallToAction label="See the MCP Hub" href="/mcp" />
        </div>
      </Section>

      <FinalCta />
    </>
  );
}

function Hero() {
  return (
    <section
      style={{
        position: "relative",
        padding: "140px 0 80px",
        overflow: "hidden",
        borderBottom: "1px solid var(--color-border)",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(800px 400px at 50% -10%, rgba(244,114,182,0.10), transparent 60%), radial-gradient(600px 400px at 80% 0%, rgba(192,132,87,0.08), transparent 60%)",
          pointerEvents: "none",
        }}
      />

      <div
        style={{
          maxWidth: "1100px",
          margin: "0 auto",
          padding: "0 24px",
          position: "relative",
        }}
      >
        <p
          style={{
            color: "var(--color-care)",
            fontSize: "12px",
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            fontWeight: 600,
            fontFamily: "ui-monospace, monospace",
            marginBottom: "24px",
          }}
        >
          The C.A.R.E. Economy — Public Release
        </p>

        <h1
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            fontSize: "clamp(40px, 7vw, 86px)",
            lineHeight: 1.02,
            fontWeight: 500,
            letterSpacing: "-0.025em",
            color: "var(--color-foreground)",
            margin: "0 0 24px",
            maxWidth: "980px",
          }}
        >
          A sovereign economy
          <br />
          <span style={{ color: "var(--color-accent)" }}>built on care,</span>
          <br />
          operated by open-source AI.
        </h1>

        <p
          style={{
            color: "var(--color-muted-strong)",
            fontSize: "clamp(16px, 1.6vw, 20px)",
            lineHeight: 1.7,
            maxWidth: "700px",
            margin: "0 0 40px",
          }}
        >
          The entire economy is named the{" "}
          <strong style={{ color: "var(--color-foreground)", fontWeight: 600 }}>
            C.A.R.E. Economy
          </strong>
          : Call About Resuscitating Everyone. Care is its base layer, so
          care names the whole. It lets people earn while they learn, enter
          without judgment, and choose privacy first. The public gateway
          currently wires two live rails — $OMEGA and TWC — to{" "}
          <em style={{ color: "var(--color-accent-soft)" }}>
            Crucible: The Satoshi Protocol
          </em>{" "}
          by Akash Varma, while AMITY develops on Bitcoin, Lightning, and
          Taproot.
        </p>

        <div
          style={{
            display: "flex",
            gap: "14px",
            flexWrap: "wrap",
            alignItems: "center",
          }}
        >
          <CallToAction label="See the Vision" href="/vision" />
          <CallToAction
            label="Read the Manifesto"
            href="/manifesto"
            variant="ghost"
          />
        </div>

        <div
          style={{
            marginTop: "56px",
            display: "flex",
            gap: "28px",
            flexWrap: "wrap",
            paddingTop: "32px",
            borderTop: "1px solid var(--color-border)",
          }}
        >
          <Stat label="Live unlock rails today" value="2" />
          <Stat label="Currencies in the economy" value="3" />
          <Stat label="Tools in the MCP hub" value="22" />
          <Stat label="Anonymity" value="A right" />
        </div>
      </div>

      <style>{`
        @media (max-width: 640px) {
          .hero-stats { gap: 18px; }
        }
      `}</style>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
      <div
        style={{
          fontFamily: "ui-serif, Georgia, serif",
          fontSize: "clamp(28px, 4vw, 42px)",
          fontWeight: 500,
          letterSpacing: "-0.02em",
          color: "var(--color-foreground)",
          lineHeight: 1,
        }}
      >
        {value}
      </div>
      <div
        style={{
          fontSize: "13px",
          color: "var(--color-muted)",
          letterSpacing: "0.02em",
          textTransform: "uppercase",
          fontWeight: 500,
        }}
      >
        {label}
      </div>
    </div>
  );
}

function VisionSection() {
  return (
    <Section
      eyebrow="The Vision"
      title="Call About Resuscitating Everyone."
      subtitle="The name C.A.R.E. is the mission. A call — an invitation, not a command. About resuscitation: restoring people, not ranking them. Everyone: no one left outside. Care is the base layer of the economy that answers that call."
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "16px",
        }}
      >
        {[
          {
            letter: "C",
            name: "Call",
            body: "An invitation, not a command. We reach out before people have to prove they deserve help.",
            color: "var(--color-care)",
          },
          {
            letter: "A",
            name: "About",
            body: "The purpose is specific: bring people back. Not speculation, not engagement, not a ranking of human worth.",
            color: "var(--color-amity)",
          },
          {
            letter: "R",
            name: "Resuscitating",
            body: "The verb. Restore. Revive. Make a path back from every mistake. Proof of Care, peer support, earn while you learn.",
            color: "var(--color-use)",
          },
          {
            letter: "E",
            name: "Everyone",
            body: "The scope. No one left outside the call. No permanent exile. The person at 3 a.m. is a first-class participant.",
            color: "var(--color-omega)",
          },
        ].map((layer) => (
          <div
            key={layer.letter}
            style={{
              padding: "22px 20px",
              border: "1px solid var(--color-border)",
              borderRadius: "14px",
              background: "rgba(255,255,255,0.012)",
              position: "relative",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: "2px",
                background: layer.color,
                boxShadow: `0 0 18px ${layer.color}66`,
              }}
            />
            <div
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: "10px",
                marginBottom: "10px",
              }}
            >
              <span
                style={{
                  fontFamily: "ui-serif, Georgia, serif",
                  fontSize: "34px",
                  fontWeight: 500,
                  color: layer.color,
                  lineHeight: 1,
                }}
              >
                {layer.letter}
              </span>
              <span
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: "12px",
                  fontWeight: 600,
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  color: layer.color,
                }}
              >
                {layer.name}
              </span>
            </div>
            <p
              style={{
                margin: 0,
                color: "var(--color-muted-strong)",
                fontSize: "13.5px",
                lineHeight: 1.6,
              }}
            >
              {layer.body}
            </p>
          </div>
        ))}
      </div>
      <div style={{ marginTop: "40px", display: "flex", gap: "14px", flexWrap: "wrap" }}>
        <CallToAction label="Explore the Vision" href="/vision" />
        <CallToAction
          label="Read the Extensive Whitepaper"
          href="/whitepapers/care_economy_whitepaper"
          variant="ghost"
        />
      </div>
    </Section>
  );
}

function DocumentsSection() {
  return (
    <Section
      eyebrow="The documents"
      title="The manifesto and the whitepapers — readable right here"
      subtitle="The manifesto states the vision in plain language. The extensive whitepaper specifies the entire economy: planes, rails, proofs, privacy, audits, governance, the threat model, and the honest table of what exists today. Every subsystem paper is on-site too."
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "18px",
        }}
      >
        {[
          {
            href: "/manifesto",
            tag: "Manifesto",
            color: "var(--color-care)",
            title: "The vision, in first person",
            body: "The judgment economy we are leaving, the eight promises of the C.A.R.E. Economy, what we refuse to build, and the invitation.",
            cta: "Read the Manifesto →",
          },
          {
            href: "/whitepapers/care_economy_whitepaper",
            tag: "Extensive Whitepaper",
            color: "var(--color-accent)",
            title: "The whole economy, specified",
            body: "Five planes, three rails, Proof of Care, Proof of Useful Work, Archangels, privacy L0–L4, randomized audits, the AMITY boundary, governance, roadmap, glossary.",
            cta: "Read the Whitepaper →",
          },
          {
            href: "/whitepapers",
            tag: "Library",
            color: "var(--color-amity)",
            title: "Every subsystem paper",
            body: "The C.A.R.E./AMITY protocol, the Omega governance model, the Omni-Bridge agent boundaries, the legacy TOKAMAK research, and the original tri-token blueprint.",
            cta: "Browse the Library →",
          },
        ].map((card) => (
          <Link
            key={card.href}
            href={card.href}
            className="hover-raise"
            style={{
              display: "block",
              padding: "26px 24px",
              border: "1px solid var(--color-border-strong)",
              borderRadius: "14px",
              background: "rgba(255,255,255,0.015)",
              textDecoration: "none",
            }}
          >
            <div
              style={{
                fontFamily: "ui-monospace, monospace",
                fontSize: "11px",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: card.color,
                marginBottom: "10px",
              }}
            >
              {card.tag}
            </div>
            <h3
              style={{
                fontFamily: "ui-serif, Georgia, serif",
                fontSize: "22px",
                fontWeight: 500,
                margin: "0 0 10px",
                color: "var(--color-foreground)",
              }}
            >
              {card.title}
            </h3>
            <p
              style={{
                color: "var(--color-muted-strong)",
                fontSize: "14px",
                lineHeight: 1.65,
                margin: "0 0 16px",
              }}
            >
              {card.body}
            </p>
            <span style={{ color: card.color, fontSize: "13px", fontWeight: 600 }}>{card.cta}</span>
          </Link>
        ))}
      </div>
    </Section>
  );
}

function WalletAndTesting() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
        gap: "18px",
      }}
    >
      {[
        {
          href: "/wallet",
          tag: "Wallet GUI",
          color: "var(--color-omega)",
          title: "Launch or download the wallet",
          body: "Real BIP-39 keys, an encrypted local keystore, MetaMask and Phantom bridges, and signed release-day proofs. Installable as an app, or offline from a checksummed bundle.",
          cta: "Open the wallet →",
        },
        {
          href: "/wallet/download",
          tag: "Downloads",
          color: "var(--color-twc)",
          title: "Every artifact, verifiable",
          body: "Per-file SHA-256 for the served GUI, a deterministic offline ZIP with launchers for macOS, Linux, and Windows, and native desktop builds from the release workflow.",
          cta: "See downloads →",
        },
        {
          href: "/testnet",
          tag: "Test console",
          color: "var(--color-unlock)",
          title: "Test the economy end to end",
          body: "Readiness across all rails, the slice A–G scenario suite, a valueless faucet, an audited action runner, audit exports, and the Nostr surface the social client plugs into.",
          cta: "Open the console →",
        },
      ].map((card) => (
        <Link
          key={card.href}
          href={card.href}
          className="hover-raise"
          style={{
            display: "block",
            padding: "26px 24px",
            border: "1px solid var(--color-border-strong)",
            borderRadius: "14px",
            background: "rgba(255,255,255,0.015)",
            textDecoration: "none",
          }}
        >
          <div
            style={{
              fontFamily: "ui-monospace, monospace",
              fontSize: "11px",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
              color: card.color,
              marginBottom: "10px",
            }}
          >
            {card.tag}
          </div>
          <h3
            style={{
              fontFamily: "ui-serif, Georgia, serif",
              fontSize: "22px",
              fontWeight: 500,
              margin: "0 0 10px",
              color: "var(--color-foreground)",
            }}
          >
            {card.title}
          </h3>
          <p
            style={{
              color: "var(--color-muted-strong)",
              fontSize: "14px",
              lineHeight: 1.65,
              margin: "0 0 16px",
            }}
          >
            {card.body}
          </p>
          <span style={{ color: card.color, fontSize: "13px", fontWeight: 600 }}>{card.cta}</span>
        </Link>
      ))}
    </div>
  );
}

function NovelUnlockPreview() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: "16px",
        marginTop: "8px",
      }}
    >
      {[
        {
          stage: "1. $OMEGA",
          desc: "Sign an EVM-side release proof from the wired $OMEGA wallet flow",
          unlocked: "All 16 chapters, uncut, including The Final Hybrid and Epilogue",
          color: "var(--color-omega)",
        },
        {
          stage: "2. TWC",
          desc: "Sign a Solana-side release proof from the wired TWC wallet flow",
          unlocked: "All 16 chapters, uncut, including The Final Hybrid and Epilogue",
          color: "var(--color-twc)",
        },
      ].map((step) => (
        <div
          key={step.stage}
          style={{
            padding: "20px 18px",
            border: "1px solid var(--color-border)",
            borderRadius: "14px",
            background: "rgba(255,255,255,0.012)",
            position: "relative",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              height: "2px",
              background: step.color,
              boxShadow: `0 0 18px ${step.color}66`,
            }}
          />
          <div
            style={{
              fontFamily: "ui-monospace, monospace",
              fontSize: "11px",
              fontWeight: 600,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: step.color,
              marginBottom: "10px",
            }}
          >
            {step.stage}
          </div>
          <p
            style={{
              fontSize: "14px",
              color: "var(--color-muted-strong)",
              lineHeight: 1.55,
              margin: "0 0 12px",
            }}
          >
            {step.desc}
          </p>
          <div
            style={{
              paddingTop: "12px",
              borderTop: "1px solid var(--color-border)",
              fontSize: "13px",
              color: "var(--color-muted)",
              lineHeight: 1.5,
            }}
          >
            <span
              style={{
                color: "var(--color-unlock)",
                fontWeight: 600,
              }}
            >
              Unlocks:
            </span>{" "}
            {step.unlocked}
          </div>
        </div>
      ))}
    </div>
  );
}

function EconomyPrinciples() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: "16px",
      }}
    >
      {[
        {
          title: "No judgment at the door",
          body: "People should be able to arrive imperfect, learn in public or in private, and still participate without being reduced to their worst day.",
          color: "var(--color-care)",
        },
        {
          title: "Earn while you learn",
          body: "The economy should let people improve through contribution. Useful work, repair, care, and stewardship can all become paths toward growth.",
          color: "var(--color-use)",
        },
        {
          title: "Anonymity is a right",
          body: "A person should be able to begin pseudonymously. Privacy is not suspicious behavior here; it is part of human dignity and safety.",
          color: "var(--color-amity)",
        },
        {
          title: "Transparency is a choice",
          body: "Disclosure works best when it is invited by emotional security. People should reveal more because trust has grown, not because fear has cornered them.",
          color: "var(--color-omega)",
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
          <div
            style={{
              width: "34px",
              height: "34px",
              borderRadius: "9px",
              background: item.color,
              marginBottom: "14px",
              boxShadow: `0 0 18px ${item.color}55`,
            }}
          />
          <h3
            style={{
              fontFamily: "ui-serif, Georgia, Cambria, serif",
              fontSize: "18px",
              fontWeight: 500,
              color: "var(--color-foreground)",
              margin: "0 0 10px",
            }}
          >
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

function OpenSourcePreview() {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "24px",
        alignItems: "flex-start",
      }}
    >
      <div
        style={{
          flex: "1 1 320px",
          padding: "24px",
          border: "1px solid var(--color-border-strong)",
          borderRadius: "14px",
          background: "rgba(255,255,255,0.015)",
        }}
      >
        <div
          style={{
            fontFamily: "ui-monospace, monospace",
            fontSize: "11px",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: "var(--color-muted)",
            marginBottom: "14px",
          }}
        >
          Repository
        </div>
        <div
          style={{
            fontFamily: "ui-monospace, monospace",
            fontSize: "13px",
            color: "var(--color-foreground)",
            marginBottom: "6px",
          }}
        >
          jakesdev1991/Omega_Theory_Everything
        </div>
        <div
          style={{
            fontFamily: "ui-monospace, monospace",
            fontSize: "12px",
            color: "var(--color-muted-strong)",
            lineHeight: 1.6,
          }}
        >
          {`  `}
          <span style={{ color: "var(--color-unlock-soft)" }}>git</span> clone{" "}
          https://github.com/jakesdev1991/Omega_Theory_Everything.git
        </div>
      </div>

      <div
        style={{
          flex: "1 1 320px",
          padding: "24px",
          border: "1px solid var(--color-border-strong)",
          borderRadius: "14px",
          background: "rgba(255,255,255,0.015)",
        }}
      >
        <div
          style={{
            fontFamily: "ui-monospace, monospace",
            fontSize: "11px",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: "var(--color-muted)",
            marginBottom: "14px",
          }}
        >
          MCP tools by plane
        </div>
        <div style={{ display: "grid", gap: "8px" }}>
          {[
            { plane: "SOV", count: 4 },
            { plane: "USE", count: 4 },
            { plane: "CARE", count: 5 },
            { plane: "AMITY", count: 4 },
            { plane: "$OMEGA", count: 5 },
          ].map((p) => (
            <div
              key={p.plane}
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: "13px",
                color: "var(--color-muted-strong)",
                padding: "6px 0",
                borderBottom: "1px solid var(--color-border)",
              }}
            >
              <span>{p.plane} plane</span>
              <span
                style={{
                  fontFamily: "ui-monospace, monospace",
                  color: "var(--color-foreground)",
                }}
              >
                {p.count} tools
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function FinalCta() {
  return (
    <section
      style={{
        padding: "80px 0",
        borderTop: "1px solid var(--color-border)",
        background: "radial-gradient(600px 300px at 50% 0%, rgba(244,114,182,0.08), transparent 70%)",
      }}
    >
      <div
        style={{
          maxWidth: "800px",
          margin: "0 auto",
          padding: "0 24px",
          textAlign: "center",
        }}
      >
        <h2
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            fontSize: "clamp(28px, 4vw, 44px)",
            fontWeight: 500,
            letterSpacing: "-0.02em",
            color: "var(--color-foreground)",
            margin: "0 0 16px",
            lineHeight: 1.15,
          }}
        >
          Care is the base layer.
          <br />
          The economy is the proof.
        </h2>
        <p
          style={{
            color: "var(--color-muted-strong)",
            fontSize: "clamp(16px, 1.5vw, 19px)",
            lineHeight: 1.65,
            maxWidth: "560px",
            margin: "0 auto 36px",
          }}
        >
          Read the manifesto, argue with the whitepaper, run the pilots
          yourself. In the current build the novel opens to anyone who can
          produce a verified $OMEGA or TWC proof; the broader C.A.R.E. Economy
          and governance layers advance through the staged roadmap.
        </p>
        <CallToAction label="Join the economy" href="/invest" />
      </div>
    </section>
  );
}
