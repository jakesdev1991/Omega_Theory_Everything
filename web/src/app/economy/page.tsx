// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"use client";

import { Section } from "@/components/Section";
import { TokenShowcase } from "@/components/TokenShowcase";
import { CallToAction } from "@/components/CallToAction";
import { MultiTokenWorkbench } from "@/components/MultiTokenWorkbench";
import Link from "next/link";

export default function EconomyPage() {
  return (
    <>
      <Hero />
      <Section
        eyebrow="The C.A.R.E. Economy"
        title="The C.A.R.E. Economy has three currencies"
        subtitle="The public economy has three currencies: $OMEGA on EVM, TWC on Solana, and AMITY on Bitcoin / Lightning / Taproot. Two currencies power the optional reader-proof demo; they do not gate the public CC BY 4.0 manuscript. The whole economy is named C.A.R.E.: Call About Resuscitating Everyone."
      >
        <TokenShowcase variant="economy" />
      </Section>

      <Section
        eyebrow="Interactive Plane Verification"
        title="Cross-Plane State Engine & Adapters"
        subtitle="Simulate and verify the 4 planes (C.A.R.E., TWC, $OMEGA, AMITY) running under shared domain state machines, verifier adapters, and fail-closed policies."
      >
        <MultiTokenWorkbench />
      </Section>

      <Section
        eyebrow="Optional reader-proof demo"
        title="Two pilot currencies demonstrate wallet verification"
        subtitle="$OMEGA and TWC are connected from wallet proof creation to server-side verification for an optional reader route. AMITY remains a separate Bitcoin / Lightning / Taproot workstream. None of these proofs is required to access the public manuscript."
      >
        <EconomyFlow />
      </Section>

      <Section
        eyebrow="Documents"
        title="The documents of the C.A.R.E. Economy"
        subtitle="The manifesto states the vision. The extensive whitepaper specifies the entire economy. The subsystem papers and the original blueprint are all readable on-site under /whitepapers."
      >
        <WhitepaperCards />
        <div style={{ marginTop: "40px" }}>
          <CallToAction label="Browse all documents" href="/whitepapers" />
        </div>
      </Section>

      <Section
        eyebrow="Economic culture"
        title="A humane economy is part of the design"
        subtitle="This project is not supposed to be a judgment machine. It should help people earn while they learn to become safer, wiser, and more useful to one another. Anonymity is a right. Transparency is a choice fueled by emotional security."
      >
        <PrinciplesGrid />
      </Section>

      <Section
        eyebrow="Governance"
        title="What the current pilot rails imply"
        subtitle="$OMEGA is the EVM-side macro-governance currency in the current build, while TWC is the Solana-side pilot currency. AMITY expands the long-term economy toward Bitcoin / Lightning / Taproot; it is not part of the optional reader-proof demo."
      >
        <GovernancePreview />
      </Section>

      <FinalCta />
    </>
  );
}

function Hero() {
  return (
    <section
      style={{
        padding: "120px 0 64px",
        borderBottom: "1px solid var(--color-border)",
        background:
          "radial-gradient(700px 360px at 30% -5%, rgba(96,165,250,0.08), transparent 60%), radial-gradient(600px 300px at 85% 0%, rgba(192,132,87,0.06), transparent 60%)",
      }}
    >
      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
        <p
          style={{
            color: "var(--color-care)",
            fontSize: "12px",
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            fontWeight: 600,
            fontFamily: "ui-monospace, monospace",
            marginBottom: "22px",
          }}
        >
          The C.A.R.E. Economy
        </p>
        <h1
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            fontSize: "clamp(38px, 6vw, 72px)",
            lineHeight: 1.05,
            fontWeight: 500,
            letterSpacing: "-0.025em",
            color: "var(--color-foreground)",
            margin: "0 0 20px",
            maxWidth: "920px",
          }}
        >
          Three currencies.
          <br />
          One economy,{" "}
          <span style={{ color: "var(--color-care)" }}>named for care.</span>
        </h1>
        <p
          style={{
            color: "var(--color-muted-strong)",
            fontSize: "clamp(16px, 1.5vw, 19px)",
            lineHeight: 1.65,
            maxWidth: "640px",
            margin: "0 0 0",
          }}
        >
          The entire economy is the C.A.R.E. Economy — Call About
          Resuscitating Everyone. It settles across three currencies: $OMEGA
          on the EVM rail, TWC on Solana, and AMITY on Bitcoin / Lightning /
          Taproot. Today, $OMEGA and TWC power an optional reader-proof demo;
          the public manuscript is available independently under CC BY 4.0.
        </p>
      </div>
    </section>
  );
}

function EconomyFlow() {
  return (
    <ol
      style={{
        listStyle: "none",
        margin: "0",
        padding: "0",
        display: "grid",
        gap: "0",
        counterReset: "flow",
      }}
    >
      {[
        {
          step: "01",
          title: "$OMEGA — EVM release rail",
          body: "$OMEGA is the EVM-side currency currently wired into an optional wallet-proof reader demonstration. The public CC BY 4.0 manuscript does not require a token or proof.",
          color: "var(--color-omega)",
        },
        {
          step: "02",
          title: "TWC — Solana release rail",
          body: "TWC is the Solana-side currency currently wired into an optional wallet-proof reader demonstration. Reading or reusing the public CC BY 4.0 manuscript does not require a token or proof.",
          color: "var(--color-twc)",
        },
      ].map((item, i) => (
        <li
          key={item.step}
          style={{
            display: "grid",
            gridTemplateColumns: "120px 1fr",
            gap: "32px",
            padding: "32px 0",
            borderTop: i === 0 ? "1px solid var(--color-border)" : "none",
            alignItems: "start",
          }}
        >
          <div
            style={{
              fontFamily: "ui-serif, Georgia, Cambria, serif",
              fontSize: "clamp(32px, 5vw, 48px)",
              fontWeight: 500,
              color: "var(--color-muted)",
              letterSpacing: "-0.02em",
              lineHeight: 1,
            }}
          >
            {item.step}
          </div>
          <div>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                marginBottom: "10px",
              }}
            >
              <div
                style={{
                  width: "10px",
                  height: "10px",
                  borderRadius: "50%",
                  background: item.color,
                  boxShadow: `0 0 16px ${item.color}77`,
                }}
              />
              <div
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: "13px",
                  fontWeight: 600,
                  letterSpacing: "0.04em",
                  color: item.color,
                  textTransform: "uppercase",
                }}
              >
                {item.title}
              </div>
            </div>
            <p
              style={{
                color: "var(--color-muted-strong)",
                fontSize: "16px",
                lineHeight: 1.65,
                margin: 0,
              }}
            >
              {item.body}
            </p>
          </div>
        </li>
      ))}
    </ol>
  );
}

function WhitepaperCards() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
        gap: "18px",
      }}
    >
      {[
        {
          title: "The C.A.R.E. Economy — A Manifesto",
          type: "Manifesto",
          desc: "The vision in first person: the judgment economy we are leaving, the eight promises, what we refuse to build, and the invitation.",
          href: "/manifesto",
          color: "var(--color-care)",
        },
        {
          title: "The C.A.R.E. Economy — Extensive Whitepaper",
          type: "Whitepaper",
          desc: "The complete specification of the entire economy: Call About Resuscitating Everyone, five planes, three rails, proofs, privacy, audits, governance, threat model, and the honest status table.",
          href: "/whitepapers/care_economy_whitepaper",
          color: "var(--color-accent)",
        },
        {
          title: "C.A.R.E. / AMITY Protocol Whitepaper",
          type: "Whitepaper",
          desc: "Proof of Care, CARE Verifiers, Archangels and the 80/20 rule, the thought-virus model, privacy, arbitration, and the exchange boundary.",
          href: "/whitepapers/care_amity_protocol_whitepaper",
          color: "var(--color-amity)",
        },
        {
          title: "Tri-Token Sovereign Economy Blueprint",
          type: "Blueprint",
          desc: "The original systems spec behind the project, including the broader economy that extends beyond the two currently wired release currencies.",
          href: "/whitepapers/tri_token_sovereign_economy_blueprint",
          color: "var(--color-use)",
        },
      ].map((card) => (
        <Link
          key={card.title}
          href={card.href}
          style={{
            display: "block",
            padding: "26px 24px",
            border: "1px solid var(--color-border-strong)",
            borderRadius: "14px",
            background: "rgba(255,255,255,0.015)",
            textDecoration: "none",
          }}
          className="hover-raise"
        >
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              marginBottom: "14px",
            }}
          >
            <div
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: card.color,
                boxShadow: `0 0 12px ${card.color}88`,
              }}
            />
            <span
              style={{
                fontFamily: "ui-monospace, monospace",
                fontSize: "11px",
                letterSpacing: "0.12em",
                textTransform: "uppercase",
                color: "var(--color-muted)",
                fontWeight: 600,
              }}
            >
              {card.type}
            </span>
          </div>

          <h3
            style={{
              fontFamily: "ui-serif, Georgia, Cambria, serif",
              fontSize: "20px",
              fontWeight: 500,
              color: "var(--color-foreground)",
              margin: "0 0 10px",
              letterSpacing: "-0.01em",
              lineHeight: 1.25,
            }}
          >
            {card.title}
          </h3>

          <p
            style={{
              color: "var(--color-muted-strong)",
              fontSize: "14px",
              lineHeight: 1.6,
              margin: 0,
            }}
          >
            {card.desc}
          </p>
        </Link>
      ))}
    </div>
  );
}

function PrinciplesGrid() {
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
          body: "Participation should begin with dignity. People do not need to be finished products to start contributing.",
          color: "var(--color-care)",
        },
        {
          title: "Earn while you learn",
          body: "The economy should reward repair, service, contribution, and self-improvement instead of forcing people to choose between growth and survival.",
          color: "var(--color-use)",
        },
        {
          title: "Anonymity is a right",
          body: "Pseudonymous participation is not a loophole here. It is part of how people stay safe enough to begin.",
          color: "var(--color-amity)",
        },
        {
          title: "Transparency is a choice",
          body: "Disclosure should be invited by trust, emotional security, and demonstrated care — not demanded as the entry fee for belonging.",
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

function GovernancePreview() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
        gap: "16px",
      }}
    >
      {[
        {
          token: "$OMEGA",
          color: "var(--color-omega)",
          role: "Current explicit governance rail",
          examples: [
            "Economy-wide parameter changes",
            "Release-policy updates",
            "Foundation stewardship votes",
            "Open-source roadmap ratification",
          ],
        },
        {
          token: "TWC",
          color: "var(--color-twc)",
          role: "Current Solana pilot rail",
          examples: [
            "Solana-side holder verification",
            "Optional reader-proof demo",
            "Cross-chain launch coordination",
            "Pilot-to-mainnet transition planning",
          ],
        },
      ].map((group) => (
        <div
          key={group.token}
          style={{
            padding: "24px",
            border: "1px solid var(--color-border)",
            borderRadius: "14px",
            background: "rgba(255,255,255,0.012)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              marginBottom: "18px",
            }}
          >
            <div
              style={{
                width: "10px",
                height: "10px",
                borderRadius: "50%",
                background: group.color,
                boxShadow: `0 0 16px ${group.color}77`,
              }}
            />
            <div
              style={{
                fontFamily: "ui-monospace, monospace",
                fontSize: "13px",
                fontWeight: 600,
                letterSpacing: "0.04em",
                color: group.color,
                textTransform: "uppercase",
              }}
            >
              {group.token}
            </div>
          </div>

          <div
            style={{
              fontSize: "13px",
              fontWeight: 600,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
              color: "var(--color-muted)",
              marginBottom: "14px",
            }}
          >
            {group.role}
          </div>

          <ul
            style={{
              listStyle: "none",
              margin: "0",
              padding: "0",
              display: "flex",
              flexDirection: "column",
              gap: "8px",
            }}
          >
            {group.examples.map((ex) => (
              <li
                key={ex}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  fontSize: "14px",
                  color: "var(--color-muted-strong)",
                  lineHeight: 1.5,
                }}
              >
                <span
                  style={{
                    width: "4px",
                    height: "4px",
                    borderRadius: "50%",
                    background: group.color,
                    flexShrink: 0,
                  }}
                />
                {ex}
              </li>
            ))}
          </ul>
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
        background:
          "radial-gradient(600px 280px at 50% 0%, rgba(192,132,87,0.06), transparent 70%)",
      }}
    >
      <div
        style={{
          maxWidth: "720px",
          margin: "0 auto",
          padding: "0 24px",
          textAlign: "center",
        }}
      >
        <h2
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            fontSize: "clamp(26px, 3.6vw, 40px)",
            fontWeight: 500,
            letterSpacing: "-0.02em",
            color: "var(--color-foreground)",
            margin: "0 0 14px",
          }}
        >
          Read the public novel. Explore the economy.
        </h2>
        <p
          style={{
            color: "var(--color-muted-strong)",
            fontSize: "17px",
            lineHeight: 1.6,
            margin: "0 0 32px",
          }}
        >
          The manuscript is publicly available under CC BY 4.0. The current
          wallet integration demonstrates two optional proof routes; the MCP hub
          is a separate in-memory simulation, not a live operating ledger. The
          whole design — manifesto, extensive whitepaper, and subsystem papers —
          is documented on this site.
        </p>
        <div
          style={{
            display: "flex",
            gap: "14px",
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <CallToAction label="Read Crucible" href="/novel" />
          <CallToAction label="Read the whitepapers" href="/whitepapers" />
          <CallToAction label="See the MCP prototype" href="/mcp" />
        </div>
      </div>
    </section>
  );
}
