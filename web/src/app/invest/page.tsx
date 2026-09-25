// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
"use client";

import { Section } from "@/components/Section";
import { CallToAction } from "@/components/CallToAction";
import { TOKENS, WIRED_TOKENS } from "@/lib/types";
import Link from "next/link";

export default function InvestPage() {
  return (
    <>
      <Hero />
      <ParticipationPrinciples />
      <UnlockTiers />
      <TokenParticipate />
      <HowItWorks />
      <WalletSection />
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
          "radial-gradient(800px 400px at 30% -10%, rgba(244,114,182,0.08), transparent 60%), radial-gradient(700px 350px at 85% 0%, rgba(167,139,250,0.06), transparent 60%)",
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
            marginBottom: "24px",
          }}
        >
          Participate on Release Day
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
            maxWidth: "860px",
          }}
        >
          Invest in the economy.
          <br />
          <span style={{ color: "var(--color-care)" }}>Unlock the novel.</span>
        </h1>
        <p
          style={{
            color: "var(--color-muted-strong)",
            fontSize: "clamp(16px, 1.5vw, 19px)",
            lineHeight: 1.7,
            maxWidth: "760px",
            margin: "0 0 36px",
          }}
        >
          This economy is meant to help people earn while they learn to become
          better people. The current release build wires two live rails into the
          wallet and web app, inside a larger three-currency economy. Sign with
          either live rail to unlock{" "}
          <em style={{ color: "var(--color-accent-soft)" }}>
            Genesis Block: The Satoshi Protocol
          </em>{" "}
          by Akash Varma.
        </p>
        <div
          style={{
            display: "flex",
            gap: "14px",
            flexWrap: "wrap",
          }}
        >
          <CallToAction label="Start participating" href="#tokens" />
          <CallToAction label="See the unlock rails" href="#tiers" variant="ghost" />
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
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <div
              style={{
                fontFamily: "ui-serif, Georgia, serif",
                fontSize: "36px",
                fontWeight: 500,
                color: "var(--color-foreground)",
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}
            >
              {TOKENS.length} currencies
            </div>
            <div style={{ fontSize: "13px", color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              In the economy
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <div
              style={{
                fontFamily: "ui-serif, Georgia, serif",
                fontSize: "36px",
                fontWeight: 500,
                color: "var(--color-foreground)",
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}
            >
              {WIRED_TOKENS.length} live rails
            </div>
            <div style={{ fontSize: "13px", color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              Wired today
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <div
              style={{
                fontFamily: "ui-serif, Georgia, serif",
                fontSize: "36px",
                fontWeight: 500,
                color: "var(--color-foreground)",
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}
            >
              16 chapters
            </div>
            <div style={{ fontSize: "13px", color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              Unlocked by participation
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            <div
              style={{
                fontFamily: "ui-serif, Georgia, serif",
                fontSize: "36px",
                fontWeight: 500,
                color: "var(--color-foreground)",
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}
            >
              Your choice
            </div>
            <div style={{ fontSize: "13px", color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
              Anonymity first, transparency optional
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}


function ParticipationPrinciples() {
  return (
    <Section
      eyebrow="Economic culture"
      title="Come in without judgment"
      subtitle="The point is not to shame people into becoming better. The point is to create conditions where people can learn, contribute, recover, and choose greater visibility when they actually feel safe enough to do so."
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
            title: "No judgment at the door",
            body: "Participation should begin with dignity. You do not need to arrive as a finished person to begin contributing.",
            color: "var(--color-care)",
          },
          {
            title: "Earn while you learn",
            body: "Useful work, care, stewardship, and repair should become paths toward both livelihood and character growth.",
            color: "var(--color-use)",
          },
          {
            title: "Anonymity is a right",
            body: "People should be able to start pseudonymously. Privacy is part of safety, not evidence of bad intent.",
            color: "var(--color-amity)",
          },
          {
            title: "Transparency is a choice",
            body: "Disclosure should grow from emotional security and trust. Coerced transparency only teaches fear.",
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
            <h3 style={{ fontFamily: "ui-serif, Georgia, Cambria, serif", fontSize: "18px", fontWeight: 500, color: "var(--color-foreground)", margin: "0 0 10px" }}>
              {item.title}
            </h3>
            <p style={{ margin: 0, color: "var(--color-muted-strong)", fontSize: "14px", lineHeight: 1.6 }}>
              {item.body}
            </p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function UnlockTiers() {
  const tiers = [
    {
      tier: "1",
      title: "$OMEGA",
      requirement: "Sign an EVM-side release proof from the wired $OMEGA wallet flow",
      unlocks: "All 16 chapters, uncut — including The Final Hybrid and Epilogue",
      color: "var(--color-omega)",
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
        </svg>
      ),
    },
    {
      tier: "2",
      title: "TWC",
      requirement: "Sign a Solana-side release proof from the wired TWC wallet flow",
      unlocks: "All 16 chapters, uncut — including The Final Hybrid and Epilogue",
      color: "var(--color-twc)",
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="9" />
          <path d="M8 12h8" />
          <path d="M12 8v8" />
        </svg>
      ),
    },
  ];

  return (
    <Section
      eyebrow="The Unlock"
      title="Two wired currencies unlock the novel"
      subtitle="The current release build does not use five chapter tiers. Instead, either of the two wired currencies — $OMEGA or TWC — can sign a proof that unlocks the full novel."
      id="tiers"
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "16px",
          marginTop: "8px",
        }}
      >
        {tiers.map((tier) => (
          <div
            key={tier.tier}
            style={{
              padding: "22px 20px",
              border: "1px solid var(--color-border)",
              borderRadius: "14px",
              background: "rgba(255,255,255,0.012)",
              position: "relative",
              overflow: "hidden",
            }}
            className="hover-raise-lg"
          >
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: "3px",
                background: tier.color,
                boxShadow: `0 0 20px ${tier.color}66`,
              }}
            />
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "10px",
                marginBottom: "14px",
              }}
            >
              <div
                style={{
                  width: "34px",
                  height: "34px",
                  borderRadius: "50%",
                  background: `${tier.color}18`,
                  color: tier.color,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontWeight: 700,
                  fontSize: "14px",
                  fontFamily: "ui-monospace, monospace",
                  flexShrink: 0,
                }}
              >
                {tier.tier}
              </div>
              <span
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: "11px",
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  color: tier.color,
                  fontWeight: 600,
                }}
              >
                Rail {tier.tier}
              </span>
            </div>

            <h3
              style={{
                fontFamily: "ui-serif, Georgia, Cambria, serif",
                fontSize: "18px",
                fontWeight: 500,
                color: "var(--color-foreground)",
                margin: "0 0 8px",
              }}
            >
              {tier.title}
            </h3>

            <p
              style={{
                fontSize: "13px",
                color: "var(--color-muted-strong)",
                lineHeight: 1.55,
                margin: "0 0 12px",
              }}
            >
              {tier.requirement}
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
              <span style={{ color: "var(--color-unlock)", fontWeight: 600 }}>Unlocks:</span>{" "}
              {tier.unlocks}
            </div>

            <div
              style={{
                marginTop: "14px",
                display: "flex",
                gap: "8px",
              }}
            >
              <div
                style={{
                  width: "24px",
                  height: "24px",
                  borderRadius: "6px",
                  background: tier.color,
                  color: "#07080c",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  opacity: 0.9,
                }}
              >
                {tier.icon}
              </div>
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}

function TokenParticipate() {
  return (
    <Section
      eyebrow="Tokens"
      title="Choose which wired currency you use"
      subtitle="Two currencies are connected end to end today. You do not need both — a verified $OMEGA proof or a verified TWC proof unlocks Genesis Block in full."
      id="tokens"
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "18px",
          marginTop: "8px",
        }}
      >
        {TOKENS.map((token) => (
          <div
            key={token.id}
            style={{
              padding: "24px 22px",
              border: "1px solid var(--color-border)",
              borderRadius: "14px",
              background: "rgba(255,255,255,0.012)",
            }}
            className="hover-raise"
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                marginBottom: "16px",
              }}
            >
              <div
                style={{
                  width: "44px",
                  height: "44px",
                  borderRadius: "10px",
                  background: token.color,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#07080c",
                  fontWeight: 700,
                  fontSize: "15px",
                  fontFamily: "ui-monospace, monospace",
                  boxShadow: `0 0 28px ${token.color}55`,
                }}
              >
                {token.symbol}
              </div>
              <div>
                <div
                  style={{
                    fontFamily: "ui-serif, Georgia, Cambria, serif",
                    fontSize: "17px",
                    fontWeight: 500,
                    color: "var(--color-foreground)",
                    marginBottom: "2px",
                    lineHeight: 1.2,
                  }}
                >
                  {token.name}
                </div>
                <div
                  style={{
                    fontFamily: "ui-monospace, monospace",
                    fontSize: "11px",
                    color: "var(--color-muted)",
                    letterSpacing: "0.04em",
                  }}
                >
                  {token.symbol}
                </div>
              </div>
            </div>

            <p
              style={{
                fontSize: "14px",
                color: "var(--color-muted-strong)",
                lineHeight: 1.6,
                margin: "0 0 16px",
              }}
            >
              {token.description}
            </p>

            <div
              style={{
                padding: "12px 0",
                borderTop: "1px solid var(--color-border)",
                fontSize: "12px",
                color: "var(--color-muted)",
                lineHeight: 1.5,
              }}
            >
              <div style={{ display: "flex", gap: "8px", marginBottom: "6px", flexWrap: "wrap" }}>
                <span
                  style={{
                    padding: "2px 8px",
                    borderRadius: "5px",
                    background: token.transferable ? "rgba(255,255,255,0.05)" : "rgba(244,114,182,0.1)",
                    color: token.transferable ? "var(--color-muted-strong)" : "var(--color-care)",
                    fontFamily: "ui-monospace, monospace",
                    fontSize: "10px",
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                  }}
                >
                  {token.transferable ? "Transferable" : "Non-transferable"}
                </span>
                <span
                  style={{
                    padding: "2px 8px",
                    borderRadius: "5px",
                    background: `${token.color}18`,
                    color: token.color,
                    fontFamily: "ui-monospace, monospace",
                    fontSize: "10px",
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                  }}
                >
                  {token.role.replace("-", " ")}
                </span>
              </div>
              {token.notionalPrice && (
                <p style={{ margin: "4px 0 0", fontFamily: "ui-monospace, monospace", fontSize: "11px", color: "var(--color-muted)" }}>
                  {token.notionalPrice}
                </p>
              )}
            </div>

            <div
              style={{
                marginTop: "14px",
                padding: "10px 12px",
                borderRadius: "8px",
                background: "rgba(52,211,153,0.06)",
                border: "1px solid var(--color-unlock-soft)",
                fontSize: "12px",
                color: "var(--color-unlock)",
                lineHeight: 1.45,
              }}
            >
              <span style={{ fontWeight: 600 }}>Unlocks:</span> {token.unlockRequirement}
            </div>
          </div>
        ))}
      </div>
    </Section>
  );
}

function HowItWorks() {
  const steps = [
    {
      step: "01",
      title: "Enter without judgment",
      body: "Start with a wallet, not a confession booth. No account and no KYC are required for the current proof flow.",
      color: "var(--color-sov)",
    },
    {
      step: "02",
      title: "Choose a live rail",
      body: "Use the wired $OMEGA EVM flow or the wired TWC Solana flow. Both currently issue a signed full-unlock proof.",
      color: "var(--color-use)",
    },
    {
      step: "03",
      title: "Unlock the novel",
      body: "On release day, a verified $OMEGA or TWC proof unlocks the full text of Genesis Block.",
      color: "var(--color-care)",
    },
    {
      step: "04",
      title: "Grow into the wider economy",
      body: "The broader economy is three-currency: $OMEGA, TWC, and AMITY. The live release build starts with two rails and expands from there.",
      color: "var(--color-amity)",
    },
    {
      step: "05",
      title: "Choose your visibility",
      body: "Anonymity is a right. Transparency should happen because trust and emotional security have been earned, not because people were pressured into exposure.",
      color: "var(--color-omega)",
    },
  ];

  return (
    <Section
      eyebrow="How it works"
      title="From participation to unlock"
      subtitle="Five steps from dignified entry to verified access. Simple, sovereign, and built around choice instead of coercion."
    >
      <ol style={{ listStyle: "none", margin: "0", padding: "0", display: "flex", flexDirection: "column", gap: "0" }}>
        {steps.map((s, i) => (
          <li
            key={s.step}
            style={{
              display: "grid",
              gridTemplateColumns: "80px 1fr",
              gap: "32px",
              padding: "32px 0",
              borderTop: i === 0 ? "1px solid var(--color-border)" : "none",
              alignItems: "start",
            }}
          >
            <div
              style={{
                fontFamily: "ui-serif, Georgia, Cambria, serif",
                fontSize: "clamp(28px, 4vw, 40px)",
                fontWeight: 500,
                color: "var(--color-muted)",
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}
            >
              {s.step}
            </div>
            <div>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  marginBottom: "8px",
                }}
              >
                <div
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    background: s.color,
                    boxShadow: `0 0 12px ${s.color}88`,
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{
                    fontFamily: "ui-monospace, monospace",
                    fontSize: "12px",
                    fontWeight: 600,
                    letterSpacing: "0.06em",
                    color: s.color,
                    textTransform: "uppercase",
                  }}
                >
                  {s.title}
                </span>
              </div>
              <p style={{ color: "var(--color-muted-strong)", fontSize: "15px", lineHeight: 1.6, margin: 0 }}>
                {s.body}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </Section>
  );
}

function WalletSection() {
  return (
    <Section
      eyebrow="Wallet"
      title="What you need to participate"
      subtitle="A wallet is all you need. No account. No KYC. Start privately if you want. Just bring a wallet, choose one of the two wired currencies, and decide later how visible you want to become."
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
          gap: "16px",
        }}
      >
        {[
          {
            title: "Wallet",
            desc: "Use an Ethereum wallet for $OMEGA or a Solana wallet for TWC. Those are the two release currencies currently wired into the app.",
            color: "var(--color-sov)",
          },
          {
            title: "Tokens",
            desc: "Choose the $OMEGA Ethereum rail or the TWC Solana rail. Both currently sign a full-unlock proof for the novel.",
            color: "var(--color-use)",
          },
          {
            title: "Release day",
            desc: "On release day, your signed $OMEGA or TWC proof unlocks Genesis Block. The current release flow opens the whole novel.",
            color: "var(--color-care)",
          },
          {
            title: "MCP hub",
            desc: "Operate your position via the Omega MCP Hub — 22 tools supporting a three-currency economy, driven by AI agents.",
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
                width: "32px",
                height: "32px",
                borderRadius: "8px",
                background: item.color,
                marginBottom: "14px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#07080c",
                fontWeight: 700,
                fontSize: "14px",
                fontFamily: "ui-monospace, monospace",
                boxShadow: `0 0 18px ${item.color}55`,
              }}
            >
              ?
            </div>
            <h3
              style={{
                fontFamily: "ui-serif, Georgia, Cambria, serif",
                fontSize: "17px",
                fontWeight: 500,
                color: "var(--color-foreground)",
                margin: "0 0 8px",
              }}
            >
              {item.title}
            </h3>
            <p style={{ fontSize: "14px", color: "var(--color-muted-strong)", lineHeight: 1.55, margin: 0 }}>
              {item.desc}
            </p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function FinalCta() {
  return (
    <section
      style={{
        padding: "80px 0",
        borderTop: "1px solid var(--color-border)",
        background: "radial-gradient(600px 280px at 50% 0%, rgba(244,114,182,0.06), transparent 70%)",
      }}
    >
      <div style={{ maxWidth: "720px", margin: "0 auto", padding: "0 24px", textAlign: "center" }}>
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
          The novel opens on release day.
        </h2>
        <p style={{ color: "var(--color-muted-strong)", fontSize: "17px", lineHeight: 1.6, margin: "0 0 32px" }}>
          Two currencies are wired in today: $OMEGA and TWC. Genesis Block is the
          reward. The MCP hub is the operating layer. The broader economy is still
          larger than this build, but the wallet-to-web unlock path already works
          across both current rails.
        </p>
        <div style={{ display: "flex", gap: "14px", justifyContent: "center", flexWrap: "wrap" }}>
          <CallToAction label="Participate now" href="#tokens" />
          <CallToAction label="Read about the tokens" href="/economy" variant="ghost" />
          <CallToAction label="See the MCP hub" href="/mcp" variant="ghost" />
        </div>
      </div>
    </section>
  );
}
