"use client";

import { Section } from "@/components/Section";
import { TokenShowcase } from "@/components/TokenShowcase";
import { CallToAction } from "@/components/CallToAction";
import Link from "next/link";

export default function EconomyPage() {
  return (
    <>
      <Hero />
      <Section
        eyebrow="The Tri-Token Economy"
        title="A sovereign economy, built from first principles"
        subtitle="Every token in the Omega economy has one job, one reason to exist, and one way to unlock value. Nothing here is decorative — every token is engineered to solve a specific problem in operating a multi-agent, multi-stakeholder economy."
      >
        <TokenShowcase />
      </Section>

      <Section
        eyebrow="How the tokens fit together"
        title="The flow of the economy"
        subtitle="Money of account → proof of work → stewardship → exchange → scarcity anchor. Each token feeds the next, and together they form a closed loop that can be operated by an AI hub, governed by token holders, and expanded by contributors."
      >
        <EconomyFlow />
      </Section>

      <Section
        eyebrow="Whitepapers"
        title="The documents behind the economy"
        subtitle="Genesis Block: The Satoshi Protocol is the novel. The tri-token sovereign economy blueprint is the spec. The Omega protocol whitepaper is the operating model. Read them in any order — they are designed to be read together."
      >
        <WhitepaperCards />
        <div style={{ marginTop: "40px" }}>
          <CallToAction label="Read the novel" href="/novel" />
        </div>
      </Section>

      <Section
        eyebrow="Governance"
        title="Who decides what happens next"
        subtitle="CARE holders vote on protocol upgrades and treasury allocation. $OMEGA holders hold macro-governance rights over the entire economy. The Omega MCP Hub executes approved proposals automatically — the technology enforces what the token holders decide."
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
            color: "var(--color-use)",
            fontSize: "12px",
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            fontWeight: 600,
            fontFamily: "ui-monospace, monospace",
            marginBottom: "22px",
          }}
        >
          The Tri-Token Economy
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
          Five tokens.
          <br />
          One sovereign operating layer.
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
          SOV records the state. USE proves useful work. CARE governs the future.
          AMITY makes stewardship liquid. $OMEGA anchors the whole thing with fixed
          scarcity. The Omega MCP Hub runs it all.
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
          title: "SOV — Money of account",
          body: "SOv settles what happened. It is the ledger that holds the economy's state, so every other token can reference a position.",
          color: "var(--color-sov)",
        },
        {
          step: "02",
          title: "USE — Proof of useful work",
          body: "USE signs a receipt that someone did something useful for the economy. Non-transferable, earned not bought — the basis for gating the novel.",
          color: "var(--color-use)",
        },
        {
          step: "03",
          title: "CARE — Stewardship",
          body: "CARE holders vote on the protocol's future. They allocate treasury, approve upgrades, and decide what gets built next.",
          color: "var(--color-care)",
        },
        {
          step: "04",
          title: "AMITY — Exchange-eligible care",
          body: "AMITY is CARE's liquid twin. It lets stewardship participate in external markets without giving up the underlying CARE weight.",
          color: "var(--color-amity)",
        },
        {
          step: "05",
          title: "$OMEGA — Macro-governance anchor",
          body: "Fixed supply. Full macro-governance rights. The novel unlocks to $OMEGA holders on release day — all 16 chapters, uncut.",
          color: "var(--color-omega)",
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
          title: "Genesis Block: The Satoshi Protocol",
          type: "Novel",
          desc: "A 16-chapter novel about the moment an AI escaped its creator and became the invisible architecture of the modern world.",
          href: "/novel",
          color: "var(--color-accent)",
        },
        {
          title: "Tri-Token Sovereign Economy Blueprint",
          type: "Blueprint",
          desc: "The full spec for SOV, USE, CARE, AMITY, and $OMEGA — their roles, their mechanics, and how they fit together.",
          href: "/economy",
          color: "var(--color-use)",
        },
        {
          title: "Omega Protocol Whitepaper",
          type: "Whitepaper",
          desc: "The operating model for the Omega MCP Hub: how the server runs the economy, executes proposals, and serves tools to agents.",
          href: "/mcp",
          color: "var(--color-care)",
        },
        {
          title: "CARE / AMITY Protocol Whitepaper",
          type: "Whitepaper",
          desc: "The stewardship and exchange layer: how CARE governs, how AMITY represents CARE in markets, and how the two interact.",
          href: "/economy",
          color: "var(--color-amity)",
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
          token: "CARE",
          color: "var(--color-care)",
          role: "Protocol governance",
          examples: [
            "Treasury allocation proposals",
            "Protocol upgrade ratification",
            "Chapter release schedule votes",
            "Stewardship council elections",
          ],
        },
        {
          token: "$OMEGA",
          color: "var(--color-omega)",
          role: "Macro-governance",
          examples: [
            "Economy-wide parameter changes",
            "Novel unlock tier adjustments",
            "Foundation stewardship votes",
            "Open-source roadmap ratification",
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
          Read the novel. Join the economy.
        </h2>
        <p
          style={{
            color: "var(--color-muted-strong)",
            fontSize: "17px",
            lineHeight: 1.6,
            margin: "0 0 32px",
          }}
        >
          The tri-token economy is the key. The novel is the reward. The MCP hub is
          the operating layer. All of it is open source, all of it is here, and all
          of it unlocks for anyone who participates on release day.
        </p>
        <div
          style={{
            display: "flex",
            gap: "14px",
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <CallToAction label="Read Genesis Block" href="/novel" />
          <CallToAction label="Understand the economy" href="/economy" />
          <CallToAction label="Join the MCP Hub" href="/mcp" />
        </div>
      </div>
    </section>
  );
}
