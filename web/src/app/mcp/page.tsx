// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
"use client";

import { Section } from "@/components/Section";
import { TokenShowcase } from "@/components/TokenShowcase";
import { CallToAction } from "@/components/CallToAction";

export default function McpPage() {
  return (
    <>
      <Hero />

      <section style={{ padding: "80px 0 0", borderTop: "1px solid var(--color-border)" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
          <p
            style={{
              color: "var(--color-omega)",
              fontSize: "12px",
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              fontWeight: 600,
              fontFamily: "ui-monospace, monospace",
              marginBottom: "18px",
            }}
          >
            Omega MCP Hub
          </p>
          <h2
            style={{
              fontFamily: "ui-serif, Georgia, Cambria, serif",
              fontSize: "clamp(28px, 4vw, 44px)",
              fontWeight: 500,
              letterSpacing: "-0.02em",
              color: "var(--color-foreground)",
              margin: "0 0 14px",
              lineHeight: 1.1,
            }}
          >
            The operating layer for the tri-token economy.
          </h2>
          <p
            style={{
              color: "var(--color-muted-strong)",
              fontSize: "clamp(15px, 1.5vw, 18px)",
              lineHeight: 1.65,
              maxWidth: "720px",
              margin: "0 0 40px",
            }}
          >
            Omega MCP Hub is a FastMCP server with 22 tools supporting a
            three-currency economy. It runs the economy: it settles accounts,
            verifies useful work, executes governance, supports participation, and
            anchors release-day coordination. Built to be driven by AI agents —
            and open source, so anyone can extend it.
          </p>

          <div
            style={{
              display: "flex",
              gap: "28px",
              flexWrap: "wrap",
              padding: "24px 0",
              borderTop: "1px solid var(--color-border)",
              borderBottom: "1px solid var(--color-border)",
            }}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <div
                style={{
                  fontFamily: "ui-serif, Georgia, serif",
                  fontSize: "32px",
                  fontWeight: 500,
                  color: "var(--color-foreground)",
                  letterSpacing: "-0.02em",
                  lineHeight: 1,
                }}
              >
                22
              </div>
              <div style={{ fontSize: "13px", color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                MCP tools
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <div
                style={{
                  fontFamily: "ui-serif, Georgia, serif",
                  fontSize: "32px",
                  fontWeight: 500,
                  color: "var(--color-foreground)",
                  letterSpacing: "-0.02em",
                  lineHeight: 1,
                }}
              >
                3
              </div>
              <div style={{ fontSize: "13px", color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                Economy currencies
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <div
                style={{
                  fontFamily: "ui-serif, Georgia, serif",
                  fontSize: "32px",
                  fontWeight: 500,
                  color: "var(--color-foreground)",
                  letterSpacing: "-0.02em",
                  lineHeight: 1,
                }}
              >
                FastMCP
              </div>
              <div style={{ fontSize: "13px", color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                MCP SDK v1
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              <div
                style={{
                  fontFamily: "ui-serif, Georgia, serif",
                  fontSize: "32px",
                  fontWeight: 500,
                  color: "var(--color-foreground)",
                  letterSpacing: "-0.02em",
                  lineHeight: 1,
                }}
              >
                Stdio + HTTP
              </div>
              <div style={{ fontSize: "13px", color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                Transport modes
              </div>
            </div>
          </div>
        </div>
      </section>

      <Section
        eyebrow="Architecture"
        title="How the hub is built"
        subtitle="Omega MCP Hub is a single FastMCP stdio server, packaged as a Python package, with a manifest for distribution. It is designed to be the backbone of the Omega three-currency economy — and to be extended by anyone who wants to add tools, internal domains, or integrations."
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "16px",
            marginTop: "8px",
          }}
        >
          {[
            {
              label: "Runtime",
              value: "Python 3.12+",
              desc: "FastMCP runs on Python. The hub is a Python package with typed tools, async handlers, and a clean stdio transport.",
            },
            {
              label: "API",
              value: "FastMCP v1",
              desc: "Built on the MCP SDK v1 FastMCP API — 22 tools at launch, grouped by token plane, with typed input/output schemas.",
            },
            {
              label: "Transport",
              value: "Stdio + HTTP",
              desc: "Run as a stdio server (for MCP clients) or as an HTTP server (for web and programmatic access). One codebase, two transports.",
            },
            {
              label: "Package",
              value: "pip-installable",
              desc: "Packaged as an installable Python package with a manifest. Add it to any MCP client or run it standalone.",
            },
          ].map((card) => (
            <div
              key={card.label}
              style={{
                padding: "22px 20px",
                border: "1px solid var(--color-border)",
                borderRadius: "14px",
                background: "rgba(255,255,255,0.012)",
                display: "flex",
                flexDirection: "column",
                gap: "10px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
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
                  {card.label}
                </span>
                <span
                  style={{
                    fontFamily: "ui-monospace, monospace",
                    fontSize: "11px",
                    color: "var(--color-omega)",
                    letterSpacing: "0.04em",
                    fontWeight: 600,
                  }}
                >
                  {card.value}
                </span>
              </div>
              <p
                style={{
                  fontSize: "14px",
                  color: "var(--color-muted-strong)",
                  lineHeight: 1.55,
                  margin: 0,
                }}
              >
                {card.desc}
              </p>
            </div>
          ))}
        </div>
      </Section>

      <Section
        eyebrow="Economy support"
        title="22 tools supporting three currencies"
        subtitle="Publicly, the economy is three-currency: $OMEGA on EVM, TWC on Solana, and AMITY on Bitcoin / Lightning / Taproot. Under the hood, the MCP package uses several internal operating domains to serve that economy."
      >
        <TokenShowcase variant="economy" />
        <div style={{ marginTop: "48px" }}>
          <ToolList />
        </div>
      </Section>

      <Section
        eyebrow="Repository"
        title="Open source on GitHub"
        subtitle="Omega MCP Hub is part of the Omega_Theory_Everything repository — the same repo that holds the tri-token economy blueprint, the whitepapers, and the novel. The hub is the operating layer. The repo is the source of truth."
      >
        <RepoCard />
        <div style={{ marginTop: "40px" }}>
          <CallToAction label="View on GitHub" href="https://github.com/jakesdev1991/Omega_Theory_Everything" />
          <CallToAction label="Read the economy docs" href="/economy" variant="ghost" />
        </div>
      </Section>

      <Section
        eyebrow="Connect"
        title="Run your own hub"
        subtitle="Spin up Omega MCP Hub locally, connect it to your MCP client, and start operating a three-currency economy with AI agents. The server is stdio-first — drop it into any MCP-compatible toolchain."
      >
        <RunCard />
        <div style={{ marginTop: "40px", display: "flex", gap: "14px", flexWrap: "wrap" }}>
          <CallToAction label="Clone the repo" href="https://github.com/jakesdev1991/Omega_Theory_Everything" />
          <CallToAction label="Explore the economy" href="/economy" variant="ghost" />
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
        padding: "140px 0 72px",
        borderBottom: "1px solid var(--color-border)",
        background:
          "radial-gradient(800px 400px at 70% -10%, rgba(251,191,36,0.08), transparent 60%), radial-gradient(700px 350px at 20% 0%, rgba(96,165,250,0.06), transparent 60%)",
      }}
    >
      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
        <p
          style={{
            color: "var(--color-omega)",
            fontSize: "12px",
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            fontWeight: 600,
            fontFamily: "ui-monospace, monospace",
            marginBottom: "24px",
          }}
        >
          The Operating Hub
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
          Omega MCP Hub.
          <br />
          <span style={{ color: "var(--color-omega)" }}>
            The economy, operationalized.
          </span>
        </h1>
        <p
          style={{
            color: "var(--color-muted-strong)",
            fontSize: "clamp(16px, 1.5vw, 19px)",
            lineHeight: 1.7,
            maxWidth: "660px",
            margin: "0",
          }}
        >
          A FastMCP server with 22 tools supporting a three-currency economy —
          operated by AI agents, governed by human choice, and open source for
          anyone to extend.
        </p>
      </div>
    </section>
  );
}

function ToolList() {
  const tools = [
    { plane: "SOv", tools: ["ledger_snapshot", "settle_position", "resolve_account", "audit_trail"] },
    { plane: "USE", tools: ["issue_receipt", "verify_work", "receipt_balance", "contribution_ledger"] },
    { plane: "CARE", tools: ["propose_governance", "cast_vote", "steward_balance", "treasury_view", "council_roster"] },
    { plane: "AMITY", tools: ["convert_care_to_amity", "amity_balance", "market_quote", "exchange_order"] },
    { plane: "$OMEGA", tools: ["macro_governance", "omega_supply", "anchor_position", "vote_weight", "release_unlock", "snapshot_epoch"] },
  ];

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        padding: "24px 0",
        borderTop: "1px solid var(--color-border)",
      }}
    >
      {tools.map((group) => (
        <div
          key={group.plane}
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: "16px",
            padding: "14px 0",
          }}
        >
          <div
            style={{
              fontFamily: "ui-monospace, monospace",
              fontSize: "11px",
              fontWeight: 700,
              letterSpacing: "0.08em",
              color: "var(--color-foreground)",
              padding: "4px 10px",
              border: "1px solid var(--color-border-strong)",
              borderRadius: "6px",
              background: "transparent",
              flexShrink: 0,
              height: "fit-content",
              marginTop: "2px",
            }}
          >
            {group.plane}
          </div>
          <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", alignItems: "center" }}>
            {group.tools.map((tool) => (
              <span
                key={tool}
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: "12px",
                  padding: "4px 10px",
                  border: "1px solid var(--color-border)",
                  borderRadius: "6px",
                  background: "rgba(255,255,255,0.02)",
                  color: "var(--color-muted-strong)",
                }}
              >
                {tool}
              </span>
            ))}
            <span
              style={{
                fontSize: "11px",
                color: "var(--color-muted)",
                fontFamily: "ui-monospace, monospace",
                marginLeft: "4px",
              }}
            >
              ({group.tools.length} tools)
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

function RepoCard() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
        gap: "16px",
      }}
    >
      <div
        style={{
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
            marginBottom: "12px",
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
            wordBreak: "break-all",
          }}
        >
          github.com/jakesdev1991/Omega_Theory_Everything
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
          <span style={{ color: "var(--color-omega)" }}>git</span> clone{" "}
          https://github.com/jakesdev1991/Omega_Theory_Everything.git
        </div>
      </div>

      <div
        style={{
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
            marginBottom: "12px",
          }}
        >
          In the repo
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
          {[
            "mcp/ — Omega MCP Hub server",
            "whitepapers/ — tri-token economy spec",
            "tri_token_sovereign_economy_blueprint.md",
            "app/ — customer-facing site source",
            "Genesis Block novel draft",
          ].map((item) => (
            <div
              key={item}
              style={{
                fontSize: "13px",
                color: "var(--color-muted-strong)",
                padding: "4px 0",
                borderBottom: "1px solid var(--color-border)",
                fontFamily: "ui-monospace, monospace",
              }}
            >
              {item}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function RunCard() {
  return (
    <div
      style={{
        padding: "28px",
        border: "1px solid var(--color-border-strong)",
        borderRadius: "14px",
        background: "rgba(0,0,0,0.25)",
        fontFamily: "ui-monospace, monospace",
        fontSize: "13px",
        color: "var(--color-foreground)",
        lineHeight: 1.7,
        overflow: "auto",
      }}
    >
      <div style={{ color: "var(--color-muted)", marginBottom: "12px" }}>
        # Clone and run the Omega MCP Hub
      </div>
      <div>
        <span style={{ color: "var(--color-omega)" }}>git</span> clone{" "}
        https://github.com/jakesdev1991/Omega_Theory_Everything.git
      </div>
      <div>cd Omega_Theory_Everything</div>
      <div>cd mcp</div>
      <div>uv venv /home/jake/.venvs/omwga-mcp</div>
      <div>source /home/jake/.venvs/omwga-mcp/bin/activate</div>
      <div>uv pip install -e .</div>
      <div>python -m omega_mcp --transport stdio</div>
      <div style={{ color: "var(--color-muted)", marginTop: "12px" }}>
        # Or run as HTTP server
      </div>
      <div>python -m omega_mcp --transport http --host 0.0.0.0 --port 8000</div>
      <div style={{ color: "var(--color-muted)", marginTop: "12px" }}>
        # Connect via any MCP client or the website
      </div>
    </div>
  );
}

function FinalCta() {
  return (
    <section
      style={{
        padding: "72px 0",
        borderTop: "1px solid var(--color-border)",
        background: "radial-gradient(600px 280px at 50% 0%, rgba(251,191,36,0.06), transparent 70%)",
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
          The hub is the economy.
        </h2>
        <p style={{ color: "var(--color-muted-strong)", fontSize: "17px", lineHeight: 1.6, margin: "0 0 32px" }}>
          Omega MCP Hub runs the economy. The novel rewards participation. The
          website connects everyone. All of it is open source, all of it is here,
          and all of it is waiting for you to join.
        </p>
        <div style={{ display: "flex", gap: "14px", justifyContent: "center", flexWrap: "wrap" }}>
          <CallToAction label="Run the hub" href="https://github.com/jakesdev1991/Omega_Theory_Everything" />
          <CallToAction label="Read the novel" href="/novel" variant="ghost" />
          <CallToAction label="Invest in the economy" href="/invest" variant="ghost" />
        </div>
      </div>
    </section>
  );
}
