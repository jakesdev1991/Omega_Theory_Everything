// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";

import { CallToAction } from "@/components/CallToAction";

export const metadata: Metadata = {
  title: "System Map",
  description:
    "A visual guide to the C.A.R.E. Economy website, whitepapers, wallet and independent Omega MCP Hub.",
};

const LINKS = [
  {
    href: "/whitepapers",
    eyebrow: "Design layer",
    title: "Browse the whitepapers",
    description: "Explore the manifesto, whole-economy specification and subsystem papers.",
    color: "var(--color-amity)",
  },
  {
    href: "/wallet",
    eyebrow: "Client-side wallet",
    title: "Open the wallet hub",
    description: "Launch the GUI, inspect its local-first design or download the offline bundle.",
    color: "var(--color-sov)",
  },
  {
    href: "/mcp",
    eyebrow: "Agent operations",
    title: "Explore the MCP Hub",
    description: "See how an MCP-compatible client connects to the separate tool-server prototype.",
    color: "var(--color-omega)",
  },
];

export default function ArchitecturePage() {
  return (
    <>
      <header
        style={{
          padding: "140px 0 56px",
          borderBottom: "1px solid var(--color-border)",
          background:
            "radial-gradient(800px 360px at 18% -5%, rgba(96,165,250,0.10), transparent 60%), radial-gradient(720px 360px at 85% 0%, rgba(167,139,250,0.08), transparent 60%)",
        }}
      >
        <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
          <p
            style={{
              color: "var(--color-use)",
              fontFamily: "ui-monospace, monospace",
              fontSize: "12px",
              fontWeight: 600,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              margin: "0 0 18px",
            }}
          >
            Repository system map · 2026-10-11
          </p>
          <h1
            style={{
              fontFamily: "ui-serif, Georgia, Cambria, serif",
              fontSize: "clamp(38px, 5.8vw, 70px)",
              lineHeight: 1.04,
              fontWeight: 500,
              letterSpacing: "-0.025em",
              color: "var(--color-foreground)",
              margin: "0 0 20px",
              maxWidth: "900px",
            }}
          >
            Website, whitepapers,
            <br />
            wallet and MCP — <span style={{ color: "var(--color-accent)" }}>one map.</span>
          </h1>
          <p
            style={{
              color: "var(--color-muted-strong)",
              fontSize: "clamp(16px, 1.6vw, 19px)",
              lineHeight: 1.7,
              maxWidth: "760px",
              margin: "0 0 30px",
            }}
          >
            See how the public Next.js site, document reader, local-first wallet and independent
            MCP server fit together—and where their boundaries are. The map reflects the repository
            snapshot dated October 11, 2026.
          </p>
          <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
            <CallToAction label="Browse the whitepapers" href="/whitepapers" />
            <CallToAction label="Open the wallet" href="/wallet" variant="ghost" />
            <CallToAction label="View MCP overview" href="/mcp" variant="ghost" />
          </div>
        </div>
      </header>

      <section style={{ padding: "48px 0 40px" }}>
        <div style={{ maxWidth: "1320px", margin: "0 auto", padding: "0 24px" }}>
          <figure style={{ margin: 0 }}>
            <div
              style={{
                overflow: "hidden",
                border: "1px solid var(--color-border-strong)",
                borderRadius: "16px",
                background: "#080b12",
                boxShadow: "0 24px 80px rgba(0,0,0,0.25)",
              }}
            >
              <Image
                src="/ecosystem-map.svg"
                alt="Architecture diagram: the website publishes whitepapers and hosts the local-first wallet; the MCP Hub is a separate in-memory server for agent clients. The diagram distinguishes three public currency rails from five MCP planes."
                width={1800}
                height={1440}
                priority
                style={{ display: "block", width: "100%", height: "auto" }}
              />
            </div>
            <figcaption
              style={{
                marginTop: "14px",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: "16px",
                flexWrap: "wrap",
                color: "var(--color-muted)",
                fontSize: "13px",
              }}
            >
              <span>
                A repo-level overview. Whitepapers are design/specification documents; the wallet
                rails are test pilots; MCP is a simulation.
              </span>
              <CallToAction label="Open full-size SVG ↗" href="/ecosystem-map.svg" variant="ghost" />
            </figcaption>
          </figure>
        </div>
      </section>

      <section style={{ padding: "32px 0 96px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
          <p
            style={{
              color: "var(--color-accent)",
              fontFamily: "ui-monospace, monospace",
              fontSize: "11px",
              fontWeight: 600,
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              margin: "0 0 14px",
            }}
          >
            Explore a layer
          </p>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: "16px",
            }}
          >
            {LINKS.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="hover-raise"
                style={{
                  display: "block",
                  padding: "22px",
                  border: "1px solid var(--color-border-strong)",
                  borderRadius: "14px",
                  background: "rgba(255,255,255,0.015)",
                  textDecoration: "none",
                }}
              >
                <span
                  style={{
                    display: "block",
                    color: item.color,
                    fontFamily: "ui-monospace, monospace",
                    fontSize: "11px",
                    fontWeight: 600,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    marginBottom: "12px",
                  }}
                >
                  {item.eyebrow}
                </span>
                <strong
                  style={{
                    display: "block",
                    color: "var(--color-foreground)",
                    fontFamily: "ui-serif, Georgia, serif",
                    fontSize: "21px",
                    fontWeight: 500,
                    marginBottom: "8px",
                  }}
                >
                  {item.title} →
                </strong>
                <span style={{ color: "var(--color-muted-strong)", fontSize: "13px", lineHeight: 1.6 }}>
                  {item.description}
                </span>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
