// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import type { Metadata } from "next";
import Link from "next/link";

import { getDoc, listDocs } from "@/lib/docs";

export const metadata: Metadata = {
  title: "Whitepapers & Manifesto",
  description:
    "The documents of the C.A.R.E. Economy: the manifesto, the extensive whitepaper, and every subsystem specification — readable on-site.",
};

export default function WhitepapersPage() {
  const docs = listDocs();
  const manifesto = docs[0];
  const flagship = docs[1];
  const rest = docs.slice(2);

  const manifestoDoc = getDoc(manifesto.slug);
  const flagshipDoc = getDoc(flagship.slug);

  return (
    <>
      <header
        style={{
          padding: "140px 0 56px",
          borderBottom: "1px solid var(--color-border)",
          background:
            "radial-gradient(700px 340px at 30% -5%, rgba(192,132,87,0.10), transparent 60%), radial-gradient(600px 300px at 85% 0%, rgba(167,139,250,0.07), transparent 60%)",
        }}
      >
        <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
          <p
            style={{
              color: "var(--color-accent)",
              fontSize: "12px",
              letterSpacing: "0.2em",
              textTransform: "uppercase",
              fontWeight: 600,
              fontFamily: "ui-monospace, monospace",
              marginBottom: "18px",
            }}
          >
            The C.A.R.E. Economy · Documents
          </p>
          <h1
            style={{
              fontFamily: "ui-serif, Georgia, Cambria, serif",
              fontSize: "clamp(36px, 5.5vw, 68px)",
              lineHeight: 1.04,
              fontWeight: 500,
              letterSpacing: "-0.025em",
              color: "var(--color-foreground)",
              margin: "0 0 20px",
              maxWidth: "880px",
            }}
          >
            The manifesto,
            <br />
            the whitepapers,{" "}
            <span style={{ color: "var(--color-accent)" }}>the whole design.</span>
          </h1>
          <p
            style={{
              color: "var(--color-muted-strong)",
              fontSize: "clamp(15px, 1.5vw, 18px)",
              lineHeight: 1.65,
              maxWidth: "660px",
              margin: 0,
            }}
          >
            Every document behind the economy, readable right here — the
            manifesto that states the vision, the extensive whitepaper that
            specifies the entire economy, and the subsystem papers underneath.
            Research drafts and system specifications; not offers of
            securities, promises of liquidity, or validated results.
          </p>
        </div>
      </header>

      <section style={{ padding: "64px 0 40px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: "18px",
            }}
          >
            <FeatureCard
              meta={manifesto}
              readingTime={manifestoDoc?.readingTimeMinutes}
              eyebrow="Start here · Manifesto"
              title={manifesto.title}
              summary={manifesto.summary}
            />
            <FeatureCard
              meta={flagship}
              readingTime={flagshipDoc?.readingTimeMinutes}
              eyebrow="The full specification"
              title={flagship.title}
              summary={flagship.summary}
            />
          </div>
        </div>
      </section>

      <section style={{ padding: "24px 0 96px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
          <div
            style={{
              fontFamily: "ui-monospace, monospace",
              fontSize: "11px",
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: "var(--color-muted)",
              fontWeight: 600,
              margin: "24px 0 18px",
            }}
          >
            Subsystem specifications
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
              gap: "18px",
            }}
          >
            {rest.map((meta) => {
              const doc = getDoc(meta.slug);
              return (
                <Link
                  key={meta.slug}
                  href={`/whitepapers/${meta.slug}`}
                  className="hover-raise"
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    padding: "26px 24px",
                    border: "1px solid var(--color-border-strong)",
                    borderRadius: "14px",
                    background: "rgba(255,255,255,0.015)",
                    textDecoration: "none",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
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
                        background: meta.accent,
                        boxShadow: `0 0 12px ${meta.accent}88`,
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
                      {meta.kind}
                    </span>
                    {doc ? (
                      <span
                        style={{
                          marginLeft: "auto",
                          fontFamily: "ui-monospace, monospace",
                          fontSize: "11px",
                          color: "var(--color-muted)",
                        }}
                      >
                        {doc.readingTimeMinutes} min
                      </span>
                    ) : null}
                  </div>
                  <h3
                    style={{
                      fontFamily: "ui-serif, Georgia, serif",
                      fontSize: "20px",
                      fontWeight: 500,
                      color: "var(--color-foreground)",
                      margin: "0 0 10px",
                      letterSpacing: "-0.01em",
                      lineHeight: 1.25,
                    }}
                  >
                    {meta.title}
                  </h3>
                  <p
                    style={{
                      color: "var(--color-muted-strong)",
                      fontSize: "14px",
                      lineHeight: 1.6,
                      margin: "0 0 14px",
                      flex: 1,
                    }}
                  >
                    {meta.summary}
                  </p>
                  <span style={{ color: meta.accent, fontSize: "13px", fontWeight: 600 }}>
                    Read on-site →
                  </span>
                </Link>
              );
            })}
          </div>

          <div
            style={{
              marginTop: "40px",
              padding: "20px 24px",
              border: "1px dashed var(--color-border-strong)",
              borderRadius: "12px",
              color: "var(--color-muted)",
              fontSize: "13px",
              lineHeight: 1.6,
            }}
          >
            The whitepapers are also in the repository under{" "}
            <code className="doc-inline-code">whitepapers/</code> — alongside
            the science materials (MIT simulations, Lean proofs, and the
            54-volume Omega research program). Product specifications in{" "}
            <code className="doc-inline-code">whitepapers/</code> are
            read-only; see{" "}
            <a
              className="doc-link"
              href="https://github.com/jakesdev1991/Omega_Theory_Everything/blob/main/docs/LICENSING.md"
              target="_blank"
              rel="noopener noreferrer"
            >
              docs/LICENSING.md
              <span className="doc-link-ext">↗</span>
            </a>
            .
          </div>
        </div>
      </section>
    </>
  );
}

function FeatureCard({
  meta,
  readingTime,
  eyebrow,
  title,
  summary,
}: {
  meta: { slug: string; accent: string; kind: string };
  readingTime?: number;
  eyebrow: string;
  title: string;
  summary: string;
}) {
  return (
    <Link
      href={`/whitepapers/${meta.slug}`}
      className="hover-raise"
      style={{
        display: "flex",
        flexDirection: "column",
        padding: "30px 28px",
        border: "1px solid var(--color-border-strong)",
        borderRadius: "16px",
        background:
          "radial-gradient(420px 200px at 15% 0%, rgba(192,132,87,0.06), transparent 65%), rgba(255,255,255,0.015)",
        textDecoration: "none",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginBottom: "16px",
        }}
      >
        <div
          style={{
            width: "9px",
            height: "9px",
            borderRadius: "50%",
            background: meta.accent,
            boxShadow: `0 0 14px ${meta.accent}88`,
          }}
        />
        <span
          style={{
            fontFamily: "ui-monospace, monospace",
            fontSize: "11px",
            letterSpacing: "0.12em",
            textTransform: "uppercase",
            color: meta.accent,
            fontWeight: 600,
          }}
        >
          {eyebrow}
        </span>
        {readingTime ? (
          <span
            style={{
              marginLeft: "auto",
              fontFamily: "ui-monospace, monospace",
              fontSize: "11px",
              color: "var(--color-muted)",
            }}
          >
            {readingTime} min
          </span>
        ) : null}
      </div>
      <h2
        style={{
          fontFamily: "ui-serif, Georgia, serif",
          fontSize: "clamp(22px, 2.4vw, 28px)",
          fontWeight: 500,
          color: "var(--color-foreground)",
          margin: "0 0 12px",
          letterSpacing: "-0.015em",
          lineHeight: 1.2,
        }}
      >
        {title}
      </h2>
      <p
        style={{
          color: "var(--color-muted-strong)",
          fontSize: "15px",
          lineHeight: 1.65,
          margin: "0 0 18px",
          flex: 1,
        }}
      >
        {summary}
      </p>
      <span style={{ color: meta.accent, fontSize: "14px", fontWeight: 600 }}>
        Read on-site →
      </span>
    </Link>
  );
}
