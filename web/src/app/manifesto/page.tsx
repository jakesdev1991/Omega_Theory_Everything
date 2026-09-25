import type { Metadata } from "next";
import Link from "next/link";

import { DocReader, DocToc } from "@/components/DocReader";
import { getDoc } from "@/lib/docs";

export const metadata: Metadata = {
  title: "The Manifesto",
  description:
    "Manifesto for a Sovereign Agentic Economy: Call About Resuscitating Everyone, participant sovereignty, agentic stewardship, and the invitation.",
};

export default function ManifestoPage() {
  const doc = getDoc("sovereign_economy_manifesto");

  if (!doc) {
    return (
      <section style={{ padding: "160px 24px 120px", maxWidth: "720px", margin: "0 auto" }}>
        <h1>The Manifesto</h1>
        <p style={{ color: "var(--color-muted-strong)" }}>
          The manifesto document could not be loaded. Run{" "}
          <code>npm run sync:docs</code> in{" "}
          <code>web/</code> and reload.
        </p>
      </section>
    );
  }

  return (
    <>
      <header
        style={{
          padding: "140px 0 48px",
          borderBottom: "1px solid var(--color-border)",
          background:
            "radial-gradient(700px 340px at 30% -5%, rgba(244,114,182,0.10), transparent 60%), radial-gradient(600px 300px at 85% 0%, rgba(192,132,87,0.08), transparent 60%)",
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
              marginBottom: "18px",
            }}
          >
            The C.A.R.E. Economy · Manifesto
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
            Call About
            <br />
            <span style={{ color: "var(--color-accent)" }}>
              Resuscitating Everyone.
            </span>
          </h1>
          <p
            style={{
              color: "var(--color-muted-strong)",
              fontSize: "clamp(15px, 1.5vw, 18px)",
              lineHeight: 1.65,
              maxWidth: "640px",
              margin: 0,
            }}
          >
            {doc.subtitle ??
              "A statement of vision and intent for the entire economy — read it in ten minutes, argue with it for years."}
          </p>
          <div
            style={{
              marginTop: "28px",
              display: "flex",
              gap: "20px",
              flexWrap: "wrap",
              fontFamily: "ui-monospace, monospace",
              fontSize: "12px",
              color: "var(--color-muted)",
              letterSpacing: "0.06em",
            }}
          >
            <span>{doc.readingTimeMinutes} min read</span>
            <span>·</span>
            <span>{doc.wordCount.toLocaleString()} words</span>
            <span>·</span>
            <Link href="/whitepapers/care_economy_whitepaper" className="doc-link">
              The mechanics live in the extensive whitepaper →
            </Link>
          </div>
        </div>
      </header>

      <section style={{ padding: "56px 0 96px" }}>
        <div
          style={{
            maxWidth: "1100px",
            margin: "0 auto",
            padding: "0 24px",
            display: "grid",
            gridTemplateColumns: "minmax(0, 1fr)",
            gap: "40px",
          }}
        >
          <div className="doc-layout">
            <aside className="doc-aside">
              <DocToc headings={doc.toc} />
            </aside>
            <article style={{ maxWidth: "720px" }}>
              <DocReader blocks={doc.blocks} />
              <Afterward />
            </article>
          </div>
        </div>
      </section>

      <style>{`
        .doc-layout {
          display: grid;
          grid-template-columns: 260px minmax(0, 1fr);
          gap: 48px;
          width: 100%;
        }
        @media (max-width: 1023px) {
          .doc-layout { grid-template-columns: minmax(0, 1fr); }
          .doc-aside { display: none; }
        }
      `}</style>
    </>
  );
}

function Afterward() {
  return (
    <div
      style={{
        marginTop: "64px",
        padding: "28px",
        border: "1px solid var(--color-border-strong)",
        borderRadius: "16px",
        background:
          "radial-gradient(500px 240px at 20% 0%, rgba(244,114,182,0.08), transparent 60%), rgba(255,255,255,0.015)",
      }}
    >
      <div
        style={{
          fontFamily: "ui-monospace, monospace",
          fontSize: "11px",
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--color-care)",
          fontWeight: 600,
          marginBottom: "12px",
        }}
      >
        Keep going
      </div>
      <h3
        style={{
          fontFamily: "ui-serif, Georgia, serif",
          fontSize: "24px",
          fontWeight: 500,
          margin: "0 0 10px",
          color: "var(--color-foreground)",
        }}
      >
        The manifesto is the why. The whitepaper is the how.
      </h3>
      <p
        style={{
          color: "var(--color-muted-strong)",
          fontSize: "15px",
          lineHeight: 1.65,
          margin: "0 0 20px",
        }}
      >
        The extensive whitepaper specifies every mechanism behind these
        promises — planes, rails, proofs, privacy, audits, governance — with
        the failure modes attached and an honest table of what exists today.
      </p>
      <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
        <Link
          href="/whitepapers/care_economy_whitepaper"
          className="nav-next-solid"
          style={{
            display: "inline-flex",
            padding: "11px 20px",
            borderRadius: "9px",
            background: "linear-gradient(135deg, var(--color-accent), #a86a36)",
            color: "#07080c",
            fontWeight: 600,
            fontSize: "14px",
            textDecoration: "none",
          }}
        >
          Read the extensive whitepaper
        </Link>
        <Link
          href="/whitepapers"
          className="btn-ghost"
          style={{
            display: "inline-flex",
            padding: "11px 20px",
            borderRadius: "9px",
            border: "1px solid var(--color-border-strong)",
            color: "var(--color-foreground)",
            fontWeight: 500,
            fontSize: "14px",
            textDecoration: "none",
          }}
        >
          All documents
        </Link>
      </div>
    </div>
  );
}
