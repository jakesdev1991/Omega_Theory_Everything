import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";

import { DocReader, DocToc } from "@/components/DocReader";
import { adjacentDocs, DOC_REGISTRY, docExists, getDoc } from "@/lib/docs";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export function generateStaticParams(): { slug: string }[] {
  return DOC_REGISTRY.map((doc) => ({ slug: doc.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const doc = getDoc(slug);
  if (!doc) {
    return { title: "Document not found" };
  }
  return {
    title: doc.meta.shortTitle,
    description: doc.meta.summary,
  };
}

export default async function WhitepaperPage({ params }: PageProps) {
  const { slug } = await params;
  if (!docExists(slug)) {
    notFound();
  }
  const doc = getDoc(slug);
  if (!doc) {
    notFound();
  }
  const { prev, next } = adjacentDocs(slug);

  return (
    <>
      <header
        style={{
          padding: "132px 0 40px",
          borderBottom: "1px solid var(--color-border)",
          background:
            "radial-gradient(700px 320px at 25% -5%, rgba(192,132,87,0.09), transparent 60%)",
        }}
      >
        <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
              marginBottom: "18px",
              flexWrap: "wrap",
            }}
          >
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                fontFamily: "ui-monospace, monospace",
                fontSize: "11px",
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                color: doc.meta.accent,
                fontWeight: 600,
              }}
            >
              <span
                style={{
                  width: "8px",
                  height: "8px",
                  borderRadius: "50%",
                  background: doc.meta.accent,
                  boxShadow: `0 0 12px ${doc.meta.accent}88`,
                  display: "inline-block",
                }}
              />
              {doc.meta.kind}
            </span>
            <span style={{ color: "var(--color-muted)", fontSize: "13px" }}>·</span>
            <span
              style={{
                fontFamily: "ui-monospace, monospace",
                fontSize: "12px",
                color: "var(--color-muted)",
                letterSpacing: "0.06em",
              }}
            >
              {doc.readingTimeMinutes} min read · {doc.wordCount.toLocaleString()} words
            </span>
          </div>

          <h1
            style={{
              fontFamily: "ui-serif, Georgia, Cambria, serif",
              fontSize: "clamp(30px, 4.5vw, 54px)",
              lineHeight: 1.08,
              fontWeight: 500,
              letterSpacing: "-0.02em",
              color: "var(--color-foreground)",
              margin: "0 0 16px",
              maxWidth: "880px",
            }}
          >
            {doc.meta.shortTitle}
          </h1>
          <p
            style={{
              color: "var(--color-muted-strong)",
              fontSize: "clamp(15px, 1.5vw, 17px)",
              lineHeight: 1.65,
              maxWidth: "680px",
              margin: "0 0 16px",
            }}
          >
            {doc.meta.summary}
          </p>
          <Link
            href="/whitepapers"
            className="doc-link"
            style={{ fontSize: "14px", fontFamily: "ui-sans-serif, system-ui, sans-serif" }}
          >
            ← All documents
          </Link>
        </div>
      </header>

      <section style={{ padding: "48px 0 96px" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
          <div className="doc-layout">
            <aside className="doc-aside">
              <DocToc headings={doc.toc} />
            </aside>
            <article style={{ maxWidth: "760px" }}>
              <DocReader blocks={doc.blocks} />
              <div
                style={{
                  marginTop: "56px",
                  display: "flex",
                  justifyContent: "space-between",
                  gap: "16px",
                  flexWrap: "wrap",
                }}
              >
                <PrevNext
                  slug={prev?.slug}
                  label={prev?.shortTitle}
                  next={false}
                />
                <PrevNext
                  slug={next?.slug}
                  label={next?.shortTitle}
                  next
                />
              </div>
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

function PrevNext({
  slug,
  label,
  next,
}: {
  slug?: string;
  label?: string;
  next: boolean;
}) {
  if (!slug || !label) return null;
  return (
    <Link
      href={`/whitepapers/${slug}`}
      className={next ? "nav-next-solid" : "nav-prev-next"}
      style={{
        display: "inline-flex",
        flexDirection: "column",
        gap: "4px",
        padding: "16px 22px",
        border: "1px solid var(--color-border-strong)",
        borderRadius: "12px",
        textDecoration: "none",
        background: next
          ? "linear-gradient(135deg, var(--color-accent), #a86a36)"
          : "rgba(255,255,255,0.015)",
        color: next ? "#07080c" : "var(--color-foreground)",
        maxWidth: "340px",
        flex: "0 1 auto",
      }}
    >
      <span
        style={{
          fontFamily: "ui-monospace, monospace",
          fontSize: "10px",
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          opacity: 0.75,
        }}
      >
        {next ? "Next document" : "Previous document"}
      </span>
      <span style={{ fontWeight: 600, fontSize: "15px" }}>
        {next ? `${label} →` : `← ${label}`}
      </span>
    </Link>
  );
}
