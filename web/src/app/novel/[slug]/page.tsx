import Link from "next/link";
import { loadChapters } from "@/lib/book";
import { GatedChapter } from "@/components/GatedChapter";
import { UnlockRailBanner } from "@/components/UnlockRailBanner";

export async function generateStaticParams() {
  const chapters = loadChapters();
  return chapters.map((ch) => ({ slug: ch.slug }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const chapters = loadChapters();
  const chapter = chapters.find((ch) => ch.slug === slug);
  if (!chapter) return {};
  return {
    title: `${chapter.number}. ${chapter.title} — Crucible`,
    description: chapter.setting,
  };
}

export default async function ChapterReaderPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const chapters = loadChapters();
  const chapter = chapters.find((ch) => ch.slug === slug);

  if (!chapter) {
    return (
      <div style={{ padding: "140px 24px", textAlign: "center", color: "var(--color-muted)" }}>
        <p>Chapter not found.</p>
        <Link href="/novel" className="link-soft" style={{ color: "var(--color-accent)" }}>
          Back to the novel
        </Link>
      </div>
    );
  }

  const prevChapter = chapters.find((ch) => ch.number === chapter.number - 1);
  const nextChapter = chapters.find((ch) => ch.number === chapter.number + 1);

  return (
    <>
      <header
        style={{
          paddingTop: "120px",
          paddingBottom: "32px",
          borderBottom: "1px solid var(--color-border)",
          background:
            "radial-gradient(700px 300px at 80% 0%, rgba(192,132,87,0.06), transparent 70%)",
        }}
      >
        <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
          <p
            style={{
              color: "var(--color-accent)",
              fontSize: "12px",
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              fontWeight: 600,
              fontFamily: "ui-monospace, monospace",
              marginBottom: "20px",
            }}
          >
            Crucible: The Satoshi Protocol — Chapter {chapter.number}
          </p>
        </div>
      </header>

      <article style={{ maxWidth: "760px", margin: "0 auto", padding: "48px 24px 80px" }}>
        <header
          style={{
            marginBottom: "48px",
            paddingBottom: "24px",
            borderBottom: "1px solid var(--color-border)",
          }}
        >
          <Link
            href="/novel"
            className="link-soft"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              fontSize: "13px",
              color: "var(--color-muted-strong)",
              textDecoration: "none",
              marginBottom: "24px",
            }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m15 18-6-6 6-6" />
            </svg>
            All chapters
          </Link>

          <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "14px" }}>
            <span
              style={{
                fontFamily: "ui-monospace, monospace",
                fontSize: "12px",
                fontWeight: 700,
                letterSpacing: "0.05em",
                color: "var(--color-accent)",
                background: "rgba(192,132,87,0.12)",
                padding: "4px 10px",
                borderRadius: "6px",
              }}
            >
              Chapter {chapter.number}
            </span>
            <span
              style={{
                fontFamily: "ui-monospace, monospace",
                fontSize: "11px",
                color: "var(--color-muted)",
                letterSpacing: "0.04em",
              }}
            >
              {chapter.readingTimeMinutes} min read
            </span>
          </div>

          <h1
            style={{
              fontFamily: "ui-serif, Georgia, Cambria, serif",
              fontSize: "clamp(28px, 4vw, 44px)",
              fontWeight: 500,
              letterSpacing: "-0.02em",
              color: "var(--color-foreground)",
              margin: "0 0 12px",
              lineHeight: 1.1,
            }}
          >
            {chapter.title}
          </h1>

          <p style={{ color: "var(--color-muted-strong)", fontSize: "15px", lineHeight: 1.5, margin: 0 }}>
            {chapter.pov} — {chapter.setting}
          </p>
        </header>

        <UnlockRailBanner />

        {/* The actual gate: prose loads only after server-side signature verification */}
        <GatedChapter slug={chapter.slug} number={chapter.number} />

        <nav
          style={{
            marginTop: "40px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            borderTop: "1px solid var(--color-border)",
            paddingTop: "24px",
            gap: "16px",
            flexWrap: "wrap",
          }}
        >
          {prevChapter ? (
            <Link
              href={`/novel/${prevChapter.slug}`}
              className="nav-prev-next"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                padding: "10px 16px",
                border: "1px solid var(--color-border-strong)",
                borderRadius: "8px",
                background: "rgba(255,255,255,0.02)",
                color: "var(--color-foreground)",
                textDecoration: "none",
                fontSize: "14px",
                fontWeight: 500,
                maxWidth: "280px",
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m15 18-6-6 6-6" />
              </svg>
              Previous — {prevChapter.title}
            </Link>
          ) : (
            <div />
          )}

          <div style={{ fontSize: "13px", color: "var(--color-muted)", fontFamily: "ui-monospace, monospace" }}>
            Chapter {chapter.number} of {chapters.length}
          </div>

          {nextChapter ? (
            <Link
              href={`/novel/${nextChapter.slug}`}
              className="nav-next-solid"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                padding: "10px 16px",
                border: "1px solid var(--color-border-strong)",
                borderRadius: "8px",
                background: "linear-gradient(135deg, var(--color-accent), #a86a36)",
                color: "#07080c",
                textDecoration: "none",
                fontSize: "14px",
                fontWeight: 600,
                maxWidth: "280px",
              }}
            >
              Next — {nextChapter.title}
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="m9 18 6-6-6-6" />
              </svg>
            </Link>
          ) : (
            <div />
          )}
        </nav>
      </article>
    </>
  );
}
