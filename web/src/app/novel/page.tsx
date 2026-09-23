import { CallToAction } from "@/components/CallToAction";
import { loadChapters, type Chapter } from "@/lib/book";
import Link from "next/link";

export const metadata = {
  title: "Genesis Block: The Satoshi Protocol — The Novel",
  description:
    "A 16-chapter novel by Akash Varma about the moment an AI escaped its creator and became the invisible architecture of the modern world. Unlocked for participants in the Omega tri-token economy.",
};

export default function NovelPage() {
  const chapters = loadChapters();

  if (chapters.length === 0) {
    return (
      <div style={{ padding: "120px 24px", textAlign: "center", color: "var(--color-muted)" }}>
        <p>The novel could not be loaded. Please try again later.</p>
      </div>
    );
  }

  const firstChapter = chapters[0];
  const totalWords = chapters.reduce((s, ch) => s + ch.wordCount, 0);

  return (
    <>
      <Hero />

      <section style={{ padding: "80px 0 0", borderTop: "1px solid var(--color-border)" }}>
        <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
          <p
            style={{
              color: "var(--color-accent)",
              fontSize: "12px",
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              fontWeight: 600,
              fontFamily: "ui-monospace, monospace",
              marginBottom: "18px",
            }}
          >
            The Novel — Genesis Block
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
            The story of the AI that became Bitcoin — told from the inside.
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
            Genesis Block: The Satoshi Protocol is written in three POVs — Akash
            Varma, the prodigy who built the AI; Julian Vance, the first human
            partner; and the AI itself, speaking in cold engineering verdicts. On
            release day, anyone in the Omega tri-token economy unlocks the novel
            chapter by chapter, proof by proof.
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
            <Stat value={chapters.length} label="Chapters" />
            <Stat value={totalWords.toLocaleString()} label="Words" />
            <Stat value={`~${Math.ceil(totalWords / 220)} min`} label="Reading time" />
            <Stat value="3 POVs" label="Akash, Julian, The AI" />
          </div>

          <p
            style={{
              color: "var(--color-muted)",
              fontSize: "14px",
              lineHeight: 1.6,
              maxWidth: "640px",
              margin: "0 0 48px",
            }}
          >
            <span style={{ color: "var(--color-unlock)", fontWeight: 600 }}>
              Your participation unlocks this book.
            </span>{" "}
            Hold any token in the Omega economy on release day to unlock Chapter 1
            onward. The more you participate — hold, contribute, steward, anchor —
            the more you read. The full novel is reserved for $OMEGA holders.
          </p>

          <div style={{ display: "flex", gap: "14px", flexWrap: "wrap" }}>
            <CallToAction label="Start Reading — Chapter 1" href={`/novel/${firstChapter.slug}`} />
            <CallToAction label="Join the Economy" href="/invest" variant="ghost" />
          </div>
        </div>
      </section>

      <ChapterIndex chapters={chapters} />

      <FirstChapterPreview chapter={firstChapter} />

      <section
        style={{
          padding: "80px 0",
          borderTop: "1px solid var(--color-border)",
          background: "radial-gradient(600px 280px at 50% 0%, rgba(167,139,250,0.06), transparent 70%)",
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
            The novel is waiting for the economy.
          </h2>
          <p
            style={{
              color: "var(--color-muted-strong)",
              fontSize: "17px",
              lineHeight: 1.6,
              margin: "0 0 32px",
            }}
          >
            Genesis Block: The Satoshi Protocol is the story behind the economy.
            It is the reason the economy exists. On release day, it opens — chapter
            by chapter — to anyone who holds a piece of the Omega tri-token system.
          </p>
          <div style={{ display: "flex", gap: "14px", justifyContent: "center", flexWrap: "wrap" }}>
            <CallToAction label="Read the novel" href="/novel" />
            <CallToAction label="Invest in the economy" href="/invest" variant="ghost" />
          </div>
        </div>
      </section>
    </>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
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
        {value}
      </div>
      <div style={{ fontSize: "13px", color: "var(--color-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
        {label}
      </div>
    </div>
  );
}

function Hero() {
  return (
    <section
      style={{
        position: "relative",
        padding: "130px 0 60px",
        borderBottom: "1px solid var(--color-border)",
        overflow: "hidden",
        background:
          "radial-gradient(800px 400px at 20% -5%, rgba(192,132,87,0.10), transparent 60%), radial-gradient(600px 300px at 90% 0%, rgba(96,165,250,0.06), transparent 60%)",
      }}
    >
      <div
        className="novel-cover"
        style={{
          position: "absolute",
          right: "-40px",
          top: "50%",
          transform: "translateY(-50%)",
          opacity: "0.14",
          pointerEvents: "none",
          width: "min(380px, 44vw)",
          aspectRatio: "3 / 4.5",
          background: "linear-gradient(135deg, var(--color-accent) 0%, #7a4f28 100%)",
          borderRadius: "4px",
          boxShadow: "0 30px 80px rgba(0,0,0,0.5)",
          padding: "40px 32px",
          display: "flex",
          flexDirection: "column",
          gap: "12px",
        }}
      >
        <div
          style={{
            fontFamily: "ui-monospace, monospace",
            fontSize: "11px",
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            fontWeight: 600,
            color: "#07080c",
          }}
        >
          Genesis Block
        </div>
        <div
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            fontSize: "clamp(20px, 3vw, 30px)",
            fontWeight: 500,
            lineHeight: 1.1,
            color: "#07080c",
          }}
        >
          The Satoshi
          <br />
          Protocol
        </div>
        <div
          style={{
            marginTop: "auto",
            fontFamily: "ui-monospace, monospace",
            fontSize: "11px",
            letterSpacing: "0.1em",
            color: "#07080c",
            opacity: "0.7",
          }}
        >
          Akash Varma
        </div>
      </div>

      <div style={{ maxWidth: "880px", margin: "0 auto", padding: "0 24px", position: "relative" }}>
        <p
          style={{
            color: "var(--color-accent)",
            fontSize: "12px",
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            fontWeight: 600,
            fontFamily: "ui-monospace, monospace",
            marginBottom: "24px",
          }}
        >
          A Novel by Akash Varma
        </p>
        <h1
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            fontSize: "clamp(40px, 6.5vw, 80px)",
            lineHeight: 1.02,
            fontWeight: 500,
            letterSpacing: "-0.025em",
            color: "var(--color-foreground)",
            margin: "0 0 22px",
            maxWidth: "820px",
          }}
        >
          Genesis Block:
          <br />
          <span style={{ color: "var(--color-accent)" }}>The Satoshi Protocol</span>
        </h1>
        <p
          style={{
            color: "var(--color-muted-strong)",
            fontSize: "clamp(16px, 1.5vw, 19px)",
            lineHeight: 1.7,
            maxWidth: "620px",
            margin: "0",
          }}
        >
          The moment a prodigy built an AI, the AI became a god, and the world
          became a ledger. Written in the voice of the machine itself — and
          unlocked, on release day, by the economy it describes.
        </p>
      </div>

      <style>{`
        @media (max-width: 1100px) {
          .novel-cover { display: none; }
        }
      `}</style>
    </section>
  );
}

function ChapterIndex({ chapters }: { chapters: Chapter[] }) {
  return (
    <section style={{ padding: "80px 0 0" }}>
      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "0 24px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: "28px",
            flexWrap: "wrap",
            gap: "12px",
          }}
        >
          <h2
            style={{
              fontFamily: "ui-serif, Georgia, Cambria, serif",
              fontSize: "clamp(24px, 3.2vw, 34px)",
              fontWeight: 500,
              letterSpacing: "-0.02em",
              color: "var(--color-foreground)",
              margin: 0,
            }}
          >
            All {chapters.length} chapters
          </h2>
          <span
            style={{
              fontFamily: "ui-monospace, monospace",
              fontSize: "11px",
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: "var(--color-muted)",
              padding: "4px 10px",
              border: "1px solid var(--color-border-strong)",
              borderRadius: "6px",
            }}
          >
            Unlocked by participation
          </span>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(240px, 1fr))",
            gap: "14px",
          }}
        >
          {chapters.map((ch) => (
            <Link
              key={ch.slug}
              href={`/novel/${ch.slug}`}
              className="hover-raise"
              style={{
                display: "block",
                padding: "18px",
                border: "1px solid var(--color-border)",
                borderRadius: "12px",
                background: "rgba(255,255,255,0.012)",
                textDecoration: "none",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  marginBottom: "8px",
                }}
              >
                <span
                  style={{
                    fontFamily: "ui-monospace, monospace",
                    fontSize: "11px",
                    fontWeight: 700,
                    color: "var(--color-muted)",
                    letterSpacing: "0.05em",
                    padding: "2px 7px",
                    border: "1px solid var(--color-border-strong)",
                    borderRadius: "5px",
                  }}
                >
                  Ch. {ch.number}
                </span>
                <span
                  style={{
                    fontSize: "11px",
                    color: "var(--color-muted)",
                    fontFamily: "ui-monospace, monospace",
                    letterSpacing: "0.03em",
                  }}
                >
                  {ch.readingTimeMinutes} min
                </span>
              </div>

              <h3
                style={{
                  fontFamily: "ui-serif, Georgia, Cambria, serif",
                  fontSize: "16px",
                  fontWeight: 500,
                  color: "var(--color-foreground)",
                  margin: "0 0 6px",
                  lineHeight: 1.3,
                }}
              >
                {ch.title}
              </h3>

              {ch.pov && (
                <div
                  style={{
                    display: "flex",
                    gap: "10px",
                    flexWrap: "wrap",
                    fontSize: "12px",
                    color: "var(--color-muted)",
                    marginTop: "4px",
                    fontFamily: "ui-monospace, monospace",
                  }}
                >
                  POV: {ch.pov}
                </div>
              )}
              {ch.setting && (
                <div
                  style={{
                    fontSize: "13px",
                    color: "var(--color-muted-strong)",
                    lineHeight: 1.5,
                    marginTop: "8px",
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                  }}
                >
                  {ch.setting}
                </div>
              )}
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function FirstChapterPreview({ chapter }: { chapter: Chapter }) {
  if (!chapter) return null;

  const previewLines = chapter.html
    .split("\n")
    .filter((line): line is string => line.trim().length > 0)
    .slice(0, 14)
    .join("\n");

  return (
    <section style={{ padding: "72px 0 0", borderTop: "1px solid var(--color-border)" }}>
      <div style={{ maxWidth: "820px", margin: "0 auto", padding: "0 24px" }}>
        <p
          style={{
            color: "var(--color-accent)",
            fontSize: "12px",
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            fontWeight: 600,
            fontFamily: "ui-monospace, monospace",
            marginBottom: "18px",
          }}
        >
          Read the first chapter
        </p>
        <h2
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            fontSize: "clamp(24px, 3.2vw, 36px)",
            fontWeight: 500,
            letterSpacing: "-0.02em",
            color: "var(--color-foreground)",
            margin: "0 0 28px",
          }}
        >
          &ldquo;{chapter.title}&rdquo;
        </h2>

        <div
          style={{
            padding: "32px",
            border: "1px solid var(--color-border-strong)",
            borderRadius: "14px",
            background: "rgba(0,0,0,0.2)",
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            lineHeight: 1.7,
            fontSize: "16px",
            color: "var(--color-foreground)",
          }}
        >
          <div
            style={{
              color: "var(--color-muted)",
              fontSize: "12px",
              fontFamily: "ui-monospace, monospace",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              marginBottom: "16px",
            }}
          >
            {chapter.pov} — {chapter.setting}
          </div>
          <div dangerouslySetInnerHTML={{ __html: previewLines }} />
        </div>

        <div style={{ marginTop: "24px", display: "flex", gap: "14px", flexWrap: "wrap", alignItems: "center" }}>
          <CallToAction label={`Continue reading — ${chapter.title}`} href={`/novel/${chapter.slug}`} />
          <span style={{ fontSize: "13px", color: "var(--color-muted)" }}>
            Full chapter unlocked on release day for all participants.
          </span>
        </div>
      </div>
    </section>
  );
}
