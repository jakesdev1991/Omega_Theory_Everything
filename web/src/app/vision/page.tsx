"use client";

import Link from "next/link";

import { Section } from "@/components/Section";
import { CallToAction } from "@/components/CallToAction";

export default function VisionPage() {
  return (
    <>
      <Hero />
      <NameSection />
      <DiagnosisSection />
      <PromisesSection />
      <ArcSection />
      <StatusSection />
    </>
  );
}

function Hero() {
  return (
    <section
      style={{
        position: "relative",
        padding: "150px 0 90px",
        overflow: "hidden",
        borderBottom: "1px solid var(--color-border)",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          background:
            "radial-gradient(900px 440px at 50% -10%, rgba(244,114,182,0.12), transparent 60%), radial-gradient(600px 400px at 80% 0%, rgba(192,132,87,0.08), transparent 60%)",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          maxWidth: "1100px",
          margin: "0 auto",
          padding: "0 24px",
          position: "relative",
        }}
      >
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
          The Vision · The C.A.R.E. Economy
        </p>

        <h1
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            fontSize: "clamp(38px, 6.5vw, 80px)",
            lineHeight: 1.03,
            fontWeight: 500,
            letterSpacing: "-0.025em",
            color: "var(--color-foreground)",
            margin: "0 0 24px",
            maxWidth: "980px",
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
            fontSize: "clamp(16px, 1.6vw, 20px)",
            lineHeight: 1.7,
            maxWidth: "700px",
            margin: "0 0 40px",
          }}
        >
          The entire economy is named the{" "}
          <strong style={{ color: "var(--color-foreground)", fontWeight: 600 }}>
            C.A.R.E. Economy
          </strong>{" "}
          because that is the call it answers. Not a slogan. An invitation to
          bring people back — from isolation, crisis, grief, and the systems
          that meet them at their worst and charge them for it — and it is
          for everyone.
        </p>

        <div
          style={{
            display: "flex",
            gap: "14px",
            flexWrap: "wrap",
            alignItems: "center",
          }}
        >
          <CallToAction label="Read the Manifesto" href="/manifesto" />
          <CallToAction
            label="Read the Extensive Whitepaper"
            href="/whitepapers/care_economy_whitepaper"
            variant="ghost"
          />
        </div>

        <div
          style={{
            marginTop: "56px",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "0",
            paddingTop: "32px",
            borderTop: "1px solid var(--color-border)",
          }}
        >
          {[
            { value: "C", label: "Call — the invitation" },
            { value: "A", label: "About — the purpose" },
            { value: "R", label: "Resuscitating — the work" },
            { value: "E", label: "Everyone — the scope" },
          ].map((item) => (
            <div
              key={item.value}
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: "12px",
                padding: "14px 18px 14px 0",
              }}
            >
              <span
                style={{
                  fontFamily: "ui-serif, Georgia, serif",
                  fontSize: "clamp(30px, 3.6vw, 44px)",
                  fontWeight: 500,
                  color: "var(--color-accent)",
                  lineHeight: 1,
                }}
              >
                {item.value}
              </span>
              <span
                style={{
                  fontSize: "13px",
                  color: "var(--color-muted-strong)",
                  lineHeight: 1.4,
                }}
              >
                {item.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function NameSection() {
  return (
    <Section
      eyebrow="The name is the mission"
      title="Four words. One call. The call is the design."
      subtitle="C.A.R.E. stands for Call About Resuscitating Everyone. Each word is load-bearing. Drop any of them and it becomes something else entirely. This is not a clinical system or an emergency service — when someone needs emergency medical care, call 911. What we build is the call that reaches people before that, and the economy that still has a place for them after."
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: "16px",
        }}
      >
        {[
          {
            letter: "C",
            name: "Call",
            color: "var(--color-care)",
            body: "An invitation, not a command. We reach out before people have to prove they deserve help. The first product is a social surface — rooms, circles, requests, a fast exit — because a call is how resuscitation starts.",
          },
          {
            letter: "A",
            name: "About",
            color: "var(--color-amity)",
            body: "The purpose is specific. This economy is not about speculation, engagement, or ranking human worth. It is about the work of bringing people back.",
          },
          {
            letter: "R",
            name: "Resuscitating",
            color: "var(--color-use)",
            body: "The verb. Restore. Revive. Make a path back from every mistake. Proof of Care, peer support, earn-while-you-learn, human Archangels over machines. Not triage. Not a leaderboard.",
          },
          {
            letter: "E",
            name: "Everyone",
            color: "var(--color-omega)",
            body: "The scope. No one left outside the call. No permanent moral exile. No judgment at the door. The person at 3 a.m. is a first-class participant.",
          },
        ].map((layer, i) => (
          <div
            key={layer.letter}
            style={{
              position: "relative",
              padding: "24px 22px",
              border: "1px solid var(--color-border)",
              borderRadius: "14px",
              background: "rgba(255,255,255,0.012)",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: "2px",
                background: layer.color,
                boxShadow: `0 0 18px ${layer.color}66`,
              }}
            />
            <div
              style={{
                display: "flex",
                alignItems: "baseline",
                gap: "12px",
                marginBottom: "12px",
              }}
            >
              <span
                style={{
                  fontFamily: "ui-serif, Georgia, serif",
                  fontSize: "40px",
                  fontWeight: 500,
                  color: layer.color,
                  lineHeight: 1,
                  textShadow: `0 0 24px ${layer.color}55`,
                }}
              >
                {layer.letter}
              </span>
              <span
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: "13px",
                  fontWeight: 600,
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  color: layer.color,
                }}
              >
                {layer.name}
              </span>
              <span
                style={{
                  marginLeft: "auto",
                  fontFamily: "ui-monospace, monospace",
                  fontSize: "11px",
                  color: "var(--color-muted)",
                }}
              >
                {i + 1}/4
              </span>
            </div>
            <p
              style={{
                margin: 0,
                color: "var(--color-muted-strong)",
                fontSize: "14px",
                lineHeight: 1.65,
              }}
            >
              {layer.body}
            </p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function DiagnosisSection() {
  return (
    <Section
      eyebrow="The diagnosis"
      title="The judgment economy meets people at their worst and charges them for it."
      subtitle="Every mechanism in the C.A.R.E. Economy answers a specific, observable failure pattern in the systems we already live inside."
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: "16px",
        }}
      >
        {[
          {
            fails: "Judgment at the door — your worst season becomes a permanent price tag.",
            answers:
              "Entry without judgment: pseudonymous participation, behavioral boundaries instead of identity labels, a path back from every mistake.",
          },
          {
            fails: "Surveillance as the price of belonging — visibility mandatory, privacy coded as suspicion.",
            answers:
              "Anonymity as a right: the L0–L4 visibility ladder, each level a separate revocable choice, and full participation at zero visibility.",
          },
          {
            fails: "Care invisible to every ledger that matters — unpaid, unrecorded, unvalued.",
            answers:
              "Proof of Care: evidence that care happened, consent-scoped and human-verified — protected from markets forever.",
          },
          {
            fails: "Extractive matching — crisis is the most profitable targeting signal.",
            answers:
              "Safeguarding by design: reflection and peer-support rooms, qualified resources, fast exit, and no engagement farming on pain.",
          },
          {
            fails: "Governance by hidden knob — rules rewritten silently, appeals impossible.",
            answers:
              "Rules on the record: auditable, replayable, versioned, change-controlled governance with timelocks and appeal routes.",
          },
          {
            fails: "Automation dropped into human systems — judgment, accelerated.",
            answers:
              "Humans govern the machines: agents flag and check, accountable humans finalize, formal proofs separate safety from the physics.",
          },
        ].map((row, i) => (
          <div
            key={i}
            style={{
              padding: "24px 22px",
              border: "1px solid var(--color-border)",
              borderRadius: "14px",
              background: "rgba(255,255,255,0.012)",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "8px",
                marginBottom: "12px",
              }}
            >
              <span
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: "10px",
                  letterSpacing: "0.14em",
                  textTransform: "uppercase",
                  color: "var(--color-muted)",
                  fontWeight: 600,
                }}
              >
                The old pattern
              </span>
              <span style={{ flex: 1, height: "1px", background: "var(--color-border)" }} />
              <span
                style={{
                  fontFamily: "ui-monospace, monospace",
                  fontSize: "10px",
                  letterSpacing: "0.14em",
                  textTransform: "uppercase",
                  color: "var(--color-unlock)",
                  fontWeight: 600,
                }}
              >
                The answer
              </span>
            </div>
            <p
              style={{
                margin: "0 0 12px",
                color: "var(--color-muted)",
                fontSize: "13.5px",
                lineHeight: 1.6,
                textDecoration: "line-through",
                textDecorationColor: "rgba(244,114,182,0.5)",
              }}
            >
              {row.fails}
            </p>
            <p
              style={{
                margin: 0,
                color: "var(--color-foreground)",
                fontSize: "14.5px",
                lineHeight: 1.65,
              }}
            >
              {row.answers}
            </p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function PromisesSection() {
  const promises = [
    {
      title: "Anonymity is a right",
      body: "Participation begins pseudonymous and can stay that way. Privacy is dignity, not suspicion.",
    },
    {
      title: "Transparency is a choice",
      body: "Disclosure is invited by earned trust and emotional security — never cornered out of someone.",
    },
    {
      title: "No judgment at the door",
      body: "Behavior, evidence, and risk are distinguishable from worth. There is always a path back.",
    },
    {
      title: "Earn while you learn",
      body: "Useful work, repair, care, and stewardship all count, from wherever you start.",
    },
    {
      title: "Humans govern the machines",
      body: "Agents flag, draft, and check. Accountable humans finalize — with reasons on the record and appeals attached.",
    },
    {
      title: "Care is never a commodity",
      body: "Care evidence never touches an exchange. The AMITY boundary is the most defended line in the design.",
    },
    {
      title: "The rules are on the record",
      body: "Every economic rule is auditable, replayable, versioned, and change-controlled.",
    },
    {
      title: "Honest about what isn't built",
      body: "What works, what's simulated, and what's a drawing is always labeled. Pilots stay valueless until the gates are passed.",
    },
  ];
  return (
    <Section
      eyebrow="The promises"
      title="Eight commitments the whole design exists to enforce."
      subtitle="These are constraints, not aspirations — every mechanism in the whitepaper makes one of them enforceable, and every one is a refusal of a real failure mode."
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: "14px",
        }}
      >
        {promises.map((p, i) => (
          <div
            key={p.title}
            style={{
              padding: "20px 20px",
              border: "1px solid var(--color-border)",
              borderRadius: "13px",
              background: "rgba(255,255,255,0.012)",
              display: "flex",
              gap: "14px",
              alignItems: "flex-start",
            }}
          >
            <span
              style={{
                fontFamily: "ui-monospace, monospace",
                fontSize: "12px",
                color: "var(--color-accent)",
                fontWeight: 700,
                paddingTop: "3px",
              }}
            >
              {String(i + 1).padStart(2, "0")}
            </span>
            <div>
              <h3
                style={{
                  fontFamily: "ui-serif, Georgia, serif",
                  fontSize: "17px",
                  fontWeight: 500,
                  color: "var(--color-foreground)",
                  margin: "0 0 6px",
                }}
              >
                {p.title}
              </h3>
              <p
                style={{
                  margin: 0,
                  color: "var(--color-muted-strong)",
                  fontSize: "13.5px",
                  lineHeight: 1.6,
                }}
              >
                {p.body}
              </p>
            </div>
          </div>
        ))}
      </div>
      <div style={{ marginTop: "36px" }}>
        <CallToAction label="Read the Manifesto" href="/manifesto" />
      </div>
    </Section>
  );
}

function ArcSection() {
  return (
    <Section
      eyebrow="How we got here"
      title="From a physics program to an economy, without losing the thread."
      subtitle="The same question runs through everything in this repository: what if the systems we build assumed care instead of judgment?"
    >
      <ol
        style={{
          listStyle: "none",
          margin: "0",
          padding: "0",
          display: "grid",
          gap: "0",
        }}
      >
        {[
          {
            step: "01",
            title: "The Omega research program",
            body: "A unified-physics program — spacetime from quantum information — across 54 formalized volumes, Lean 4 proofs, and simulations. Evidence discipline included: negative results are published, and the security proofs are kernel-checked separately from the physics.",
            color: "var(--color-omega)",
          },
          {
            step: "02",
            title: "The sovereign economy blueprint",
            body: "The research program becomes a systems specification: separate money, contribution, and governance powers; privacy by construction; auditable, replayable rules; reversible pilots with hard caps.",
            color: "var(--color-use)",
          },
          {
            step: "03",
            title: "The C.A.R.E. Economy",
            body: "The blueprint gets its name and its soul: Call About Resuscitating Everyone. Care as the base layer, five planes, three rails, Proof of Care and Proof of Useful Work, human Archangels over machine agents, and the manifesto that states the intent in plain language.",
            color: "var(--color-care)",
          },
          {
            step: "04",
            title: "The build, in the open",
            body: "A wallet where keys never leave your device. An MCP hub with 22 tools over a replayable ledger. Test consoles, a Nostr app backplane, an agent-governance bridge — all valueless pilots, all public, with the honest table attached.",
            color: "var(--color-unlock)",
          },
        ].map((item, i) => (
          <li
            key={item.step}
            style={{
              display: "grid",
              gridTemplateColumns: "110px 1fr",
              gap: "32px",
              padding: "30px 0",
              borderTop: i === 0 ? "1px solid var(--color-border)" : "none",
              alignItems: "start",
            }}
          >
            <div
              style={{
                fontFamily: "ui-serif, Georgia, serif",
                fontSize: "clamp(30px, 4.4vw, 44px)",
                fontWeight: 500,
                color: item.color,
                letterSpacing: "-0.02em",
                lineHeight: 1,
                textShadow: `0 0 22px ${item.color}44`,
              }}
            >
              {item.step}
            </div>
            <div>
              <h3
                style={{
                  fontFamily: "ui-serif, Georgia, serif",
                  fontSize: "20px",
                  fontWeight: 500,
                  margin: "0 0 8px",
                  color: "var(--color-foreground)",
                }}
              >
                {item.title}
              </h3>
              <p
                style={{
                  color: "var(--color-muted-strong)",
                  fontSize: "15px",
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
    </Section>
  );
}

function StatusSection() {
  return (
    <section
      style={{
        padding: "96px 0",
        borderTop: "1px solid var(--color-border)",
        background:
          "radial-gradient(600px 300px at 50% 0%, rgba(244,114,182,0.07), transparent 70%)",
      }}
    >
      <div
        style={{
          maxWidth: "900px",
          margin: "0 auto",
          padding: "0 24px",
          textAlign: "center",
        }}
      >
        <p
          style={{
            color: "var(--color-care)",
            fontSize: "12px",
            letterSpacing: "0.2em",
            textTransform: "uppercase",
            fontWeight: 600,
            fontFamily: "ui-monospace, monospace",
            marginBottom: "20px",
          }}
        >
          Honesty is part of the vision
        </p>
        <h2
          style={{
            fontFamily: "ui-serif, Georgia, Cambria, serif",
            fontSize: "clamp(28px, 4vw, 44px)",
            fontWeight: 500,
            letterSpacing: "-0.02em",
            color: "var(--color-foreground)",
            margin: "0 0 16px",
            lineHeight: 1.15,
          }}
        >
          What works, what&apos;s simulated, and what&apos;s a drawing — always labeled.
        </h2>
        <p
          style={{
            color: "var(--color-muted-strong)",
            fontSize: "clamp(15px, 1.5vw, 18px)",
            lineHeight: 1.7,
            maxWidth: "640px",
            margin: "0 auto 36px",
          }}
        >
          The wallet, the hub, the consoles, and two unlock rails are real
          code you can run today. Everything is a valueless pilot by design,
          and the extensive whitepaper carries the honest table: what exists,
          what is simulated, and what does not exist yet. An economy asking
          for trust in care should model honesty first.
        </p>
        <div
          style={{
            display: "flex",
            gap: "14px",
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <CallToAction
            label="Read the Extensive Whitepaper"
            href="/whitepapers/care_economy_whitepaper"
          />
          <Link
            href="/testnet"
            className="btn-ghost"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              padding: "10px 20px",
              borderRadius: "8px",
              border: "1px solid var(--color-border-strong)",
              color: "var(--color-foreground)",
              fontSize: "14px",
              fontWeight: 500,
              textDecoration: "none",
            }}
          >
            Test the economy yourself
          </Link>
        </div>
      </div>
    </section>
  );
}
