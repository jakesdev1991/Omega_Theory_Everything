// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"use client";

import { useMemo, useState, type FormEvent } from "react";

import { demoCarePosts, demoCareProfile, demoCareRequests } from "@/lib/care/demo-data";
import type { CareFeed, CarePost, CareRequest } from "@/lib/care/types";

const FEED_LABELS: Array<{ id: CareFeed; label: string; count: string }> = [
  { id: "circle", label: "Circle feed", count: "3" },
  { id: "requests", label: "Care requests", count: "3" },
  { id: "projects", label: "Projects", count: "soon" },
];

export function CareSocialPrototype() {
  const [activeFeed, setActiveFeed] = useState<CareFeed>("circle");
  const [posts, setPosts] = useState<CarePost[]>(demoCarePosts);
  const [requests, setRequests] = useState<CareRequest[]>(demoCareRequests);
  const [draft, setDraft] = useState("");
  const [notice, setNotice] = useState<string | null>(null);

  const visiblePosts = useMemo(() => posts, [posts]);

  function createPost(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const body = draft.trim();
    if (!body) return;

    const newPost: CarePost = {
      id: `local-post-${Date.now()}`,
      author: demoCareProfile.displayName,
      role: demoCareProfile.role,
      privacyLevel: demoCareProfile.privacyLevel,
      circle: "My circles",
      body,
      createdAt: "just now",
      audienceCount: 1,
      attentionEvents: 1,
      supportLabel: "Local draft",
      accent: "var(--color-care)",
    };

    setPosts((current) => [newPost, ...current]);
    setDraft("");
    setNotice("Saved to this local prototype only. No account, token, or public post was created.");
  }

  function supportRequest(requestId: string) {
    setRequests((current) =>
      current.map((request) =>
        request.id === requestId
          ? { ...request, supportCount: request.supportCount + 1, status: "covered" }
          : request,
      ),
    );
    setNotice("Solidarity intent recorded locally. No resource transfer has occurred.");
  }

  return (
    <div className="care-shell">
      <section className="care-hero">
        <div>
          <div className="care-kicker">C.A.R.E. / social prototype</div>
          <h1>Call About Resuscitating Everyone.</h1>
          <p>
            A first social surface for circles, care requests, community reach, and
            user-controlled boundaries. It is intentionally local and valueless while
            the architecture is being tested.
          </p>
        </div>
        <div className="care-hero-status">
          <span className="care-dot" />
          <span>local prototype</span>
          <small>No wallet or token wiring</small>
        </div>
      </section>

      <section className="care-status-grid" aria-label="Your local CARE status">
        <StatusCard label="Privacy level" value={demoCareProfile.privacyLevel} detail="Pseudonymous by default" tone="pink" />
        <StatusCard label="Resource quota" value={`${demoCareProfile.selectedQuotaPercent}%`} detail="User-selected device budget" tone="cyan" />
        <StatusCard label="CARE status" value="Active" detail={`${demoCareProfile.contributedUnits}/${demoCareProfile.requiredUnits} local units`} tone="gold" />
        <StatusCard label="Role path" value="Participant" detail="X1 → X2 → Archangel" tone="purple" />
      </section>

      <div className="care-layout">
        <main>
          <div className="care-tabs" role="tablist" aria-label="CARE social feeds">
            {FEED_LABELS.map((feed) => (
              <button
                key={feed.id}
                type="button"
                role="tab"
                aria-selected={activeFeed === feed.id}
                className={activeFeed === feed.id ? "care-tab active" : "care-tab"}
                onClick={() => setActiveFeed(feed.id)}
              >
                {feed.label}
                <span>{feed.count}</span>
              </button>
            ))}
          </div>

          {notice && (
            <div className="care-notice" role="status">
              <span>{notice}</span>
              <button type="button" onClick={() => setNotice(null)} aria-label="Dismiss notice">×</button>
            </div>
          )}

          {activeFeed === "circle" && (
            <>
              <form className="care-compose" onSubmit={createPost}>
                <div className="care-compose-topline">
                  <span className="care-avatar">N</span>
                  <div>
                    <strong>{demoCareProfile.displayName}</strong>
                    <span> · {demoCareProfile.privacyLevel} pseudonymous circle post</span>
                  </div>
                </div>
                <label className="sr-only" htmlFor="care-draft">Share with your circles</label>
                <textarea
                  id="care-draft"
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder="Share a useful presence, a request, or something your circle can carry together…"
                  rows={3}
                />
                <div className="care-compose-actions">
                  <span>Local draft · no public identity required</span>
                  <button className="care-button solid" type="submit" disabled={!draft.trim()}>Share locally</button>
                </div>
              </form>

              <div className="care-feed-heading">
                <div>
                  <span className="care-kicker">Your circles</span>
                  <h2>Attention as a care signal</h2>
                </div>
                <span className="care-muted">No moral score</span>
              </div>

              <div className="care-feed">
                {visiblePosts.map((post) => <PostCard key={post.id} post={post} />)}
              </div>
            </>
          )}

          {activeFeed === "requests" && (
            <div className="care-request-list">
              <div className="care-feed-heading">
                <div>
                  <span className="care-kicker">Solidarity layer</span>
                  <h2>Requests people can carry together</h2>
                </div>
                <span className="care-muted">No debt created</span>
              </div>
              {requests.map((request) => (
                <RequestCard key={request.id} request={request} onSupport={supportRequest} />
              ))}
            </div>
          )}

          {activeFeed === "projects" && (
            <div className="care-empty-state">
              <span className="care-kicker">Project circles</span>
              <h2>Community projects are next.</h2>
              <p>
                This surface will hold voluntary local projects, resource needs, and
                in-person touchpoints. Nothing here will require a token balance.
              </p>
            </div>
          )}
        </main>

        <aside className="care-sidebar">
          <div className="care-panel accent-panel">
            <div className="care-panel-label">Resource budget</div>
            <div className="care-budget-row">
              <strong>{demoCareProfile.selectedQuotaPercent}%</strong>
              <span>user-selected quota</span>
            </div>
            <div className="care-progress"><span style={{ width: `${demoCareProfile.contributedUnits}%` }} /></div>
            <p>
              The wallet may contribute a small visible device budget while powered on.
              You can pause it; this prototype does not use your device.
            </p>
            <button className="care-button ghost" type="button" onClick={() => setNotice("Resource controls will be wired to the wallet budget after the local policy is finalized.")}>Review quota</button>
          </div>

          <div className="care-panel">
            <div className="care-panel-label">Privacy ladder</div>
            <div className="care-privacy-list">
              <PrivacyRow level="P0" title="Pseudonymous" text="Basic circles and requests" active />
              <PrivacyRow level="P1" title="Verified person" text="Stronger accountability" />
              <PrivacyRow level="P2" title="Service history" text="Expanded proof scope" />
              <PrivacyRow level="P3" title="Senior steward" text="Archangel consideration" />
            </div>
            <p>More disclosure can unlock trust scope. It never becomes a requirement for ordinary participation.</p>
          </div>

          <div className="care-panel">
            <div className="care-panel-label">Protocol boundary</div>
            <ul className="care-boundary-list">
              <li>CARE is not wired to AMITY here.</li>
              <li>Hardship has no expiry cap.</li>
              <li>Attention is a measurable signal, not a moral score.</li>
              <li>Agents cannot issue strikes or expel people.</li>
            </ul>
          </div>
        </aside>
      </div>

      <style>{`
        .care-shell { max-width: 1180px; margin: 0 auto; padding: 118px 24px 100px; }
        .care-hero { display: flex; align-items: end; justify-content: space-between; gap: 32px; padding: 34px 0 44px; border-bottom: 1px solid var(--color-border); }
        .care-kicker { color: var(--color-accent-soft); font: 600 11px/1.2 ui-monospace, SFMono-Regular, monospace; letter-spacing: .16em; text-transform: uppercase; }
        .care-hero h1 { margin: 12px 0 14px; color: var(--color-foreground); font: 500 clamp(40px, 7vw, 74px)/.98 ui-serif, Georgia, serif; letter-spacing: -.04em; max-width: 760px; }
        .care-hero p { max-width: 650px; margin: 0; color: var(--color-muted-strong); font-size: 17px; line-height: 1.65; }
        .care-hero-status { min-width: 190px; display: grid; gap: 8px; align-content: center; padding: 16px; border: 1px solid rgba(52, 211, 153, .3); border-radius: 14px; background: rgba(52, 211, 153, .05); color: var(--color-unlock-soft); font: 600 12px ui-monospace, SFMono-Regular, monospace; text-transform: uppercase; letter-spacing: .08em; }
        .care-hero-status small { color: var(--color-muted); font: 400 11px ui-sans-serif, system-ui, sans-serif; text-transform: none; letter-spacing: 0; }
        .care-dot { width: 8px; height: 8px; border-radius: 99px; background: var(--color-unlock); box-shadow: 0 0 16px var(--color-unlock); }
        .care-status-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; padding: 24px 0; }
        .care-status-card, .care-panel, .care-compose, .care-post, .care-request-card, .care-empty-state { border: 1px solid var(--color-border); border-radius: 16px; background: rgba(255,255,255,.018); }
        .care-status-card { padding: 17px; }
        .care-status-card .label, .care-panel-label { color: var(--color-muted); font: 600 10px ui-monospace, SFMono-Regular, monospace; letter-spacing: .14em; text-transform: uppercase; }
        .care-status-card strong { display: block; margin: 10px 0 4px; font: 500 25px ui-serif, Georgia, serif; }
        .care-status-card p { margin: 0; color: var(--color-muted-strong); font-size: 12px; }
        .tone-pink strong { color: var(--color-care); } .tone-cyan strong { color: var(--color-twc); } .tone-gold strong { color: var(--color-omega); } .tone-purple strong { color: var(--color-amity); }
        .care-layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 28px; align-items: start; }
        .care-tabs { display: flex; gap: 6px; padding: 6px; border: 1px solid var(--color-border); border-radius: 12px; background: rgba(0,0,0,.16); }
        .care-tab { flex: 1; display: flex; justify-content: center; gap: 8px; padding: 11px 12px; border: 0; border-radius: 8px; color: var(--color-muted-strong); background: transparent; font: 600 12px ui-monospace, SFMono-Regular, monospace; cursor: pointer; }
        .care-tab span { color: var(--color-muted); font-weight: 400; } .care-tab.active { color: var(--color-foreground); background: rgba(192,132,87,.13); } .care-tab.active span { color: var(--color-accent-soft); }
        .care-notice { display: flex; justify-content: space-between; gap: 12px; margin: 14px 0; padding: 12px 14px; border: 1px solid rgba(52,211,153,.3); border-radius: 10px; background: rgba(52,211,153,.07); color: var(--color-unlock-soft); font-size: 12px; line-height: 1.45; }
        .care-notice button { border: 0; background: none; color: inherit; cursor: pointer; font-size: 18px; line-height: 1; }
        .care-compose { margin-top: 18px; padding: 18px; }
        .care-compose-topline { display: flex; align-items: center; gap: 9px; margin-bottom: 14px; color: var(--color-muted-strong); font-size: 13px; } .care-compose-topline strong { color: var(--color-foreground); } .care-compose-topline span:last-child { color: var(--color-muted); }
        .care-avatar { display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; border-radius: 50%; background: linear-gradient(135deg, var(--color-care), var(--color-accent)); color: #120b10; font: 700 12px ui-monospace, monospace; }
        .care-compose textarea { width: 100%; resize: vertical; border: 1px solid var(--color-border-strong); border-radius: 10px; padding: 13px; background: rgba(0,0,0,.2); color: var(--color-foreground); font: 14px/1.55 ui-sans-serif, system-ui, sans-serif; outline: none; }
        .care-compose textarea:focus { border-color: var(--color-accent); } .care-compose textarea::placeholder { color: var(--color-muted); }
        .care-compose-actions { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-top: 12px; color: var(--color-muted); font-size: 11px; }
        .care-button { border-radius: 8px; padding: 9px 13px; cursor: pointer; font: 600 11px ui-monospace, SFMono-Regular, monospace; letter-spacing: .02em; } .care-button:disabled { opacity: .4; cursor: not-allowed; }
        .care-button.solid { border: 1px solid var(--color-accent); background: var(--color-accent); color: #120b08; } .care-button.ghost { border: 1px solid var(--color-border-strong); background: transparent; color: var(--color-muted-strong); }
        .care-feed-heading { display: flex; align-items: end; justify-content: space-between; gap: 20px; margin: 34px 0 14px; } .care-feed-heading h2 { margin: 7px 0 0; font: 500 28px ui-serif, Georgia, serif; letter-spacing: -.02em; } .care-muted { color: var(--color-muted); font-size: 12px; }
        .care-feed, .care-request-list { display: grid; gap: 12px; }
        .care-post { padding: 19px; transition: border-color .15s ease, transform .15s ease; } .care-post:hover, .care-request-card:hover { border-color: var(--color-border-strong); transform: translateY(-1px); }
        .care-post-head, .care-request-head { display: flex; justify-content: space-between; gap: 16px; } .care-post-person { display: flex; gap: 10px; align-items: center; } .care-post-avatar { width: 34px; height: 34px; border-radius: 10px; display: inline-flex; align-items: center; justify-content: center; color: #0b0910; font: 700 12px ui-monospace, monospace; }
        .care-post-person strong { display: block; font-size: 14px; } .care-post-person span { color: var(--color-muted); font-size: 11px; } .care-post-time { color: var(--color-muted); font-size: 11px; }
        .care-post-body { margin: 18px 0; color: var(--color-muted-strong); font-size: 15px; line-height: 1.65; } .care-post-label { display: inline-flex; padding: 5px 8px; border-radius: 99px; background: rgba(192,132,87,.1); color: var(--color-accent-soft); font: 600 10px ui-monospace, monospace; }
        .care-post-metrics { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 16px; padding-top: 13px; border-top: 1px solid var(--color-border); color: var(--color-muted); font-size: 11px; } .care-post-metrics strong { color: var(--color-muted-strong); font-weight: 600; }
        .care-sidebar { display: grid; gap: 12px; } .care-panel { padding: 18px; } .accent-panel { border-color: rgba(192,132,87,.35); background: linear-gradient(150deg, rgba(192,132,87,.08), rgba(255,255,255,.015)); }
        .care-budget-row { display: flex; align-items: baseline; gap: 9px; margin: 14px 0 9px; } .care-budget-row strong { font: 500 34px ui-serif, Georgia, serif; color: var(--color-accent-soft); } .care-budget-row span { color: var(--color-muted); font-size: 11px; }
        .care-progress { height: 6px; border-radius: 99px; overflow: hidden; background: var(--color-border); } .care-progress span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, var(--color-accent), var(--color-omega)); }
        .care-panel p { color: var(--color-muted); font-size: 12px; line-height: 1.55; } .care-panel .care-button { margin-top: 4px; }
        .care-privacy-list { display: grid; gap: 10px; margin: 16px 0; } .care-privacy-row { display: grid; grid-template-columns: 30px 1fr; gap: 9px; align-items: center; } .care-privacy-level { width: 28px; height: 23px; display: inline-flex; align-items: center; justify-content: center; border: 1px solid var(--color-border-strong); border-radius: 6px; color: var(--color-muted); font: 600 10px ui-monospace, monospace; } .care-privacy-row.active .care-privacy-level { border-color: var(--color-care); color: var(--color-care); background: rgba(244,114,182,.08); } .care-privacy-row strong { display: block; color: var(--color-muted-strong); font-size: 12px; } .care-privacy-row span { color: var(--color-muted); font-size: 11px; }
        .care-boundary-list { display: grid; gap: 10px; margin: 16px 0 2px; padding: 0; list-style: none; color: var(--color-muted-strong); font-size: 12px; line-height: 1.45; } .care-boundary-list li { padding-left: 14px; position: relative; } .care-boundary-list li::before { content: ""; position: absolute; left: 0; top: .55em; width: 5px; height: 5px; border-radius: 50%; background: var(--color-care); }
        .care-request-card { padding: 18px; transition: border-color .15s ease, transform .15s ease; } .care-request-head strong { font-size: 15px; } .care-request-head span { color: var(--color-muted); font-size: 11px; } .care-request-card p { margin: 13px 0 16px; color: var(--color-muted-strong); font-size: 13px; line-height: 1.55; } .care-request-foot { display: flex; align-items: center; justify-content: space-between; gap: 12px; color: var(--color-muted); font-size: 11px; }
        .care-empty-state { padding: 44px 28px; margin-top: 18px; } .care-empty-state h2 { margin: 10px 0; font: 500 30px ui-serif, Georgia, serif; } .care-empty-state p { max-width: 510px; margin: 0; color: var(--color-muted-strong); line-height: 1.65; }
        .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
        @media (max-width: 900px) { .care-status-grid { grid-template-columns: repeat(2, 1fr); } .care-layout { grid-template-columns: 1fr; } .care-sidebar { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        @media (max-width: 640px) { .care-shell { padding: 98px 16px 72px; } .care-hero { display: block; } .care-hero-status { margin-top: 24px; } .care-status-grid { grid-template-columns: 1fr 1fr; } .care-sidebar { grid-template-columns: 1fr; } .care-compose-actions { align-items: stretch; flex-direction: column; } .care-compose-actions .care-button { width: 100%; } .care-tab { flex-direction: column; gap: 3px; align-items: center; font-size: 10px; } }
      `}</style>
    </div>
  );
}

function StatusCard({ label, value, detail, tone }: { label: string; value: string; detail: string; tone: string }) {
  return (
    <div className={`care-status-card tone-${tone}`}>
      <div className="label">{label}</div>
      <strong>{value}</strong>
      <p>{detail}</p>
    </div>
  );
}

function PostCard({ post }: { post: CarePost }) {
  return (
    <article className="care-post">
      <div className="care-post-head">
        <div className="care-post-person">
          <span className="care-post-avatar" style={{ background: post.accent }}>{post.author.slice(0, 1)}</span>
          <div>
            <strong>{post.author}</strong>
            <span>{post.circle} · {post.privacyLevel} · {post.role}</span>
          </div>
        </div>
        <span className="care-post-time">{post.createdAt}</span>
      </div>
      <p className="care-post-body">{post.body}</p>
      <span className="care-post-label">{post.supportLabel}</span>
      <div className="care-post-metrics">
        <span><strong>{post.audienceCount}</strong> people reached</span>
        <span><strong>{post.attentionEvents}</strong> attention events</span>
        <span>No moral score attached</span>
      </div>
    </article>
  );
}

function RequestCard({ request, onSupport }: { request: CareRequest; onSupport: (id: string) => void }) {
  return (
    <article className="care-request-card">
      <div className="care-request-head">
        <div>
          <strong>{request.title}</strong>
          <span>{request.circle} · {request.privacy}</span>
        </div>
        <span>{request.status === "covered" ? "covered" : "open"}</span>
      </div>
      <p>{request.detail}</p>
      <div className="care-request-foot">
        <span>{request.supportCount} people carrying this</span>
        <button className="care-button ghost" type="button" onClick={() => onSupport(request.id)} disabled={request.status === "covered"}>
          {request.status === "covered" ? "Carried" : "Carry this"}
        </button>
      </div>
    </article>
  );
}

function PrivacyRow({ level, title, text, active = false }: { level: string; title: string; text: string; active?: boolean }) {
  return (
    <div className={active ? "care-privacy-row active" : "care-privacy-row"}>
      <span className="care-privacy-level">{level}</span>
      <div><strong>{title}</strong><span>{text}</span></div>
    </div>
  );
}
