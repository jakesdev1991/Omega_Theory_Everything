"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { SimplePool, generateSecretKey, getPublicKey, nip19 } from "nostr-tools";
import type { Event as NostrEvent } from "nostr-tools";

import { ECONOMY_NOSTR_KINDS } from "@/lib/nostr";
import {
  buildJobRequestTemplate,
  bytesToHex,
  demoJobResult,
  demoListings,
  dedupeListings,
  hexToBytes,
  jobResultFilter,
  parseJobResult,
  parseListingEvent,
  signJobRequest,
  type AppListing,
  type JobResult,
} from "@/lib/nostr-store";

const OPERATOR_KEY_STORAGE = "omega.store.operatorKey.v1";

interface JobState {
  requestId: string;
  appId: string;
  jobKind: number;
  phase: "publishing" | "awaiting" | "done" | "error";
  result: JobResult | null;
  error?: string;
  startedAt: number;
}

function decodePubkey(input: string): string | null {
  const value = input.trim();
  if (!value) return null;
  if (value.startsWith("npub1")) {
    try {
      const decoded = nip19.decode(value);
      if (decoded.type === "npub") return decoded.data as string;
    } catch {
      return null;
    }
    return null;
  }
  return /^[0-9a-fA-F]{64}$/.test(value) ? value.toLowerCase() : null;
}

function decodeSecret(input: string): Uint8Array | null {
  const value = input.trim();
  try {
    if (value.startsWith("nsec1")) {
      const decoded = nip19.decode(value);
      if (decoded.type === "nsec") return decoded.data as Uint8Array;
      return null;
    }
    return hexToBytes(value);
  } catch {
    return null;
  }
}

export function AppStore() {
  const [relaysInput, setRelaysInput] = useState("wss://nos.lol, wss://relay.damus.io");
  const [rootInput, setRootInput] = useState("");
  const [mode, setMode] = useState<"demo" | "live">("demo");
  const [connected, setConnected] = useState(false);
  const [connectError, setConnectError] = useState<string | null>(null);
  const [listings, setListings] = useState<AppListing[]>([]);
  const [jobs, setJobs] = useState<Record<string, JobState>>({});
  const [paramDrafts, setParamDrafts] = useState<Record<string, string>>({});
  const [operatorNpub, setOperatorNpub] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const poolRef = useRef<SimplePool | null>(null);
  const operatorRef = useRef<Uint8Array | null>(null);
  const rootHexRef = useRef<string | null>(null);
  const closeSubRef = useRef<{ close: () => void } | null>(null);

  const relays = useMemo(
    () =>
      relaysInput
        .split(",")
        .map((relay) => relay.trim())
        .filter((relay) => relay.startsWith("wss://")),
    [relaysInput],
  );

  /* Restore or create the operator key (local only, never uploaded). */
  useEffect(() => {
    let secret: Uint8Array | null = null;
    try {
      const stored = localStorage.getItem(OPERATOR_KEY_STORAGE);
      if (stored) secret = decodeSecret(stored);
    } catch {
      secret = null;
    }
    if (!secret) {
      secret = generateSecretKey();
      try {
        localStorage.setItem(OPERATOR_KEY_STORAGE, bytesToHex(secret));
      } catch {
        /* private mode: session-only key */
      }
    }
    operatorRef.current = secret;
    setOperatorNpub(nip19.npubEncode(getPublicKey(secret)));
  }, []);

  const disconnect = useCallback(() => {
    closeSubRef.current?.close();
    closeSubRef.current = null;
    poolRef.current?.close(relays);
    poolRef.current = null;
    setConnected(false);
  }, [relays]);

  useEffect(() => disconnect, [disconnect]);

  async function connect() {
    setConnectError(null);
    setListings([]);

    if (mode === "demo") {
      const demoSecret = hexToBytes("0000000000000000000000000000000000000000000000000000000000000001");
      const demoRoot = getPublicKey(demoSecret);
      rootHexRef.current = rootInput.trim() ? decodePubkey(rootInput) : demoRoot;
      const events = demoListings(rootHexRef.current ?? demoRoot);
      const parsed = events.map(parseListingEvent).filter((listing): listing is AppListing => !!listing);
      setListings(dedupeListings(parsed));
      setConnected(true);
      setNotice(
        `Demo mode: ${parsed.length} fixture listings loaded, and "Run" is answered by an in-page responder. No relay traffic.`,
      );
      return;
    }

    const rootHex = decodePubkey(rootInput);
    if (!rootHex) {
      setConnectError("Enter the store root key as npub1… or 64-char hex (NOSTR_STORE_ROOT_NPUB on the server).");
      return;
    }
    if (relays.length === 0) {
      setConnectError("Add at least one wss:// relay, or switch to demo mode.");
      return;
    }

    rootHexRef.current = rootHex;
    const pool = new SimplePool();
    poolRef.current = pool;

    const directorySubscription = pool.subscribeMany(
      relays,
      { kinds: [ECONOMY_NOSTR_KINDS.storeHandler, ECONOMY_NOSTR_KINDS.storeListing], authors: [rootHex] },
      {
        onevent(event: NostrEvent) {
          const listing = parseListingEvent(event);
          if (!listing) return;
          setListings((current) => dedupeListings([...current, listing]));
        },
        onclose(reasons: { url: string; reason: string }[]) {
          setNotice(`Relay subscription closed: ${reasons.map((entry) => `${entry.url}: ${entry.reason}`).join(", ") || "unknown reason"}`);
        },
      },
    );
    closeSubRef.current = directorySubscription;
    setConnected(true);
    setNotice(`Listening on ${relays.length} relay(s) for directory kinds 31990/30017 from ${rootInput.trim().slice(0, 16)}…`);
  }

  async function runApp(listing: AppListing) {
    if (!operatorRef.current || !rootHexRef.current) return;
    const operatorPubkey = getPublicKey(operatorRef.current);

    let params: Record<string, unknown>;
    try {
      const draft = paramDrafts[listing.appId];
      params = draft ? (JSON.parse(draft) as Record<string, unknown>) : listing.paramsTemplate;
    } catch {
      setNotice(`Parameters for ${listing.name} are not valid JSON.`);
      return;
    }

    const request = signJobRequest(
      buildJobRequestTemplate({ listing, serverPubkey: rootHexRef.current, params }),
      operatorRef.current,
    );

    const job: JobState = {
      requestId: request.event.id,
      appId: listing.appId,
      jobKind: request.jobKind,
      phase: "publishing",
      result: null,
      startedAt: Date.now(),
    };
    setJobs((current) => ({ ...current, [request.event.id]: job }));

    if (mode === "demo") {
      window.setTimeout(() => {
        const resultEvent = demoJobResult(request.event, operatorPubkey);
        const result = parseJobResult(resultEvent, request.event.id);
        setJobs((current) => ({
          ...current,
          [request.event.id]: { ...job, phase: "done", result },
        }));
      }, 450);
      setJobs((current) => ({ ...current, [request.event.id]: { ...job, phase: "awaiting" } }));
      return;
    }

    const pool = poolRef.current;
    if (!pool) {
      setJobs((current) => ({
        ...current,
        [request.event.id]: { ...job, phase: "error", error: "Not connected." },
      }));
      return;
    }

    try {
      await pool.publish(relays, request.event);
    } catch (error) {
      setJobs((current) => ({
        ...current,
        [request.event.id]: {
          ...job,
          phase: "error",
          error: error instanceof Error ? error.message : "Publish failed",
        },
      }));
      return;
    }

    setJobs((current) => ({ ...current, [request.event.id]: { ...job, phase: "awaiting" } }));

    const subscription = pool.subscribeMany(relays, jobResultFilter(request.event.id, operatorPubkey, request.jobKind), {
      onevent(event: NostrEvent) {
        const result = parseJobResult(event, request.event.id);
        if (!result) return;
        subscription.close();
        setJobs((current) => ({
          ...current,
          [request.event.id]: { ...job, phase: "done", result },
        }));
      },
    });

    window.setTimeout(() => {
      subscription.close();
      setJobs((current) => {
        const state = current[request.event.id];
        if (state && state.phase === "awaiting") {
          return {
            ...current,
            [request.event.id]: { ...state, phase: "error", error: "Timed out waiting for the mobile node (60s)." },
          };
        }
        return current;
      });
    }, 60_000);
  }

  async function settleToEconomy(listing: AppListing, job: JobState) {
    if (!job.result) return;
    try {
      const response = await fetch("/api/economy/state", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          action: "work_propose_and_execute",
          contributorCommitment: `nostr:${job.result.serverPubkey.slice(0, 16)}`,
          workClass: listing.workClass,
          objective: `NIP-90 job ${job.requestId.slice(0, 8)} via ${listing.name}`,
          resourceBudget: 100,
        }),
      });
      const data = await response.json();
      setNotice(
        data.ok
          ? `Settled: receipt ${data.receipt.workId} credited ${data.receipt.issuedTwcUnits} TWC for the verified job.`
          : `Settlement refused: ${data.error}`,
      );
    } catch (error) {
      setNotice(`Settlement failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  return (
    <div className="store">
      <header className="store-hero">
        <p className="store-eyebrow">App Store · Nostr backplane</p>
        <h1>
          A static storefront.
          <br />
          <span className="store-accent">Your phone is the backend.</span>
        </h1>
        <p className="store-lede">
          The directory is read straight from relays: parameterized replaceable events (kinds 31990
          and 30017) published by your sovereign root key. Pressing Run signs a NIP-90 job request
          (5000–5999); the Termux daemon on the Pixel 8a verifies it, executes an allowlisted
          algorithm in the Debian sandbox, and answers with a job result (request kind + 1000) that
          this page correlates by <code>[&quot;e&quot;, requestId]</code>.
        </p>

        <div className="store-config">
          <div className="store-config-row">
            <label>
              mode
              <select value={mode} onChange={(event) => { setMode(event.target.value as "demo" | "live"); disconnect(); }}>
                <option value="demo">demo (offline responder)</option>
                <option value="live">live relays</option>
              </select>
            </label>
            <label className="grow">
              relays
              <input value={relaysInput} onChange={(event) => setRelaysInput(event.target.value)} disabled={mode === "demo"} />
            </label>
            <label className="grow">
              store root key (npub or hex)
              <input value={rootInput} onChange={(event) => setRootInput(event.target.value)} placeholder="npub1… / server pubkey" />
            </label>
            <button type="button" className="store-btn" onClick={() => void connect()}>
              {connected ? "↻ Reconnect" : "⚡ Connect"}
            </button>
            {connected ? (
              <button type="button" className="store-btn ghost" onClick={disconnect}>
                Disconnect
              </button>
            ) : null}
          </div>
          <div className="store-statusline">
            <span className={connected ? "store-pill ok" : "store-pill idle"}>
              {connected ? (mode === "demo" ? "demo connected" : "connected") : "disconnected"}
            </span>
            <span>{relays.length} relay(s)</span>
            <span>{listings.length} listing(s)</span>
            <span>
              operator <code title={operatorNpub ?? ""}>{operatorNpub ? `${operatorNpub.slice(0, 12)}…` : "…"}</code>
            </span>
            <span>kinds 31990/30017 → 5000-5999 → 6000-6999</span>
          </div>
          {connectError ? <p className="store-error">{connectError}</p> : null}
          {notice ? <p className="store-notice">{notice}</p> : null}
        </div>
      </header>

      <section className="store-grid">
        {listings.length === 0 ? (
          <div className="store-empty">
            <p>
              {connected
                ? "No directory events yet. Publish a kind 31990 handler announcement from the root key (mobile-node/README.md shows how), or stay in demo mode."
                : "Connect to load the directory. Demo mode needs no relays and answers Run locally so the whole flow stays testable."}
            </p>
          </div>
        ) : null}

        {listings.map((listing) => {
          const draft = paramDrafts[listing.appId] ?? JSON.stringify(listing.paramsTemplate, null, 2);
          const listingJobs = Object.values(jobs).filter((job) => job.appId === listing.appId);
          const latest = listingJobs[listingJobs.length - 1];
          return (
            <article key={`${listing.publisher}:${listing.appId}`} className="store-card">
              <div className="store-card-head">
                <span className="store-kind-tag">kind {listing.sourceKind}</span>
                <span className="store-kind-tag dvm">job {listing.jobKind} → {listing.jobKind + 1000}</span>
              </div>
              <h2>{listing.name}</h2>
              <p className="store-about">{listing.about}</p>
              <code className="store-appid">d={listing.appId} · workClass={listing.workClass}</code>

              <label className="store-params">
                job parameters (JSON)
                <textarea
                  rows={3}
                  value={draft}
                  onChange={(event) => setParamDrafts((current) => ({ ...current, [listing.appId]: event.target.value }))}
                />
              </label>

              <div className="store-card-actions">
                <button type="button" className="store-btn" disabled={!connected} onClick={() => void runApp(listing)}>
                  ▶ Run on mobile node
                </button>
                {latest?.result ? (
                  <button type="button" className="store-btn ghost" onClick={() => void settleToEconomy(listing, latest)}>
                    ⛓ Settle result as TWC work
                  </button>
                ) : null}
              </div>

              {listingJobs.slice(-1).map((job) => (
                <div key={job.requestId} className="store-job">
                  <div className="store-job-head">
                    <span className={job.phase === "done" ? "store-pill ok" : job.phase === "error" ? "store-pill bad" : "store-pill idle"}>
                      {job.phase}
                    </span>
                    <code>req {job.requestId.slice(0, 12)}…</code>
                  </div>
                  {job.error ? <p className="store-error">{job.error}</p> : null}
                  {job.result ? <pre className="store-result">{job.result.content}</pre> : null}
                </div>
              ))}
            </article>
          );
        })}
      </section>

      <section className="store-security">
        <h2>Security contract of this storefront</h2>
        <ul>
          <li>Directory events and job results are signature-verified (<code>verifyEvent</code>) before rendering or correlation; unverifiable events are dropped.</li>
          <li>Job requests are signed by a local operator key kept in this browser profile only; the page never transmits it.</li>
          <li>The mobile node only executes allowlisted algorithm ids, only for operator pubkeys, with argv arrays and timeouts — relay content is untrusted input, never shell code.</li>
          <li>Results settle into the economy ledger as audited TWC work receipts; the ledger never trusts a relay event as settlement authority on its own.</li>
        </ul>
      </section>

      <style>{`
        .store { max-width: 1180px; margin: 0 auto; padding: 120px 24px 96px; }
        .store-eyebrow { color: var(--color-amity); font: 600 12px ui-monospace, monospace; letter-spacing: .2em; text-transform: uppercase; margin: 0 0 20px; }
        .store h1 { font-family: ui-serif, Georgia, serif; font-size: clamp(36px, 5.5vw, 64px); line-height: 1.06; font-weight: 500; letter-spacing: -0.025em; margin: 0 0 20px; max-width: 900px; }
        .store-accent { color: var(--color-amity); }
        .store-lede { color: var(--color-muted-strong); font-size: clamp(15px, 1.5vw, 18px); line-height: 1.7; max-width: 800px; margin: 0 0 28px; }
        .store-lede code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; font-size: 12px; }

        .store-config { border: 1px solid var(--color-border); border-radius: 14px; padding: 16px 18px; background: rgba(255,255,255,0.015); display: flex; flex-direction: column; gap: 12px; }
        .store-config-row { display: flex; gap: 12px; flex-wrap: wrap; align-items: flex-end; }
        .store-config-row label { display: flex; flex-direction: column; gap: 6px; color: var(--color-muted); font: 600 10px ui-monospace, monospace; text-transform: uppercase; letter-spacing: .1em; }
        .store-config-row label.grow { flex: 1; min-width: 220px; }
        .store-config-row input, .store-config-row select { border: 1px solid var(--color-border-strong); border-radius: 8px; padding: 9px 12px; background: rgba(0,0,0,0.35); color: var(--color-foreground); font: 12.5px ui-monospace, monospace; width: 100%; }
        .store-statusline { display: flex; gap: 18px; flex-wrap: wrap; color: var(--color-muted); font: 11.5px ui-monospace, monospace; align-items: center; }
        .store-statusline code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; }
        .store-notice { margin: 0; color: var(--color-muted-strong); font-size: 12.5px; }
        .store-error { margin: 4px 0 0; color: var(--color-care); font-size: 12.5px; }

        .store-pill { font: 700 10px ui-monospace, monospace; letter-spacing: .08em; text-transform: uppercase; padding: 3px 9px; border-radius: 999px; }
        .store-pill.ok { background: rgba(52,211,153,0.14); color: var(--color-unlock); }
        .store-pill.idle { background: rgba(251,191,36,0.14); color: var(--color-omega); }
        .store-pill.bad { background: rgba(244,114,182,0.14); color: var(--color-care); }

        .store-btn { border: 1px solid var(--color-amity); background: rgba(167,139,250,0.14); color: var(--color-amity); border-radius: 8px; padding: 10px 16px; font: 700 12px ui-monospace, monospace; cursor: pointer; }
        .store-btn:hover { background: rgba(167,139,250,0.24); }
        .store-btn:disabled { opacity: .5; cursor: not-allowed; }
        .store-btn.ghost { border-color: var(--color-border-strong); background: transparent; color: var(--color-muted-strong); }

        .store-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 34px; }
        .store-empty { grid-column: 1 / -1; border: 1px dashed var(--color-border-strong); border-radius: 14px; padding: 26px; color: var(--color-muted-strong); font-size: 13.5px; line-height: 1.6; }
        .store-card { border: 1px solid var(--color-border); border-radius: 16px; padding: 20px 22px; background: rgba(255,255,255,0.015); display: flex; flex-direction: column; gap: 10px; }
        .store-card-head { display: flex; gap: 8px; flex-wrap: wrap; }
        .store-kind-tag { font: 700 10px ui-monospace, monospace; padding: 3px 8px; border-radius: 5px; background: rgba(167,139,250,0.14); color: var(--color-amity); }
        .store-kind-tag.dvm { background: rgba(34,211,238,0.12); color: var(--color-twc); }
        .store-card h2 { font-family: ui-serif, Georgia, serif; font-size: 22px; font-weight: 500; margin: 0; }
        .store-about { color: var(--color-muted-strong); font-size: 13px; line-height: 1.6; margin: 0; }
        .store-appid { color: var(--color-muted); font-size: 11px; }
        .store-params { display: flex; flex-direction: column; gap: 6px; color: var(--color-muted); font: 600 10px ui-monospace, monospace; text-transform: uppercase; letter-spacing: .1em; }
        .store-params textarea { border: 1px solid var(--color-border-strong); border-radius: 8px; padding: 10px 12px; background: rgba(0,0,0,0.35); color: var(--color-foreground); font: 12px ui-monospace, monospace; resize: vertical; }
        .store-card-actions { display: flex; gap: 10px; flex-wrap: wrap; }
        .store-job { border-top: 1px dashed var(--color-border-strong); padding-top: 10px; display: flex; flex-direction: column; gap: 8px; }
        .store-job-head { display: flex; gap: 10px; align-items: center; }
        .store-job-head code { color: var(--color-muted); font-size: 11px; }
        .store-result { margin: 0; padding: 10px 12px; background: rgba(0,0,0,0.4); border: 1px solid var(--color-border); border-radius: 8px; color: var(--color-unlock-soft); font: 11.5px ui-monospace, monospace; overflow-x: auto; max-height: 220px; }

        .store-security { margin-top: 48px; border: 1px solid var(--color-border); border-radius: 16px; padding: 24px 26px; background: rgba(255,255,255,0.015); }
        .store-security h2 { font-family: ui-serif, Georgia, serif; font-size: 24px; font-weight: 500; margin: 0 0 14px; }
        .store-security ul { margin: 0; padding-left: 20px; color: var(--color-muted-strong); font-size: 13.5px; line-height: 1.75; display: grid; gap: 8px; }
        .store-security code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
      `}</style>
    </div>
  );
}
