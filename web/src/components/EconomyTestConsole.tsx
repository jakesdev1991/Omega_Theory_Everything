// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import type { SuiteResult } from "@/lib/domain/scenarios";

type Tab = "readiness" | "scenarios" | "ledger" | "actions" | "audit" | "nostr";

interface HealthCheck {
  id: string;
  label: string;
  status: "pass" | "warn" | "fail";
  detail: string;
}

interface HealthPayload {
  ok: boolean;
  readiness: string;
  policyVersion: string;
  testMode: boolean;
  checks: HealthCheck[];
  ledger: Record<string, unknown>;
  rails: {
    rails: {
      omega: { configured: boolean; network: string; rpcUrl: string; gateAddress: string | null; message: string };
      twc: { configured: boolean; network: string; rpcUrl: string; mintAddress: string | null; message: string };
    };
  };
  nostr: { configured: boolean; relays: string[]; status: string };
  scenarios: string[];
}

interface ScenarioSummary {
  id: string;
  title: string;
  plane: string;
  slice: string;
  description: string;
  stepCount: number;
}

interface EconomyState {
  ok: boolean;
  testMode?: boolean;
  participants: Array<{ participantId: string; role: string; privacyLevel: string; status: string }>;
  budgets: Array<{ participantId: string; budgetBps: number; minimumQuotaUnits: number; contributedUnits: number; paused: boolean }>;
  consents: Array<{ participantId: string; scope: string; granted: boolean }>;
  balances: { care: Record<string, number>; twc: Record<string, number>; omega: Record<string, number>; amity: Record<string, number> };
  receipts: Array<{ workId: string; workClass: string; contributorCommitment: string; issuedTwcUnits: number; status: string }>;
  appeals: Array<{ appealId: string; workId: string; status: string }>;
  conversions: Array<{ conversionId: string; participantId: string; careAmount: number; netAmityAmount: number }>;
  governance: Array<{ proposalId: string; title: string; status: string }>;
  hardships: Array<{ requestId: string; participantId: string; status: string }>;
  auditEvents: Array<Record<string, unknown>>;
}

interface AuditPayload {
  ok: boolean;
  count: number;
  total: number;
  actions: string[];
  planes: string[];
  events: Array<{
    eventId: string;
    timestamp: string;
    plane: string;
    action: string;
    actorCommitment: string;
    inputCommitments: string[];
    outputCommitments: string[];
  }>;
}

interface SocialPayload {
  ok: boolean;
  status: string;
  configured: boolean;
  relays: string[];
  publisherNpub: string | null;
  kinds: Record<string, number>;
  requiredFromClient: string[];
  notes: string[];
  consumers: Array<{ route: string; uses: string }>;
}

const ACTION_TEMPLATES: Record<string, Record<string, unknown>> = {
  register_participant: { action: "register_participant", participantId: "ana", role: "participant", privacyLevel: "P1" },
  consent_record: { action: "consent_record", participantId: "ana", scope: "resource_contribution", granted: true },
  resource_update: { action: "resource_update", participantId: "ana", deltaUnits: 25, paused: false, consent: true },
  reach_record: { action: "reach_record", participantId: "ana", audiences: 4, attentions: 6 },
  distribute_pool: { action: "distribute_pool" },
  hardship_request: { action: "hardship_request", participantId: "ana", uncoveredUnits: 30 },
  hardship_sponsor: { action: "hardship_sponsor", requestId: "req-…", sponsorId: "ben" },
  work_propose_and_execute: {
    action: "work_propose_and_execute",
    contributorCommitment: "ana",
    workClass: "engineering_protocol",
    objective: "Ship the audit adapter",
    resourceBudget: 120,
  },
  work_appeal_open: { action: "work_appeal_open", workId: "work-…", appellantCommitment: "watchdog", reason: "Evidence not reproducible" },
  work_appeal_resolve: { action: "work_appeal_resolve", appealId: "appeal-…", reviewerCommitment: "archangel-1", decision: "reversed", note: "Confirmed by second reviewer" },
  care_amity_convert: { action: "care_amity_convert", participantId: "ana", careAmount: 500, extremeHoldback: false },
  governance_propose: {
    action: "governance_propose",
    proposerCommitment: "ana",
    targetPlane: "AMITY",
    title: "Bound conversion ceiling",
    description: "Cap extreme holdback at 1500 bps",
    diff: { ceilingBps: 3500 },
    timelockSeconds: 5,
  },
  governance_execute: { action: "governance_execute", proposalId: "gov-…" },
  credit_balance: { action: "credit_balance", currency: "omega", participantId: "ana", units: 1000, reason: "test-mode operator credit" },
};

function StatusPill({ status }: { status: "pass" | "warn" | "fail" | "ready" | "not_configured" }) {
  const tone =
    status === "pass" || status === "ready" ? "ok" : status === "warn" || status === "not_configured" ? "warn" : "bad";
  return <span className={`etc-pill ${tone}`}>{status}</span>;
}

export function EconomyTestConsole() {
  const [tab, setTab] = useState<Tab>("readiness");
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [state, setState] = useState<EconomyState | null>(null);
  const [summaries, setSummaries] = useState<ScenarioSummary[]>([]);
  const [suite, setSuite] = useState<SuiteResult | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [audit, setAudit] = useState<AuditPayload | null>(null);
  const [auditPlane, setAuditPlane] = useState("");
  const [auditAction, setAuditAction] = useState("");
  const [social, setSocial] = useState<SocialPayload | null>(null);
  const [actionName, setActionName] = useState("work_propose_and_execute");
  const [actionBody, setActionBody] = useState(JSON.stringify(ACTION_TEMPLATES.work_propose_and_execute, null, 2));
  const [actionResult, setActionResult] = useState<string | null>(null);
  const [faucet, setFaucet] = useState({ participantId: "demo-user", currency: "twc", units: "1000" });
  const [notice, setNotice] = useState<string | null>(null);

  const loadHealth = useCallback(async () => {
    try {
      const response = await fetch("/api/economy/health");
      setHealth(await response.json());
    } catch {
      setHealth(null);
    }
  }, []);

  const loadState = useCallback(async () => {
    try {
      const response = await fetch("/api/economy/state");
      setState(await response.json());
    } catch {
      setState(null);
    }
  }, []);

  const loadAudit = useCallback(async () => {
    const params = new URLSearchParams({ limit: "300" });
    if (auditPlane) params.set("plane", auditPlane);
    if (auditAction) params.set("action", auditAction);
    try {
      const response = await fetch(`/api/economy/audit?${params.toString()}`);
      setAudit(await response.json());
    } catch {
      setAudit(null);
    }
  }, [auditPlane, auditAction]);

  const loadSummaries = useCallback(async () => {
    try {
      const response = await fetch("/api/economy/scenarios");
      const data = await response.json();
      setSummaries(data.scenarios ?? []);
    } catch {
      setSummaries([]);
    }
  }, []);

  const loadSocial = useCallback(async () => {
    try {
      const response = await fetch("/api/social/status");
      setSocial(await response.json());
    } catch {
      setSocial(null);
    }
  }, []);

  useEffect(() => {
    void loadHealth();
    void loadState();
    void loadSummaries();
    void loadSocial();
    void loadAudit();
  }, [loadHealth, loadState, loadSummaries, loadSocial, loadAudit]);

  async function runScenarios(ids?: string[]) {
    setRunning(true);
    setNotice(null);
    try {
      const response = await fetch("/api/economy/scenarios", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(ids && ids.length > 0 ? { ids } : {}),
      });
      const data = await response.json();
      setSuite(data.suite ?? null);
      setNotice(
        data.suite
          ? `Suite finished: ${data.suite.passed}/${data.suite.total} scenarios, ${data.suite.stepPassed}/${data.suite.stepTotal} steps ${data.suite.ok ? "passed" : "with failures"}.`
          : `Scenario run failed: ${data.error ?? "unknown error"}`,
      );
    } catch (error) {
      setNotice(`Scenario run failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setRunning(false);
    }
  }

  async function runAction() {
    setActionResult(null);
    try {
      const payload = JSON.parse(actionBody);
      const response = await fetch("/api/economy/state", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      setActionResult(JSON.stringify(data, null, 2));
      await loadState();
      await loadAudit();
    } catch (error) {
      setActionResult(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) }, null, 2));
    }
  }

  async function grantFaucet() {
    try {
      const response = await fetch("/api/economy/faucet", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ...faucet, units: Number(faucet.units) }),
      });
      const data = await response.json();
      setNotice(data.ok ? `Faucet granted ${data.granted} ${data.currency} to ${data.participantId}.` : `Faucet refused: ${data.error}`);
      await loadState();
      await loadAudit();
    } catch (error) {
      setNotice(`Faucet failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  async function resetLedger() {
    try {
      const response = await fetch("/api/economy/reset", { method: "POST" });
      const data = await response.json();
      setNotice(data.ok ? "Ledger reset to a fresh engine." : `Reset refused: ${data.error}`);
      await loadState();
      await loadAudit();
      await loadHealth();
    } catch (error) {
      setNotice(`Reset failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  const balances = state?.balances;
  const participants = useMemo(() => state?.participants ?? [], [state]);

  return (
    <div className="etc">
      <header className="etc-header">
        <p className="etc-eyebrow">Economy Test Console</p>
        <h1>
          Complete testing for the
          <span className="etc-accent"> crypto economy</span>
        </h1>
        <p className="etc-lede">
          One surface for the whole four-plane economy: readiness across every rail, the offline
          scenario suite from <code>docs/tri-token-integration-v1.md</code> §7, a valueless faucet,
          every audited mutation, and the Nostr surface your client plugs into. Nothing here touches
          a chain; chain verification stays in the unlock rails.
        </p>
        <div className="etc-tabs">
          {(
            [
              ["readiness", "Readiness"],
              ["scenarios", "Scenarios"],
              ["ledger", "Ledger & faucet"],
              ["actions", "Action runner"],
              ["audit", "Audit trail"],
              ["nostr", "Nostr surface"],
            ] as Array<[Tab, string]>
          ).map(([id, label]) => (
            <button key={id} type="button" className={tab === id ? "etc-tab active" : "etc-tab"} onClick={() => setTab(id)}>
              {label}
            </button>
          ))}
        </div>
        {notice ? <div className="etc-notice">{notice}</div> : null}
      </header>

      {tab === "readiness" ? (
        <section className="etc-section">
          <div className="etc-row">
            <div>
              <h2>Deployment readiness</h2>
              <p>
                Overall: <StatusPill status={health?.readiness === "ready" ? "pass" : health?.readiness === "ready-with-warnings" ? "warn" : "fail"} />{" "}
                <code>{health?.readiness ?? "loading"}</code> · policy <code>{health?.policyVersion ?? "…"}</code> · test mode{" "}
                <code>{health ? String(health.testMode) : "…"}</code>
              </p>
            </div>
            <button type="button" className="etc-btn" onClick={() => { void loadHealth(); void loadState(); }}>
              ↻ Re-check
            </button>
          </div>
          <div className="etc-checks">
            {(health?.checks ?? []).map((check) => (
              <div key={check.id} className="etc-check">
                <StatusPill status={check.status} />
                <div>
                  <strong>{check.label}</strong>
                  <p>{check.detail}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="etc-grid-2">
            <div className="etc-card">
              <h3>$OMEGA rail</h3>
              <ul className="etc-kv">
                <li><span>network</span><code>{health?.rails?.rails?.omega?.network}</code></li>
                <li><span>rpc</span><code>{health?.rails?.rails?.omega?.rpcUrl}</code></li>
                <li><span>gate</span><code>{health?.rails?.rails?.omega?.gateAddress ?? "not configured"}</code></li>
              </ul>
              <p className="etc-fine">{health?.rails?.rails?.omega?.message}</p>
            </div>
            <div className="etc-card">
              <h3>TWC rail</h3>
              <ul className="etc-kv">
                <li><span>network</span><code>{health?.rails?.rails?.twc?.network}</code></li>
                <li><span>rpc</span><code>{health?.rails?.rails?.twc?.rpcUrl}</code></li>
                <li><span>mint</span><code>{health?.rails?.rails?.twc?.mintAddress ?? "not configured"}</code></li>
              </ul>
              <p className="etc-fine">{health?.rails?.rails?.twc?.message}</p>
            </div>
          </div>
          <div className="etc-card">
            <h3>Ledger snapshot</h3>
            <pre className="etc-pre">{JSON.stringify(health?.ledger ?? {}, null, 2)}</pre>
          </div>
        </section>
      ) : null}

      {tab === "scenarios" ? (
        <section className="etc-section">
          <div className="etc-row">
            <div>
              <h2>Scenario suite</h2>
              <p>
                {summaries.length} scenarios covering slices A–G, both unlock rails, and the
                cross-plane invariants. Runs are deterministic and use fresh engines.
              </p>
            </div>
            <div className="etc-btn-row">
              <button type="button" className="etc-btn" disabled={running} onClick={() => void runScenarios()}>
                ▶ Run all
              </button>
              <button
                type="button"
                className="etc-btn secondary"
                disabled={running || selected.length === 0}
                onClick={() => void runScenarios(selected)}
              >
                ▶ Run selected ({selected.length})
              </button>
            </div>
          </div>

          <div className="etc-scenario-list">
            {summaries.map((scenario) => (
              <label key={scenario.id} className="etc-scenario">
                <input
                  type="checkbox"
                  checked={selected.includes(scenario.id)}
                  onChange={(event) =>
                    setSelected((current) =>
                      event.target.checked ? [...current, scenario.id] : current.filter((id) => id !== scenario.id),
                    )
                  }
                />
                <div>
                  <div className="etc-scenario-head">
                    <span className="etc-plane-tag">{scenario.plane}</span>
                    <strong>{scenario.title}</strong>
                    <span className="etc-fine">slice {scenario.slice} · {scenario.stepCount} steps</span>
                  </div>
                  <p>{scenario.description}</p>
                </div>
              </label>
            ))}
          </div>

          {suite ? (
            <div className="etc-suite">
              <div className={suite.ok ? "etc-suite-banner ok" : "etc-suite-banner bad"}>
                {suite.ok ? "✓" : "✕"} {suite.passed}/{suite.total} scenarios · {suite.stepPassed}/{suite.stepTotal} steps ·{" "}
                {suite.durationMs} ms · ran {new Date(suite.ranAt).toLocaleTimeString()}
              </div>
              {suite.results.map((result) => (
                <details key={result.id} className="etc-result" open={!result.passed}>
                  <summary>
                    <span className={result.passed ? "etc-pill ok" : "etc-pill bad"}>{result.passed ? "pass" : "fail"}</span>
                    <strong>{result.title}</strong>
                    <span className="etc-fine">{result.steps.filter((s) => s.passed).length}/{result.steps.length} steps · {result.durationMs} ms · {result.auditEventCount} audit events</span>
                  </summary>
                  <ul className="etc-steps">
                    {result.steps.map((step) => (
                      <li key={step.id} className={step.passed ? "step ok" : "step bad"}>
                        <span className="etc-step-mark">{step.passed ? "✓" : "✕"}</span>
                        <div>
                          <strong>{step.title}</strong>
                          {!step.passed ? (
                            <p className="etc-fine">
                              expected: {step.expected}
                              <br />
                              actual: {step.actual}
                            </p>
                          ) : null}
                        </div>
                      </li>
                    ))}
                  </ul>
                </details>
              ))}
            </div>
          ) : (
            <p className="etc-fine">No run yet. Press “Run all” to execute the entire suite server-side.</p>
          )}
        </section>
      ) : null}

      {tab === "ledger" ? (
        <section className="etc-section">
          <div className="etc-row">
            <div>
              <h2>Ledger &amp; faucet</h2>
              <p>
                The shared in-memory engine. Test mode: <code>{state?.testMode === undefined ? "…" : String(state.testMode)}</code>.
                Faucet grants are valueless and fully audited.
              </p>
            </div>
            <button type="button" className="etc-btn danger" onClick={() => void resetLedger()}>
              ⟲ Reset ledger
            </button>
          </div>

          <div className="etc-card">
            <h3>Faucet</h3>
            <div className="etc-form-row">
              <label>
                participant
                <input value={faucet.participantId} onChange={(e) => setFaucet((f) => ({ ...f, participantId: e.target.value }))} />
              </label>
              <label>
                currency
                <select value={faucet.currency} onChange={(e) => setFaucet((f) => ({ ...f, currency: e.target.value }))}>
                  <option value="care">CARE</option>
                  <option value="twc">TWC</option>
                  <option value="omega">$OMEGA</option>
                  <option value="amity">AMITY</option>
                </select>
              </label>
              <label>
                units
                <input value={faucet.units} onChange={(e) => setFaucet((f) => ({ ...f, units: e.target.value }))} />
              </label>
              <button type="button" className="etc-btn" onClick={() => void grantFaucet()}>
                ⛲ Grant
              </button>
            </div>
          </div>

          <div className="etc-grid-2">
            <div className="etc-card">
              <h3>Participants</h3>
              <div className="etc-table-wrap">
                <table className="etc-table">
                  <thead><tr><th>id</th><th>role</th><th>privacy</th><th>status</th></tr></thead>
                  <tbody>
                    {participants.map((participant) => (
                      <tr key={participant.participantId}>
                        <td><code>{participant.participantId}</code></td>
                        <td>{participant.role}</td>
                        <td>{participant.privacyLevel}</td>
                        <td>{participant.status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="etc-card">
              <h3>Balances</h3>
              <div className="etc-table-wrap">
                <table className="etc-table">
                  <thead><tr><th>participant</th><th>CARE</th><th>TWC</th><th>$OMEGA</th><th>AMITY</th></tr></thead>
                  <tbody>
                    {participants.map((participant) => (
                      <tr key={participant.participantId}>
                        <td><code>{participant.participantId}</code></td>
                        <td>{balances?.care?.[participant.participantId] ?? 0}</td>
                        <td>{balances?.twc?.[participant.participantId] ?? 0}</td>
                        <td>{balances?.omega?.[participant.participantId] ?? 0}</td>
                        <td>{balances?.amity?.[participant.participantId] ?? 0}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="etc-grid-2">
            <div className="etc-card">
              <h3>Work receipts &amp; appeals</h3>
              <ul className="etc-mini-list">
                {(state?.receipts ?? []).slice(-8).reverse().map((receipt) => (
                  <li key={receipt.workId}>
                    <code>{receipt.workId}</code> · {receipt.workClass} · +{receipt.issuedTwcUnits} TWC ·{" "}
                    <span className={receipt.status === "settled" ? "etc-pill ok" : "etc-pill bad"}>{receipt.status}</span>
                  </li>
                ))}
                {(state?.appeals ?? []).slice(-5).reverse().map((appeal) => (
                  <li key={appeal.appealId}>
                    <code>{appeal.appealId}</code> on <code>{appeal.workId}</code> · {appeal.status}
                  </li>
                ))}
                {(state?.receipts ?? []).length === 0 && (state?.appeals ?? []).length === 0 ? <li className="etc-fine">No receipts yet.</li> : null}
              </ul>
            </div>
            <div className="etc-card">
              <h3>Conversions, hardships, governance</h3>
              <ul className="etc-mini-list">
                {(state?.conversions ?? []).slice(-5).reverse().map((conversion) => (
                  <li key={conversion.conversionId}>
                    <code>{conversion.participantId}</code> burned {conversion.careAmount} CARE → {conversion.netAmityAmount} AMITY
                  </li>
                ))}
                {(state?.hardships ?? []).slice(-5).reverse().map((hardship) => (
                  <li key={hardship.requestId}>
                    <code>{hardship.requestId}</code> · {hardship.participantId} · {hardship.status}
                  </li>
                ))}
                {(state?.governance ?? []).slice(-5).reverse().map((proposal) => (
                  <li key={proposal.proposalId}>
                    <code>{proposal.proposalId}</code> · {proposal.title} · {proposal.status}
                  </li>
                ))}
                {(state?.conversions ?? []).length + (state?.hardships ?? []).length + (state?.governance ?? []).length === 0 ? (
                  <li className="etc-fine">Nothing yet — use the action runner.</li>
                ) : null}
              </ul>
            </div>
          </div>
        </section>
      ) : null}

      {tab === "actions" ? (
        <section className="etc-section">
          <div className="etc-row">
            <div>
              <h2>Action runner</h2>
              <p>Every mutation the shared engine accepts, as editable JSON. Failures return the fail-closed error.</p>
            </div>
          </div>
          <div className="etc-card">
            <div className="etc-form-row">
              <label>
                action
                <select
                  value={actionName}
                  onChange={(event) => {
                    setActionName(event.target.value);
                    setActionBody(JSON.stringify(ACTION_TEMPLATES[event.target.value], null, 2));
                  }}
                >
                  {Object.keys(ACTION_TEMPLATES).map((name) => (
                    <option key={name} value={name}>{name}</option>
                  ))}
                </select>
              </label>
              <button type="button" className="etc-btn" onClick={() => void runAction()}>
                ⚡ Execute
              </button>
            </div>
            <textarea className="etc-textarea" rows={10} value={actionBody} onChange={(event) => setActionBody(event.target.value)} />
            {actionResult ? <pre className="etc-pre">{actionResult}</pre> : null}
          </div>
          <div className="etc-card">
            <h3>curl recipes</h3>
            <pre className="etc-pre">{`# run the whole scenario suite
curl -sX POST localhost:3000/api/economy/scenarios -H 'content-type: application/json' -d '{}'

# grant test balances
curl -sX POST localhost:3000/api/economy/faucet -H 'content-type: application/json' \\
  -d '{"participantId":"ana","currency":"twc","units":1000}'

# settle a piece of useful work
curl -sX POST localhost:3000/api/economy/state -H 'content-type: application/json' \\
  -d '{"action":"work_propose_and_execute","contributorCommitment":"ana","workClass":"zk_proving","resourceBudget":120}'

# export the audit trail
curl -s 'localhost:3000/api/economy/audit?format=csv' -o audit.csv`}</pre>
          </div>
        </section>
      ) : null}

      {tab === "audit" ? (
        <section className="etc-section">
          <div className="etc-row">
            <div>
              <h2>Audit trail</h2>
              <p>
                Showing {audit?.count ?? 0} of {audit?.total ?? 0} events. Every balance mutation in
                this economy is recorded with its policy version, actor, and input/output
                commitments.
              </p>
            </div>
            <div className="etc-btn-row">
              <a className="etc-btn secondary" href={`/api/economy/audit?format=csv&limit=10000${auditPlane ? `&plane=${auditPlane}` : ""}`}>⬇ CSV</a>
              <a className="etc-btn secondary" href={`/api/economy/audit?format=ndjson&limit=10000${auditPlane ? `&plane=${auditPlane}` : ""}`}>⬇ NDJSON</a>
              <button type="button" className="etc-btn" onClick={() => void loadAudit()}>↻ Refresh</button>
            </div>
          </div>
          <div className="etc-form-row">
            <label>
              plane
              <select value={auditPlane} onChange={(event) => setAuditPlane(event.target.value)}>
                <option value="">all</option>
                {(audit?.planes ?? ["CARE", "TWC", "OMEGA", "AMITY"]).map((plane) => (
                  <option key={plane} value={plane}>{plane}</option>
                ))}
              </select>
            </label>
            <label>
              action
              <select value={auditAction} onChange={(event) => setAuditAction(event.target.value)}>
                <option value="">all</option>
                {(audit?.actions ?? []).map((action) => (
                  <option key={action} value={action}>{action}</option>
                ))}
              </select>
            </label>
          </div>
          <div className="etc-table-wrap">
            <table className="etc-table">
              <thead>
                <tr><th>time</th><th>plane</th><th>action</th><th>actor</th><th>inputs → outputs</th></tr>
              </thead>
              <tbody>
                {(audit?.events ?? []).slice().reverse().map((event) => (
                  <tr key={event.eventId}>
                    <td className="etc-fine">{new Date(event.timestamp).toLocaleTimeString()}</td>
                    <td><span className="etc-plane-tag">{event.plane}</span></td>
                    <td><code>{event.action}</code></td>
                    <td><code>{event.actorCommitment}</code></td>
                    <td className="etc-fine">
                      {event.inputCommitments.join(", ")} → {event.outputCommitments.join(", ")}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {tab === "nostr" ? (
        <section className="etc-section">
          <div className="etc-row">
            <div>
              <h2>Nostr surface</h2>
              <p>
                The economy&apos;s social layer rides on the Nostr client you are building. This is
                the contract it plugs into — status:{" "}
                <StatusPill status={social?.status === "ready" ? "pass" : "not_configured"} />
              </p>
            </div>
          </div>
          <div className="etc-grid-2">
            <div className="etc-card">
              <h3>Configuration</h3>
              <ul className="etc-kv">
                <li><span>relays</span><code>{social?.relays.length ? social.relays.join(", ") : "none (set NOSTR_RELAYS)"}</code></li>
                <li><span>publisher npub</span><code>{social?.publisherNpub ?? "user-signed (set NOSTR_PUBLISHER_NPUB to override)"}</code></li>
              </ul>
              <h3>Website consumers</h3>
              <ul className="etc-mini-list">
                {(social?.consumers ?? []).map((consumer) => (
                  <li key={consumer.route}><code>{consumer.route}</code> — {consumer.uses}</li>
                ))}
              </ul>
            </div>
            <div className="etc-card">
              <h3>Economy event kinds</h3>
              <ul className="etc-kv">
                {Object.entries(social?.kinds ?? {}).map(([name, kind]) => (
                  <li key={name}><span>{name}</span><code>{kind}</code></li>
                ))}
              </ul>
            </div>
          </div>
          <div className="etc-card">
            <h3>What the client must do</h3>
            <ol className="etc-mini-list ordered">
              {(social?.requiredFromClient ?? []).map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ol>
            <ul className="etc-mini-list">
              {(social?.notes ?? []).map((note) => (
                <li key={note} className="etc-fine">{note}</li>
              ))}
            </ul>
          </div>
        </section>
      ) : null}

      <style>{`
        .etc { max-width: 1180px; margin: 0 auto; padding: 120px 24px 96px; }
        .etc-eyebrow { color: var(--color-unlock); font: 600 12px ui-monospace, monospace; letter-spacing: .2em; text-transform: uppercase; margin: 0 0 20px; }
        .etc h1 { font-family: ui-serif, Georgia, serif; font-size: clamp(36px, 5.5vw, 64px); line-height: 1.06; font-weight: 500; letter-spacing: -0.025em; margin: 0 0 20px; max-width: 900px; }
        .etc-accent { color: var(--color-unlock); }
        .etc-lede { color: var(--color-muted-strong); font-size: clamp(15px, 1.5vw, 18px); line-height: 1.7; max-width: 780px; margin: 0 0 30px; }
        .etc-lede code, .etc code { background: rgba(255,255,255,0.06); padding: 1px 5px; border-radius: 4px; font-size: 12px; }
        .etc-tabs { display: flex; gap: 8px; flex-wrap: wrap; border-bottom: 1px solid var(--color-border); padding-bottom: 14px; }
        .etc-tab { border: 1px solid var(--color-border-strong); background: transparent; color: var(--color-muted-strong); border-radius: 8px; padding: 8px 14px; font: 600 12px ui-monospace, monospace; cursor: pointer; }
        .etc-tab.active { color: var(--color-foreground); border-color: var(--color-accent); background: rgba(192,132,87,0.12); }
        .etc-notice { margin-top: 14px; padding: 10px 14px; border: 1px solid var(--color-border-strong); border-radius: 10px; background: rgba(96,165,250,0.08); color: var(--color-muted-strong); font-size: 13px; }

        .etc-section { margin-top: 34px; display: flex; flex-direction: column; gap: 18px; }
        .etc-row { display: flex; justify-content: space-between; align-items: flex-end; gap: 18px; flex-wrap: wrap; }
        .etc-row h2 { font-family: ui-serif, Georgia, serif; font-size: 26px; font-weight: 500; margin: 0 0 6px; }
        .etc-row p { color: var(--color-muted-strong); font-size: 13.5px; margin: 0; max-width: 640px; }
        .etc-btn-row { display: flex; gap: 10px; flex-wrap: wrap; }
        .etc-btn { border: 1px solid var(--color-accent); background: rgba(192,132,87,0.14); color: var(--color-accent-soft); border-radius: 8px; padding: 9px 16px; font: 700 12px ui-monospace, monospace; cursor: pointer; text-decoration: none; }
        .etc-btn:hover { background: rgba(192,132,87,0.24); }
        .etc-btn:disabled { opacity: .5; cursor: not-allowed; }
        .etc-btn.secondary { border-color: var(--color-border-strong); background: transparent; color: var(--color-muted-strong); }
        .etc-btn.danger { border-color: rgba(244,114,182,0.6); background: rgba(244,114,182,0.1); color: var(--color-care); }

        .etc-pill { font: 700 10px ui-monospace, monospace; letter-spacing: .08em; text-transform: uppercase; padding: 3px 9px; border-radius: 999px; }
        .etc-pill.ok { background: rgba(52,211,153,0.14); color: var(--color-unlock); }
        .etc-pill.warn { background: rgba(251,191,36,0.14); color: var(--color-omega); }
        .etc-pill.bad { background: rgba(244,114,182,0.14); color: var(--color-care); }

        .etc-checks { display: grid; gap: 10px; }
        .etc-check { display: flex; gap: 14px; align-items: flex-start; border: 1px solid var(--color-border); border-radius: 12px; padding: 14px 16px; background: rgba(255,255,255,0.015); }
        .etc-check strong { font-size: 14px; }
        .etc-check p { margin: 4px 0 0; color: var(--color-muted); font-size: 12.5px; line-height: 1.55; }

        .etc-grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
        .etc-card { border: 1px solid var(--color-border); border-radius: 14px; padding: 18px 20px; background: rgba(255,255,255,0.015); }
        .etc-card h3 { font-size: 14px; font-weight: 600; margin: 0 0 12px; letter-spacing: .04em; text-transform: uppercase; color: var(--color-muted-strong); font-family: ui-monospace, monospace; }
        .etc-kv { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
        .etc-kv li { display: flex; justify-content: space-between; gap: 14px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 7px; font-size: 12.5px; }
        .etc-kv li span { color: var(--color-muted); }
        .etc-fine { color: var(--color-muted); font-size: 11.5px; line-height: 1.5; }

        .etc-pre { margin: 0; padding: 12px 14px; background: rgba(0,0,0,0.4); border: 1px solid var(--color-border); border-radius: 10px; color: var(--color-unlock-soft); font: 12px ui-monospace, monospace; overflow-x: auto; max-height: 420px; }
        .etc-textarea { width: 100%; margin-top: 12px; padding: 12px 14px; background: rgba(0,0,0,0.4); border: 1px solid var(--color-border-strong); border-radius: 10px; color: var(--color-foreground); font: 12.5px ui-monospace, monospace; resize: vertical; }
        .etc-form-row { display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-end; }
        .etc-form-row label { display: flex; flex-direction: column; gap: 6px; color: var(--color-muted); font: 600 10px ui-monospace, monospace; text-transform: uppercase; letter-spacing: .1em; }
        .etc-form-row input, .etc-form-row select { border: 1px solid var(--color-border-strong); border-radius: 8px; padding: 9px 12px; background: rgba(0,0,0,0.35); color: var(--color-foreground); font: 13px ui-monospace, monospace; min-width: 150px; }

        .etc-scenario-list { display: grid; gap: 10px; }
        .etc-scenario { display: flex; gap: 12px; align-items: flex-start; border: 1px solid var(--color-border); border-radius: 12px; padding: 14px 16px; cursor: pointer; background: rgba(255,255,255,0.015); }
        .etc-scenario input { margin-top: 3px; accent-color: var(--color-unlock); }
        .etc-scenario p { margin: 6px 0 0; color: var(--color-muted); font-size: 12.5px; line-height: 1.55; }
        .etc-scenario-head { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
        .etc-plane-tag { font: 700 10px ui-monospace, monospace; padding: 2px 8px; border-radius: 5px; background: rgba(96,165,250,0.14); color: var(--color-use); }

        .etc-suite { display: flex; flex-direction: column; gap: 10px; }
        .etc-suite-banner { padding: 12px 16px; border-radius: 10px; font: 700 13px ui-monospace, monospace; }
        .etc-suite-banner.ok { background: rgba(52,211,153,0.12); color: var(--color-unlock); border: 1px solid rgba(52,211,153,0.4); }
        .etc-suite-banner.bad { background: rgba(244,114,182,0.12); color: var(--color-care); border: 1px solid rgba(244,114,182,0.4); }
        .etc-result { border: 1px solid var(--color-border); border-radius: 12px; padding: 10px 14px; background: rgba(255,255,255,0.015); }
        .etc-result summary { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; cursor: pointer; }
        .etc-steps { list-style: none; margin: 12px 0 4px; padding: 0; display: grid; gap: 8px; }
        .etc-steps .step { display: flex; gap: 10px; align-items: flex-start; }
        .etc-step-mark { font-weight: 700; }
        .etc-steps .step.ok .etc-step-mark { color: var(--color-unlock); }
        .etc-steps .step.bad .etc-step-mark { color: var(--color-care); }
        .etc-steps .step p { margin: 4px 0 0; }

        .etc-table-wrap { overflow-x: auto; }
        .etc-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
        .etc-table th { text-align: left; padding: 8px 10px; color: var(--color-muted); font: 600 10px ui-monospace, monospace; text-transform: uppercase; letter-spacing: .08em; border-bottom: 1px solid var(--color-border); }
        .etc-table td { padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.04); color: var(--color-muted-strong); }
        .etc-mini-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; font-size: 12.5px; color: var(--color-muted-strong); }
        .etc-mini-list.ordered { list-style: decimal; padding-left: 18px; }
      `}</style>
    </div>
  );
}
