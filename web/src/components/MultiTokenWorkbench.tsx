// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
"use client";

import { useEffect, useState } from "react";

type PlaneTab = "overview" | "amity" | "twc" | "omega";

export function MultiTokenWorkbench() {
  const [activeTab, setActiveTab] = useState<PlaneTab>("overview");
  const [state, setState] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  // Form states
  const [workClass, setWorkClass] = useState("lean_formalization");
  const [workObjective, setWorkObjective] = useState("Verify Lean 4 APPA tactic & sorry audit");
  const [workBudget, setWorkBudget] = useState(150);

  const [convertAmount, setConvertAmount] = useState(500);
  const [extremeHoldback, setExtremeHoldback] = useState(false);

  const [govTitle, setGovTitle] = useState("Bound AMITY conversion ceiling to 3500 bps");
  const [govTarget, setGovTarget] = useState("AMITY");

  async function fetchState() {
    try {
      const res = await fetch("/api/economy/state");
      const data = await res.json();
      if (data.ok) setState(data);
    } catch (e) {
      console.error(e);
    }
  }

  useEffect(() => {
    fetchState();
  }, []);

  async function callAction(action: string, payload: any) {
    setLoading(true);
    setMessage(null);
    try {
      const res = await fetch("/api/economy/state", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, ...payload }),
      });
      const data = await res.json();
      if (data.ok) {
        setMessage(`Action '${action}' succeeded!`);
        await fetchState();
      } else {
        setMessage(`Action failed: ${data.error}`);
      }
    } catch (err: any) {
      setMessage(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  const userCare = state?.balances?.care?.["demo-user"] ?? 0;
  const userTwc = state?.balances?.twc?.["demo-user"] ?? 0;
  const userOmega = state?.balances?.omega?.["demo-user"] ?? 0;
  const userAmity = state?.balances?.amity?.["demo-user"] ?? 0;

  return (
    <div className="workbench-shell">
      <div className="workbench-header">
        <div>
          <span className="workbench-badge">Tri-Token & Cross-Plane Maturity</span>
          <h2>Integrated Economic Workbench</h2>
          <p>
            Simulate and verify the 4 planes (C.A.R.E., TWC, $OMEGA, AMITY) running under
            shared domain state machines, verifier adapters, and fail-closed policies.
          </p>
        </div>
        <div className="workbench-metrics">
          <div className="metric-pill">
            <span className="dot care" />
            <span className="label">CARE:</span>
            <strong>{userCare}</strong>
          </div>
          <div className="metric-pill">
            <span className="dot twc" />
            <span className="label">TWC:</span>
            <strong>{userTwc}</strong>
          </div>
          <div className="metric-pill">
            <span className="dot omega" />
            <span className="label">$OMEGA:</span>
            <strong>{userOmega}</strong>
          </div>
          <div className="metric-pill">
            <span className="dot amity" />
            <span className="label">AMITY:</span>
            <strong>{userAmity}</strong>
          </div>
        </div>
      </div>

      <div className="workbench-tabs">
        <button
          type="button"
          className={activeTab === "overview" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("overview")}
        >
          Integration Overview
        </button>
        <button
          type="button"
          className={activeTab === "amity" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("amity")}
        >
          AMITY / Taproot Plane
        </button>
        <button
          type="button"
          className={activeTab === "twc" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("twc")}
        >
          TWC / Useful Work Plane
        </button>
        <button
          type="button"
          className={activeTab === "omega" ? "tab-btn active" : "tab-btn"}
          onClick={() => setActiveTab("omega")}
        >
          $OMEGA / Governance Plane
        </button>
      </div>

      {message && (
        <div className="workbench-alert">
          <span>{message}</span>
          <button type="button" onClick={() => setMessage(null)}>×</button>
        </div>
      )}

      <div className="workbench-body">
        {activeTab === "overview" && (
          <div className="tab-pane">
            <div className="grid-2">
              <div className="panel">
                <h3>Slice A & B: Social Reach & Resource Quota</h3>
                <p>
                  Wallet contributes user-selected device resource units. Missed quota alters activity
                  state without deleting identity or history.
                </p>
                <div className="btn-row">
                  <button
                    className="action-btn"
                    disabled={loading}
                    onClick={() => callAction("reach_record", { participantId: "demo-user", audiences: 5, attentions: 12 })}
                  >
                    + Log Reach Event (5 aud, 12 att)
                  </button>
                  <button
                    className="action-btn"
                    disabled={loading}
                    onClick={() => callAction("distribute_pool", {})}
                  >
                    Settle Weekly Pool Distribution
                  </button>
                </div>
              </div>

              <div className="panel">
                <h3>Slice C: Hardship Solidarity (No Debt)</h3>
                <p>
                  Hardship is unbounded in duration. Community sponsors fulfill quota deficits and
                  receive a 10% useful-work TWC bonus.
                </p>
                <div className="btn-row">
                  <button
                    className="action-btn secondary"
                    disabled={loading}
                    onClick={() => callAction("hardship_request", { participantId: "demo-user", uncoveredUnits: 50 })}
                  >
                    Request Hardship Coverage
                  </button>
                </div>
              </div>
            </div>

            <div className="panel" style={{ marginTop: "16px" }}>
              <h3>Append-Only Audit Ledger (Cross-Plane Commitments)</h3>
              <div className="audit-table-wrapper">
                <table className="audit-table">
                  <thead>
                    <tr>
                      <th>Plane</th>
                      <th>Action</th>
                      <th>Actor</th>
                      <th>Inputs</th>
                      <th>Outputs</th>
                      <th>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(state?.recentEvents || []).slice(-6).map((evt: any) => (
                      <tr key={evt.eventId}>
                        <td><span className={`plane-tag ${evt.plane.toLowerCase()}`}>{evt.plane}</span></td>
                        <td>{evt.action}</td>
                        <td><code>{evt.actorCommitment}</code></td>
                        <td>{evt.inputCommitments.join(", ") || "-"}</td>
                        <td>{evt.outputCommitments.join(", ") || "-"}</td>
                        <td>{new Date(evt.timestamp).toLocaleTimeString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === "amity" && (
          <div className="tab-pane">
            <div className="grid-2">
              <div className="panel">
                <h3>Slice D: CARE to AMITY Bounded Conversion</h3>
                <p>
                  CARE is converted to market-facing AMITY under strict policy bounds:
                  5% participant dock, optional 15% extreme unissued holdback, ceiling 35%.
                  No base supply burn.
                </p>
                <div className="form-group">
                  <label>CARE Amount to Convert:</label>
                  <input
                    type="number"
                    value={convertAmount}
                    onChange={(e) => setConvertAmount(Number(e.target.value))}
                    max={userCare}
                    min={10}
                  />
                </div>
                <div className="form-group checkbox">
                  <label>
                    <input
                      type="checkbox"
                      checked={extremeHoldback}
                      onChange={(e) => setExtremeHoldback(e.target.checked)}
                    />
                    Apply Extreme Holdback (1500 bps / 15%)
                  </label>
                </div>
                <button
                  className="action-btn"
                  disabled={loading || userCare < convertAmount}
                  onClick={() =>
                    callAction("care_amity_convert", {
                      participantId: "demo-user",
                      careAmount: convertAmount,
                      extremeHoldback,
                    })
                  }
                >
                  Execute Bounded Conversion
                </button>
              </div>

              <div className="panel">
                <h3>AMITY Testnet Scaffold Verification</h3>
                <p>
                  Bitcoin / Lightning / Taproot holder proof verification operates fail-closed with
                  strict Taproot Asset IDs and canonical challenges.
                </p>
                <ul className="info-list">
                  <li><strong>Status:</strong> Testnet Scaffold / Offline Fixture Mode</li>
                  <li><strong>Dock Rate:</strong> 500 bps (5%)</li>
                  <li><strong>Max Combined Dock:</strong> 3500 bps (35%)</li>
                  <li><strong>Asset Gate:</strong> Separate from live EVM/Solana unlock until mainnet safety audit</li>
                </ul>
              </div>
            </div>
          </div>
        )}

        {activeTab === "twc" && (
          <div className="tab-pane">
            <div className="grid-2">
              <div className="panel">
                <h3>Slice E: TWC Proof-of-Useful-Work</h3>
                <p>
                  Submit work proposals across engineering, Lean formalization, ZK circuits,
                  and infrastructure. Adapters verify artifacts before issuing TWC settlement units.
                </p>
                <div className="form-group">
                  <label>Work Class:</label>
                  <select value={workClass} onChange={(e) => setWorkClass(e.target.value)}>
                    <option value="engineering_protocol">Engineering & Protocol (Build & Test)</option>
                    <option value="lean_formalization">Lean Formalization (Kernel Checked)</option>
                    <option value="zk_proving">ZK Circuits & Proving</option>
                    <option value="infrastructure_resource">Infrastructure & Relay Uptime</option>
                    <option value="physics_simulation">Physics & Simulation Model</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Objective:</label>
                  <input
                    type="text"
                    value={workObjective}
                    onChange={(e) => setWorkObjective(e.target.value)}
                  />
                </div>
                <div className="form-group">
                  <label>Resource Budget (units):</label>
                  <input
                    type="number"
                    value={workBudget}
                    onChange={(e) => setWorkBudget(Number(e.target.value))}
                  />
                </div>
                <button
                  className="action-btn"
                  disabled={loading}
                  onClick={() =>
                    callAction("work_propose_and_execute", {
                      contributorCommitment: "demo-user",
                      workClass,
                      objective: workObjective,
                      resourceBudget: workBudget,
                    })
                  }
                >
                  Submit & Verify Useful Work
                </button>
              </div>

              <div className="panel">
                <h3>Recent Work Receipts</h3>
                <div className="receipts-list">
                  {(state?.receipts || []).slice(-4).map((r: any) => (
                    <div key={r.workId} className="receipt-card">
                      <div className="receipt-head">
                        <strong>{r.workClass}</strong>
                        <span className="award">+{r.issuedTwcUnits} TWC</span>
                      </div>
                      <p className="receipt-hash">Artifact: {r.artifactHash}</p>
                      <small>Adapter: {r.verificationResult?.adapterName} (Verified)</small>
                    </div>
                  ))}
                  {(!state?.receipts || state.receipts.length === 0) && (
                    <p className="empty-text">No useful-work receipts submitted yet.</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "omega" && (
          <div className="tab-pane">
            <div className="grid-2">
              <div className="panel">
                <h3>Slice F: OMEGA Governance & Timelock</h3>
                <p>
                  Proposals undergo simulation verification, peer review, and timelocked delay
                  before execution can mutate plane parameters.
                </p>
                <div className="form-group">
                  <label>Target Plane:</label>
                  <select value={govTarget} onChange={(e) => setGovTarget(e.target.value)}>
                    <option value="AMITY">AMITY</option>
                    <option value="TWC">TWC</option>
                    <option value="CARE">CARE</option>
                    <option value="OMEGA">OMEGA</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>Policy Title:</label>
                  <input
                    type="text"
                    value={govTitle}
                    onChange={(e) => setGovTitle(e.target.value)}
                  />
                </div>
                <button
                  className="action-btn"
                  disabled={loading}
                  onClick={() =>
                    callAction("governance_propose", {
                      proposerCommitment: "demo-user",
                      targetPlane: govTarget,
                      title: govTitle,
                      description: "Steward timelock proposal simulation",
                      diff: { updatedBps: 2000 },
                      timelockSeconds: 5,
                    })
                  }
                >
                  Submit Timelocked Proposal (5s timelock)
                </button>
              </div>

              <div className="panel">
                <h3>Governance Proposals</h3>
                <div className="proposals-list">
                  {(state?.governance || []).slice(-3).map((gov: any) => (
                    <div key={gov.proposalId} className="gov-card">
                      <div className="gov-head">
                        <strong>{gov.title}</strong>
                        <span className={`status-tag ${gov.status}`}>{gov.status}</span>
                      </div>
                      <p>{gov.description}</p>
                      {gov.status === "queued" && (
                        <button
                          className="action-btn small"
                          onClick={() =>
                            callAction("governance_execute", {
                              proposalId: gov.proposalId,
                              timestamp: Date.now() + 10000,
                            })
                          }
                        >
                          Execute After Timelock
                        </button>
                      )}
                    </div>
                  ))}
                  {(!state?.governance || state.governance.length === 0) && (
                    <p className="empty-text">No governance proposals submitted yet.</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .workbench-shell {
          border: 1px solid var(--color-border);
          border-radius: 16px;
          background: rgba(255, 255, 255, 0.02);
          padding: 28px;
          margin: 32px 0;
        }
        .workbench-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          flex-wrap: wrap;
          gap: 20px;
          border-bottom: 1px solid var(--color-border);
          padding-bottom: 24px;
        }
        .workbench-badge {
          color: var(--color-accent);
          font: 600 11px/1 ui-monospace, SFMono-Regular, monospace;
          letter-spacing: 0.15em;
          text-transform: uppercase;
        }
        .workbench-header h2 {
          font: 500 28px/1.2 ui-serif, Georgia, serif;
          margin: 8px 0;
          color: var(--color-foreground);
        }
        .workbench-header p {
          color: var(--color-muted-strong);
          font-size: 14px;
          max-width: 620px;
          margin: 0;
        }
        .workbench-metrics {
          display: flex;
          gap: 12px;
          flex-wrap: wrap;
        }
        .metric-pill {
          display: flex;
          align-items: center;
          gap: 6px;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid var(--color-border);
          border-radius: 8px;
          padding: 8px 12px;
          font-size: 13px;
        }
        .metric-pill .dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
        }
        .dot.care { background: var(--color-care); }
        .dot.twc { background: var(--color-twc); }
        .dot.omega { background: var(--color-omega); }
        .dot.amity { background: var(--color-amity); }
        .metric-pill .label { color: var(--color-muted); }
        .metric-pill strong { color: var(--color-foreground); }

        .workbench-tabs {
          display: flex;
          gap: 8px;
          margin: 20px 0;
          border-bottom: 1px solid var(--color-border);
          padding-bottom: 12px;
        }
        .tab-btn {
          background: transparent;
          border: 1px solid transparent;
          border-radius: 8px;
          padding: 8px 14px;
          color: var(--color-muted-strong);
          font: 600 12px ui-monospace, SFMono-Regular, monospace;
          cursor: pointer;
          transition: all 0.15s ease;
        }
        .tab-btn:hover {
          color: var(--color-foreground);
          background: rgba(255, 255, 255, 0.04);
        }
        .tab-btn.active {
          color: var(--color-foreground);
          background: rgba(192, 132, 87, 0.14);
          border-color: rgba(192, 132, 87, 0.3);
        }

        .workbench-alert {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 10px 14px;
          border-radius: 8px;
          background: rgba(52, 211, 153, 0.08);
          border: 1px solid rgba(52, 211, 153, 0.3);
          color: var(--color-unlock-soft);
          font-size: 13px;
          margin-bottom: 16px;
        }
        .workbench-alert button {
          background: none;
          border: none;
          color: inherit;
          font-size: 18px;
          cursor: pointer;
        }

        .grid-2 {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
          gap: 16px;
        }
        .panel {
          border: 1px solid var(--color-border);
          border-radius: 12px;
          background: rgba(0, 0, 0, 0.2);
          padding: 18px;
        }
        .panel h3 {
          font: 500 18px ui-serif, Georgia, serif;
          margin: 0 0 8px;
          color: var(--color-foreground);
        }
        .panel p {
          color: var(--color-muted-strong);
          font-size: 13px;
          line-height: 1.5;
          margin: 0 0 16px;
        }

        .btn-row {
          display: flex;
          gap: 10px;
          flex-wrap: wrap;
        }
        .action-btn {
          background: var(--color-accent);
          color: #120b10;
          border: none;
          border-radius: 8px;
          padding: 8px 14px;
          font: 600 12px ui-monospace, SFMono-Regular, monospace;
          cursor: pointer;
        }
        .action-btn.secondary {
          background: rgba(255, 255, 255, 0.08);
          color: var(--color-foreground);
          border: 1px solid var(--color-border);
        }
        .action-btn.small {
          padding: 4px 10px;
          font-size: 11px;
        }
        .action-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .form-group {
          margin-bottom: 12px;
        }
        .form-group label {
          display: block;
          color: var(--color-muted);
          font: 600 11px ui-monospace, SFMono-Regular, monospace;
          margin-bottom: 4px;
          text-transform: uppercase;
        }
        .form-group input, .form-group select {
          width: 100%;
          border: 1px solid var(--color-border);
          border-radius: 6px;
          padding: 8px 10px;
          background: rgba(0, 0, 0, 0.3);
          color: var(--color-foreground);
          font-size: 13px;
        }
        .form-group.checkbox label {
          display: flex;
          align-items: center;
          gap: 8px;
          text-transform: none;
          cursor: pointer;
        }

        .audit-table-wrapper {
          overflow-x: auto;
        }
        .audit-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 12px;
          text-align: left;
        }
        .audit-table th {
          padding: 8px;
          border-bottom: 1px solid var(--color-border);
          color: var(--color-muted);
          font-weight: 500;
        }
        .audit-table td {
          padding: 8px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.04);
          color: var(--color-muted-strong);
        }
        .plane-tag {
          padding: 2px 6px;
          border-radius: 4px;
          font: 600 10px ui-monospace, monospace;
        }
        .plane-tag.care { background: rgba(244, 114, 182, 0.15); color: var(--color-care); }
        .plane-tag.twc { background: rgba(56, 189, 248, 0.15); color: var(--color-twc); }
        .plane-tag.omega { background: rgba(251, 191, 36, 0.15); color: var(--color-omega); }
        .plane-tag.amity { background: rgba(192, 132, 252, 0.15); color: var(--color-amity); }

        .info-list {
          list-style: none;
          padding: 0;
          margin: 0;
          font-size: 13px;
        }
        .info-list li {
          padding: 6px 0;
          border-bottom: 1px solid rgba(255, 255, 255, 0.04);
          color: var(--color-muted-strong);
        }

        .receipts-list, .proposals-list {
          display: grid;
          gap: 10px;
        }
        .receipt-card, .gov-card {
          border: 1px solid var(--color-border);
          border-radius: 8px;
          padding: 10px 12px;
          background: rgba(0, 0, 0, 0.2);
        }
        .receipt-head, .gov-head {
          display: flex;
          justify-content: space-between;
          font-size: 12px;
          margin-bottom: 4px;
        }
        .award {
          color: var(--color-twc);
          font-weight: 600;
        }
        .receipt-hash {
          font: 11px ui-monospace, monospace;
          color: var(--color-muted);
          margin: 0 0 4px;
        }
        .status-tag {
          font-size: 10px;
          padding: 2px 6px;
          border-radius: 4px;
          text-transform: uppercase;
        }
        .status-tag.queued { background: rgba(251, 191, 36, 0.15); color: var(--color-omega); }
        .status-tag.executed { background: rgba(52, 211, 153, 0.15); color: var(--color-unlock-soft); }
        .empty-text {
          color: var(--color-muted);
          font-size: 12px;
          font-style: italic;
        }
      `}</style>
    </div>
  );
}
