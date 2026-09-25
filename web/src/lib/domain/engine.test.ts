// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import test from "node:test";
import assert from "node:assert/strict";
import { TriTokenEngine } from "./engine";
import { VerifierAdapters } from "./adapters";

test("Slice A: Resource Quota to CARE Status transition", () => {
  const engine = new TriTokenEngine();
  const participantId = "alice";
  engine.registerParticipant(participantId, "participant", "P0");
  engine.recordConsent(participantId, "resource_contribution", true);

  // Initial state: contributedUnits = 50, minimumQuota = 50 -> active
  const initial = engine.participants.get(participantId);
  assert.equal(initial?.status, "active");

  // Reduce quota to below minimum: 30 / 50 -> grace (>= 50%)
  const graceResult = engine.updateResourceContribution(participantId, -20);
  assert.equal(graceResult.status, "grace");

  // Reduce quota further: 10 / 50 -> under_quota (< 50%)
  const underResult = engine.updateResourceContribution(participantId, -20);
  assert.equal(underResult.status, "under_quota");

  // Restore quota back above 50 -> active
  const activeResult = engine.updateResourceContribution(participantId, 45);
  assert.equal(activeResult.status, "active");
});

test("Slice B: CARE Social Reach and Pro-Rata Distribution", () => {
  const engine = new TriTokenEngine();
  const alice = "alice";
  const bob = "bob";

  engine.registerParticipant(alice);
  engine.registerParticipant(bob);

  // Alice: 3 audience + 7 attention = 10 reach units (+ 2 initial = 12)
  engine.recordReachEvent(alice, 3, 7);
  // Bob: 1 audience + 3 attention = 4 reach units (+ 2 initial = 6)
  engine.recordReachEvent(bob, 1, 3);

  // Demo user also exists with 2 initial reach units. Total reach = 12 + 6 + 2 = 20
  const allocations = engine.distributeWeeklyCarePool();
  assert.ok(allocations.has(alice));
  assert.ok(allocations.has(bob));

  // Alice has 12 / 20 = 60% of the 100,000 pool -> 60,000
  assert.equal(allocations.get(alice), 60000);
  // Bob has 6 / 20 = 30% of the 100,000 pool -> 30,000
  assert.equal(allocations.get(bob), 30000);
  assert.equal(engine.careBalances.get(alice), 60000);
});

test("Slice C: Hardship Solidarity (Unbounded, No Debt, Sponsor Bonus)", () => {
  const engine = new TriTokenEngine();
  const recipient = "charlie";
  const sponsor = "sponsor-dan";

  engine.registerParticipant(recipient);
  engine.registerParticipant(sponsor);
  engine.recordConsent(sponsor, "resource_contribution", true);

  // Sponsor contributes excess resources
  engine.updateResourceContribution(sponsor, 100);

  // Recipient enters hardship
  const hardship = engine.requestHardship(recipient, 40);
  assert.equal(hardship.status, "open");
  assert.equal(engine.participants.get(recipient)?.status, "hardship");

  // Sponsor covers hardship
  const result = engine.sponsorHardship(hardship.requestId, sponsor);
  assert.equal(result.request.status, "sponsored");
  assert.equal(result.sponsorBonusUnits, 4); // 10% bonus
  assert.equal(engine.twcBalances.get(sponsor), 4);
  assert.equal(engine.participants.get(recipient)?.status, "active");
});

test("Slice D: CARE to AMITY Bounded Conversion", () => {
  const engine = new TriTokenEngine();
  const alice = "alice";
  engine.registerParticipant(alice);

  // Seed CARE balance
  engine.careBalances.set(alice, 1000);

  // Standard conversion: 5% dock (500 bps), 0% holdback -> 95% net AMITY
  const conv1 = engine.requestCareToAmityConversion(alice, 400, false);
  assert.equal(conv1.careAmount, 400);
  assert.equal(conv1.participantDockBps, 500);
  assert.equal(conv1.unissuedHoldbackBps, 0);
  assert.equal(conv1.netAmityAmount, 380);
  assert.equal(engine.amityBalances.get(alice), 380);
  assert.equal(engine.careBalances.get(alice), 600);

  // Extreme holdback conversion: 5% dock + 15% holdback = 20% total reduction -> 80% net AMITY
  const conv2 = engine.requestCareToAmityConversion(alice, 500, true);
  assert.equal(conv2.netAmityAmount, 400); // 500 * 0.8
  assert.equal(engine.amityBalances.get(alice), 780);
  assert.equal(engine.careBalances.get(alice), 100);
});

test("Slice E: TWC Proof-of-Useful-Work Lifecycle and Adapters", () => {
  const engine = new TriTokenEngine();
  const dev = "developer-eva";
  engine.registerParticipant(dev);

  // 1. Engineering work
  const p1 = engine.proposeWork(
    dev,
    "engineering_protocol",
    "Implement fail-closed Taproot verification adapter",
    ["unit_tests_pass", "build_reproducible"],
    100
  );
  const r1 = engine.executeAndVerifyWork(
    p1.workId,
    "0xcommitsha256abcdef1234567890",
    { cpuSeconds: 12 },
    VerifierAdapters.engineering
  );
  assert.equal(r1.verificationResult.verified, true);
  assert.equal(r1.issuedTwcUnits, 150);
  assert.equal(engine.twcBalances.get(dev), 150);

  // 2. Lean Formalization (kernel checked)
  const p2 = engine.proposeWork(
    dev,
    "lean_formalization",
    "Prove topological invariance in Lean 4",
    ["kernel_checked", "audit_axioms"],
    200
  );
  const r2 = engine.executeAndVerifyWork(
    p2.workId,
    "0xleanproofhash987654321",
    { cpuSeconds: 60 },
    VerifierAdapters.leanFormalization
  );
  assert.equal(r2.verificationResult.verified, true);
  assert.equal(r2.issuedTwcUnits, 300);
  assert.equal(engine.twcBalances.get(dev), 450);
});

test("Slice F: OMEGA Governance Proposal and Timelock Execution", () => {
  const engine = new TriTokenEngine();
  const governor = "archangel-steward";
  engine.registerParticipant(governor, "archangel", "P3");

  const proposal = engine.proposeGovernancePolicy(
    governor,
    "AMITY",
    "Adjust Conversion Cap Ceiling",
    "Set unissued holdback maximum to 1500 bps",
    { extremeUnissuedHoldbackBps: 1500 },
    10 // 10 seconds timelock
  );

  assert.equal(proposal.status, "queued");

  // Attempt immediate execution should fail due to timelock
  assert.throws(() => {
    engine.executeGovernancePolicy(proposal.proposalId, Date.now());
  }, /Timelock has not expired/);

  // Execute after timelock simulated expiry
  const future = Date.now() + 15 * 1000;
  const executed = engine.executeGovernancePolicy(proposal.proposalId, future);
  assert.equal(executed, true);
  assert.equal(engine.governanceProposals.get(proposal.proposalId)?.status, "executed");
});

test("Consent gate: resource contribution fails closed without explicit consent", () => {
  const engine = new TriTokenEngine();
  engine.registerParticipant("no-consent");

  assert.throws(
    () => engine.updateResourceContribution("no-consent", 10),
    /explicit consent/
  );

  engine.recordConsent("no-consent", "resource_contribution", true);
  assert.equal(engine.updateResourceContribution("no-consent", 10).status, "active");

  engine.recordConsent("no-consent", "resource_contribution", false);
  assert.throws(() => engine.updateResourceContribution("no-consent", 10), /explicit consent/);
});

test("Governance fails closed without voting power when bootstrap is disabled", () => {
  const engine = new TriTokenEngine({ bootstrapVotingPower: false });
  engine.registerParticipant("powerless");

  assert.throws(
    () => engine.proposeGovernancePolicy("powerless", "AMITY", "t", "d", {}),
    /Insufficient \$OMEGA voting power/
  );

  engine.creditBalance("omega", "powerless", 1000, "observed on-chain holding import");
  const proposal = engine.proposeGovernancePolicy("powerless", "AMITY", "t", "d", {});
  assert.equal(proposal.status, "queued");
});

test("Appeals: reversal claws back settled TWC and marks the receipt reversed", () => {
  const engine = new TriTokenEngine();
  const dev = "dev-appeal";
  engine.registerParticipant(dev);

  const proposal = engine.proposeWork(dev, "engineering_protocol", "Ship adapter", ["tests"], 100);
  const receipt = engine.executeAndVerifyWork(
    proposal.workId,
    "0xartifact-appeal-1",
    { cpuSeconds: 5 },
    VerifierAdapters.engineering
  );
  assert.equal(receipt.status, "settled");
  assert.equal(engine.twcBalances.get(dev), receipt.issuedTwcUnits);

  const appeal = engine.openWorkAppeal(proposal.workId, "reviewer-a", "Benchmark was fabricated");
  assert.throws(() => engine.resolveWorkAppeal(appeal.appealId, "reviewer-a", "dismissed"), /own appeal/);

  const resolved = engine.resolveWorkAppeal(appeal.appealId, "reviewer-b", "reversed", "Evidence not reproducible");
  assert.equal(resolved.status, "reversed");
  assert.equal(resolved.clawbackTwcUnits, receipt.issuedTwcUnits);
  assert.equal(engine.twcBalances.get(dev), 0);
  assert.equal(engine.workReceipts.get(proposal.workId)?.status, "reversed");

  assert.throws(() => engine.resolveWorkAppeal(appeal.appealId, "reviewer-b", "upheld"), /already resolved/);
  assert.throws(() => engine.openWorkAppeal(proposal.workId, "reviewer-a", "again"), /already reversed/);
});

test("Faucet grants are disabled outside test mode", () => {
  const production = new TriTokenEngine();
  production.registerParticipant("alice");
  assert.throws(() => production.faucetGrant("twc", "alice", 500), /not in test mode/);

  const testEngine = new TriTokenEngine({ testMode: true });
  testEngine.registerParticipant("alice");
  assert.equal(testEngine.faucetGrant("twc", "alice", 500), 500);
  assert.ok(testEngine.auditEvents.some((event) => event.action === "credit_balance"));
});
