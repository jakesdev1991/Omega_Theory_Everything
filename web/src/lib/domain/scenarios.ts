// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * End-to-end scenario suite for the four-plane crypto economy.
 *
 * Each scenario boots a fresh TriTokenEngine, drives one vertical slice of the
 * economy (docs/tri-token-integration-v1.md §7 plus the appeal and rail slices),
 * and asserts the invariants the slice must hold. The same suite runs in three
 * places:
 *
 *   1. In the browser, through the Economy Test Console (/testnet)
 *   2. In CI / locally, through `npm test` (src/lib/domain/scenarios.test.ts)
 *   3. Server-side, through POST /api/economy/scenarios
 *
 * Everything here is deterministic and offline. Chain-dependent verification is
 * reported separately by GET /api/economy/health and is never faked here.
 */

import { Wallet } from "ethers";
import nacl from "tweetnacl";
import bs58 from "bs58";

import {
  FULL_NOVEL_UNLOCK_CHAPTERS,
  OMEGA_NETWORK,
  TWC_NETWORK,
  verifyUnlockProof,
} from "../unlock";
import { VerifierAdapters } from "./adapters";
import {
  CONVERSION_DOCK_BPS,
  CONVERSION_EXTREME_HOLDBACK_BPS,
  MIN_GOVERNANCE_PROPOSAL_POWER,
  POLICY_VERSION,
  SPONSOR_BONUS_BPS,
  TriTokenEngine,
} from "./engine";
import type { BalanceCurrency, EngineOptions, Plane, WorkClass } from "./types";

export type ScenarioPlane = Plane | "RAILS" | "CROSS";

export interface ScenarioStepResult {
  id: string;
  title: string;
  plane: ScenarioPlane;
  passed: boolean;
  expected: string;
  actual: string;
  durationMs: number;
}

export interface ScenarioResult {
  id: string;
  title: string;
  plane: ScenarioPlane;
  slice: string;
  description: string;
  passed: boolean;
  steps: ScenarioStepResult[];
  durationMs: number;
  auditEventCount: number;
}

export interface ScenarioSummary {
  id: string;
  title: string;
  plane: ScenarioPlane;
  slice: string;
  description: string;
  stepCount: number;
}

export interface SuiteResult {
  ok: boolean;
  policyVersion: string;
  ranAt: string;
  durationMs: number;
  total: number;
  passed: number;
  failed: number;
  stepTotal: number;
  stepPassed: number;
  results: ScenarioResult[];
}

interface StepContext {
  engine: TriTokenEngine;
  /** Assert JSON-serialisable equality. */
  equal(actual: unknown, expected: unknown, label: string): void;
  /** Assert a boolean condition. */
  ok(condition: boolean, label: string, detail?: string): void;
  /** Assert that fn throws, optionally with a message fragment. */
  throws(fn: () => unknown, label: string, messagePart?: string): void;
}

interface ScenarioStep {
  title: string;
  plane: ScenarioPlane;
  run(ctx: StepContext): void;
}

export interface ScenarioDefinition {
  id: string;
  title: string;
  plane: ScenarioPlane;
  slice: string;
  description: string;
  engineOptions?: EngineOptions;
  steps: ScenarioStep[];
}

class StepFailure extends Error {}

function format(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

function createStepContext(engine: TriTokenEngine): StepContext {
  return {
    engine,
    equal(actual, expected, label) {
      const a = format(actual);
      const b = format(expected);
      if (a !== b) {
        throw new StepFailure(`${label}: expected ${b}, got ${a}`);
      }
    },
    ok(condition, label, detail) {
      if (!condition) {
        throw new StepFailure(`${label}${detail ? ` (${detail})` : ""}: expected true, got false`);
      }
    },
    throws(fn, label, messagePart) {
      try {
        fn();
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (messagePart && !message.includes(messagePart)) {
          throw new StepFailure(`${label}: expected error containing "${messagePart}", got "${message}"`);
        }
        return;
      }
      throw new StepFailure(`${label}: expected a rejection, but the call succeeded`);
    },
  };
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

const WORK_CLASSES: WorkClass[] = [
  "engineering_protocol",
  "lean_formalization",
  "zk_proving",
  "infrastructure_resource",
  "physics_simulation",
];

const FIXED_TIMESTAMP = "2026-01-01T00:00:00.000Z";
const FIXED_ORIGIN = "https://economy-test.omega.local";

function adapterFor(workClass: WorkClass) {
  switch (workClass) {
    case "lean_formalization":
      return VerifierAdapters.leanFormalization;
    case "zk_proving":
      return VerifierAdapters.zkProving;
    case "infrastructure_resource":
      return VerifierAdapters.infrastructure;
    case "physics_simulation":
      return VerifierAdapters.physicsSimulation;
    default:
      return VerifierAdapters.engineering;
  }
}

function rawUnlockMessage(currency: "OMEGA" | "TWC", address: string, network: string) {
  return [
    "OMEGA RELEASE-DAY NOVEL UNLOCK",
    `Currency: ${currency}`,
    `Address: ${address}`,
    `Network: ${network}`,
    `Origin: ${FIXED_ORIGIN}`,
    `Timestamp: ${FIXED_TIMESTAMP}`,
    "Unlock: full novel",
  ].join("\n");
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

function settleOneWorkReceipt(engine: TriTokenEngine, participant: string, workClass: WorkClass, budget: number) {
  const proposal = engine.proposeWork(
    participant,
    workClass,
    `Scenario artifact for ${workClass}`,
    workClass === "lean_formalization" ? ["kernel_checked", "audit_axioms"] : ["reproducible_build"],
    budget
  );
  return engine.executeAndVerifyWork(
    proposal.workId,
    `0xartifact-${workClass}-0000000000`,
    { cpuSeconds: 12 },
    adapterFor(workClass)
  );
}

/* ------------------------------------------------------------------ */
/* Scenarios                                                           */
/* ------------------------------------------------------------------ */

export const SCENARIOS: ScenarioDefinition[] = [
  {
    id: "slice-a-resource-quota",
    title: "Slice A — resource quota drives CARE status",
    plane: "CARE",
    slice: "A",
    description:
      "Wallet quota receipts move a participant between active, grace, and under_quota. Explicit consent is required before any device contribution counts.",
    steps: [
      {
        title: "New participant starts active at quota with no consent recorded",
        plane: "CARE",
        run: ({ engine, equal }) => {
          engine.registerParticipant("alice", "participant", "P0");
          equal(engine.participants.get("alice")?.status, "active", "initial status");
          equal(engine.budgets.get("alice")?.minimumQuotaUnits, 50, "minimum quota");
          equal(engine.hasConsent("alice", "resource_contribution"), false, "consent absent");
        },
      },
      {
        title: "Contribution without consent fails closed",
        plane: "CARE",
        run: ({ engine, throws }) => {
          throws(() => engine.updateResourceContribution("alice", 10), "consent gate", "explicit consent");
        },
      },
      {
        title: "Quota ladder: active → grace → under_quota → active",
        plane: "CARE",
        run: ({ engine, equal }) => {
          engine.recordConsent("alice", "resource_contribution", true);
          equal(engine.updateResourceContribution("alice", -20).status, "grace", "30/50 units");
          equal(engine.updateResourceContribution("alice", -20).status, "under_quota", "10/50 units");
          equal(engine.updateResourceContribution("alice", 45).status, "active", "55/50 units");
        },
      },
      {
        title: "Withdrawn consent blocks further contribution",
        plane: "CARE",
        run: ({ engine, throws }) => {
          engine.recordConsent("alice", "resource_contribution", false);
          throws(() => engine.updateResourceContribution("alice", 5), "withdrawn consent", "explicit consent");
        },
      },
    ],
  },
  {
    id: "slice-b-reach-allocation",
    title: "Slice B — reach records drive the pro-rata weekly CARE pool",
    plane: "CARE",
    slice: "B",
    description:
      "Reach and attention records settle a fixed weekly pool pro-rata across eligible participants, and every settlement is audited with the policy version.",
    steps: [
      {
        title: "Pro-rata allocation is exact for two participants",
        plane: "CARE",
        run: ({ engine, equal }) => {
          engine.registerParticipant("alice");
          engine.registerParticipant("bob");
          engine.recordReachEvent("alice", 3, 7); // 10 + 2 seeded = 12
          engine.recordReachEvent("bob", 1, 3); // 4 + 2 seeded = 6
          // The seeded demo-user holds 2 reach units, so the total is 20.
          const allocations = engine.distributeWeeklyCarePool();
          equal(allocations.get("alice"), 60000, "alice allocation");
          equal(allocations.get("bob"), 30000, "bob allocation");
          equal(engine.balanceFor("care", "alice"), 60000, "alice CARE balance");
        },
      },
      {
        title: "Under-quota participants are excluded from the pool",
        plane: "CARE",
        run: ({ engine, equal }) => {
          engine.registerParticipant("carol");
          engine.recordConsent("carol", "resource_contribution", true);
          engine.updateResourceContribution("carol", -45); // 5/50 -> under_quota
          engine.recordReachEvent("carol", 9, 9);
          const allocations = engine.distributeWeeklyCarePool();
          equal(allocations.has("carol"), false, "carol excluded from the pool");
        },
      },
      {
        title: "Every settlement writes an audited event with the policy version",
        plane: "CARE",
        run: ({ engine, ok }) => {
          const settlements = engine.auditEvents.filter((event) => event.action === "settle_weekly_allocation");
          ok(settlements.length > 0, "settlement events exist");
          ok(settlements.every((event) => event.policyVersion === POLICY_VERSION), "policy version stamped");
        },
      },
    ],
  },
  {
    id: "slice-c-hardship-solidarity",
    title: "Slice C — hardship solidarity is unbounded and debt-free",
    plane: "CARE",
    slice: "C",
    description:
      "A hardship request restores the recipient to active with no debt when a sponsor with sufficient verified contribution covers it, and pays the sponsor a bounded solidarity bonus.",
    steps: [
      {
        title: "Hardship moves the recipient to hardship status",
        plane: "CARE",
        run: ({ engine, equal }) => {
          engine.registerParticipant("recipient");
          engine.registerParticipant("sponsor");
          engine.recordConsent("sponsor", "resource_contribution", true);
          engine.updateResourceContribution("sponsor", 100);
          const request = engine.requestHardship("recipient", 40);
          equal(request.status, "open", "request open");
          equal(engine.participants.get("recipient")?.status, "hardship", "recipient status");
        },
      },
      {
        title: "Sponsor bonus is exactly 10% of the uncovered quota",
        plane: "CARE",
        run: ({ engine, equal }) => {
          const request = Array.from(engine.hardshipRequests.values())[0];
          const result = engine.sponsorHardship(request.requestId, "sponsor");
          equal(result.sponsorBonusUnits, Math.floor((40 * SPONSOR_BONUS_BPS) / 10000), "bonus units");
          equal(engine.balanceFor("twc", "sponsor"), 4, "sponsor TWC credit");
          equal(engine.participants.get("recipient")?.status, "active", "recipient restored with no debt");
        },
      },
      {
        title: "Double sponsorship fails closed",
        plane: "CARE",
        run: ({ engine, throws }) => {
          const request = Array.from(engine.hardshipRequests.values())[0];
          throws(() => engine.sponsorHardship(request.requestId, "sponsor"), "double sponsor", "already handled");
        },
      },
      {
        title: "A sponsor without sufficient verified contribution is rejected",
        plane: "CARE",
        run: ({ engine, throws }) => {
          engine.registerParticipant("poor-sponsor");
          const request = engine.requestHardship("recipient", 90);
          throws(
            () => engine.sponsorHardship(request.requestId, "poor-sponsor"),
            "insufficient sponsor contribution",
            "insufficient verified contribution"
          );
        },
      },
    ],
  },
  {
    id: "slice-d-care-amity-conversion",
    title: "Slice D — CARE → AMITY conversion is bounded and value-conserving",
    plane: "AMITY",
    slice: "D",
    description:
      "Conversion docks 5% (plus a 15% unissued holdback in extreme cases), never issues more AMITY than the CARE burned, and fails closed on insufficient balance.",
    steps: [
      {
        title: "Standard dock: 500 CARE converts to 475 AMITY",
        plane: "AMITY",
        run: ({ engine, equal }) => {
          engine.registerParticipant("alice");
          engine.creditBalance("care", "alice", 1000, "seed: weekly pool settlement");
          const conversion = engine.requestCareToAmityConversion("alice", 500);
          equal(conversion.participantDockBps, CONVERSION_DOCK_BPS, "dock bps");
          equal(conversion.unissuedHoldbackBps, 0, "no holdback");
          equal(conversion.netAmityAmount, 475, "net AMITY");
          equal(engine.balanceFor("care", "alice"), 500, "CARE deducted");
          equal(engine.balanceFor("amity", "alice"), 475, "AMITY credited");
        },
      },
      {
        title: "Extreme holdback: 500 CARE converts to 400 AMITY",
        plane: "AMITY",
        run: ({ engine, equal }) => {
          const conversion = engine.requestCareToAmityConversion("alice", 500, true);
          equal(conversion.unissuedHoldbackBps, CONVERSION_EXTREME_HOLDBACK_BPS, "holdback bps");
          equal(conversion.netAmityAmount, 400, "net AMITY with holdback");
          equal(engine.balanceFor("care", "alice"), 0, "CARE fully burned");
          equal(engine.balanceFor("amity", "alice"), 875, "cumulative AMITY");
        },
      },
      {
        title: "Insufficient CARE fails closed and mints nothing",
        plane: "AMITY",
        run: ({ engine, equal, throws }) => {
          throws(() => engine.requestCareToAmityConversion("alice", 1), "insufficient CARE", "Insufficient CARE");
          equal(engine.balanceFor("amity", "alice"), 875, "AMITY unchanged");
        },
      },
      {
        title: "Conversion never creates value: net ≤ burned",
        plane: "CROSS",
        run: ({ engine, ok }) => {
          ok(
            engine.conversionRequests.size > 0 &&
              Array.from(engine.conversionRequests.values()).every(
                (conversion) => conversion.netAmityAmount <= conversion.careAmount
              ),
            "every conversion is value-conserving"
          );
        },
      },
    ],
  },
  {
    id: "slice-e-twc-work-lifecycle",
    title: "Slice E — Proof-of-Useful-Work across all five work classes",
    plane: "TWC",
    slice: "E",
    description:
      "Each work class runs proposal → verifier adapter → receipt → TWC settlement. A failing adapter rejects the receipt, and awards follow max(10, ⌊budget × 1.5⌋).",
    steps: [
      ...WORK_CLASSES.map((workClass): ScenarioStep => ({
        title: `${workClass}: proposal settles a verified receipt and credits TWC`,
        plane: "TWC",
        run: ({ engine, equal, ok }) => {
          engine.registerParticipant("builder");
          const budget = 100;
          const before = engine.balanceFor("twc", "builder");
          const receipt = settleOneWorkReceipt(engine, "builder", workClass, budget);
          const expectedAward = Math.max(10, Math.floor(budget * 1.5));
          equal(receipt.issuedTwcUnits, expectedAward, "TWC award");
          equal(receipt.status, "settled", "receipt status");
          equal(engine.balanceFor("twc", "builder"), before + expectedAward, "balance credited");
          ok(receipt.verificationResult.verified, "verifier marked the artifact verified");
          equal(receipt.policyVersion, POLICY_VERSION, "receipt policy version");
        },
      })),
      {
        title: "A failing verifier adapter rejects the receipt (fail closed)",
        plane: "TWC",
        run: ({ engine, equal, throws }) => {
          engine.registerParticipant("builder");
          const before = engine.balanceFor("twc", "builder");
          const proposal = engine.proposeWork("builder", "engineering_protocol", "Broken build", ["tests"], 100);
          throws(
            () => engine.executeAndVerifyWork(proposal.workId, "short", { cpuSeconds: 1 }, VerifierAdapters.engineering),
            "invalid artifact hash",
            "Work verification failed"
          );
          equal(engine.workReceipts.has(proposal.workId), false, "no receipt on rejection");
          equal(engine.balanceFor("twc", "builder"), before, "no TWC without a receipt");
        },
      },
      {
        title: "Lean artifacts keep the kernel-checked / research-artifact distinction",
        plane: "TWC",
        run: ({ engine, equal }) => {
          engine.registerParticipant("formalizer");
          const sorry = engine.proposeWork("formalizer", "lean_formalization", "Attempt with a sorry", ["contains_sorry"], 40);
          equal(
            VerifierAdapters.leanFormalization(sorry, "0xartifact-lean-0000000000").category,
            "formalization_attempt",
            "sorry-bearing file is not a checked theorem"
          );
          const clean = engine.proposeWork("formalizer", "lean_formalization", "Kernel-checked lemma", ["kernel_checked"], 40);
          equal(
            VerifierAdapters.leanFormalization(clean, "0xartifact-lean-0000000001").category,
            "kernel_checked_theorem",
            "kernel-checked, sorry-free artifact"
          );
        },
      },
    ],
  },
  {
    id: "slice-f-governance-timelock",
    title: "Slice F — $OMEGA governance enforces voting power and the timelock",
    plane: "OMEGA",
    slice: "F",
    description:
      "Proposals require real voting power when bootstrap is disabled, and execution is impossible before the timelock expires.",
    engineOptions: { bootstrapVotingPower: false },
    steps: [
      {
        title: `Proposing below ${MIN_GOVERNANCE_PROPOSAL_POWER} voting power fails closed`,
        plane: "OMEGA",
        run: ({ engine, throws }) => {
          engine.registerParticipant("citizen");
          throws(
            () => engine.proposeGovernancePolicy("citizen", "AMITY", "Raise conversion ceiling", "…", {}),
            "no voting power",
            "Insufficient $OMEGA voting power"
          );
        },
      },
      {
        title: "A funded proposer can queue a proposal",
        plane: "OMEGA",
        run: ({ engine, equal }) => {
          engine.creditBalance("omega", "citizen", 1000, "observed escrow import");
          const proposal = engine.proposeGovernancePolicy(
            "citizen",
            "AMITY",
            "Bound AMITY conversion ceiling",
            "Cap the extreme holdback at 1500 bps",
            { ceilingBps: 3500 },
            3600
          );
          equal(proposal.status, "queued", "queued");
          equal(proposal.simulationVerified, true, "simulation recorded");
        },
      },
      {
        title: "Execution before the timelock expires is rejected",
        plane: "OMEGA",
        run: ({ engine, throws }) => {
          const proposal = Array.from(engine.governanceProposals.values())[0];
          throws(
            () => engine.executeGovernancePolicy(proposal.proposalId, Date.parse(proposal.queuedAt ?? "") + 10),
            "early execution",
            "Timelock has not expired"
          );
        },
      },
      {
        title: "Execution after the timelock succeeds exactly once",
        plane: "OMEGA",
        run: ({ engine, equal, throws }) => {
          const proposal = Array.from(engine.governanceProposals.values())[0];
          const after = Date.parse(proposal.executableAfter ?? "") + 1;
          equal(engine.executeGovernancePolicy(proposal.proposalId, after), true, "executed");
          equal(engine.governanceProposals.get(proposal.proposalId)?.status, "executed", "status executed");
          throws(() => engine.executeGovernancePolicy(proposal.proposalId, after + 1000), "second execution", "not queued");
        },
      },
    ],
  },
  {
    id: "slice-g-appeals",
    title: "Slice G — work receipts are appealable and reversible",
    plane: "TWC",
    slice: "G",
    description:
      "Every settlement is appealable. A reversal claws back settled TWC, marks the receipt reversed, and the appellant may never review their own appeal.",
    steps: [
      {
        title: "A settled receipt can be appealed exactly once while open",
        plane: "TWC",
        run: ({ engine, equal, throws }) => {
          engine.registerParticipant("dev");
          const receipt = settleOneWorkReceipt(engine, "dev", "zk_proving", 80);
          const appeal = engine.openWorkAppeal(receipt.workId, "watchdog", "Proof vector was reused");
          equal(appeal.status, "open", "appeal open");
          throws(() => engine.openWorkAppeal(receipt.workId, "watchdog", "duplicate"), "duplicate appeal", "already exists");
        },
      },
      {
        title: "Self-review is rejected",
        plane: "TWC",
        run: ({ engine, throws }) => {
          const appeal = Array.from(engine.workAppeals.values())[0];
          throws(() => engine.resolveWorkAppeal(appeal.appealId, "watchdog", "dismissed"), "self review", "own appeal");
        },
      },
      {
        title: "Reversal claws back the settled TWC and marks the receipt reversed",
        plane: "TWC",
        run: ({ engine, equal }) => {
          const appeal = Array.from(engine.workAppeals.values())[0];
          const receipt = engine.workReceipts.get(appeal.workId);
          const credited = receipt?.issuedTwcUnits ?? 0;
          const before = engine.balanceFor("twc", "dev");
          const resolved = engine.resolveWorkAppeal(appeal.appealId, "archangel-1", "reversed", "Not reproducible");
          equal(resolved.status, "reversed", "appeal reversed");
          equal(resolved.clawbackTwcUnits, Math.min(before, credited), "clawback recorded");
          equal(engine.balanceFor("twc", "dev"), before - Math.min(before, credited), "balance clawed back");
          equal(engine.workReceipts.get(appeal.workId)?.status, "reversed", "receipt reversed");
        },
      },
      {
        title: "Resolved appeals cannot be re-resolved or re-appealed",
        plane: "TWC",
        run: ({ engine, throws }) => {
          const appeal = Array.from(engine.workAppeals.values())[0];
          throws(() => engine.resolveWorkAppeal(appeal.appealId, "archangel-2", "upheld"), "re-resolve", "already resolved");
          throws(() => engine.openWorkAppeal(appeal.workId, "watchdog", "again"), "re-appeal", "already reversed");
        },
      },
    ],
  },
  {
    id: "rails-omega-signature",
    title: "Rail — $OMEGA unlock proof signature round-trip",
    plane: "RAILS",
    slice: "R1",
    description:
      "A secp256k1 signature over the canonical unlock message verifies against the embedded EVM address. Tampering fails closed before any chain call.",
    steps: [
      {
        title: "A valid signature verifies and unlocks the full novel",
        plane: "RAILS",
        run: ({ equal }) => {
          const wallet = Wallet.createRandom();
          const message = rawUnlockMessage("OMEGA", wallet.address, OMEGA_NETWORK);
          const signature = wallet.signMessageSync(message);
          const verified = verifyUnlockProof({ message, signature });
          equal(verified.address, wallet.address, "recovered address");
          equal(verified.currency, "OMEGA", "currency");
          equal(verified.unlocked, FULL_NOVEL_UNLOCK_CHAPTERS, "unlocked chapters");
        },
      },
      {
        title: "A signature from a different key is rejected",
        plane: "RAILS",
        run: ({ throws }) => {
          const wallet = Wallet.createRandom();
          const stranger = Wallet.createRandom();
          const message = rawUnlockMessage("OMEGA", wallet.address, OMEGA_NETWORK);
          throws(
            () => verifyUnlockProof({ message, signature: stranger.signMessageSync(message) }),
            "foreign signature",
            "Signature verification failed"
          );
        },
      },
      {
        title: "Editing the embedded address invalidates the proof",
        plane: "RAILS",
        run: ({ throws }) => {
          const wallet = Wallet.createRandom();
          const stranger = Wallet.createRandom();
          const message = rawUnlockMessage("OMEGA", wallet.address, OMEGA_NETWORK);
          const signature = wallet.signMessageSync(message);
          const tampered = message.replace(wallet.address, stranger.address);
          throws(() => verifyUnlockProof({ message: tampered, signature }), "tampered address", "Signature verification failed");
        },
      },
      {
        title: "An $OMEGA proof aimed at the wrong network is rejected",
        plane: "RAILS",
        run: ({ throws }) => {
          const wallet = Wallet.createRandom();
          const message = rawUnlockMessage("OMEGA", wallet.address, TWC_NETWORK);
          const signature = wallet.signMessageSync(message);
          throws(() => verifyUnlockProof({ message, signature }), "wrong network", "must target");
        },
      },
    ],
  },
  {
    id: "rails-twc-signature",
    title: "Rail — TWC unlock proof signature round-trip",
    plane: "RAILS",
    slice: "R2",
    description:
      "An ed25519 signature over the canonical unlock message verifies against the embedded Solana address. Tampering fails closed before any chain call.",
    steps: [
      {
        title: "A valid ed25519 signature verifies and unlocks the full novel",
        plane: "RAILS",
        run: ({ equal }) => {
          const keyPair = nacl.sign.keyPair();
          const address = bs58.encode(keyPair.publicKey);
          const message = rawUnlockMessage("TWC", address, TWC_NETWORK);
          const signature = bytesToBase64(nacl.sign.detached(new TextEncoder().encode(message), keyPair.secretKey));
          const verified = verifyUnlockProof({ message, signature });
          equal(verified.address, address, "embedded address");
          equal(verified.currency, "TWC", "currency");
          equal(verified.unlocked, FULL_NOVEL_UNLOCK_CHAPTERS, "unlocked chapters");
        },
      },
      {
        title: "A single flipped signature byte is rejected",
        plane: "RAILS",
        run: ({ throws }) => {
          const keyPair = nacl.sign.keyPair();
          const address = bs58.encode(keyPair.publicKey);
          const message = rawUnlockMessage("TWC", address, TWC_NETWORK);
          const signatureBytes = nacl.sign.detached(new TextEncoder().encode(message), keyPair.secretKey);
          signatureBytes[7] ^= 0xff;
          throws(
            () => verifyUnlockProof({ message, signature: bytesToBase64(signatureBytes) }),
            "flipped byte",
            "Signature verification failed"
          );
        },
      },
      {
        title: "A non-base58 Solana address is rejected",
        plane: "RAILS",
        run: ({ throws }) => {
          const keyPair = nacl.sign.keyPair();
          const message = rawUnlockMessage("TWC", "not-a-base58-address!!", TWC_NETWORK);
          const signature = bytesToBase64(nacl.sign.detached(new TextEncoder().encode(message), keyPair.secretKey));
          void keyPair;
          throws(() => verifyUnlockProof({ message, signature }), "bad address", "base58");
        },
      },
    ],
  },
  {
    id: "cross-invariants",
    title: "Cross-plane invariants — no value from nothing, full audit coverage",
    plane: "CROSS",
    slice: "X",
    description:
      "After a mixed workload across all four planes, balances reconcile against the audited ledger, nothing goes negative, and every event carries the policy version.",
    steps: [
      {
        title: "Mixed workload across all four planes runs clean",
        plane: "CROSS",
        run: ({ engine }) => {
          for (const id of ["ana", "ben"]) {
            engine.registerParticipant(id);
            engine.recordConsent(id, "resource_contribution", true);
            engine.updateResourceContribution(id, 60);
            engine.recordReachEvent(id, 4, 6);
          }
          engine.distributeWeeklyCarePool();
          engine.creditBalance("omega", "ana", 500, "observed escrow import");
          settleOneWorkReceipt(engine, "ana", "engineering_protocol", 120);
          settleOneWorkReceipt(engine, "ben", "infrastructure_resource", 60);
          engine.requestCareToAmityConversion("ana", 1000);
          const hardship = engine.requestHardship("ben", 30);
          engine.sponsorHardship(hardship.requestId, "ana");
          engine.proposeGovernancePolicy("ana", "TWC", "Raise work award cap", "Pilot tuning", { cap: 400 }, 60);
        },
      },
      {
        title: "No balance is negative on any plane",
        plane: "CROSS",
        run: ({ engine, ok }) => {
          const currencies: BalanceCurrency[] = ["care", "twc", "omega", "amity"];
          for (const currency of currencies) {
            for (const [participant, balance] of engine.balances(currency)) {
              ok(balance >= 0, `${currency}/${participant} non-negative`, String(balance));
            }
          }
        },
      },
      {
        title: "TWC balances reconcile with receipts, sponsor bonuses, credits, and clawbacks",
        plane: "CROSS",
        run: ({ engine, equal }) => {
          const expected = new Map<string, number>();
          const add = (participant: string, units: number) =>
            expected.set(participant, (expected.get(participant) || 0) + units);

          for (const receipt of engine.workReceipts.values()) {
            if (receipt.status === "settled") add(receipt.contributorCommitment, receipt.issuedTwcUnits);
          }
          for (const hardship of engine.hardshipRequests.values()) {
            if (hardship.status === "sponsored" && hardship.sponsoredBy) {
              add(hardship.sponsoredBy, hardship.sponsorBonusUnits || 0);
            }
          }
          for (const event of engine.auditEvents) {
            if (event.action === "credit_balance" && event.inputCommitments[0] === "twc") {
              add(event.actorCommitment, Number(event.inputCommitments[2]));
            }
            if (event.action === "reverse_work_receipt") {
              const receipt = engine.workReceipts.get(event.inputCommitments[1]);
              if (receipt) add(receipt.contributorCommitment, -Number(event.inputCommitments[2]));
            }
          }

          for (const [participant, balance] of engine.twcBalances) {
            equal(balance, expected.get(participant) || 0, `twc/${participant} ledger reconciliation`);
          }
        },
      },
      {
        title: "CARE balances reconcile with pool settlements, credits, and conversion burns",
        plane: "CROSS",
        run: ({ engine, equal }) => {
          const expected = new Map<string, number>();
          const add = (participant: string, units: number) =>
            expected.set(participant, (expected.get(participant) || 0) + units);

          for (const event of engine.auditEvents) {
            if (event.action === "settle_weekly_allocation") add(event.actorCommitment, Number(event.outputCommitments[0]));
            if (event.action === "credit_balance" && event.inputCommitments[0] === "care") {
              add(event.actorCommitment, Number(event.inputCommitments[2]));
            }
          }
          for (const conversion of engine.conversionRequests.values()) {
            add(conversion.participantId, -conversion.careAmount);
          }

          for (const [participant, balance] of engine.careBalances) {
            equal(balance, expected.get(participant) || 0, `care/${participant} ledger reconciliation`);
          }
        },
      },
      {
        title: "AMITY in existence equals the sum of conversion issuances and credits",
        plane: "CROSS",
        run: ({ engine, equal }) => {
          let issued = 0;
          for (const conversion of engine.conversionRequests.values()) issued += conversion.netAmityAmount;
          for (const event of engine.auditEvents) {
            if (event.action === "credit_balance" && event.inputCommitments[0] === "amity") {
              issued += Number(event.inputCommitments[2]);
            }
          }
          const total = Array.from(engine.amityBalances.values()).reduce((sum, value) => sum + value, 0);
          equal(total, issued, "total AMITY supply");
        },
      },
      {
        title: "Every audit event carries the policy version and a named actor",
        plane: "CROSS",
        run: ({ engine, ok }) => {
          ok(engine.auditEvents.length > 0, "audit trail not empty");
          ok(
            engine.auditEvents.every(
              (event) => event.policyVersion === POLICY_VERSION && event.actorCommitment.length > 0
            ),
            "policy version and actor on every event"
          );
        },
      },
    ],
  },
];

/* ------------------------------------------------------------------ */
/* Runner                                                              */
/* ------------------------------------------------------------------ */

export function runScenario(definition: ScenarioDefinition): ScenarioResult {
  const startedAt = Date.now();
  const engine = new TriTokenEngine(definition.engineOptions ?? { testMode: true });
  const ctx = createStepContext(engine);
  const steps: ScenarioStepResult[] = [];

  for (let index = 0; index < definition.steps.length; index += 1) {
    const step = definition.steps[index];
    const stepStartedAt = Date.now();
    let passed = true;
    let expected = "step completes without failing an assertion";
    let actual = "ok";
    try {
      step.run(ctx);
    } catch (error) {
      passed = false;
      actual = error instanceof Error ? error.message : String(error);
      expected = error instanceof StepFailure ? "assertion holds" : expected;
    }
    steps.push({
      id: `${definition.id}:${index + 1}`,
      title: step.title,
      plane: step.plane,
      passed,
      expected,
      actual,
      durationMs: Date.now() - stepStartedAt,
    });
  }

  return {
    id: definition.id,
    title: definition.title,
    plane: definition.plane,
    slice: definition.slice,
    description: definition.description,
    passed: steps.every((step) => step.passed),
    steps,
    durationMs: Date.now() - startedAt,
    auditEventCount: engine.auditEvents.length,
  };
}

export function runAllScenarios(): SuiteResult {
  const startedAt = Date.now();
  const results = SCENARIOS.map(runScenario);
  const stepResults = results.flatMap((result) => result.steps);
  const passed = results.filter((result) => result.passed).length;

  return {
    ok: passed === results.length,
    policyVersion: POLICY_VERSION,
    ranAt: new Date().toISOString(),
    durationMs: Date.now() - startedAt,
    total: results.length,
    passed,
    failed: results.length - passed,
    stepTotal: stepResults.length,
    stepPassed: stepResults.filter((step) => step.passed).length,
    results,
  };
}

export function scenarioSummaries(): ScenarioSummary[] {
  return SCENARIOS.map((definition) => ({
    id: definition.id,
    title: definition.title,
    plane: definition.plane,
    slice: definition.slice,
    description: definition.description,
    stepCount: definition.steps.length,
  }));
}
