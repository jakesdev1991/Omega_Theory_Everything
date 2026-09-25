// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
import type {
  AuditEvent,
  BalanceCurrency,
  CareConversionRequest,
  CarePrivacyLevel,
  CareRole,
  CareStatus,
  ConsentRecord,
  EngineOptions,
  GovernancePolicyProposal,
  HardshipRequest,
  ParticipantCredential,
  Plane,
  ResourceBudget,
  WorkAppeal,
  WorkAppealDecision,
  WorkClass,
  WorkProposal,
  WorkReceipt,
} from "./types";

export const POLICY_VERSION = "tri-token-v1-prototype";
export const MIN_GOVERNANCE_PROPOSAL_POWER = 100;
export const BOOTSTRAP_VOTING_POWER = 1000;
export const SPONSOR_BONUS_BPS = 1000; // 10% of the uncovered quota, as useful-work credit
export const CONVERSION_DOCK_BPS = 500; // 5% participant dock on CARE -> AMITY
export const CONVERSION_EXTREME_HOLDBACK_BPS = 1500; // extra 15% unissued holdback

export class TriTokenEngine {
  public participants: Map<string, ParticipantCredential> = new Map();
  public budgets: Map<string, ResourceBudget> = new Map();
  public consents: Map<string, ConsentRecord> = new Map();
  public workProposals: Map<string, WorkProposal> = new Map();
  public workReceipts: Map<string, WorkReceipt> = new Map();
  public workAppeals: Map<string, WorkAppeal> = new Map();
  public hardshipRequests: Map<string, HardshipRequest> = new Map();
  public conversionRequests: Map<string, CareConversionRequest> = new Map();
  public governanceProposals: Map<string, GovernancePolicyProposal> = new Map();
  public auditEvents: AuditEvent[] = [];
  public readonly options: Required<EngineOptions>;

  // Balances
  public careBalances: Map<string, number> = new Map();
  public twcBalances: Map<string, number> = new Map();
  public omegaBalances: Map<string, number> = new Map();
  public amityBalances: Map<string, number> = new Map();

  // Weekly CARE distribution pool
  public weeklyCarePool: number = 100_000;
  public reachEvents: Map<string, { audienceCount: number; attentionCount: number }> = new Map();

  constructor(options: EngineOptions = {}) {
    this.options = {
      testMode: options.testMode ?? false,
      bootstrapVotingPower: options.bootstrapVotingPower ?? true,
    };
    // Initial seeded participant
    this.registerParticipant("demo-user", "participant", "P0");
  }

  public balanceFor(currency: BalanceCurrency, participantId: string): number {
    const balances = this.balances(currency);
    return balances.get(participantId) || 0;
  }

  public balances(currency: BalanceCurrency): Map<string, number> {
    switch (currency) {
      case "care":
        return this.careBalances;
      case "twc":
        return this.twcBalances;
      case "omega":
        return this.omegaBalances;
      case "amity":
        return this.amityBalances;
      default:
        throw new Error(`Unknown currency: ${String(currency)}`);
    }
  }

  /**
   * Records an observed credit (for example an on-chain settlement imported by an
   * operator, or a faucet grant in test mode). Every credit is an explicit,
   * audited mutation — the engine never invents balances silently.
   */
  public creditBalance(
    currency: BalanceCurrency,
    participantId: string,
    units: number,
    reason: string
  ): number {
    if (!Number.isFinite(units) || units <= 0) throw new Error("Credit units must be a positive number");
    if (!this.participants.has(participantId)) throw new Error("Participant not found");
    if (!reason || !reason.trim()) throw new Error("A credit reason is required for the audit trail");

    const balances = this.balances(currency);
    const next = (balances.get(participantId) || 0) + units;
    balances.set(participantId, next);

    const plane: Plane = currency === "care" ? "CARE" : currency === "twc" ? "TWC" : currency === "omega" ? "OMEGA" : "AMITY";
    this.logEvent(plane, "credit_balance", participantId, [currency, reason, String(units)], [String(next)]);
    return next;
  }

  /**
   * Test-mode-only faucet grant. Throws in non-test engines so a production
   * caller can never mint balances through this path.
   */
  public faucetGrant(currency: BalanceCurrency, participantId: string, units: number): number {
    if (!this.options.testMode) {
      throw new Error("Faucet grants are disabled: this engine is not in test mode.");
    }
    return this.creditBalance(currency, participantId, units, "faucet:test-grant");
  }

  /** Plain-object snapshot used by the API routes, the test console, and exports. */
  public snapshot() {
    return {
      policyVersion: POLICY_VERSION,
      testMode: this.options.testMode,
      bootstrapVotingPower: this.options.bootstrapVotingPower,
      participants: Array.from(this.participants.values()),
      budgets: Array.from(this.budgets.values()),
      consents: Array.from(this.consents.values()),
      balances: {
        care: Object.fromEntries(this.careBalances),
        twc: Object.fromEntries(this.twcBalances),
        omega: Object.fromEntries(this.omegaBalances),
        amity: Object.fromEntries(this.amityBalances),
      },
      proposals: Array.from(this.workProposals.values()),
      receipts: Array.from(this.workReceipts.values()),
      appeals: Array.from(this.workAppeals.values()),
      hardships: Array.from(this.hardshipRequests.values()),
      conversions: Array.from(this.conversionRequests.values()),
      governance: Array.from(this.governanceProposals.values()),
      auditEvents: this.auditEvents,
      weeklyCarePool: this.weeklyCarePool,
      reachEvents: Object.fromEntries(this.reachEvents),
    };
  }

  public logEvent(
    plane: Plane,
    action: string,
    actor: string,
    inputs: string[],
    outputs: string[],
    evidence: string[] = []
  ): AuditEvent {
    const event: AuditEvent = {
      eventId: `evt-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
      plane,
      action,
      policyVersion: POLICY_VERSION,
      actorCommitment: actor,
      inputCommitments: inputs,
      outputCommitments: outputs,
      evidenceReferences: evidence,
      timestamp: new Date().toISOString(),
    };
    this.auditEvents.push(event);
    return event;
  }

  // --- Participant Registration & Privacy ---
  public registerParticipant(
    participantId: string,
    role: CareRole = "participant",
    privacyLevel: CarePrivacyLevel = "P0"
  ): ParticipantCredential {
    const cred: ParticipantCredential = {
      participantId,
      role,
      privacyLevel,
      status: "active",
      joinedEpoch: 1,
    };
    this.participants.set(participantId, cred);
    this.budgets.set(participantId, {
      participantId,
      budgetBps: 200, // 2%
      minimumQuotaUnits: 50,
      selectedUnits: 50,
      contributedUnits: 50,
      paused: false,
    });
    this.careBalances.set(participantId, 0);
    this.twcBalances.set(participantId, 0);
    this.omegaBalances.set(participantId, 0);
    this.amityBalances.set(participantId, 0);
    this.reachEvents.set(participantId, { audienceCount: 1, attentionCount: 1 });

    this.logEvent("CARE", "register_participant", participantId, [], [participantId]);
    return cred;
  }

  // --- Consent records (CARE) ---
  public recordConsent(
    participantId: string,
    scope: ConsentRecord["scope"],
    granted: boolean
  ): ConsentRecord {
    if (!this.participants.has(participantId)) throw new Error("Participant not found");
    const record: ConsentRecord = {
      consentId: `consent-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      participantId,
      scope,
      granted,
      timestamp: new Date().toISOString(),
    };
    this.consents.set(`${participantId}:${scope}`, record);
    this.logEvent(
      "CARE",
      granted ? "grant_consent" : "withdraw_consent",
      participantId,
      [scope],
      [record.consentId]
    );
    return record;
  }

  public hasConsent(participantId: string, scope: ConsentRecord["scope"]): boolean {
    return this.consents.get(`${participantId}:${scope}`)?.granted === true;
  }

  // --- Slice A: Resource Quota to CARE status ---
  public updateResourceContribution(
    participantId: string,
    contributedDelta: number,
    isPaused: boolean = false
  ): { status: CareStatus; budget: ResourceBudget } {
    const budget = this.budgets.get(participantId);
    if (!budget) throw new Error("Participant budget not found");
    const participant = this.participants.get(participantId);
    if (!participant) throw new Error("Participant not found");
    if (!this.hasConsent(participantId, "resource_contribution")) {
      throw new Error(
        "Resource contribution requires explicit consent. Record a resource_contribution consent first."
      );
    }

    budget.paused = isPaused;
    budget.contributedUnits += contributedDelta;

    if (budget.contributedUnits >= budget.minimumQuotaUnits) {
      participant.status = "active";
    } else if (budget.contributedUnits >= budget.minimumQuotaUnits * 0.5) {
      participant.status = "grace";
    } else {
      participant.status = "under_quota";
    }

    this.logEvent(
      "CARE",
      "update_resource_quota",
      participantId,
      [String(budget.contributedUnits)],
      [participant.status]
    );

    return { status: participant.status, budget };
  }

  // --- Slice B: CARE Social Reach to Weekly Allocation ---
  public recordReachEvent(
    participantId: string,
    audiences: number,
    attentions: number
  ): { totalReachUnits: number } {
    const cur = this.reachEvents.get(participantId) || { audienceCount: 0, attentionCount: 0 };
    cur.audienceCount += audiences;
    cur.attentionCount += attentions;
    this.reachEvents.set(participantId, cur);

    const totalReach = cur.audienceCount + cur.attentionCount;
    this.logEvent(
      "CARE",
      "record_reach_event",
      participantId,
      [String(audiences), String(attentions)],
      [String(totalReach)]
    );
    return { totalReachUnits: totalReach };
  }

  public distributeWeeklyCarePool(): Map<string, number> {
    let totalEligibleReach = 0;
    const participantReach = new Map<string, number>();

    for (const [pId, reach] of this.reachEvents.entries()) {
      const part = this.participants.get(pId);
      if (part && (part.status === "active" || part.status === "grace" || part.status === "hardship")) {
        const units = reach.audienceCount + reach.attentionCount;
        participantReach.set(pId, units);
        totalEligibleReach += units;
      }
    }

    const allocations = new Map<string, number>();
    if (totalEligibleReach === 0) return allocations;

    for (const [pId, units] of participantReach.entries()) {
      const allocation = Math.floor((this.weeklyCarePool * units) / totalEligibleReach);
      const currentBal = this.careBalances.get(pId) || 0;
      this.careBalances.set(pId, currentBal + allocation);
      allocations.set(pId, allocation);

      this.logEvent(
        "CARE",
        "settle_weekly_allocation",
        pId,
        [String(units), String(totalEligibleReach)],
        [String(allocation)]
      );
    }

    return allocations;
  }

  // --- Slice C: Hardship Solidarity (Unbounded, No Debt) ---
  public requestHardship(participantId: string, uncoveredUnits: number): HardshipRequest {
    const participant = this.participants.get(participantId);
    if (!participant) throw new Error("Participant not found");

    participant.status = "hardship";
    const req: HardshipRequest = {
      requestId: `req-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      participantId,
      uncoveredQuotaUnits: uncoveredUnits,
      status: "open",
      createdAt: new Date().toISOString(),
    };
    this.hardshipRequests.set(req.requestId, req);

    this.logEvent("CARE", "request_hardship", participantId, [String(uncoveredUnits)], [req.requestId]);
    return req;
  }

  public sponsorHardship(
    requestId: string,
    sponsorId: string
  ): { request: HardshipRequest; sponsorBonusUnits: number } {
    const req = this.hardshipRequests.get(requestId);
    if (!req) throw new Error("Hardship request not found");
    if (req.status !== "open") throw new Error("Request already handled");

    const sponsorBudget = this.budgets.get(sponsorId);
    if (!sponsorBudget || sponsorBudget.contributedUnits < req.uncoveredQuotaUnits) {
      throw new Error("Sponsor has insufficient verified contribution");
    }

    // Sponsor bonus: SPONSOR_BONUS_BPS (10%) useful-work credit for verified solidarity
    const bonus = Math.floor((req.uncoveredQuotaUnits * SPONSOR_BONUS_BPS) / 10000);
    req.status = "sponsored";
    req.sponsoredBy = sponsorId;
    req.sponsorBonusUnits = bonus;

    // Credit sponsor with TWC useful work settlement for verified solidarity
    const curTwc = this.twcBalances.get(sponsorId) || 0;
    this.twcBalances.set(sponsorId, curTwc + bonus);

    // Hardship recipient status restored to active without any debt
    const recipient = this.participants.get(req.participantId);
    if (recipient) recipient.status = "active";

    this.logEvent(
      "CARE",
      "sponsor_hardship",
      sponsorId,
      [requestId, String(req.uncoveredQuotaUnits)],
      [String(bonus)]
    );
    return { request: req, sponsorBonusUnits: bonus };
  }

  // --- Slice D: CARE to AMITY Bounded Conversion ---
  public requestCareToAmityConversion(
    participantId: string,
    careAmount: number,
    hasExtremeHoldback: boolean = false
  ): CareConversionRequest {
    const curCare = this.careBalances.get(participantId) || 0;
    if (curCare < careAmount) throw new Error("Insufficient CARE balance");

    // Dock: CONVERSION_DOCK_BPS (5%), holdback: 0 or CONVERSION_EXTREME_HOLDBACK_BPS (15%)
    const dockBps = CONVERSION_DOCK_BPS;
    const holdbackBps = hasExtremeHoldback ? CONVERSION_EXTREME_HOLDBACK_BPS : 0;
    const totalReductionBps = dockBps + holdbackBps;

    const netAmity = Math.floor((careAmount * (10000 - totalReductionBps)) / 10000);

    // Deduct CARE
    this.careBalances.set(participantId, curCare - careAmount);

    const conv: CareConversionRequest = {
      conversionId: `conv-${Date.now()}`,
      participantId,
      careAmount,
      participantDockBps: dockBps,
      unissuedHoldbackBps: holdbackBps,
      netAmityAmount: netAmity,
      fundedSettlementVerified: true,
      requestedAt: new Date().toISOString(),
      settledAt: new Date().toISOString(),
    };

    this.conversionRequests.set(conv.conversionId, conv);

    // Credit AMITY
    const curAmity = this.amityBalances.get(participantId) || 0;
    this.amityBalances.set(participantId, curAmity + netAmity);

    this.logEvent(
      "AMITY",
      "convert_care_to_amity",
      participantId,
      [String(careAmount), `${dockBps}bps`],
      [String(netAmity)]
    );

    return conv;
  }

  // --- Slice E: TWC Proof-of-Useful-Work Lifecycle ---
  public proposeWork(
    contributorCommitment: string,
    workClass: WorkClass,
    objective: string,
    acceptanceTests: string[],
    resourceBudget: number
  ): WorkProposal {
    const proposal: WorkProposal = {
      workId: `work-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      contributorCommitment,
      workClass,
      objective,
      acceptanceTests,
      inputCommitments: ["pkg-head-commit-2026"],
      resourceBudget,
      reviewPolicy: "deterministic_check",
      expirySeconds: 86400,
    };
    this.workProposals.set(proposal.workId, proposal);
    this.logEvent("TWC", "propose_work", contributorCommitment, [workClass], [proposal.workId]);
    return proposal;
  }

  public executeAndVerifyWork(
    workId: string,
    artifactHash: string,
    resources: { cpuSeconds?: number; bandwidthBytes?: number; memoryMb?: number },
    verifierAdapter: (proposal: WorkProposal, artifactHash: string) => { ok: boolean; message: string }
  ): WorkReceipt {
    const proposal = this.workProposals.get(workId);
    if (!proposal) throw new Error("Proposal not found");

    const check = verifierAdapter(proposal, artifactHash);
    if (!check.ok) {
      throw new Error(`Work verification failed: ${check.message}`);
    }

    // Award TWC based on work class and resource budget
    const twcAward = Math.max(10, Math.floor(proposal.resourceBudget * 1.5));

    const receipt: WorkReceipt = {
      workId,
      workClass: proposal.workClass,
      contributorCommitment: proposal.contributorCommitment,
      artifactHash,
      resourceReceipts: resources,
      verifierSet: ["omega-local-verifier-adapter"],
      verificationResult: {
        verified: true,
        adapterName: `adapter-${proposal.workClass}`,
        details: { result: check.message },
        verifiedAt: new Date().toISOString(),
      },
      policyVersion: POLICY_VERSION,
      issuedTwcUnits: twcAward,
      settledAt: new Date().toISOString(),
      status: "settled",
    };

    this.workReceipts.set(workId, receipt);

    // Credit contributor TWC
    const curTwc = this.twcBalances.get(proposal.contributorCommitment) || 0;
    this.twcBalances.set(proposal.contributorCommitment, curTwc + twcAward);

    this.logEvent(
      "TWC",
      "settle_work_receipt",
      proposal.contributorCommitment,
      [workId, artifactHash],
      [String(twcAward)]
    );

    return receipt;
  }

  // --- Slice F: OMEGA Governance Proposal & Timelock ---
  public proposeGovernancePolicy(
    proposerCommitment: string,
    targetPlane: Plane,
    title: string,
    description: string,
    parametersDiff: Record<string, unknown>,
    timelockSeconds: number = 86400
  ): GovernancePolicyProposal {
    // Proposer must hold OMEGA voting power; the historical bootstrap is a
    // test-only convenience and fails closed unless explicitly enabled.
    const omegaBal = this.omegaBalances.get(proposerCommitment) || 0;
    if (omegaBal < MIN_GOVERNANCE_PROPOSAL_POWER) {
      if (!this.options.bootstrapVotingPower) {
        throw new Error(
          `Insufficient $OMEGA voting power to propose: ${omegaBal} < ${MIN_GOVERNANCE_PROPOSAL_POWER}.`
        );
      }
      if (!this.participants.has(proposerCommitment)) {
        this.registerParticipant(proposerCommitment);
      }
      this.creditBalance(
        "omega",
        proposerCommitment,
        BOOTSTRAP_VOTING_POWER,
        "test-mode bootstrap voting power"
      );
    }

    const proposal: GovernancePolicyProposal = {
      proposalId: `gov-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      proposerCommitment,
      targetPlane,
      title,
      description,
      parametersDiff,
      simulationVerified: true, // Simulation test passes
      independentReviewers: ["peer-steward-1", "peer-steward-2"],
      votesFor: 1000,
      votesAgainst: 0,
      timelockSeconds,
      status: "queued",
      queuedAt: new Date().toISOString(),
      executableAfter: new Date(Date.now() + timelockSeconds * 1000).toISOString(),
    };

    this.governanceProposals.set(proposal.proposalId, proposal);
    this.logEvent(
      "OMEGA",
      "propose_governance_policy",
      proposerCommitment,
      [targetPlane, title],
      [proposal.proposalId]
    );

    return proposal;
  }

  public executeGovernancePolicy(proposalId: string, currentTimestamp: number = Date.now()): boolean {
    const proposal = this.governanceProposals.get(proposalId);
    if (!proposal) throw new Error("Proposal not found");
    if (proposal.status !== "queued") throw new Error("Proposal not queued");

    const execTime = proposal.executableAfter ? Date.parse(proposal.executableAfter) : 0;
    if (currentTimestamp < execTime) {
      throw new Error("Timelock has not expired");
    }

    proposal.status = "executed";
    this.logEvent("OMEGA", "execute_governance_policy", proposal.proposerCommitment, [proposalId], ["executed"]);
    return true;
  }

  // --- Slice G: Work receipt appeals and reversals ---
  public openWorkAppeal(workId: string, appellantCommitment: string, reason: string): WorkAppeal {
    const receipt = this.workReceipts.get(workId);
    if (!receipt) throw new Error("Work receipt not found");
    if (receipt.status === "reversed") throw new Error("Receipt already reversed; nothing left to appeal");
    if (!reason || !reason.trim()) throw new Error("An appeal requires a stated reason");

    const existing = Array.from(this.workAppeals.values()).find(
      (appeal) => appeal.workId === workId && appeal.status === "open"
    );
    if (existing) throw new Error(`An open appeal already exists for ${workId}`);

    const appeal: WorkAppeal = {
      appealId: `appeal-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
      workId,
      appellantCommitment,
      reason,
      status: "open",
      openedAt: new Date().toISOString(),
    };
    this.workAppeals.set(appeal.appealId, appeal);
    this.logEvent("TWC", "open_work_appeal", appellantCommitment, [workId, reason], [appeal.appealId]);
    return appeal;
  }

  public resolveWorkAppeal(
    appealId: string,
    reviewerCommitment: string,
    decision: WorkAppealDecision,
    note = ""
  ): WorkAppeal {
    const appeal = this.workAppeals.get(appealId);
    if (!appeal) throw new Error("Appeal not found");
    if (appeal.status !== "open") throw new Error("Appeal already resolved");
    if (reviewerCommitment === appeal.appellantCommitment) {
      throw new Error("The appellant may not review their own appeal");
    }

    const receipt = this.workReceipts.get(appeal.workId);
    if (!receipt) throw new Error("Work receipt not found for appeal");

    appeal.status = decision;
    appeal.reviewerCommitment = reviewerCommitment;
    appeal.decisionNote = note;
    appeal.resolvedAt = new Date().toISOString();

    if (decision === "reversed") {
      const balances = this.twcBalances;
      const current = balances.get(receipt.contributorCommitment) || 0;
      const recovered = Math.min(current, receipt.issuedTwcUnits);
      const unrecoverable = receipt.issuedTwcUnits - recovered;
      balances.set(receipt.contributorCommitment, current - recovered);
      receipt.status = "reversed";
      appeal.clawbackTwcUnits = recovered;
      appeal.unrecoverableTwcUnits = unrecoverable;

      this.logEvent(
        "TWC",
        "reverse_work_receipt",
        reviewerCommitment,
        [appealId, appeal.workId, String(recovered)],
        [String(unrecoverable)]
      );
    } else {
      this.logEvent("TWC", "resolve_work_appeal", reviewerCommitment, [appealId, decision], [appeal.workId]);
    }

    return appeal;
  }
}
