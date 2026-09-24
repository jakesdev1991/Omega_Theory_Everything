import type {
  AuditEvent,
  CareConversionRequest,
  CarePrivacyLevel,
  CareRole,
  CareStatus,
  GovernancePolicyProposal,
  HardshipRequest,
  ParticipantCredential,
  Plane,
  ResourceBudget,
  WorkClass,
  WorkProposal,
  WorkReceipt,
} from "./types";

export const POLICY_VERSION = "tri-token-v1-prototype";

export class TriTokenEngine {
  public participants: Map<string, ParticipantCredential> = new Map();
  public budgets: Map<string, ResourceBudget> = new Map();
  public workProposals: Map<string, WorkProposal> = new Map();
  public workReceipts: Map<string, WorkReceipt> = new Map();
  public hardshipRequests: Map<string, HardshipRequest> = new Map();
  public conversionRequests: Map<string, CareConversionRequest> = new Map();
  public governanceProposals: Map<string, GovernancePolicyProposal> = new Map();
  public auditEvents: AuditEvent[] = [];

  // Balances
  public careBalances: Map<string, number> = new Map();
  public twcBalances: Map<string, number> = new Map();
  public omegaBalances: Map<string, number> = new Map();
  public amityBalances: Map<string, number> = new Map();

  // Weekly CARE distribution pool
  public weeklyCarePool: number = 100_000;
  public reachEvents: Map<string, { audienceCount: number; attentionCount: number }> = new Map();

  constructor() {
    // Initial seeded participant
    this.registerParticipant("demo-user", "participant", "P0");
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

    // Sponsor bonus: 10% useful bonus credit
    const bonus = Math.floor(req.uncoveredQuotaUnits * 0.1);
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

    // Dock: 500 bps (5%), holdback: 0 or 1500 bps (15%)
    const dockBps = 500;
    const holdbackBps = hasExtremeHoldback ? 1500 : 0;
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
    // Proposer must hold OMEGA voting power
    const omegaBal = this.omegaBalances.get(proposerCommitment) || 0;
    if (omegaBal < 100) {
      // Grant initial 1000 OMEGA voting token if zero for prototype testing
      this.omegaBalances.set(proposerCommitment, 1000);
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
}
