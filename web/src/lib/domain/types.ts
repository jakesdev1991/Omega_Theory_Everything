// Shared domain types for the 4 economic planes: CARE, TWC, OMEGA, AMITY
// Reference: docs/tri-token-integration-v1.md and docs/care-architecture-v2.md

export type Plane = "CARE" | "TWC" | "OMEGA" | "AMITY";

export type CarePrivacyLevel = "P0" | "P1" | "P2" | "P3";
export type CareRole = "participant" | "X1" | "X2" | "archangel";
export type CareStatus = "active" | "grace" | "under_quota" | "hardship";

export interface ParticipantCredential {
  participantId: string;
  role: CareRole;
  privacyLevel: CarePrivacyLevel;
  status: CareStatus;
  joinedEpoch: number;
}

export interface ConsentRecord {
  consentId: string;
  participantId: string;
  scope: "resource_contribution" | "circle_sharing" | "service_attestation";
  granted: boolean;
  timestamp: string;
}

export interface ResourceBudget {
  participantId: string;
  budgetBps: number; // e.g. 200 bps = 2.0%
  minimumQuotaUnits: number;
  selectedUnits: number;
  contributedUnits: number;
  paused: boolean;
}

export type WorkClass =
  | "engineering_protocol"
  | "lean_formalization"
  | "zk_proving"
  | "infrastructure_resource"
  | "physics_simulation";

export interface WorkProposal {
  workId: string;
  contributorCommitment: string;
  workClass: WorkClass;
  objective: string;
  acceptanceTests: string[];
  inputCommitments: string[];
  resourceBudget: number;
  reviewPolicy: "deterministic_check" | "independent_reviewer" | "dual_review";
  expirySeconds: number;
}

export interface VerificationResult {
  verified: boolean;
  adapterName: string;
  details: Record<string, unknown>;
  verifiedAt: string;
}

export interface WorkReceipt {
  workId: string;
  workClass: WorkClass;
  contributorCommitment: string;
  artifactHash: string;
  resourceReceipts: {
    cpuSeconds?: number;
    bandwidthBytes?: number;
    memoryMb?: number;
    storageBytes?: number;
  };
  verifierSet: string[];
  verificationResult: VerificationResult;
  policyVersion: string;
  issuedTwcUnits: number;
  settledAt: string;
}

export interface HardshipRequest {
  requestId: string;
  participantId: string;
  uncoveredQuotaUnits: number;
  status: "open" | "sponsored" | "fulfilled";
  sponsoredBy?: string;
  sponsorBonusUnits?: number;
  createdAt: string;
}

export interface CareConversionRequest {
  conversionId: string;
  participantId: string;
  careAmount: number;
  participantDockBps: number; // 500 bps (5%) default, max 2000 bps (20%)
  unissuedHoldbackBps: number; // 0 to 1500 bps (15%)
  netAmityAmount: number;
  fundedSettlementVerified: boolean;
  requestedAt: string;
  settledAt?: string;
}

export interface GovernancePolicyProposal {
  proposalId: string;
  proposerCommitment: string;
  targetPlane: Plane;
  title: string;
  description: string;
  parametersDiff: Record<string, unknown>;
  simulationVerified: boolean;
  independentReviewers: string[];
  votesFor: number;
  votesAgainst: number;
  timelockSeconds: number;
  status: "draft" | "review" | "active" | "queued" | "executed" | "rejected";
  queuedAt?: string;
  executableAfter?: string;
}

export interface AuditEvent {
  eventId: string;
  plane: Plane;
  action: string;
  policyVersion: string;
  actorCommitment: string;
  inputCommitments: string[];
  outputCommitments: string[];
  evidenceReferences: string[];
  timestamp: string;
}
