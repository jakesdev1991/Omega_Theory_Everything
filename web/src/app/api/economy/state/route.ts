import { NextResponse } from "next/server";
import { TriTokenEngine } from "@/lib/domain/engine";
import { VerifierAdapters } from "@/lib/domain/adapters";

// Persistent in-memory engine singleton for local dev server
const globalForEngine = globalThis as unknown as {
  triTokenEngine?: TriTokenEngine;
};

export const engineInstance = globalForEngine.triTokenEngine ?? new TriTokenEngine();
if (process.env.NODE_ENV !== "production") globalForEngine.triTokenEngine = engineInstance;

/**
 * GET /api/economy/state
 * Returns the multi-plane integration snapshot across CARE, TWC, $OMEGA, and AMITY.
 */
export async function GET() {
  const participants = Array.from(engineInstance.participants.values());
  const budgets = Array.from(engineInstance.budgets.values());
  const proposals = Array.from(engineInstance.workProposals.values());
  const receipts = Array.from(engineInstance.workReceipts.values());
  const hardships = Array.from(engineInstance.hardshipRequests.values());
  const conversions = Array.from(engineInstance.conversionRequests.values());
  const governance = Array.from(engineInstance.governanceProposals.values());
  const recentEvents = engineInstance.auditEvents.slice(-20);

  const balances = {
    care: Object.fromEntries(engineInstance.careBalances),
    twc: Object.fromEntries(engineInstance.twcBalances),
    omega: Object.fromEntries(engineInstance.omegaBalances),
    amity: Object.fromEntries(engineInstance.amityBalances),
  };

  return NextResponse.json({
    ok: true,
    policyVersion: "tri-token-v1-prototype",
    participants,
    budgets,
    balances,
    proposals,
    receipts,
    hardships,
    conversions,
    governance,
    recentEvents,
  });
}

/**
 * POST /api/economy/state
 * Handles interactive simulations:
 * - work_propose & work_execute (TWC Proof of Useful Work)
 * - hardship_request & hardship_sponsor (CARE solidarity)
 * - care_amity_convert (AMITY bounded conversion)
 * - governance_propose & governance_execute ($OMEGA timelock)
 * - resource_update (Device quota to status)
 */
export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { action } = body;

    switch (action) {
      case "resource_update": {
        const { participantId, deltaUnits, paused } = body;
        const res = engineInstance.updateResourceContribution(participantId, Number(deltaUnits || 0), !!paused);
        return NextResponse.json({ ok: true, result: res });
      }

      case "reach_record": {
        const { participantId, audiences, attentions } = body;
        const res = engineInstance.recordReachEvent(participantId, Number(audiences || 0), Number(attentions || 0));
        return NextResponse.json({ ok: true, result: res });
      }

      case "distribute_pool": {
        const res = engineInstance.distributeWeeklyCarePool();
        return NextResponse.json({ ok: true, allocations: Object.fromEntries(res) });
      }

      case "hardship_request": {
        const { participantId, uncoveredUnits } = body;
        const res = engineInstance.requestHardship(participantId, Number(uncoveredUnits || 40));
        return NextResponse.json({ ok: true, hardship: res });
      }

      case "hardship_sponsor": {
        const { requestId, sponsorId } = body;
        const res = engineInstance.sponsorHardship(requestId, sponsorId);
        return NextResponse.json({ ok: true, sponsored: res });
      }

      case "work_propose_and_execute": {
        const { contributorCommitment, workClass, objective, resourceBudget } = body;
        const prop = engineInstance.proposeWork(
          contributorCommitment || "demo-user",
          workClass || "engineering_protocol",
          objective || "Formalize test adapter",
          ["kernel_checked", "audit_axioms"],
          Number(resourceBudget || 100)
        );

        let adapter = VerifierAdapters.engineering;
        if (workClass === "lean_formalization") adapter = VerifierAdapters.leanFormalization;
        else if (workClass === "zk_proving") adapter = VerifierAdapters.zkProving;
        else if (workClass === "infrastructure_resource") adapter = VerifierAdapters.infrastructure;
        else if (workClass === "physics_simulation") adapter = VerifierAdapters.physicsSimulation;

        const receipt = engineInstance.executeAndVerifyWork(
          prop.workId,
          `0xartifact-${Date.now()}`,
          { cpuSeconds: 15 },
          adapter
        );

        return NextResponse.json({ ok: true, proposal: prop, receipt });
      }

      case "care_amity_convert": {
        const { participantId, careAmount, extremeHoldback } = body;
        const conv = engineInstance.requestCareToAmityConversion(
          participantId,
          Number(careAmount),
          !!extremeHoldback
        );
        return NextResponse.json({ ok: true, conversion: conv });
      }

      case "governance_propose": {
        const { proposerCommitment, targetPlane, title, description, diff, timelockSeconds } = body;
        const gov = engineInstance.proposeGovernancePolicy(
          proposerCommitment || "demo-user",
          targetPlane || "OMEGA",
          title || "Community Parameter Proposal",
          description || "Updating parameter bounds",
          diff || {},
          Number(timelockSeconds || 10)
        );
        return NextResponse.json({ ok: true, proposal: gov });
      }

      case "governance_execute": {
        const { proposalId, timestamp } = body;
        const executed = engineInstance.executeGovernancePolicy(proposalId, timestamp ? Number(timestamp) : Date.now());
        return NextResponse.json({ ok: true, executed });
      }

      default:
        return NextResponse.json({ ok: false, error: `Unknown action: ${action}` }, { status: 400 });
    }
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Error executing action" },
      { status: 500 }
    );
  }
}
