import { NextRequest, NextResponse } from "next/server";

import { VerifierAdapters } from "@/lib/domain/adapters";
import { getEngine } from "@/lib/domain/server-engine";
import type {
  BalanceCurrency,
  CarePrivacyLevel,
  CareRole,
  ConsentRecord,
  Plane,
  WorkAppealDecision,
} from "@/lib/domain/types";

/**
 * GET  /api/economy/state — multi-plane integration snapshot (CARE/TWC/$OMEGA/AMITY)
 * POST /api/economy/state — interactive economy actions used by the workbench and
 *                           the Test Console. Every action is an audited mutation.
 */

export async function GET() {
  const engine = getEngine();
  const snapshot = engine.snapshot();

  return NextResponse.json({
    ok: true,
    ...snapshot,
    recentEvents: engine.auditEvents.slice(-20),
  });
}

function badRequest(error: unknown) {
  return NextResponse.json(
    { ok: false, error: error instanceof Error ? error.message : "Invalid request." },
    { status: 400 },
  );
}

export async function POST(req: NextRequest) {
  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body." }, { status: 400 });
  }

  const engine = getEngine();
  const { action } = body;

  try {
    switch (action) {
      case "register_participant": {
        const { participantId, role, privacyLevel } = body as {
          participantId?: string;
          role?: CareRole;
          privacyLevel?: CarePrivacyLevel;
        };
        if (!participantId) return badRequest(new Error("participantId is required"));
        const credential = engine.registerParticipant(String(participantId), role, privacyLevel);
        return NextResponse.json({ ok: true, result: credential });
      }

      case "consent_record": {
        const { participantId, scope, granted } = body as {
          participantId?: string;
          scope?: ConsentRecord["scope"];
          granted?: boolean;
        };
        if (!participantId || !scope) return badRequest(new Error("participantId and scope are required"));
        const record = engine.recordConsent(String(participantId), scope, granted !== false);
        return NextResponse.json({ ok: true, result: record });
      }

      case "resource_update": {
        const { participantId, deltaUnits, paused, consent } = body as {
          participantId?: string;
          deltaUnits?: number;
          paused?: boolean;
          consent?: boolean;
        };
        if (!participantId) return badRequest(new Error("participantId is required"));
        if (consent === true && !engine.hasConsent(String(participantId), "resource_contribution")) {
          engine.recordConsent(String(participantId), "resource_contribution", true);
        }
        const res = engine.updateResourceContribution(String(participantId), Number(deltaUnits || 0), !!paused);
        return NextResponse.json({ ok: true, result: res });
      }

      case "reach_record": {
        const { participantId, audiences, attentions } = body;
        const res = engine.recordReachEvent(
          String(participantId || "demo-user"),
          Number(audiences || 0),
          Number(attentions || 0),
        );
        return NextResponse.json({ ok: true, result: res });
      }

      case "distribute_pool": {
        const res = engine.distributeWeeklyCarePool();
        return NextResponse.json({ ok: true, allocations: Object.fromEntries(res) });
      }

      case "hardship_request": {
        const { participantId, uncoveredUnits } = body;
        const res = engine.requestHardship(String(participantId || "demo-user"), Number(uncoveredUnits || 40));
        return NextResponse.json({ ok: true, hardship: res });
      }

      case "hardship_sponsor": {
        const { requestId, sponsorId } = body;
        const res = engine.sponsorHardship(String(requestId), String(sponsorId));
        return NextResponse.json({ ok: true, sponsored: res });
      }

      case "work_propose_and_execute": {
        const { contributorCommitment, workClass, objective, resourceBudget } = body as {
          contributorCommitment?: string;
          workClass?: Parameters<typeof engine.proposeWork>[1];
          objective?: string;
          resourceBudget?: number;
        };
        const contributor = String(contributorCommitment || "demo-user");
        if (!engine.participants.has(contributor)) engine.registerParticipant(contributor);

        const prop = engine.proposeWork(
          contributor,
          workClass || "engineering_protocol",
          String(objective || "Formalize test adapter"),
          ["kernel_checked", "audit_axioms"],
          Number(resourceBudget || 100),
        );

        const adapter =
          workClass === "lean_formalization"
            ? VerifierAdapters.leanFormalization
            : workClass === "zk_proving"
              ? VerifierAdapters.zkProving
              : workClass === "infrastructure_resource"
                ? VerifierAdapters.infrastructure
                : workClass === "physics_simulation"
                  ? VerifierAdapters.physicsSimulation
                  : VerifierAdapters.engineering;

        const receipt = engine.executeAndVerifyWork(
          prop.workId,
          `0xartifact-${Date.now()}`,
          { cpuSeconds: 15 },
          adapter,
        );

        return NextResponse.json({ ok: true, proposal: prop, receipt });
      }

      case "work_appeal_open": {
        const { workId, appellantCommitment, reason } = body;
        const appeal = engine.openWorkAppeal(String(workId), String(appellantCommitment || "watchdog"), String(reason || ""));
        return NextResponse.json({ ok: true, appeal });
      }

      case "work_appeal_resolve": {
        const { appealId, reviewerCommitment, decision, note } = body as {
          appealId?: string;
          reviewerCommitment?: string;
          decision?: WorkAppealDecision;
          note?: string;
        };
        const appeal = engine.resolveWorkAppeal(
          String(appealId),
          String(reviewerCommitment || "archangel-1"),
          decision || "dismissed",
          String(note || ""),
        );
        return NextResponse.json({ ok: true, appeal });
      }

      case "care_amity_convert": {
        const { participantId, careAmount, extremeHoldback } = body;
        const conv = engine.requestCareToAmityConversion(
          String(participantId || "demo-user"),
          Number(careAmount),
          !!extremeHoldback,
        );
        return NextResponse.json({ ok: true, conversion: conv });
      }

      case "governance_propose": {
        const { proposerCommitment, targetPlane, title, description, diff, timelockSeconds } = body;
        const gov = engine.proposeGovernancePolicy(
          String(proposerCommitment || "demo-user"),
          (targetPlane as Plane) || "OMEGA",
          String(title || "Community Parameter Proposal"),
          String(description || "Updating parameter bounds"),
          (diff as Record<string, unknown>) || {},
          Number(timelockSeconds || 10),
        );
        return NextResponse.json({ ok: true, proposal: gov });
      }

      case "governance_execute": {
        const { proposalId, timestamp } = body;
        const executed = engine.executeGovernancePolicy(
          String(proposalId),
          timestamp ? Number(timestamp) : Date.now(),
        );
        return NextResponse.json({ ok: true, executed });
      }

      case "credit_balance": {
        const { currency, participantId, units, reason } = body as {
          currency?: BalanceCurrency;
          participantId?: string;
          units?: number;
          reason?: string;
        };
        if (!engine.options.testMode) {
          return NextResponse.json(
            { ok: false, error: "Balance credits are disabled outside test mode (ECONOMY_TEST_MODE=1)." },
            { status: 403 },
          );
        }
        const balance = engine.creditBalance(
          (currency || "twc") as BalanceCurrency,
          String(participantId || "demo-user"),
          Number(units || 0),
          String(reason || "operator credit"),
        );
        return NextResponse.json({ ok: true, balance });
      }

      default:
        return NextResponse.json({ ok: false, error: `Unknown action: ${String(action)}` }, { status: 400 });
    }
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Error executing action" },
      { status: 500 },
    );
  }
}
