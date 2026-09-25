import { NextResponse } from "next/server";

import { getAmityScaffoldStatus } from "@/lib/amity-server";
import { economyTestModeEnabled, getEngine } from "@/lib/domain/server-engine";
import { scenarioSummaries } from "@/lib/domain/scenarios";
import { getNostrIntegrationStatus } from "@/lib/nostr";
import { getUnlockRailStatus } from "@/lib/unlock-server";
import { readWalletBuildInfo, WALLET_LAUNCH_URL } from "@/lib/wallet-artifacts";

export const dynamic = "force-dynamic";

export interface HealthCheck {
  id: string;
  label: string;
  status: "pass" | "warn" | "fail";
  detail: string;
}

/**
 * GET /api/economy/health
 * Single readiness report for the Test Console: rails, scaffold, wallet GUI,
 * Nostr surface, test controls, and the offline scenario suite inventory.
 */
export async function GET() {
  const checks: HealthCheck[] = [];
  const engine = getEngine();

  const rails = await getUnlockRailStatus();
  checks.push({
    id: "rail-omega",
    label: "$OMEGA EVM unlock rail",
    status: rails.rails.omega.configured ? "pass" : "warn",
    detail: rails.rails.omega.message,
  });
  checks.push({
    id: "rail-twc",
    label: "TWC Solana unlock rail",
    status: rails.rails.twc.configured ? "pass" : "warn",
    detail: rails.rails.twc.message,
  });

  let amityStatus: unknown = null;
  try {
    const amity = await getAmityScaffoldStatus();
    amityStatus = amity;
    checks.push({
      id: "rail-amity",
      label: "AMITY Bitcoin/Lightning scaffold",
      status: "warn",
      detail: "Separate scaffold workstream — operator visibility only, not a live unlock rail.",
    });
  } catch (error) {
    checks.push({
      id: "rail-amity",
      label: "AMITY Bitcoin/Lightning scaffold",
      status: "fail",
      detail: error instanceof Error ? error.message : "AMITY scaffold status unavailable.",
    });
  }

  const walletBuild = await readWalletBuildInfo();
  checks.push({
    id: "wallet-gui",
    label: "Wallet GUI served by this site",
    status: walletBuild ? "pass" : "fail",
    detail: walletBuild
      ? `Synced ${walletBuild.version} (${walletBuild.channel}) at ${WALLET_LAUNCH_URL} on ${walletBuild.generatedAt}.`
      : "public/omega-wallet is missing. Run `npm run sync:wallet` in web/.",
  });

  const nostr = getNostrIntegrationStatus();
  checks.push({
    id: "nostr",
    label: "Nostr social layer",
    status: nostr.configured ? "pass" : "warn",
    detail: nostr.configured
      ? `${nostr.relays.length} relay(s) configured; awaiting the Nostr client.`
      : "Not configured (set NOSTR_RELAYS). The economy social layer stays local until the client lands.",
  });

  const testMode = economyTestModeEnabled();
  checks.push({
    id: "test-controls",
    label: "Economy test controls (faucet, credits, reset)",
    status: testMode ? "pass" : "warn",
    detail: testMode
      ? "Enabled — this deployment allows valueless test grants on the in-memory ledger."
      : "Disabled — production build without ECONOMY_TEST_MODE=1. Scenarios still run offline.",
  });

  const scenarios = scenarioSummaries();
  checks.push({
    id: "scenario-suite",
    label: "Offline economy scenario suite",
    status: scenarios.length >= 9 ? "pass" : "fail",
    detail: `${scenarios.length} scenarios / ${scenarios.reduce((sum, s) => sum + s.stepCount, 0)} steps available via POST /api/economy/scenarios.`,
  });

  const snapshot = engine.snapshot();
  const ledgerStats = {
    participants: snapshot.participants.length,
    receipts: snapshot.receipts.length,
    appeals: snapshot.appeals.length,
    conversions: snapshot.conversions.length,
    governanceProposals: snapshot.governance.length,
    auditEvents: snapshot.auditEvents.length,
    balances: snapshot.balances,
  };

  const failed = checks.filter((check) => check.status === "fail");
  const warned = checks.filter((check) => check.status === "warn");

  return NextResponse.json({
    ok: true,
    readiness: failed.length === 0 ? (warned.length === 0 ? "ready" : "ready-with-warnings") : "blocked",
    policyVersion: snapshot.policyVersion,
    testMode,
    checks,
    ledger: ledgerStats,
    rails,
    amity: amityStatus,
    nostr,
    scenarios: scenarios.map((scenario) => scenario.id),
  });
}
