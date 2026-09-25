import { NextRequest, NextResponse } from "next/server";

import { assertEconomyTestMode, economyTestModeEnabled, getEngine } from "@/lib/domain/server-engine";
import type { BalanceCurrency } from "@/lib/domain/types";

const CURRENCIES: BalanceCurrency[] = ["care", "twc", "omega", "amity"];
const MAX_GRANT = 1_000_000;

/**
 * POST /api/economy/faucet
 * Test-mode faucet: grants valueless balances on the shared in-memory ledger so
 * the whole economy can be exercised end to end without touching any chain.
 * Disabled in production builds unless ECONOMY_TEST_MODE=1.
 */
export async function POST(req: NextRequest) {
  if (!economyTestModeEnabled()) {
    return NextResponse.json(
      {
        ok: false,
        error: "The faucet is disabled in this deployment. Set ECONOMY_TEST_MODE=1 to enable test controls.",
      },
      { status: 403 },
    );
  }

  let body: { participantId?: string; currency?: BalanceCurrency; units?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body." }, { status: 400 });
  }

  const participantId = String(body.participantId || "demo-user");
  const currency = (body.currency || "twc") as BalanceCurrency;
  const units = Number(body.units ?? 1000);

  if (!CURRENCIES.includes(currency)) {
    return NextResponse.json({ ok: false, error: `currency must be one of ${CURRENCIES.join(", ")}` }, { status: 400 });
  }
  if (!Number.isFinite(units) || units <= 0 || units > MAX_GRANT) {
    return NextResponse.json({ ok: false, error: `units must be between 1 and ${MAX_GRANT}` }, { status: 400 });
  }

  try {
    assertEconomyTestMode();
    const engine = getEngine();
    if (!engine.participants.has(participantId)) {
      engine.registerParticipant(participantId);
    }
    const balance = engine.faucetGrant(currency, participantId, units);
    return NextResponse.json({
      ok: true,
      participantId,
      currency,
      granted: units,
      balance,
      testMode: engine.options.testMode,
    });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Faucet failed." }, { status: 403 });
  }
}
