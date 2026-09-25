// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
import { NextResponse } from "next/server";

import { assertEconomyTestMode, resetEngine } from "@/lib/domain/server-engine";

/**
 * POST /api/economy/reset
 * Replaces the shared in-memory ledger with a fresh engine. Test control only.
 */
export async function POST() {
  try {
    assertEconomyTestMode();
    const engine = resetEngine();
    return NextResponse.json({ ok: true, reset: true, testMode: engine.options.testMode, participants: engine.participants.size });
  } catch (error) {
    return NextResponse.json({ ok: false, error: error instanceof Error ? error.message : "Reset refused." }, { status: 403 });
  }
}
