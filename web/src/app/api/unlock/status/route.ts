import { NextResponse } from "next/server";

import { getUnlockRailStatus } from "@/lib/unlock-server";

/**
 * GET /api/unlock/status
 * Returns whether the current web app has enough local configuration to perform
 * independent on-chain verification for the wired $OMEGA and TWC rails.
 */
export async function GET() {
  try {
    const status = await getUnlockRailStatus();
    return NextResponse.json({ ok: true, ...status });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Unable to inspect unlock rail status.",
      },
      { status: 500 },
    );
  }
}
