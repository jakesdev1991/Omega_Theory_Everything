import { NextResponse } from "next/server";

import { getAmityScaffoldStatus } from "@/lib/amity-server";

/**
 * GET /api/amity/status
 * Returns the separate AMITY testnet scaffold status for operator visibility.
 * This does not make AMITY a third live unlock rail.
 */
export async function GET() {
  try {
    const status = await getAmityScaffoldStatus();
    return NextResponse.json({ ok: true, amity: status });
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Unable to inspect AMITY scaffold status.",
      },
      { status: 500 },
    );
  }
}
