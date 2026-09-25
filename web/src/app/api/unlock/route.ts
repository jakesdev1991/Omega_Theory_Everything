// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import { NextRequest, NextResponse } from "next/server";
import { verifyUnlockProofOnChain } from "@/lib/unlock-server";

/**
 * POST /api/unlock
 * Body: { message, signature }
 * Verifies either an EVM $OMEGA proof or a Solana TWC proof, then independently
 * checks the configured on-chain rail before returning the unlocked chapter
 * count. The signer address is always derived from the signature itself.
 */
export async function POST(req: NextRequest) {
  let body: { message?: string; signature?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body." }, { status: 400 });
  }

  try {
    const result = await verifyUnlockProofOnChain(body);
    return NextResponse.json({
      ok: true,
      address: result.address,
      currency: result.currency,
      network: result.network,
      chapters: result.unlocked,
      message: result.message,
      chain: result.chain,
    });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unlock verification failed." },
      { status: 400 },
    );
  }
}
