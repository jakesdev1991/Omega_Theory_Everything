import { NextRequest, NextResponse } from "next/server";
import { verifyMessage } from "ethers";

/**
 * POST /api/unlock
 * Body: { message, signature, tier }
 * Verifies an EIP-191 personal_sign signature over the unlock statement and
 * returns which chapters the signer may read. Verification is cryptographic:
 * the recovered address is derived from the signature itself — no trust in the
 * client-supplied address.
 */
export async function POST(req: NextRequest) {
  let body: { message?: string; signature?: string; tier?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body." }, { status: 400 });
  }

  const { message, signature, tier } = body;
  if (!message || !signature) {
    return NextResponse.json(
      { ok: false, error: "message and signature are required." },
      { status: 400 }
    );
  }

  // The message must be a genuine unlock statement, not arbitrary text.
  if (!message.startsWith("OMEGA TRI-TOKEN ECONOMY — RELEASE-DAY NOVEL UNLOCK")) {
    return NextResponse.json(
      { ok: false, error: "Unexpected message preamble." },
      { status: 400 }
    );
  }

  let address: string;
  try {
    address = verifyMessage(message, signature);
  } catch {
    return NextResponse.json(
      { ok: false, error: "Signature verification failed." },
      { status: 400 }
    );
  }

  const tierId = typeof tier === "string" && tier.startsWith("tier-") ? tier : "tier-1";
  const tierNum = parseInt(tierId.replace("tier-", ""), 10);

  // Chapter unlock map: how many of the 16 chapters each tier opens.
  const UNLOCKED_CHAPTERS: Record<number, number> = {
    1: 1,
    2: 6,
    3: 12,
    4: 15,
    5: 16,
  };
  const chapters = UNLOCKED_CHAPTERS[tierNum] ?? 1;

  return NextResponse.json({
    ok: true,
    address,
    tier: tierId,
    chapters,
    message: `Verified. Address ${address} unlocked ${chapters} of 16 chapters.`,
  });
}
