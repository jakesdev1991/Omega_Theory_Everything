import { NextRequest, NextResponse } from "next/server";
import { verifyMessage } from "ethers";
import { chapterBySlug } from "@/lib/book";

/**
 * POST /api/chapter/[slug]
 * Body: { message, signature, tier? }
 *
 * Verifies the EIP-191 signature server-side, derives the signer from the
 * signature itself, maps the claimed tier to an unlocked chapter count, and
 * returns the chapter HTML only if the signer is allowed to read it.
 * Unverified requests never receive chapter prose.
 */
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  const { slug } = await params;

  let body: { message?: string; signature?: string; tier?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body." }, { status: 400 });
  }

  const { message, signature } = body;
  if (!message || !signature) {
    return NextResponse.json(
      { ok: false, error: "message and signature are required." },
      { status: 400 }
    );
  }

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

  // Tier is claimed in the signed message itself, not just the request body.
  const tierMatch = message.match(/Tier: (tier-(\d))/);
  const tierNum = tierMatch ? parseInt(tierMatch[2], 10) : 1;

  const UNLOCKED_CHAPTERS: Record<number, number> = {
    1: 1,
    2: 6,
    3: 12,
    4: 15,
    5: 16,
  };
  const unlocked = UNLOCKED_CHAPTERS[tierNum] ?? 1;

  const chapter = chapterBySlug(slug);
  if (!chapter) {
    return NextResponse.json({ ok: false, error: "Chapter not found." }, { status: 404 });
  }

  if (chapter.number > unlocked) {
    return NextResponse.json(
      {
        ok: false,
        error: `Chapter ${chapter.number} requires a higher participation tier.`,
        unlocked,
        address,
      },
      { status: 403 }
    );
  }

  return NextResponse.json({
    ok: true,
    address,
    tier: `tier-${tierNum}`,
    unlocked,
    number: chapter.number,
    title: chapter.title,
    html: chapter.html,
  });
}
