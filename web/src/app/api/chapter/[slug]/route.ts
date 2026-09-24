import { NextRequest, NextResponse } from "next/server";
import { chapterBySlug } from "@/lib/book";
import { verifyUnlockProofOnChain } from "@/lib/unlock-server";

/**
 * POST /api/chapter/[slug]
 * Body: { message, signature }
 *
 * Verifies either a $OMEGA or TWC unlock proof server-side, independently checks
 * the configured on-chain rail, and only returns chapter HTML to verified
 * readers. Unverified requests never receive chapter prose.
 */
export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ slug: string }> }
) {
  const { slug } = await params;

  let body: { message?: string; signature?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON body." }, { status: 400 });
  }

  let proof;
  try {
    proof = await verifyUnlockProofOnChain(body);
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : "Unlock verification failed." },
      { status: 400 }
    );
  }

  const chapter = chapterBySlug(slug);
  if (!chapter) {
    return NextResponse.json({ ok: false, error: "Chapter not found." }, { status: 404 });
  }

  if (chapter.number > proof.unlocked) {
    return NextResponse.json(
      {
        ok: false,
        error: `Chapter ${chapter.number} is not unlocked for this proof.`,
        unlocked: proof.unlocked,
        address: proof.address,
        currency: proof.currency,
        network: proof.network,
      },
      { status: 403 }
    );
  }

  return NextResponse.json({
    ok: true,
    address: proof.address,
    currency: proof.currency,
    network: proof.network,
    unlocked: proof.unlocked,
    chain: proof.chain,
    number: chapter.number,
    title: chapter.title,
    html: chapter.html,
  });
}
