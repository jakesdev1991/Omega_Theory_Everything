import { NextResponse } from "next/server";

import { getWalletManifest } from "@/lib/wallet-artifacts";

/**
 * GET /api/wallet/manifest
 * Everything the download page needs: versions, per-file SHA-256 hashes for the
 * served web app, the offline bundle hash, PWA installability, and the status of
 * native desktop builds.
 */
export async function GET() {
  try {
    return NextResponse.json(await getWalletManifest());
  } catch (error) {
    return NextResponse.json(
      {
        ok: false,
        error: error instanceof Error ? error.message : "Unable to build the wallet manifest.",
        hint: "Run `npm run sync:wallet` in web/ to regenerate public/omega-wallet/.",
      },
      { status: 500 },
    );
  }
}
