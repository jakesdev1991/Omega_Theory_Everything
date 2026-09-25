// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
import { NextResponse } from "next/server";

import { buildWalletBundle, WALLET_BUNDLE_FILE_NAME } from "@/lib/wallet-artifacts";

export const dynamic = "force-dynamic";

/**
 * GET /api/wallet/download
 * Streams the offline Omega Wallet bundle (the same GUI served at
 * /omega-wallet/, plus local launchers, checksums, and the product license).
 * The ZIP is built deterministically on every request, so the SHA-256 published
 * by /api/wallet/manifest always matches what this route returns.
 */
export async function GET() {
  try {
    const bundle = await buildWalletBundle();
    const headers = new Headers();
    headers.set("content-type", "application/zip");
    headers.set("content-disposition", `attachment; filename="${bundle.fileName}"`);
    headers.set("content-length", String(bundle.bytes.length));
    headers.set("x-wallet-sha256", bundle.sha256);
    headers.set("cache-control", "no-store");
    return new NextResponse(Buffer.from(bundle.bytes), { headers });
  } catch (error) {
    const payload = {
      ok: false,
      error: error instanceof Error ? error.message : "Unable to build the wallet bundle.",
      hint: "Run `npm run sync:wallet` in web/ first.",
      fileName: WALLET_BUNDLE_FILE_NAME,
    };
    return NextResponse.json(payload, { status: 500 });
  }
}
