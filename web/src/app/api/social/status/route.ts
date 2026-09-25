import { NextResponse } from "next/server";

import { getNostrIntegrationStatus } from "@/lib/nostr";

/**
 * GET /api/social/status
 * Reports the Nostr integration surface the (separately developed) Nostr client
 * plugs into: relays, publisher identity, economy event kinds, and the checklist
 * the client must satisfy. Fails closed until relays are configured.
 */
export async function GET() {
  const status = getNostrIntegrationStatus();
  return NextResponse.json({
    ok: true,
    surface: "omega-economy-nostr-v0",
    ...status,
    consumers: [
      { route: "/care", uses: "care attestation summaries, consent-scoped only" },
      { route: "/testnet", uses: "work receipts and governance mirrors for public discussion" },
      { route: "/novel", uses: "reader notes (future)" },
    ],
  });
}
