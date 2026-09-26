// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

/**
 * GET /.well-known/nostr.json
 * NIP-05 identifier service (https://github.com/nostr-protocol/nips/blob/master/05.md).
 *
 * Serves `{ names: {...}, relays: {...} }`:
 * - `names` comes from NOSTR_NIP05_NAMES ("name=hexpubkey,name2=hexpubkey2").
 *   Fails closed: when the variable is empty/unset we still return 200 with an
 *   empty names object — spec-safe, since unknown names resolve to the same
 *   empty mapping — rather than erroring out on a health probe.
 * - `relays` maps every name in `names` to the NOSTR_RELAYS wss:// list, which
 *   is the optional relay recommendation section of NIP-05.
 *
 * NIP-05 requires CORS: any origin must be able to fetch this document.
 * Content-Type is application/json per the NIP-05 verification flow.
 */
export const dynamic = "force-dynamic";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
} as const;

function parseNames(raw: string | undefined): Record<string, string> {
  const names: Record<string, string> = {};
  if (!raw) {
    return names;
  }
  for (const entry of raw.split(",")) {
    const trimmed = entry.trim();
    if (!trimmed) {
      continue;
    }
    const separator = trimmed.indexOf("=");
    if (separator <= 0 || separator === trimmed.length - 1) {
      continue;
    }
    const name = trimmed.slice(0, separator).trim().toLowerCase();
    const pubkey = trimmed.slice(separator + 1).trim().toLowerCase();
    if (name && pubkey) {
      names[name] = pubkey;
    }
  }
  return names;
}

function parseRelays(raw: string | undefined): string[] {
  return (raw ?? "")
    .split(",")
    .map((relay) => relay.trim())
    .filter((relay) => relay.startsWith("wss://"));
}

export async function GET() {
  const names = parseNames(process.env.NOSTR_NIP05_NAMES);
  const relays = parseRelays(process.env.NOSTR_RELAYS);

  const relayMap: Record<string, string[]> = {};
  for (const name of Object.keys(names)) {
    relayMap[name] = relays;
  }

  return new Response(JSON.stringify({ names, relays: relayMap }), {
    status: 200,
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/json",
    },
  });
}
