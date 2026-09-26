#!/usr/bin/env node
// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * Omega sovereign nostr client daemon.
 *
 * Publishes the app-store directory (kind 31990, same shape as
 * mobile-node/publish-directory.mjs) signed with the root key, then stays
 * connected and republishes on demand (SIGHUP) or every INTERVAL.
 * Economy event surfaces are exported for tests and future callers.
 *
 * Env (see lib/policy.mjs): OMEGA_NOSTR_RELAYS, OMEGA_NOSTR_ROOT_SECRET,
 * optional OMEGA_NOSTR_LN_ADDRESS, OMEGA_NOSTR_INTERVAL_SEC.
 * Fails closed on any missing requirement.
 */

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { loadPolicy } from "./lib/policy.mjs";
import { RelayPool } from "./lib/relay.mjs";
import { signEvent, getPublicKeyHex } from "./lib/events.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const ALGORITHMS_PATH = resolve(here, "..", "mobile-node", "algorithms.json");

/** Kinds from the web contract (web/src/lib/nostr.ts ECONOMY_NOSTR_KINDS). */
export const ECONOMY_KINDS = {
  participantNote: 1,
  longFormClaim: 30023,
  workReceipt: 31331,
  careAttestation: 31332,
  governanceProposal: 31333,
  auditEvent: 31334,
  storeHandler: 31990,
  storeListing: 30017,
  storeLicense: 31335,
};

/** Build the signed kind-31990 directory events from algorithms.json. */
export function buildDirectoryEvents(algorithms, secretHex) {
  return algorithms.algorithms.map((app) =>
    signEvent(
      {
        kind: ECONOMY_KINDS.storeHandler,
        created_at: Math.floor(Date.now() / 1000),
        tags: [
          ["d", app.id],
          ["k", String(app.kind)],
          ["name", app.name],
          ["workClass", app.workClass],
        ],
        content: JSON.stringify({
          id: app.id,
          kind: app.kind,
          name: app.name,
          workClass: app.workClass,
          timeoutMs: app.timeoutMs,
          maxOutputBytes: app.maxOutputBytes,
          license: app.license ?? null,
        }),
      },
      secretHex,
    ),
  );
}

/** Sign any economy event with the root key. */
export function publishEconomyEvent(kind, content, tags, secretHex) {
  return signEvent(
    {
      kind,
      created_at: Math.floor(Date.now() / 1000),
      tags,
      content,
    },
    secretHex,
  );
}

export async function main() {
  const policy = loadPolicy();
  console.log(
    `omega nostr client: root ${policy.rootPubkeyHex.slice(0, 16)}… ` +
      `relays ${policy.relays.join(", ")}`,
  );

  const algorithms = JSON.parse(readFileSync(ALGORITHMS_PATH, "utf8"));
  const events = buildDirectoryEvents(algorithms, policy.rootSecretHex);
  console.log(`directory: ${events.length} store handler(s) to publish (kind 31990)`);

  const pool = new RelayPool(policy.relays, policy.rootSecretHex);
  pool.on("connected", (u) => console.log(`connected: ${u}`));
  pool.on("disconnected", (u) => console.log(`disconnected: ${u} (will retry)`));
  pool.on("notice", (u, n) => console.log(`notice from ${u}: ${n}`));

  pool.connect();

  const publishAll = async (why) => {
    for (const ev of events) {
      try {
        const res = await pool.publish(ev);
        console.log(`[directory ${why}] ${ev.id.slice(0, 8)}… OK on ${res.relay}`);
      } catch (err) {
        console.error(`[directory ${why}] FAILED: ${err.message}`);
      }
    }
  };

  await publishAll("startup");

  const intervalSec = Number(process.env.OMEGA_NOSTR_INTERVAL_SEC ?? 0);
  if (intervalSec > 0) {
    setInterval(() => publishAll("refresh"), intervalSec * 1000).unref?.();
  }
  process.on("SIGHUP", () => publishAll("sighup"));

  console.log("client daemon: staying connected (Ctrl+C to stop)");
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error(err.message);
    process.exit(1);
  });
}
