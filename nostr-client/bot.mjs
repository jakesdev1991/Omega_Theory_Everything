#!/usr/bin/env node
// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * Omega funding/promo bot — mention-responder ONLY.
 *
 * Anti-spam contract (deliberately conservative):
 *   - replies only to kind-1 notes that tag the root key (#p) or contain
 *     its npub in content,
 *   - at most one reply per pubkey per 24h (persistent JSON state),
 *   - never DMs strangers, never posts unsolicited, no scraping,
 *   - NIP-57 lightningAddress tag ONLY when OMEGA_NOSTR_LN_ADDRESS is set
 *     (fails closed otherwise: no zap tag),
 *   - honest pitch: cites the Lean corpus status factually.
 *
 * The pitch (fixed text, no LLM needed at this stage):
 *   - the repo, 54 Lean 4 volumes kernel-checked vs Mathlib v4.32.0,
 *     0 sorry / 0 trivial,
 *   - C.A.R.E. economy blueprint, Crucible novel, app store over Nostr,
 *   - funding welcome via the zap tag when configured.
 */

import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { loadPolicy } from "./lib/policy.mjs";
import { RelayPool } from "./lib/relay.mjs";
import { signEvent, verifyEventStrict } from "./lib/events.mjs";
import { hexToNpub } from "./lib/bech32.mjs";

const here = dirname(fileURLToPath(import.meta.url));

export const REPO_URL = "https://github.com/jakesdev1991/Omega_Theory_Everything";

export const PITCH = [
  "The Omega Theory of Everything: a unified physics model whose math is",
  "formally verified — 54 Lean 4 volumes kernel-checked against Mathlib",
  "v4.32.0, zero sorry, zero trivial proofs. The same repo specifies the",
  "C.A.R.E. economy (Call About Resuscitating Everyone), the novel",
  "Crucible: The Satoshi Protocol, and an app store that runs over Nostr.",
  `Repo: ${REPO_URL}`,
  "Funding welcome via zaps if configured — see the repo's /invest page.",
].join(" ");

/** 24h dedup state, injectable clock for tests. */
export class ReplyLedger {
  constructor(stateFile, { now = () => Date.now() } = {}) {
    this.stateFile = stateFile;
    this.now = now;
    this.replied = new Map(); // pubkey -> timestamp ms
    this._load();
  }

  _load() {
    try {
      const raw = JSON.parse(readFileSync(this.stateFile, "utf8"));
      for (const [k, v] of Object.entries(raw.replied ?? {})) this.replied.set(k, v);
    } catch {
      /* fresh state */
    }
  }

  _save() {
    mkdirSync(dirname(this.stateFile), { recursive: true });
    writeFileSync(this.stateFile, JSON.stringify({ replied: Object.fromEntries(this.replied) }, null, 2));
  }

  mayReply(pubkey, cooldownMs = 24 * 3600 * 1000) {
    const last = this.replied.get(pubkey) ?? 0;
    return this.now() - last >= cooldownMs;
  }

  markReplied(pubkey) {
    this.replied.set(pubkey, this.now());
    this._save();
  }
}

/** Decide whether an incoming event is a mention worth answering. */
export function isMention(event, rootPubkeyHex, rootNpub) {
  if (event.kind !== 1) return false;
  if (!verifyEventStrict(event)) return false;
  const tagsP = event.tags.filter((t) => t[0] === "p").map((t) => t[1]);
  if (tagsP.includes(rootPubkeyHex)) return true;
  return event.content.includes(rootNpub);
}

export function buildReply(event, rootSecretHex, lnAddress) {
  const tags = [
    ["p", event.pubkey],
    ["e", event.id],
    ["r", REPO_URL],
  ];
  if (lnAddress) tags.push(["lnurl", "lightningAddress", lnAddress, "NIP-57 zap"]);
  return signEvent(
    {
      kind: 1,
      created_at: Math.floor(Date.now() / 1000),
      tags,
      content: PITCH,
    },
    rootSecretHex,
  );
}

export async function main() {
  const policy = loadPolicy();
  const rootNpub = hexToNpub(policy.rootPubkeyHex);
  const ledger = new ReplyLedger(policy.stateFile);
  console.log(`omega bot: watching for mentions of ${rootNpub} on ${policy.relays.join(", ")}`);

  const pool = new RelayPool(policy.relays, policy.rootSecretHex);
  pool.on("connected", (u) => console.log(`connected: ${u}`));
  pool.connect();

  pool.subscribe(
    { kinds: [1], "#p": [policy.rootPubkeyHex], since: Math.floor(Date.now() / 1000) - 60 },
    {
      onEvent: (ev) => {
        if (!isMention(ev, policy.rootPubkeyHex, rootNpub)) return;
        if (!ledger.mayReply(ev.pubkey)) {
          console.log(`skip (cooldown): ${ev.pubkey.slice(0, 8)}…`);
          return;
        }
        const reply = buildReply(ev, policy.rootSecretHex, policy.lnAddress);
        pool
          .publish(reply)
          .then((res) => {
            ledger.markReplied(ev.pubkey);
            console.log(`replied to ${ev.pubkey.slice(0, 8)}… (${reply.id.slice(0, 8)}… on ${res.relay})`);
          })
          .catch((err) => console.error(`reply failed: ${err.message}`));
      },
    },
  );

  console.log("bot: listening (Ctrl+C to stop)");
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error(err.message);
    process.exit(1);
  });
}
