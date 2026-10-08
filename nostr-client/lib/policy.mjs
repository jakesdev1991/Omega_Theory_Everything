// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * Fail-closed environment policy for the Omega nostr client.
 * Mirrors mobile-node/lib/policy.mjs strictness: no defaults that fake
 * success, no silent fallbacks; a missing requirement throws with a
 * message that says exactly what to set.
 */

import { homedir } from "node:os";
import { join } from "node:path";

import { getPublicKeyHex } from "./events.mjs";
import { nsecToHex } from "./bech32.mjs";

export const REQUIRED_ENV = ["OMEGA_NOSTR_RELAYS", "OMEGA_NOSTR_ROOT_SECRET"];

export class ClientPolicy {
  constructor({ relays, rootSecretHex, rootPubkeyHex, lnAddress, stateFile }) {
    this.relays = relays;
    this.rootSecretHex = rootSecretHex;
    this.rootPubkeyHex = rootPubkeyHex;
    this.lnAddress = lnAddress;
    this.stateFile = stateFile;
  }
}

const ACCEPTABLE_PROTOCOLS = new Set(["ws:", "wss:"]);
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);

export function isAcceptableRelayUrl(url) {
  try {
    const parsed = new URL(url);
    if (!ACCEPTABLE_PROTOCOLS.has(parsed.protocol)) return false;
    // wss:// required off-loopback (matches the web contract); plain ws://
    // is allowed for loopback hosts only (local bench + integration tests).
    if (parsed.protocol === "ws:" && !LOOPBACK_HOSTS.has(parsed.hostname)) {
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

export function parseRelayList(raw) {
  return (raw ?? "")
    .split(",")
    .map((r) => r.trim())
    .filter((r) => r.length > 0 && isAcceptableRelayUrl(r));
}

export function loadPolicy(env = process.env) {
  const missing = REQUIRED_ENV.filter((key) => env[key] === undefined || env[key] === null);
  if (missing.length > 0) {
    throw new Error(
      `omega nostr client: missing required env ${missing.join(", ")} — ` +
        `set them in ~/.config/omega-nostr/keys/env.sh (see deploy/README.md). ` +
        `Failing closed; nothing was published.`,
    );
  }

  const relays = parseRelayList(env.OMEGA_NOSTR_RELAYS);
  if (relays.length === 0) {
    throw new Error(
      "omega nostr client: OMEGA_NOSTR_RELAYS contains no acceptable relay URL " +
        "(wss:// required off-loopback, ws:// allowed for 127.0.0.1/localhost/::1). Failing closed.",
    );
  }

  let rootSecretHex;
  try {
    rootSecretHex = env.OMEGA_NOSTR_ROOT_SECRET.startsWith("nsec")
      ? nsecToHex(env.OMEGA_NOSTR_ROOT_SECRET)
      : env.OMEGA_NOSTR_ROOT_SECRET.trim();
  } catch (err) {
    throw new Error(
      `omega nostr client: OMEGA_NOSTR_ROOT_SECRET is not a valid hex/nsec key: ${err.message}`,
    );
  }
  if (!/^[0-9a-fA-F]{64}$/.test(rootSecretHex)) {
    throw new Error("omega nostr client: OMEGA_NOSTR_ROOT_SECRET must be 32-byte hex or nsec.");
  }
  rootSecretHex = rootSecretHex.toLowerCase();

  return new ClientPolicy({
    relays,
    rootSecretHex,
    rootPubkeyHex: getPublicKeyHex(rootSecretHex),
    lnAddress: env.OMEGA_NOSTR_LN_ADDRESS?.trim() || null,
    stateFile: env.OMEGA_NOSTR_STATE_FILE ?? defaultStateFile(env),
  });
}

/**
 * Where the reply ledger lives when OMEGA_NOSTR_STATE_FILE is not set.
 *
 * This used to fall back to the literal string "/home/jake", which is this
 * project's developer's home directory: on any other machine the client would
 * write its state into a path that either does not exist or belongs to someone
 * else. Fail closed instead, the way the rest of this file does — ask the OS for
 * the home directory, and if even that is unavailable, say exactly what to set.
 *
 * `env` is the object `loadPolicy` was handed (its contract is that it reads the
 * environment it is given, and the test suite relies on that); `process.env` is
 * only consulted when the caller's environment does not carry HOME.
 */
function defaultStateFile(env) {
  const home = env.HOME?.trim() || process.env.HOME?.trim() || homedir();
  if (!home) {
    throw new Error(
      "omega nostr client: cannot determine a home directory for the reply-ledger " +
        "state file; set OMEGA_NOSTR_STATE_FILE (or HOME) explicitly.",
    );
  }
  return join(home, ".local", "state", "omega-nostr", "bot-state.json");
}
