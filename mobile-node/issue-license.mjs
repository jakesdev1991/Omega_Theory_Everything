#!/usr/bin/env node
// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
/**
 * Issue, renew, or revoke an app-store license (kind 31335) signed by the
 * store root / publisher key. The secret key never leaves this terminal.
 *
 * Usage:
 *   LUCIFER_RELAYS=wss://nos.lol LUCIFER_ROOT_SECRET=nsec1… \
 *     node issue-license.mjs issue  --app echo --to npub1… [--tier standard] [--days 30] [--payment manual] [--note "…"]
 *   … node issue-license.mjs revoke --app echo --to npub1…
 *   … node issue-license.mjs show   --app echo --to npub1… [--tier …] [--days …]   (prints the signed event, publishes nothing)
 *
 * Terms: the license records the terms id the licensee accepted. By default it
 * is the `license.terms` of the algorithm in algorithms.json; override with
 * --terms. Revocation publishes a newer version of the same `d` tag with
 * status "revoked" (newest wins; the daemon also tracks it live).
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import WebSocket from "ws";

import { hexToNpub } from "./lib/bech32.mjs";
import { getPublicKeyHex, npubOrHexToHex, signEvent } from "./lib/events.mjs";
import { algorithmLicensePolicy, buildLicenseTemplate } from "./lib/license.mjs";

const here = dirname(fileURLToPath(import.meta.url));

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    app: { type: "string" },
    to: { type: "string" },
    tier: { type: "string", default: "standard" },
    days: { type: "string" },
    terms: { type: "string" },
    payment: { type: "string" },
    note: { type: "string" },
  },
});

const action = positionals[0];
if (!["issue", "revoke", "show"].includes(action ?? "")) {
  fail("usage: node issue-license.mjs <issue|revoke|show> --app <id> --to <npub|hex> [--tier t] [--days n] [--terms id] [--payment ref] [--note text]");
}
if (!values.app || !values.to) fail("--app and --to are required");

const rootSecret = process.env.LUCIFER_ROOT_SECRET ?? process.env.LUCIFER_SECRET;
if (!rootSecret) fail("LUCIFER_ROOT_SECRET (hex or nsec) is required");

const algorithmsPath = resolve(process.env.LUCIFER_ALGORITHMS ?? resolve(here, "algorithms.json"));
const algorithmsConfig = JSON.parse(readFileSync(algorithmsPath, "utf8"));
const algorithm = algorithmsConfig.algorithms.find((entry) => entry.id === values.app);
if (!algorithm) fail(`unknown app "${values.app}" (not in ${algorithmsPath})`);
const policy = algorithmLicensePolicy(algorithm);
if (policy.access !== "licensed") {
  process.stderr.write(`warning: "${values.app}" has license.access="${policy.access}"; the daemon only honours licenses for "licensed" apps.\n`);
}
if (action !== "revoke" && policy.tiers.length > 0 && !policy.tiers.includes(values.tier)) {
  fail(`tier "${values.tier}" is not offered for ${values.app} (offered: ${policy.tiers.join(", ")})`);
}

let licensee;
try {
  licensee = npubOrHexToHex(values.to);
} catch (error) {
  fail(error.message);
}

const issuerPubkey = getPublicKeyHex(rootSecret);
const now = Math.floor(Date.now() / 1000);
let expiresAt;
if (values.days !== undefined) {
  const days = Number(values.days);
  if (!Number.isFinite(days) || days <= 0) fail("--days must be a positive number");
  expiresAt = now + Math.round(days * 86400);
}

const template = buildLicenseTemplate({
  issuerPubkey,
  licenseePubkey: licensee,
  appId: values.app,
  tier: values.tier,
  status: action === "revoke" ? "revoked" : "active",
  terms: values.terms ?? policy.terms ?? undefined,
  expiresAt,
  payment: values.payment,
  note: values.note,
  createdAt: now,
});
const event = signEvent(template, rootSecret);

process.stdout.write(
  `${action} license app=${values.app} licensee=${hexToNpub(licensee)} tier=${values.tier}` +
    `${expiresAt ? ` expires=${new Date(expiresAt * 1000).toISOString()}` : " expires=never"} issuer=${hexToNpub(issuerPubkey)}\n`,
);

if (action === "show") {
  process.stdout.write(`${JSON.stringify(event, null, 2)}\n`);
  process.exit(0);
}

const relays = (process.env.LUCIFER_RELAYS ?? "")
  .split(",")
  .map((relay) => relay.trim())
  .filter((relay) => relay.startsWith("wss://"));
if (relays.length === 0) fail("LUCIFER_RELAYS must list at least one wss:// relay (or use `show`)");

let pending = relays.length;
let accepted = 0;
for (const url of relays) {
  const socket = new WebSocket(url);
  let finished = false;
  const done = () => {
    if (finished) return;
    finished = true;
    socket.close();
    pending -= 1;
    if (pending === 0) process.exit(accepted > 0 ? 0 : 2);
  };
  socket.on("open", () => socket.send(JSON.stringify(["EVENT", event])));
  socket.on("message", (raw) => {
    try {
      const message = JSON.parse(raw.toString("utf8"));
      if (message[0] === "OK" && message[1] === event.id) {
        if (message[2]) accepted += 1;
        process.stdout.write(`  ${url}: ${message[2] ? "accepted" : `rejected (${message[3] ?? "no reason"})`}\n`);
        done();
      }
    } catch {
      /* ignore */
    }
  });
  socket.on("error", (error) => {
    process.stderr.write(`  ${url} failed: ${error.message}\n`);
    done();
  });
  setTimeout(done, 5000);
}
