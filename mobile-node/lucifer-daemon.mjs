#!/usr/bin/env node
// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * Lucifer mobile node — sovereign NIP-90 execution daemon for Termux.
 *
 * Listens on the configured relay pool for app-store job requests addressed to
 * this node (kinds from algorithms.json, #p = this node's pubkey), verifies
 * them strictly, runs allowlisted algorithms in the Debian sandbox, and
 * publishes NIP-90 job results (request kind + 1000) correlated by
 * ["e", requestId] / ["p", requester].
 *
 * Configuration (env, or config.json next to this file):
 *   LUCIFER_RELAYS        comma separated wss:// URLs            (required)
 *   LUCIFER_SECRET        hex or nsec server key                  (required)
 *   LUCIFER_OPERATORS     comma separated npub/hex operator keys  (required)
 *   LUCIFER_ALGORITHMS    path to algorithms.json                 (default ./algorithms.json)
 *   LUCIFER_CONFIG        path to a JSON file with the same keys  (optional)
 *   LUCIFER_LICENSE_ISSUERS comma separated npub/hex keys whose kind 31335
 *                         licenses are honoured for algorithms marked
 *                         license.access = "licensed"   (default: this node's key)
 *
 * TLS is always verified. There is no flag to disable certificate checks.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import WebSocket from "ws";

import { hexToNpub } from "./lib/bech32.mjs";
import { getPublicKeyHex, npubOrHexToHex, signEvent } from "./lib/events.mjs";
import { evaluateRequest, loadPolicy } from "./lib/policy.mjs";
import { LICENSE_KIND, LicenseLedger } from "./lib/license.mjs";
import { runAlgorithm } from "./lib/executor.mjs";

const here = dirname(fileURLToPath(import.meta.url));

/**
 * TLS is mandatory for public relays. Plain ws:// is accepted only for
 * loopback hosts so the integration suite (and local bench relays) can run
 * without certificates; there is no other way to disable TLS verification.
 */
function isAcceptableRelayUrl(url) {
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "wss:") return true;
    return parsed.protocol === "ws:" && ["127.0.0.1", "localhost", "::1"].includes(parsed.hostname);
  } catch {
    return false;
  }
}

function loadConfig() {
  let fileConfig = {};
  const configPath = process.env.LUCIFER_CONFIG;
  if (configPath) {
    fileConfig = JSON.parse(readFileSync(resolve(configPath), "utf8"));
  }
  const relays = (process.env.LUCIFER_RELAYS ?? fileConfig.relays ?? "")
    .split(",")
    .map((relay) => relay.trim())
    .filter(isAcceptableRelayUrl);
  const secret = process.env.LUCIFER_SECRET ?? fileConfig.secret;
  const operators = (process.env.LUCIFER_OPERATORS ?? (fileConfig.operators ?? []).join(","))
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean)
    .map(npubOrHexToHex);
  const algorithmsPath = resolve(process.env.LUCIFER_ALGORITHMS ?? fileConfig.algorithms ?? resolve(here, "algorithms.json"));

  if (relays.length === 0) throw new Error("LUCIFER_RELAYS must list at least one wss:// relay");
  if (!secret) throw new Error("LUCIFER_SECRET (hex or nsec) is required");
  if (operators.length === 0) throw new Error("LUCIFER_OPERATORS must list at least one operator pubkey");

  const licenseIssuers = (process.env.LUCIFER_LICENSE_ISSUERS ?? (fileConfig.licenseIssuers ?? []).join(","))
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean)
    .map(npubOrHexToHex);

  return { relays, secret, operators, algorithmsPath, licenseIssuers };
}

const config = loadConfig();
const algorithmsConfig = JSON.parse(readFileSync(config.algorithmsPath, "utf8"));
const serverPubkey = getPublicKeyHex(config.secret);
const licenseIssuers = config.licenseIssuers.length > 0 ? config.licenseIssuers : [serverPubkey];
const policy = loadPolicy({ algorithmsConfig, operators: config.operators, licenseIssuers });
const ledger = new LicenseLedger({ trustedIssuers: licenseIssuers });
const licensedAlgorithms = algorithmsConfig.algorithms.filter((algorithm) => algorithm.license?.access === "licensed");
const jobKinds = algorithmsConfig.algorithms.map((algorithm) => algorithm.kind);

const processed = new Set();
const PROCESSED_CAP = 5000;

function remember(requestId) {
  if (processed.has(requestId)) return false;
  processed.add(requestId);
  if (processed.size > PROCESSED_CAP) {
    const oldest = processed.values().next().value;
    processed.delete(oldest);
  }
  return true;
}

function log(line) {
  process.stdout.write(`[${new Date().toISOString()}] ${line}\n`);
}

async function handleEvent(connection, event) {
  if (event && event.kind === LICENSE_KIND) {
    if (ledger.ingest(event)) log(`license update ${event.id.slice(0, 12)} (${ledger.size} known)`);
    return;
  }
  if (!jobKinds.includes(event.kind)) return;
  if (!remember(event.id)) return;

  const decision = evaluateRequest(policy, event, { ledger });
  if (!decision.ok && decision.paymentRequired) {
    log(`license check failed for ${event.id.slice(0, 12)} from ${(event.pubkey ?? "?").slice(0, 12)}: ${decision.reason}`);
    publish(
      connection,
      event,
      "payment-required",
      JSON.stringify({ status: "payment-required", code: decision.code, reason: decision.reason, terms: decision.terms }),
    );
    return;
  }
  if (!decision.ok) {
    log(`refused ${event.id.slice(0, 12)} from ${(event.pubkey ?? "?").slice(0, 12)}: ${decision.reason}`);
    publish(connection, event, "error", JSON.stringify({ status: "error", reason: decision.reason }));
    return;
  }

  const via = decision.access === "licensed" ? ` via license ${decision.license.id.slice(0, 12)} (${decision.license.tier})` : "";
  log(`executing ${decision.algorithm.id} for job ${event.id.slice(0, 12)} (kind ${event.kind})${via}`);
  const result = await runAlgorithm({
    algorithm: decision.algorithm,
    sandboxArgv: algorithmsConfig.sandboxArgv ?? [],
    params: decision.params,
    cwd: here,
    env: { OMEGA_REPO: resolve(here, "..") },
  });

  if (result.ok) {
    log(`job ${event.id.slice(0, 12)} succeeded (${result.stdout.length} bytes of output)`);
    publish(connection, event, "success", result.stdout.trim() || JSON.stringify({ status: "success" }));
  } else {
    log(`job ${event.id.slice(0, 12)} failed: ${result.error}`);
    publish(
      connection,
      event,
      "error",
      JSON.stringify({ status: "error", reason: result.error, stderr: (result.stderr ?? "").slice(0, 2000) }),
    );
  }
}

function publish(connection, request, status, content) {
  const event = signEvent(
    {
      kind: request.kind + 1000,
      created_at: Math.floor(Date.now() / 1000),
      tags: [
        ["e", request.id],
        ["p", request.pubkey],
        ["status", status],
      ],
      content,
    },
    config.secret,
  );
  connection.send(JSON.stringify(["EVENT", event]));
}

function connectRelay(url) {
  const backoff = { delay: 1000 };
  const open = () => {
    const ws = new WebSocket(url);

    ws.on("open", () => {
      backoff.delay = 1000;
      log(`connected ${url}`);
      ws.send(
        JSON.stringify([
          "REQ",
          "lucifer-store-jobs",
          { kinds: jobKinds, "#p": [serverPubkey] },
        ]),
      );
      if (licensedAlgorithms.length > 0) {
        ws.send(JSON.stringify(["REQ", "lucifer-store-licenses", { kinds: [LICENSE_KIND], authors: licenseIssuers }]));
      }
    });

    ws.on("message", (raw) => {
      let message;
      try {
        message = JSON.parse(raw.toString("utf8"));
      } catch {
        return;
      }
      if (message[0] === "EVENT") {
        // NIP-01: ["EVENT", <subscription_id>, <event>]; tolerate the 2-element
        // form some minimal relays emit.
        const event = message.length >= 3 ? message[2] : message[1];
        if (event) handleEvent(ws, event).catch((error) => log(`handler error: ${error.message}`));
      }
    });

    ws.on("close", () => {
      log(`disconnected ${url}; retrying in ${backoff.delay} ms`);
      setTimeout(open, backoff.delay);
      backoff.delay = Math.min(backoff.delay * 2, 60000);
    });

    ws.on("error", (error) => {
      log(`relay error ${url}: ${error.message}`);
      ws.terminate();
    });
  };
  open();
}

log(`Lucifer node ${hexToNpub(serverPubkey)}`);
log(`allowlisted kinds: ${jobKinds.join(", ")} · operators: ${config.operators.length}`);
log(
  `licensed algorithms: ${licensedAlgorithms.map((algorithm) => algorithm.id).join(", ") || "none"} · license issuers: ${licenseIssuers.length}`,
);
log(`sandbox argv prefix: ${JSON.stringify(algorithmsConfig.sandboxArgv ?? [])}`);
for (const relay of config.relays) connectRelay(relay);

process.on("SIGINT", () => {
  log("shutting down");
  process.exit(0);
});
