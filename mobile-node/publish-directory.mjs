#!/usr/bin/env node
/**
 * Publishes the app-store directory (kind 31990 handler announcements) for the
 * algorithms in algorithms.json, signed by the store ROOT key.
 *
 * The static storefront (/store on the website) renders exactly these events:
 * parameterized replaceable per `d` tag, newest wins.
 *
 * Usage:
 *   LUCIFER_RELAYS=wss://nos.lol LUCIFER_ROOT_SECRET=nsec1… node publish-directory.mjs
 *
 * The root key is typically the same key the mobile node answers as
 * (LUCIFER_SECRET), so job requests addressed to the store root reach the
 * node's #p filter.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import WebSocket from "ws";

import { hexToNpub } from "./lib/bech32.mjs";
import { getPublicKeyHex, signEvent } from "./lib/events.mjs";

const here = dirname(fileURLToPath(import.meta.url));

const relays = (process.env.LUCIFER_RELAYS ?? "")
  .split(",")
  .map((relay) => relay.trim())
  .filter((relay) => relay.startsWith("wss://"));
const rootSecret = process.env.LUCIFER_ROOT_SECRET ?? process.env.LUCIFER_SECRET;
const algorithmsPath = resolve(process.env.LUCIFER_ALGORITHMS ?? resolve(here, "algorithms.json"));

if (relays.length === 0 || !rootSecret) {
  process.stderr.write("usage: LUCIFER_RELAYS=wss://… LUCIFER_ROOT_SECRET=… node publish-directory.mjs\n");
  process.exit(1);
}

const algorithmsConfig = JSON.parse(readFileSync(algorithmsPath, "utf8"));
const rootPubkey = getPublicKeyHex(rootSecret);
process.stdout.write(`publishing directory as ${hexToNpub(rootPubkey)}\n`);

const events = algorithmsConfig.algorithms.map((algorithm) =>
  signEvent(
    {
      kind: 31990,
      created_at: Math.floor(Date.now() / 1000),
      tags: [
        ["d", algorithm.id],
        ["k", String(algorithm.kind)],
        ["title", algorithm.name],
      ],
      content: JSON.stringify({
        name: algorithm.name,
        about: `Allowlisted algorithm ${algorithm.id} executed by the Lucifer mobile node (kind ${algorithm.kind} → ${algorithm.kind + 1000}).`,
        workClass: algorithm.workClass ?? "engineering_protocol",
        paramsTemplate: algorithm.paramsTemplate ?? {},
      }),
    },
    rootSecret,
  ),
);

let pending = relays.length;
for (const url of relays) {
  const socket = new WebSocket(url);
  const done = () => {
    socket.close();
    pending -= 1;
    if (pending === 0) process.exit(0);
  };
  socket.on("open", () => {
    for (const event of events) {
      const dTag = event.tags.find((tag) => tag[0] === "d");
      socket.send(JSON.stringify(["EVENT", event]));
      process.stdout.write(`  ${url} ← kind 31990 d=${dTag ? dTag[1] : "?"}\n`);
    }
    setTimeout(done, 1500);
  });
  socket.on("error", (error) => {
    process.stderr.write(`  ${url} failed: ${error.message}\n`);
    done();
  });
}

setTimeout(() => process.exit(0), 10000);
