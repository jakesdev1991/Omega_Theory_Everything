// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * Live integration against the local chorus smoke relay (ws://127.0.0.1:18080).
 * Skips gracefully when the relay is not running.
 */
import test from "node:test";
import assert from "node:assert/strict";

import WebSocket from "ws";

import { getPublicKeyHex, signEvent, verifyEventStrict } from "../lib/events.mjs";
import { RelayPool } from "../lib/relay.mjs";

const SECRET = "13".repeat(32);
const RELAYS = ["ws://127.0.0.1:18080"];

async function relayUp(url) {
  // NOTE: chorus bans the IP for ~2s after a DISCONNECT. So the probe
  // socket stays open for the life of the test — closing it would ban
  // 127.0.0.1 and reset the pool's real connection. Callers must keep
  // the returned socket alive and close everything at test end.
  return new Promise((resolve) => {
    const ws = new WebSocket(url);
    ws.once("open", () => resolve(ws));
    ws.once("error", () => resolve(null));
    setTimeout(() => resolve(null), 1500).unref?.();
  });
}

test("live chorus: publish directory + read back with strict verification", async (t) => {
  const probe = await relayUp(RELAYS[0]);
  if (probe === null) {
    t.skip(`live relay ${RELAYS[0]} not running — start chorus with /tmp/chorus-smoke/chorus.conf to enable`);
    return;
  }
  try {
  const pool = new RelayPool(RELAYS, SECRET);
  pool.connect();
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("live relay: connect timeout")), 5000);
    pool.once("connected", () => { clearTimeout(timer); resolve(); });
  });

  const dirEvent = signEvent(
    {
      kind: 31990,
      created_at: Math.floor(Date.now() / 1000),
      tags: [["d", "live-smoke"], ["k", "5099"]],
      content: JSON.stringify({ name: "live smoke" }),
    },
    SECRET,
  );
  const res = await pool.publish(dirEvent);
  assert.equal(res.accepted, true);

  const got = [];
  const close = pool.subscribe(
    { kinds: [31990], authors: [getPublicKeyHex(SECRET)] },
    { onEvent: (ev) => got.push(ev) },
  );
  await new Promise((r) => setTimeout(r, 500));
  assert.ok(got.some((e) => e.id === dirEvent.id), "read back our own 31990");
  for (const ev of got) assert.equal(verifyEventStrict(ev), true);
  close();
  pool.close();
  } finally {
    // close probe LAST — chorus 2s IP ban fires on disconnect; closing it
    // first would reset the pool socket mid-test.
    probe.close();
  }
});
