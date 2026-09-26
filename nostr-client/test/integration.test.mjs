// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * Integration: in-memory mini-relay (ws server) proving the pool's
 * publish + AUTH + subscribe roundtrip with ephemeral keys.
 * Mirrors mobile-node/test/daemon.integration.test.mjs.
 */
import test from "node:test";
import assert from "node:assert/strict";

import { WebSocketServer } from "ws";

import { getPublicKeyHex, signEvent, verifyEventStrict } from "../lib/events.mjs";
import { RelayConnection } from "../lib/relay.mjs";

const SECRET = "11".repeat(32);

function waitFor(predicate, { timeoutMs = 10_000, label = "condition" } = {}) {
  return new Promise((resolvePromise, rejectPromise) => {
    const startedAt = Date.now();
    const tick = () => {
      let value;
      try {
        value = predicate();
      } catch (err) {
        return rejectPromise(err);
      }
      if (value) return resolvePromise(value);
      if (Date.now() - startedAt > timeoutMs)
        return rejectPromise(new Error(`timed out waiting for ${label}`));
      setTimeout(tick, 25);
    };
    tick();
  });
}

test("RelayConnection: AUTH challenge answered with exact relay URL, publish OK, subscribe roundtrip", async () => {
  // --- mini relay ---
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  await new Promise((r) => wss.once("listening", r));
  const port = wss.address().port;
  const url = `ws://127.0.0.1:${port}`;

  const received = [];
  const pushes = new Set();
  let authEvent = null;

  wss.on("connection", (socket) => {
    // NIP-42 like chorus: challenge immediately on connection setup
    socket.send(JSON.stringify(["AUTH", "challenge-abc"]));
    const subIds = new Set();
    socket.on("message", (raw) => {
      const msg = JSON.parse(raw.toString());
      received.push(msg);
      if (msg[0] === "AUTH") {
        authEvent = msg[1];
        socket.send(JSON.stringify(["OK", msg[1].id, true, "auth ok"]));
      }
      if (msg[0] === "EVENT") {
        socket.send(JSON.stringify(["OK", msg[1].id, true, ""]));
        // echo the event back to every subscriber (this test: one socket,
        // whatever subId it actually used)
        for (const id of subIds) socket.send(JSON.stringify(["EVENT", id, msg[1]]));
        for (const peer of pushes) if (peer !== socket && peer.readyState === 1)
          for (const id of subIds) peer.send(JSON.stringify(["EVENT", id, msg[1]]));
        pushes.add(socket);
      }
      if (msg[0] === "REQ") {
        subIds.add(msg[1]);
        // EOSE immediately (empty store)
        socket.send(JSON.stringify(["EOSE", msg[1]]));
        pushes.add(socket);
      }
    });
    // push helper is only used for the single-subscriber case: reply on the
    // same subId the client actually used (captured in the REQ branch)
    socket.on("push", (event) => {
      for (const id of subIds) socket.send(JSON.stringify(["EVENT", id, event]));
    });
  });

  // --- client ---
  const conn = new RelayConnection(url, SECRET);
  const seenEvents = [];
  conn.connect();
  await waitFor(() => received.some((m) => m[0] === "AUTH"), { label: "AUTH reply" });

  // the AUTH event must carry the EXACT dialed URL and the challenge
  assert.equal(authEvent.kind, 22242);
  assert.deepEqual(
    authEvent.tags.find((t) => t[0] === "relay"),
    ["relay", url],
  );
  assert.deepEqual(
    authEvent.tags.find((t) => t[0] === "challenge"),
    ["challenge", "challenge-abc"],
  );
  assert.equal(verifyEventStrict(authEvent), true);
  assert.equal(authEvent.pubkey, getPublicKeyHex(SECRET));

  // publish
  const note = signEvent(
    { kind: 31990, created_at: Math.floor(Date.now() / 1000), tags: [["d", "echo"]], content: "{}" },
    SECRET,
  );
  const res = await conn.publish(note);
  assert.equal(res.accepted, true);

  // subscribe and receive pushes
  const close = conn.subscribe({ kinds: [31990] }, { onEvent: (ev) => seenEvents.push(ev) });
  const note2 = signEvent(
    { kind: 31990, created_at: Math.floor(Date.now() / 1000), tags: [["d", "echo2"]], content: "{}" },
    SECRET,
  );
  await conn.publish(note2);
  await waitFor(() => seenEvents.some((e) => e.id === note2.id), { label: "pushed event" });
  assert.equal(seenEvents.at(-1).id, note2.id);

  close();
  conn.close();
  wss.close();
});

test("RelayConnection: publish rejects on relay OK:false", async () => {
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  await new Promise((r) => wss.once("listening", r));
  const url = `ws://127.0.0.1:${wss.address().port}`;

  wss.on("connection", (socket) => {
    socket.on("message", (raw) => {
      const msg = JSON.parse(raw.toString());
      if (msg[0] === "EVENT") socket.send(JSON.stringify(["OK", msg[1].id, false, "blocked: rejected for test"]));
    });
  });

  const conn = new RelayConnection(url, SECRET);
  conn.connect();
  await new Promise((r) => conn.once("connected", r));
  const note = signEvent({ kind: 1, created_at: 1, tags: [], content: "x" }, SECRET);
  await assert.rejects(() => conn.publish(note), /rejected/);
  conn.close();
  wss.close();
});
