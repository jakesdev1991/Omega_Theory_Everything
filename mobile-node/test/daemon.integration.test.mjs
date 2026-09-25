// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { WebSocketServer } from "ws";

import { getPublicKeyHex, signEvent, verifyEventStrict } from "../lib/events.mjs";
import { buildLicenseTemplate } from "../lib/license.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const nodeRoot = resolve(here, "..");

const SERVER_SECRET = "01".repeat(32);
const OPERATOR_SECRET = "02".repeat(32);
const STRANGER_SECRET = "03".repeat(32);

function waitFor(predicate, { timeoutMs = 15000, label = "condition" } = {}) {
  const startedAt = Date.now();
  return new Promise((resolvePromise, rejectPromise) => {
    const tick = () => {
      const value = predicate();
      if (value) return resolvePromise(value);
      if (Date.now() - startedAt > timeoutMs) return rejectPromise(new Error(`timed out waiting for ${label}`));
      setTimeout(tick, 50);
    };
    tick();
  });
}

test("The daemon executes allowlisted jobs over a relay and refuses strangers", async () => {
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  await new Promise((resolvePromise) => wss.once("listening", resolvePromise));
  const port = wss.address().port;

  const connections = new Set();
  const received = [];
  wss.on("connection", (socket) => {
    connections.add(socket);
    socket.on("message", (raw) => {
      try {
        received.push(JSON.parse(raw.toString()));
      } catch {
        /* ignore non-json frames */
      }
    });
  });

  const child = spawn(process.execPath, ["lucifer-daemon.mjs"], {
    cwd: nodeRoot,
    env: {
      ...process.env,
      LUCIFER_RELAYS: `ws://127.0.0.1:${port}`,
      LUCIFER_SECRET: SERVER_SECRET,
      LUCIFER_OPERATORS: getPublicKeyHex(OPERATOR_SECRET),
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  let daemonLog = "";
  child.stdout.on("data", (chunk) => {
    daemonLog += chunk.toString();
  });
  child.stderr.on("data", (chunk) => {
    daemonLog += chunk.toString();
  });

  try {
    await waitFor(() => daemonLog.includes("connected"), { label: "daemon relay connection" });
    await waitFor(() => received.some((message) => message[0] === "REQ"), { label: "REQ subscription" });

    const reqMessage = received.find((message) => message[0] === "REQ");
    assert.deepEqual(reqMessage[2], { kinds: [5001, 5002, 5003, 5099], "#p": [getPublicKeyHex(SERVER_SECRET)] });

    // 1. Operator job for the allowlisted echo algorithm.
    const request = signEvent(
      {
        kind: 5099,
        created_at: Math.floor(Date.now() / 1000),
        tags: [],
        content: JSON.stringify({ app: "echo", params: { ping: "pong" } }),
      },
      OPERATOR_SECRET,
    );
    for (const socket of connections) socket.send(JSON.stringify(["EVENT", "lucifer-store-jobs", request]));

    const resultMessage = await waitFor(
      () => received.find((message) => message[0] === "EVENT" && message[1]?.kind === 6099),
      { label: "job result event" },
    );
    const result = resultMessage[1];
    assert.equal(verifyEventStrict(result), true, "result verifies strictly");
    assert.equal(result.pubkey, getPublicKeyHex(SERVER_SECRET));
    assert.ok(result.tags.some((tag) => tag[0] === "e" && tag[1] === request.id), "correlates by request id");
    assert.ok(result.tags.some((tag) => tag[0] === "p" && tag[1] === getPublicKeyHex(OPERATOR_SECRET)), "addresses requester");
    assert.ok(result.tags.some((tag) => tag[0] === "status" && tag[1] === "success"));
    assert.match(result.content, /pong/);

    const resultFor = (requestId, label) =>
      waitFor(
        () =>
          received.find(
            (message) =>
              message[0] === "EVENT" &&
              message[1]?.kind >= 6000 &&
              message[1].tags.some((tag) => tag[0] === "e" && tag[1] === requestId),
          ),
        { label },
      );
    const statusOf = (message) => message[1].tags.find((tag) => tag[0] === "status")?.[1];
    const sendJob = (secret, kind, content) => {
      const event = signEvent({ kind, created_at: Math.floor(Date.now() / 1000), tags: [], content: JSON.stringify(content) }, secret);
      for (const socket of connections) socket.send(JSON.stringify(["EVENT", "lucifer-store-jobs", event]));
      return event;
    };

    // The daemon subscribes to licenses issued by its own key (default issuer).
    await waitFor(() => received.some((message) => message[0] === "REQ" && message[1] === "lucifer-store-licenses"), {
      label: "license subscription",
    });
    const licenseReq = received.find((message) => message[0] === "REQ" && message[1] === "lucifer-store-licenses");
    assert.deepEqual(licenseReq[2], { kinds: [31335], authors: [getPublicKeyHex(SERVER_SECRET)] });

    // 2. Stranger on an operator-only algorithm is refused with an error, never executed.
    const operatorOnly = sendJob(STRANGER_SECRET, 5001, { app: "radial-metric-sim", params: {} });
    const refusal = await resultFor(operatorOnly.id, "operator-only refusal");
    assert.equal(statusOf(refusal), "error");
    assert.match(refusal[1].content, /not an operator/);

    // 3. Stranger on a licensed algorithm without a license → payment-required.
    const unlicensed = sendJob(STRANGER_SECRET, 5099, { app: "echo", params: {} });
    const paymentRequired = await resultFor(unlicensed.id, "payment-required");
    assert.equal(statusOf(paymentRequired), "payment-required");
    assert.match(paymentRequired[1].content, /omega-store-eula-1\.0/);

    // 4. With a license signed by the node key attached, the job executes.
    const now = Math.floor(Date.now() / 1000);
    const license = signEvent(
      buildLicenseTemplate({
        issuerPubkey: getPublicKeyHex(SERVER_SECRET),
        licenseePubkey: getPublicKeyHex(STRANGER_SECRET),
        appId: "echo",
        tier: "trial",
        terms: "omega-store-eula-1.0",
        expiresAt: now + 3600,
        createdAt: now - 5,
      }),
      SERVER_SECRET,
    );
    const licensed = sendJob(STRANGER_SECRET, 5099, { app: "echo", params: { licensed: "yes" }, license });
    const licensedResult = await resultFor(licensed.id, "licensed result");
    assert.equal(statusOf(licensedResult), "success");
    assert.match(licensedResult[1].content, /licensed/);

    // 5. A revocation seen on the relay beats the stale license still attached.
    const revocation = signEvent(
      buildLicenseTemplate({
        issuerPubkey: getPublicKeyHex(SERVER_SECRET),
        licenseePubkey: getPublicKeyHex(STRANGER_SECRET),
        appId: "echo",
        tier: "trial",
        status: "revoked",
        terms: "omega-store-eula-1.0",
        createdAt: now,
      }),
      SERVER_SECRET,
    );
    for (const socket of connections) socket.send(JSON.stringify(["EVENT", "lucifer-store-licenses", revocation]));
    await waitFor(() => daemonLog.includes("license update"), { label: "revocation ingested" });
    const afterRevoke = sendJob(STRANGER_SECRET, 5099, { app: "echo", params: {}, license });
    const revokedResult = await resultFor(afterRevoke.id, "revoked refusal");
    assert.equal(statusOf(revokedResult), "payment-required");
    assert.match(revokedResult[1].content, /revoked/);
  } finally {
    child.kill("SIGTERM");
    for (const socket of connections) socket.terminate();
    await new Promise((resolvePromise) => wss.close(resolvePromise));
  }
});
