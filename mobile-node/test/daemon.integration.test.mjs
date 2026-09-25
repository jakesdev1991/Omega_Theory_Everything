import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { WebSocketServer } from "ws";

import { getPublicKeyHex, signEvent, verifyEventStrict } from "../lib/events.mjs";

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

    // 2. Stranger job is refused with an error result, never executed.
    const strangerRequest = signEvent(
      {
        kind: 5099,
        created_at: Math.floor(Date.now() / 1000),
        tags: [],
        content: JSON.stringify({ app: "echo", params: {} }),
      },
      STRANGER_SECRET,
    );
    for (const socket of connections) socket.send(JSON.stringify(["EVENT", "lucifer-store-jobs", strangerRequest]));

    const refusal = await waitFor(
      () =>
        received.find(
          (message) =>
            message[0] === "EVENT" &&
            message[1]?.kind === 6099 &&
            message[1].tags.some((tag) => tag[0] === "e" && tag[1] === strangerRequest.id),
        ),
      { label: "stranger refusal" },
    );
    assert.ok(refusal[1].tags.some((tag) => tag[0] === "status" && tag[1] === "error"));
    assert.match(refusal[1].content, /not an operator/);
  } finally {
    child.kill("SIGTERM");
    for (const socket of connections) socket.terminate();
    await new Promise((resolvePromise) => wss.close(resolvePromise));
  }
});
