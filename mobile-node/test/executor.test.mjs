// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import test from "node:test";
import assert from "node:assert/strict";

import { runAlgorithm } from "../lib/executor.mjs";

test("Allowlisted argv runs with params on stdin and bounded output", async () => {
  const result = await runAlgorithm({
    algorithm: { id: "echo", argv: ["cat"], timeoutMs: 5000, maxOutputBytes: 4096 },
    params: { hello: "world" },
    cwd: process.cwd(),
  });
  assert.equal(result.ok, true);
  assert.deepEqual(JSON.parse(result.stdout), { hello: "world" });
});

test("The bundled radial metric algorithm produces a convergence payload", async () => {
  const result = await runAlgorithm({
    algorithm: {
      id: "radial-metric-sim",
      argv: ["python3", "algorithms/radial_metric.py"],
      timeoutMs: 20000,
      maxOutputBytes: 65536,
    },
    params: { phi: 0.618, chainLength: 128 },
    cwd: new URL("..", import.meta.url).pathname,
  });
  assert.equal(result.ok, true, result.stderr);
  const payload = JSON.parse(result.stdout);
  assert.equal(payload.status, "success");
  assert.match(payload.matrix_output, /stable_convergence/);
});

test("Runaway jobs are killed at the timeout", async () => {
  const result = await runAlgorithm({
    algorithm: { id: "slow", argv: ["sleep", "5"], timeoutMs: 300, maxOutputBytes: 1024 },
    params: {},
    cwd: process.cwd(),
  });
  assert.equal(result.ok, false);
  assert.match(result.error, /timed out/);
});

test("Failing exit codes surface as errors with stderr", async () => {
  const result = await runAlgorithm({
    algorithm: { id: "bad", argv: ["python3", "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"], timeoutMs: 10000 },
    params: {},
    cwd: process.cwd(),
  });
  assert.equal(result.ok, false);
  assert.match(result.error, /exit code 3/);
  assert.match(result.stderr, /boom/);
});
