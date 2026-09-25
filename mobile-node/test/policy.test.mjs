// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
import test from "node:test";
import assert from "node:assert/strict";

import { getPublicKeyHex, signEvent } from "../lib/events.mjs";
import { evaluateRequest, loadPolicy } from "../lib/policy.mjs";

const OPERATOR_SECRET = "0000000000000000000000000000000000000000000000000000000000000001";
const STRANGER_SECRET = "0000000000000000000000000000000000000000000000000000000000000002";

const algorithmsConfig = {
  maxParamsBytes: 128,
  algorithms: [
    { id: "echo", kind: 5099, argv: ["cat"], timeoutMs: 1000 },
    { id: "radial-metric-sim", kind: 5001, argv: ["python3", "algorithms/radial_metric.py"], timeoutMs: 1000 },
  ],
};

const policy = loadPolicy({ algorithmsConfig, operators: [getPublicKeyHex(OPERATOR_SECRET)] });

function request(secret, overrides = {}) {
  return signEvent(
    {
      kind: 5099,
      created_at: 1767225600,
      tags: [],
      content: JSON.stringify({ app: "echo", params: { hello: "world" }, ...overrides }),
    },
    secret,
  );
}

test("Operators pass, strangers are refused before anything executes", () => {
  const ok = evaluateRequest(policy, request(OPERATOR_SECRET));
  assert.equal(ok.ok, true);
  assert.equal(ok.algorithm.id, "echo");
  assert.deepEqual(ok.params, { hello: "world" });

  const stranger = evaluateRequest(policy, request(STRANGER_SECRET));
  assert.equal(stranger.ok, false);
  assert.match(stranger.reason, /not an operator/);
});

test("Tampered requests fail the strict verification gate", () => {
  const event = request(OPERATOR_SECRET);
  assert.equal(evaluateRequest(policy, { ...event, content: '{"app":"echo","params":{"evil":true}}' }).ok, false);
});

test("Unknown kinds and mismatched algorithm ids are refused", () => {
  const wrongKind = signEvent(
    { kind: 5555, created_at: 1767225600, tags: [], content: JSON.stringify({ app: "echo" }) },
    OPERATOR_SECRET,
  );
  assert.match(evaluateRequest(policy, wrongKind).reason, /not allowlisted/);

  const wrongApp = evaluateRequest(policy, request(OPERATOR_SECRET, { app: "rm-rf" }));
  assert.match(wrongApp.reason, /not the allowlisted id/);
});

test("Params must be an object inside the size cap", () => {
  const arrayParams = evaluateRequest(policy, request(OPERATOR_SECRET, { params: [1, 2, 3] }));
  assert.match(arrayParams.reason, /must be a JSON object/);

  const huge = evaluateRequest(policy, request(OPERATOR_SECRET, { params: { blob: "x".repeat(4096) } }));
  assert.match(huge.reason, /size cap/);
});
