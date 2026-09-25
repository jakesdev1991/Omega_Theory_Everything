/**
 * Execution policy for the mobile node.
 *
 * The daemon faces the public relay pool, i.e. untrusted input from anyone on
 * the internet. Nothing executes unless every gate below passes:
 *
 *   1. strict NIP-01 verification (recomputed id + schnorr signature)
 *   2. requester pubkey is in the operator allowlist
 *   3. event kind matches an allowlisted algorithm kind
 *   4. content names an allowlisted algorithm id registered for that kind
 *   5. parameters are a JSON object within the configured size cap
 *
 * Execution itself (executor.mjs) never touches a shell: argv arrays only.
 */

import { verifyEventStrict } from "./events.mjs";

export function loadPolicy({ algorithmsConfig, operators }) {
  const byKind = new Map();
  for (const algorithm of algorithmsConfig.algorithms ?? []) {
    byKind.set(algorithm.kind, algorithm);
  }
  return {
    algorithmsConfig,
    byKind,
    operators: new Set(operators),
    maxParamsBytes: algorithmsConfig.maxParamsBytes ?? 65536,
  };
}

export function evaluateRequest(policy, event) {
  if (!event || typeof event !== "object") {
    return { ok: false, reason: "malformed envelope" };
  }
  if (!verifyEventStrict(event)) {
    return { ok: false, reason: "signature or event id verification failed" };
  }
  if (!policy.operators.has(String(event.pubkey).toLowerCase())) {
    return { ok: false, reason: "requester is not an operator" };
  }
  const algorithm = policy.byKind.get(event.kind);
  if (!algorithm) {
    return { ok: false, reason: `kind ${event.kind} is not allowlisted` };
  }

  let payload;
  try {
    payload = JSON.parse(event.content || "{}");
  } catch {
    return { ok: false, reason: "content is not valid JSON" };
  }
  if (typeof payload !== "object" || payload === null) {
    return { ok: false, reason: "content JSON must be an object" };
  }
  if (payload.app !== algorithm.id) {
    return { ok: false, reason: `algorithm "${payload.app ?? "?"}" is not the allowlisted id for kind ${event.kind}` };
  }

  const params = payload.params;
  if (params !== undefined && (typeof params !== "object" || params === null || Array.isArray(params))) {
    return { ok: false, reason: "params must be a JSON object" };
  }
  const paramsJson = JSON.stringify(params ?? {});
  if (Buffer.byteLength(paramsJson, "utf8") > policy.maxParamsBytes) {
    return { ok: false, reason: "params exceed the configured size cap" };
  }

  return { ok: true, algorithm, params: params ?? {} };
}

export function buildResultEvent({ request, serverSecretSign, status, content }) {
  // Kept separate so the daemon supplies signing; policy stays crypto-free here.
  return {
    kind: request.kind + 1000,
    created_at: Math.floor(Date.now() / 1000),
    tags: [
      ["e", request.id],
      ["p", request.pubkey],
      ["status", status],
    ],
    content,
    sign: serverSecretSign,
  };
}
