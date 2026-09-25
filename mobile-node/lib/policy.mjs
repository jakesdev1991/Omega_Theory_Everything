// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * Execution policy for the mobile node.
 *
 * The daemon faces the public relay pool, i.e. untrusted input from anyone on
 * the internet. Nothing executes unless every gate below passes:
 *
 *   1. strict NIP-01 verification (recomputed id + schnorr signature)
 *   2. requester pubkey is in the operator allowlist, OR the algorithm is
 *      marked `license.access: "licensed"` and the requester holds a valid
 *      license (kind 31335) from a trusted issuer — see lib/license.mjs
 *   3. event kind matches an allowlisted algorithm kind
 *   4. content names an allowlisted algorithm id registered for that kind
 *   5. parameters are a JSON object within the configured size cap
 *
 * Execution itself (executor.mjs) never touches a shell: argv arrays only.
 */

import { verifyEventStrict } from "./events.mjs";
import { algorithmLicensePolicy, evaluateLicense } from "./license.mjs";

/** Extra room in the request content for an attached license event. */
const LICENSE_ENVELOPE_BYTES = 16384;

export function loadPolicy({ algorithmsConfig, operators, licenseIssuers = [] }) {
  const byKind = new Map();
  for (const algorithm of algorithmsConfig.algorithms ?? []) {
    byKind.set(algorithm.kind, algorithm);
  }
  return {
    algorithmsConfig,
    byKind,
    operators: new Set(operators),
    licenseIssuers: licenseIssuers.map((key) => String(key).toLowerCase()),
    maxParamsBytes: algorithmsConfig.maxParamsBytes ?? 65536,
  };
}

/**
 * @param {object} policy   from loadPolicy()
 * @param {object} event    the raw NIP-90 job request
 * @param {object} [context]
 * @param {number} [context.now]     unix seconds (defaults to the wall clock)
 * @param {object} [context.ledger]  LicenseLedger with revocations seen on relays
 */
export function evaluateRequest(policy, event, context = {}) {
  if (!event || typeof event !== "object") {
    return { ok: false, reason: "malformed envelope" };
  }
  if (!verifyEventStrict(event)) {
    return { ok: false, reason: "signature or event id verification failed" };
  }
  const requester = String(event.pubkey).toLowerCase();
  const isOperator = policy.operators.has(requester);
  const algorithm = policy.byKind.get(event.kind);
  if (!algorithm) {
    return { ok: false, reason: isOperator ? `kind ${event.kind} is not allowlisted` : "requester is not an operator" };
  }
  const licensePolicy = algorithmLicensePolicy(algorithm);
  if (!isOperator && licensePolicy.access !== "licensed") {
    return { ok: false, reason: "requester is not an operator" };
  }
  if (Buffer.byteLength(event.content || "", "utf8") > policy.maxParamsBytes + LICENSE_ENVELOPE_BYTES) {
    return { ok: false, reason: "request content exceeds the configured size cap" };
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

  let license = null;
  if (!isOperator) {
    const attached = payload.license && typeof payload.license === "object" ? [payload.license] : [];
    const known = context.ledger ? context.ledger.candidatesFor(algorithm.id, requester) : [];
    const decision = evaluateLicense({
      appId: algorithm.id,
      requester,
      trustedIssuers: policy.licenseIssuers,
      candidates: [...attached, ...known],
      now: context.now ?? Math.floor(Date.now() / 1000),
      allowedTiers: licensePolicy.tiers,
      acceptedTerms: licensePolicy.acceptedTerms,
    });
    if (!decision.ok) {
      return {
        ok: false,
        paymentRequired: true,
        code: decision.code,
        reason: decision.reason,
        terms: licensePolicy.terms,
      };
    }
    license = decision.license;
  }

  const params = payload.params;
  if (params !== undefined && (typeof params !== "object" || params === null || Array.isArray(params))) {
    return { ok: false, reason: "params must be a JSON object" };
  }
  const paramsJson = JSON.stringify(params ?? {});
  if (Buffer.byteLength(paramsJson, "utf8") > policy.maxParamsBytes) {
    return { ok: false, reason: "params exceed the configured size cap" };
  }

  return {
    ok: true,
    algorithm,
    params: params ?? {},
    access: isOperator ? "operator" : "licensed",
    license: license ? { id: license.id, tier: license.tier, expiresAt: license.expiresAt } : null,
  };
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
