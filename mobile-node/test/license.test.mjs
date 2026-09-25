// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
import test from "node:test";
import assert from "node:assert/strict";

import { getPublicKeyHex, signEvent } from "../lib/events.mjs";
import { LicenseLedger, buildLicenseTemplate, evaluateLicense, parseLicenseEvent } from "../lib/license.mjs";
import { evaluateRequest, loadPolicy } from "../lib/policy.mjs";

const ISSUER = "11".repeat(32);
const ROGUE = "22".repeat(32);
const BUYER = "33".repeat(32);
const OPERATOR = "44".repeat(32);
const NOW = 1_800_000_000;

const issuerPub = getPublicKeyHex(ISSUER);
const buyerPub = getPublicKeyHex(BUYER);

function license(overrides = {}, secret = ISSUER) {
  return signEvent(
    buildLicenseTemplate({
      issuerPubkey: getPublicKeyHex(secret),
      licenseePubkey: buyerPub,
      appId: "echo",
      tier: "standard",
      terms: "omega-store-eula-1.0",
      createdAt: NOW - 100,
      ...overrides,
    }),
    secret,
  );
}

const evalWith = (candidates, extra = {}) =>
  evaluateLicense({ appId: "echo", requester: buyerPub, trustedIssuers: [issuerPub], candidates, now: NOW, ...extra });

test("A valid license from a trusted issuer is accepted", () => {
  const decision = evalWith([license()]);
  assert.equal(decision.ok, true);
  assert.equal(decision.license.tier, "standard");
  assert.equal(parseLicenseEvent(license()).d, `echo:${buyerPub}`);
});

test("Licenses fail closed: untrusted issuer, wrong app, wrong licensee, tampering", () => {
  assert.equal(evalWith([license({}, ROGUE)]).code, "license-required");
  assert.equal(evalWith([license({ appId: "lean-audit" })]).code, "license-required");
  assert.equal(
    evaluateLicense({ appId: "echo", requester: getPublicKeyHex(OPERATOR), trustedIssuers: [issuerPub], candidates: [license()], now: NOW }).code,
    "license-required",
  );
  const tampered = { ...license(), tags: [...license().tags.filter((tag) => tag[0] !== "tier"), ["tier", "pro"]] };
  assert.equal(evalWith([tampered]).code, "license-required");
  assert.equal(evalWith([]).code, "license-required");
});

test("Expiry, tier and terms are enforced", () => {
  assert.equal(evalWith([license({ expiresAt: NOW - 1 })]).code, "license-expired");
  assert.equal(evalWith([license({ expiresAt: NOW + 3600 })]).ok, true);
  assert.equal(evalWith([license({ tier: "trial" })], { allowedTiers: ["pro"] }).code, "license-tier");
  assert.equal(evalWith([license({ terms: "omega-store-eula-0.9" })], { acceptedTerms: ["omega-store-eula-1.0"] }).code, "license-terms");
});

test("Newest version wins, so a revocation beats an older active license", () => {
  const active = license({ createdAt: NOW - 1000 });
  const revoked = license({ createdAt: NOW - 10, status: "revoked" });
  assert.equal(evalWith([active, revoked]).code, "license-revoked");
  const renewed = license({ createdAt: NOW - 5, tier: "pro" });
  assert.equal(evalWith([active, revoked, renewed]).license.tier, "pro");
});

test("Future-dated licenses (beyond clock skew) are ignored", () => {
  assert.equal(evalWith([license({ createdAt: NOW + 3600 })]).code, "license-required");
});

test("The ledger keeps only the newest trusted version per license", () => {
  const ledger = new LicenseLedger({ trustedIssuers: [issuerPub] });
  assert.equal(ledger.ingest(license({ createdAt: NOW - 1000 })), true);
  assert.equal(ledger.ingest(license({ createdAt: NOW - 10, status: "revoked" })), true);
  assert.equal(ledger.ingest(license({ createdAt: NOW - 500 })), false, "older version ignored");
  assert.equal(ledger.ingest(license({}, ROGUE)), false, "untrusted issuer ignored");
  assert.equal(ledger.size, 1);
  assert.equal(ledger.candidatesFor("echo", buyerPub).length, 1);
});

/* ---------------- policy integration ---------------- */

const algorithmsConfig = {
  maxParamsBytes: 1024,
  algorithms: [
    { id: "echo", kind: 5099, argv: ["cat"], license: { access: "licensed", terms: "omega-store-eula-1.0", tiers: ["standard", "pro"] } },
    { id: "radial-metric-sim", kind: 5001, argv: ["python3", "x.py"] },
  ],
};
const policy = loadPolicy({ algorithmsConfig, operators: [getPublicKeyHex(OPERATOR)], licenseIssuers: [issuerPub] });

function job(secret, kind, content) {
  return signEvent({ kind, created_at: NOW, tags: [], content: JSON.stringify(content) }, secret);
}

test("Licensed apps run for license holders; operator-only apps still refuse them", () => {
  const ok = evaluateRequest(policy, job(BUYER, 5099, { app: "echo", params: { a: 1 }, license: license() }), { now: NOW });
  assert.equal(ok.ok, true);
  assert.equal(ok.access, "licensed");
  assert.equal(ok.license.tier, "standard");

  const operatorOnly = evaluateRequest(policy, job(BUYER, 5001, { app: "radial-metric-sim", license: license() }), { now: NOW });
  assert.equal(operatorOnly.ok, false);
  assert.match(operatorOnly.reason, /not an operator/);
});

test("Missing or revoked licenses get payment-required, operators bypass licensing", () => {
  const missing = evaluateRequest(policy, job(BUYER, 5099, { app: "echo" }), { now: NOW });
  assert.equal(missing.ok, false);
  assert.equal(missing.paymentRequired, true);
  assert.equal(missing.terms, "omega-store-eula-1.0");

  const ledger = new LicenseLedger({ trustedIssuers: [issuerPub] });
  ledger.ingest(license({ createdAt: NOW - 1, status: "revoked" }));
  const stale = evaluateRequest(policy, job(BUYER, 5099, { app: "echo", license: license({ createdAt: NOW - 500 }) }), { now: NOW, ledger });
  assert.equal(stale.code, "license-revoked", "relay revocation beats the stale attached license");

  const operator = evaluateRequest(policy, job(OPERATOR, 5099, { app: "echo" }), { now: NOW });
  assert.equal(operator.ok, true);
  assert.equal(operator.access, "operator");
});
