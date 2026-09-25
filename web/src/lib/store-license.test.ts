// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import test from "node:test";
import assert from "node:assert/strict";

import { finalizeEvent, generateSecretKey, getPublicKey } from "nostr-tools";

import {
  DEMO_ROOT_SECRET_HEX,
  buildJobRequestTemplate,
  demoIssueLicense,
  demoJobResult,
  demoListings,
  hexToBytes,
  parseJobResult,
  parseListingEvent,
  signJobRequest,
} from "./nostr-store";
import {
  STORE_TERMS_ID,
  buildLicenseTemplate,
  describeDecision,
  evaluateLicense,
  mergeLicenses,
  parseLicenseEvent,
  type StoreLicense,
} from "./store-license";

const issuerSecret = hexToBytes(DEMO_ROOT_SECRET_HEX);
const issuer = getPublicKey(issuerSecret);
const buyerSecret = generateSecretKey();
const buyer = getPublicKey(buyerSecret);
const NOW = 1_800_000_000;

function license(overrides: Partial<Parameters<typeof buildLicenseTemplate>[0]> = {}, secret = issuerSecret) {
  return finalizeEvent(
    buildLicenseTemplate({
      issuerPubkey: getPublicKey(secret),
      licenseePubkey: buyer,
      appId: "echo",
      tier: "standard",
      terms: STORE_TERMS_ID,
      createdAt: NOW - 100,
      ...overrides,
    }),
    secret,
  );
}

const decide = (candidates: ReturnType<typeof license>[], extra: Partial<Parameters<typeof evaluateLicense>[0]> = {}) =>
  evaluateLicense({ appId: "echo", requester: buyer, trustedIssuers: [issuer], candidates, now: NOW, ...extra });

test("Listings expose license terms; unknown access modes fail closed", () => {
  const listings = demoListings(issuer).map(parseListingEvent);
  const echo = listings.find((listing) => listing?.appId === "echo");
  assert.equal(echo?.license.access, "licensed");
  assert.equal(echo?.license.terms, STORE_TERMS_ID);
  assert.deepEqual(echo?.license.tiers, ["trial", "standard", "pro"]);

  const radial = listings.find((listing) => listing?.appId === "radial-metric-sim");
  assert.equal(radial?.license.access, "operators");

  const weird = finalizeEvent(
    { kind: 31990, created_at: NOW, tags: [["d", "x"], ["access", "public"]], content: "{}" },
    issuerSecret,
  );
  assert.equal(parseListingEvent(weird)?.license.access, "operators");
});

test("License evaluation matches the mobile node rules", () => {
  assert.equal(decide([license()]).ok, true);
  assert.equal(decide([license({}, generateSecretKey())]).code, "license-required", "untrusted issuer");
  assert.equal(decide([license({ expiresAt: NOW - 1 })]).code, "license-expired");
  assert.equal(decide([license({ tier: "trial" })], { allowedTiers: ["pro"] }).code, "license-tier");
  assert.equal(decide([license({ terms: "old" })], { acceptedTerms: [STORE_TERMS_ID] }).code, "license-terms");
  assert.equal(
    decide([license({ createdAt: NOW - 1000 }), license({ createdAt: NOW - 1, status: "revoked" })]).code,
    "license-revoked",
  );
  const tampered = { ...license(), tags: license().tags.map((tag) => (tag[0] === "tier" ? ["tier", "pro"] : tag)) };
  assert.equal(parseLicenseEvent(tampered), null);
});

test("mergeLicenses keeps only the newest version", () => {
  const old = parseLicenseEvent(license({ createdAt: NOW - 1000 })) as StoreLicense;
  const revoked = parseLicenseEvent(license({ createdAt: NOW - 1, status: "revoked" })) as StoreLicense;
  let state = mergeLicenses([], old);
  state = mergeLicenses(state, revoked);
  state = mergeLicenses(state, old);
  assert.equal(state.length, 1);
  assert.equal(state[0].status, "revoked");
});

test("Demo responder: payment-required without a license, success with one", () => {
  const echo = demoListings(issuer).map(parseListingEvent).find((listing) => listing?.appId === "echo");
  assert.ok(echo);
  const licensing = { terms: echo.license, issuers: [issuer], operators: [] as string[] };

  const unlicensed = signJobRequest(buildJobRequestTemplate({ listing: echo, serverPubkey: issuer, params: {} }), buyerSecret);
  const refusal = parseJobResult(demoJobResult(unlicensed.event, buyer, licensing), unlicensed.event.id);
  assert.equal(refusal?.status, "payment-required");
  assert.equal(refusal?.ok, false);

  const granted = demoIssueLicense(buyer, "echo");
  const licensed = signJobRequest(
    buildJobRequestTemplate({ listing: echo, serverPubkey: issuer, params: { ping: 1 }, license: granted }),
    buyerSecret,
  );
  const result = parseJobResult(demoJobResult(licensed.event, buyer, licensing), licensed.event.id);
  assert.equal(result?.status, "success");

  const operatorRun = parseJobResult(
    demoJobResult(unlicensed.event, buyer, { ...licensing, operators: [buyer] }),
    unlicensed.event.id,
  );
  assert.equal(operatorRun?.status, "success", "operators bypass licensing");
});

test("Badge text summarizes the license state", () => {
  const echo = demoListings(issuer).map(parseListingEvent).find((listing) => listing?.appId === "echo");
  assert.ok(echo);
  assert.equal(describeDecision(echo.license, null, false), "License required");
  assert.match(describeDecision(echo.license, decide([license({ expiresAt: NOW + 86400 })]), false), /^Licensed · standard · until /);
  assert.equal(describeDecision(echo.license, decide([license({ expiresAt: NOW - 1 })]), false), "License expired");
});

test("The web and mobile node serialize identical license tags", () => {
  const template = buildLicenseTemplate({
    issuerPubkey: issuer,
    licenseePubkey: buyer,
    appId: "echo",
    tier: "trial",
    terms: STORE_TERMS_ID,
    expiresAt: NOW + 10,
    payment: "manual",
    createdAt: NOW,
  });
  assert.deepEqual(template.tags, [
    ["d", `echo:${buyer}`],
    ["p", buyer],
    ["app", "echo"],
    ["a", `31990:${issuer}:echo`],
    ["tier", "trial"],
    ["status", "active"],
    ["terms", STORE_TERMS_ID],
    ["expiration", String(NOW + 10)],
    ["payment", "manual"],
  ]);
});
