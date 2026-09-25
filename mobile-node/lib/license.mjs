// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * App-store licensing over Nostr (mobile-node side).
 *
 * Mirrors web/src/lib/store-license.ts: both sides MUST agree on these rules.
 * The full protocol is documented in docs/store/LICENSE-PROTOCOL.md.
 *
 * A license is a parameterized replaceable event (kind 31335, provisional)
 * signed by a trusted issuer (normally the store root / publisher key):
 *
 *   ["d", "<appId>:<licenseePubkeyHex>"]   one live license per app+licensee
 *   ["p", "<licenseePubkeyHex>"]
 *   ["app", "<appId>"]
 *   ["a", "31990:<issuerPubkeyHex>:<appId>"]
 *   ["tier", "<tier>"]                     e.g. trial | standard | pro
 *   ["status", "active" | "revoked"]
 *   ["terms", "<termsId>"]                 e.g. omega-store-eula-1.0
 *   ["expiration", "<unix seconds>"]       optional, NIP-40
 *   ["payment", "<ref>"]                   optional hook (manual, zap:<id>, …)
 *
 * Newest created_at per (issuer, d) wins, exactly like any NIP-01
 * parameterized replaceable event. Revocation = publish a newer version with
 * status "revoked". Fails closed on anything malformed.
 */

import { verifyEventStrict } from "./events.mjs";

export const LICENSE_KIND = 31335;
export const LICENSE_STATUSES = ["active", "revoked"];
/** Accept at most 10 minutes of clock skew for licenses dated in the future. */
export const MAX_FUTURE_SKEW_SECONDS = 600;

const TIER_PATTERN = /^[a-z0-9][a-z0-9-]{0,31}$/;
const APP_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const HEX64 = /^[0-9a-f]{64}$/;

function firstTag(event, name) {
  const tag = (event.tags ?? []).find((entry) => Array.isArray(entry) && entry[0] === name && typeof entry[1] === "string");
  return tag ? tag[1] : undefined;
}

export function licenseDTag(appId, licenseePubkey) {
  return `${appId}:${licenseePubkey}`;
}

/**
 * Structural parse of a license event (signature is verified here too).
 * Returns null for anything that is not a well-formed license.
 */
export function parseLicenseEvent(event) {
  if (!event || typeof event !== "object" || event.kind !== LICENSE_KIND) return null;
  if (!verifyEventStrict(event)) return null;

  const appId = firstTag(event, "app");
  const licensee = (firstTag(event, "p") ?? "").toLowerCase();
  const d = firstTag(event, "d");
  const status = firstTag(event, "status") ?? "active";
  const tier = firstTag(event, "tier") ?? "standard";
  const expirationRaw = firstTag(event, "expiration");

  if (!appId || !APP_ID_PATTERN.test(appId)) return null;
  if (!HEX64.test(licensee)) return null;
  if (d !== licenseDTag(appId, licensee)) return null;
  if (!LICENSE_STATUSES.includes(status)) return null;
  if (!TIER_PATTERN.test(tier)) return null;

  let expiresAt = null;
  if (expirationRaw !== undefined) {
    const value = Number(expirationRaw);
    if (!Number.isInteger(value) || value <= 0) return null;
    expiresAt = value;
  }

  return {
    id: event.id,
    issuer: String(event.pubkey).toLowerCase(),
    licensee,
    appId,
    d,
    tier,
    status,
    terms: firstTag(event, "terms") ?? null,
    payment: firstTag(event, "payment") ?? null,
    expiresAt,
    createdAt: event.created_at,
    raw: event,
  };
}

/** NIP-01 replaceable ordering: newest created_at wins; ties → lowest id. */
function newer(a, b) {
  if (!b) return true;
  if (a.createdAt !== b.createdAt) return a.createdAt > b.createdAt;
  return a.id < b.id;
}

/**
 * Evaluates whether `requester` holds a valid license for `appId`.
 *
 * @param {object}   options
 * @param {string}   options.appId
 * @param {string}   options.requester        hex pubkey of the job requester
 * @param {string[]} options.trustedIssuers   hex pubkeys allowed to issue licenses
 * @param {object[]} options.candidates       raw license events (attached + ledger)
 * @param {number}   options.now              unix seconds
 * @param {string[]} [options.allowedTiers]   if set, license tier must be listed
 * @param {string[]} [options.acceptedTerms]  if set, license terms id must be listed
 */
export function evaluateLicense({ appId, requester, trustedIssuers, candidates, now, allowedTiers, acceptedTerms }) {
  const issuers = new Set((trustedIssuers ?? []).map((key) => String(key).toLowerCase()));
  const who = String(requester ?? "").toLowerCase();
  const wantedD = licenseDTag(appId, who);

  // Newest version per issuer for this exact (app, licensee) pair.
  const newestByIssuer = new Map();
  for (const candidate of candidates ?? []) {
    const license = parseLicenseEvent(candidate);
    if (!license) continue;
    if (!issuers.has(license.issuer)) continue;
    if (license.d !== wantedD || license.licensee !== who || license.appId !== appId) continue;
    if (license.createdAt > now + MAX_FUTURE_SKEW_SECONDS) continue;
    if (newer(license, newestByIssuer.get(license.issuer))) newestByIssuer.set(license.issuer, license);
  }

  if (newestByIssuer.size === 0) {
    return { ok: false, code: "license-required", reason: `no license for app "${appId}" from a trusted issuer` };
  }

  // Any issuer's newest version being revoked wins over another issuer's active
  // one only for that issuer; we accept if at least one issuer has a valid
  // newest version. Collect the most useful failure reason otherwise.
  let failure = null;
  for (const license of newestByIssuer.values()) {
    if (license.status !== "active") {
      failure = failure ?? { ok: false, code: "license-revoked", reason: "license has been revoked", license };
      continue;
    }
    if (license.expiresAt !== null && license.expiresAt <= now) {
      failure = { ok: false, code: "license-expired", reason: `license expired at ${license.expiresAt}`, license };
      continue;
    }
    if (Array.isArray(allowedTiers) && allowedTiers.length > 0 && !allowedTiers.includes(license.tier)) {
      failure = { ok: false, code: "license-tier", reason: `tier "${license.tier}" does not include this app`, license };
      continue;
    }
    if (Array.isArray(acceptedTerms) && acceptedTerms.length > 0 && !acceptedTerms.includes(license.terms)) {
      failure = {
        ok: false,
        code: "license-terms",
        reason: `license was issued under terms "${license.terms ?? "none"}"; current terms must be accepted`,
        license,
      };
      continue;
    }
    return { ok: true, code: "licensed", license };
  }
  return failure;
}

/**
 * In-memory view of the newest license versions seen on relays, so a stale
 * "active" license attached to a job request cannot outrun a newer revocation.
 */
export class LicenseLedger {
  constructor({ trustedIssuers, cap = 50000 } = {}) {
    this.trustedIssuers = new Set((trustedIssuers ?? []).map((key) => String(key).toLowerCase()));
    this.byKey = new Map();
    this.cap = cap;
  }

  /** Returns true if the event became the newest known version for its key. */
  ingest(event) {
    const license = parseLicenseEvent(event);
    if (!license || !this.trustedIssuers.has(license.issuer)) return false;
    const key = `${license.issuer}:${license.d}`;
    const current = this.byKey.get(key);
    if (!newer(license, current)) return false;
    this.byKey.delete(key);
    this.byKey.set(key, license);
    if (this.byKey.size > this.cap) this.byKey.delete(this.byKey.keys().next().value);
    return true;
  }

  /** Raw events known for this app+licensee (all trusted issuers). */
  candidatesFor(appId, licensee) {
    const d = licenseDTag(appId, String(licensee).toLowerCase());
    const out = [];
    for (const issuer of this.trustedIssuers) {
      const license = this.byKey.get(`${issuer}:${d}`);
      if (license) out.push(license.raw);
    }
    return out;
  }

  get size() {
    return this.byKey.size;
  }
}

/** Builds an unsigned license template (sign with events.mjs signEvent). */
export function buildLicenseTemplate({
  issuerPubkey,
  licenseePubkey,
  appId,
  tier = "standard",
  status = "active",
  terms,
  expiresAt,
  payment,
  note,
  createdAt = Math.floor(Date.now() / 1000),
}) {
  const licensee = String(licenseePubkey).toLowerCase();
  if (!HEX64.test(licensee)) throw new Error("licensee must be a 64-char hex pubkey");
  if (!APP_ID_PATTERN.test(appId)) throw new Error(`invalid app id: ${appId}`);
  if (!TIER_PATTERN.test(tier)) throw new Error(`invalid tier: ${tier}`);
  if (!LICENSE_STATUSES.includes(status)) throw new Error(`invalid status: ${status}`);

  const tags = [
    ["d", licenseDTag(appId, licensee)],
    ["p", licensee],
    ["app", appId],
    ["a", `31990:${issuerPubkey}:${appId}`],
    ["tier", tier],
    ["status", status],
  ];
  if (terms) tags.push(["terms", terms]);
  if (expiresAt) tags.push(["expiration", String(Math.floor(expiresAt))]);
  if (payment) tags.push(["payment", payment]);

  return {
    kind: LICENSE_KIND,
    created_at: createdAt,
    tags,
    content: JSON.stringify(note ? { note } : {}),
  };
}

/** Normalizes the `license` block of an algorithms.json entry. */
export function algorithmLicensePolicy(algorithm) {
  const block = algorithm && typeof algorithm.license === "object" && algorithm.license ? algorithm.license : {};
  const access = block.access === "licensed" ? "licensed" : "operators";
  const terms = typeof block.terms === "string" ? block.terms : null;
  const acceptedTerms = Array.isArray(block.acceptedTerms)
    ? block.acceptedTerms.filter((entry) => typeof entry === "string")
    : terms
      ? [terms]
      : [];
  const tiers = Array.isArray(block.tiers) ? block.tiers.filter((entry) => typeof entry === "string") : [];
  return {
    access,
    terms,
    acceptedTerms,
    tiers,
    spdx: typeof block.spdx === "string" ? block.spdx : "LicenseRef-Omega-Product-Proprietary",
    price: typeof block.price === "string" ? block.price : null,
  };
}
