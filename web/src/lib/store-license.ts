// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * App-store licensing over Nostr (storefront side).
 *
 * Mirrors mobile-node/lib/license.mjs: the daemon enforces, the storefront
 * displays and attaches. Both MUST agree on these rules. Spec:
 * docs/store/LICENSE-PROTOCOL.md.
 *
 *   kind 31335 (provisional, parameterized replaceable), signed by the issuer
 *   ["d", "<appId>:<licenseeHex>"] ["p", licensee] ["app", appId]
 *   ["a", "31990:<issuer>:<appId>"] ["tier", t] ["status", active|revoked]
 *   ["terms", termsId] ["expiration", unix]? ["payment", ref]?
 */

import { getEventHash, verifyEvent, type Event as NostrEvent, type EventTemplate } from "nostr-tools";

export const LICENSE_KIND = 31335;
export const MAX_FUTURE_SKEW_SECONDS = 600;

/** Current store-wide end-user terms. Bump the id when the text changes materially. */
export const STORE_TERMS_ID = "omega-store-eula-1.0";
export const PUBLISHER_AGREEMENT_ID = "omega-store-publisher-1.0";

export type AccessMode = "operators" | "licensed";
export type LicenseStatus = "active" | "revoked";

const TIER_PATTERN = /^[a-z0-9][a-z0-9-]{0,31}$/;
const APP_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const HEX64 = /^[0-9a-f]{64}$/;

/** License terms advertised by a directory listing (31990/30017). */
export interface ListingLicenseTerms {
  access: AccessMode;
  terms: string | null;
  tiers: string[];
  spdx: string;
  price: string | null;
}

export interface StoreLicense {
  id: string;
  issuer: string;
  licensee: string;
  appId: string;
  d: string;
  tier: string;
  status: LicenseStatus;
  terms: string | null;
  payment: string | null;
  expiresAt: number | null;
  createdAt: number;
  raw: NostrEvent;
}

export type LicenseDecisionCode =
  | "licensed"
  | "license-required"
  | "license-revoked"
  | "license-expired"
  | "license-tier"
  | "license-terms";

export interface LicenseDecision {
  ok: boolean;
  code: LicenseDecisionCode;
  reason: string;
  license?: StoreLicense;
}

function verifyStrict(event: NostrEvent): boolean {
  try {
    return getEventHash(event) === event.id && verifyEvent(event);
  } catch {
    return false;
  }
}

function firstTag(event: Pick<NostrEvent, "tags">, name: string): string | undefined {
  const tag = event.tags.find((entry) => Array.isArray(entry) && entry[0] === name && typeof entry[1] === "string");
  return tag ? tag[1] : undefined;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];
}

export function licenseDTag(appId: string, licenseePubkey: string): string {
  return `${appId}:${licenseePubkey.toLowerCase()}`;
}

/**
 * Reads license terms from a listing. Tags win over content (tags are
 * indexable by relays); unknown access modes fail closed to "operators".
 */
export function parseListingLicense(event: Pick<NostrEvent, "tags">, content: Record<string, unknown>): ListingLicenseTerms {
  const block =
    typeof content.license === "object" && content.license !== null ? (content.license as Record<string, unknown>) : {};
  const accessRaw = firstTag(event, "access") ?? (typeof block.access === "string" ? block.access : "operators");
  return {
    access: accessRaw === "licensed" ? "licensed" : "operators",
    terms: firstTag(event, "terms") ?? (typeof block.terms === "string" ? block.terms : null),
    tiers: stringList(block.tiers),
    spdx:
      firstTag(event, "license") ?? (typeof block.spdx === "string" ? block.spdx : "LicenseRef-Omega-Product-Proprietary"),
    price: typeof block.price === "string" ? block.price : null,
  };
}

export function parseLicenseEvent(event: NostrEvent): StoreLicense | null {
  if (!event || event.kind !== LICENSE_KIND) return null;
  if (!verifyStrict(event)) return null;

  const appId = firstTag(event, "app");
  const licensee = (firstTag(event, "p") ?? "").toLowerCase();
  const d = firstTag(event, "d");
  const status = firstTag(event, "status") ?? "active";
  const tier = firstTag(event, "tier") ?? "standard";
  const expirationRaw = firstTag(event, "expiration");

  if (!appId || !APP_ID_PATTERN.test(appId)) return null;
  if (!HEX64.test(licensee)) return null;
  if (d !== licenseDTag(appId, licensee)) return null;
  if (status !== "active" && status !== "revoked") return null;
  if (!TIER_PATTERN.test(tier)) return null;

  let expiresAt: number | null = null;
  if (expirationRaw !== undefined) {
    const value = Number(expirationRaw);
    if (!Number.isInteger(value) || value <= 0) return null;
    expiresAt = value;
  }

  return {
    id: event.id,
    issuer: event.pubkey.toLowerCase(),
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

function isNewer(a: StoreLicense, b: StoreLicense | undefined): boolean {
  if (!b) return true;
  if (a.createdAt !== b.createdAt) return a.createdAt > b.createdAt;
  return a.id < b.id;
}

export function evaluateLicense({
  appId,
  requester,
  trustedIssuers,
  candidates,
  now,
  allowedTiers = [],
  acceptedTerms = [],
}: {
  appId: string;
  requester: string;
  trustedIssuers: string[];
  candidates: NostrEvent[];
  now: number;
  allowedTiers?: string[];
  acceptedTerms?: string[];
}): LicenseDecision {
  const issuers = new Set(trustedIssuers.map((key) => key.toLowerCase()));
  const who = requester.toLowerCase();
  const wantedD = licenseDTag(appId, who);

  const newestByIssuer = new Map<string, StoreLicense>();
  for (const candidate of candidates) {
    const license = parseLicenseEvent(candidate);
    if (!license || !issuers.has(license.issuer)) continue;
    if (license.d !== wantedD || license.licensee !== who || license.appId !== appId) continue;
    if (license.createdAt > now + MAX_FUTURE_SKEW_SECONDS) continue;
    if (isNewer(license, newestByIssuer.get(license.issuer))) newestByIssuer.set(license.issuer, license);
  }

  if (newestByIssuer.size === 0) {
    return { ok: false, code: "license-required", reason: `no license for app "${appId}" from a trusted issuer` };
  }

  let failure: LicenseDecision | null = null;
  for (const license of newestByIssuer.values()) {
    if (license.status !== "active") {
      failure = failure ?? { ok: false, code: "license-revoked", reason: "license has been revoked", license };
      continue;
    }
    if (license.expiresAt !== null && license.expiresAt <= now) {
      failure = { ok: false, code: "license-expired", reason: `license expired at ${license.expiresAt}`, license };
      continue;
    }
    if (allowedTiers.length > 0 && !allowedTiers.includes(license.tier)) {
      failure = { ok: false, code: "license-tier", reason: `tier "${license.tier}" does not include this app`, license };
      continue;
    }
    if (acceptedTerms.length > 0 && !acceptedTerms.includes(license.terms ?? "")) {
      failure = {
        ok: false,
        code: "license-terms",
        reason: `license was issued under terms "${license.terms ?? "none"}"; current terms must be accepted`,
        license,
      };
      continue;
    }
    return { ok: true, code: "licensed", reason: "valid license", license };
  }
  return failure as LicenseDecision;
}

/** Keeps the newest license version per (issuer, d). */
export function mergeLicenses(current: StoreLicense[], incoming: StoreLicense): StoreLicense[] {
  const key = `${incoming.issuer}:${incoming.d}`;
  const existing = current.find((license) => `${license.issuer}:${license.d}` === key);
  if (existing && !isNewer(incoming, existing)) return current;
  return [...current.filter((license) => license !== existing), incoming];
}

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
}: {
  issuerPubkey: string;
  licenseePubkey: string;
  appId: string;
  tier?: string;
  status?: LicenseStatus;
  terms?: string | null;
  expiresAt?: number | null;
  payment?: string | null;
  note?: string;
  createdAt?: number;
}): EventTemplate {
  const licensee = licenseePubkey.toLowerCase();
  if (!HEX64.test(licensee)) throw new Error("licensee must be a 64-char hex pubkey");
  if (!APP_ID_PATTERN.test(appId)) throw new Error(`invalid app id: ${appId}`);
  if (!TIER_PATTERN.test(tier)) throw new Error(`invalid tier: ${tier}`);

  const tags: string[][] = [
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

  return { kind: LICENSE_KIND, created_at: createdAt, tags, content: JSON.stringify(note ? { note } : {}) };
}

/** Relay filter for the licenses addressed to one buyer from the store's issuers. */
export function licenseFilter(issuers: string[], licensee: string) {
  return { kinds: [LICENSE_KIND], authors: issuers, "#p": [licensee.toLowerCase()] };
}

/** Human summary used by the storefront badge. */
export function describeDecision(terms: ListingLicenseTerms, decision: LicenseDecision | null, isOperator: boolean): string {
  if (terms.access === "operators") return isOperator ? "Operator access" : "Operator-only app";
  if (isOperator) return "Operator access (license not needed)";
  if (!decision) return "License required";
  if (decision.ok && decision.license) {
    const expiry = decision.license.expiresAt
      ? `until ${new Date(decision.license.expiresAt * 1000).toISOString().slice(0, 10)}`
      : "no expiry";
    return `Licensed · ${decision.license.tier} · ${expiry}`;
  }
  switch (decision.code) {
    case "license-expired":
      return "License expired";
    case "license-revoked":
      return "License revoked";
    case "license-tier":
      return "Tier does not include this app";
    case "license-terms":
      return "New terms must be accepted";
    default:
      return "License required";
  }
}
