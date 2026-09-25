// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * App-store protocol over the Nostr backplane.
 *
 * The store is a static frontend: it reads its directory from parameterized
 * replaceable events published by the sovereign root key, and turns "Run"
 * clicks into NIP-90 Data Vending Machine job requests that the mobile node
 * (mobile-node/lucifer-daemon.mjs on the Pixel 8a) answers with job results.
 *
 * Directory kinds:
 *   31990  NIP-89 handler announcement  (preferred; tags: d, k)
 *   30017  NIP-99 classified listing    (fallback index; tags: d, title)
 *
 * Licensing (see store-license.ts, docs/store/LICENSE-PROTOCOL.md):
 *   31335  license grant / revocation from the store's issuer key
 *   listings carry ["access", operators|licensed], ["terms", id], ["license", SPDX]
 *
 * Execution kinds (NIP-90):
 *   request  5000-5999   (default 5001; per-app override via the "k" tag)
 *   result   request + 1000, tagged ["e", requestId] and ["p", requester]
 *
 * Everything here is pure and offline-testable; relay I/O lives in the page
 * component (nostr-tools SimplePool) or in the demo responder below.
 */

import {
  finalizeEvent,
  getEventHash,
  getPublicKey,
  verifyEvent,
  type Event as NostrEvent,
  type EventTemplate,
} from "nostr-tools";

import { ECONOMY_NOSTR_KINDS } from "./nostr";
import {
  STORE_TERMS_ID,
  buildLicenseTemplate,
  evaluateLicense,
  parseListingLicense,
  type ListingLicenseTerms,
} from "./store-license";

export interface AppListing {
  appId: string;
  name: string;
  about: string;
  jobKind: number;
  sourceKind: number;
  publisher: string;
  updatedAt: number;
  eventId: string;
  paramsTemplate: Record<string, unknown>;
  workClass: string;
  license: ListingLicenseTerms;
  raw: NostrEvent;
}

export interface JobRequest {
  event: NostrEvent;
  appId: string;
  jobKind: number;
  serverPubkey: string;
}

export const DEFAULT_JOB_KIND = 5001;

/**
 * Strict NIP-01 verification.
 *
 * nostr-tools' verifyEvent() only checks the schnorr signature against the
 * event's *declared* id; it does not recompute the id from the serialized
 * [0, pubkey, created_at, kind, tags, content] array. A relay (or anyone
 * upstream) could therefore rewrite content/tags while keeping the original
 * id and signature and still pass verifyEvent(). Anything the store renders
 * or executes against must recompute the hash first.
 */
export function verifyEventStrict(event: NostrEvent): boolean {
  try {
    return getEventHash(event) === event.id && verifyEvent(event);
  } catch {
    return false;
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

function parseContentJson(content: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(content);
    return asRecord(parsed);
  } catch {
    return {};
  }
}

function tagValues(event: NostrEvent, name: string): string[] {
  return event.tags.filter((tag) => tag[0] === name && tag[1]).map((tag) => tag[1]);
}

function firstTag(event: NostrEvent, name: string): string | undefined {
  return tagValues(event, name)[0];
}

function clampJobKind(candidate: unknown): number {
  const value = Number(candidate);
  if (
    Number.isInteger(value) &&
    value >= ECONOMY_NOSTR_KINDS.dvmRequestMin &&
    value <= ECONOMY_NOSTR_KINDS.dvmRequestMax
  ) {
    return value;
  }
  return DEFAULT_JOB_KIND;
}

/**
 * Parses a 31990/30017 event into a store listing. Returns null for events that
 * are not directory entries (missing d tag, wrong kind, unverifiable signature).
 */
export function parseListingEvent(event: NostrEvent): AppListing | null {
  if (event.kind !== ECONOMY_NOSTR_KINDS.storeHandler && event.kind !== ECONOMY_NOSTR_KINDS.storeListing) {
    return null;
  }
  if (!verifyEventStrict(event)) return null;

  const appId = firstTag(event, "d");
  if (!appId) return null;

  const content = parseContentJson(event.content);
  const name =
    firstTag(event, "title") ||
    (typeof content.name === "string" && content.name) ||
    (typeof content.title === "string" && content.title) ||
    appId;
  const about =
    (typeof content.about === "string" && content.about) ||
    (typeof content.description === "string" && content.description) ||
    (typeof content.summary === "string" && content.summary) ||
    "";
  const jobKind = clampJobKind(firstTag(event, "k") ?? content.jobKind);
  const paramsTemplate = asRecord(content.paramsTemplate ?? content.params ?? {});
  const workClass =
    (typeof content.workClass === "string" && content.workClass) || "engineering_protocol";

  return {
    appId,
    name,
    about,
    jobKind,
    sourceKind: event.kind,
    publisher: event.pubkey,
    updatedAt: event.created_at,
    eventId: event.id,
    paramsTemplate,
    workClass,
    license: parseListingLicense(event, content),
    raw: event,
  };
}

/**
 * Deduplicates parameterized-replaceable directory events: newest per
 * (publisher, appId) wins.
 */
export function dedupeListings(listings: AppListing[]): AppListing[] {
  const best = new Map<string, AppListing>();
  for (const listing of listings) {
    const key = `${listing.publisher}:${listing.appId}`;
    const current = best.get(key);
    if (!current || listing.updatedAt >= current.updatedAt) best.set(key, listing);
  }
  return Array.from(best.values()).sort((a, b) => a.name.localeCompare(b.name));
}

export function buildJobRequestTemplate({
  listing,
  serverPubkey,
  params,
  inputs = [],
  license,
}: {
  listing: Pick<AppListing, "appId" | "jobKind">;
  serverPubkey: string;
  params: Record<string, unknown>;
  inputs?: string[];
  /** Signed kind 31335 license to attach; the node re-verifies it. */
  license?: NostrEvent | null;
}): EventTemplate {
  return {
    kind: listing.jobKind,
    created_at: Math.floor(Date.now() / 1000),
    tags: [
      ["p", serverPubkey],
      ["a", listing.appId],
      ...inputs.map((input) => ["i", input]),
    ],
    content: JSON.stringify(license ? { app: listing.appId, params, license } : { app: listing.appId, params }),
  };
}

/** Signs a job request with the operator secret key (raw 32 bytes). */
export function signJobRequest(
  template: EventTemplate,
  secretKey: Uint8Array,
): JobRequest {
  const event = finalizeEvent(template, secretKey);
  const serverPubkey = firstTag(event, "p") ?? "";
  const appId = firstTag(event, "a") ?? "";
  return { event, appId, jobKind: event.kind, serverPubkey };
}

/** The filter that correlates a job result back to its request. */
export function jobResultFilter(requestId: string, requesterPubkey: string, jobKind: number) {
  return {
    kinds: [jobKind + 1000],
    "#e": [requestId],
    "#p": [requesterPubkey],
  };
}

export interface JobResult {
  ok: boolean;
  requestId: string;
  resultKind: number;
  serverPubkey: string;
  content: string;
  status: "success" | "error" | "pending" | "payment-required";
  eventId: string;
  createdAt: number;
}

export function parseJobResult(event: NostrEvent, requestId: string): JobResult | null {
  if (!verifyEventStrict(event)) return null;
  if (event.kind < ECONOMY_NOSTR_KINDS.dvmResultMin || event.kind > ECONOMY_NOSTR_KINDS.dvmResultMax) {
    return null;
  }
  if (!tagValues(event, "e").includes(requestId)) return null;

  const statusTag = firstTag(event, "status");
  const status: JobResult["status"] =
    statusTag === "payment-required" ? "payment-required" : statusTag === "error" ? "error" : "success";

  return {
    ok: status === "success",
    requestId,
    resultKind: event.kind,
    serverPubkey: event.pubkey,
    content: event.content,
    status,
    eventId: event.id,
    createdAt: event.created_at,
  };
}

/** Demo mode only: the fixture store key issues a license to `licensee`. */
export function demoIssueLicense(
  licenseePubkey: string,
  appId: string,
  { tier = "trial", days = 14, status = "active" as "active" | "revoked", now = Math.floor(Date.now() / 1000) } = {},
): NostrEvent {
  const secret = hexToBytes(DEMO_ROOT_SECRET_HEX);
  return finalizeEvent(
    buildLicenseTemplate({
      issuerPubkey: getPublicKey(secret),
      licenseePubkey,
      appId,
      tier,
      status,
      terms: STORE_TERMS_ID,
      expiresAt: status === "active" ? now + days * 86400 : null,
      payment: "demo",
      createdAt: now,
    }),
    secret,
  );
}

/* ------------------------------------------------------------------ */
/* Demo mode: an in-page responder so the whole flow is testable with  */
/* zero relays configured (same philosophy as the Economy Test         */
/* Console's offline scenario suite).                                  */
/* ------------------------------------------------------------------ */

export const DEMO_ROOT_SECRET_HEX = "0000000000000000000000000000000000000000000000000000000000000001";

export function demoListings(publisherPubkey: string, createdAt = 1767225600): NostrEvent[] {
  const templates: EventTemplate[] = [
    {
      kind: ECONOMY_NOSTR_KINDS.storeHandler,
      created_at: createdAt,
      tags: [
        ["d", "radial-metric-sim"],
        ["k", String(DEFAULT_JOB_KIND)],
        ["title", "Radial Metric Simulator"],
        ["access", "operators"],
        ["license", "LicenseRef-Omega-Product-Proprietary"],
      ],
      content: JSON.stringify({
        name: "Radial Metric Simulator",
        about: "Checks the macroscopic radial metric g_rr(Φ) bottleneck on the mobile node's Debian sandbox.",
        workClass: "physics_simulation",
        paramsTemplate: { phi: 0.618, chainLength: 64 },
      }),
    },
    {
      kind: ECONOMY_NOSTR_KINDS.storeHandler,
      created_at: createdAt,
      tags: [
        ["d", "lean-audit"],
        ["k", "5002"],
        ["title", "Lean Axiom Audit"],
      ],
      content: JSON.stringify({
        name: "Lean Axiom Audit",
        about: "Runs the sorry/axiom inventory over a named lean_proofs module and returns the count.",
        workClass: "lean_formalization",
        paramsTemplate: { module: "RadialMetric" },
      }),
    },
    {
      kind: ECONOMY_NOSTR_KINDS.storeListing,
      created_at: createdAt - 60,
      tags: [
        ["d", "rcod-benchmark"],
        ["title", "RCOD Noise Recovery"],
      ],
      content: JSON.stringify({
        name: "RCOD Noise Recovery",
        about: "Benchmark governor run from rcod/ — returns the honest verdict for the spec thresholds.",
        workClass: "engineering_protocol",
        jobKind: 5003,
        paramsTemplate: { noisePct: 20 },
      }),
    },
    {
      kind: ECONOMY_NOSTR_KINDS.storeHandler,
      created_at: createdAt,
      tags: [
        ["d", "echo"],
        ["k", "5099"],
        ["title", "Echo (licensed self-test)"],
        ["access", "licensed"],
        ["terms", STORE_TERMS_ID],
        ["license", "LicenseRef-Omega-Product-Proprietary"],
      ],
      content: JSON.stringify({
        name: "Echo (licensed self-test)",
        about: "Returns its parameters. Open to anyone holding a license from the store key: exercises the full license → job → result path.",
        workClass: "engineering_protocol",
        paramsTemplate: { ping: "pong" },
        license: {
          access: "licensed",
          terms: STORE_TERMS_ID,
          tiers: ["trial", "standard", "pro"],
          spdx: "LicenseRef-Omega-Product-Proprietary",
          price: "Free self-test; license issued on request",
        },
      }),
    },
  ];

  const secret = hexToBytes(DEMO_ROOT_SECRET_HEX);
  return templates.map((template) => finalizeEvent(template, secret));
}

/**
 * Deterministic in-page DVM responder used by demo mode. When `licensing` is
 * given for a licensed listing it enforces the same license rules as the
 * mobile node (operators bypass; everyone else needs a valid license).
 */
export function demoJobResult(
  request: NostrEvent,
  requesterPubkey: string,
  licensing?: { terms: ListingLicenseTerms; issuers: string[]; operators: string[]; now?: number },
): NostrEvent {
  const payload = parseContentJson(request.content);
  const params = asRecord(asRecord(payload).params);

  if (licensing && licensing.terms.access === "licensed" && !licensing.operators.includes(request.pubkey)) {
    const attached = asRecord(payload.license);
    const decision = evaluateLicense({
      appId: typeof payload.app === "string" ? payload.app : "",
      requester: request.pubkey,
      trustedIssuers: licensing.issuers,
      candidates: Object.keys(attached).length > 0 ? [attached as unknown as NostrEvent] : [],
      now: licensing.now ?? Math.floor(Date.now() / 1000),
      allowedTiers: licensing.terms.tiers,
      acceptedTerms: licensing.terms.terms ? [licensing.terms.terms] : [],
    });
    if (!decision.ok) {
      return finalizeEvent(
        {
          kind: request.kind + 1000,
          created_at: Math.floor(Date.now() / 1000),
          tags: [
            ["e", request.id],
            ["p", requesterPubkey],
            ["status", "payment-required"],
          ],
          content: JSON.stringify(
            { status: "payment-required", code: decision.code, reason: decision.reason, terms: licensing.terms.terms },
            null,
            2,
          ),
        },
        hexToBytes(DEMO_ROOT_SECRET_HEX),
      );
    }
  }
  const output = {
    status: "success",
    node: "pixel-8a-demo (simulated)",
    app: payload.app ?? "unknown",
    echoParams: params,
    matrix_output: "0.354_stable_convergence",
    note: "Demo responder: no device execution happened. Configure relays + mobile-node/ for real runs.",
  };

  const template: EventTemplate = {
    kind: request.kind + 1000,
    created_at: Math.floor(Date.now() / 1000),
    tags: [
      ["e", request.id],
      ["p", requesterPubkey],
      ["status", "success"],
    ],
    content: JSON.stringify(output, null, 2),
  };
  return finalizeEvent(template, hexToBytes(DEMO_ROOT_SECRET_HEX));
}

export function hexToBytes(hex: string): Uint8Array {
  const normalized = hex.startsWith("0x") ? hex.slice(2) : hex;
  if (normalized.length % 2 !== 0) throw new Error("Hex string must have an even length");
  const bytes = new Uint8Array(normalized.length / 2);
  for (let index = 0; index < bytes.length; index += 1) {
    const byte = Number.parseInt(normalized.slice(index * 2, index * 2 + 2), 16);
    if (Number.isNaN(byte)) throw new Error("Invalid hex character");
    bytes[index] = byte;
  }
  return bytes;
}

export function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}
