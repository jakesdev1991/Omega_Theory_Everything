// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
/**
 * NIP-01 event primitives with STRICT verification.
 *
 * Mirrors web/src/lib/nostr-store.ts verifyEventStrict(): the event id is
 * always recomputed from the serialized [0, pubkey, created_at, kind, tags,
 * content] array before the schnorr signature is checked. Signature-only
 * checks are insufficient: they validate the signature against the *declared*
 * id, so rewritten content/tags would pass.
 */

import { randomBytes } from "node:crypto";

import { schnorr } from "@noble/curves/secp256k1";
import { sha256 } from "@noble/hashes/sha256";

import { hexToNpub, nsecToHex, npubToHex } from "./bech32.mjs";

export function hexToBytes(hex) {
  return Uint8Array.from(Buffer.from(hex, "hex"));
}

export function hexToSecretBytes(secret) {
  const trimmed = String(secret).trim();
  const normalized = trimmed.startsWith("nsec1") ? nsecToHex(trimmed) : trimmed.replace(/^0x/, "");
  const bytes = Uint8Array.from(Buffer.from(normalized, "hex"));
  if (bytes.length !== 32) throw new Error("Secret key must be 32 bytes");
  return bytes;
}

export function getPublicKeyHex(secret) {
  return Buffer.from(schnorr.getPublicKey(hexToSecretBytes(secret))).toString("hex");
}

export function npubOrHexToHex(value) {
  const trimmed = String(value).trim();
  if (trimmed.startsWith("npub1")) return npubToHex(trimmed);
  if (!/^[0-9a-fA-F]{64}$/.test(trimmed)) throw new Error(`Not a pubkey: ${trimmed}`);
  return trimmed.toLowerCase();
}

export function serializeEvent(event) {
  return JSON.stringify([0, event.pubkey, event.created_at, event.kind, event.tags, event.content]);
}

export function getEventHashHex(event) {
  return Buffer.from(sha256(new TextEncoder().encode(serializeEvent(event)))).toString("hex");
}

export function signEvent(template, secret) {
  const secretBytes = hexToSecretBytes(secret);
  const event = { ...template, pubkey: Buffer.from(schnorr.getPublicKey(secretBytes)).toString("hex") };
  event.id = getEventHashHex(event);
  event.sig = Buffer.from(schnorr.sign(hexToBytes(event.id), secretBytes, randomBytes(32))).toString("hex");
  return event;
}

export function verifyEventStrict(event) {
  try {
    if (getEventHashHex(event) !== event.id) return false;
    return schnorr.verify(hexToBytes(event.sig), hexToBytes(event.id), hexToBytes(event.pubkey));
  } catch {
    return false;
  }
}

export { hexToNpub, npubToHex };
