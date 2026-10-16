// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import { getAddress, isAddress, verifyMessage } from "ethers";
import bs58 from "bs58";
import nacl from "tweetnacl";

export const UNLOCK_STORAGE_KEY = "omega.unlock.proof";
export const FULL_NOVEL_UNLOCK_CHAPTERS = 16;
export const UNLOCK_MESSAGE_PREAMBLE = "OMEGA OPTIONAL NOVEL READER PROOF";
export const OMEGA_NETWORK = "ethereum-sepolia";
export const TWC_NETWORK = "solana-devnet";

export type UnlockCurrency = "OMEGA" | "TWC";

export type UnlockNetwork = typeof OMEGA_NETWORK | typeof TWC_NETWORK;

export interface UnlockProofPayload {
  message?: string;
  signature?: string;
}

export interface VerifiedUnlock {
  ok: true;
  address: string;
  currency: UnlockCurrency;
  network: UnlockNetwork;
  origin: string;
  timestamp: string;
  unlocked: number;
  message: string;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

function normalizeMessage(message: string) {
  const normalized = message.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
  assert(normalized.length > 0, "Unlock message is required.");
  return normalized;
}

function parseLabeledLine(line: string, label: string) {
  assert(line.startsWith(`${label}: `), `Expected line "${label}: ..." in unlock message.`);
  return line.slice(label.length + 2).trim();
}

export function buildUnlockMessage({
  currency,
  address,
  network,
  origin,
  timestamp = new Date().toISOString(),
}: {
  currency: UnlockCurrency;
  address: string;
  network: UnlockNetwork;
  origin: string;
  timestamp?: string;
}) {
  return [
    UNLOCK_MESSAGE_PREAMBLE,
    `Currency: ${currency}`,
    `Address: ${address}`,
    `Network: ${network}`,
    `Origin: ${origin}`,
    `Timestamp: ${timestamp}`,
    "Reader demo: full chapter view",
  ].join("\n");
}

export function parseUnlockMessage(message: string) {
  const normalized = normalizeMessage(message);
  const lines = normalized.split("\n");
  assert(lines.length === 7, "Unlock message must contain exactly 7 lines.");
  assert(lines[0] === UNLOCK_MESSAGE_PREAMBLE, `Unlock message preamble must equal "${UNLOCK_MESSAGE_PREAMBLE}".`);

  const currency = parseLabeledLine(lines[1], "Currency");
  assert(currency === "OMEGA" || currency === "TWC", "Currency must be OMEGA or TWC.");

  const address = parseLabeledLine(lines[2], "Address");
  const network = parseLabeledLine(lines[3], "Network");
  const origin = parseLabeledLine(lines[4], "Origin");
  const timestamp = parseLabeledLine(lines[5], "Timestamp");
  const readerDemo = parseLabeledLine(lines[6], "Reader demo");

  let parsedOrigin: URL;
  try {
    parsedOrigin = new URL(origin);
  } catch {
    throw new Error("Origin must be a canonical HTTP(S) origin.");
  }
  assert(
    (parsedOrigin.protocol === "http:" || parsedOrigin.protocol === "https:") && parsedOrigin.origin === origin,
    "Origin must be a canonical HTTP(S) origin (scheme and host only).",
  );
  assert(
    readerDemo === "full chapter view",
    "Reader demo line must equal " + JSON.stringify("full chapter view") + ".",
  );

  const parsedTimestamp = Date.parse(timestamp);
  assert(Number.isFinite(parsedTimestamp), "Timestamp must be a valid ISO-8601 datetime.");
  assert(new Date(parsedTimestamp).toISOString() === timestamp, "Timestamp must be a canonical UTC ISO-8601 datetime.");

  if (currency === "OMEGA") {
    assert(network === OMEGA_NETWORK, `OMEGA proofs must target ${OMEGA_NETWORK}.`);
    assert(isAddress(address), "OMEGA address must be a valid EVM address.");
    return {
      currency,
      address: getAddress(address),
      network,
      origin,
      timestamp,
      message: normalized,
    } as const;
  }

  assert(network === TWC_NETWORK, `TWC proofs must target ${TWC_NETWORK}.`);
  let publicKey: Uint8Array;
  try {
    publicKey = bs58.decode(address);
  } catch {
    throw new Error("TWC address must be a valid base58 Solana address.");
  }
  assert(publicKey.length === 32, "TWC address must decode to a 32-byte Solana public key.");

  return {
    currency,
    address,
    publicKey,
    network,
    origin,
    timestamp,
    message: normalized,
  } as const;
}

function decodeBase64Signature(signature: string) {
  const normalized = signature.trim();
  assert(normalized.length > 0, "Signature is required.");
  const bytes = Uint8Array.from(Buffer.from(normalized, "base64"));
  assert(bytes.length === 64, "TWC signatures must be 64-byte base64 ed25519 signatures.");
  return bytes;
}

export function verifyUnlockProof(
  { message, signature }: UnlockProofPayload,
  expectedOrigin?: string,
): VerifiedUnlock {
  assert(typeof message === "string" && message.trim().length > 0, "message and signature are required.");
  assert(typeof signature === "string" && signature.trim().length > 0, "message and signature are required.");

  const parsed = parseUnlockMessage(message);
  if (expectedOrigin !== undefined) {
    let expectedUrl: URL;
    try {
      expectedUrl = new URL(expectedOrigin);
    } catch {
      throw new Error("Expected origin must be a canonical HTTP(S) origin.");
    }
    assert(
      (expectedUrl.protocol === "http:" || expectedUrl.protocol === "https:") && expectedUrl.origin === expectedOrigin,
      "Expected origin must be a canonical HTTP(S) origin.",
    );
    assert(parsed.origin === expectedUrl.origin, "Unlock proof origin does not match this site's origin.");
  }

  if (parsed.currency === "OMEGA") {
    let recovered: string;
    try {
      recovered = getAddress(verifyMessage(parsed.message, signature));
    } catch {
      throw new Error("Signature verification failed.");
    }
    assert(recovered === parsed.address, "Signature verification failed.");
    return {
      ok: true,
      address: parsed.address,
      currency: parsed.currency,
      network: parsed.network,
      origin: parsed.origin,
      timestamp: parsed.timestamp,
      unlocked: FULL_NOVEL_UNLOCK_CHAPTERS,
      message: `Verified ${parsed.currency} proof for ${parsed.address}. Signature matched.`,
    };
  }

  const signatureBytes = decodeBase64Signature(signature);
  const verified = nacl.sign.detached.verify(
    new TextEncoder().encode(parsed.message),
    signatureBytes,
    parsed.publicKey,
  );
  assert(verified, "Signature verification failed.");

  return {
    ok: true,
    address: parsed.address,
    currency: parsed.currency,
    network: parsed.network,
    origin: parsed.origin,
    timestamp: parsed.timestamp,
    unlocked: FULL_NOVEL_UNLOCK_CHAPTERS,
    message: `Verified ${parsed.currency} proof for ${parsed.address}. Signature matched.`,
  };
}
