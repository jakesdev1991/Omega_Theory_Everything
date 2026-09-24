import {
  createHash,
  createPublicKey,
  createSign,
  createVerify,
  generateKeyPairSync,
} from "node:crypto";

import { parseHolderUnlockChallenge } from "./proof.mjs";
import { TESTNET_NETWORK } from "./constants.mjs";

export const HOLDER_PROOF_ENVELOPE_VERSION = 1;
export const DEFAULT_MAX_CHALLENGE_AGE_MS = 5 * 60 * 1000;
export const DEFAULT_CLOCK_SKEW_MS = 30 * 1000;
const FIXTURE_SIGNATURE_ALGORITHM = "ecdsa-secp256k1-sha256";

function assert(condition, message) {
  if (!condition) throw new Error(`Holder verification error: ${message}`);
}

function stringValue(value, label, { maxLength = 512 } = {}) {
  assert(typeof value === "string", `${label} must be a string.`);
  const normalized = value.trim();
  assert(normalized.length > 0, `${label} is required.`);
  assert(normalized.length <= maxLength, `${label} is too long.`);
  assert(!/[\r\n]/.test(normalized), `${label} must be a single line.`);
  return normalized;
}

function assetIdValue(value, label = "assetId") {
  const assetId = stringValue(value, label, { maxLength: 64 }).toLowerCase();
  assert(/^[a-f0-9]{64}$/.test(assetId), `${label} must be a 64-character hexadecimal Taproot Asset ID.`);
  return assetId;
}

function httpUrlValue(value, label) {
  const url = stringValue(value, label);
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error(`Holder verification error: ${label} must be an absolute HTTP(S) URL.`);
  }
  assert(["http:", "https:"].includes(parsed.protocol), `${label} must be an absolute HTTP(S) URL.`);
  assert(!parsed.username && !parsed.password, `${label} must not contain credentials.`);
  return url;
}

function positiveIntegerString(value, label) {
  const normalized = stringValue(value, label, { maxLength: 80 });
  assert(/^[1-9][0-9]*$/.test(normalized), `${label} must be a positive integer string.`);
  return normalized;
}

function base64Value(value, label) {
  const normalized = stringValue(value, label, { maxLength: 8192 });
  assert(/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(normalized), `${label} must be base64.`);
  return normalized;
}

function proofHash(message) {
  return createHash("sha256").update(message, "utf8").digest("hex");
}

function assertFresh(issuedAt, now, maxAgeMs, clockSkewMs) {
  const issuedAtMs = Date.parse(issuedAt);
  const nowMs = now instanceof Date ? now.getTime() : now;
  assert(Number.isFinite(nowMs), "now must be a valid Date or millisecond timestamp.");
  assert(issuedAtMs <= nowMs + clockSkewMs, "challenge was issued in the future.");
  assert(nowMs - issuedAtMs <= maxAgeMs, "challenge has expired.");
}

function proofSigningPayload(challenge, proof) {
  return [
    "AMITY HOLDER PROOF ENVELOPE V1",
    `Challenge: ${challenge.message}`,
    `Source: ${proof.source}`,
    `Asset ID: ${proof.assetId}`,
    `Holder Address: ${proof.holderAddress}`,
    `Universe: ${proof.universeUrl}`,
    `Amount: ${proof.amount}`,
    `Proof ID: ${proof.proofId}`,
  ].join("\n");
}

function verifyProofSignature(challenge, proof) {
  assert(proof.signatureAlgorithm === FIXTURE_SIGNATURE_ALGORITHM, "unsupported proof signature algorithm; Schnorr/Taproot wallet verification is not implemented yet.");
  assert(typeof proof.publicKeyPem === "string" && proof.publicKeyPem.length > 0, "proof.publicKeyPem is required.");
  assert(proof.publicKeyPem.length <= 4096, "proof.publicKeyPem is too long.");
  const publicKeyPem = proof.publicKeyPem;
  const signature = Buffer.from(base64Value(proof.signature, "proof.signature"), "base64");

  let publicKey;
  try {
    publicKey = createPublicKey(publicKeyPem);
  } catch {
    throw new Error("Holder verification error: proof.publicKeyPem is not a valid public key.");
  }

  assert(publicKey.asymmetricKeyType === "ec", "proof.publicKeyPem must contain an EC public key.");
  assert(publicKey.asymmetricKeyDetails?.namedCurve === "secp256k1", "proof.publicKeyPem must use secp256k1.");

  const verifier = createVerify("SHA256");
  verifier.update(proofSigningPayload(challenge, proof), "utf8");
  verifier.end();
  assert(verifier.verify(publicKey, signature), "proof signature does not validate the canonical challenge and holdings claim.");
}

/**
 * A source-neutral proof envelope for the future tapd + Universe adapter.
 *
 * `acceptedBySource` means an injected source accepted the envelope. The
 * canonical claim is also ECDSA-verified here for the local test path. The
 * fixture public key is not bound to a wallet address, so this is integrity
 * checking only. It is not a Taproot wallet signature check: Schnorr/Taproot
 * support and live holdings verification remain intentionally unavailable.
 */
export function validateHolderProofEnvelope({
  challenge,
  proof,
  expected,
  now = Date.now(),
  maxAgeMs = DEFAULT_MAX_CHALLENGE_AGE_MS,
  clockSkewMs = DEFAULT_CLOCK_SKEW_MS,
  allowFixture = false,
}) {
  const parsedChallenge = typeof challenge === "string" ? parseHolderUnlockChallenge(challenge) : challenge;
  assert(parsedChallenge && typeof parsedChallenge === "object", "challenge is required.");

  const expectedNetwork = expected?.network ?? TESTNET_NETWORK;
  assert(expectedNetwork === TESTNET_NETWORK, `network must remain ${TESTNET_NETWORK}.`);
  assert(parsedChallenge.network === expectedNetwork, `challenge network must be ${expectedNetwork}.`);
  assertFresh(parsedChallenge.issuedAt, now, maxAgeMs, clockSkewMs);

  const expectedAssetId = assetIdValue(expected?.assetId, "expected.assetId");
  const challengeAssetId = assetIdValue(parsedChallenge.assetId, "challenge assetId");
  assert(challengeAssetId === expectedAssetId, "challenge asset ID does not match configured AMITY asset.");

  const expectedUniverseUrl = httpUrlValue(expected?.universeUrl, "expected.universeUrl");
  assert(parsedChallenge.universeUrl === expectedUniverseUrl, "challenge Universe URL does not match configured Universe.");

  if (expected?.origin !== undefined) {
    assert(parsedChallenge.origin === httpUrlValue(expected.origin, "expected.origin"), "challenge origin does not match the requesting origin.");
  }

  assert(proof && typeof proof === "object", "proof envelope is required.");
  assert(proof.version === HOLDER_PROOF_ENVELOPE_VERSION, `proof version must be ${HOLDER_PROOF_ENVELOPE_VERSION}.`);

  const source = stringValue(proof.source, "proof.source", { maxLength: 64 });
  assert(source !== "fixture" || allowFixture, "fixture proofs are disabled outside the test harness.");
  assert(source === "fixture" || source === "tapd-universe", "proof.source must be fixture or tapd-universe.");
  assert(proof.acceptedBySource === true, "proof was not accepted by its source verifier.");
  assert(proof.network === expectedNetwork, `proof network must be ${expectedNetwork}.`);
  assert(assetIdValue(proof.assetId, "proof.assetId") === expectedAssetId, "proof asset ID does not match configured AMITY asset.");
  assert(stringValue(proof.holderAddress, "proof.holderAddress", { maxLength: 256 }) === parsedChallenge.address, "proof holder does not match the challenge address.");
  assert(httpUrlValue(proof.universeUrl, "proof.universeUrl") === expectedUniverseUrl, "proof Universe URL does not match the configured Universe.");
  const amount = positiveIntegerString(proof.amount, "proof.amount");
  const id = stringValue(proof.proofId, "proof.proofId", { maxLength: 256 });
  const digest = stringValue(proof.challengeDigest, "proof.challengeDigest", { maxLength: 64 }).toLowerCase();
  assert(/^[a-f0-9]{64}$/.test(digest), "proof.challengeDigest must be a SHA-256 hex digest.");
  assert(digest === proofHash(parsedChallenge.message), "proof.challengeDigest does not bind to the canonical challenge.");

  verifyProofSignature(parsedChallenge, {
    ...proof,
    source,
    assetId: expectedAssetId,
    holderAddress: parsedChallenge.address,
    universeUrl: expectedUniverseUrl,
    amount,
    proofId: id,
  });

  return Object.freeze({
    status: "accepted",
    source,
    claimSignatureChecked: true,
    holderIdentityBound: false,
    taprootAssetOwnershipChecked: false,
    challenge: parsedChallenge,
    proofId: id,
    amount,
    assetId: expectedAssetId,
    network: expectedNetwork,
    universeUrl: expectedUniverseUrl,
  });
}

/**
 * Verify a holder claim through an injected source adapter. The adapter is
 * intentionally small so a real tapd/Universe client can replace the fixture
 * without changing the policy checks or the web API contract.
 */
export async function verifyHolderProof({
  challenge,
  expected,
  proofSource,
  now = Date.now(),
  maxAgeMs = DEFAULT_MAX_CHALLENGE_AGE_MS,
  clockSkewMs = DEFAULT_CLOCK_SKEW_MS,
  allowFixture = false,
}) {
  assert(proofSource && typeof proofSource.verify === "function", "proofSource.verify must be provided.");
  const parsedChallenge = typeof challenge === "string" ? parseHolderUnlockChallenge(challenge) : challenge;
  const proof = await proofSource.verify({ challenge: parsedChallenge });
  return validateHolderProofEnvelope({
    challenge: parsedChallenge,
    proof,
    expected,
    now,
    maxAgeMs,
    clockSkewMs,
    allowFixture,
  });
}

/**
 * Explicitly test-only proof source. It creates a real secp256k1 ECDSA
 * signature over the canonical test claim, but it does not prove Taproot Asset
 * ownership and must never be used as a production or web-unlock source.
 */
export function createFixtureProofSource({ amount = "1", proofId = "fixture-proof-1" } = {}) {
  const { privateKey, publicKey } = generateKeyPairSync("ec", { namedCurve: "secp256k1" });
  const publicKeyPem = publicKey.export({ type: "spki", format: "pem" }).toString();

  return Object.freeze({
    mode: "fixture",
    async verify({ challenge }) {
      const proof = {
        version: HOLDER_PROOF_ENVELOPE_VERSION,
        source: "fixture",
        acceptedBySource: true,
        signatureAlgorithm: FIXTURE_SIGNATURE_ALGORITHM,
        publicKeyPem,
        network: challenge.network,
        assetId: challenge.assetId.toLowerCase(),
        holderAddress: challenge.address,
        universeUrl: challenge.universeUrl,
        amount,
        proofId,
        challengeDigest: proofHash(challenge.message),
      };
      const signer = createSign("SHA256");
      signer.update(proofSigningPayload(challenge, proof), "utf8");
      signer.end();
      return {
        ...proof,
        signature: signer.sign(privateKey).toString("base64"),
      };
    },
  });
}
