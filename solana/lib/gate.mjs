import { address as parseAddress, getBase58Encoder, getPublicKeyFromAddress, verifySignature } from "@solana/kit";

import {
  DEVNET_GENESIS_HASH,
  FULL_NOVEL_UNLOCK_CHAPTERS,
  HOLDER_PROOF_MAX_MESSAGE_BYTES,
  HOLDER_PROOF_NETWORK,
  HOLDER_PROOF_PREAMBLE,
  HOLDER_PROOF_PURPOSE,
  PILOT_TOKEN_DECIMALS,
  STANDARD_TOKEN_PROGRAM_ADDRESS,
} from "./constants.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function normalizeLineValue(value, label, { maxLength = 512 } = {}) {
  assert(typeof value === "string", `${label} must be a string.`);
  const normalized = value.trim();
  assert(normalized.length > 0, `${label} is required.`);
  assert(!/[\r\n]/.test(normalized), `${label} must be a single line.`);
  assert(Buffer.byteLength(normalized, "utf8") <= maxLength, `${label} must be at most ${maxLength} UTF-8 bytes.`);
  return normalized;
}

function normalizeAddress(value, label) {
  try {
    return parseAddress(normalizeLineValue(value, label, { maxLength: 64 }));
  } catch (error) {
    throw new Error(`${label} must be a valid base58 Solana address: ${error.message}`);
  }
}

function normalizeNonce(value) {
  const nonce = normalizeLineValue(value, "Nonce", { maxLength: 128 });
  assert(/^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$/.test(nonce), "Nonce must be 8-128 URL-safe characters.");
  return nonce;
}

function normalizeIssuedAt(value) {
  const issuedAt = normalizeLineValue(value, "Issued At", { maxLength: 64 });
  const timestamp = Date.parse(issuedAt);
  assert(Number.isFinite(timestamp), "Issued At must be a valid ISO-8601 timestamp.");
  assert(new Date(timestamp).toISOString() === issuedAt, "Issued At must be a canonical UTC ISO-8601 timestamp.");
  return issuedAt;
}

function normalizeMessage(message) {
  assert(typeof message === "string", "Proof message must be a string.");
  const normalized = message.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trimEnd();
  assert(normalized.length > 0, "Proof message is required.");
  assert(
    Buffer.byteLength(normalized, "utf8") <= HOLDER_PROOF_MAX_MESSAGE_BYTES,
    `Proof message must be at most ${HOLDER_PROOF_MAX_MESSAGE_BYTES} UTF-8 bytes.`,
  );
  return normalized;
}

function parseLabeledLine(line, label) {
  assert(typeof line === "string" && line.startsWith(`${label}: `), `Expected line "${label}: ..." in holder proof message.`);
  return line.slice(label.length + 2);
}

function ensureExpectedOrigin(origin, expectedOrigin) {
  if (expectedOrigin === undefined) return;
  const normalizedExpectedOrigin = normalizeLineValue(expectedOrigin, "Expected origin", { maxLength: 512 });
  assert(origin === normalizedExpectedOrigin, `Holder proof origin ${origin} did not match the expected origin ${normalizedExpectedOrigin}.`);
}

function ensureFreshIssuedAt(issuedAt, maxAgeMs, now) {
  if (maxAgeMs === undefined || maxAgeMs === null) return;
  assert(Number.isFinite(maxAgeMs) && maxAgeMs > 0, "maxAgeMs must be a positive finite number when provided.");
  const issuedAtMs = Date.parse(issuedAt);
  const referenceMs = now instanceof Date ? now.getTime() : Number(now);
  assert(Number.isFinite(referenceMs), "A finite verification time is required when maxAgeMs is provided.");
  const ageMs = referenceMs - issuedAtMs;
  assert(ageMs >= -300_000, "Holder proof issuedAt is too far in the future.");
  assert(ageMs <= maxAgeMs, `Holder proof is older than the allowed ${maxAgeMs}ms window.`);
}

async function jsonRpcRequest(rpcUrl, method, params, fetchImpl = globalThis.fetch) {
  if (typeof fetchImpl !== "function") {
    throw new Error("No Fetch API is available to query Solana RPC.");
  }

  let response;
  try {
    response = await fetchImpl(rpcUrl, {
      method: "POST",
      headers: {
        accept: "application/json",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: `${method}-request`,
        method,
        params,
      }),
      signal: AbortSignal.timeout(15_000),
    });
  } catch (error) {
    throw new Error(`Unable to query Solana RPC (${method}): ${error.message}`);
  }

  if (!response.ok) {
    throw new Error(`Solana RPC (${method}) returned HTTP ${response.status}.`);
  }

  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new Error(`Solana RPC (${method}) returned invalid JSON: ${error.message}`);
  }

  if (payload?.error) {
    throw new Error(`Solana RPC (${method}) error ${payload.error.code}: ${payload.error.message}`);
  }

  return payload?.result;
}

function validateHolderProofShape(proof) {
  assert(proof && typeof proof === "object" && !Array.isArray(proof), "Holder proof must be a JSON object.");
  const address = normalizeAddress(proof.address, "Proof address");
  const message = normalizeMessage(proof.message);
  const signature = normalizeLineValue(proof.signature, "Proof signature", { maxLength: 128 });
  return Object.freeze({ address, message, signature });
}

export function buildHolderUnlockChallenge({ address, mintAddress, origin, nonce, issuedAt = new Date().toISOString() }) {
  const normalizedAddress = normalizeAddress(address, "Address");
  const normalizedMintAddress = normalizeAddress(mintAddress, "Mint");
  const normalizedOrigin = normalizeLineValue(origin, "Origin", { maxLength: 512 });
  const normalizedNonce = normalizeNonce(nonce);
  const normalizedIssuedAt = normalizeIssuedAt(issuedAt);

  return [
    HOLDER_PROOF_PREAMBLE,
    `Address: ${normalizedAddress}`,
    `Mint: ${normalizedMintAddress}`,
    `Network: ${HOLDER_PROOF_NETWORK}`,
    `Origin: ${normalizedOrigin}`,
    `Nonce: ${normalizedNonce}`,
    `Issued At: ${normalizedIssuedAt}`,
    `Purpose: ${HOLDER_PROOF_PURPOSE}`,
  ].join("\n");
}

export function parseHolderUnlockChallenge(message) {
  const normalized = normalizeMessage(message);
  const lines = normalized.split("\n");

  assert(lines.length === 8, "Holder proof message must contain exactly 8 lines.");
  assert(lines[0] === HOLDER_PROOF_PREAMBLE, `Holder proof preamble must equal "${HOLDER_PROOF_PREAMBLE}".`);

  const address = normalizeAddress(parseLabeledLine(lines[1], "Address"), "Address");
  const mintAddress = normalizeAddress(parseLabeledLine(lines[2], "Mint"), "Mint");
  const network = normalizeLineValue(parseLabeledLine(lines[3], "Network"), "Network", { maxLength: 32 });
  const origin = normalizeLineValue(parseLabeledLine(lines[4], "Origin"), "Origin", { maxLength: 512 });
  const nonce = normalizeNonce(parseLabeledLine(lines[5], "Nonce"));
  const issuedAt = normalizeIssuedAt(parseLabeledLine(lines[6], "Issued At"));
  const purpose = normalizeLineValue(parseLabeledLine(lines[7], "Purpose"), "Purpose", { maxLength: 256 });

  assert(network === HOLDER_PROOF_NETWORK, `Holder proof network must be ${HOLDER_PROOF_NETWORK}.`);
  assert(purpose === HOLDER_PROOF_PURPOSE, `Holder proof purpose must equal "${HOLDER_PROOF_PURPOSE}".`);

  return Object.freeze({
    address,
    mintAddress,
    network,
    origin,
    nonce,
    issuedAt,
    purpose,
    message: normalized,
  });
}

export async function verifyHolderProofSignature({ address, message, signature }) {
  const normalizedAddress = normalizeAddress(address, "Proof address");
  const normalizedMessage = normalizeMessage(message);
  const normalizedSignature = normalizeLineValue(signature, "Proof signature", { maxLength: 128 });

  let signatureBytes;
  try {
    signatureBytes = getBase58Encoder().encode(normalizedSignature);
  } catch (error) {
    throw new Error(`Proof signature must be valid base58: ${error.message}`);
  }
  assert(signatureBytes.byteLength === 64, "Proof signature must decode to 64 ed25519 signature bytes.");

  const publicKey = await getPublicKeyFromAddress(normalizedAddress);
  return await verifySignature(publicKey, signatureBytes, new TextEncoder().encode(normalizedMessage));
}

export async function assertDevnetRpc(rpcUrl, fetchImpl = globalThis.fetch) {
  const result = await jsonRpcRequest(rpcUrl, "getGenesisHash", [], fetchImpl);
  assert(
    result === DEVNET_GENESIS_HASH,
    `Refusing to continue: RPC genesis hash ${result} is not the Solana Devnet genesis hash ${DEVNET_GENESIS_HASH}.`,
  );
  return result;
}

export async function fetchHolderTokenBalance({ rpcUrl, ownerAddress, mintAddress, fetchImpl = globalThis.fetch }) {
  const normalizedOwnerAddress = normalizeAddress(ownerAddress, "Owner address");
  const normalizedMintAddress = normalizeAddress(mintAddress, "Mint address");
  await assertDevnetRpc(rpcUrl, fetchImpl);

  const result = await jsonRpcRequest(
    rpcUrl,
    "getTokenAccountsByOwner",
    [
      normalizedOwnerAddress,
      { mint: normalizedMintAddress },
      { encoding: "jsonParsed", commitment: "confirmed" },
    ],
    fetchImpl,
  );

  const accounts = Array.isArray(result?.value) ? result.value : [];
  let baseUnits = 0n;
  let decimals = PILOT_TOKEN_DECIMALS;

  for (const account of accounts) {
    assert(account?.account?.owner === STANDARD_TOKEN_PROGRAM_ADDRESS, "RPC returned a non-standard SPL token account.");
    const parsed = account.account?.data?.parsed;
    assert(parsed?.type === "account", "RPC returned an unexpected token-account payload.");

    const info = parsed.info ?? {};
    assert(info.owner === normalizedOwnerAddress, "RPC returned a token account owned by a different address.");
    assert(info.mint === normalizedMintAddress, "RPC returned a token account for a different mint.");

    const tokenAmount = info.tokenAmount ?? {};
    assert(typeof tokenAmount.amount === "string" && /^[0-9]+$/.test(tokenAmount.amount), "RPC returned an invalid token amount.");
    assert(Number.isInteger(tokenAmount.decimals) && tokenAmount.decimals >= 0, "RPC returned invalid token decimals.");

    if (baseUnits === 0n) {
      decimals = tokenAmount.decimals;
    } else {
      assert(tokenAmount.decimals === decimals, "RPC returned inconsistent token decimals across holder accounts.");
    }

    baseUnits += BigInt(tokenAmount.amount);
  }

  return Object.freeze({
    ownerAddress: normalizedOwnerAddress,
    mintAddress: normalizedMintAddress,
    accountCount: accounts.length,
    baseUnits,
    decimals,
  });
}

export function formatTokenAmount(baseUnits, decimals) {
  const rawBaseUnits = typeof baseUnits === "bigint" ? baseUnits : BigInt(baseUnits);
  assert(Number.isInteger(decimals) && decimals >= 0, "Token decimals must be a non-negative integer.");

  const negative = rawBaseUnits < 0n;
  const absolute = negative ? -rawBaseUnits : rawBaseUnits;
  const divisor = 10n ** BigInt(decimals);
  const whole = absolute / divisor;
  const fraction = absolute % divisor;

  if (fraction === 0n || decimals === 0) {
    return `${negative ? "-" : ""}${whole.toString()}`;
  }

  return `${negative ? "-" : ""}${whole.toString()}.${fraction
    .toString()
    .padStart(decimals, "0")
    .replace(/0+$/, "")}`;
}

export async function verifyHolderUnlock({
  rpcUrl,
  manifest,
  proof,
  fetchImpl = globalThis.fetch,
  expectedOrigin,
  maxAgeMs,
  now = Date.now(),
}) {
  assert(manifest?.mint?.address, "A deployment manifest with mint.address is required.");
  assert(manifest?.pilot?.network, "A deployment manifest with pilot.network is required.");
  assert(Number.isInteger(manifest?.mint?.decimals), "A deployment manifest with mint.decimals is required.");

  const normalizedProof = validateHolderProofShape(proof);
  const challenge = parseHolderUnlockChallenge(normalizedProof.message);

  assert(challenge.address === normalizedProof.address, "Proof address does not match the signed challenge address.");
  assert(challenge.mintAddress === manifest.mint.address, "Signed challenge mint does not match the deployment manifest mint.");
  assert(challenge.network === manifest.pilot.network, "Signed challenge network does not match the deployment manifest network.");
  ensureExpectedOrigin(challenge.origin, expectedOrigin);
  ensureFreshIssuedAt(challenge.issuedAt, maxAgeMs, now);

  const signatureValid = await verifyHolderProofSignature(normalizedProof);
  assert(signatureValid, "Holder signature verification failed.");

  const balance = await fetchHolderTokenBalance({
    rpcUrl,
    ownerAddress: normalizedProof.address,
    mintAddress: manifest.mint.address,
    fetchImpl,
  });

  assert(balance.decimals === manifest.mint.decimals, "Holder balance decimals do not match the deployment manifest.");
  assert(balance.baseUnits > 0n, `Holder ${normalizedProof.address} does not currently hold any ${manifest.identity?.onChainSymbol ?? "tTWC"}.`);

  return Object.freeze({
    status: "verified",
    address: normalizedProof.address,
    network: manifest.pilot.network,
    mintAddress: manifest.mint.address,
    balanceBaseUnits: balance.baseUnits.toString(),
    balanceTokens: formatTokenAmount(balance.baseUnits, balance.decimals),
    tokenDecimals: balance.decimals,
    tokenAccounts: balance.accountCount,
    unlockedChapters: FULL_NOVEL_UNLOCK_CHAPTERS,
    fullNovelUnlock: true,
    origin: challenge.origin,
    nonce: challenge.nonce,
    issuedAt: challenge.issuedAt,
  });
}
