import {
  AMITY_HOLDER_PROOF_MAX_MESSAGE_BYTES,
  AMITY_HOLDER_PROOF_PREAMBLE,
  AMITY_HOLDER_PROOF_PURPOSE,
  TESTNET_NETWORK,
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
    Buffer.byteLength(normalized, "utf8") <= AMITY_HOLDER_PROOF_MAX_MESSAGE_BYTES,
    `Proof message must be at most ${AMITY_HOLDER_PROOF_MAX_MESSAGE_BYTES} UTF-8 bytes.`,
  );
  return normalized;
}

function parseLabeledLine(line, label) {
  assert(typeof line === "string" && line.startsWith(`${label}: `), `Expected line "${label}: ..." in AMITY holder proof message.`);
  return line.slice(label.length + 2);
}

export function buildHolderUnlockChallenge({
  address,
  assetId,
  universeUrl,
  origin,
  nonce,
  issuedAt = new Date().toISOString(),
}) {
  const normalizedAddress = normalizeLineValue(address, "Address", { maxLength: 256 });
  const normalizedAssetId = normalizeLineValue(assetId, "Asset ID", { maxLength: 128 });
  const normalizedUniverseUrl = normalizeLineValue(universeUrl, "Universe", { maxLength: 512 });
  const normalizedOrigin = normalizeLineValue(origin, "Origin", { maxLength: 512 });
  const normalizedNonce = normalizeNonce(nonce);
  const normalizedIssuedAt = normalizeIssuedAt(issuedAt);

  return [
    AMITY_HOLDER_PROOF_PREAMBLE,
    `Address: ${normalizedAddress}`,
    `Asset ID: ${normalizedAssetId}`,
    `Network: ${TESTNET_NETWORK}`,
    `Universe: ${normalizedUniverseUrl}`,
    `Origin: ${normalizedOrigin}`,
    `Nonce: ${normalizedNonce}`,
    `Issued At: ${normalizedIssuedAt}`,
    `Purpose: ${AMITY_HOLDER_PROOF_PURPOSE}`,
  ].join("\n");
}

export function parseHolderUnlockChallenge(message) {
  const normalized = normalizeMessage(message);
  const lines = normalized.split("\n");

  assert(lines.length === 9, "AMITY holder proof message must contain exactly 9 lines.");
  assert(lines[0] === AMITY_HOLDER_PROOF_PREAMBLE, `AMITY holder proof preamble must equal "${AMITY_HOLDER_PROOF_PREAMBLE}".`);

  const address = normalizeLineValue(parseLabeledLine(lines[1], "Address"), "Address", { maxLength: 256 });
  const assetId = normalizeLineValue(parseLabeledLine(lines[2], "Asset ID"), "Asset ID", { maxLength: 128 });
  const network = normalizeLineValue(parseLabeledLine(lines[3], "Network"), "Network", { maxLength: 32 });
  const universeUrl = normalizeLineValue(parseLabeledLine(lines[4], "Universe"), "Universe", { maxLength: 512 });
  const origin = normalizeLineValue(parseLabeledLine(lines[5], "Origin"), "Origin", { maxLength: 512 });
  const nonce = normalizeNonce(parseLabeledLine(lines[6], "Nonce"));
  const issuedAt = normalizeIssuedAt(parseLabeledLine(lines[7], "Issued At"));
  const purpose = normalizeLineValue(parseLabeledLine(lines[8], "Purpose"), "Purpose", { maxLength: 256 });

  assert(network === TESTNET_NETWORK, `AMITY holder proof network must be ${TESTNET_NETWORK}.`);
  assert(purpose === AMITY_HOLDER_PROOF_PURPOSE, `AMITY holder proof purpose must equal "${AMITY_HOLDER_PROOF_PURPOSE}".`);

  return Object.freeze({
    address,
    assetId,
    network,
    universeUrl,
    origin,
    nonce,
    issuedAt,
    purpose,
    message: normalized,
  });
}
