import { createHash } from "node:crypto";

import { MAX_METADATA_BYTES, PILOT_TOKEN_NAME, PILOT_TOKEN_SYMBOL } from "./constants.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

export function sha256Hex(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

export function parseAndValidatePilotMetadata(bytes) {
  let document;
  try {
    document = JSON.parse(Buffer.from(bytes).toString("utf8"));
  } catch {
    throw new Error("Metadata document is not valid UTF-8 JSON.");
  }

  assert(document && typeof document === "object" && !Array.isArray(document), "Metadata document must be a JSON object.");
  assert(document.name === PILOT_TOKEN_NAME, `Metadata name must equal "${PILOT_TOKEN_NAME}".`);
  assert(document.symbol === PILOT_TOKEN_SYMBOL, `Metadata symbol must equal "${PILOT_TOKEN_SYMBOL}".`);
  assert(typeof document.description === "string" && document.description.length > 0, "Metadata document must include a non-empty description.");
  assert(/devnet/i.test(document.description), "Metadata description must identify this as a Devnet pilot.");
  assert(/valueless|no monetary value|not.*investment/i.test(document.description), "Metadata description must state that this is valueless or not an investment.");

  return document;
}

export async function fetchMetadataCommitment(uri, expectedSha256, fetchImpl = globalThis.fetch) {
  if (typeof fetchImpl !== "function") {
    throw new Error("No Fetch API is available to retrieve the metadata document.");
  }

  let response;
  try {
    response = await fetchImpl(uri, {
      headers: { accept: "application/json" },
      redirect: "error",
      signal: AbortSignal.timeout(15_000),
    });
  } catch (error) {
    throw new Error(`Unable to retrieve TWC_METADATA_URI without redirects: ${error.message}`);
  }

  if (!response.ok) {
    throw new Error(`Metadata URI returned HTTP ${response.status}.`);
  }

  const contentLength = response.headers.get("content-length");
  if (contentLength && Number(contentLength) > MAX_METADATA_BYTES) {
    throw new Error(`Metadata document exceeds the ${MAX_METADATA_BYTES}-byte limit.`);
  }

  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes.byteLength === 0 || bytes.byteLength > MAX_METADATA_BYTES) {
    throw new Error(`Metadata document must be between 1 and ${MAX_METADATA_BYTES} bytes.`);
  }

  const actualSha256 = sha256Hex(bytes);
  if (actualSha256 !== expectedSha256) {
    throw new Error(`Metadata SHA-256 mismatch: expected ${expectedSha256}, received ${actualSha256}.`);
  }

  const document = parseAndValidatePilotMetadata(bytes);
  return Object.freeze({
    bytes,
    byteLength: bytes.byteLength,
    sha256: actualSha256,
    document,
  });
}
