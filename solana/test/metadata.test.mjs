import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  fetchMetadataCommitment,
  parseAndValidatePilotMetadata,
  sha256Hex,
} from "../lib/metadata.mjs";

test("the checked-in tTWC metadata template declares the pilot identity and warnings", async () => {
  const bytes = await readFile(new URL("../metadata/twc-devnet.template.json", import.meta.url));
  const document = parseAndValidatePilotMetadata(bytes);
  assert.equal(document.name, "Token of the World Citizen");
  assert.equal(document.symbol, "tTWC");
  assert.match(document.description, /Devnet/i);
});

test("off-chain metadata is hash-pinned before deployment", async () => {
  const bytes = new TextEncoder().encode(JSON.stringify({
    name: "Token of the World Citizen",
    symbol: "tTWC",
    description: "A valueless Solana Devnet pilot; it is not an investment.",
  }));
  const expected = sha256Hex(bytes);
  const fakeFetch = async () => new Response(bytes, {
    status: 200,
    headers: { "content-type": "application/json", "content-length": String(bytes.length) },
  });

  const result = await fetchMetadataCommitment("https://arweave.net/example", expected, fakeFetch);
  assert.equal(result.sha256, expected);
  assert.equal(result.document.symbol, "tTWC");

  await assert.rejects(
    fetchMetadataCommitment("https://arweave.net/example", "0".repeat(64), fakeFetch),
    /SHA-256 mismatch/,
  );
});

test("metadata safety checks reject a document that could misrepresent the pilot", () => {
  const bytes = new TextEncoder().encode(JSON.stringify({
    name: "Token of the World Citizen",
    symbol: "TWC",
    description: "A new token",
  }));
  assert.throws(() => parseAndValidatePilotMetadata(bytes), /symbol/);
});
