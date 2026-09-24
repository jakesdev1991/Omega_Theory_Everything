// Offline fixture harness for verifying Solana TWC holder proofs against mock or devnet manifests
// Mirrors amity/scripts/run-fixture-harness.mjs

import assert from "node:assert/strict";
import { generateKeyPairSigner, getBase58Decoder, signBytes } from "@solana/kit";

import {
  DEVNET_GENESIS_HASH,
  PILOT_TOKEN_DECIMALS,
  PILOT_TOKEN_SYMBOL,
  STANDARD_TOKEN_PROGRAM_ADDRESS,
} from "../lib/constants.mjs";
import { buildHolderUnlockChallenge, verifyHolderUnlock } from "../lib/gate.mjs";

const mintSigner = await generateKeyPairSigner();
const holderSigner = await generateKeyPairSigner();

const issuedAt = "2026-09-24T00:00:00.000Z";
const now = Date.parse("2026-09-24T00:01:00.000Z");

const manifest = {
  manifestVersion: "0.1.0",
  pilot: { network: "devnet", genesisHash: DEVNET_GENESIS_HASH, valueless: true },
  identity: { onChainSymbol: PILOT_TOKEN_SYMBOL },
  mint: { address: mintSigner.address, decimals: PILOT_TOKEN_DECIMALS },
};

const message = buildHolderUnlockChallenge({
  address: holderSigner.address,
  mintAddress: mintSigner.address,
  origin: "https://omega.example/novel",
  nonce: "fixture-twc-nonce-1234",
  issuedAt,
});

const signatureBytes = await signBytes(holderSigner.keyPair.privateKey, new TextEncoder().encode(message));
const signature = getBase58Decoder().decode(signatureBytes);

const mockFetch = async (_url, options) => {
  const payload = JSON.parse(options.body);
  if (payload.method === "getGenesisHash") {
    return {
      ok: true,
      async json() {
        return { jsonrpc: "2.0", id: payload.id, result: DEVNET_GENESIS_HASH };
      },
    };
  }

  if (payload.method === "getTokenAccountsByOwner") {
    return {
      ok: true,
      async json() {
        return {
          jsonrpc: "2.0",
          id: payload.id,
          result: {
            value: [
              {
                pubkey: "FixtureTokenAccountPubkey",
                account: {
                  owner: STANDARD_TOKEN_PROGRAM_ADDRESS,
                  data: {
                    program: "spl-token",
                    parsed: {
                      type: "account",
                      info: {
                        mint: mintSigner.address,
                        owner: holderSigner.address,
                        tokenAmount: {
                          amount: "5000000000", // 5 tokens
                          decimals: PILOT_TOKEN_DECIMALS,
                          uiAmount: 5,
                          uiAmountString: "5",
                        },
                      },
                    },
                  },
                },
              },
            ],
          },
        };
      },
    };
  }

  throw new Error(`Unexpected mock method ${payload.method}`);
};

const result = await verifyHolderUnlock({
  rpcUrl: "https://api.devnet.solana.com",
  manifest,
  proof: {
    address: holderSigner.address,
    message,
    signature,
  },
  expectedOrigin: "https://omega.example/novel",
  maxAgeMs: 300_000,
  now,
  fetchImpl: mockFetch,
});

assert.equal(result.status, "verified");
assert.equal(result.address, holderSigner.address);
assert.equal(result.mintAddress, mintSigner.address);
assert.equal(result.balanceTokens, "5");
assert.equal(result.fullNovelUnlock, true);

console.log("TWC Solana fixture readiness harness");
console.log("  network                 devnet");
console.log(`  mint address            ${result.mintAddress}`);
console.log(`  holder                  ${result.address}`);
console.log(`  balance tokens          ${result.balanceTokens} ${PILOT_TOKEN_SYMBOL}`);
console.log("  signature verification  valid (Ed25519 signer)");
console.log("  origin match            verified (https://omega.example/novel)");
console.log("  novel unlock status     granted (all chapters)");
console.log("\nFixture policy, signature binding, and balance verification passed.");
