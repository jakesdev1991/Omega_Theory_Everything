import assert from "node:assert/strict";
import test from "node:test";

import { generateKeyPairSigner, getBase58Decoder, signBytes } from "@solana/kit";

import {
  DEVNET_GENESIS_HASH,
  FULL_NOVEL_UNLOCK_CHAPTERS,
  PILOT_TOKEN_DECIMALS,
  PILOT_TOKEN_SYMBOL,
  STANDARD_TOKEN_PROGRAM_ADDRESS,
} from "../lib/constants.mjs";
import {
  buildHolderUnlockChallenge,
  fetchHolderTokenBalance,
  formatTokenAmount,
  parseHolderUnlockChallenge,
  verifyHolderProofSignature,
  verifyHolderUnlock,
} from "../lib/gate.mjs";

function createManifest(mintAddress) {
  return {
    pilot: { network: "devnet" },
    identity: { onChainSymbol: PILOT_TOKEN_SYMBOL },
    mint: { address: mintAddress, decimals: PILOT_TOKEN_DECIMALS },
  };
}

function createRpcFetchStub({ ownerAddress, mintAddress, tokenAmounts = ["0"] }) {
  return async (_url, options) => {
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
              value: tokenAmounts.map((amount, index) => ({
                pubkey: `TokenAccount${index}`,
                account: {
                  owner: STANDARD_TOKEN_PROGRAM_ADDRESS,
                  data: {
                    program: "spl-token",
                    parsed: {
                      type: "account",
                      info: {
                        mint: mintAddress,
                        owner: ownerAddress,
                        tokenAmount: {
                          amount,
                          decimals: PILOT_TOKEN_DECIMALS,
                          uiAmount: Number(amount) / 10 ** PILOT_TOKEN_DECIMALS,
                          uiAmountString: amount,
                        },
                      },
                    },
                  },
                },
              })),
            },
          };
        },
      };
    }

    throw new Error(`Unexpected RPC method ${payload.method}`);
  };
}

test("holder unlock challenges are canonicalized and signature-verifiable", async () => {
  const signer = await generateKeyPairSigner();
  const mint = await generateKeyPairSigner();
  const issuedAt = "2026-09-23T21:00:00.000Z";
  const message = buildHolderUnlockChallenge({
    address: signer.address,
    mintAddress: mint.address,
    origin: "https://omega.example/novel",
    nonce: "nonce-12345678",
    issuedAt,
  });

  const parsed = parseHolderUnlockChallenge(message);
  assert.equal(parsed.address, signer.address);
  assert.equal(parsed.mintAddress, mint.address);
  assert.equal(parsed.origin, "https://omega.example/novel");
  assert.equal(parsed.nonce, "nonce-12345678");
  assert.equal(parsed.issuedAt, issuedAt);

  const signatureBytes = await signBytes(signer.keyPair.privateKey, new TextEncoder().encode(message));
  const signature = getBase58Decoder().decode(signatureBytes);

  assert.equal(
    await verifyHolderProofSignature({ address: signer.address, message, signature }),
    true,
  );

  assert.equal(
    await verifyHolderProofSignature({ address: signer.address, message: `${message}\n`, signature }),
    true,
  );
});

test("holder balance queries sum standard SPL token accounts for the deployed mint", async () => {
  const holder = await generateKeyPairSigner();
  const mint = await generateKeyPairSigner();
  const fetchStub = createRpcFetchStub({
    ownerAddress: holder.address,
    mintAddress: mint.address,
    tokenAmounts: ["1000000000", "250000000"],
  });

  const result = await fetchHolderTokenBalance({
    rpcUrl: "https://api.devnet.solana.com",
    ownerAddress: holder.address,
    mintAddress: mint.address,
    fetchImpl: fetchStub,
  });

  assert.equal(result.accountCount, 2);
  assert.equal(result.baseUnits, 1_250_000_000n);
  assert.equal(result.decimals, PILOT_TOKEN_DECIMALS);
  assert.equal(formatTokenAmount(result.baseUnits, result.decimals), "1.25");
});

test("holder verification requires a valid signature, matching origin, fresh challenge, and positive balance", async () => {
  const holder = await generateKeyPairSigner();
  const mint = await generateKeyPairSigner();
  const manifest = createManifest(mint.address);
  const proofIssuedAt = "2026-09-23T21:15:00.000Z";
  const proofMessage = buildHolderUnlockChallenge({
    address: holder.address,
    mintAddress: mint.address,
    origin: "https://omega.example/novel",
    nonce: "nonce-abcdefgh",
    issuedAt: proofIssuedAt,
  });
  const proofSignature = getBase58Decoder().decode(
    await signBytes(holder.keyPair.privateKey, new TextEncoder().encode(proofMessage)),
  );

  const verified = await verifyHolderUnlock({
    rpcUrl: "https://api.devnet.solana.com",
    manifest,
    proof: {
      address: holder.address,
      message: proofMessage,
      signature: proofSignature,
    },
    expectedOrigin: "https://omega.example/novel",
    maxAgeMs: 15 * 60_000,
    now: Date.parse("2026-09-23T21:20:00.000Z"),
    fetchImpl: createRpcFetchStub({
      ownerAddress: holder.address,
      mintAddress: mint.address,
      tokenAmounts: ["1000000000000000000"],
    }),
  });

  assert.equal(verified.status, "verified");
  assert.equal(verified.address, holder.address);
  assert.equal(verified.balanceBaseUnits, "1000000000000000000");
  assert.equal(verified.balanceTokens, "1000000000");
  assert.equal(verified.unlockedChapters, FULL_NOVEL_UNLOCK_CHAPTERS);
  assert.equal(verified.fullNovelUnlock, true);

  await assert.rejects(
    () =>
      verifyHolderUnlock({
        rpcUrl: "https://api.devnet.solana.com",
        manifest,
        proof: {
          address: holder.address,
          message: proofMessage,
          signature: proofSignature,
        },
        expectedOrigin: "https://different.example/novel",
        fetchImpl: createRpcFetchStub({
          ownerAddress: holder.address,
          mintAddress: mint.address,
          tokenAmounts: ["1000000000"],
        }),
      }),
    /expected origin/,
  );

  await assert.rejects(
    () =>
      verifyHolderUnlock({
        rpcUrl: "https://api.devnet.solana.com",
        manifest,
        proof: {
          address: holder.address,
          message: proofMessage,
          signature: proofSignature,
        },
        expectedOrigin: "https://omega.example/novel",
        maxAgeMs: 60_000,
        now: Date.parse("2026-09-23T21:17:00.000Z") + 120_000,
        fetchImpl: createRpcFetchStub({
          ownerAddress: holder.address,
          mintAddress: mint.address,
          tokenAmounts: ["1000000000"],
        }),
      }),
    /older than the allowed/,
  );

  await assert.rejects(
    () =>
      verifyHolderUnlock({
        rpcUrl: "https://api.devnet.solana.com",
        manifest,
        proof: {
          address: holder.address,
          message: proofMessage,
          signature: proofSignature,
        },
        expectedOrigin: "https://omega.example/novel",
        fetchImpl: createRpcFetchStub({
          ownerAddress: holder.address,
          mintAddress: mint.address,
          tokenAmounts: [],
        }),
      }),
    /does not currently hold any/,
  );

  const stranger = await generateKeyPairSigner();
  const strangerSignature = getBase58Decoder().decode(
    await signBytes(stranger.keyPair.privateKey, new TextEncoder().encode(proofMessage)),
  );
  await assert.rejects(
    () =>
      verifyHolderUnlock({
        rpcUrl: "https://api.devnet.solana.com",
        manifest,
        proof: {
          address: holder.address,
          message: proofMessage,
          signature: strangerSignature,
        },
        expectedOrigin: "https://omega.example/novel",
        fetchImpl: createRpcFetchStub({
          ownerAddress: holder.address,
          mintAddress: mint.address,
          tokenAmounts: ["1000000000"],
        }),
      }),
    /signature verification failed/,
  );
});
