import assert from "node:assert/strict";
import test from "node:test";

import {
  TRANSACTION_SIZE_LIMIT,
  appendTransactionMessageInstructions,
  createTransactionMessage,
  generateKeyPairSigner,
  getTransactionSize,
  pipe,
  setTransactionMessageFeePayerSigner,
  setTransactionMessageLifetimeUsingBlockhash,
  signTransactionMessageWithSigners,
} from "@solana/kit";
import {
  AuthorityType,
  getInitializeMintInstructionDataDecoder,
  getSetAuthorityInstructionDataDecoder,
} from "@solana-program/token";
import {
  TokenStandard,
  getCreateV1InstructionDataDecoder,
  getMintV1InstructionDataDecoder,
} from "@metaplex-foundation/mpl-token-metadata-kit";

import { buildPilotInstructions } from "../lib/chain.mjs";
import { PILOT_TOKEN_DECIMALS } from "../lib/constants.mjs";

test("tTWC issuance plan is atomic, fixed-supply, freeze-free, and packet-bounded", async () => {
  const payer = await generateKeyPairSigner();
  const mint = await generateKeyPairSigner();
  const treasury = (await generateKeyPairSigner()).address;
  const config = {
    treasuryAddress: treasury,
    metadataUri: "https://arweave.net/metadata-transaction",
    initialSupplyBaseUnits: 1_000_000_000_000_000_000n,
  };

  const plan = await buildPilotInstructions({
    payer,
    mint,
    config,
    mintRentLamports: 1_461_600n,
  });

  assert.equal(plan.instructions.length, 5);
  const initializeMintData = getInitializeMintInstructionDataDecoder().decode(plan.instructions[1].data);
  assert.equal(initializeMintData.decimals, PILOT_TOKEN_DECIMALS);
  assert.equal(initializeMintData.mintAuthority, payer.address);
  assert.equal(initializeMintData.freezeAuthority.__option, "None");

  const metadataData = getCreateV1InstructionDataDecoder().decode(plan.instructions[2].data);
  assert.equal(metadataData.name, "Token of the World Citizen");
  assert.equal(metadataData.symbol, "tTWC");
  assert.equal(metadataData.tokenStandard, TokenStandard.Fungible);
  assert.equal(metadataData.isMutable, false);
  assert.equal(metadataData.sellerFeeBasisPoints, 0);

  const mintData = getMintV1InstructionDataDecoder().decode(plan.instructions[3].data);
  assert.equal(mintData.amount, config.initialSupplyBaseUnits);

  const revokeData = getSetAuthorityInstructionDataDecoder().decode(plan.instructions[4].data);
  assert.equal(revokeData.authorityType, AuthorityType.MintTokens);
  assert.equal(revokeData.newAuthority.__option, "None");

  const message = pipe(
    createTransactionMessage({ version: 0 }),
    (transaction) => setTransactionMessageFeePayerSigner(payer, transaction),
    (transaction) => setTransactionMessageLifetimeUsingBlockhash({
      blockhash: "11111111111111111111111111111111",
      lastValidBlockHeight: 1n,
    }, transaction),
    (transaction) => appendTransactionMessageInstructions(plan.instructions, transaction),
  );
  const signedTransaction = await signTransactionMessageWithSigners(message);
  assert.ok(getTransactionSize(signedTransaction) <= TRANSACTION_SIZE_LIMIT);
});
