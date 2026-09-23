import {
  appendTransactionMessageInstructions,
  createSolanaRpc,
  createSolanaRpcSubscriptions,
  createTransactionMessage,
  generateKeyPairSigner,
  getSignatureFromTransaction,
  pipe,
  sendAndConfirmTransactionFactory,
  setTransactionMessageFeePayerSigner,
  setTransactionMessageLifetimeUsingBlockhash,
  signTransactionMessageWithSigners,
} from "@solana/kit";
import { getCreateAccountInstruction } from "@solana-program/system";
import {
  AuthorityType,
  TOKEN_PROGRAM_ADDRESS,
  fetchMint,
  fetchToken,
  findAssociatedTokenPda,
  getInitializeMintInstruction,
  getMintSize,
  getSetAuthorityInstruction,
} from "@solana-program/token";
import {
  MPL_TOKEN_METADATA_PROGRAM_ADDRESS,
  TokenStandard,
  fetchMetadata,
  findMetadataPda,
  getCreateV1InstructionAsync,
  getMintV1InstructionAsync,
} from "@metaplex-foundation/mpl-token-metadata-kit";

import {
  DEVNET_GENESIS_HASH,
  MIN_DEPLOYER_BALANCE_LAMPORTS,
  PILOT_TOKEN_DECIMALS,
  PILOT_TOKEN_NAME,
  PILOT_TOKEN_SYMBOL,
} from "./constants.mjs";
import { fetchMetadataCommitment } from "./metadata.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function isNone(option) {
  return option?.__option === "None";
}

function isSome(option, expectedValue) {
  return option?.__option === "Some" && (expectedValue === undefined || option.value === expectedValue);
}

export function createDevnetClients({ rpcUrl, wsUrl }) {
  return {
    rpc: createSolanaRpc(rpcUrl),
    rpcSubscriptions: wsUrl ? createSolanaRpcSubscriptions(wsUrl) : undefined,
  };
}

export async function assertDevnetGenesis(rpc) {
  const genesisHash = await rpc.getGenesisHash().send();
  assert(
    genesisHash === DEVNET_GENESIS_HASH,
    `Refusing to continue: RPC genesis hash ${genesisHash} is not the Solana Devnet genesis hash ${DEVNET_GENESIS_HASH}.`,
  );
  return genesisHash;
}

export async function assertPayerBalance(rpc, payerAddress) {
  const { value: lamports } = await rpc.getBalance(payerAddress, { commitment: "confirmed" }).send();
  assert(
    lamports >= MIN_DEPLOYER_BALANCE_LAMPORTS,
    `Deployer ${payerAddress} has ${lamports} lamports; at least ${MIN_DEPLOYER_BALANCE_LAMPORTS} Devnet lamports are required before deployment.`,
  );
  return lamports;
}

export async function buildPilotInstructions({ payer, mint, config, mintRentLamports }) {
  const mintAddress = mint.address;
  const [metadataAddress] = await findMetadataPda({ mint: mintAddress });
  const [treasuryAtaAddress] = await findAssociatedTokenPda({
    owner: config.treasuryAddress,
    mint: mintAddress,
    tokenProgram: TOKEN_PROGRAM_ADDRESS,
  });

  // Build the mint manually so the SPL Token Program initializes its freeze
  // authority as None. CreateV1 then attaches standard Metaplex fungible
  // metadata to this already-initialized standard SPL mint.
  const createMintInstruction = getCreateAccountInstruction({
    payer,
    newAccount: mint,
    lamports: mintRentLamports,
    space: BigInt(getMintSize()),
    programAddress: TOKEN_PROGRAM_ADDRESS,
  });
  const initializeMintInstruction = getInitializeMintInstruction({
    mint: mintAddress,
    decimals: PILOT_TOKEN_DECIMALS,
    mintAuthority: payer.address,
    freezeAuthority: null,
  });
  const createMetadataInstruction = await getCreateV1InstructionAsync({
    mint: mintAddress,
    authority: payer,
    payer,
    updateAuthority: payer,
    name: PILOT_TOKEN_NAME,
    symbol: PILOT_TOKEN_SYMBOL,
    uri: config.metadataUri,
    sellerFeeBasisPoints: 0,
    isMutable: false,
    tokenStandard: TokenStandard.Fungible,
    decimals: PILOT_TOKEN_DECIMALS,
    splTokenProgram: TOKEN_PROGRAM_ADDRESS,
  });
  const mintSupplyInstruction = await getMintV1InstructionAsync({
    mint: mintAddress,
    authority: payer,
    payer,
    tokenOwner: config.treasuryAddress,
    amount: config.initialSupplyBaseUnits,
    tokenStandard: TokenStandard.Fungible,
    splTokenProgram: TOKEN_PROGRAM_ADDRESS,
  });
  const revokeMintAuthorityInstruction = getSetAuthorityInstruction({
    owned: mintAddress,
    owner: payer,
    authorityType: AuthorityType.MintTokens,
    newAuthority: null,
  });

  return Object.freeze({
    mintAddress,
    metadataAddress,
    treasuryAtaAddress,
    instructions: [
      createMintInstruction,
      initializeMintInstruction,
      createMetadataInstruction,
      mintSupplyInstruction,
      revokeMintAuthorityInstruction,
    ],
  });
}

export async function buildPilotTransaction({ rpc, payer, config }) {
  const mint = await generateKeyPairSigner();
  const mintRentLamports = await rpc
    .getMinimumBalanceForRentExemption(BigInt(getMintSize()), { commitment: "confirmed" })
    .send();
  const plan = await buildPilotInstructions({ payer, mint, config, mintRentLamports });
  const { value: latestBlockhash } = await rpc.getLatestBlockhash({ commitment: "confirmed" }).send();

  const transactionMessage = pipe(
    createTransactionMessage({ version: 0 }),
    (message) => setTransactionMessageFeePayerSigner(payer, message),
    (message) => setTransactionMessageLifetimeUsingBlockhash(latestBlockhash, message),
    (message) => appendTransactionMessageInstructions(plan.instructions, message),
  );
  const signedTransaction = await signTransactionMessageWithSigners(transactionMessage);

  return Object.freeze({
    ...plan,
    mint,
    signedTransaction,
    transactionSignature: getSignatureFromTransaction(signedTransaction),
  });
}

export async function sendPilotTransaction({ rpc, rpcSubscriptions, signedTransaction }) {
  if (!rpcSubscriptions) {
    throw new Error("A Devnet WS(S) endpoint is required to send and confirm the deployment transaction.");
  }
  await sendAndConfirmTransactionFactory({ rpc, rpcSubscriptions })(signedTransaction, {
    commitment: "confirmed",
  });
}

export async function verifyPilotState({ rpc, manifest, fetchImpl = globalThis.fetch }) {
  await assertDevnetGenesis(rpc);

  const mintAddress = manifest.mint.address;
  const mint = await fetchMint(rpc, mintAddress, { commitment: "confirmed" });
  assert(mint.owner === TOKEN_PROGRAM_ADDRESS, `Mint owner is ${mint.owner}, not the standard SPL Token Program.`);
  assert(mint.data.decimals === PILOT_TOKEN_DECIMALS, `Mint decimals are ${mint.data.decimals}, expected ${PILOT_TOKEN_DECIMALS}.`);
  assert(mint.data.supply === BigInt(manifest.mint.initialSupplyBaseUnits), "Mint supply does not match the manifest.");
  assert(isNone(mint.data.mintAuthority), "Mint authority is not permanently disabled.");
  assert(isNone(mint.data.freezeAuthority), "Freeze authority is not permanently disabled.");

  const [expectedTreasuryAta] = await findAssociatedTokenPda({
    owner: manifest.treasury.address,
    mint: mintAddress,
    tokenProgram: TOKEN_PROGRAM_ADDRESS,
  });
  assert(expectedTreasuryAta === manifest.treasury.associatedTokenAccount, "Manifest treasury ATA does not match the canonical ATA derivation.");
  const treasuryToken = await fetchToken(rpc, expectedTreasuryAta, { commitment: "confirmed" });
  assert(treasuryToken.owner === TOKEN_PROGRAM_ADDRESS, "Treasury token account is not owned by the standard SPL Token Program.");
  assert(treasuryToken.data.mint === mintAddress, "Treasury ATA references a different mint.");
  assert(treasuryToken.data.owner === manifest.treasury.address, "Treasury ATA owner differs from the manifest treasury.");
  assert(treasuryToken.data.amount === BigInt(manifest.mint.initialSupplyBaseUnits), "Treasury ATA does not hold the complete initial supply.");

  const [expectedMetadataAddress] = await findMetadataPda({ mint: mintAddress });
  assert(expectedMetadataAddress === manifest.metadata.address, "Manifest metadata address does not match the canonical Metaplex PDA.");
  const metadata = await fetchMetadata(rpc, expectedMetadataAddress, { commitment: "confirmed" });
  assert(metadata.owner === MPL_TOKEN_METADATA_PROGRAM_ADDRESS, "Metadata account is not owned by the Metaplex Token Metadata Program.");
  assert(metadata.data.mint === mintAddress, "Metadata references a different mint.");
  assert(metadata.data.name === PILOT_TOKEN_NAME, "Metadata name differs from the pilot identity.");
  assert(metadata.data.symbol === PILOT_TOKEN_SYMBOL, "Metadata symbol differs from the pilot identity.");
  assert(metadata.data.uri === manifest.metadata.uri, "Metadata URI differs from the manifest.");
  assert(metadata.data.sellerFeeBasisPoints === 0, "Metadata seller fee basis points must be zero.");
  assert(metadata.data.isMutable === false, "Metadata is mutable.");
  assert(isSome(metadata.data.tokenStandard, TokenStandard.Fungible), "Metadata token standard is not Fungible.");
  assert(metadata.data.updateAuthority === manifest.metadata.updateAuthority, "Metadata update authority differs from the manifest.");

  const offChainMetadata = await fetchMetadataCommitment(
    manifest.metadata.uri,
    manifest.metadata.sha256,
    fetchImpl,
  );

  return Object.freeze({
    mintAddress,
    metadataAddress: expectedMetadataAddress,
    treasuryAtaAddress: expectedTreasuryAta,
    supplyBaseUnits: mint.data.supply.toString(),
    metadataSha256: offChainMetadata.sha256,
    metadataByteLength: offChainMetadata.byteLength,
  });
}
