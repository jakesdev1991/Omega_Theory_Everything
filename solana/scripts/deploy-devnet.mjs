import { existsSync } from "node:fs";

import {
  DEFAULT_MANIFEST_PATH,
  DEPLOYMENT_CONFIRMATION_VALUE,
  devnetAddressExplorerUrl,
  devnetTransactionExplorerUrl,
} from "../lib/constants.mjs";
import {
  deploymentConfigForManifest,
  loadKeypairSigner,
  readDeploymentConfig,
} from "../lib/config.mjs";
import {
  assertDevnetGenesis,
  assertPayerBalance,
  buildPilotTransaction,
  createDevnetClients,
  sendPilotTransaction,
  verifyPilotState,
} from "../lib/chain.mjs";
import { fetchMetadataCommitment } from "../lib/metadata.mjs";
import { createManifest, writeManifestNew } from "../lib/manifest.mjs";

const confirmationFlag = process.argv.includes("--confirm-devnet");

try {
  if (existsSync(DEFAULT_MANIFEST_PATH)) {
    throw new Error(`Refusing to overwrite an existing deployment manifest: ${DEFAULT_MANIFEST_PATH}`);
  }

  const config = readDeploymentConfig();
  const payer = await loadKeypairSigner(config.deployerKeypairPath);
  const { rpc, rpcSubscriptions } = createDevnetClients(config);

  await assertDevnetGenesis(rpc);
  const payerLamports = await assertPayerBalance(rpc, payer.address);
  const offChainMetadata = await fetchMetadataCommitment(config.metadataUri, config.metadataSha256);

  console.log(JSON.stringify({
    action: "Solana Devnet tTWC pilot preflight",
    payer: payer.address,
    payerLamports: payerLamports.toString(),
    treasury: config.treasuryAddress,
    supplyTokens: config.initialSupplyTokens.toString(),
    supplyBaseUnits: config.initialSupplyBaseUnits.toString(),
    metadataUri: config.metadataUri,
    metadataSha256: offChainMetadata.sha256,
    metadataBytes: offChainMetadata.byteLength,
    manifestPath: DEFAULT_MANIFEST_PATH,
  }, null, 2));

  if (!confirmationFlag || process.env.TWC_DEPLOY_CONFIRM !== DEPLOYMENT_CONFIRMATION_VALUE) {
    throw new Error(
      `Preflight passed but no deployment was sent. To send the irreversible Devnet pilot transaction, pass --confirm-devnet and set TWC_DEPLOY_CONFIRM=${DEPLOYMENT_CONFIRMATION_VALUE}.`,
    );
  }

  const transaction = await buildPilotTransaction({ rpc, payer, config });
  console.log(`Prepared atomic Devnet deployment for mint ${transaction.mintAddress}.`);
  console.log(`Prepared transaction: ${devnetTransactionExplorerUrl(transaction.transactionSignature)}`);

  await sendPilotTransaction({
    rpc,
    rpcSubscriptions,
    signedTransaction: transaction.signedTransaction,
  });

  const manifest = createManifest({
    deploymentConfig: deploymentConfigForManifest(config),
    payerAddress: payer.address,
    mintAddress: transaction.mintAddress,
    metadataAddress: transaction.metadataAddress,
    treasuryAtaAddress: transaction.treasuryAtaAddress,
    transactionSignature: transaction.transactionSignature,
    metadataByteLength: offChainMetadata.byteLength,
  });

  const verification = await verifyPilotState({ rpc, manifest });
  await writeManifestNew(DEFAULT_MANIFEST_PATH, manifest);

  console.log("Devnet-only tTWC pilot deployed and independently verified.");
  console.log(`Mint: ${devnetAddressExplorerUrl(verification.mintAddress)}`);
  console.log(`Metadata: ${devnetAddressExplorerUrl(verification.metadataAddress)}`);
  console.log(`Treasury ATA: ${devnetAddressExplorerUrl(verification.treasuryAtaAddress)}`);
  console.log(`Manifest (ignored): ${DEFAULT_MANIFEST_PATH}`);
} catch (error) {
  console.error(`Deployment stopped: ${error.message}`);
  process.exitCode = 1;
}
