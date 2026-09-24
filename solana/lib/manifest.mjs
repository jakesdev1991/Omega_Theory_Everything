import { access, mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import {
  CANONICAL_TOKEN_NAME,
  CANONICAL_TOKEN_SYMBOL,
  DEVNET_CLUSTER,
  DEVNET_GENESIS_HASH,
  MANIFEST_VERSION,
  PILOT_TOKEN_DECIMALS,
  PILOT_TOKEN_NAME,
  PILOT_TOKEN_SYMBOL,
  STANDARD_TOKEN_PROGRAM_ADDRESS,
} from "./constants.mjs";

function assert(condition, message) {
  if (!condition) throw new Error(`Manifest error: ${message}`);
}

export function createManifest({
  deploymentConfig,
  payerAddress,
  mintAddress,
  metadataAddress,
  treasuryAtaAddress,
  transactionSignature,
  metadataByteLength,
}) {
  return {
    manifestVersion: MANIFEST_VERSION,
    generatedAt: new Date().toISOString(),
    pilot: {
      network: DEVNET_CLUSTER,
      genesisHash: DEVNET_GENESIS_HASH,
      valueless: true,
      purpose: "Test-only SPL Token / Metaplex metadata deployment",
    },
    identity: {
      canonicalName: CANONICAL_TOKEN_NAME,
      canonicalSymbol: CANONICAL_TOKEN_SYMBOL,
      onChainName: PILOT_TOKEN_NAME,
      onChainSymbol: PILOT_TOKEN_SYMBOL,
    },
    mint: {
      address: mintAddress,
      decimals: PILOT_TOKEN_DECIMALS,
      initialSupplyTokens: deploymentConfig.initialSupplyTokens,
      initialSupplyBaseUnits: deploymentConfig.initialSupplyBaseUnits,
      tokenProgram: STANDARD_TOKEN_PROGRAM_ADDRESS,
      mintAuthority: null,
      freezeAuthority: null,
    },
    treasury: {
      address: deploymentConfig.treasuryAddress,
      associatedTokenAccount: treasuryAtaAddress,
    },
    metadata: {
      address: metadataAddress,
      uri: deploymentConfig.metadataUri,
      sha256: deploymentConfig.metadataSha256,
      byteLength: metadataByteLength,
      isMutable: false,
      sellerFeeBasisPoints: 0,
      updateAuthority: payerAddress,
    },
    deployment: {
      payer: payerAddress,
      transactionSignature,
      rpcUrl: deploymentConfig.rpcUrl,
    },
  };
}

export function assertValidManifest(manifest) {
  assert(manifest && typeof manifest === "object", "root must be an object.");
  assert(manifest.manifestVersion === MANIFEST_VERSION, `unsupported manifestVersion (expected ${MANIFEST_VERSION}).`);
  assert(manifest.pilot?.network === DEVNET_CLUSTER, "network must be devnet.");
  assert(manifest.pilot?.genesisHash === DEVNET_GENESIS_HASH, "genesis hash must be Solana Devnet.");
  assert(manifest.pilot?.valueless === true, "pilot must be marked valueless.");
  assert(manifest.identity?.onChainName === PILOT_TOKEN_NAME, "unexpected on-chain name.");
  assert(manifest.identity?.onChainSymbol === PILOT_TOKEN_SYMBOL, "unexpected on-chain symbol.");
  assert(typeof manifest.mint?.address === "string", "mint address is required.");
  assert(manifest.mint?.decimals === PILOT_TOKEN_DECIMALS, "unexpected decimals.");
  assert(typeof manifest.mint?.initialSupplyBaseUnits === "string", "initialSupplyBaseUnits is required.");
  assert(manifest.mint?.mintAuthority === null, "mintAuthority must be null in manifest.");
  assert(manifest.mint?.freezeAuthority === null, "freezeAuthority must be null in manifest.");
  assert(typeof manifest.treasury?.address === "string", "treasury address is required.");
  assert(typeof manifest.treasury?.associatedTokenAccount === "string", "treasury ATA is required.");
  assert(typeof manifest.metadata?.address === "string", "metadata address is required.");
  assert(typeof manifest.metadata?.uri === "string", "metadata URI is required.");
  assert(/^[a-f0-9]{64}$/.test(manifest.metadata?.sha256 ?? ""), "metadata SHA-256 is required.");
  assert(manifest.metadata?.isMutable === false, "metadata must be immutable.");
  assert(manifest.metadata?.sellerFeeBasisPoints === 0, "seller fee basis points must be zero.");
  assert(typeof manifest.deployment?.transactionSignature === "string", "deployment signature is required.");
  return manifest;
}

export async function readManifest(manifestPath) {
  let manifest;
  try {
    manifest = JSON.parse(await readFile(manifestPath, "utf8"));
  } catch (error) {
    throw new Error(`Cannot read deployment manifest at ${manifestPath}: ${error.message}`);
  }
  return assertValidManifest(manifest);
}

export async function writeManifestNew(manifestPath, manifest) {
  assertValidManifest(manifest);
  try {
    await access(manifestPath);
    throw new Error(`Refusing to overwrite existing manifest: ${manifestPath}`);
  } catch (error) {
    if (error?.code !== "ENOENT") throw error;
  }

  await mkdir(dirname(manifestPath), { recursive: true });
  const temporaryPath = `${manifestPath}.${process.pid}.${Date.now()}.tmp`;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(manifest, null, 2)}\n`, { mode: 0o600 });
    await rename(temporaryPath, manifestPath);
  } catch (error) {
    throw new Error(`Could not write deployment manifest: ${error.message}`);
  }
}
