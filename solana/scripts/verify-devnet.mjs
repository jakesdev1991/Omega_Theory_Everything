import { existsSync } from "node:fs";

import { DEFAULT_MANIFEST_PATH, devnetAddressExplorerUrl } from "../lib/constants.mjs";
import { createDevnetClients, verifyPilotState } from "../lib/chain.mjs";
import { readVerificationConfig } from "../lib/config.mjs";
import { readManifest } from "../lib/manifest.mjs";

const manifestPath = process.argv[2] ?? DEFAULT_MANIFEST_PATH;

try {
  if (!existsSync(manifestPath)) {
    throw new Error(`No local deployment manifest found at ${manifestPath}. This repository does not claim a deployed tTWC pilot.`);
  }

  const manifest = await readManifest(manifestPath);
  const { rpc } = createDevnetClients(readVerificationConfig());
  const result = await verifyPilotState({ rpc, manifest });

  console.log(JSON.stringify({
    status: "verified",
    mint: result.mintAddress,
    mintExplorer: devnetAddressExplorerUrl(result.mintAddress),
    metadata: result.metadataAddress,
    treasuryAta: result.treasuryAtaAddress,
    supplyBaseUnits: result.supplyBaseUnits,
    metadataSha256: result.metadataSha256,
    metadataBytes: result.metadataByteLength,
  }, null, 2));
} catch (error) {
  console.error(`Verification failed: ${error.message}`);
  process.exitCode = 1;
}
