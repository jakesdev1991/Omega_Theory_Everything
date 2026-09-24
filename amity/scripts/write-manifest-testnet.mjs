#!/usr/bin/env node

import { inspectOperatorFiles, readAmityConfig } from "../lib/config.mjs";
import { DEFAULT_MANIFEST_PATH } from "../lib/constants.mjs";
import { createManifest, writeManifest } from "../lib/manifest.mjs";

const overwrite = process.argv.includes("--force");

try {
  const config = await readAmityConfig();
  const operatorFiles = await inspectOperatorFiles(config);
  const manifest = createManifest({ config, operatorFiles });
  await writeManifest(DEFAULT_MANIFEST_PATH, manifest, { overwrite });

  const presentCount = manifest.operatorFiles.filter((entry) => entry.exists).length;

  console.log(`Wrote ${DEFAULT_MANIFEST_PATH}`);
  console.log(`  network               ${manifest.pilot.network}`);
  console.log(`  asset id              ${manifest.asset.assetId}`);
  console.log(`  universe              ${manifest.asset.universeUrl}`);
  console.log(`  litd rpc              ${manifest.services.litd.rpcHost}`);
  console.log(`  tapd rpc              ${manifest.services.tapd.rpcHost}`);
  console.log(`  operator files        ${presentCount}/${manifest.operatorFiles.length} present`);
  console.log("  unlock wiring         false (the live web unlock flow still only wires $OMEGA and TWC)");
  console.log("\nNo network calls were made and no asset was issued.");
  console.log("This manifest is only a local status scaffold for future testnet node/proof integration.");
} catch (error) {
  console.error(`Manifest write stopped: ${error.message}`);
  process.exitCode = 1;
}
