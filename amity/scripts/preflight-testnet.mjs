import { amityEnvExists, inspectOperatorFiles, readAmityConfig } from "../lib/config.mjs";

function status(ok) {
  return ok ? "OK" : "MISSING";
}

try {
  const envPresent = await amityEnvExists();
  const config = await readAmityConfig();
  const files = await inspectOperatorFiles(config);

  console.log("\nAMITY testnet preflight");
  console.log(`  local .env            ${status(envPresent)}${envPresent ? "" : " (using process env or defaults where available)"}`);
  console.log(`  network               ${config.network}`);
  console.log(`  asset id              ${config.assetId}`);
  console.log(`  universe              ${config.universeUrl}`);
  console.log(`  litd rpc              ${config.litdRpcHost}`);
  console.log(`  tapd rpc              ${config.tapdRpcHost}`);
  console.log(`  asset name            ${config.assetName}`);
  console.log(`  unit name             ${config.unitName}`);
  console.log("");
  console.log("Operator file checks");
  for (const file of files) {
    const state = file.configured ? (file.exists ? "present" : "configured but missing") : "not configured yet";
    console.log(`  ${file.label.padEnd(20)} ${state}${file.filePath ? ` (${file.filePath})` : ""}`);
  }
  console.log("\nNo network calls were made and no asset was issued.");
  console.log("Next steps: stand up litd/tapd testnet infrastructure, issue a valueless AMITY test asset, and publish a Universe endpoint before building holder verification.");
} catch (error) {
  console.error(`Preflight stopped: ${error.message}`);
  process.exitCode = 1;
}
