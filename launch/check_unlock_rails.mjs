#!/usr/bin/env node

import { readFile, access } from "node:fs/promises";
import { constants as fsConstants } from "node:fs";
import { join, resolve } from "node:path";

const ROOT = process.cwd();
const EVM_MANIFEST = resolve(ROOT, "evm/deployments/sepolia.json");
const SOLANA_MANIFEST = resolve(ROOT, "solana/deployments/twc-devnet.json");
const AMITY_MANIFEST = resolve(ROOT, "amity/deployments/testnet.json");
const WEB_ENV_LOCAL = resolve(ROOT, "web/.env.local");

function green(value) {
  return `\x1b[32m${value}\x1b[0m`;
}

function yellow(value) {
  return `\x1b[33m${value}\x1b[0m`;
}

function cyan(value) {
  return `\x1b[36m${value}\x1b[0m`;
}

function dim(value) {
  return `\x1b[2m${value}\x1b[0m`;
}

async function exists(path) {
  try {
    await access(path, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

async function readJsonIfExists(path) {
  if (!(await exists(path))) return null;
  return JSON.parse(await readFile(path, "utf8"));
}

async function readEnvFile(path) {
  if (!(await exists(path))) return {};
  const text = await readFile(path, "utf8");
  const entries = {};
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const index = line.indexOf("=");
    if (index === -1) continue;
    const key = line.slice(0, index).trim();
    let value = line.slice(index + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    entries[key] = value;
  }
  return entries;
}

function line(label, value, ok = true) {
  const prefix = ok ? green("✓") : yellow("!");
  return `${prefix} ${label.padEnd(20)} ${value}`;
}

async function main() {
  const [evmManifest, solanaManifest, amityManifest, webEnv] = await Promise.all([
    readJsonIfExists(EVM_MANIFEST),
    readJsonIfExists(SOLANA_MANIFEST),
    readJsonIfExists(AMITY_MANIFEST),
    readEnvFile(WEB_ENV_LOCAL),
  ]);

  const omegaManifestReady = !!evmManifest?.contracts?.omegaNovelGate;
  const twcManifestReady = !!solanaManifest?.mint?.address;
  const amityManifestReady = !!amityManifest?.asset?.assetId;
  const omegaEnvReady = !!webEnv.OMEGA_NOVEL_GATE_ADDRESS;
  const twcEnvReady = !!webEnv.TWC_MINT_ADDRESS;
  const amityEnvReady = !!webEnv.AMITY_ASSET_ID;

  const omegaReady = omegaManifestReady || omegaEnvReady;
  const twcReady = twcManifestReady || twcEnvReady;
  const amityReady = amityManifestReady || amityEnvReady;

  console.log(`\n${cyan("Unlock rail readiness")}`);
  console.log(dim(`root: ${ROOT}`));
  console.log("");

  console.log(cyan("$OMEGA / Sepolia"));
  console.log(line("manifest", omegaManifestReady ? EVM_MANIFEST : "missing", omegaManifestReady));
  if (omegaManifestReady) {
    console.log(line("gate", evmManifest.contracts.omegaNovelGate));
    console.log(line("token", evmManifest.contracts.omegaTestToken));
  } else {
    console.log(line("env fallback", omegaEnvReady ? WEB_ENV_LOCAL : "missing OMEGA_NOVEL_GATE_ADDRESS", omegaEnvReady));
  }
  console.log("");

  console.log(cyan("TWC / Devnet"));
  console.log(line("manifest", twcManifestReady ? SOLANA_MANIFEST : "missing", twcManifestReady));
  if (twcManifestReady) {
    console.log(line("mint", solanaManifest.mint.address));
    console.log(line("rpc", solanaManifest.deployment?.rpcUrl ?? "n/a"));
  } else {
    console.log(line("env fallback", twcEnvReady ? WEB_ENV_LOCAL : "missing TWC_MINT_ADDRESS", twcEnvReady));
  }
  console.log("");

  console.log(cyan("AMITY / Bitcoin Taproot (Testnet Scaffold)"));
  console.log(line("manifest", amityManifestReady ? AMITY_MANIFEST : "missing (scaffold / fixture mode)", amityManifestReady));
  if (amityManifestReady) {
    console.log(line("asset id", amityManifest.asset.assetId));
    console.log(line("universe", amityManifest.asset.universeUrl ?? "n/a"));
  } else {
    console.log(line("fixture mode", amityEnvReady ? "active via env" : "fixture testnet ready", true));
  }
  console.log("");

  console.log(cyan("Web operator env"));
  console.log(line("web/.env.local", Object.keys(webEnv).length ? WEB_ENV_LOCAL : "missing", Object.keys(webEnv).length > 0));
  if (Object.keys(webEnv).length > 0) {
    console.log(line("omega gate env", webEnv.OMEGA_NOVEL_GATE_ADDRESS ?? "not set", !!webEnv.OMEGA_NOVEL_GATE_ADDRESS));
    console.log(line("twc mint env", webEnv.TWC_MINT_ADDRESS ?? "not set", !!webEnv.TWC_MINT_ADDRESS));
    console.log(line("amity asset env", webEnv.AMITY_ASSET_ID ?? "not set", !!webEnv.AMITY_ASSET_ID));
  }
  console.log("");

  const allReady = omegaReady && twcReady;
  console.log(allReady ? green("Both wired rails are locally activatable.") : yellow("At least one wired rail is still missing local activation data."));

  if (!allReady) {
    console.log("\nNext steps:");
    if (!omegaReady) {
      console.log(`  - create ${join("evm", ".env")}, run ${cyan("cd evm && npm run preflight:sepolia && npm run deploy:sepolia")}`);
    }
    if (!twcReady) {
      console.log(`  - create ${join("solana", ".env")}, run ${cyan("cd solana && npm run deploy:devnet")}`);
    }
    console.log(`  - once both manifests exist, optionally generate ${join("web", ".env.local")} with ${cyan("node launch/sync_web_unlock_env.mjs")}`);
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
