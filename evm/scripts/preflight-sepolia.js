// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
/* eslint-disable no-console */

const hre = require("hardhat");

const { ethers, network } = hre;
const SEPOLIA_CHAIN_ID = 11155111n;
const NINETY_DAYS = 90 * 24 * 60 * 60;

function required(name) {
  const value = process.env[name];
  if (!value || !value.trim()) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value.trim();
}

function integer(name, fallback, { min = 0, max = Number.MAX_SAFE_INTEGER } = {}) {
  const raw = process.env[name] ?? String(fallback);
  if (!/^\d+$/.test(raw)) {
    throw new Error(`${name} must be a whole-number string; received ${JSON.stringify(raw)}.`);
  }
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new Error(`${name} must be an integer between ${min} and ${max}; received ${raw}.`);
  }
  return value;
}

function tokenAmount(name, fallback) {
  const raw = process.env[name] ?? fallback;
  if (!/^\d+(\.\d+)?$/.test(raw)) {
    throw new Error(`${name} must be a decimal token amount; received ${JSON.stringify(raw)}.`);
  }
  return ethers.parseUnits(raw, 18);
}

function address(name) {
  const value = required(name);
  if (!ethers.isAddress(value) || value === ethers.ZeroAddress) {
    throw new Error(`${name} must be a non-zero EVM address; received ${JSON.stringify(value)}.`);
  }
  return ethers.getAddress(value);
}

async function main() {
  if (!process.env.DEPLOYER_PRIVATE_KEY) {
    throw new Error(
      "DEPLOYER_PRIVATE_KEY is required. Use a newly-created Sepolia-only wallet; never use a mainnet key.",
    );
  }

  const chainId = BigInt(await network.provider.send("eth_chainId"));
  if (chainId !== SEPOLIA_CHAIN_ID) {
    throw new Error(
      `Refusing to continue: expected Sepolia chain ID ${SEPOLIA_CHAIN_ID}, received ${chainId}.`,
    );
  }

  const [deployer] = await ethers.getSigners();
  const treasury = address("TREASURY_ADDRESS");
  const guardian = address("GUARDIAN_ADDRESS");

  const initialSupply = tokenAmount("OMEGA_INITIAL_SUPPLY", "1000000000");
  const claimThreshold = tokenAmount("CLAIM_THRESHOLD_TOKENS", "1");
  const timelockDelay = integer("TIMELOCK_DELAY_SECONDS", 86400, { min: 60 });
  const votingDelay = integer("VOTING_DELAY_BLOCKS", 1, { min: 1 });
  const votingPeriod = integer("VOTING_PERIOD_BLOCKS", 7200, { min: 1 });
  const proposalThreshold = tokenAmount("PROPOSAL_THRESHOLD_TOKENS", "1000");
  const quorumPercent = integer("QUORUM_PERCENT", 4, { min: 1, max: 100 });

  const latestBlock = await ethers.provider.getBlock("latest");
  const defaultClaimStart = Number(latestBlock.timestamp);
  const claimStart = integer("CLAIM_START_UNIX", defaultClaimStart, { min: defaultClaimStart });
  const claimEnd = integer("CLAIM_END_UNIX", claimStart + NINETY_DAYS, { min: claimStart + 1 });

  if (proposalThreshold > initialSupply) {
    throw new Error("PROPOSAL_THRESHOLD_TOKENS cannot exceed OMEGA_INITIAL_SUPPLY.");
  }
  if (claimThreshold > initialSupply) {
    throw new Error("CLAIM_THRESHOLD_TOKENS cannot exceed OMEGA_INITIAL_SUPPLY.");
  }

  console.log("\n$OMEGA Sepolia preflight passed");
  console.log(`  deployer              ${deployer.address}`);
  console.log(`  treasury              ${treasury}`);
  console.log(`  guardian              ${guardian}`);
  console.log(`  initial supply        ${ethers.formatUnits(initialSupply, 18)} tOMEGA`);
  console.log(`  claim threshold       ${ethers.formatUnits(claimThreshold, 18)} tOMEGA`);
  console.log(`  claim window          ${claimStart} → ${claimEnd}`);
  console.log(`  governance            ${votingDelay} block delay, ${votingPeriod} block period, ${quorumPercent}% quorum`);
  console.log(`  timelock delay        ${timelockDelay} seconds`);
  console.log("\nNo transactions were sent.");
  console.log("Next step: run npm run deploy:sepolia only after an independent review of the parameters above.");
}

main().catch((error) => {
  console.error(`Preflight stopped: ${error.message}`);
  process.exitCode = 1;
});
