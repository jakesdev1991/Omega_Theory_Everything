// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
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

function octetCount(hex) {
  let nonZero = 0n;
  for (let index = 2; index < hex.length; index += 2) {
    if (hex.slice(index, index + 2) !== "00") {
      nonZero += 1n;
    }
  }
  return nonZero;
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

  // Funds. The deploy script creates six contracts and then sends governance
  // setup transactions; a deployer that runs out of ETH part way through leaves
  // a half-deployed system and a lot of manual cleanup. Estimate the creation
  // cost from the compiled artifacts rather than trusting a hand-written number:
  // 32,000 gas base per CREATE, 200 gas per runtime byte deposited, the initcode
  // calldata cost (16 gas per non-zero byte, 4 per zero byte), and EIP-3860's 2
  // gas per 32-byte initcode word. Constructor execution is not counted, so a
  // 1.5x safety factor plus a fixed reserve for the two setup writes covers it.
  const deploymentPlan = [
    "OmegaTestToken",
    "OmegaVotingEscrow",
    "TimelockController",
    "OmegaLocking",
    "OmegaGovernor",
    "OmegaNovelGate",
  ];
  let creationGas = 0n;
  for (const name of deploymentPlan) {
    const artifact = await hre.artifacts.readArtifact(name);
    const initcodeBytes = BigInt((artifact.bytecode.length - 2) / 2);
    const nonZeroBytes = octetCount(artifact.bytecode);
    creationGas +=
      32000n +
      200n * BigInt((artifact.deployedBytecode.length - 2) / 2) +
      16n * nonZeroBytes +
      4n * (initcodeBytes - nonZeroBytes) +
      2n * ((initcodeBytes + 31n) / 32n);
  }
  const setupGasReserve = 200000n; // grant governor proposer + renounce timelock admin

  const feeData = await ethers.provider.getFeeData();
  const gasPrice = feeData.maxFeePerGas ?? feeData.gasPrice;
  const balance = await ethers.provider.getBalance(deployer.address);
  const explicitFloor = process.env.MIN_DEPLOYER_BALANCE_ETH
    ? tokenAmount("MIN_DEPLOYER_BALANCE_ETH", "0")
    : null;

  let requiredWei = explicitFloor;
  let fundsExplanation;
  if (gasPrice === null || gasPrice === undefined) {
    if (explicitFloor === null) {
      throw new Error(
        "Could not read a gas price from the RPC to estimate deployment cost. Set MIN_DEPLOYER_BALANCE_ETH to an explicit floor if this network's fee data is unavailable.",
      );
    }
    fundsExplanation = "explicit MIN_DEPLOYER_BALANCE_ETH floor (no fee data from the RPC)";
  } else {
    const estimated = ((creationGas + setupGasReserve) * gasPrice * 3n) / 2n;
    requiredWei = estimated > (explicitFloor ?? 0n) ? estimated : explicitFloor;
    fundsExplanation = `${ethers.formatEther(estimated)} ETH estimated at ${ethers.formatUnits(gasPrice, "gwei")} gwei`;
  }

  if (balance < requiredWei) {
    throw new Error(
      `Deployer ${deployer.address} has ${ethers.formatEther(balance)} ETH but the deployment needs about ` +
        `${ethers.formatEther(requiredWei)} ETH (${fundsExplanation}). Fund a Sepolia-only wallet from a testnet ` +
        "faucet before deploying; the preflight sends no transactions.",
    );
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
  console.log(
    `  deployer balance      ${ethers.formatEther(balance)} ETH (needs ~${ethers.formatEther(requiredWei)}; ${fundsExplanation})`,
  );
  console.log("\nNo transactions were sent.");
  console.log("Next step: run npm run deploy:sepolia only after an independent review of the parameters above.");
}

main().catch((error) => {
  console.error(`Preflight stopped: ${error.message}`);
  process.exitCode = 1;
});
