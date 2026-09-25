// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
/* eslint-disable no-console */

const fs = require("node:fs");
const path = require("node:path");
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

async function deploy(name, factory, args) {
  const contract = await factory.deploy(...args);
  await contract.waitForDeployment();
  const contractAddress = await contract.getAddress();
  console.log(`  ${name.padEnd(21)} ${contractAddress}`);
  return contract;
}

async function send(label, transactionPromise) {
  const transaction = await transactionPromise;
  const receipt = await transaction.wait();
  console.log(`  ${label.padEnd(21)} ${receipt.hash}`);
  return receipt;
}

function jsonReplacer(_key, value) {
  return typeof value === "bigint" ? value.toString() : value;
}

async function main() {
  if (!process.env.DEPLOYER_PRIVATE_KEY) {
    throw new Error(
      "DEPLOYER_PRIVATE_KEY is required. Use a newly-created Sepolia-only wallet; never use a mainnet key."
    );
  }

  const chainId = BigInt(await network.provider.send("eth_chainId"));
  if (chainId !== SEPOLIA_CHAIN_ID) {
    throw new Error(
      `Refusing to deploy: expected Sepolia chain ID ${SEPOLIA_CHAIN_ID}, received ${chainId}.`
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

  console.log("\nDeploying valueless $OMEGA Sepolia pilot suite");
  console.log(`  deployer              ${deployer.address}`);
  console.log(`  treasury              ${treasury}`);
  console.log(`  guardian              ${guardian}`);
  console.log(`  initial supply        ${ethers.formatUnits(initialSupply, 18)} tOMEGA`);
  console.log(`  claim threshold       ${ethers.formatUnits(claimThreshold, 18)} tOMEGA`);
  console.log(`  claim window          ${claimStart} → ${claimEnd}`);
  console.log(`  governance            ${votingDelay} block delay, ${votingPeriod} block period, ${quorumPercent}% quorum`);
  console.log(`  timelock delay        ${timelockDelay} seconds\n`);

  const OmegaTestToken = await ethers.getContractFactory("OmegaTestToken");
  const OmegaVotingEscrow = await ethers.getContractFactory("OmegaVotingEscrow");
  const TimelockController = await ethers.getContractFactory("TimelockController");
  const OmegaLocking = await ethers.getContractFactory("OmegaLocking");
  const OmegaGovernor = await ethers.getContractFactory("OmegaGovernor");
  const OmegaNovelGate = await ethers.getContractFactory("OmegaNovelGate");

  const token = await deploy("OmegaTestToken", OmegaTestToken, [treasury, initialSupply]);
  const votingEscrow = await deploy("OmegaVotingEscrow", OmegaVotingEscrow, [deployer.address]);

  // The deployer is the temporary setup admin only. The governor is granted
  // proposer rights below, and the deployer subsequently renounces this role.
  const timelock = await deploy("TimelockController", TimelockController, [
    timelockDelay,
    [],
    [ethers.ZeroAddress],
    deployer.address,
  ]);

  const timelockAddress = await timelock.getAddress();
  const locking = await deploy("OmegaLocking", OmegaLocking, [
    await token.getAddress(),
    await votingEscrow.getAddress(),
    treasury,
    timelockAddress,
    guardian,
  ]);

  await send("bind escrow locker", votingEscrow.setLocker(await locking.getAddress()));
  await send("renounce escrow owner", votingEscrow.renounceOwnership());

  const governor = await deploy("OmegaGovernor", OmegaGovernor, [
    await votingEscrow.getAddress(),
    timelockAddress,
    votingDelay,
    votingPeriod,
    proposalThreshold,
    quorumPercent,
  ]);

  const gate = await deploy("OmegaNovelGate", OmegaNovelGate, [
    await token.getAddress(),
    claimThreshold,
    claimStart,
    claimEnd,
    timelockAddress,
    guardian,
  ]);

  const proposerRole = await timelock.PROPOSER_ROLE();
  const adminRole = await timelock.DEFAULT_ADMIN_ROLE();
  await send("grant governor proposer", timelock.grantRole(proposerRole, await governor.getAddress()));
  await send("renounce timelock admin", timelock.renounceRole(adminRole, deployer.address));

  // Fail closed if the setup handoff did not complete as expected.
  if (await timelock.hasRole(adminRole, deployer.address)) {
    throw new Error("Setup handoff failed: deployer still has Timelock DEFAULT_ADMIN_ROLE.");
  }
  if (!(await timelock.hasRole(proposerRole, await governor.getAddress()))) {
    throw new Error("Setup handoff failed: governor is missing Timelock PROPOSER_ROLE.");
  }
  if ((await votingEscrow.owner()) !== ethers.ZeroAddress) {
    throw new Error("Setup handoff failed: voting escrow ownership was not renounced.");
  }

  const deployedAt = new Date().toISOString();
  const manifest = {
    schemaVersion: 1,
    network: "sepolia",
    chainId: Number(chainId),
    testnetOnly: true,
    deployedAt,
    deployer: deployer.address,
    treasury,
    guardian,
    contracts: {
      omegaTestToken: await token.getAddress(),
      omegaVotingEscrow: await votingEscrow.getAddress(),
      timelockController: timelockAddress,
      omegaLocking: await locking.getAddress(),
      omegaGovernor: await governor.getAddress(),
      omegaNovelGate: await gate.getAddress(),
    },
    configuration: {
      initialSupply,
      claimThreshold,
      claimStart,
      claimEnd,
      timelockDelay,
      votingDelay,
      votingPeriod,
      proposalThreshold,
      quorumPercent,
    },
    constructorArguments: {
      omegaTestToken: [treasury, initialSupply],
      omegaVotingEscrow: [deployer.address],
      timelockController: [timelockDelay, [], [ethers.ZeroAddress], deployer.address],
      omegaLocking: [
        await token.getAddress(),
        await votingEscrow.getAddress(),
        treasury,
        timelockAddress,
        guardian,
      ],
      omegaGovernor: [
        await votingEscrow.getAddress(),
        timelockAddress,
        votingDelay,
        votingPeriod,
        proposalThreshold,
        quorumPercent,
      ],
      omegaNovelGate: [
        await token.getAddress(),
        claimThreshold,
        claimStart,
        claimEnd,
        timelockAddress,
        guardian,
      ],
    },
  };

  const deploymentDirectory = path.join(__dirname, "..", "deployments");
  fs.mkdirSync(deploymentDirectory, { recursive: true });
  const manifestPath = path.join(deploymentDirectory, "sepolia.json");
  fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, jsonReplacer, 2)}\n`);

  console.log(`\nDeployment manifest written to ${manifestPath}`);
  console.log("This manifest contains public addresses and parameters only; it is intentionally gitignored.");
  console.log("Before distributing test tokens, verify the contracts and complete the post-deploy checklist in evm/README.md.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
