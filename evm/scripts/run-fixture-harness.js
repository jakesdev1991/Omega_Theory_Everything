// Offline fixture harness for verifying EVM $OMEGA signer and novel gate state
// Mirrors amity/scripts/run-fixture-harness.mjs and solana/scripts/run-fixture-harness.mjs

const assert = require("node:assert/strict");
const hre = require("hardhat");

async function main() {
  const [deployer, holder] = await hre.ethers.getSigners();
  const initialSupply = hre.ethers.parseEther("1000000");
  const claimThreshold = hre.ethers.parseEther("1");

  // Deploy OmegaTestToken with deployer as initial treasury
  const OmegaTestToken = await hre.ethers.getContractFactory("OmegaTestToken");
  const token = await OmegaTestToken.deploy(deployer.address, initialSupply);
  await token.waitForDeployment();
  const tokenAddress = await token.getAddress();

  const latestBlock = await hre.ethers.provider.getBlock("latest");
  const claimStart = Number(latestBlock.timestamp);
  const claimEnd = claimStart + 90 * 86400;

  // Deploy OmegaNovelGate
  const OmegaNovelGate = await hre.ethers.getContractFactory("OmegaNovelGate");
  const gate = await OmegaNovelGate.deploy(
    tokenAddress,
    claimThreshold,
    claimStart,
    claimEnd,
    deployer.address, // timelock address
    deployer.address  // guardian address
  );
  await gate.waitForDeployment();
  const gateAddress = await gate.getAddress();

  // Transfer 100 $OMEGA to holder
  const amount = hre.ethers.parseEther("100");
  await token.transfer(holder.address, amount);
  assert.equal(await token.balanceOf(holder.address), amount);

  // Holder qualifies for novel gate
  assert.equal(await gate.canClaim(holder.address), true);

  // Holder records novel claim
  await gate.connect(holder).claim();
  assert.equal(await gate.hasClaimed(holder.address), true);

  console.log("$OMEGA EVM fixture readiness harness");
  console.log("  network                 hardhat-simulated-sepolia");
  console.log(`  token address           ${tokenAddress}`);
  console.log(`  gate address            ${gateAddress}`);
  console.log(`  holder address          ${holder.address}`);
  console.log(`  holder balance          100 OMEGA`);
  console.log(`  qualifies novel gate    true`);
  console.log(`  has claimed novel gate  true`);
  console.log("\nFixture deployment, balance qualification, and novel gate claim passed.");
}

main().catch((err) => {
  console.error("Fixture error:", err);
  process.exitCode = 1;
});
