// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary

const { expect } = require("chai");
const { ethers } = require("hardhat");
const { anyValue } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");
const { mine, time } = require("@nomicfoundation/hardhat-network-helpers");

const DAY = 24 * 60 * 60;
const YEAR = 365 * DAY;

async function deploySystem({ votingPeriod = 5, timelockDelay = 3600 } = {}) {
  const [admin, treasury, guardian, alice, bob] = await ethers.getSigners();
  const initialSupply = ethers.parseUnits("1000000", 18);

  const Token = await ethers.getContractFactory("OmegaTestToken");
  const token = await Token.deploy(treasury.address, initialSupply);
  await token.waitForDeployment();

  const Escrow = await ethers.getContractFactory("OmegaVotingEscrow");
  const escrow = await Escrow.deploy(admin.address);
  await escrow.waitForDeployment();

  const Timelock = await ethers.getContractFactory("TimelockController");
  const timelock = await Timelock.deploy(
    timelockDelay,
    [],
    [ethers.ZeroAddress],
    admin.address
  );
  await timelock.waitForDeployment();

  const Locking = await ethers.getContractFactory("OmegaLocking");
  const locking = await Locking.deploy(
    await token.getAddress(),
    await escrow.getAddress(),
    treasury.address,
    await timelock.getAddress(),
    guardian.address
  );
  await locking.waitForDeployment();

  await (await escrow.setLocker(await locking.getAddress())).wait();
  await (await escrow.renounceOwnership()).wait();

  const Governor = await ethers.getContractFactory("OmegaGovernor");
  const governor = await Governor.deploy(
    await escrow.getAddress(),
    await timelock.getAddress(),
    1,
    votingPeriod,
    ethers.parseUnits("1", 18),
    4
  );
  await governor.waitForDeployment();

  const now = await time.latest();
  const Gate = await ethers.getContractFactory("OmegaNovelGate");
  const gate = await Gate.deploy(
    await token.getAddress(),
    ethers.parseUnits("1", 18),
    now - 1,
    now + 90 * DAY,
    await timelock.getAddress(),
    guardian.address
  );
  await gate.waitForDeployment();

  await (
    await timelock.grantRole(
      await timelock.PROPOSER_ROLE(),
      await governor.getAddress()
    )
  ).wait();
  await (
    await timelock.renounceRole(
      await timelock.DEFAULT_ADMIN_ROLE(),
      admin.address
    )
  ).wait();

  return {
    admin,
    treasury,
    guardian,
    alice,
    bob,
    initialSupply,
    token,
    escrow,
    timelock,
    locking,
    governor,
    gate,
  };
}

async function fundAndLock(system, amount = ethers.parseUnits("100", 18)) {
  const { token, treasury, alice, locking } = system;
  await (await token.connect(treasury).transfer(alice.address, amount)).wait();
  await (await token.connect(alice).approve(await locking.getAddress(), amount)).wait();

  const unlockTime = BigInt((await time.latest()) + YEAR);
  await (await locking.connect(alice).lock(amount, unlockTime)).wait();
  return { amount, unlockTime };
}

describe("$OMEGA Sepolia pilot contracts", function () {
  it("mints the complete fixed test supply once and exposes no mint function", async function () {
    const { token, treasury, initialSupply } = await deploySystem();

    expect(await token.name()).to.equal("Omega Test Token");
    expect(await token.symbol()).to.equal("tOMEGA");
    expect(await token.totalSupply()).to.equal(initialSupply);
    expect(await token.balanceOf(treasury.address)).to.equal(initialSupply);
    expect(await token.initialSupply()).to.equal(initialSupply);
    expect(token.interface.getFunction("mint")).to.equal(null);

    await (await token.connect(treasury).burn(ethers.parseUnits("1", 18))).wait();
    expect(await token.totalSupply()).to.equal(initialSupply - ethers.parseUnits("1", 18));
  });

  it("turns a bounded time lock into non-transferable voting power and returns the principal at expiry", async function () {
    const system = await deploySystem();
    const { token, escrow, locking, alice, bob } = system;
    expect(await locking.durationFactorBps(0, 7 * DAY)).to.equal(10000);
    expect(await locking.durationFactorBps(0, YEAR)).to.equal(20000);

    const { amount, unlockTime } = await fundAndLock(system);

    const position = await locking.positions(1);
    expect(position.owner).to.equal(alice.address);
    expect(position.amount).to.equal(amount);
    expect(position.votingWeight).to.be.greaterThan(amount);
    expect(await escrow.balanceOf(alice.address)).to.equal(position.votingWeight);
    expect(await escrow.delegates(alice.address)).to.equal(alice.address);
    expect(await token.balanceOf(await locking.getAddress())).to.equal(amount);

    await expect(escrow.connect(alice).transfer(bob.address, 1)).to.be.revertedWithCustomError(
      escrow,
      "NonTransferable"
    );
    await expect(locking.connect(alice).withdraw(1)).to.be.revertedWithCustomError(
      locking,
      "PositionStillLocked"
    );

    await time.increaseTo(Number(unlockTime) + 1);
    await expect(locking.connect(alice).withdraw(1))
      .to.emit(locking, "Withdrawn")
      .withArgs(1, alice.address, amount, 0, false);

    expect(await escrow.balanceOf(alice.address)).to.equal(0);
    expect(await token.balanceOf(alice.address)).to.equal(amount);
    expect((await locking.positions(1)).withdrawn).to.equal(true);
  });

  it("provides a guardian-controlled emergency exit without granting seizure or mint powers", async function () {
    const system = await deploySystem();
    const { locking, guardian, alice, token } = system;
    const { amount } = await fundAndLock(system);

    await expect(locking.connect(alice).emergencyWithdraw(1)).to.be.revertedWithCustomError(
      locking,
      "EmergencyExitDisabled"
    );
    await expect(locking.connect(guardian).setEmergencyExitEnabled(true))
      .to.emit(locking, "EmergencyExitUpdated")
      .withArgs(true, guardian.address);
    await expect(locking.connect(alice).emergencyWithdraw(1))
      .to.emit(locking, "Withdrawn")
      .withArgs(1, alice.address, amount, 0, true);

    expect(await token.balanceOf(alice.address)).to.equal(amount);
  });

  it("records exactly one qualifying gate claim and lets the guardian pause only the gate", async function () {
    const system = await deploySystem();
    const { token, treasury, gate, alice, bob, guardian } = system;
    const threshold = await gate.claimThreshold();

    await expect(gate.connect(alice).claim()).to.be.revertedWithCustomError(
      gate,
      "InsufficientTokenBalance"
    );

    await (await token.connect(treasury).transfer(alice.address, threshold)).wait();
    await expect(gate.connect(alice).claim())
      .to.emit(gate, "NovelClaimed")
      .withArgs(alice.address, threshold, threshold, anyValue);
    expect(await gate.hasClaimed(alice.address)).to.equal(true);
    expect(await gate.canClaim(alice.address)).to.equal(false);
    await expect(gate.connect(alice).claim()).to.be.revertedWithCustomError(gate, "AlreadyClaimed");

    await (await token.connect(treasury).transfer(bob.address, threshold)).wait();
    await (await gate.connect(guardian).pause()).wait();
    await expect(gate.connect(bob).claim()).to.be.reverted;
    expect(await token.balanceOf(bob.address)).to.equal(threshold);
    await (await gate.connect(guardian).unpause()).wait();
    await expect(gate.connect(bob).claim()).to.emit(gate, "NovelClaimed");
  });

  it("hands governance to a vote-escrow snapshot, queues through the timelock, and executes only after delay", async function () {
    const system = await deploySystem({ votingPeriod: 5, timelockDelay: 3600 });
    const { timelock, governor, alice } = system;
    await fundAndLock(system);

    const Target = await ethers.getContractFactory("TimelockTarget");
    const target = await Target.deploy(await timelock.getAddress());
    await target.waitForDeployment();

    const description = "Set the pilot governance test value to 42";
    const descriptionHash = ethers.id(description);
    const targets = [await target.getAddress()];
    const values = [0];
    const calldatas = [target.interface.encodeFunctionData("setValue", [42])];
    const proposalId = await governor.hashProposal(targets, values, calldatas, descriptionHash);

    await expect(governor.connect(alice).propose(targets, values, calldatas, description))
      .to.emit(governor, "ProposalCreated");

    await mine(2);
    await expect(governor.connect(alice).castVote(proposalId, 1)).to.emit(governor, "VoteCast");
    await mine(6);

    await expect(governor.queue(targets, values, calldatas, descriptionHash)).to.emit(
      governor,
      "ProposalQueued"
    );
    await expect(governor.execute(targets, values, calldatas, descriptionHash)).to.be.reverted;

    await time.increase(3601);
    await expect(governor.execute(targets, values, calldatas, descriptionHash)).to.emit(
      governor,
      "ProposalExecuted"
    );
    expect(await target.value()).to.equal(42);
  });

  it("removes temporary setup authority while keeping the governor as timelock proposer", async function () {
    const { admin, timelock, governor, escrow } = await deploySystem();
    const adminRole = await timelock.DEFAULT_ADMIN_ROLE();
    const proposerRole = await timelock.PROPOSER_ROLE();

    expect(await timelock.hasRole(adminRole, admin.address)).to.equal(false);
    expect(await timelock.hasRole(proposerRole, await governor.getAddress())).to.equal(true);
    expect(await escrow.owner()).to.equal(ethers.ZeroAddress);
  });
});
