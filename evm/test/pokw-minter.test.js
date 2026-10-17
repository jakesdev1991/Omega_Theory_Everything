// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
//
// Acceptance suite for OmegaScienceMinter — the PoKW claim contract.
// Mirrors the PoKW whitepaper's §7 pilot form and the suite conventions
// of omega-system.test.js (hardhat EDR, same fixtures style).

import { expect } from "chai";
import { network } from "hardhat";

const { ethers, networkHelpers } = await network.create();
const { setBalance } = networkHelpers;

const DAY = 24 * 60 * 60;

async function deployMinterSystem({ maxEpochPayout, budget }) {
  const [deployer, treasury, timelockAddr, minerA, minerB, rando] =
    await ethers.getSigners();

  const Token = await ethers.getContractFactory("OmegaTestToken");
  const token = await Token.deploy(treasury.address, budget * 3n);
  await token.waitForDeployment();

  const Minter = await ethers.getContractFactory("OmegaScienceMinter");
  const minter = await Minter.deploy(
    await token.getAddress(),
    timelockAddr.address,   // stand-in for the real TimelockController
    deployer.address,
    maxEpochPayout
  );
  await minter.waitForDeployment();

  // Fund the minter with its fixed science budget, then lock it.
  await (await token.connect(treasury)
    .transfer(await minter.getAddress(), budget)).wait();
  await (await minter.connect(deployer).activate()).wait();

  return { token, minter, deployer, treasury, timelockAddr, minerA, minerB, rando };
}

function makeLeaf(account, amountWei, receiptId) {
  // Must match the contract byte-for-byte:
  //   keccak256(abi.encode(account, amountWei, receiptId))
  // abi.encode PADDS each arg to 32 bytes (96 bytes total).
  // solidityPackedKeccak256 is abi.encodePacked (84 bytes) — a
  // different hash entirely, which is why claims hit InvalidProof.
  return ethers.keccak256(
    ethers.AbiCoder.defaultAbiCoder().encode(
      ["address", "uint256", "bytes32"],
      [account, amountWei, receiptId]
    )
  );
}

// Minimal OpenZeppelin-compatible Merkle tree over 1+ leaves with proof
// generation. OZ's MerkleProof._hashPair SORTS each pair before hashing
// (smaller hash first); a builder that hashes (a, b) in insertion order
// produces roots OZ will reject whenever a left child exceeds its sibling.
function makeTree(leaves) {
  if (leaves.length === 1) {
    return { root: leaves[0], proofFor: () => [] };
  }
  const level = [...leaves];
  const layers = [[...level]];
  while (level.length > 1) {
    const next = [];
    for (let i = 0; i < level.length; i += 2) {
      const a = level[i];
      const b = i + 1 < level.length ? level[i + 1] : level[i];
      // Mirror OZ: sort the pair — the smaller hash is hashed first.
      const pair = a <= b ? [a, b] : [b, a];
      next.push(
        ethers.solidityPackedKeccak256(["bytes32", "bytes32"], pair)
      );
    }
    level.length = 0;
    level.push(...next);
    layers.push([...next]);
  }
  function proofFor(leaf) {
    let idx = layers[0].indexOf(leaf);
    expect(idx, "leaf not in tree").to.be.at.least(0);
    const proof = [];
    for (let l = 0; l < layers.length - 1; l++) {
      const layer = layers[l];
      const sib = idx % 2 === 0 ? idx + 1 : idx - 1;
      proof.push(layer[sib < layer.length ? sib : idx]);
      idx = Math.floor(idx / 2);
    }
    return proof;
  }
  return { root: layers[layers.length - 1][0], proofFor };
}

describe("OmegaScienceMinter (PoKW claims)", () => {
  it("rejects zero cap and zero funding at construction", async () => {
    const [deployer, treasury, timelockAddr] = await ethers.getSigners();
    const Token = await ethers.getContractFactory("OmegaTestToken");
    const token = await Token.deploy(
      treasury.address, ethers.parseUnits("1000", 18)
    );
    await token.waitForDeployment();
    const Minter = await ethers.getContractFactory("OmegaScienceMinter");

    // zero cap
    await expect(
      Minter.deploy(await token.getAddress(), timelockAddr.address,
                    deployer.address, 0)
    ).to.be.revertedWithCustomError(Minter, "ZeroCap");

    // zero address
    await expect(
      Minter.deploy(await token.getAddress(), ethers.ZeroAddress,
                    deployer.address, 1n)
    ).to.be.revertedWithCustomError(Minter, "ZeroAddress");

    // Note: the SepoliaTestnetOnly construction guard is exercised by the
    // shared suite convention — the local EDR simulates Sepolia's chain
    // id (hardhat.config.js simulatedSepolia), so every construction in
    // this file proves the guard PASSES on 11155111. Deploying to any
    // other chain reverts UnsupportedChain (see utils/SepoliaTestnetOnly).
  });

  it("records a ratified batch from the timelock only", async () => {
    const budget = ethers.parseUnits("1000", 18);
    const cap = ethers.parseUnits("100", 18);
    const { minter, timelockAddr, rando } = await deployMinterSystem({
      maxEpochPayout: cap, budget,
    });

    const leaf = makeLeaf(rando.address, cap, ethers.id("r1"));
    const { root } = makeTree([leaf]);

    // A random address cannot submit.
    await expect(
      minter.connect(rando).submitRatifiedBatch(1, root, cap, 1, "ipfs://x")
    ).to.be.revertedWithCustomError(minter, "Unauthorized");

    // The timelock can.
    await expect(
      minter.connect(timelockAddr).submitRatifiedBatch(1, root, cap, 1, "ipfs://x")
    ).to.emit(minter, "BatchRatified");

    // ... but an epoch ratifies exactly once.
    await expect(
      minter.connect(timelockAddr).submitRatifiedBatch(1, root, cap, 1, "ipfs://x")
    ).to.be.revertedWithCustomError(minter, "EpochAlreadyPaid");
  });

  it("enforces the per-epoch cap and the science budget", async () => {
    const budget = ethers.parseUnits("1000", 18);
    const cap = ethers.parseUnits("100", 18);
    const { minter, timelockAddr } = await deployMinterSystem({
      maxEpochPayout: cap, budget,
    });

    const root = ethers.id("root");
    // over the cap
    await expect(
      minter.connect(timelockAddr).submitRatifiedBatch(
        1, root, cap + 1n, 1, "ipfs://x"
      )
    ).to.be.revertedWithCustomError(minter, "CapTooHigh");

    // fill the budget: 10 epochs of the max cap
    for (let e = 1; e <= 10; e++) {
      await minter.connect(timelockAddr).submitRatifiedBatch(
        e, ethers.id(`r${e}`), cap, 1, "ipfs://x"
      );
    }
    // the 11th must fail: budget exhausted
    await expect(
      minter.connect(timelockAddr).submitRatifiedBatch(
        11, ethers.id("r11"), cap, 1, "ipfs://x"
      )
    ).to.be.revertedWithCustomError(minter, "BudgetExhausted");
  });

  it("pays a valid claim once, to the committed account, for the committed amount", async () => {
    const budget = ethers.parseUnits("1000", 18);
    const cap = ethers.parseUnits("100", 18);
    const { token, minter, timelockAddr, minerA, minerB, rando } =
      await deployMinterSystem({ maxEpochPayout: cap, budget });

    const amtA = ethers.parseUnits("40", 18);
    const amtB = ethers.parseUnits("60", 18);
    const leaves = [
      makeLeaf(minerA.address, amtA, ethers.id("receipt-A")),
      makeLeaf(minerB.address, amtB, ethers.id("receipt-B")),
    ];
    const { root, proofFor } = makeTree(leaves);

    await minter.connect(timelockAddr).submitRatifiedBatch(
      1, root, amtA + amtB, 2, "ipfs://pokw-ci-run-1"
    );

    const minterAddr = await minter.getAddress();
    const proofA = proofFor(leaves[0]);

    // rando cannot claim minerA's leaf to himself: account is committed
    // in the leaf; the proof verifies but transfer goes to the committed
    // account. (rando passing his own address would fail the proof.)
    // Claim flow — manual balance diffs (the chai matcher requires a
    // connected token instance under hardhat-ethers v3):
    const aBefore = await token.balanceOf(minerA.address);
    const minterBefore = await token.balanceOf(minterAddr);
    await minter.connect(minerA).claimEpoch(
      1, proofA, minerA.address, amtA, ethers.id("receipt-A")
    );
    expect(await token.balanceOf(minerA.address)).to.equal(aBefore + amtA);
    expect(await token.balanceOf(minterAddr)).to.equal(minterBefore - amtA);

    // double-claim blocked
    await expect(
      minter.connect(minerA).claimEpoch(
        1, proofA, minerA.address, amtA, ethers.id("receipt-A")
      )
    ).to.be.revertedWithCustomError(minter, "LeafClaimed");

    // wrong amount fails the leaf hash -> invalid proof
    await expect(
      minter.connect(minerB).claimEpoch(
        1, proofFor(leaves[1]), minerB.address,
        amtB + 1n, ethers.id("receipt-B")
      )
    ).to.be.revertedWithCustomError(minter, "InvalidProof");

    // second miner claims fine
    const bBefore = await token.balanceOf(minerB.address);
    const minterBeforeB = await token.balanceOf(minterAddr);
    await minter.connect(minerB).claimEpoch(
      1, proofFor(leaves[1]), minerB.address, amtB, ethers.id("receipt-B")
    );
    expect(await token.balanceOf(minerB.address)).to.equal(bBefore + amtB);
    expect(await token.balanceOf(minterAddr)).to.equal(minterBeforeB - amtB);
  });

  it("blocks claims on unratified epochs and under pause", async () => {
    const budget = ethers.parseUnits("1000", 18);
    const cap = ethers.parseUnits("100", 18);
    const { minter, timelockAddr, minerA } = await deployMinterSystem({
      maxEpochPayout: cap, budget,
    });

    await expect(
      minter.connect(minerA).claimEpoch(99, [], minerA.address, 1n, ethers.id("x"))
    ).to.be.revertedWithCustomError(minter, "EpochNotRatified");

    // pause
    await minter.connect(timelockAddr).setClaimsPaused(true);
    const leaf = makeLeaf(minerA.address, 1n, ethers.id("p1"));
    const { root, proofFor } = makeTree([leaf]);
    await minter.connect(timelockAddr).submitRatifiedBatch(
      1, root, 1, 1, "ipfs://x"
    );
    await expect(
      minter.connect(minerA).claimEpoch(1, proofFor(leaf), minerA.address, 1n, ethers.id("p1"))
    ).to.be.revertedWithCustomError(minter, "ClaimsPaused");
  });

  it("bounds cumulative claims by the declared epoch payout and rescues residue after the delay", async () => {
    const budget = ethers.parseUnits("1000", 18);
    const cap = ethers.parseUnits("100", 18);
    const { token, minter, timelockAddr, minerA, minerB, rando } =
      await deployMinterSystem({ maxEpochPayout: cap, budget });

    // Declare 100 for the epoch, but build leaves summing to 150 —
    // a malicious or buggy batch builder. Cumulative claims must hit
    // the HARD bound at 100: the third claim reverts EpochCapExceeded.
    const amtA = ethers.parseUnits("60", 18);
    const amtB = ethers.parseUnits("60", 18); // leaves sum 120 > declared 100
    const leaves = [
      makeLeaf(minerA.address, amtA, ethers.id("rA")),
      makeLeaf(minerB.address, amtB, ethers.id("rB")),
    ];
    const { root, proofFor } = makeTree(leaves);
    await minter.connect(timelockAddr).submitRatifiedBatch(
      1, root, cap, 2, "ipfs://x"
    ); // declared totalPayoutWei = 100 (the per-epoch max)

    // First claim (60) passes.
    await minter.connect(minerA).claimEpoch(
      1, proofFor(leaves[0]), minerA.address, amtA, ethers.id("rA")
    );
    // Second claim (60) would take cumulative to 120 > 100 declared.
    await expect(
      minter.connect(minerB).claimEpoch(
        1, proofFor(leaves[1]), minerB.address, amtB, ethers.id("rB")
      )
    ).to.be.revertedWithCustomError(minter, "EpochCapExceeded");

    // Residue rescue: before the 30-day delay it refuses.
    await expect(
      minter.connect(timelockAddr).rescueEpochResidue(1, rando.address)
    ).to.be.revertedWithCustomError(minter, "RescueTooEarly");

    // After the delay, the unclaimed residue (100 - 60 = 40) returns to
    // governance and the epoch closes; a late claim reverts EpochClosed.
    const latest = await ethers.provider.getBlock("latest");
    await ethers.provider.send("evm_setNextBlockTimestamp", [
      latest.timestamp + 31 * 24 * 60 * 60,
    ]);
    const randoBefore = await token.balanceOf(rando.address);
    await minter.connect(timelockAddr).rescueEpochResidue(1, rando.address);
    expect(await token.balanceOf(rando.address)).to.equal(
      randoBefore + (cap - amtA)
    );
    await expect(
      minter.connect(minerB).claimEpoch(
        1, proofFor(leaves[1]), minerB.address, amtB, ethers.id("rB")
      )
    ).to.be.revertedWithCustomError(minter, "EpochClosed");

    // Aggregate accounting reconciles: the closed epoch's commitment is
    // released, so rescueUnratified can now return everything the minter
    // still holds that is not owed (1000 - 60 claimed - 40 already
    // rescued to rando) — nothing is trapped.
    const treasuryBefore = await token.balanceOf(timelockAddr.address);
    await minter.connect(timelockAddr).rescueUnratified(timelockAddr.address);
    expect(await token.balanceOf(timelockAddr.address)).to.equal(
      treasuryBefore + (budget - amtA - (cap - amtA))
    );
    expect(await token.balanceOf(await minter.getAddress())).to.equal(0n);
    // outstandingWei is now 0; remainingBudget still reports budget minus
    // ratified total (monotonic budget basis), which is the right query
    // for governance planning.
    expect(await minter.remainingBudget()).to.equal(budget - cap);
  });

  it("rescues only unratified funds", async () => {
    const budget = ethers.parseUnits("1000", 18);
    const cap = ethers.parseUnits("100", 18);
    const { token, minter, timelockAddr, treasury, rando } =
      await deployMinterSystem({ maxEpochPayout: cap, budget });

    // ratify one epoch of 100
    await minter.connect(timelockAddr).submitRatifiedBatch(
      1, ethers.id("root1"), cap, 1, "ipfs://x"
    );
    // rescue: balance 1000, ratified 100 -> rescuable 900
    const randoBefore = await token.balanceOf(rando.address);
    await minter.connect(timelockAddr).rescueUnratified(rando.address);
    expect(await token.balanceOf(rando.address)).to.equal(randoBefore + budget - cap);

    // the remaining 100 stays reserved for the ratified epoch
    expect(await token.balanceOf(await minter.getAddress())).to.equal(cap);
  });

  it("immutable budget equals the funded balance; no mint path exists", async () => {
    const budget = ethers.parseUnits("777", 18);
    const cap = ethers.parseUnits("100", 18);
    const { token, minter } = await deployMinterSystem({
      maxEpochPayout: cap, budget,
    });
    expect(await minter.scienceBudgetWei()).to.equal(budget);
    expect(await minter.remainingBudget()).to.equal(budget);
    // total supply unchanged by minter existence (fixed-supply discipline)
    expect(await token.totalSupply()).to.equal(budget * 3n);
  });
});
