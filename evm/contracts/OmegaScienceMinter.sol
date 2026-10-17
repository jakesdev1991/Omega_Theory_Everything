// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
pragma solidity ^0.8.24;

import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {MerkleProof} from "@openzeppelin/contracts/utils/cryptography/MerkleProof.sol";
import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";

import {SepoliaTestnetOnly} from "./utils/SepoliaTestnetOnly.sol";

/// @title OmegaScienceMinter
/// @notice Proof-of-Kernel-Verified-Work (PoKW) claim contract for the
///         valueless Sepolia $OMEGA pilot. Implements the receipt-claim
///         leg of the PoKW whitepaper (§7): governance-ratified batches of
///         proof-work receipts authorize capped epoch payouts, and miners
///         claim their epoch payout against a Merkle root.
/// @dev Safety boundaries (mirroring the pilot suite):
///      * SepoliaTestnetOnly: cannot be constructed off Sepolia (chainId
///        11155111 or the local EDR simulation of it).
///      * No mint path exists here at all. The contract holds a FIXED
///        science budget of tOMEGA transferred to it at deployment; when
///        the budget is exhausted, payouts end. Total claimable is
///        therefore bounded by deploy-time funding, not by governance
///        whim — the fixed-supply discipline of the pilot is preserved.
///      * Only the timelock may submit ratified batches. Governance can
///        pause claims (guardian-style), but can never exceed the
///        remaining budget and can never pay twice for the same leaf.
///      * Nothing here verifies Lean proofs on-chain. The chain records
///        governance's RATIFICATION of an off-chain, publicly
///        reproducible verification run (CI kernel audit); the kernel
///        remains the arbiter of mathematical truth.
contract OmegaScienceMinter is SepoliaTestnetOnly, Ownable {
    // ------------------------------------------------------------------
    // Errors
    // ------------------------------------------------------------------
    error ZeroAddress();
    error ZeroCap();
    error CapTooHigh(uint256 requested, uint256 maxAllowed);
    error ZeroRoot();
    error EpochAlreadyPaid(uint256 epoch);
    error EpochCapExceeded(uint256 requested, uint256 cap);
    error BudgetExhausted(uint256 requested, uint256 remaining);
    error EpochNotRatified(uint256 epoch);
    error LeafClaimed(bytes32 leaf);
    error InvalidProof();
    error ClaimsPaused();
    error BatchTooLarge(uint256 size, uint256 max);
    error ZeroSupply();
    error Unauthorized(address caller);
    error TransferFailed();
    error NotActivated();
    error AlreadyActivated();
    error EpochClosed(uint256 epoch);
    error RescueTooEarly(uint256 epoch, uint256 unlockAt);

    // ------------------------------------------------------------------
    // Events
    // ------------------------------------------------------------------
    event BatchRatified(
        uint256 indexed epoch,
        bytes32 indexed batchRoot,
        uint256 epochPayoutWei,
        uint256 totalWeight,
        uint256 leafCount,
        string evidenceUri
    );
    event EpochClaimed(
        uint256 indexed epoch,
        address indexed claimant,
        uint256 amountWei,
        bytes32 leaf
    );
    event ClaimsPauseToggled(bool paused);
    event EpochCapSet(uint256 indexed epoch, uint256 capWei);
    event MaxEpochPayoutSet(uint256 maxWei);
    event Rescue(address indexed token, address indexed to, uint256 amount);
    event EpochRescued(uint256 indexed epoch, address indexed to, uint256 residueWei);

    // ------------------------------------------------------------------
    // Types
    // ------------------------------------------------------------------
    /// @notice One ratified receipt batch. The root commits to leaves of
    ///         keccak256(abi.encode(account, amountWei, receiptId)); the
    ///         payout scalar converts aggregate receipt weight to wei.
    struct Batch {
        bytes32 root;            // Merkle root of the receipt leaves
        uint256 totalPayoutWei;  // declared payout for the epoch; HARD bound
                                 // on cumulative claims for the epoch
        uint256 claimedWei;      // cumulative amount actually claimed
        uint256 leafCount;       // number of receipts in the batch
        uint64  ratifiedAt;      // block.timestamp of ratification
        bool    paid;            // true once the epoch payout executed
        bool    closed;          // true once residue-rescued; claims closed
    }

    // ------------------------------------------------------------------
    // Storage
    // ------------------------------------------------------------------
    IERC20 public immutable omega;
    address public immutable timelock;

    /// @notice Hard ceiling on any single epoch's payout. Immutable.
    uint256 public immutable maxEpochPayoutWei;

    /// @notice Total science budget this contract may ever pay out,
    ///         locked once at activation and never writable again
    ///         (guarded by `activated`; the deployer's role ends at
    ///         activation — the timelock holds all post-activate power).
    uint256 public scienceBudgetWei;

    /// @notice Set true by activate(); claims require it.
    bool public activated;

    /// @notice Sum of all ratified epoch payouts to date. Monotonic —
    ///         the basis for the science-budget bound (total ever
    ///         ratified can never exceed the locked budget).
    uint256 public ratifiedTotalWei;

    /// @notice Payout still owed to miners across open ratified epochs:
    ///         ratified amounts minus what claims have paid out and minus
    ///         closed epochs' rescued residues. This is what
    ///         rescueUnratified must keep reserved; it can never exceed
    ///         ratifiedTotalWei.
    uint256 public outstandingWei;

    /// @notice Global kill-style pause for claims (governance only).
    bool public claimsPaused;

    /// @notice Absolute cap on leaves per batch (gas/DoS bound).
    uint256 public maxBatchLeaves = 500;

    /// @notice Delay after ratification before an epoch's unclaimed
    ///         residue can be rescued (in-flight claim protection).
    uint256 public constant RESCUE_DELAY_SECONDS = 30 days;

    mapping(uint256 => Batch) public batches;
    mapping(bytes32 => bool) public claimedLeaf;
    mapping(address => uint256) public claimedByAccount;

    // ------------------------------------------------------------------
    // Constructor
    // ------------------------------------------------------------------
    constructor(
        address omegaToken,
        address timelockController,
        address deployer,
        uint256 maxEpochPayout
    )
        Ownable(deployer)
    {
        if (omegaToken == address(0) || timelockController == address(0)
            || deployer == address(0)) revert ZeroAddress();
        if (maxEpochPayout == 0) revert ZeroCap();

        omega = IERC20(omegaToken);
        timelock = timelockController;
        maxEpochPayoutWei = maxEpochPayout;
    }

    /// @notice Two-phase deployment step 2: after the treasury has
    ///         transferred the fixed science budget to this contract,
    ///         lock it. The budget is whatever the contract actually
    ///         holds at activation — there is no mint path; when the
    ///         budget is spent, PoKW payouts end.
    function activate() external onlyOwner {
        if (activated) revert AlreadyActivated();
        uint256 balance = omega.balanceOf(address(this));
        if (balance == 0) revert ZeroSupply();
        scienceBudgetWei = balance;
        activated = true;
    }

    // ------------------------------------------------------------------
    // Modifiers
    // ------------------------------------------------------------------
    modifier onlyTimelock() {
        if (msg.sender != timelock) revert Unauthorized(msg.sender);
        _;
    }

    // ------------------------------------------------------------------
    // Governance (timelock-routed) administration
    // ------------------------------------------------------------------

    /// @notice Record a governance-ratified receipt batch (PoKW §3-§6).
    ///         Callable only by the timelock, i.e. after a Governor vote
    ///         and the timelock delay.
    function submitRatifiedBatch(
        uint256 epoch,
        bytes32 batchRoot,
        uint256 totalPayoutWei,
        uint256 leafCount,
        string calldata evidenceUri
    )
        external
        onlyTimelock
    {
        if (!activated) revert NotActivated();
        if (batchRoot == bytes32(0)) revert ZeroRoot();
        if (totalPayoutWei == 0) revert ZeroCap();
        if (totalPayoutWei > maxEpochPayoutWei) {
            revert CapTooHigh(totalPayoutWei, maxEpochPayoutWei);
        }
        if (leafCount == 0 || leafCount > maxBatchLeaves) {
            revert BatchTooLarge(leafCount, maxBatchLeaves);
        }
        Batch storage b = batches[epoch];
        if (b.root != bytes32(0)) revert EpochAlreadyPaid(epoch);
        if (ratifiedTotalWei + totalPayoutWei > scienceBudgetWei) {
            revert BudgetExhausted(
                totalPayoutWei,
                scienceBudgetWei - ratifiedTotalWei
            );
        }

        b.root = batchRoot;
        b.totalPayoutWei = totalPayoutWei;
        b.leafCount = leafCount;
        b.ratifiedAt = uint64(block.timestamp);
        b.paid = false;

        ratifiedTotalWei += totalPayoutWei;
        outstandingWei += totalPayoutWei;

        emit BatchRatified(
            epoch, batchRoot, totalPayoutWei, totalPayoutWei, leafCount,
            evidenceUri
        );
    }

    /// @notice Mark an epoch's payout as executed (called by the timelock
    ///         after the treasury transfer for that epoch completes, when
    ///         the pilot uses treasury-funded payouts). For the common
    ///         case here — the minter holding its own budget — claims
    ///         drain the budget directly and this flag is set when the
    ///         epoch's ratified payout is first claimed.
    function markEpochPaid(uint256 epoch) external onlyTimelock {
        Batch storage b = batches[epoch];
        if (b.root == bytes32(0)) revert EpochNotRatified(epoch);
        b.paid = true;
    }

    function setClaimsPaused(bool paused) external onlyTimelock {
        claimsPaused = paused;
        emit ClaimsPauseToggled(paused);
    }

    function setMaxBatchLeaves(uint256 max) external onlyTimelock {
        if (max == 0) revert ZeroCap();
        maxBatchLeaves = max;
    }

    // ------------------------------------------------------------------
    // Miner claims (PoKW §3 last stage)
    // ------------------------------------------------------------------

    /// @notice Claim an epoch payout against a receipt leaf in a ratified
    ///         batch. The leaf commits to
    ///         keccak256(abi.encode(account, amountWei, receiptId)).
    /// @dev Each leaf is claimable exactly once, by the committed
    ///      account, for the committed amount. A receipt cannot be
    ///      double-paid across epochs because receiptId uniqueness is
    ///      enforced off-chain by the batch builder and committed to the
    ///      root; on-chain we enforce one-claim-per-leaf.
    function claimEpoch(
        uint256 epoch,
        bytes32[] calldata proof,
        address account,
        uint256 amountWei,
        bytes32 receiptId
    )
        external
    {
        if (claimsPaused) revert ClaimsPaused();
        if (!activated) revert NotActivated();
        Batch storage b = batches[epoch];
        if (b.root == bytes32(0)) revert EpochNotRatified(epoch);
        if (b.closed) revert EpochClosed(epoch);

        bytes32 leaf = keccak256(
            abi.encode(account, amountWei, receiptId)
        );
        if (claimedLeaf[leaf]) revert LeafClaimed(leaf);

        if (!MerkleProof.verify(proof, b.root, leaf)) revert InvalidProof();

        // The declared epoch payout is a HARD bound: cumulative claims
        // can never exceed what governance ratified for the epoch, even
        // if the off-chain batch builder's leaves sum higher. Fail closed.
        if (b.claimedWei + amountWei > b.totalPayoutWei) {
            revert EpochCapExceeded(b.claimedWei + amountWei, b.totalPayoutWei);
        }

        claimedLeaf[leaf] = true;
        claimedByAccount[account] += amountWei;
        b.claimedWei += amountWei;
        outstandingWei -= amountWei;
        b.paid = true;

        if (!omega.transfer(account, amountWei)) revert TransferFailed();

        emit EpochClaimed(epoch, account, amountWei, leaf);
    }

    // ------------------------------------------------------------------
    // Views
    // ------------------------------------------------------------------

    function remainingBudget() external view returns (uint256) {
        return scienceBudgetWei - ratifiedTotalWei;
    }

    function epochInfo(uint256 epoch)
        external
        view
        returns (Batch memory)
    {
        return batches[epoch];
    }

    function leafClaimed(bytes32 leaf) external view returns (bool) {
        return claimedLeaf[leaf];
    }

    // ------------------------------------------------------------------
    // Rescue (governance; never touches payable funds)
    // ------------------------------------------------------------------

    /// @notice Return UNRATIFIED tOMEGA to the timelock: anything the
    ///         contract holds above what is still OWED to miners across
    ///         open ratified epochs (see outstandingWei). Can never touch
    ///         funds backing ratified-but-unclaimed epochs — those are
    ///         governed by rescueEpochResidue below.
    function rescueUnratified(address to)
        external
        onlyTimelock
        returns (uint256)
    {
        uint256 balance = omega.balanceOf(address(this));
        uint256 rescuable = balance > outstandingWei
            ? balance - outstandingWei
            : 0;
        if (rescuable > 0) {
            if (!omega.transfer(to, rescuable)) revert TransferFailed();
            emit Rescue(address(omega), to, rescuable);
        }
        return rescuable;
    }

    /// @notice After RESCUE_DELAY_SECONDS, return an epoch's unclaimed
    ///         residue (totalPayoutWei - claimedWei) to the timelock and
    ///         close the epoch. Before the delay, in-flight claims are
    ///         untouchable; after it, late claims are sacrificed by
    ///         design so abandoned leaves can never trap pilot funds.
    /// @dev The epoch is closed BEFORE the transfer (checks-effects-
    ///      interactions). outstandingWei is decremented by the residue
    ///      (it was decremented by claims already): the epoch can never
    ///      re-ratify (b.root != 0) and never pay another claim (b.closed),
    ///      so its unclaimed remainder genuinely returns to the pool.
    ///      ratifiedTotalWei stays monotonic as the budget basis. Cumulative
    ///      payouts remain bounded by the immutable science budget; no
    ///      underflow is possible because each epoch contributes its
    ///      totalPayoutWei once to outstandingWei and can be closed at
    ///      most once, after claims decremented it by claimedWei.
    function rescueEpochResidue(uint256 epoch, address to)
        external
        onlyTimelock
        returns (uint256)
    {
        Batch storage b = batches[epoch];
        if (b.root == bytes32(0)) revert EpochNotRatified(epoch);
        if (b.closed) revert EpochClosed(epoch);
        if (block.timestamp < b.ratifiedAt + RESCUE_DELAY_SECONDS) {
            revert RescueTooEarly(epoch, b.ratifiedAt + RESCUE_DELAY_SECONDS);
        }
        uint256 residue = b.totalPayoutWei - b.claimedWei;
        b.closed = true;
        outstandingWei -= residue;
        if (residue > 0) {
            if (!omega.transfer(to, residue)) revert TransferFailed();
            emit Rescue(address(omega), to, residue);
        }
        emit EpochRescued(epoch, to, residue);
        return residue;
    }
}
