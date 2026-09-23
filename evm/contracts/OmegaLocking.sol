// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
pragma solidity ^0.8.24;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import {Math} from "@openzeppelin/contracts/utils/math/Math.sol";
import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

import {IOmegaVotingEscrow} from "./interfaces/IOmegaVotingEscrow.sol";
import {SepoliaTestnetOnly} from "./utils/SepoliaTestnetOnly.sol";

/// @title OmegaLocking
/// @notice Time-locks tOMEGA and issues non-transferable veOMEGA voting units.
/// @dev The pilot uses a fixed, inspectable duration multiplier: 1x at the
///      shortest allowed lock and up to 2x at a one-year lock. It does not
///      promise a reward or an economic return. Each position is independently
///      withdrawable after its visible expiry.
contract OmegaLocking is AccessControl, ReentrancyGuard, SepoliaTestnetOnly {
    using SafeERC20 for IERC20;

    bytes32 public constant EMERGENCY_ROLE = keccak256("EMERGENCY_ROLE");

    uint256 public constant BASIS_POINTS = 10_000;
    uint64 public constant MIN_LOCK_DURATION = 7 days;
    uint64 public constant MAX_LOCK_DURATION = 365 days;
    uint16 public constant MAX_BONUS_BPS = 10_000;

    // A zero penalty is intentional for the valueless Sepolia pilot. A future,
    // separately reviewed production design must not assume this policy.
    uint16 public constant EMERGENCY_EXIT_PENALTY_BPS = 0;

    error ZeroAddress();
    error ZeroAmount();
    error InvalidUnlockTime(uint256 requested, uint256 minimum, uint256 maximum);
    error PositionNotFound(uint256 positionId);
    error NotPositionOwner(uint256 positionId, address caller);
    error PositionAlreadyWithdrawn(uint256 positionId);
    error PositionStillLocked(uint256 positionId, uint256 unlockTime);
    error EmergencyExitDisabled();

    struct Position {
        address owner;
        uint64 lockedAt;
        uint64 unlockTime;
        uint256 amount;
        uint256 votingWeight;
        bool withdrawn;
    }

    event Locked(
        uint256 indexed positionId,
        address indexed owner,
        uint256 amount,
        uint64 unlockTime,
        uint256 votingWeight,
        uint256 durationFactorBps
    );
    event Withdrawn(
        uint256 indexed positionId,
        address indexed owner,
        uint256 amountReturned,
        uint256 penalty,
        bool emergency
    );
    event EmergencyExitUpdated(bool enabled, address indexed operator);

    IERC20 public immutable omega;
    IOmegaVotingEscrow public immutable votingEscrow;
    address public immutable treasury;

    uint256 public nextPositionId = 1;
    bool public emergencyExitEnabled;
    mapping(uint256 positionId => Position) public positions;

    constructor(
        IERC20 omega_,
        IOmegaVotingEscrow votingEscrow_,
        address treasury_,
        address governanceAdmin,
        address guardian
    ) {
        if (
            address(omega_) == address(0) || address(votingEscrow_) == address(0)
                || treasury_ == address(0) || governanceAdmin == address(0) || guardian == address(0)
        ) revert ZeroAddress();

        omega = omega_;
        votingEscrow = votingEscrow_;
        treasury = treasury_;

        _grantRole(DEFAULT_ADMIN_ROLE, governanceAdmin);
        _grantRole(EMERGENCY_ROLE, guardian);
    }

    /// @notice Lock tOMEGA until `unlockTime` and receive bounded veOMEGA voting units.
    function lock(uint256 amount, uint64 unlockTime) external nonReentrant returns (uint256 positionId) {
        if (amount == 0) revert ZeroAmount();

        uint256 minimum = block.timestamp + MIN_LOCK_DURATION;
        uint256 maximum = block.timestamp + MAX_LOCK_DURATION;
        if (unlockTime < minimum || unlockTime > maximum) {
            revert InvalidUnlockTime(unlockTime, minimum, maximum);
        }

        uint64 lockedAt = uint64(block.timestamp);
        uint256 factor = durationFactorBps(lockedAt, unlockTime);
        uint256 weight = Math.mulDiv(amount, factor, BASIS_POINTS);

        positionId = nextPositionId++;
        positions[positionId] = Position({
            owner: msg.sender,
            lockedAt: lockedAt,
            unlockTime: unlockTime,
            amount: amount,
            votingWeight: weight,
            withdrawn: false
        });

        omega.safeTransferFrom(msg.sender, address(this), amount);
        votingEscrow.mint(msg.sender, weight);

        emit Locked(positionId, msg.sender, amount, unlockTime, weight, factor);
    }

    /// @notice Withdraw a completed position. Only its original owner can withdraw it.
    function withdraw(uint256 positionId) external nonReentrant {
        Position storage position = _ownedActivePosition(positionId, msg.sender);
        if (block.timestamp < position.unlockTime) {
            revert PositionStillLocked(positionId, position.unlockTime);
        }
        _withdraw(positionId, position, false);
    }

    /// @notice Exit an active position before expiry only while the guardian has
    ///         activated the pilot emergency path. The policy and action are logged.
    function emergencyWithdraw(uint256 positionId) external nonReentrant {
        if (!emergencyExitEnabled) revert EmergencyExitDisabled();
        Position storage position = _ownedActivePosition(positionId, msg.sender);
        _withdraw(positionId, position, true);
    }

    /// @notice Toggle the emergency early-exit path. This cannot transfer, mint,
    ///         seize, or alter another holder's position.
    function setEmergencyExitEnabled(bool enabled) external onlyRole(EMERGENCY_ROLE) {
        emergencyExitEnabled = enabled;
        emit EmergencyExitUpdated(enabled, msg.sender);
    }

    /// @notice Compute the bounded multiplier for a lock interval.
    function durationFactorBps(uint64 startTime, uint64 unlockTime) public pure returns (uint256) {
        if (unlockTime <= startTime) return BASIS_POINTS;

        uint256 duration = uint256(unlockTime) - uint256(startTime);
        if (duration <= MIN_LOCK_DURATION) return BASIS_POINTS;
        if (duration > MAX_LOCK_DURATION) duration = MAX_LOCK_DURATION;

        return BASIS_POINTS
            + Math.mulDiv(
                duration - MIN_LOCK_DURATION,
                MAX_BONUS_BPS,
                MAX_LOCK_DURATION - MIN_LOCK_DURATION
            );
    }

    function _ownedActivePosition(uint256 positionId, address caller)
        internal
        view
        returns (Position storage position)
    {
        position = positions[positionId];
        if (position.owner == address(0)) revert PositionNotFound(positionId);
        if (position.owner != caller) revert NotPositionOwner(positionId, caller);
        if (position.withdrawn) revert PositionAlreadyWithdrawn(positionId);
    }

    function _withdraw(uint256 positionId, Position storage position, bool emergency) internal {
        position.withdrawn = true;

        uint256 penalty = emergency
            ? Math.mulDiv(position.amount, EMERGENCY_EXIT_PENALTY_BPS, BASIS_POINTS)
            : 0;
        uint256 amountReturned = position.amount - penalty;

        votingEscrow.burn(position.owner, position.votingWeight);
        if (penalty != 0) omega.safeTransfer(treasury, penalty);
        omega.safeTransfer(position.owner, amountReturned);

        emit Withdrawn(positionId, position.owner, amountReturned, penalty, emergency);
    }
}
