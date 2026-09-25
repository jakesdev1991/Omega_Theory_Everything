// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
pragma solidity ^0.8.24;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";
import {IERC20} from "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import {Pausable} from "@openzeppelin/contracts/utils/Pausable.sol";

import {SepoliaTestnetOnly} from "./utils/SepoliaTestnetOnly.sol";

/// @title OmegaNovelGate
/// @notice Records one Sepolia pilot claim per qualifying tOMEGA wallet.
/// @dev This contract records an eligibility receipt only. It stores no content
///      key, personal data, private evidence, or entitlement that has value off
///      the test network. Threshold and claim window are immutable.
contract OmegaNovelGate is AccessControl, Pausable, SepoliaTestnetOnly {
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");

    error ZeroAddress();
    error ZeroThreshold();
    error InvalidClaimWindow(uint64 startTime, uint64 endTime);
    error ClaimNotOpen(uint256 currentTime, uint64 startTime);
    error ClaimClosed(uint256 currentTime, uint64 endTime);
    error AlreadyClaimed(address claimant);
    error InsufficientTokenBalance(uint256 actualBalance, uint256 requiredBalance);

    event NovelClaimed(
        address indexed claimant,
        uint256 observedTokenBalance,
        uint256 requiredTokenBalance,
        uint64 indexed claimedAt
    );

    IERC20 public immutable omega;
    uint256 public immutable claimThreshold;
    uint64 public immutable claimStart;
    uint64 public immutable claimEnd;

    mapping(address claimant => bool) public hasClaimed;

    constructor(
        IERC20 omega_,
        uint256 claimThreshold_,
        uint64 claimStart_,
        uint64 claimEnd_,
        address governanceAdmin,
        address guardian
    ) {
        if (address(omega_) == address(0) || governanceAdmin == address(0) || guardian == address(0)) {
            revert ZeroAddress();
        }
        if (claimThreshold_ == 0) revert ZeroThreshold();
        if (claimStart_ >= claimEnd_) revert InvalidClaimWindow(claimStart_, claimEnd_);

        omega = omega_;
        claimThreshold = claimThreshold_;
        claimStart = claimStart_;
        claimEnd = claimEnd_;

        _grantRole(DEFAULT_ADMIN_ROLE, governanceAdmin);
        _grantRole(PAUSER_ROLE, guardian);
    }

    /// @notice Record a claim while the caller holds at least the immutable threshold.
    function claim() external whenNotPaused {
        if (block.timestamp < claimStart) revert ClaimNotOpen(block.timestamp, claimStart);
        if (block.timestamp > claimEnd) revert ClaimClosed(block.timestamp, claimEnd);
        if (hasClaimed[msg.sender]) revert AlreadyClaimed(msg.sender);

        uint256 balance = omega.balanceOf(msg.sender);
        if (balance < claimThreshold) {
            revert InsufficientTokenBalance(balance, claimThreshold);
        }

        hasClaimed[msg.sender] = true;
        emit NovelClaimed(msg.sender, balance, claimThreshold, uint64(block.timestamp));
    }

    /// @notice Return whether an address can make a new claim at the current block.
    function canClaim(address claimant) external view returns (bool) {
        return !paused() && !hasClaimed[claimant] && block.timestamp >= claimStart
            && block.timestamp <= claimEnd && omega.balanceOf(claimant) >= claimThreshold;
    }

    /// @notice Guardian-only circuit breaker for the gate. It cannot pause tOMEGA transfers.
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }
}
