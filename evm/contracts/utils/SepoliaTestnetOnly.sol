// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
pragma solidity ^0.8.24;

/// @notice Construction guard for the valueless $OMEGA pilot suite.
/// @dev This is intentionally a deployment-time guard, not a claim that Sepolia
///      itself is a security boundary. The suite must not be deployed to a live
///      value-bearing network without a separate specification, audit, and review.
abstract contract SepoliaTestnetOnly {
    uint256 public constant SEPOLIA_CHAIN_ID = 11_155_111;

    error UnsupportedChain(uint256 expectedChainId, uint256 actualChainId);

    constructor() {
        if (block.chainid != SEPOLIA_CHAIN_ID) {
            revert UnsupportedChain(SEPOLIA_CHAIN_ID, block.chainid);
        }
    }
}
