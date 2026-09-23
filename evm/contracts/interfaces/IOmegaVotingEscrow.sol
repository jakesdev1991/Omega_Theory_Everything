// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
pragma solidity ^0.8.24;

/// @notice Minimal privileged interface used by OmegaLocking.
interface IOmegaVotingEscrow {
    function mint(address recipient, uint256 amount) external;

    function burn(address account, uint256 amount) external;
}
