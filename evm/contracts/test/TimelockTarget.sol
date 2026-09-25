// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
pragma solidity ^0.8.24;

/// @dev Test-only target used to prove the governor -> timelock -> execution path.
contract TimelockTarget {
    error OnlyTimelock(address caller);

    address public immutable timelock;
    uint256 public value;

    constructor(address timelock_) {
        timelock = timelock_;
    }

    function setValue(uint256 nextValue) external {
        if (msg.sender != timelock) revert OnlyTimelock(msg.sender);
        value = nextValue;
    }
}
