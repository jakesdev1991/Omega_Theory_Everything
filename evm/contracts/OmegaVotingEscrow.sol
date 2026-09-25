// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
pragma solidity ^0.8.24;

import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ERC20Permit} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import {ERC20Votes} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import {Nonces} from "@openzeppelin/contracts/utils/Nonces.sol";

import {SepoliaTestnetOnly} from "./utils/SepoliaTestnetOnly.sol";

/// @title OmegaVotingEscrow
/// @notice Non-transferable voting units minted only against an active OMEGA lock.
/// @dev This contract is deliberately not a liquid asset. Its one-time locker is
///      configured during deployment and the owner is then renounced. Locking
///      automatically self-delegates a new holder who has not delegated before.
contract OmegaVotingEscrow is ERC20, ERC20Permit, ERC20Votes, Ownable, SepoliaTestnetOnly {
    error ZeroAddress();
    error ZeroAmount();
    error LockerAlreadySet();
    error NotLocker(address caller);
    error NonTransferable();

    event LockerSet(address indexed locker);

    address public locker;

    constructor(address initialOwner)
        ERC20("Vote-Escrowed Omega Test", "veOMEGA")
        ERC20Permit("Vote-Escrowed Omega Test")
        Ownable(initialOwner)
    {
        if (initialOwner == address(0)) revert ZeroAddress();
    }

    /// @notice Bind this voting escrow to its sole locking contract exactly once.
    function setLocker(address locker_) external onlyOwner {
        if (locker_ == address(0)) revert ZeroAddress();
        if (locker != address(0)) revert LockerAlreadySet();

        locker = locker_;
        emit LockerSet(locker_);
    }

    /// @notice Create voting units when the associated OMEGA is locked.
    function mint(address recipient, uint256 amount) external onlyLocker {
        if (recipient == address(0)) revert ZeroAddress();
        if (amount == 0) revert ZeroAmount();

        _mint(recipient, amount);

        // ERC20Votes requires explicit delegation. Defaulting a first lock to
        // self-delegation makes the voting rule visible and prevents accidental
        // zero voting weight while preserving later delegation choices.
        if (delegates(recipient) == address(0)) {
            _delegate(recipient, recipient);
        }
    }

    /// @notice Destroy voting units as the matching OMEGA lock exits.
    function burn(address account, uint256 amount) external onlyLocker {
        if (amount == 0) revert ZeroAmount();
        _burn(account, amount);
    }

    modifier onlyLocker() {
        if (msg.sender != locker) revert NotLocker(msg.sender);
        _;
    }

    /// @dev Only mint and burn are permitted; transfers between accounts revert.
    function _update(address from, address to, uint256 value)
        internal
        virtual
        override(ERC20, ERC20Votes)
    {
        if (from != address(0) && to != address(0)) revert NonTransferable();
        super._update(from, to, value);
    }

    /// @dev Required by Solidity for ERC20Permit nonce bookkeeping.
    function nonces(address owner)
        public
        view
        virtual
        override(ERC20Permit, Nonces)
        returns (uint256)
    {
        return super.nonces(owner);
    }
}
