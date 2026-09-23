// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
pragma solidity ^0.8.24;

import {ERC20} from "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import {ERC20Burnable} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import {ERC20Permit} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Permit.sol";
import {ERC20Votes} from "@openzeppelin/contracts/token/ERC20/extensions/ERC20Votes.sol";
import {Nonces} from "@openzeppelin/contracts/utils/Nonces.sol";

import {SepoliaTestnetOnly} from "./utils/SepoliaTestnetOnly.sol";

/// @title OmegaTestToken
/// @notice Fixed-supply, valueless ERC-20 used only for the Sepolia $OMEGA pilot.
/// @dev The entire supply is minted exactly once in the constructor. There is no
///      post-deployment mint path or administrative transfer/pause path. Holders
///      can voluntarily burn their own balance through ERC20Burnable.
contract OmegaTestToken is
    ERC20,
    ERC20Burnable,
    ERC20Permit,
    ERC20Votes,
    SepoliaTestnetOnly
{
    error ZeroAddress();
    error ZeroSupply();

    /// @notice Amount minted at construction; total supply can only decrease by burning.
    uint256 public immutable initialSupply;

    constructor(address initialTreasury, uint256 supply)
        ERC20("Omega Test Token", "tOMEGA")
        ERC20Permit("Omega Test Token")
    {
        if (initialTreasury == address(0)) revert ZeroAddress();
        if (supply == 0) revert ZeroSupply();

        initialSupply = supply;
        _mint(initialTreasury, supply);
    }

    /// @dev Required by Solidity for ERC20 + ERC20Votes accounting hooks.
    function _update(address from, address to, uint256 value)
        internal
        virtual
        override(ERC20, ERC20Votes)
    {
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
