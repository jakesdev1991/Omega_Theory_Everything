// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
//! Versioned, deterministic wire types for the local sovereign-economy prototype.
//! No identity, telemetry, or clinical data belongs in these types.

pub type AccountId = u64;
pub type ClaimId = u64;
pub type Amount = u128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Asset {
    Sov,
    Use,
    Care,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Balance {
    pub account: AccountId,
    pub asset: Asset,
    pub amount: Amount,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimStatus {
    Proposed,
    Accepted,
    Finalized,
    Rejected,
    Reversed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkCategory {
    PublicGood,
    Caregiving,
    Maintenance,
    Research,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkClaim {
    pub id: ClaimId,
    pub contributor: AccountId,
    pub category: WorkCategory,
    pub quantity: Amount,
    pub attestation_count: u16,
    pub status: ClaimStatus,
    pub protocol_version: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    ClaimCreated { id: ClaimId },
    ClaimFinalized { id: ClaimId, issued: Amount },
    ClaimReversed { id: ClaimId, revoked: Amount },
    Transfer { from: AccountId, to: AccountId, asset: Asset, amount: Amount },
}
