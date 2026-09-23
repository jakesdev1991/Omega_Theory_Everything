//! Local Proof-of-Useful-Work claim lifecycle.
use ledger::{Ledger, LedgerError};
use protocol_types::{AccountId, Amount, Asset, ClaimId, ClaimStatus, Event, WorkCategory, WorkClaim};
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Eq)]
pub enum ClaimError { NotFound, InvalidTransition, TooFewAttestations, Ledger(LedgerError) }
impl From<LedgerError> for ClaimError { fn from(e: LedgerError) -> Self { Self::Ledger(e) } }

pub struct ClaimRegistry { claims: BTreeMap<ClaimId, WorkClaim>, next_id: ClaimId }
impl Default for ClaimRegistry { fn default() -> Self { Self { claims: BTreeMap::new(), next_id: 1 } } }
impl ClaimRegistry {
    pub fn propose(&mut self, contributor: AccountId, category: WorkCategory, quantity: Amount) -> Result<ClaimId, ClaimError> {
        if quantity == 0 { return Err(ClaimError::InvalidTransition); }
        let id = self.next_id; self.next_id += 1;
        self.claims.insert(id, WorkClaim { id, contributor, category, quantity, attestation_count: 0, status: ClaimStatus::Proposed, protocol_version: 1 });
        Ok(id)
    }
    pub fn attest(&mut self, id: ClaimId) -> Result<(), ClaimError> {
        let c = self.claims.get_mut(&id).ok_or(ClaimError::NotFound)?;
        if !matches!(c.status, ClaimStatus::Proposed | ClaimStatus::Accepted) { return Err(ClaimError::InvalidTransition); }
        c.attestation_count = c.attestation_count.saturating_add(1);
        c.status = ClaimStatus::Accepted;
        Ok(())
    }
    pub fn finalize(&mut self, id: ClaimId, ledger: &mut Ledger) -> Result<Amount, ClaimError> {
        let c = self.claims.get_mut(&id).ok_or(ClaimError::NotFound)?;
        if c.status != ClaimStatus::Accepted { return Err(ClaimError::InvalidTransition); }
        if c.attestation_count < 2 { return Err(ClaimError::TooFewAttestations); }
        let issued = c.quantity.min(c.quantity.saturating_mul(c.attestation_count as Amount) / 2);
        ledger.credit(c.contributor, Asset::Use, issued)?;
        c.status = ClaimStatus::Finalized;
        Ok(issued)
    }
    pub fn get(&self, id: ClaimId) -> Option<&WorkClaim> { self.claims.get(&id) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn issuance_requires_two_attestations() {
        let mut r = ClaimRegistry::default(); let mut l = Ledger::default();
        let id = r.propose(7, WorkCategory::PublicGood, 10).unwrap();
        r.attest(id).unwrap();
        assert_eq!(r.finalize(id, &mut l), Err(ClaimError::TooFewAttestations));
        r.attest(id).unwrap();
        assert_eq!(r.finalize(id, &mut l), Ok(10));
        assert_eq!(l.balance(7, Asset::Use), 10);
    }
}
