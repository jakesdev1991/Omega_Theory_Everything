//! Deterministic append-only accounting state machine.
use protocol_types::{AccountId, Amount, Asset, Event};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerError { InsufficientBalance, Overflow, InvalidAmount, SameAccount }

#[derive(Debug, Default)]
pub struct Ledger {
    balances: BTreeMap<(AccountId, Asset), Amount>,
    events: Vec<Event>,
}

impl Ledger {
    pub fn balance(&self, account: AccountId, asset: Asset) -> Amount {
        self.balances.get(&(account, asset)).copied().unwrap_or(0)
    }
    pub fn events(&self) -> &[Event] { &self.events }
    pub fn credit(&mut self, account: AccountId, asset: Asset, amount: Amount) -> Result<(), LedgerError> {
        if amount == 0 { return Err(LedgerError::InvalidAmount); }
        let key = (account, asset);
        let next = self.balance(account, asset).checked_add(amount).ok_or(LedgerError::Overflow)?;
        self.balances.insert(key, next);
        Ok(())
    }
    pub fn transfer(&mut self, from: AccountId, to: AccountId, asset: Asset, amount: Amount) -> Result<(), LedgerError> {
        if amount == 0 { return Err(LedgerError::InvalidAmount); }
        if from == to { return Err(LedgerError::SameAccount); }
        let source = self.balance(from, asset);
        if source < amount { return Err(LedgerError::InsufficientBalance); }
        let destination = self.balance(to, asset).checked_add(amount).ok_or(LedgerError::Overflow)?;
        self.balances.insert((from, asset), source - amount);
        self.balances.insert((to, asset), destination);
        self.events.push(Event::Transfer { from, to, asset, amount });
        Ok(())
    }
    pub fn total(&self, asset: Asset) -> Amount {
        self.balances.iter().filter(|((_, a), _)| *a == asset).map(|(_, n)| *n).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transfer_preserves_supply() {
        let mut l = Ledger::default();
        l.credit(1, Asset::Sov, 100).unwrap();
        let before = l.total(Asset::Sov);
        l.transfer(1, 2, Asset::Sov, 40).unwrap();
        assert_eq!(before, l.total(Asset::Sov));
        assert_eq!(l.balance(1, Asset::Sov), 60);
        assert_eq!(l.balance(2, Asset::Sov), 40);
    }
    #[test]
    fn failed_transfer_is_atomic() {
        let mut l = Ledger::default();
        l.credit(1, Asset::Sov, 10).unwrap();
        assert_eq!(l.transfer(1, 2, Asset::Sov, 11), Err(LedgerError::InsufficientBalance));
        assert_eq!(l.total(Asset::Sov), 10);
        assert!(l.events().is_empty());
    }
}
