//! A deterministic, valueless local pilot. This is not a market and has no oracle.
use ledger::Ledger;
use protocol_types::{Asset, WorkCategory};
use work_claims::ClaimRegistry;

fn main() {
    let mut ledger = Ledger::default();
    let mut claims = ClaimRegistry::default();
    let id = claims.propose(1, WorkCategory::PublicGood, 100).expect("claim");
    claims.attest(id).expect("attestation");
    claims.attest(id).expect("attestation");
    let issued = claims.finalize(id, &mut ledger).expect("finalization");
    println!("local pilot: claim={id}, issued_use={issued}, supply={}", ledger.total(Asset::Use));
}
