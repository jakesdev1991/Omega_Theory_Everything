# Local protocol prototype

This workspace is the first implementation step from the Tri-Token Sovereign Economy blueprint. It is intentionally local and valueless.

- `protocol-types`: versioned ledger and work-claim types.
- `ledger`: deterministic accounting state machine with atomic transfers.
- `work-claims`: two-attestation minimum before `USE` issuance.
- `simulation`: tiny deterministic local pilot executable.

Run from a machine with Rust installed:

```bash
cargo test --workspace
cargo run -p simulation
```

Not implemented yet: networking, identity, telemetry ingestion, psychosocial inference, AMM/FINN clearing, smart contracts, custody, or real-value issuance.
