<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary. -->

# Local protocol prototype

This workspace is the first implementation step from the Tri-Token Sovereign Economy blueprint. It is intentionally local and valueless.

**License:** All Rust source and workspace files under `rust/` are product materials and all rights reserved (`LicenseRef-Omega-Product-Proprietary`), not open-source licensed. Each crate uses a proprietary `license-file` and is marked non-publishable. A commercial license requires a separate signed agreement with percentage-based royalty terms. See [`../docs/LICENSING.md`](../docs/LICENSING.md).

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
