<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

# Launch controls and readiness ledger

This directory contains planning and local-control tools for the public manuscript and the
separate valueless pilot rails. It does **not** authorize deployment, accept funds, contain
private keys, or replace legal, security, privacy, safeguarding, or operational review.

## Readiness ledger

[`readiness-gates.json`](readiness-gates.json) is the public, evidence-backed summary of what is
complete and what remains blocked. Its three scopes are deliberately different:

- `public-manuscript` — the already-public CC BY 4.0 draft and its non-exclusive access posture;
- `testnet-pilot` — a reversible, valueless Sepolia/Devnet/Testnet drill only;
- `mainnet` — any value-bearing or production launch.

Validate and inspect the ledger from the repository root:

```bash
python launch/check_readiness.py --validate
python launch/check_readiness.py --report --scope testnet-pilot
python launch/check_readiness.py --require mainnet
```

The last command exits non-zero while any mainnet condition is incomplete. That is expected and
is the point: a green code build is not permission to cross a legal, security, or human-safety
gate.

Evidence references in the public ledger must be public repository paths or HTTPS URLs. Keep
keys, deployment manifests, personal contact information, legal work product, incident details,
and other confidential evidence in an access-controlled system.

## Pilot helpers

- [`check_unlock_rails.mjs`](check_unlock_rails.mjs) reports local optional-reader-proof rail
  configuration. A missing manifest means the rail is not activated; it does not mean a failure
  to be worked around.
- [`sync_web_unlock_env.mjs`](sync_web_unlock_env.mjs) copies public pilot metadata from local
  EVM/Solana manifests into ignored `web/.env.local`. It does not handle private keys.
- [`novel_day_one_plan.md`](novel_day_one_plan.md) remains the planning/risk document. The
  ledger provides a short, machine-validated control plane for its release blockers.

See [`../docs/SOFTWARE_SUPPLY_CHAIN.md`](../docs/SOFTWARE_SUPPLY_CHAIN.md) for the dependency
inventory, production and bounded development npm advisory controls, Python resolver audit, and
Dependabot coverage check.
