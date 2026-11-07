# Accepted npm audit advisories (EVM full-tree gate)

The `evm` full-tree audit step (`npm audit`, no level filter) in
`.github/workflows/ci.yml` is intentionally stricter than the runtime gate.
The advisories below are accepted with reasons and review dates. Any NEW
advisory fails CI until it is triaged here.

| Advisory | Package | Severity | Reason accepted | Review by |
|---|---|---|---|---|
| GHSA-…zod | zod@3.25.76 (via Hardhat) | (transitive) | No non-breaking upgrade exists; Hardhat pins it; dev-only transitive, not in shipped runtime bundle | 2027-05-06 |

Review policy: re-run `npm audit` in `evm/`, re-confirm each row, bump the
review date. If zod gains a non-breaking patched release, remove the row and
upgrade instead.
