# Solana kit 8 migration block

**Status (2026-11-06):** documented block, unchanged — migration deferred.

`@solana-program/system` 0.16 and `@solana-program/token` 0.18 require
`@solana/kit@8`, but `@metaplex-foundation/mpl-token-metadata-kit` (currently
0.0.3, peering `^6.5.0`) has no kit-8-compatible release. The repo pins
`@solana/kit@6.5.0` with `@solana-program/system@0.12.2` and
`@solana-program/token@0.13.0` (see `solana/package.json`).

Unblock condition: mpl-token-metadata-kit publishes a kit-8-compatible
release, or the metaplex dependency is replaced. Until then, do not bump
`@solana/kit` — it will break the metaplex peer and the devnet pilot.

Review by: 2027-05-06.
