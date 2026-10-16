<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-ReadOnly -->

# Engineering security and evidence review — 2026-10-16

> This is a code-level maintenance record for PR #85, not a penetration test, formal audit, certification, release approval, or claim that the repository is production-ready.

## Scope and changes

### Browser wallet and optional reader proof

- Removed recovery-mnemonic output from the browser console. A newly generated phrase stays visible until the user confirms an offline backup and hides it; another wallet cannot be created while the phrase remains unconfirmed. Copy is explicit, and if encrypted storage fails the page still surfaces the generated phrase so it is not silently discarded.
- Removed the `Math.random()` entropy fallback. Wallet creation now stops unless `crypto.getRandomValues` is available and succeeds.
- Set the V3 keystore scrypt parameters to ethers' default-strength `N=131072, r=8, p=1`. Keystore/account writes are checked; partial writes are rolled back on failure where browser storage permits. Legacy keystore migration no longer deletes the source if writing the new key fails.
- Masked imported mnemonic/private-key input. The browser UI no longer triggers an automatic public-RPC balance lookup on opening the wallet view; the user must press Refresh. The interface documents that the active address is sent to public Ethereum RPCs for this optional mainnet balance read.
- `forgetWallet()` clears the local EVM keystore, active address, account metadata, Solana address metadata, old/new proof history, and tab-staged proof. It does not revoke MetaMask/Phantom extension permissions or clear a clipboard/export the user saved elsewhere.
- Removed automatic localStorage persistence of signed reader proofs and the proof-history API. Existing proof-history keys are discarded. A proof is staged in session storage only after the user opens the optional reader and is consumed on the reader's first load. The server rechecks the current on-chain rail; this flow is not an authentication system and is not needed to read the public manuscript.
- MetaMask proof signing now requires chain ID `0xaa36a7` (Ethereum Sepolia) before it creates a message labelled `ethereum-sepolia`.
- Unlock messages now require a canonical HTTP(S) origin, and the API binds the signed origin to the current request origin. This is same-request-origin binding, not a deployment-domain allowlist; reverse-proxy/host configuration must be correct.

### Nostr storefront key handling

- Replaced persistent plaintext storage of the browser operator secret with an in-memory, tab-only key. A legacy key is recovered into memory where possible and its localStorage copy is removed. Users can explicitly import, reveal, or copy an `nsec`; those actions expose a private credential and require user care. Reloading without importing an exported key creates a new identity.
- The UI and security text now state that signed job requests are public Nostr events, while the raw operator secret is not sent to relays.

### Web response headers

- Added a baseline CSP, `Referrer-Policy`, `X-Content-Type-Options`, same-origin framing protection, a restrictive `Permissions-Policy`, and HSTS in `web/next.config.ts`.
- The CSP deliberately permits inline scripts/styles needed by this app, the existing Google Fonts stylesheet, HTTPS/WSS connections for public RPC/Nostr use, and WS for development. It is not a nonce-based strict CSP and is not a substitute for XSS prevention or deployment review.

### MCP simulation

- Replaced Python `assert` statements used for simulation invariants with explicit runtime validation, so invalid transfers/claims are still rejected under `python -O`. The HTTP transport remains unauthenticated and loopback-only by default; this hardening does not make it safe to expose to untrusted networks.

## Dependency advisory results

Manual npm audits on 2026-10-16 used the committed lockfiles; Python audits used the declared runtime requirements:

| Package root | Full dependency audit |
|---|---:|
| `web/` | 0 vulnerabilities |
| `mobile-node/` | 0 |
| `nostr-client/` | 0 |
| `amity/` | 0 |
| `desktop/` | 0 |
| `solana/` | 0 |
| `evm/` | 0 after the reviewed Hardhat 3 migration |

The web app's Next.js framework is `16.4.2`; safe transitive fixes were applied, unused lint packages were removed, and remaining lint dependencies are in `devDependencies`. Its full `npm audit` is clean. `pip-audit 2.10.1` reported no known vulnerabilities for `requirements.txt` or the `mcp/` project. These Python requirements are not a full lock/SBOM, and advisory results are time- and database-dependent. The existing Trivy filesystem scan remains in CI.

The EVM development stack was migrated from Hardhat 2.29.1/toolbox 5 to Hardhat `3.20.0` with `@nomicfoundation/hardhat-toolbox-mocha-ethers` `4.0.0`; the config, Mocha tests, and scripts now use the ESM/Hardhat 3 APIs. The project retains the pinned Solidity compiler `solc` `0.8.26` to avoid changing contract bytecode. Its old `tmp` transitive dependency is overridden to patched `0.2.7`. A fresh `npm ci` and full `npm audit` report zero vulnerabilities. The H3 compile, size check, six contract tests, local fixture, preflight rehearsal, and a simulated local deployment all passed; the local deployment manifest was removed afterward. No live Sepolia transaction was sent.

### Rust, Tauri, and Lean/Mathlib

- The first-party `rust/` workspace manifests contain only local path dependencies; they declare no third-party crates. The Tauri wrapper does have external dependencies. GitHub's Tauri advisories identify `tauri` versions through `2.11.0` ([GHSA-7gmj-67g7-phm9](https://github.com/advisories/GHSA-7gmj-67g7-phm9)) and through `2.11.5` ([GHSA-w28w-mhc8-qvjv](https://github.com/tauri-apps/tauri/security/advisories/GHSA-w28w-mhc8-qvjv)) as affected. `desktop/src-tauri/Cargo.toml` now pins `tauri` `=2.11.6` and `tauri-build` `=2.6.3`; the Tauri 2.11.6 release retains the project's Rust 1.77 line (minimum 1.77.2), so the declared MSRV was corrected to `1.77.2` rather than raised to the later Tauri 2.12 line. The direct Tauri version is outside both affected ranges.
- This environment has no `cargo`/`rustc`, and neither Rust nor Tauri has a committed `Cargo.lock`; therefore no resolved Cargo graph or local `cargo audit` result is available. CI now installs pinned `cargo-audit` `0.22.2`, generates a lockfile for each Cargo root, and audits the Rust workspace and Tauri tree. On PR #85, GitHub Actions completed the Rust workspace build/tests and both generated-lock audits successfully. Because the generated lockfiles are ephemeral, this confirms the PR-time resolved graphs but does not pin the dependency graph for future runs.
- `lean_proofs/lake-manifest.json` pins nine Git packages by revision; the root toolchain and Mathlib are both Lean `4.32.0`. A GitHub advisory review found [GHSA-qrv2-rm2g-m65v](https://github.com/leanprover-community/import-graph/security/advisories/GHSA-qrv2-rm2g-m65v) for `import-graph`: the locked revision `7e9612bf0b9ee66db3cb5b9988a35afc706f5a12` contains the pre-`0.1.0` HTML graph tool, whose `docs_url` query parameter allowed open redirects/XSS in generated graphs. The upstream fix is at [merge commit `1b60e1ed6c0d152aed0c782ad2fdf09365700b72`](https://github.com/leanprover-community/import-graph/commit/1b60e1ed6c0d152aed0c782ad2fdf09365700b72) (package version `0.1.0`), but that revision updates its `Cli` dependency to `v4.35.0-rc3`; the current project is pinned to `Cli` `v4.32.0`. `lake`/`lean` are unavailable here, so a coordinated toolchain/Mathlib update cannot be safely validated. The affected executable is not invoked by this repository's `lake build ToE` CI job, and no generated graph HTML is shipped here, but the dependency advisory remains an explicit exception until a tested Lean/Mathlib upgrade. Do not publish HTML graphs generated by this pinned revision or direct users to those graph pages.

CI gates the full EVM npm tree, root/MCP Python dependencies, and generated-lock Cargo audits. The PR #85 run passed the Rust workspace build/tests and both Cargo audits. The existing Lean CI still builds `ToE` and applies source/kernel audits; it does not currently perform a Lake advisory scan.

## Evidence and validation

- `npm ci` in `web/` after the ESLint 10.12.0 update completed with zero reported vulnerabilities; `npm audit --audit-level=high` also reported zero.
- `npm test` in `web/`: 43 tests passed, including signed EVM/Solana proof origin-binding tests, MCP tool inventory checks, and session operator-key migration checks.
- `npm run test:wallet` in `web/`: 14 browser-wallet regression tests passed, covering secure entropy, scrypt work factor, failed storage writes/rollback, legacy proof cleanup, forget behavior and deletion errors, Sepolia signing guard, masked import, no modal prompts/mnemonic logging, interactive backup confirmation and replacement-wallet blocking, phrase recovery on storage failure, local signing-password clearing, and opt-in balance RPC.
- `npm run typecheck`, `npm run lint`, and `npm run build` in `web/`: passed with Next.js 16.4.2; the build generated all 55 static pages.
- Root Python checks: 104 pytest tests plus 81 subtests passed; Ruff lint and format checks and MyPy passed.
- MCP smoke test: 22 tools, manifest parity, loopback Streamable HTTP bind, guarded CARE/AMITY/SOV mutations, and stdio JSON-RPC round-trip passed on Python 3.11.2 / MCP SDK 1.30.0. A separate `python -O` check confirmed business invariants remain active.
- `pip-audit 2.10.1` reported no known vulnerabilities for root requirements and `mcp/`.
- In `evm/`, clean `npm ci`, `npm run check` (9 Solidity files compiled, bytecode-size gates passed, 6/6 contract tests), `npm run test:fixture`, and `npm run preflight:rehearsal` passed. Full `npm audit` reported 0 vulnerabilities. A local simulated deployment exercised the migrated deploy script with dummy values; its ignored manifest was removed. No live Sepolia transaction was sent.
- The Tauri minimum version was checked against upstream Tauri 2.11.6 release metadata and the two published advisory ranges. GitHub Actions passed the Cargo audits for the Rust workspace and Tauri dependency tree using generated lockfiles. Cargo is absent locally, neither Cargo root commits a lockfile, and the PR audit job did not build the Tauri application itself.
- The Lean GHSA was checked against the pinned `importGraph` commit and upstream fix commit. No Lean build or Lake update was performed because `lean`/`lake` are absent and the fixed dependency revision requires a newer `Cli` than the project's pinned toolchain.
- `.github/workflows/ci.yml` parsed as valid YAML with 14 jobs. Final `git diff --check` passed before submission; the reviewed changes are committed on the PR branch.

## Remaining risks and evidence gaps

- This repository has not received an external security audit or penetration test. The web application still relies on browser localStorage for an encrypted keystore; same-origin script compromise can access the encrypted file and interact with UI-held secrets. Browser extensions, compromised dependencies, phishing, weak passwords, and clipboard history remain relevant risks.
- The wallet can import keys controlling real EVM assets even though the GUI does not send transactions. It is an unaudited prototype; do not use it with valuable funds.
- Public balance RPC requests disclose the selected address to the provider(s) attempted. Optional proof submission discloses the signed statement/address to the site's server and configured chain RPC. A signed statement can be replayed; the server rechecks current on-chain eligibility, and the manuscript itself remains public.
- CSP still includes `unsafe-inline`; HTTPS/WSS connection and HTTPS image/font sources remain broad. Review and tighten these only with a tested nonce-based design and deployment-specific endpoint policy.
- The Lean/Mathlib dependency graph retains the pre-0.1.0 `import-graph` advisory described above. Its HTML graph executable is not part of this repository's `ToE` build or shipped output, but upgrading it safely requires a coordinated Lean/Mathlib/Cli change and a successful Lake build.
- Cargo/Tauri's direct advisory ranges are addressed by the `2.11.6` minimum pin, and the generated-lock `cargo audit` jobs passed in PR #85. No `Cargo.lock` is committed for either Cargo root, so the dependency graph is not reproducible across future resolutions; the Tauri application itself was not compiled by the audit job.
- No complete SPDX/license inventory or project-wide CycloneDX SBOM was produced. The completed vulnerability scans cover seven npm roots plus root/MCP Python dependencies; Cargo/Tauri has a direct-version remediation and a passing PR-time generated-lock audit, while Lean/Mathlib has a manually triaged exception. Python runtime requirements remain minimum-version ranges; third-party media/content provenance needs separate review.
- Repository history is shallow. See [`PROVENANCE.md`](PROVENANCE.md) for the historical rights and disclosure evidence limitations.
