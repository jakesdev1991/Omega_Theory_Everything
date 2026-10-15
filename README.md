<!-- Copyright (c) 2025-2026 Jacob See. Split licensing: science is MIT, the public novel is CC BY 4.0, product software is PolyForm Noncommercial 1.0.0, and crypto/C.A.R.E. materials are read-only — see LICENSE and docs/LICENSING.md. -->

# Omega Theory Everything

> **Scientific status (2026-10-11): exploratory research program, not a completed or empirically validated Theory of Everything.** The information-geometric ideas are hypotheses under development. The operational map from Q-region mutual information to \(\Phi\), the macroscopic metric, a covariant dynamics derived from the Q-region model that recovers general relativity, and independent empirical tests remain open. One minimally coupled scalar action has a conditional GR branch; a curvature-coupled gravitational-scalar branch is the provisional next-test model and has a linear zero-mode diagnostic but no nonlinear solution. Neither action nor branch is derived from Q-regions. Lean proofs establish only their stated mathematical consequences under formal assumptions; simulations establish only the behavior of the coded models. Neither constitutes physical validation.
>
> Start with [`docs/research/QREGION_MARKOV_MODEL.md`](docs/research/QREGION_MARKOV_MODEL.md) for the analytically tractable Markov benchmark, [`docs/research/ENTANGLED_TFIM_BENCHMARK.md`](docs/research/ENTANGLED_TFIM_BENCHMARK.md) for a genuinely entangled finite-chain check, [`docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md`](docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md) for the separate covariant branches and empirical-test gate, and [`docs/research/GAUSS_BONNET_SCALARIZATION.md`](docs/research/GAUSS_BONNET_SCALARIZATION.md) for the linear gravitational-scalar zero-mode calculation. The [`Omega_Theory_v4.0_Radial_Metric.md`](Omega_Theory_v4.0_Radial_Metric.md) file is exploratory working material, not a derivation of a complete gravitational theory.

## Related systems prototypes

The agent, web, token, and proposed economy projects below are separate engineering workstreams. Their existence, tests, and branding do not validate the physics hypotheses described in this repository.

The economy described by this repository is named **the C.A.R.E. Economy** — **Call About Resuscitating Everyone.** Care is the base layer of the entire system. Its vision is stated in [`docs/sovereign_economy_manifesto.md`](docs/sovereign_economy_manifesto.md) (the manifesto) and fully specified in [`whitepapers/care_economy_whitepaper.md`](whitepapers/care_economy_whitepaper.md) (the extensive whitepaper). The website carries both: `/vision`, `/manifesto`, and a `/whitepapers` reader. The novel's canonical title is **Crucible: The Satoshi Protocol**.

The repository also includes [`tri_token_sovereign_economy_blueprint.md`](tri_token_sovereign_economy_blueprint.md), a research-and-systems specification for a proposed Tri-Token Sovereign Economy. It connects the Omega research program to Proof of Useful Work, C.A.R.E. privacy and arbitration, active-inference simulation, telemetry safeguards, asymmetric AMM/FINN clearing, governance, and staged implementation.

The blueprint is intentionally explicit about what is hypothetical, what requires independent validation, and what must not be used for automatic financial, medical, or public-benefit decisions.

The expanded whitepaper set is in [`whitepapers/`](whitepapers/): $OMEGA, legacy TOKAMAK domain research (not the Solana token identity), C.A.R.E./AMITY, and Lucifer–Hermes Omni-Bridge Prime.

The Omega MCP Hub — a sovereign MCP (Model Context Protocol) server exposing 22 tools across the sov/use/care/amity/omega planes over stdio JSON-RPC — is in [`mcp/`](mcp/) (see [`mcp/README_MCP_HUB.md`](mcp/README_MCP_HUB.md); its smoke-test evidence is committed alongside it).

The Omni-Bridge control-boundary implementation — the Hermes ⇄ Lucifer decision layer (APPA capability algebra, human-governed capabilities, hash-chained evidence ledger, T0–T5 sandbox tier selection), header-only C++23 with its acceptance suite — is in [`omni-bridge/`](omni-bridge/); implementation status, test evidence, and documented deviations are in [`omni-bridge/README.md`](omni-bridge/README.md). The CBwK shadow-price pacer (Lucifer–Hermes routing governor) is in [`cpp/`](cpp/).

A wallet-first web prototype is in [`app/`](app/). It performs real client-side wallet flows — BIP-39 mnemonic creation and import, MetaMask connection, Phantom connection, and an encrypted local Ethereum keystore — with keys never leaving the device. In the current build, **two pilot currencies power an optional reader-proof demo**: `$OMEGA` on the EVM rail (current pilot: Ethereum Sepolia) and `TWC` on Solana. `AMITY` remains the separate Bitcoin/Lightning-with-Taproot workstream. The site runs no backend of its own and collects nothing. The same GUI is also served by the forward-facing website in [`web/`](web/) at `/omega-wallet/` (page: `/wallet`), where it is installable as an app and downloadable as a checksummed offline bundle (`/wallet/download`, `GET /api/wallet/download`); per-file SHA-256 hashes are published at `GET /api/wallet/manifest`, and native installers are produced from the Tauri wrapper in [`desktop/`](desktop/) by `.github/workflows/wallet-desktop.yml`.

The current release plan for the public novel **Crucible: The Satoshi Protocol** under the C.A.R.E. Protocol ($OMEGA on EVM, **Token of the World Citizen (TWC)** on Solana, AMITY on Bitcoin/Lightning with Taproot Assets) is in [`launch/novel_day_one_plan.md`](launch/novel_day_one_plan.md). The Ethereum leg has a **Sepolia-only, valueless pilot suite** in [`evm/`](evm/): a fixed-supply `tOMEGA`, non-transferable vote escrow, bounded locking, OpenZeppelin Governor + TimelockController, and a claim-receipt gate. The Solana leg has a **Devnet-only, valueless `tTWC` pilot issuer/verifier** in [`solana/`](solana/): a standard fixed-supply SPL mint, null mint/freeze authorities, immutable Metaplex fungible metadata, and offline transaction-plan tests. The AMITY leg now has an expanded **testnet-only operator scaffold** in [`amity/`](amity/): strict testnet config parsing, canonical holder-challenge formatting, a non-broadcasting preflight, and a local status-manifest snapshot for the future Taproot Assets workstream. The web app also exposes a separate AMITY operator-status endpoint at `GET /api/amity/status` for visibility only. Neither pilot nor scaffold has been deployed, independently audited, or approved for mainnet/value-bearing use. The web reader experience is implemented in [`web/`](web/): a Next.js site whose **optional wallet-to-web proof flow is wired for two currencies so far — `$OMEGA` and `TWC`** — with server-side signature verification plus independent on-chain rail checks once the local pilot manifests or equivalent environment variables are present. The current manuscript, [`web/public/book/full-book.md`](web/public/book/full-book.md), is public and licensed under CC BY 4.0: anyone may read, share, and adapt the original text with attribution, subject to the license and third-party-rights limitations in [`web/public/book/LICENSE.md`](web/public/book/LICENSE.md). No token or wallet is required to access it. The site retains an optional proof-verified reader flow for the $OMEGA and TWC pilots; that flow does not restrict the licensed text. The legacy `novel/seal.sh` helper is not applied to this public draft. The web app additionally hosts the **Economy Test Console** at `/testnet`: complete offline testing of the four-plane economy (readiness across all rails, the slice A–G scenario suite from `docs/tri-token-integration-v1.md`, a valueless faucet, an audited action runner, audit exports, and the Nostr integration surface reserved for the separately developed Nostr client), and an **App Store** at `/store`: a static frontend over a Nostr backplane whose directory is read from kinds 31990/30017 and whose Run buttons issue NIP-90 job requests to the sovereign mobile node in [`mobile-node/`](mobile-node/) (Termux/Droid@Debian daemon on the Pixel 8a), with verified results settleable into the TWC ledger.

An RCOD multi-scale optimizer-governor research prototype (Omega informational-geometry metrics as optimizer governors) with its label-noise-recovery benchmark is in [`rcod/`](rcod/) — the benchmark's honest verdict, including a negative result for the spec-as-written thresholds, is in [`rcod/RESULTS.md`](rcod/RESULTS.md).


## Physics research program and simulation suite

The repository contains an exploratory information-geometric program and a set of legacy Omega v4.0 simulations. The scripts exercise chosen mathematical models; they do not derive spacetime, gravity, or cosmology from quantum information, and they have not established agreement with independent physical data. Five of the seven historical Sim entries are executable scripts; the other two are manuscripts that were previously stored with a `.py` extension and described as runnable, which they never were:

- **Sim2**: phenomenological information-depletion ODE with an exponential scale factor — `Sim2_Cosmology.py` (executable; not a Friedmann derivation)
- **Sim3**: chosen scale-profile and disformal-causality toy calculations — `Sim3_Dynamic_Scale.py` (executable; not a physical Planck-scale prediction)
- **Sim5**: toy entropic-force model — `Sim5_Emergent_Gravity.py` (executable; not a derivation of gravity)
- **Sim6**: phenomenological v14 depletion model — `sim6_v14_depletion.py` (executable; calibrates γ to reproduce the chosen H₀ = 73.5 km/s/Mpc, so that value is not an independent prediction)
- **Sim7**: numerical checks of candidate radial-metric identities — `Sim7_Radial_Metric.py` (executable; companion to exploratory notes in `Omega_Theory_v4.0_Radial_Metric.md`)
- **Sim1**: historical MDS exercise on a prescribed exponential kernel (not a physical distance derivation; damaged code listing): [`docs/manuscripts/sim1_emergent_geometry.md`](docs/manuscripts/sim1_emergent_geometry.md)
- **Sim4**: historical LTEE-inspired toy bundle (not an LTEE reproduction or empirical evidence; see status warning): [`docs/manuscripts/sim4_evolution.md`](docs/manuscripts/sim4_evolution.md)

## Scientific status and open problems

Omega explores the hypothesis that information-theoretic structure may help characterize geometry. No equation in this repository currently derives the spacetime metric from mutual information. In particular, the former shorthand `g_μν = δS_ent/δρ` is not a well-formed tensor equation and has been removed. The logarithmic-distance proposal is conditional on a multiplicative correlation kernel; pairwise mutual information itself generally does not satisfy that composition law.

The sequenced plan for tightening every claim and derivation in the repository, from the core notes through the LaTeX volumes and Lean headers, with each step's exit gate, is in [`docs/research/CLAIM_TIGHTENING_ROADMAP.md`](docs/research/CLAIM_TIGHTENING_ROADMAP.md). The derivations in the core notes are re-checked by [`research/verify_derivations.py`](research/verify_derivations.py).

The immediate research results and limits are documented here:

- [`Omega_Theory_v4.0_Technical.md`](Omega_Theory_v4.0_Technical.md) — exploratory technical note; its scalar action is a candidate model, not a unique derivation.
- [`Omega_Theory_Laymans_Guide.md`](Omega_Theory_Laymans_Guide.md) — plain-language account with hypotheses and established results separated.
- [`Omega_Theory_v4.0_Radial_Metric.md`](Omega_Theory_v4.0_Radial_Metric.md) — conditional radial identities and the still-open Q-region-to-metric bridge.
- [`docs/research/QREGION_MARKOV_MODEL.md`](docs/research/QREGION_MARKOV_MODEL.md) — a classical-diagonal Q-region toy model; its derived correlation decay does not recover the required Schwarzschild radial coefficient.
- [`docs/research/ENTANGLED_TFIM_BENCHMARK.md`](docs/research/ENTANGLED_TFIM_BENCHMARK.md) — an exact finite-chain diagnostic with positive pairwise negativity; its fitted decay-rate trend also disagrees with the Schwarzschild-required trend.
- [`docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md`](docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md) — separate matter-scalar and gravitational-scalar branches, conditional GR limits, and the empirical-test gate.
- [`docs/research/GAUSS_BONNET_SCALARIZATION.md`](docs/research/GAUSS_BONNET_SCALARIZATION.md) — a linear scalar–Gauss–Bonnet zero-mode calculation; it is not a nonlinear black-hole solution or Q-region derivation.

The Lean corpus should be read with [`lean_proofs/PROOF_AUDIT.md`](lean_proofs/PROOF_AUDIT.md): kernel checking verifies formal statements under their assumptions, not their physical interpretation or empirical truth.

## Formal Verification (Lean 4)

The formal corpus contains a mixture of model definitions, conditional lemmas, and research results; it is not a formal proof of the physical theory as a whole. The build is the source of truth for which individual statements are kernel-checked, and the proof audit explains the limits of that evidence:

```
lean_proofs/
├── ToE.lean                          # Main entry point
├── OmegaAxioms.lean                  # Concrete minimal model and derived lemmas
├── OmegaUnifiedFoundation.lean       # Unified foundation
├── OmegaDimensionalHierarchy.lean    # Dimensional hierarchy
├── OmegaProtocol.lean                # Cross-volume theorems
├── InformationPhysics.lean           # Landauer-Einstein information mass
├── DynamicPlanckScale.lean           # density-driven Planck scale
├── DynamicCODScale.lean              # COD-driven Planck scale, l_P(Phi) = l_P0 sqrt(1 - Phi^2)
├── RadialMetric.lean                 # golden-ratio bottleneck, finite-chain two-factor law, Phi-form Schwarzschild identities
├── RadialDilation.lean              # lP-only obstruction (g_RR<=1, negative quasi-local mass) + required-kappa dilation reconciliation
├── Vol01_ClassicalMechanics.lean     # through
└── Vol54_TheoryOfNothing.lean        # 54 physics volumes
```

The security-boundary proofs are kept separate from the physics model:
- `lean_proofs/APPA_Context_Branching.lean` — axiom-free parent/child isolation and declassification gate
- `lean_proofs/CBwK_Budget_Pacer.lean` — axiom-free budget-pacer invariants
- `lean_proofs/leandojo_tactic_harness.py` — bounded tactic search plus an explicit `lake env lean` kernel-check boundary

Companion files for the physics volumes:
- **Text**: `txt_proofs/Vol##_*.txt` - plain-text copies of the Lean sources; all 54 volumes have one.
- **LaTeX**: `latex_docs/Vol##_*.tex` - write-ups with layman's summaries. Only 41 of the 54 volumes have one (Vol01–08 and Vol36–40 do not yet). These write-ups use claim language that is being tightened under [`docs/research/CLAIM_TIGHTENING_ROADMAP.md`](docs/research/CLAIM_TIGHTENING_ROADMAP.md); they are not evidence beyond the Lean statements they describe.

## Structure

```
Omega_Theory_Everything/
├── lean_proofs/              # 69 Lean 4 modules (54 physics volumes + foundations)
│   ├── ToE.lean
│   ├── OmegaAxioms.lean
│   ├── OmegaUnifiedFoundation.lean
│   ├── OmegaDimensionalHierarchy.lean
│   ├── OmegaProtocol.lean
│   ├── Vol01_ClassicalMechanics.lean
│   └── ... (through Vol54)
├── latex_docs/               # 41 LaTeX documents (Vol09-Vol54 with gaps: Vol01-08 and Vol36-40 have none)
│   ├── Vol09_HolographicPrinciple.tex
│   └── ... (through Vol54)
├── txt_proofs/               # 59 plain-text companions (54 volumes + 5 core)
│   ├── Vol01_ClassicalMechanics.txt
│   └── ... (companions for the core modules and Vol01-Vol54)
├── Sim2_Cosmology.py
├── Sim3_Dynamic_Scale.py
├── Sim5_Emergent_Gravity.py
├── sim6_v14_depletion.py
├── Sim7_Radial_Metric.py
├── tests/                    # Simulation and repository-hygiene regression tests
├── docs/manuscripts/         # Sim1 and Sim4 (prose; not executable)
├── docs/research/            # Q-region benchmarks, covariant branches, and scalarization diagnostic
├── research/                 # Reproducible Python research checks
├── rcod/                     # RCOD optimizer governor (research prototype + benchmark)
├── cpp/                      # CBwK shadow-price pacer (Lucifer–Hermes routing governor, C++23)
├── omni-bridge/              # Omni-Bridge control boundary (Hermes ⇄ Lucifer), C++23 + host audit kit
├── mcp/                      # Omega MCP Hub server (sov/use/care/amity/omega planes, stdio JSON-RPC)
├── evm/                      # Sepolia-only, valueless tOMEGA governance + gate pilot
├── solana/                   # Devnet-only, valueless tTWC SPL mint issuer + verifier
├── amity/                    # Testnet-only AMITY Taproot Assets operator scaffold
├── web/                      # Next.js site: public CC BY 4.0 novel + optional $OMEGA/TWC reader-proof flow
├── novel/                    # Public novel notes + future-material staging helper
├── launch/                   # Day-one launch plan documents
├── conjecture_pilot/         # Measurement-first conjecture generator + pre-registration
├── requirements.txt
└── *.md
```

## Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the executable simulations (matplotlib must be headless in CI: MPLBACKEND=Agg)
python Sim2_Cosmology.py
python Sim3_Dynamic_Scale.py
python Sim5_Emergent_Gravity.py
python sim6_v14_depletion.py
python Sim7_Radial_Metric.py --no-plots

# Reproduce the diagonal Markov-chain Q-region benchmark
python research/qregion_markov.py
python -m pytest tests/test_qregion_markov.py -q

# Reproduce the genuinely entangled finite Ising-chain diagnostic
python research/quantum_ising_chain.py
python -m pytest tests/test_quantum_ising_chain.py -q

# Reproduce the curvature-coupled scalar zero-mode diagnostic
python research/gauss_bonnet_scalarization.py
python -m pytest tests/test_gauss_bonnet_scalarization.py -q
python research/verify_derivations.py
python research/verify_covariant.py
python research/verify_tfim.py
python research/check_latex_citations.py
python research/check_latex_structure.py
python research/claim_scan.py

# Or run every local check at once (needs requirements.txt for ruff/mypy/pip-audit)
python research/consistency_gate.py

# Validate the public launch-control ledger and generated dependency inventory.
# A valid ledger is deliberately not launch approval; --require mainnet exits non-zero today.
python launch/check_readiness.py --validate
python tools/generate_dependency_inventory.py --check
python tools/check_dependabot_coverage.py
pip-audit -r requirements.txt --strict
# MCP declares Python >=3.12; run this with that interpreter/environment.
pip-audit mcp --strict
node tools/audit_npm_production.mjs
node tools/audit_npm_development.mjs

# Sim1 and Sim4 are historical manuscripts, not executable scripts; read their status notes in docs/manuscripts/

# Build Lean 4 proofs (requires Lean 4 + Mathlib4)
cd lean_proofs && lake build

# The cited-name check is CI-only: it needs a pinned mathlib4 checkout and the
# toolchain's own sources, which no local check here can download.
#   python3 lean_proofs/check_mathlib_names.py \
#       --mathlib <mathlib4> --lean <lean-prefix>/src/lean --max 0
```

## Development

```bash
# Python linting
ruff check .
mypy .

# Lean 4 build
cd lean_proofs && lake build

# Test
pytest -v
```

## License and intellectual property

This repository uses a **split license**: scientific materials are MIT, the public novel text is CC BY 4.0, product software is PolyForm Noncommercial, and the crypto materials are read-only.

| Materials | Terms |
|---|---|
| Simulations, Omega theory notes, Lean proofs, LaTeX and text companions, `rcod/`, `research/`, `docs/research/` | **MIT** — see [`LICENSES/MIT.txt`](LICENSES/MIT.txt) |
| `web/public/book/full-book.md` and its public license notice | **CC BY 4.0** — sharing and adaptation, including commercial use, allowed with attribution. Third-party rights are not included. See [`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt) |
| `web/` (website, wallet hub, App Store storefront; except the public book files and store legal texts), `mobile-node/`, `app/`, `desktop/`, `omni-bridge/`, `cpp/`, `mcp/`, `launch/`, `novel/` (release/staging tools), `docs/store/` | **PolyForm Noncommercial 1.0.0** — free for noncommercial use; commercial use requires a separate paid, signed written license with percentage-based compensation. See [`LICENSES/PolyForm-Noncommercial-1.0.0.txt`](LICENSES/PolyForm-Noncommercial-1.0.0.txt) |
| `rust/`, `evm/`, `solana/`, `amity/`, [`whitepapers/`](whitepapers/), the [Tri-Token blueprint](tri_token_sovereign_economy_blueprint.md), and the C.A.R.E. docs | **Read-only, no license** — view-only in this repository; no copying, reproduction, or derivative works without written permission. See [`LICENSES/LicenseRef-Omega-ReadOnly.txt`](LICENSES/LicenseRef-Omega-ReadOnly.txt) |

See [`LICENSE`](LICENSE) for exact path scope and [`docs/LICENSING.md`](docs/LICENSING.md) for important limits, including the prior MIT license and public-disclosure history. Copyright does not generally make underlying ideas, methods, or systems exclusive. Patent status is not claimed; see [`docs/PATENT-POSTURE.md`](docs/PATENT-POSTURE.md). Project-name notices are in [`docs/TRADEMARKS.md`](docs/TRADEMARKS.md).
