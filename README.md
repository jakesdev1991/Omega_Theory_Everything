<!-- Copyright (c) 2025-2026 Jacob See. Split licensing: this repository mixes Apache-2.0 science materials with all-rights-reserved product materials — see LICENSE and docs/LICENSING.md. -->

# Omega Theory Everything

## Integrated systems blueprint

The economy described by this repository is named **the C.A.R.E. Economy** — care as the base layer of the entire system. Its vision is stated in [`whitepapers/care_economy_manifesto.md`](whitepapers/care_economy_manifesto.md) (the manifesto) and fully specified in [`whitepapers/care_economy_whitepaper.md`](whitepapers/care_economy_whitepaper.md) (the extensive whitepaper). The website carries both: `/vision`, `/manifesto`, and a `/whitepapers` reader.

The repository also includes [`tri_token_sovereign_economy_blueprint.md`](tri_token_sovereign_economy_blueprint.md), a research-and-systems specification for a proposed Tri-Token Sovereign Economy. It connects the Omega research program to Proof of Useful Work, C.A.R.E. privacy and arbitration, active-inference simulation, telemetry safeguards, asymmetric AMM/FINN clearing, governance, and staged implementation.

The blueprint is intentionally explicit about what is hypothetical, what requires independent validation, and what must not be used for automatic financial, medical, or public-benefit decisions.

The expanded whitepaper set is in [`whitepapers/`](whitepapers/): $OMEGA, legacy TOKAMAK domain research (not the Solana token identity), C.A.R.E./AMITY, and Lucifer–Hermes Omni-Bridge Prime.

The Omega MCP Hub — a sovereign MCP (Model Context Protocol) server exposing 22 tools across the sov/use/care/amity/omega planes over stdio JSON-RPC — is in [`mcp/`](mcp/) (see [`mcp/README_MCP_HUB.md`](mcp/README_MCP_HUB.md); its smoke-test evidence is committed alongside it).

The Omni-Bridge control-boundary implementation — the Hermes ⇄ Lucifer decision layer (APPA capability algebra, human-governed capabilities, hash-chained evidence ledger, T0–T5 sandbox tier selection), header-only C++23 with its acceptance suite — is in [`omni-bridge/`](omni-bridge/); implementation status, test evidence, and documented deviations are in [`omni-bridge/README.md`](omni-bridge/README.md). The CBwK shadow-price pacer (Lucifer–Hermes routing governor) is in [`cpp/`](cpp/).

A wallet-first web prototype is in [`app/`](app/). It performs real client-side wallet flows — BIP-39 mnemonic creation and import, MetaMask connection, Phantom connection, and an encrypted local Ethereum keystore — with keys never leaving the device. In the current build, **two release-day currencies are wired into the wallet flow**: `$OMEGA` on the EVM rail (current pilot: Ethereum Sepolia) and `TWC` on Solana. `AMITY` remains the separate Bitcoin/Lightning-with-Taproot workstream. The site runs no backend of its own and collects nothing. The same GUI is also served by the forward-facing website in [`web/`](web/) at `/omega-wallet/` (page: `/wallet`), where it is installable as an app and downloadable as a checksummed offline bundle (`/wallet/download`, `GET /api/wallet/download`); per-file SHA-256 hashes are published at `GET /api/wallet/manifest`, and native installers are produced from the Tauri wrapper in [`desktop/`](desktop/) by `.github/workflows/wallet-desktop.yml`.

The day-one launch plan for the token-gated release of the novel **Crucible: The Satoshi Protocol** under the C.A.R.E. Protocol ($OMEGA on EVM, **Token of the World Citizen (TWC)** on Solana, AMITY on Bitcoin/Lightning with Taproot Assets) is in [`launch/novel_day_one_plan.md`](launch/novel_day_one_plan.md). The Ethereum leg has a **Sepolia-only, valueless pilot suite** in [`evm/`](evm/): a fixed-supply `tOMEGA`, non-transferable vote escrow, bounded locking, OpenZeppelin Governor + TimelockController, and a claim-receipt gate. The Solana leg has a **Devnet-only, valueless `tTWC` pilot issuer/verifier** in [`solana/`](solana/): a standard fixed-supply SPL mint, null mint/freeze authorities, immutable Metaplex fungible metadata, and offline transaction-plan tests. The AMITY leg now has an expanded **testnet-only operator scaffold** in [`amity/`](amity/): strict testnet config parsing, canonical holder-challenge formatting, a non-broadcasting preflight, and a local status-manifest snapshot for the future Taproot Assets workstream. The web app also exposes a separate AMITY operator-status endpoint at `GET /api/amity/status` for visibility only. Neither pilot nor scaffold has been deployed, independently audited, or approved for mainnet/value-bearing use. The reading experience is implemented in [`web/`](web/): a Next.js site whose **current wallet-to-web unlock flow is wired for two currencies so far — `$OMEGA` and `TWC`** — with server-side signature verification plus independent on-chain rail checks once the local pilot manifests or equivalent environment variables are present. **Open release decision:** the manuscript the web app reads, `web/public/book/full-book.md`, is currently committed in plaintext, while the sealed-staging tooling in [`novel/`](novel/) (`seal.sh`; ciphertext + SHA-256 commitment) is not yet applied — sealing the book or accepting the public plaintext must be decided before launch day. The web app additionally hosts the **Economy Test Console** at `/testnet`: complete offline testing of the four-plane economy (readiness across all rails, the slice A–G scenario suite from `docs/tri-token-integration-v1.md`, a valueless faucet, an audited action runner, audit exports, and the Nostr integration surface reserved for the separately developed Nostr client), and an **App Store** at `/store`: a static frontend over a Nostr backplane whose directory is read from kinds 31990/30017 and whose Run buttons issue NIP-90 job requests to the sovereign mobile node in [`mobile-node/`](mobile-node/) (Termux/Droid@Debian daemon on the Pixel 8a), with verified results settleable into the TWC ledger.

An RCOD multi-scale optimizer-governor research prototype (Omega informational-geometry metrics as optimizer governors) with its label-noise-recovery benchmark is in [`rcod/`](rcod/) — the benchmark's honest verdict, including a negative result for the spec-as-written thresholds, is in [`rcod/RESULTS.md`](rcod/RESULTS.md).


Complete implementation of the Omega Theory framework - a unified physics model deriving spacetime, gravity, and cosmology from quantum information principles.

## Overview

This repository contains the full simulation suite for Omega Theory v4.0, implementing:

- **Sim1**: Emergent Geometry from quantum entanglement
- **Sim2**: Cosmological dynamics with dynamic scale factor
- **Sim3**: Renormalization group flow in spacetime
- **Sim4**: System evolution with v14 depletion mechanics
- **Sim5**: Emergent gravity as entropic force
- **Sim6**: v14 depletion dynamics
- **Sim7**: Checks for the macroscopic radial metric g_rr(Φ) (companion to `Omega_Theory_v4.0_Radial_Metric.md`)

## Theory Foundation

Based on the principle that spacetime geometry emerges from quantum entanglement entropy:

```
S_ent = A/4G  →  g_μν = δS_ent/δρ
```

Key papers:
- `Omega_Theory_v4.0_Technical.md` - Full technical specification
- `Omega_Theory_Laymans_Guide.md` - Accessible overview
- `Omega_Theory_v4.0_Radial_Metric.md` - Working notes toward the macroscopic metric g_rr(Φ) (exploratory; the open item of Technical §2.2)

## Formal Verification (Lean 4)

The mathematical framework is represented in Lean 4; the build is the source of truth for which statements are kernel-checked:

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
├── Vol01_ClassicalMechanics.lean     # through
└── Vol54_TheoryOfNothing.lean        # 54 physics volumes
```

The security-boundary proofs are kept separate from the physics model:
- `lean_proofs/APPA_Context_Branching.lean` — axiom-free parent/child isolation and declassification gate
- `lean_proofs/CBwK_Budget_Pacer.lean` — axiom-free budget-pacer invariants
- `lean_proofs/leandojo_tactic_harness.py` — bounded tactic search plus an explicit `lake env lean` kernel-check boundary

Each physics proof has companion files:
- **LaTeX**: `latex_docs/Vol##_*.tex` - Mathematical typesetting
- **Text**: `txt_proofs/Vol##_*.txt` - Plain text for accessibility

## Structure

```
Omega_Theory_Everything/
├── lean_proofs/              # 58 Lean 4 formalizations
│   ├── ToE.lean
│   ├── OmegaAxioms.lean
│   ├── OmegaUnifiedFoundation.lean
│   ├── OmegaDimensionalHierarchy.lean
│   ├── OmegaProtocol.lean
│   ├── Vol01_ClassicalMechanics.lean
│   └── ... (through Vol54)
├── latex_docs/               # 46 LaTeX documents (Vol09-Vol54)
│   ├── Vol09_HolographicPrinciple.tex
│   └── ... (through Vol54)
├── txt_proofs/               # 104 plain text companions
│   ├── Vol01_ClassicalMechanics.txt
│   └── ... (through Vol54, both Lean + LaTeX)
├── Sim1_Emergent_Geometry.py
├── Sim2_Cosmology.py
├── Sim3_Dynamic_Scale.py
├── Sim4_Evolution.py
├── Sim5_Emergent_Gravity.py
├── sim6_v14_depletion.py
├── Sim7_Radial_Metric.py
├── rcod/                     # RCOD optimizer governor (research prototype + benchmark)
├── cpp/                      # CBwK shadow-price pacer (Lucifer–Hermes routing governor, C++23)
├── omni-bridge/              # Omni-Bridge control boundary (Hermes ⇄ Lucifer), C++23 + host audit kit
├── mcp/                      # Omega MCP Hub server (sov/use/care/amity/omega planes, stdio JSON-RPC)
├── evm/                      # Sepolia-only, valueless tOMEGA governance + gate pilot
├── solana/                   # Devnet-only, valueless tTWC SPL mint issuer + verifier
├── amity/                    # Testnet-only AMITY Taproot Assets operator scaffold
├── web/                      # Next.js site: current $OMEGA + TWC unlock rails + gated novel chapters
├── novel/                    # Novel release plan + sealed-manuscript staging
├── launch/                   # Day-one launch plan documents
├── update_discovery.sh
├── requirements.txt
└── *.md
```

## Quick Start

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run simulations
python Sim1_Emergent_Geometry.py
python Sim2_Cosmology.py
# ... etc

# Build Lean 4 proofs (requires Lean 4 + Mathlib4)
cd lean_proofs && lake build
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

This repository uses a **split license**: the science is open; the product materials are proprietary.

| Materials | Terms |
|---|---|
| Simulations, Omega theory notes, Lean proofs, LaTeX and text companions, and listed tooling | Apache-2.0 — see [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) |
| `app/`, `rust/`, `evm/`, `solana/`, [`whitepapers/`](whitepapers/), and the [Tri-Token blueprint](tri_token_sovereign_economy_blueprint.md) | All rights reserved. Commercial licenses may be negotiated under a separate signed agreement with a percentage-based royalty; no rate or commercial permission is set by this repository. |

See [`LICENSE`](LICENSE) for exact path scope and [`docs/LICENSING.md`](docs/LICENSING.md) for important limits, including the prior MIT license and public-disclosure history. Copyright does not generally make underlying ideas, methods, or systems exclusive. Patent status is not claimed; see [`docs/PATENT-POSTURE.md`](docs/PATENT-POSTURE.md). Project-name notices are in [`docs/TRADEMARKS.md`](docs/TRADEMARKS.md).
