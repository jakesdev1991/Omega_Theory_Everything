<!-- Copyright (c) 2025-2026 Jacob See. Licensed under Apache-2.0; see ../LICENSE. -->

# Lean 4 Proofs - Omega Theory

This directory contains the complete formal verification of the Omega Theory framework in Lean 4.

## Structure

### Core Foundations
- `ToE.lean` - Main entry point importing all volumes
- `OmegaAxioms.lean` - Foundational axioms (Hilbert space, modular flow, QFIM)
- `OmegaUnifiedFoundation.lean` - Unified mathematical foundation
- `OmegaDimensionalHierarchy.lean` - Dimensional hierarchy structure
- `OmegaProtocol.lean` - Protocol definitions and cross-volume theorems

### Physics Volumes (1-54)
Each volume formalizes a major domain of physics:

| Volume | Domain | Key Theorems |
|--------|--------|--------------|
| Vol01 | Classical Mechanics | Newton's laws, symplectic structure, energy conservation |
| Vol02 | Electromagnetism | Maxwell's equations, gauge invariance |
| Vol03 | Thermodynamics | Laws of thermodynamics, entropy |
| Vol04 | Quantum Mechanics | Schrödinger equation, uncertainty principle |
| Vol05 | General Relativity | Einstein field equations, geodesics |
| Vol06 | Quantum Field Theory | Path integrals, renormalization |
| Vol07 | Cosmology | FLRW, inflation |
| Vol08 | Black Hole Thermodynamics | Bekenstein-Hawking entropy |
| Vol09 | Holographic Principle | AdS/CFT correspondence |
| Vol10 | Standard Model | Particle content, interactions |
| Vol11 | Condensed Matter | Phase transitions, topology |
| Vol12 | Quantum Information | Entanglement, quantum channels |
| Vol13 | Quantum Gravity | Emergent spacetime |
| Vol14 | Dark Sector | Dark matter, dark energy |
| Vol15 | Early Universe | Big Bang nucleosynthesis |
| Vol16 | Non-Equilibrium Thermodynamics | Fluctuation theorems |
| Vol17 | Fluid Dynamics | Navier-Stokes, turbulence |
| Vol18 | Statistical Mechanics | Ensembles, partition functions |
| Vol19 | Complex Systems | Emergence, self-organization |
| Vol20 | Chaos Theory | Lyapunov exponents, strange attractors |
| Vol21 | Arrow of Time | Entropy gradient, irreversibility |
| Vol22 | ER=EPR | Entanglement = wormholes |
| Vol23 | Measurement Problem | Decoherence, Born rule |
| Vol24 | Non-Locality | Bell inequalities, CHSH |
| Vol25 | Black Hole Information | Page curve, unitarity |
| Vol26 | Network Theory | Graph dynamics, percolation |
| Vol27 | Consciousness | Integrated information theory |
| Vol28 | Evolutionary Algorithms | Fitness landscapes |
| Vol29 | Ecosystem Dynamics | Lotka-Volterra, stability |
| Vol30 | Planetary Systems | Orbital mechanics |
| Vol31 | Game Theory | Nash equilibria |
| Vol32 | Cybernetics | Control theory, feedback |
| Vol33 | AGI | Alignment, capability |
| Vol34 | Quantum Computing | Algorithms, error correction |
| Vol35 | Theory of Computation | Complexity classes |
| Vol36 | Topos Theory | Categorical logic |
| Vol37 | Morphogenesis | Pattern formation |
| Vol38 | Economics | General equilibrium |
| Vol39 | Societal Networks | Social dynamics |
| Vol40 | Fermi Paradox | Drake equation, filters |
| Vol41 | Post-Biological Evolution | Transhumanism |
| Vol42 | Stellar Engineering | Dyson spheres |
| Vol43 | Galactic Ecosystems | Galactic dynamics |
| Vol44 | Universal Expansion | Dark energy equation of state |
| Vol45 | Multiverse Theory | Landscape, measure problem |
| Vol46 | Simulation Hypothesis | Computational bounds |
| Vol47 | Transcendent Architectures | Hypercomputation |
| Vol48 | Extra Dimensions | Compactification |
| Vol49 | Quantum Reference Frames | Relational quantum mechanics |
| Vol50 | Ultimate Ensemble | Mathematical universe |
| Vol51 | Closed Timelike Curves | Causality constraints |
| Vol52 | Omega Point Theory | Final singularity |
| Vol53 | Universal Compiler | Self-compiling systems |
| Vol54 | Theory of Nothing | Existence from non-existence |

## Building

```bash
# Requires Lean 4 and Mathlib4
lake build
```

## Security-gate proofs and tactic search

`APPA_Context_Branching.lean` is a self-contained, axiom-free proof of the
parent non-tainting, child capability bounding, and declassification-gate
invariants. `CBwK_Budget_Pacer.lean` contains the three small budget-pacer
invariants used by the search harness.

`leandojo_tactic_harness.py` is the repository boundary for a LeanDojo REPL
integration. It performs bounded best-first tactic discovery and then invokes
`lake env lean` for the authoritative kernel check. It never reports a static
search result as a proof when Lean is unavailable:

```bash
python3 leandojo_tactic_harness.py --json
python3 leandojo_tactic_harness.py --require-lean
```

The legacy volume bridges now use explicit concrete model values or theorem
hypotheses instead of declaration-level axioms. `audit_axioms.py` enforces
that policy lexically, while Lean's kernel remains authoritative for the final
build. This distinction keeps a security proof from being presented as
stronger than its formal model.

## Companion Files

- **LaTeX versions**: `../latex_docs/` - Mathematical notation for each volume
- **Text versions**: `../txt_proofs/` - Plain text for accessibility
