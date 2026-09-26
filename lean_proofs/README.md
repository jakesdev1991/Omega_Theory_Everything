<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../LICENSE and ../LICENSES/MIT.txt. -->

# Lean 4 Proofs - Omega Theory

This directory contains Lean 4 formalizations associated with Omega Theory:
conditional mathematical results, small executable models, and legacy
zero-information consistency checks. It is **not** a complete formal verification
or physical derivation of the theory. A kernel-checked statement can still be
vacuous in its chosen model.

See [PROOF_AUDIT.md](PROOF_AUDIT.md) for the source-review inventory, outstanding
limitations, and the latest strengthening pass (kernel validation pending).

## Structure

### Core Foundations
- `ToE.lean` - Main entry point importing all volumes
- `OmegaAxioms.lean` - Concrete singleton/zero-information legacy model
- `OmegaUnifiedFoundation.lean` - Unified mathematical foundation
- `OmegaDimensionalHierarchy.lean` - Dimensional hierarchy structure
- `OmegaProtocol.lean` - Protocol definitions and cross-volume theorems
- `InformationPhysics.lean` - Landauer–Einstein bridge: information-mass-energy
  equivalence, conservation of total state energy during mass↔information
  conversion, and positivity of information mass
- `DynamicPlanckScale.lean` - Dynamic Planck scale, built on
  `InformationPhysics` as a prerequisite module:
  information mass `m(I)` ⟹ local density `ρ = m(I)/V` ⟹ scale `ℓ_P(ρ)`.
  Proves that any positive density contracts the operational Planck length
  below its baseline `ℓ_P0` (strictly decreasing in `ρ`), and that any
  positive information in a positive volume therefore contracts it. The
  profile `ℓ_P(ρ) = ℓ_P0 / √(1 + ρ/ρ₀)` is a stated model assumption, not a
  derivation.
- `DynamicCODScale.lean` - Dynamic Planck scale driven directly by
  Chain-Overlap Density `Φ ∈ [0, 1]`: `ℓ_P(Φ) = ℓ_P0 · √(1 − Φ²)`. Proves the
  vacuum baseline `ℓ_P(0) = ℓ_P0`, the collapse `ℓ_P(1) = 0` at maximal
  overlap, `0 ≤ ℓ_P(Φ) ≤ ℓ_P0`, that any positive overlap contracts the scale
  strictly below baseline, and strict monotonicity of the contraction on
  `[0, 1]`. The profile is a stated model assumption, not a derivation.

### Physics Volumes (1-54)
The table lists intended domains and legacy topic labels, **not established
derivations**. Several volumes currently contain only toy models or structural
checks; consult the audit before interpreting a theorem name physically.

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
cd lean_proofs  # from the repository root
lake build ToE
```

## Proof-honesty policy

This corpus is maintained under an explicit honesty policy, enforced in CI:

1. **No unsound escapes.** `sorry`, `admit`, `native_decide`, `sorryAx`,
   `@[extern]`, and `implemented_by` are banned from sources (grep gate in
   `.github/workflows/lean-ci.yml`).
2. **No declaration-level axioms.** `audit_axioms.py --max 0` counts
   `axiom` commands; physical postulates are instead encoded as concrete
   definitions in `OmegaAxioms.lean` or as explicit theorem hypotheses.
3. **No dressed-up tautologies.** The Q-region model in `OmegaAxioms.lean`
   is intentionally degenerate (e.g. `vonNeumannEntropy _ = 0`), so
   statements like `vonNeumannEntropy R ≥ 0` are structural consistency
   checks of the formalization, *not* physics. Such declarations must carry
   the `bridge_*` naming convention and an honest docstring saying so;
   `audit_vacuity.py` fails CI if any other theorem states one of those
   tautologies. This lexical check is incomplete: it does not unfold aliases
   or discover contradictory hypotheses; a passing audit is not a semantic
   non-vacuity certificate. Legacy `*_Stmt` exports are separately inventoried
   by `legacy_statement_aliases.json` (96 outstanding entries); CI rejects
   new aliases and requires retired entries to be removed.
4. **No `Unit` stubs.** Existence proofs of the form `Nonempty X := ⟨()⟩`
   and Unit-valued type definitions (including parameterized types) inside the
   volumes are tracked by
   `audit_vacuity.py` and ratcheted so they can only decrease.

The ratchet values live in `.github/workflows/lean-ci.yml`; lowering them is
always welcome, raising them is not.

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

## Strengthened models and regression checks

- `LogCorrelationMetric.lean`: independent real-line correlation model, separation
  of distinct points, triangle inequality, and the correct direction of the
  multiplicative correlation condition. Does not replace or derive the legacy
  Q-region model.
- `RadialMetric.lean`: quantitative golden-bottleneck gap and unique equality
  case; exact chain-factorization error and necessary/sufficient condition.
- `CBwK_Budget_Pacer.lean`: state-changing spending and arbitrary finite cost
  sequences with an exact success criterion and conservation law. Reservation
  certificates now require their cost to equal the indexed request.
- `APPA_Context_Branching.lean`: executable fail-closed gate with acceptance and
  rejection equivalences; validator correctness remains an external assumption.
- `Vol18`, `Vol28`: unconditional probability normalization/strict bounds and
  exact optimal fitness of tournament selection.
- `Vol53`: an expressiveness limitation theorem: the current grammar cannot
  represent identity because every program is input-independent.
- `ProofRegression.lean`: boundary cases and non-degenerate witnesses, included
  in both the entry point and Lake roots.

Run the lightweight build-coverage tests without installing Lean:

```bash
python3 -m unittest discover -s lean_proofs -p 'test_*.py'
```

These tests and the lexical audits do not replace `lake build ToE`.

### Second pass: nontrivial witnesses and sharp domains

- Vol12 now uses two complex amplitudes, not `Unit`. A normalized pair with
  overlap `3/5` witnesses failure of the universal copying overlap condition.
- Vol49 now uses complex unit phases. Frame transformations can be nonidentity
  and have composition, round-trip and norm-preservation results.
- Vol25 now distinguishes joint from marginal entropy. Pure-state equality of
  marginals follows from an explicitly assumed Araki–Lieb bound; positive
  marginals are witnessed. This corrects the former, physically invalid
  “marginal entropies sum to zero” interpretation. `BHState` construction changed;
  `entropies_opposite` was removed and `unitarity_total` now concerns joint purity.
- Information mass now has linearity, additivity, strict monotonicity and a
  zero characterization. Conservation includes pre-existing information and
  is distinguished from nonnegative-output feasibility.
- Both scale profiles have domain-qualified converses/injectivity. The COD
  profile also has a proof that global injectivity would be false.
- Source-audit tests cover nested comments/strings, parameterized types, Unicode
  and attributes, long theorem declarations and per-export alias baselines.
  This pass reduced the remaining Unit-valued volume type definitions to 14;
  the fourth pass lowers that to 13.

These additions are still pending the pinned Lean kernel build. Python audit
success does not certify Lean elaboration or physical applicability.

### Third pass: positive-parameter models and actual limits

- Vol30 replaces zero central mass/zero period with `KeplerSystem`, requiring
  positive gravitational constant and mass. Period-square and ratio laws have
  explicit radius hypotheses; positivity, strict period growth, and the zero
  endpoint are derived from the chosen formula. This is not an ODE derivation.
- Vol38 replaces identically zero supply/demand with `LinearMarket`. Positive
  slopes/intercept give a unique positive equilibrium, positive traded quantity,
  and excess-demand signs on either side. The equilibrium is below choke price.
- Vol31 now defines pure Nash equilibrium using **all** unilateral deviations
  and classifies it uniquely as mutual defection. Pareto superiority of mutual
  cooperation is recorded separately from Nash stability.
- Vol32 now proves a genuine limit to zero for gain `0 ≤ g < 1`, rather than
  calling a one-step inequality convergence. Unit gain with nonzero initial
  state is an explicit counterexample to decay.

**API changes:** Vol30 period quantities/laws now take a `KeplerSystem`, and the
ratio theorem requires strictly positive radii. Vol38 quantities/laws take a
`LinearMarket`. Their `OmegaProtocol` wrappers were updated. The old standalone
`Vol30.CentralMass` zero constant was removed.

### Transitive kernel-axiom gate

```bash
python3 lean_proofs/audit_kernel.py
```

This command first runs `lake build ToE`, then asks Lean for the transitive
axioms of the 76 declarations in `kernel_audit_targets.json`. Only `propext`,
`Classical.choice`, and `Quot.sound` are permitted. Missing Lake, failed builds,
missing/duplicate reports, malformed output, and unapproved axioms all fail;
there is no static-success fallback. The scratch report lives under ignored
`.lake/` and is removed afterward. CI runs this gate after the main Lean build.

Coverage is explicitly **selected declarations**, not every theorem. Even a
successful axiom audit would not prove physical applicability or non-vacuity.
The report parser and failure paths have Python tests, but the live Lean audit
has not run here: it correctly exits 2 while Lake remains unavailable.

### Fourth pass: graph incidence and complete minimum conditions

- Vol26 now bundles `SimpleGraph (Fin n)`. Degrees count neighbors and edges
  count unordered edges independently; the degree-sum identity uses Mathlib's
  finite graph theorem. Added parity of odd-degree vertices, another-odd-vertex
  existence, a degree-floor bound, and empty/two-vertex witnesses.
- Vol39 reuses those graphs instead of zero constants. Its edge lower bound
  requires no isolated vertices and includes integer rounding. The singleton
  edgeless graph refutes the bound without that hypothesis. The empty graph is
  covered without assuming a vertex exists.
- Vol10 uses real scalar amplitudes, replacing another `Unit` carrier. The
  potential's signed minimizers are exactly ±v; zero energy determines +v only
  under nonnegative amplitude. This is not the full electroweak Higgs field.
- Vol11 completes the Landau quartic square, derives its lower bound and exact
  equality case, and classifies all minimizers when the quadratic coefficient
  is nonpositive and the quartic coefficient is positive. At criticality b=0,
  the potential is flat, so the former uniqueness wording was unjustified.

**API changes:** `Vol26.Graph` is no longer a list. `Vol39` count functions take
that graph, and its lower-bound theorems require `NoIsolatedVertices`. The
protocol social-network wrapper was updated. `Vol10.HiggsField` now aliases ℝ.
The source-audit ceiling is now **13** remaining Unit-valued volume types.

The kernel target list now covers **51** declarations. There are **31** Python
tests, including mocked command-order, timeout, failed-build, invalid-target and
scratch-cleanup tests. Mocked tests are not Lean runs. The live kernel gate still
fails closed because Lake is unavailable; all Lean changes remain pending the
pinned compiler build.

### Fifth pass: real derivatives and bounded settlement

- Vol16 now carries differentiable entropy/flux/production functions, with a
  `HasDerivAt` balance hypothesis. Isolation implies global monotonicity; strictly
  positive production implies strict growth. A positive, decreasing entropy
  history with entropy export shows that nonnegative production alone is not
  enough. Absolute entropy nonnegativity is propagated from explicit initial data.
- Vol17 uses an actual positive density trajectory and a derivative-level scalar
  continuity equation. Global incompressibility gives equal density at any two
  times. The pointwise converse requires positive density; vacuum is excluded.
  A decaying exponential density with divergence one is a nonconstant witness.
- Vol21 reuses Vol16's dynamics instead of assigning zero to cosmic entropy and
  its rate. Its arrow theorem is conditional, not a cosmological time derivation.
  The spurious identification with computational latency was removed.
- CBwK adds a capacity-bounded `Ledger`, reserving funds and then charging or
  releasing only existing reserves. Mixed finite traces preserve original
  capacity and never reduce spent funds. Excess settlement and a second positive
  release after full release are rejected. `Dimension.inv` now means actual
  allocated funds ≤ budget rather than budget ≤ itself.

**Scope:** the entropy balance and scalar material continuity laws are explicit
model hypotheses, not microscopic thermodynamics or a spatial fluid solution.
The ledger is aggregate sequential accounting, not ticket authentication,
reservation ownership, concurrency control, persistent storage or anti-rollback.
The old headroom-only `settle` remains a documented arithmetic compatibility
helper; new bounded-accounting integrations should use `settleLedger`.

**API changes:** Vol16 quantities/theorems take `EntropyProcess`; Vol17 takes
`FluidTrajectory`; Vol21 takes `CosmicHistory` with isolation/rate conditions.
`qregion_at` and `time_as_computational_latency` were removed from Vol21.
`Dimension` now requires an `allocated` field and its corresponding bound.
Protocol callers were updated, and the tactic harness still uses the same
invariant projection proof.

At the end of pass five there were **35** Python/tooling tests, including local import-cycle and
missing-module checks, and **67** selected declarations in the kernel audit.
The actual Lean build remains pending: the live kernel gate still fails closed
when Lake is absent. Source tests and written Lean regressions are not a
substitute for compiler execution.

### Sixth pass: non-ideal erasure and batch affordability

`InformationPhysics.Erasure` adds nonnegative excess energy to the chosen
Landauer-limit cost. Single-erasure equality holds exactly at zero excess;
zero total cost requires both zero information and zero excess. An explicit
positive-excess witness refutes universal saturation.

Finite batches have an exact ideal-plus-excess decomposition. A batch saturates
the bound iff every member has zero excess, so nonnegative overhead cannot
cancel. Paying a concatenation agrees with threading the remaining mass through
its two parts. The remaining mass is nonnegative iff the entire cost fits the
initial mass-energy budget; complete affordability also implies affordability
of every prefix. Conservation includes spent energy, not only surviving mass.

Twelve additional Lean regression examples cover ideal/overhead cases, empty
batches, exact exhaustion, and the failure of separate per-job checks to ensure
joint affordability. They are included in the build but **have not run locally**.
The 35 Python/tooling tests still pass, and the selected kernel inventory now
contains **76 declarations**; the live gate exits 2 because Lake is absent.

These are conditional accounting results in one fixed environment, not a
microscopic derivation of Landauer's bound or an erasure/conversion mechanism.
Real-valued information quantities need not be integer bitstrings. The old
ideal information-mass and density/scale APIs remain unchanged and must not be
interpreted as actual storage mass or general irreversible erasure cost.
