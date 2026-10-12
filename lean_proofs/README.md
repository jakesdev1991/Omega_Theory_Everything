<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../LICENSE and ../LICENSES/MIT.txt. -->

# Lean 4 Proofs - Omega Theory

This directory contains Lean 4 formalizations associated with Omega Theory:
conditional mathematical results, small executable models, and legacy
zero-information consistency checks. It is **not** a complete formal verification
or physical derivation of the theory. A kernel-checked statement can still be
vacuous in its chosen model.

See [PROOF_AUDIT.md](PROOF_AUDIT.md) for the source-review inventory, outstanding
limitations, and compiler-validation results.

**Integrated CI status (2026-10-12):** the thirteenth pass (PR #77) and the
claim-tightening passes (PR #78, PR #79) are merged into `main`. `lake build ToE`
and the transitive kernel-axiom gate — **260** selected declarations, zero
unexpected axioms — are green on `main`
([Lean CI run 38119532226](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38119532226),
the merge of PR #79), were re-confirmed on
[PR #83](https://github.com/jakesdev1991/Omega_Theory_Everything/pull/83)
(runs 38251369698 and, at the head with the rename pass, 38254475070), and are now green
on `main` at the merge commit of PR #83, `a764f34`
([Lean CI run 38255776907](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38255776907),
[CI run 38255777137](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38255777137)).
Reproducible
locally without a toolchain: 107 Python/tooling
tests, `audit_axioms --max 0` → 0, all eight `audit_vacuity` ratchets and the
alias baseline at 0, the comment-parity gate across 69 files, 68
byte-identical `txt_proofs/` mirrors, and the Phase 6 statement audit — 163 rows
for the 161 physics-sounding volume declarations, with the 33 `carried` rows
capped by the audit file's own ratchet and the 16 names the rename pass retired
recorded as `previous_name` on their rows. Lean/mathlib remain pinned to
v4.32.0.
Kernel acceptance cannot be reproduced here: the toolchain and the mathlib olean
cache are served from release-asset hosts that are unreachable in this
environment, so `lake build` is CI-only. The same is true of the cited-name
check (`check_mathlib_names.py`), which needs the pinned mathlib and Lean
sources and therefore runs, like the build, only in Lean CI. The
[adversarial audit](ADVERSARIAL_AUDIT.md) is dated 2026-09-26; its executive
summary is a pre-remediation record, and its status note carries the counts
above. Earlier pass-status notes below are historical.

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
- `RadialMetric.lean` - Macroscopic radial metric in `Φ`: the golden-ratio
  bottleneck of Sim3's disformal causality band, the finite-chain two-factor
  law (proper length is scale × weight per link), and the `Φ`-form
  Schwarzschild identities (areal radius endpoints, lapse–conformal
  identity). Companion to `Omega_Theory_v4.0_Radial_Metric.md`.
- `RadialDilation.lean` - Obstruction and reconciliation for the radial
  metric, built on `DynamicCODScale`: `ℓ_P` alone gives
  `g_RR = (1−Φ²)² ≤ 1` with negative quasi-local mass and no linear term
  (`M_ADM = 0` content), while the required `κ/κ₀ = (1+Φ)²/√(1−Φ²)`
  reproduces the isotropic conformal factor exactly, outruns the inverse
  rendering scale, and yields areal dilation with exact lapse reciprocity.
  The `Φ`-form Schwarzschild dictionary is identified, not derived.

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
   tautologies. Hypotheses unsatisfiable in the model (e.g. `Φ = 1`, positive
   information in the zero model) are retired rather than stated; the
   non-degenerate restatements live over explicit network/fluid/entropy
   structures. This lexical check is incomplete: it does not unfold aliases
   or discover contradictory hypotheses; a passing audit is not a semantic
   non-vacuity certificate. Legacy `*_Stmt` exports are separately inventoried
   by `legacy_statement_aliases.json` (now empty); CI rejects any new alias.
   The same gate ratchets three further defect classes at zero: decorative
   hypotheses (unused in both statement and proof,
   `--max-decorative-hypotheses`), hypotheses whose type is a closed arithmetic
   truth such as `0 ≤ 0` (`--max-trivial-hypotheses`), and identity proofs
   whose conclusion restates a hypothesis (`--max-identity-proofs`; both
   tactic-mode `:= by exact h` and term-mode `:= h` bodies are detected).
4. **No `Unit` stubs.** Existence proofs of the form `Nonempty X := ⟨()⟩`
   and Unit-valued type definitions (including parameterized types) inside the
   volumes are tracked by
   `audit_vacuity.py` and ratcheted so they can only decrease.
5. **Comment edits cannot change a theorem.** The label-replacement pass
   (roadmap Phase 5) touched comments and docstrings only.
   `check_declaration_parity.py` reduces every `.lean` file to a canonical
   *code projection* — nested block comments, docstrings and line comments
   removed, string and character literals kept, blank lines and trailing
   whitespace dropped — and hashes it into `declaration_parity.json`. A
   comment-only edit leaves every digest untouched; a change to a name,
   statement, or proof body fails the gate until the baseline is regenerated:

   ```bash
   python3 check_declaration_parity.py            # gate (CI)
   python3 check_declaration_parity.py --update   # after a deliberate code change
   python3 check_declaration_parity.py --against main   # prove a branch is comment-only
   ```

   It is a lexical check, not a kernel check: it cannot see semantic
   equivalence, so two spellings of one proof give two digests.
6. **Plain-text mirrors track the sources.** `txt_proofs/` is generated, not
   hand-edited:

   ```bash
   python3 sync_txt_mirrors.py          # gate (CI)
   python3 sync_txt_mirrors.py --write  # regenerate after editing a module
   ```

   Every module except `lakefile.lean` has a byte-identical companion; orphans
   are reported but never deleted.
7. **No axiom-*named* declarations.** The corpus contains zero Lean `axiom`
   primitives, so a theorem or field *named* `axiom_*`/`*_axiom` misrepresents
   its content. `audit_vacuity.py --max-axiom-names 0` (CI) rejects such
   names; model laws carried as structure data are named `law_*` with an
   honest docstring, and witness theorems are named `*_witness`/`*_exists`.
8. **Cited names exist in the pinned sources.** `check_mathlib_names.py`
   indexes every declaration in a Mathlib4 checkout plus the Lean core sources,
   then reports qualified names used by the corpus that appear in neither —
   `Finset.sum_range_succs` where the real lemma is `Finset.sum_range_succ`,
   or a structure field that was renamed. It is a ratchet: `--max N` fails when
   the count exceeds `N`, and CI runs it at `0`:

   ```bash
   python3 check_mathlib_names.py \
       --mathlib /path/to/mathlib4 --lean /path/to/lean4/src --max 0
   ```

   It is a *lexical* net with known blind spots — no `open` scoping, no
   notation, no type-class resolution, and it cannot see that a name exists but
   needs an import the file lacks — so a clean run is not well-typedness. It
   needs the pinned sources (the mathlib checkout and the toolchain's own
   `src/lean`), which exist only in CI, so unlike the rest of this list it
   cannot run in the offline consistency gate; `.github/workflows/lean-ci.yml`
   invokes it after the Lean build, and it exits 2 rather than reporting a
   clean run when a source tree yields no declarations.
9. **The physics volumes have a per-statement audit.** Phase 6 of
   `../docs/research/CLAIM_TIGHTENING_ROADMAP.md` requires a recorded judgement
   for every theorem or lemma in `Vol*.lean` whose name carries a
   physics-sounding word (a documented lexicon, matched after splitting names on
   `_`, `.` and camelCase). `check_statement_audit.py` is the gate and
   `statement_audit.json` is the record: **163 rows** for the current corpus
   (161 candidates plus two rows for renamed declarations the lexicon no longer
   matches), each with a disposition and a note —

   ```bash
   python3 check_statement_audit.py            # gate (also in Lean CI)
   python3 check_statement_audit.py --review   # the candidates, with statements
   python3 check_statement_audit.py --list     # one line per row
   ```

   A row is **witness** (names a declared, non-degenerate object or instance;
   73 rows), **conditional** (quotes a hypothesis that occurs verbatim in the
   statement; 43), **renamed** (records a name retired earlier, which must
   survive nowhere once comments are masked; 14), or **carried** (true only from
   data the model carries — a structure field, a postulate, a placeholder or
   constant definition, a definitional restatement, or a degenerate instance —
   and names that declaration; 33, the count the file's `carried_cap` ratchets,
   which may only be lowered). Any row may also carry a `previous_name`: the
   Phase 6 rename pass renamed sixteen declarations whose names asserted results
   their statements do not contain, kept each row's disposition, and recorded
   the retired name, which must then survive nowhere in the corpus — the match is
   on identifier boundaries, so `zeroCurvature_bianchi_identity` is not a
   surviving `bianchi_identity`. The fourth disposition is a deliberate addition
   to the roadmap's sketch: the three-way split has no honest place for
   `theorem photonMass_eq_zero : PhotonMass = 0 := rfl` (formerly
   `photon_massless`) or for `|0| ≤ |0|` in a placeholder layer, and calling
   those "witnessed" is the misreport this phase exists to stop. The gate is
   lexical, so it also runs offline and inside `research/consistency_gate.py`;
   it checks that rows exist and are well formed — a witness or source is a
   declaration, a condition occurs, a retired name is gone — but not that a
   judgement is *right*. That half is CI: `lake build ToE` compiles every
   witness and source a row names.

The ratchet values live in `.github/workflows/lean-ci.yml`, except the statement
audit's `carried_cap`, which lives in `statement_audit.json`; lowering them is
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
axioms of the 164 declarations in `kernel_audit_targets.json`. Only `propext`,
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

### Seventh pass: computation, categories, pattern, expansion

- Vol35 now has a toy computability semantics (`Machine` over ℕ
  configurations) with halting propagation, bounded-halt decidability,
  halt/loop witnesses, and a halting diagonal theorem — not a
  Turing-machine equivalence.
- Vol36 now defines a skeletal FinSet category with products,
  equalizers, exponentials, and subset characteristic maps — honestly
  scoped below a full elementary topos.
- Vol37 now carries a 2-species Jacobian/diffusivity model with the
  Turing instability criterion, the exact inhibitor-faster threshold,
  and a verified activator-inhibitor witness.
- Vol44/Vol48 replace fixed constants with variable positive
  parameters, scaling/inverse laws, antitonicity, and benchmarks.
- Vol54 now proves the empty type's universal property; Vol29
  classifies all Lotka–Volterra equilibria under nonzero rates; the
  Vol04 Robertson header overclaim is corrected (comment-only).
- Ten regression examples and ten kernel-audit targets added
  (142 selected declarations). All changes are additive; no protocol
  signatures changed. Kernel validation passes on PR #47 (Lean CI
  run 36270581122, 2026-09-26): full build green, all 142 targets
  resolve with zero unexpected axioms.

### Eighth pass: honest names and realizable hypotheses (P1 remediation)

This pass closes the P1 items from the adversarial audit remediation log:
naming honesty, vacuous hypotheses, and the two fixed-constant clusters.
No gate was weakened; the kernel-audit inventory grew from 142 to
**155** selected declarations, and the Python/tooling suite from 39 to
**41** tests.

- **Axiom-name gate.** `audit_vacuity.py` gains `--max-axiom-names`
  (CI: 0): no declaration, structure field, or instance assignment may be
  *named* `axiom_*`/`*_axiom`. Model laws carried as structure data are
  named `law_*` (13 fields renamed in `OmegaUnifiedFoundation.lean` and
  `Vol05_GeneralRelativity.lean`); witness theorems are named
  `*_witness`/`*_exists` (10 volume renames); three pure-duplicate protocol
  wrappers were deleted. Prose in comments/docstrings is exempt
  (comment-masked), and the retired legacy names are recorded in
  in-file retirement comments for traceability.
- **Protocol renames.** All 44 `*_axiom` protocol delegations now carry the
  name of the content they prove (`handshaking_lemma`,
  `feedback_convergence`, `hadamard_unitary`, `cantor_diagonal`,
  `wheeler_dewitt_balance`, `dark_energy_deduction` (since qualified to
  `legacy_density_constants_dark_energy_eq` by the rename pass below), …). The P→P
  `schrodinger_equation` wrapper was removed.
- **Vacuous hypotheses retired.** The unsatisfiable premises are gone at
  the source and replaced by non-degenerate restatements:
  `InformationNetwork.log_ratio_nonpos`,
  `InformationNetwork.log_inequality_of_multiplicative_DPI` (the
  log-triangle ingredient behind the metric triangle inequality),
  `InformationNetwork.overlap_one_identifies` and the Vol22 network
  `perfect_overlap_same_entity` theorems under an explicit
  strict-separation law, Vol15 `CosmologicalFluid` acceleration, and the
  Vol27 `IntegratedSystem` positivity results with an honestly scoped
  `Experiences` dictionary (`def Experiences S := 0 < S.phi`, a model
  definition, not a claim about consciousness).
- **Vol09 honest ER=EPR dictionary.** The `EPR_Entanglement := False`
  poisoned well is replaced by a two-qubit separability model:
  `BipartiteState`, `IsProduct` (coefficient matrix factors),
  `Entangled`, `correlationDet` (ad − bc), product ⇒ det = 0,
  det ≠ 0 ⇒ entangled, and the Bell state witnessing both sides. Only the
  sound direction (bridge ⇒ entanglement) is claimed; the converse is not.
- **Λ/G reconciliation.** Vol44's fixed `Λ := 1` legacy cluster and
  Vol48's fixed `G := 1, V := 1` cluster were retired; each module has
  exactly one Λ/G notion (the `DeSitterSpace`/`Compactification`
  structure fields) and `OmegaProtocol.lean` delegates to the
  parameterized theorems. Vol02's `PhotonMass := 0` carries an explicit
  model-postulate docstring.
- **P→P theorems.** `Vol04.schrodinger_equation` (hypothesis restated as
  conclusion) was removed; Vol03 `kms_transitivity`/`zeroth_law` now take
  the KMS-temperature uniqueness law as a genuine hypothesis over every
  state; Vol19's zero-entropy legacy cluster was retired.

**API changes:** protocol theorem names renamed as listed above (old
names recorded in retirement comments); `OmegaUnifiedFoundation`/
`Vol05` law-fields renamed `axiom_*` → `law_*`; Vol09 `EPR_Entanglement`
and `EinsteinRosenBridge` now take a `BipartiteState`; Vol15/Vol19/Vol44/
Vol48 legacy zero/fixed-constant clusters removed. No proof content was
weakened: every rename delegates to the same or a stronger kernel-checked
statement, and the new targets extend the kernel audit.

Local validation: 41 Python/tooling tests, `audit_axioms --max 0` → 0,
`audit_vacuity` → all-zero metrics including the new axiom-name gate,
CI grep gate clean, ruff and mypy clean. The Lean build and kernel gate
run in CI on this branch's PR (no local Lean toolchain is obtainable in
this environment).

### Ninth pass: Robertson, the exact ER=EPR dictionary, real dynamics

- **Vol04 `Robertson` section (general complex inner product space):**
  the full Robertson uncertainty relation
  `Var(A)·Var(B) ≥ (|⟨ψ,[A,B]ψ⟩|/2)²` for self-adjoint `A`, `B` and a
  normalized state `⟨ψ,ψ⟩ = 1` — via the self-adjoint transfer
  (`inner_dev_eq`) and the commutator link (`comm_link`), the two steps
  the file previously documented as MISSING. Nontrivial mathematics
  (Cauchy–Schwarz + exact cancellation of the scalar corrections under
  normalization + the triangle bound `‖z − conj z‖ ≤ 2‖z‖`); in the
  one-dimensional minimal model all commutators vanish, which is why
  the theorem is stated for an arbitrary `H`.
- **Vol09 exact dictionary:** `product_of_correlation_det_zero`
  (2×2 rank-one factorization: `IsProduct ↔ det = 0`) upgrades the
  ER=EPR dictionary to an equivalence `er_epr_iff : bridge ↔
  entanglement`, with the Bell state witnessing both sides. Still a
  model dictionary: no spacetime geometry is derived.
- **Vol04 genuine Schrödinger dynamics:** `schrodingerTrajectory ω ψ₀ t
  = exp(−iωt)·ψ₀` with a real `HasDerivAt` derivative solves
  `iℏ·dψ/dt = H_{ℏω} ψ` for the self-adjoint Hamiltonian `z ↦ ℏω·z`
  (`hamiltonian_self_adjoint`). The retired P→P theorem stays
  retired; this is a constructed and verified solution, not a
  postulate restated.
- Protocol wrappers: `robertson_uncertainty`, `schrodinger_dynamics`,
  `cross_vol09_vol22_er_epr_dictionary_iff`. Kernel inventory: 155 →
  **164** selected declarations; 41 tooling tests; all lexical audits
  at zero.

**API changes:** purely additive. New Vol04 namespace `Robertson.*`,
new Vol04 top-level `hamiltonian`/`schrodingerTrajectory`/
`schrodinger_equation` (the name retired in the eighth pass is
reused for the genuine theorem — the retirement comment explains the
history), new Vol09 `product_of_correlation_det_zero`,
`isProduct_iff_correlation_det`,
`entangled_iff_correlation_det_ne_zero`, `er_epr_iff`.

### Eleventh pass: the four degenerate-model stacks redesigned

The tenth pass renamed `*_axiom` to content names; the eleventh pass removed
the degeneracies those names still sat on.

- **Vol02 electromagnetism**: the zero-form model (`ExteriorDerivative := 0`,
  `HodgeStar := 0`, `J := 0`) and its constructed witnesses are retired.
  Forms are Mathlib `ContinuousAlternatingMap`s on `EuclideanSpace ℝ (Fin 4)`,
  `d` is Mathlib's `extDeriv`, `d_squared_zero` is `extDeriv_extDeriv`
  (symmetry of second derivatives) under `ContDiff ℝ ∞`, `d_add`/`d_smul`/
  `contDiff_d` come from the `fderiv` API, and gauge invariance/composition
  follow from `d² = 0`. The dual Maxwell law is carried as model-law data
  (`SourcedField.law_dual_maxwell`) with a docstring saying so — Mathlib has
  no Hodge star. The form variable is `η` because `open scoped ContDiff`
  turns `ω` into a notation.
- **Vol13 quantum gravity**: `H_matter := 0`/`H_gravity := 0`/`H_total` are
  retired; `wheeler_dewitt_balance` is additive-group algebra under the
  constraint hypothesis `(H_m + H_g) Ψ = 0`, and `MiniSuperspace` gives a
  satisfiable, non-automatic instance with real `p²/(2a³)`/`-k·a` terms.
- **KMS**: the concrete model's `KMSState` is the trace equation
  `ω_ρ (A * B) = ω_ρ (B * A)` proved by `ω_ρ_trace`
  (`LinearMap.trace_mul_comm`), not a constantly-true predicate;
  `kms_temperature_uniqueness_fails` proves the model determines no
  temperature; Vol08 proves `kms_exterior_vacuum` (KMS at `β = 2π/κ`) and
  `kms_beta_hawking` (`(2π/κ)·T_H = 1`). The model is the type-I tracial case
  (`Δ = 1`, identity flow) — stated as such, not presented as type III.
- **APPA**: the nested `low_le_all`/`high_le_high` re-exports are deleted; the
  public names are derivations by case analysis over the label type.

Honest remainder: `StateSpace := ℂ`/`QRegion := Unit`, the placeholder
`op_pow A z := A` used by the modular-operator law, the `PhotonMass := 0`
postulate, and the toy minisuperspace. Kernel audit grew to **173** selected
declarations (ten new targets, the retired `kms_transitivity` replaced);
47 tooling tests; all lexical audits at zero. Lean CI runs the pinned
`lake build ToE` and the transitive kernel-axiom gate; the compile defects CI
surfaced (stuck `Module` metavariables, missing `Decidable`, a no-goals
tactic, the `ω` scope clash, missing `noncomputable`, a form-degree mismatch
in the sourced-Maxwell carrier) are fixed by the follow-up commits, and Lean
CI is green on `7a5fbf2` (push run `37588987044`: pinned `lake build ToE` plus
the 173-declaration transitive-axiom gate, all job steps success).

### Twelfth and Thirteenth passes: non-trivial `QRegion`, zero `Unit` definitions corpus-wide, non-tracial 2×2 KMS, unitary gate matrices, pullback subobject classifier, and verified stack compiler

- **Twelfth pass (`QRegion` upgrade, PR #57):** Upgraded `QRegion` from `Unit`
  to `{ p : ℝ // 0 ≤ p ∧ p ≤ 1 }` with non-zero von Neumann entropy, mutual
  information, overlap density, pair distance, COD profile, and asymmetry flux
  (kernel audit expanded to 226 targets).
- **Thirteenth pass:**
  - **Core foundation & Volumes 01, 03, 05:** Replaced the remaining `Unit`
    definitions in `OmegaAxioms.lean` (`StateType`, `PureState`, `TraceOp`,
    `ConnesInvariant`, `Observable`, `commutator`) and tightened
    `audit_vacuity.py` so `--max-unit-types 0` applies to all 69 `.lean` files.
    Added `concreteTypeIII1Algebra` and the parametric Fisher-metric = KL-Hessian
    derivative proof (`klSecondOrder_hessian_eq_fisher`) in
    `OmegaUnifiedFoundation.lean`, exact Weyl-commutator to Poisson-bracket
    correspondence (`weyl_commutator_quotient_eq_poisson`) in `Vol01`, the
    non-tracial $2\times 2$ matrix modular/KMS model (`Obs2`,
    `gibbsState_not_tracial`, `twoLevel_kms_iff_eq`, `twoLevel_kms_zeroth_law`)
    in `Vol03`, and derivative-based covariant divergence
    (`real_conservation_of_energy_momentum`) in `Vol05`.
  - **Volume 34 (Quantum Computing):** Added a faithful $2\times 2$ unitary
    matrix representation (`GateMatrix`, `matH_unitary`,
    `hadamard_conjugates_X_to_Z`, `interpGate_compose`, `interpGate_injective`,
    `matH_preserves_norm`).
  - **Volume 36 (Topos Theory):** Added pullback-based subobject classification
    (`IsSubobjectPullback`, `skel_subobject_classifier`,
    `skel_subobject_pullback_universal`).
  - **Volume 53 (Universal Compiler):** Added an input-reading arithmetic
    expression language (`Expr`, `evalExpr`, `subst`, `evalExpr_subst`) and a
    verified McCarthy–Painter stack-machine compiler (`Instr`, `compile`,
    `compile_correct_stack`, `compile_correct`).
  - **Volumes 40–43, 45–47, 50–52:** Added real-valued filter chains on $[0,1]$
    (`Vol40`), multi-step path costs (`Vol41`), closed-form Dyson shell scaling
    (`Vol42`), `Fin stars`-indexed multi-hop networks (`Vol43`), dyadic branch
    measures (`Vol45`), injective simulation depth and exponential host overhead
    (`Vol46`), continuous Kardashev power scaling (`Vol47`), subset-indexing
    Cantor diagonalization (`Vol50`), arbitrary finite-state relaxation
    convergence (`Vol51`), and conformal finite-horizon time compression
    (`Vol52`).
  - **Kernel inventory:** **260** declarations in `kernel_audit_targets.json`
    (recounted on 2026-10-12; the pass note originally said 262), 47 Python
    audit tests, all 8 static/vacuity ratchets plus the alias baseline at 0.

### Fourteenth pass: the Phase 5 exit gate (tooling, no proof content changed)

The label-replacement pass above edited comments and docstrings only, and said
so without a machine check behind it. This pass supplies the check and closes
the rest of Phase 5 of the claim-tightening roadmap. **No declaration was
added, removed, or restated**: the code projection of the corpus is unchanged.

- **`check_declaration_parity.py`** reduces every `.lean` file to a canonical
  code projection and hashes it into `declaration_parity.json` (69 files, 1,448
  declarations, 7,492 comment-free code lines). Appending a comment leaves all
  69 digests untouched; one inserted token inside a proof body fails the gate
  with the file and its line-count delta. `--against <ref>` proves a branch is
  comment-only against any revision. Deliberate code changes need `--update` in
  the same commit.
- **`sync_txt_mirrors.py`** makes `txt_proofs/` generated rather than
  hand-edited. Nine modules that had no mirror at all (`RadialMetric`,
  `RadialDilation`, `InformationPhysics`, `DynamicPlanckScale`,
  `DynamicCODScale`, `LogCorrelationMetric`, `ProofRegression`,
  `CBwK_Budget_Pacer`, `APPA_Context_Branching`) were added; all 68 mirrors are
  byte-identical to their sources. Orphans are reported, never deleted.
- 19 new regression tests (66 total), and both gates wired into Lean CI.
- Documentation drift corrected: the stale "Integrated CI status" paragraph
  above, the `262` → `260` kernel-target count, and a status note added to
  `ADVERSARIAL_AUDIT.md` marking its executive summary as the pre-remediation
  record it is.

Verified: green on PR #83 (Lean CI run 38234115307), including `lake build ToE`
and the 260-declaration kernel-axiom gate; the head of the branch re-ran
everything green at run 38251369698, adding the Phase 6 statement-audit step.

### Fifteenth pass: the cited-name check becomes a gate (tooling, no proof content changed)

`docs/PROJECT_STATUS_2026-10-08.md` recorded `check_mathlib_names.py` as
existing "for the next round": a lexical check that would catch a typo like
`Finset.sum_range_succs` offline, before a CI build has to walk to the failing
module one dependency layer at a time. This pass completes that round. **No
declaration was added, removed, or restated.**

- **The indexer was rebuilt for recall, because every miss is a false
  "suspicious" name and a ratchet over false positives is pressure to raise
  it.** It now registers declarations whose attribute block spans preceding
  lines, constructors of `inductive`/`structure`/`class` with the names Lean
  generates (`X.rec`, `X.injEq`, `X.noConfusion`, ...), structure fields where
  they are declared, the `nonrec`/`public`/`expose` modifiers, same-line
  `| ctor`, unicode-dotted components, and the `to_additive`/`to_dual` twins of
  all of the above. Namespace and section scopes are tracked separately, so
  `end someSection` can no longer pop the enclosing namespace.
- **Baseline measured, not assumed.** Against the pinned sources (mathlib4
  `81a5d257…`, lean4 `v4.32.0`): 563,126 mathlib declarations + 292,520 core
  declarations + 7,429 repository declarations indexed, **0 suspicious names**
  (the previous script reported 205 on the same trees). Negative control on a
  copy of the tree with four injected typos (`Real.sqrtt`,
  `Finset.sum_range_succs`, `RadialMetric.chainLenght`, `Set.mem_Ioi.mpq`): all
  four reported; the unmodified copy reports 0.
- **A hollow pass is an error, not a result.** Exit status 2 when any tree —
  mathlib, core, or the repository itself, so a wrong `--repo` counts too —
  yields no declarations, instead of reporting a clean run from an empty index.
- **13 new regression tests** (80 total in this directory, all `unittest` so
  the CI audit step does not need pytest). One pins that a `.lake/` dependency
  checkout inside the repository tree is not scanned as corpus.
- **Wired into Lean CI** after the build, with `if: always()` so a compile
  failure still produces a suspect list. The trees are located by derivation,
  not assumption: the mathlib checkout under `.lake/packages/`, and the core
  sources at `<lean --print-prefix>/src/lean` — the location the pinned
  `src/CMakeLists.txt` installs to (`install(... DESTINATION src/lean)`).
- **The first CI run of the gate failed, and the failure was the useful kind.**
  The local fixtures had used the directory name `mathlib`, which is what the
  checkout is called on disk here, while the requirement is named `mathlib4`
  in `lakefile.lean` and the manifest — so CI's `.lake/packages/mathlib4` did
  not match, and the step exited 1 instead of passing hollow. The step now
  finds the checkout that holds `Mathlib/` (explicit candidates, then a search)
  rather than trusting one name, and the local fixture was rebuilt to mirror
  the CI layout (`.lake/packages/mathlib4`) so it can fail the same way again.
  Five branches are exercised: sources present (exit 0, 0 suspicious), the
  renamed requirement found by the search, missing mathlib, unexpected
  toolchain layout, and no toolchain on `PATH` (each of the last three exits 1
  with an actionable message).
- The offline consistency gate states plainly that this check is one of the
  three it cannot run (with `lake build ToE` and the kernel audit).

Verified in CI: [Lean CI run 38247512557](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38247512557)
step 6, "Gate - cited names exist in the pinned mathlib/Lean sources", green on
the same commit as `lake build ToE` and the 260-declaration kernel gate. The
first attempt is [run 38246572524](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38246572524),
recorded here because its failure is the reason the step searches for the
mathlib checkout: it looked for `.lake/packages/mathlib`, the package is named
`mathlib4`, and it exited 1 rather than reporting a clean run from a tree it
had never read.

### Sixteenth pass: the Phase 6 statement audit (audit record and tooling, no proof content changed)

Phase 6 of the claim-tightening roadmap asked for a per-statement judgement over
the physics volumes, and has been deferred as the one phase that needs the Lean
toolchain to be meaningful. This pass implements it. **No Lean declaration was
added, removed, or renamed**: the parity baseline and the mirrors are untouched,
and the two files added are a checker, an audit record, and its tests.

- **One row per physics-sounding declaration, and "physics-sounding" is a
  documented rule.** `check_statement_audit.py` accepts a `theorem`/`lemma` in
  `Vol*.lean` when one of the words in its name — split on `_`, `.` and
  camelCase — is a token of a 92-word lexicon kept in the file (quantities,
  relativity and quantum theory, thermodynamics and information, and the
  disciplines in the volume titles). The lexicon is a floor, not a ceiling: a
  false positive costs one row, a false negative hides a claim, and adding a
  word immediately demands rows for everything it matches. Measured on the
  corpus: **160 candidates in 49 volumes**, indexed against 1,438 declarations.
- **Four dispositions, each with a checkable obligation.** *witness* (73 rows)
  names a declared, non-degenerate object or instance the statement is about or
  that realizes its setting; *conditional* (43) quotes a hypothesis that must
  occur verbatim in the statement; *renamed* (14) records a name retired by an
  earlier pass, and the retired name must survive nowhere in the corpus once
  comments are masked; *carried* (30) records a statement that is true only from
  data the model carries (a structure field, a postulate, a placeholder or
  constant definition, a definitional restatement, a degenerate instance) and
  names the declaration carrying it.
- **The fourth disposition is an addition, and it is a tightening.** The
  roadmap's sketch listed three. Writing the rows showed the three-way split has
  no honest place for `photon_massless` (`PhotonMass := 0` restated by `rfl`),
  for `bridge_decoherence_suppresses` (`|0| ≤ |0|` in the placeholder layer), or
  for a structure field returned unchanged: those rows would have to name a
  "witness" for a degenerate instance. The audit file carries a `carried_cap`
  ratchet (30, the measured count) that the gate enforces, so a new `carried`
  row fails until the cap is raised in a visible diff and fixing one means
  lowering the cap.
- **The gate fails closed.** Exit 2 when the audit is unreadable or lacks the
  cap; exit 1 on a missing row, a malformed row, a witness or source that is not
  a declaration, a witness that is the theorem itself, a condition that does not
  occur, a retired name that survives, or a `carried` count over the cap. It is
  lexical and offline, so it also runs in `research/consistency_gate.py` — but
  what it cannot do is decide whether a judgement is *right*, and its own output
  says so: that half is `lake build ToE`, which has to compile every witness and
  source a row names.
- **23 new regression tests** (103 in this directory, all `unittest` so the CI
  audit step does not need pytest): one negative control per disposition
  (witness missing, witness self-referential, condition absent, condition
  unquoted, source absent, source self-referential, `renamed_to` unknown,
  retired name still declared, retired name live in code while a comment mention
  is ignored), the missing-row and duplicate-row cases, the cap, the wrong-file
  case, unreadable and cap-less audits exiting 2, the lexicon's own
  no-prose-words regression, and one test that the shipped audit covers the
  shipped corpus.
- **Wired into Lean CI as its own step** with `if: always()` (a compile failure
  still reports an unreadable or incomplete audit), and into the consistency
  gate as a sixteenth command.
- **What the rows say, in aggregate.** 73 of the 160 declarations are about
  declared, non-degenerate objects; 43 carry their hypotheses in their own
  signature; 14 are renames recorded rather than performed; 30 are true only
  from carried data. The `carried` list is the phase's finding: it names the
  declarations whose truth is a definitional or degenerate-model artifact —
  `metric_is_qfim` (both sides 0 on the zero-curvature instance),
  `clausius_inequality` (the structure's own balance field), `optical_expansion`
  (`SMatrix := 1`, `TMatrix := 0`), `flat_universe_sum` (the legacy (0,0,1)
  constants), and the tracial-model KMS rows, among others — and each of those
  notes says what is carried and by which declaration instead of implying a
  derivation. (The names above are the phase-6 names; the rename pass below
  qualified sixteen of them and grew the audit to 163 rows with a cap of 33.)
- **Verified in CI.** Green on PR #83 at
  [Lean CI run 38251369698](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38251369698)
  — the new `Gate - statement audit of the physics volumes` step passes next to
  `lake build ToE`, the cited-name gate and the 260-declaration kernel audit
  ([CI run 38251369679](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38251369679)
  for the pytest side). The merge into `main` (`a764f34`) re-ran both green
  ([Lean CI 38255776907](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38255776907),
  [CI 38255777137](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38255777137)).

### Seventeenth pass: the Phase 6 rename pass (names qualified, code changed)

The audit's follow-up: sixteen declarations whose *name* asserted a result the statement
does not contain were renamed to names that state the content. This is the one pass in the
report that changes code, so it ships the parity baseline regenerated
(`check_declaration_parity.py --update`), the `txt_proofs/` mirrors re-synced
(`sync_txt_mirrors.py --write`), the updated `ProofRegression.lean` reference and
kernel-audit target, and the `OmegaProtocol.lean` re-exports under the same names — in the
same commit.

| old name | new name | why the old name overstated |
|---|---|---|
| `photon_massless` | `photonMass_eq_zero` | the statement is the postulate `PhotonMass := 0` unfolded by `rfl` |
| `maxwells_equations` | `maxwell_equations_of_sourced_field` | one of the two conjuncts is the carried model law |
| `metric_is_qfim` | `zeroCurvature_metric_eq_qfim` | restates the carried law field at the degenerate instance (both sides 0) |
| `einstein_field_equations` | `zeroCurvature_einstein_field_equations` | same, `law_einstein_equations`; both sides 0 |
| `bianchi_identity` | `zeroCurvature_bianchi_identity` | same, `law_bianchi_identity`; ∇·0 = 0 |
| `cosmological_constant` | `zeroCurvature_cosmological_constant` | same, `law_cosmological_constant`; Λ = 0 |
| `conservation_of_energy_momentum` | `zeroCurvature_conservation_of_energy_momentum` | the chain is real, but both tensors vanish at this instance |
| `em_sources_gravity` | `zeroCurvature_stress_energy_eq_em_stress_energy` | `EMStressEnergy` is *defined* as `stress_energy`; no coupling is shown |
| `optical_expansion` | `placeholder_optical_expansion` | `SMatrix := 1`, `TMatrix := 0`; no optical theorem is derived |
| `standard_model_gauge_group` | `GaugeSymmetry_eq_product` | the carrier is defined as the product; no Lie-group construction |
| `bridge_bh_entropy_is_holographic` | `bridge_entanglement_entropy_eq_bh_entropy` | the identity is definitional; no holographic dictionary |
| `gauge_symmetry` | `SM_Gauge_Group_eq_product` | the same definitional equality, not a gauge-group construction |
| `flat_universe_sum` | `legacy_density_constants_sum_eq_one` | flatness is the legacy constant `1`, not a derived property |
| `dark_energy_deduction` | `legacy_density_constants_dark_energy_eq` | the "deduction" is `linarith` on the legacy constants |
| `quantum_computing_from_omega` | `hadamard_unitary_restated` | restates `hadamard_unitary`; no construction from Ω |
| `turing_instability_ratio` | `diffusion_ratio_gt_one_of_activator_pos` | a ratio of two constant diffusivities is not an instability proof |

- **The gate grew with the pass.** `previous_name` is now validated on *any* row, not only
  on `renamed` rows: a renamed declaration keeps the disposition of its statement and still
  records what it used to be called. Matching is on identifier boundaries, so
  `zeroCurvature_bianchi_identity` is not a surviving `bianchi_identity`, while a
  deliberately kept law field beside it (`law_bianchi_identity`, `law_metric_is_qfim`) is a
  different name and not a stale use. A `previous_name` equal to the row's own name is
  rejected.
- **Coverage grew with the cap visible.** Three renamed declarations
  (`einstein_field_equations`, `bianchi_identity`, `maxwells_equations`) were never lexicon
  candidates and now have `carried` rows naming their carriers, so the audit grew
  160 → 163 rows and `carried_cap` 30 → 33 in one visible diff, with no existing row
  reclassified. Two rows (`maxwell_equations_of_sourced_field`,
  `hadamard_unitary_restated`) record renamed declarations whose new names the lexicon no
  longer matches; the gate prints them as extras.
- **4 new regression tests** (27 in `test_statement_audit.py`, 107 in this directory): the
  retirement check on a non-`renamed` disposition, the token-boundary match, a row whose
  `previous_name` is its own name, and a test that the shipped audit's rename records are
  complete (every retired name absent from the corpus, every replacement declared).
- **Verified locally**: 107 `lean_proofs` tests, 215 repo tests, the statement audit green
  at 163 rows (33 carried, cap 33, 30 retirement records), mirrors and parity in sync, all
  eight `audit_vacuity` ratchets and the alias baseline at 0, the cited-name check 0
  suspicious names against the pinned trees, `ruff`/`mypy` clean, the consistency gate
  16/16. Verified in CI at the pass's commit `9dd2379`: Lean CI
  [run 38254475070](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38254475070)
  — `lake build ToE` compiling every renamed declaration, the kernel audit over its 260
  targets (including the renamed `OmegaProtocol.Vol09.bridge_entanglement_entropy_eq_bh_entropy`),
  the cited-name gate and the statement-audit step — and CI
  [run 38254475046](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38254475046).
  PR #83 was then merged into `main` as merge commit `a764f34`, and `main`'s own push runs
  are green too ([Lean CI 38255776907](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38255776907),
  [CI 38255777137](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/38255777137)).
