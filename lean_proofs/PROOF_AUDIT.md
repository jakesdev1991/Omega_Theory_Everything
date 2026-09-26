# Lean proof audit and strengthening pass

Date: 2026-09-26. Scope: source review of all 65 pre-existing proof modules,
plus Lake configuration; the two new proof modules are inventoried below.
The full pinned Lean build and the selected transitive kernel-axiom audit have
now **passed in GitHub CI**. This does not certify physical interpretations or
semantic non-vacuity. Earlier pass-status notes below are historical.

## Main finding and logical next step

The most important distinction is **correct proof of a weak model** versus
**proof of the intended physical claim**. Absence of declaration-level axioms
or proof holes is necessary but insufficient. The existing singleton model
makes information, distance, flux and asymmetry identically zero. Many later
physical-sounding theorems inherit that degeneracy; others assume the conclusion.

The next step is to construct witnessed non-degenerate models, expose physical
assumptions, and establish converses, sharp bounds and state-transition
invariants. This pass starts that work without silently redefining the legacy
QRegion API or pretending to derive physical laws.

### Highest-priority findings

1. **Log-metric sign:** for positive normalized correlations at a common scale,
   `Kxy * Kyz ≤ Kxz` suffices for the negative-log triangle inequality. An upper
   multiplicative bound points the other way. The legacy zero-information model
   conceals this distinction. `LogCorrelationMetric.lean` adds the general
   sufficient-condition lemma, a real-line example with distance `|x-y|`, and
   a counterexample to the opposite bound. Pair-dependent scales still need a
   separate argument; this does not validate the legacy metric for general data.
2. **Radial sharpness:** the original bound did not state the promised unique
   equality case. Added a weighted squared-error bound and equality iff `Φ=Φ*`.
   Constant chain scale is sufficient, NOT necessary for factorization; corrected
   the prose and added the exact error and an iff characterization.
3. **Budget safety:** per-call admissibility against unchanged headroom is not
   cumulative safety. Added state-changing spend and finite-run success iff
   total cost fits, with exact remaining-balance conservation. Certificates now
   tie their stored cost to the indexed request. Pass five adds a separate
   capacity-bounded ledger and mixed reserve/settle trace conservation. Legacy
   headroom-only settlement still permits arbitrary reclamation; the new ledger
   bounds aggregate refunds but makes no ownership or concurrency claim.
4. **Security gate:** flags supplied by callers are not sanitizer correctness.
   Added executable rejection/acceptance equivalences for the two flags and
   whole-parent preservation, without claiming host isolation is implemented.
5. **Compiler expressiveness:** the current grammar has no input constructor.
   Added input-independence and non-representability of identity, instead of
   bolstering the misleading universality interpretation.
6. **Audit blind spots, partly closed in pass two:** the old direct-conclusion
   patterns missed `*_Stmt` wrappers (including foundation exports) and
   parameterized `DifferentialForm`. The audit now counts parameterized Unit
   types and separately tracks all 96 legacy statement-alias exports in an
   exact per-declaration baseline. It still does not unfold arbitrary Lean
   definitions, detect impossible premises, or inspect transitive kernel axioms.
   A zero direct-pattern count is NOT a semantic non-vacuity certificate.

## Validation status

- **Remote compiler validation passed:** [Lean CI run 36240687825](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/36240687825)
  at commit `8329f951a31b2bb44a5d5ff62938de0032290662`, using the pinned
  Lean/mathlib v4.32.0. `lake build ToE` compiled the full configured target,
  including `ProofRegression.lean`. The subsequent build-before-report kernel
  gate passed for all **76 selected declarations**, allowing only `propext`,
  `Classical.choice`, and `Quot.sound` as foundational axioms.
- Lean/Lake remain absent locally; the successful compiler execution was on a
  GitHub-hosted runner, not this workspace. This supersedes earlier pending-build
  notes. Selected axiom inspection is not a whole-corpus axiom inventory.
- Passed after pass six: 35 Python source-audit/build-wiring/report-gate tests;
  declaration-axiom audit (0); direct misleading-pattern matches (0);
  Unit stubs (0); remaining counted Unit types (13); exact legacy alias
  baseline (96); prohibited-proof-escape grep; Ruff checks/formatting on
  changed Python tooling; `git diff --check`.
  These results do not establish elaboration or semantic non-vacuity.
- Reproduce with `cd lean_proofs && lake build ToE && python3 audit_kernel.py`.
  `ProofRegression.lean` is included in that target and `ToE.lean`.
- Plain `lake build` now has an explicit default target. No dependency versions
  were changed. Most public signatures were retained. Intentional API changes:
  `CBwK.Reserved` now requires `matches_request`; Vol12 state values are pairs;
  Vol49 frame values are unit phases; Vol25 `BHState` requires joint-entropy
  data and explicit bounds, `unitarity_total` states joint purity, and the false
  marginal-sum interpretation (`entropies_opposite`) was removed. Vol30 now
  requires `KeplerSystem` and radius sign hypotheses; Vol38 requires
  `LinearMarket`. Vol26 now bundles finite simple graphs; Vol39 count functions
  take that graph and connection bounds require `NoIsolatedVertices`.
  Vol10 `HiggsField` now aliases ℝ. Vol16/17/21 now require explicit process or
  trajectory arguments; their derivatives are real derivatives. Vol21's Q-region
  time/latency identification was removed. CBwK dimensions now include actual
  allocated funds. Protocol wrappers were updated accordingly.

## Module-by-module inventory and next work

“Substantive” below means content within the stated mathematical model, not
experimental support or a derivation of the physical topic from Omega Theory.
Unchanged limitations are deliberately recorded rather than hidden by renaming.

### Foundations, applications and build

| Source | Findings / next step |
|---|---|
| `OmegaAxioms.lean` | Explicit complex one-space and Unit regions; zero information and stress quantities, placeholder powers/commutator/limit. Positive information, perfect overlap and nonzero asymmetry premises are unrealizable. Replace primitives through an explicit model interface with nontrivial witnesses before porting physical claims. |
| `OmegaUnifiedFoundation.lean` | Modular flow is zero and lacks a zero-time identity law; KMS always true; QFIM zero. Most named volume/cross-volume results are aliases for elementary zero-model facts, not the named laws. Symplectic closedness field is not exterior-derivative closedness. Requires model-interface redesign. |
| `OmegaDimensionalHierarchy.lean` | Import-only compatibility module; no dimensional hierarchy is constructed here. |
| `OmegaProtocol.lean` | Mostly wrappers; some genuine Vol01/04 mathematics, many degenerate or hypothesis-only conclusions. Wrapper names do not strengthen imported statements. Vol30/38 wrappers now expose positive-parameter models; misleading game-theory wrapper names have explicit scope notes. |
| `InformationPhysics.lean` | Positive environmental constants and model-dependent energy/mass bookkeeping. Added linearity, strict monotonicity, conversion conservation and output feasibility. Non-ideal Erasure now includes explicit nonnegative excess costs, sharp saturation/zero-cost criteria, batch decomposition and exact aggregate/prefix affordability. The ideal mass/scale API is unchanged. Next: connect excess cost and lower-bound assumptions to an actual thermodynamic protocol; no microscopic derivation is claimed. |
| `DynamicPlanckScale.lean` | Positive chosen density profile; added global upper bound, contraction iff positive density, baseline iff nonpositive density, and positive-domain identifiability. Next: boundary continuity and limit at infinity. |
| `DynamicCODScale.lean` | Chosen sqrt profile; added physical-domain injectivity, exact endpoint converses, squared-overlap recovery, evenness and failure of global injectivity. Outside [-1,1], Lean sqrt truncates to zero; no global physical interpretation. |
| `RadialMetric.lean` | Added quantitative gap/unique maximizer and exact chain error. Schwarzschild identities remain algebraic identifications, not Einstein-equation derivations. |
| `CBwK_Budget_Pacer.lean` | Added a bounded available/reserved/spent ledger and successful-trace capacity conservation; spent funds cannot decrease and settlement cannot exceed outstanding reserves. Dimension.inv now bounds actual allocation. Legacy headroom-only settle remains arithmetic. Next: identified reservations, ownership, persistence and concurrency refinement. |
| `APPA_Context_Branching.lean` | Added full-parent preservation and executable fail-closed gate. Caller-provided flags and immutable records do not prove OS isolation or semantic sanitization. Next: validator specification and integration refinement. |
| `ToE.lean` | Aggregating imports, not a unification theorem. New modules and regression checks included. |
| `lakefile.lean` | Pinned dependencies unchanged; all local proof modules rooted; added explicit default target and coverage tests. |
| `LogCorrelationMetric.lean` | New positive real-line kernel with separated points, symmetry, triangle and multiplicative lower bound. Independent reference model, not an assertion that real mutual information has these properties. |
| `ProofRegression.lean` | New boundary witnesses for budget rejection/exhaustion, gate flag combinations, equal Boltzmann weights, fitter selection, nonconstant-scale factorization and compiler limitation. Pending kernel build. |

### Volumes

| Source | Findings / next step |
|---|---|
| `Vol01_ClassicalMechanics.lean` | Substantive symplectic/derivative/energy/action algebra; legacy commutator is a singleton operation and mass is zero. Next: general Poisson Jacobi under smoothness and strict action equality cases. |
| `Vol02_Electromagnetism.lean` | Differential forms (including the parameterized type), spacetime and derivatives are Unit-valued. Maxwell statements do not describe nonzero fields. Next: actual differential forms and exterior derivative. |
| `Vol03_Thermodynamics.lean` | KMS is always true; zeroth law assumes the requested temperature equality; positive-temperature premise is impossible. Next: nonzero thermodynamic states and an equilibrium relation. |
| `Vol04_QuantumMechanics.lean` | Projection/Born and Cauchy–Schwarz arguments are mathematical; StateSpace is only complex one-space. Time derivative is zero and Schrödinger conclusion is a hypothesis. Next: general Hilbert space and actual time derivatives. |
| `Vol05_GeneralRelativity.lean` | Unit spacetime, zero tensors and zero divergence. Next: nondegenerate metric and genuine connection before claiming field-equation consequences. |
| `Vol06_QuantumFieldTheory.lean` | Region inclusion/separation are False; local nets empty; gauge groups Unit; adjoint is identity. Next: nonempty net with realizable separation, actual groups and adjoint. |
| `Vol07_Cosmology.lean` | Scale factor is one; density, pressure and derivative zero. Next: parameterized differentiable scale-factor solutions rather than a static zero-density instance. |
| `Vol08_BlackHoleThermodynamics.lean` | All black-hole observables zero. First-law rewriting is conditional algebra, not a black-hole solution. Next: positive mass/area model. |
| `Vol09_HolographicPrinciple.lean` | Bulk/boundary and areas trivial; ER/EPR predicates false. Next: a nontrivial boundary map and explicit reconstruction hypotheses. |
| `Vol10_StandardModel.lean` | Generation count and singleton gauge carriers remain legacy parameters. The field carrier is now real, and all signed quartic minimizers are classified as ±v. Nonnegative amplitude gives uniqueness at +v. Next: a genuine complex doublet and gauge action; no full Standard Model is derived. |
| `Vol11_CondensedMatter.lean` | Added square completion, sharp lower-bound equality, attainable branches and full minimizer characterization for positive quartic/nonpositive quadratic coefficients. Critical b=0 is explicitly flat; b>0 gives a unique zero. Chern number remains defined as zero. Next: positive-quadratic regime and an actual topological invariant. |
| `Vol12_QuantumInformation.lean` | Replaced Unit by two complex amplitudes and the Hermitian overlap. A normalized (3/5,4/5) witness obstructs universal overlap copying. The overlap-preservation condition is still an explicit necessary condition, not derived from a tensor-product unitary. QubitSystem entropy stub remains. Next: actual tensor products and unitary-copy specification. |
| `Vol13_QuantumGravity.lean` | Both Hamiltonians zero, spin networks/gravitons Unit. Coupling reduces to metric reflexivity. Next: nonzero constrained operators. |
| `Vol14_DarkSector.lean` | Density fractions fixed to (0,0,1). Next: parameterized nonnegative fractions on a simplex. |
| `Vol15_EarlyUniverse.lean` | Inflation hypothesis contradicts the zero density/pressure definitions. Next: a realizable negative-pressure model. |
| `Vol16_NonEquilibriumThermodynamics.lean` | Replaced zero functions with EntropyProcess and a HasDerivAt balance hypothesis. Isolation implies monotonicity, positive production implies strict growth, and initial nonnegativity propagates forward. An entropy-exporting witness defeats monotonicity without controlled flux. Next: derive balance/production assumptions from a thermodynamic system. |
| `Vol17_FluidDynamics.lean` | Replaced zero quantities with a positive FluidTrajectory satisfying material continuity at the derivative level. Incompressibility implies constant density across times; pointwise zero rate iff zero divergence uses strict positivity. Exponential decay at unit divergence witnesses nonconstant behavior. Next: actual spatial velocity fields/PDEs and vacuum handling. |
| `Vol18_StatisticalMechanics.lean` | Two-state Boltzmann model is nontrivial. Added derived partition positivity, unconditional normalization and strict probability bounds; next: arbitrary finite ensembles. |
| `Vol19_ComplexSystems.lean` | ComplexityEntropy constant zero, so emergence premise impossible. Next: a nonconstant observable and witnessed increase. |
| `Vol20_ChaosTheory.lean` | Lyapunov exponent defined as one, no dynamics. Next: derivative growth along a concrete iterated map. |
| `Vol21_ArrowOfTime.lean` | Now reuses Vol16 entropy histories and actual derivatives; ordering requires rate/isolation hypotheses. Initial entropy nonnegativity is explicit. Removed the spurious time/latency identification. Next: a specified cosmological system; this remains a conditional entropy arrow, not a derivation of time orientation. |
| `Vol22_EREqualsEPR.lean` | Both entanglement and throat area zero; perfect overlap impossible. Next: independently specified nonzero geometry/correlation model, not identifications by definition. |
| `Vol23_MeasurementProblem.lean` | DensityMatrix aliases Unit, trace one, off-diagonal zero, decoherence identity. Next: finite density matrices and a dephasing channel. |
| `Vol24_NonLocality.lean` | Proves the numerical inequality 2√2 > 2, not existence of a CHSH experiment or the operator upper bound. Next: concrete observables/state witness. |
| `Vol25_BlackHoleInformation.lean` | Corrected the marginal-sum model: joint entropy is separate; nonnegativity, Araki–Lieb difference bound and joint purity are explicit hypotheses. Derived equal marginals and endpoint iff; added positive-marginal witness. Next: derive the assumed inequality from density matrices and construct dynamics. |
| `Vol26_NetworkTheory.lean` | Replaced the definitional identity with a finite SimpleGraph bundle, independent neighbor/edge counts, degree-sum theorem, odd-degree parity and a degree-floor bound. Empty graphs and a one-edge/two-vertex graph are witnessed. Next: path/connectivity and weighted information-bearing edges. |
| `Vol27_Consciousness.lean` | Asymmetry always zero; consciousness-positivity premises impossible. Next: abstract nonzero integration data, without claiming phenomenology follows from an absolute value. |
| `Vol28_EvolutionaryAlgorithms.lean` | Real OneMax list model. Added exact maximum fitness and domination of both parents; next: population-level elitist invariants. |
| `Vol29_EcosystemDynamics.lean` | Lotka–Volterra equilibrium is substantive algebra under nonzero parameters. Next: classify all equilibria and analyze stability under positive parameters. |
| `Vol30_PlanetarySystems.lean` | Replaced the zero-denominator model with positive-parameter KeplerSystem and a chosen sqrt period formula. Added period positivity, endpoint iff, strictly increasing periods and a positive-radius ratio law. Next: derive this relation from actual two-body dynamics; no ODE derivation is claimed. |
| `Vol31_GameTheory.lean` | Concrete prisoner’s-dilemma table now has a full pure-Nash predicate, unique equilibrium classification, strict dominance and a separate Pareto-superiority result. Next: explicitly modeled repeated-game dynamics, not evolution-of-cooperation claims from a static table. |
| `Vol32_Cybernetics.lean` | Added a scalar feedback trajectory, step equation and actual convergence for 0 ≤ gain < 1. Nonzero initial states at gain one provably do not decay. Next: disturbed or time-varying gains and multidimensional stability. |
| `Vol33_AGI.lean` | Pointwise demand domination model, not intelligence/alignment. Integration-positive premise still impossible. Next: finite resource allocation with feasibility witnesses. |
| `Vol34_QuantumComputing.lean` | Two-symbol gate algebra, not matrices or arbitrary quantum circuits. Next: faithful unitary matrix interpretation. |
| `Vol35_TheoryOfComputation.lean` | Valid diagonal non-surjectivity statement; repeated proof. Next: explicit computability semantics before halting/complexity claims. |
| `Vol36_ToposTheory.lean` | Excluded middle only; no topos/category defined. Next: a concrete category and subobject classifier. |
| `Vol37_Morphogenesis.lean` | Diffusion constants defined as 1 and 2. Faster inhibitor diffusion alone is not a Turing instability proof. Next: reaction Jacobian and stability conditions. |
| `Vol38_Economics.lean` | Replaced zero curves with a LinearMarket with positive slopes/intercept. Derived unique positive equilibrium, positive quantity, choke-price bound and excess-demand sign. Next: nonlinear or clipped curves; total real extensions are not interpreted physically. |
| `Vol39_SocietalNetworks.lean` | Replaced constants by Vol26 graph counts. No-isolated-vertices explicitly implies n ≤ 2E and the rounded edge bound; the singleton edgeless graph refutes an unconditional version. Next: distinguish reachability/connectivity from the weaker degree assumption. |
| `Vol40_FermiParadox.lean` | Boolean filter chains and natural-number products are toy combinatorics. Next: survival probabilities and normalization, not physical existence claims. |
| `Vol41_PostBiologicalEvolution.lean` | Finite three-substrate cost model, with genuine triangle/separation checks. Next: compose longer paths; physical costs remain stipulated. |
| `Vol42_StellarEngineering.lean` | Recurrence and fractional capture inequalities have content. Capture bound lacks a lower fraction bound for nonnegativity. Next: fraction in [0,1] and resource conservation. |
| `Vol43_GalacticEcosystems.lean` | Symmetric irreflexive adjacency has content, but indices need not lie below stars. Next: vertices indexed by Fin stars and actual paths. |
| `Vol44_UniversalExpansion.lean` | Positive de Sitter entropy for a fixed positive Lambda; not a dynamical expansion theorem. Next: variable positive Lambda. |
| `Vol45_MultiverseTheory.lean` | Binary syntax tree with depth and doubling weight. Weight is not a normalized probability. Next: finite-level counts/measures. |
| `Vol46_SimulationHypothesis.lean` | Unary layer syntax ordered by depth. Does not establish simulated reality. Next: prove order/level classification and executable resource bounds. |
| `Vol47_TranscendentArchitectures.lean` | Four stipulated energy tiers; finite case analysis. Next: parameterized resource model with feasibility conditions. |
| `Vol48_ExtraDimensions.lean` | Higher-dimensional coupling and volume fixed to one. Next: positive variable volume and inverse monotonicity. |
| `Vol49_QuantumReferenceFrames.lean` | Replaced Unit by norm-one complex phases. Ratio transformations compose, have inverse round trips and preserve norms; ±1 phases give a nonidentity witness. Next: multidimensional unitary frames and observer semantics. |
| `Vol50_UltimateEnsemble.lean` | Inhabited carrier bundle plus Cantor diagonal theorem on Boolean sequences. The diagonal theorem is not directly about enumeration of the bundle. Next: precise universe/encoding statement. |
| `Vol51_ClosedTimelikeCurves.lean` | Three-state transition system has a unique fixed point and settles in two steps. Next: general finite transition-system convergence, not spacetime causality claims. |
| `Vol52_OmegaPointTheory.lean` | Natural-number complexity chain has no terminal stage. Does not prove a physical final singularity. Next: distinguish unbounded discrete growth from finite-time blow-up. |
| `Vol53_UniversalCompiler.lean` | Added proof that every current program is input-independent and identity is unrepresentable. Next: input constructor, semantics-preserving compilation, then an explicit universality target. |
| `Vol54_TheoryOfNothing.lean` | AbsoluteNothingness is defined as metric reflexivity. Next: a meaningful formal proposition; no metaphysical existence conclusion follows. |

## Second-pass changes and compatibility notes

The second pass deliberately repairs a false physical interpretation rather than
preserving its theorem name: purity is zero **joint** entropy and can coexist
with positive marginal entropies. The replacement Vol25 model assumes the
Araki–Lieb bound explicitly and proves its pure-state consequences. Existing
protocol endpoint callers keep the same theorem signature, but external code
constructing `BHState` must provide the new fields. The old
`entropies_opposite` statement is not available because it is wrong for the
replacement model.

Two singleton carriers were removed (Vol12 and Vol49). The more complete Unit
scanner finds one previously missed parameterized type, so the reported CI
threshold falls from 15 to 14; under the corrected counting rule the original
corpus had 16. This is a real reduction of two, not an audit exemption.

`lean_source.py` masks nested comments and strings while preserving line numbers;
`test_source_audits.py` exercises comment nesting, escaped quotes, attributes,
Unicode, nested modules, parameterized Unit types, long theorem declarations,
and alias-baseline replacement attacks. `legacy_statement_aliases.json` is an
explicit inventory of outstanding scope debt, not an approval of the historical
physical names. New aliases fail the gate; retired entries must be removed.

The direct Lean release download was retried in pass two and again failed at the
GitHub asset host with SSL errors. All added and modified Lean declarations
remain **pending kernel validation**, including the first pass's proof scripts.

## Third-pass changes and trust-boundary gate

The zero-valued Vol30 and Vol38 models have been replaced rather than retained
as vacuous compatibility examples. Their signatures intentionally change to
expose `KeplerSystem` and `LinearMarket`. The Kepler ratio theorem requires
**strictly** positive radii: a regression shows that including zero radius
would equate zero with a positive value because Lean totalizes division by zero.
The market witness has intercept 12, supply slope 2 and demand slope 1; its
price is 4 and its traded quantity is 8. Prices 3 and 5 have excess demands
3 and -3, respectively, so clearing is not a universal zero-function identity.

The game-theory additions quantify over every unilateral deviation and
classify all pure equilibria. The feedback additions use the actual topological
limit, with regression cases at gains 0, 1/2, and 1. These conclusions remain
within their chosen static-game/scalar-dynamics models.

`audit_kernel.py` adds a fail-closed transitive-axiom gate for 39 selected key
declarations, including earlier strengthened results. It first builds `ToE`,
then uses Lean's `#print axioms` output. Only the standard foundations
`propext`, `Classical.choice`, and `Quot.sound` are accepted. A proof-hole axiom,
custom postulate, malformed/incomplete report, duplicate report, failed command,
or absent toolchain cannot be reported as success. This is **not whole-corpus
coverage** and does not replace the semantic review of assumptions. CI invokes
it after building, and the scratch file is isolated under ignored `.lake/`.

The 26 Python tests include 10 fixtures for the gate/parser/failure paths.
The actual gate was invoked locally and returned exit code 2 with
“Lake is unavailable; no kernel audit was performed.” The installation retry
again failed at the official raw-content host. Consequently, the three passes'
Lean scripts and the gate's end-to-end integration remain pending a real pinned
Lean build; passing Python fixtures is not compiler certification.

## Fourth-pass changes: incidence is not a definition of edge count

Vol26 now counts adjacency neighbors and unordered edges independently. The
handshaking theorem invokes Mathlib's finite graph result, and parity of the
number of odd-degree vertices is a separate consequence. Vertices use `Fin n`,
so they are distinct and in bounds. The two-vertex complete graph witnesses
degree sum 2 and edge count 1; empty graphs exist for any vertex count.

Vol39 shares that graph model. The no-isolated-vertices assumption is exposed
rather than hidden behind an informal connectivity claim. It gives a rounded
edge lower bound, but says nothing about reachability or connectedness. A
singleton with no edge is an explicit counterexample to dropping the assumption.

The potential results now distinguish a zero-energy witness from a complete
minimizer classification. The signed scalar Higgs potential has ±v minima;
+v uniqueness needs the nonnegative domain. Landau square completion requires
b>0, and its bound is attainable when a(T−Tc)≤0. At b=0 and T=Tc every real m
has zero energy. These are polynomial results, not microscopic physical proofs.
Replacing the Higgs amplitude carrier lowers the Unit-type ceiling to 13.

The selected kernel-axiom inventory grew from 39 to 51 declarations, including
the graph and potential results. Five additional mocked control-flow tests
exercise build-before-inspection, no inspection after failed builds, timeouts,
invalid targets, and scratch-file cleanup on both accepted and rejected reports.
Declaration-name validation also rejects malformed dotted names.

Validation remains split: 31 Python tests and the lexical/style checks pass;
`audit_kernel.py` was invoked and again returned exit code 2 because Lake is
absent. The mocked runner tests deliberately do not invoke Lean. All four passes'
new or changed Lean proof scripts still require the pinned kernel build.

## Fifth-pass changes: derivatives and accounting transitions

Vol16, Vol17 and Vol21 no longer encode evolution by returning zero from a
function named derivative. Their `HasDerivAt` fields are explicit model laws,
and Mathlib calculus supplies the derivative-to-monotonicity and zero-derivative-
to-constancy implications. The new trajectory arguments are intentional API
changes, and the protocol wrappers were updated. Concrete histories include
constant entropy, exp(t) with positive production, exp(-t) with negative entropy
flux, constant positive fluid density and exp(-t) density at unit divergence.
Nonnegative production alone is insufficient for entropy monotonicity, and
positive density is essential for recovering divergence from a zero density rate.
There is still no microscopic entropy derivation or spatial fluid-field model.

The bounded ledger preserves `available + reserved + spent = capacity` under
checked reserve/settle operations. Both charge and release must jointly fit the
outstanding reserve; a trace cannot release spent funds or change its original
capacity. Regression traces cover successful partial/full settlement, aggregate
over-settlement, double reservation, repeated full refunds, spent-fund recovery
attempts and zero-capacity no-ops. Dimension invariants now compare actual
allocation with budget, and `dimensionOfLedger` constructs that view safely.

This does not secure a runtime pacer against caller-authentication failures,
reservation mix-ups, forked snapshots, rollback or concurrent writes. Those
require identified reservations and an implementation refinement. The existing
headroom-only `settle` is retained for compatibility and explicitly documented
as unsafe for refund-bounded accounting; the new ledger is a separate interface.

The source tooling now checks local import cycles and unresolved local imports,
with fixtures for self/cross cycles, shared dependencies, comment masking and
nested modules. There are 35 Python/tooling tests and 67 selected kernel-audit
declarations. The live kernel gate was invoked again and returned exit code 2
because Lake is unavailable. All five passes' Lean changes and the executable
Lean regression examples therefore remain pending the pinned compiler build.

## Sixth-pass changes: non-ideal erasure is not universal saturation

Added `InformationPhysics.Erasure` with nonnegative real information quantity
and excess energy. Its cost is the chosen ideal Landauer formula plus excess:
the lower-bound assumption is explicit in the model, not presented as a derived
physical law. Sharp single-process results characterize saturation, strict
excess, and zero cost. `lossyBit` is a concrete positive-excess witness against
universal equality. Zero information with positive overhead is also realizable.

For finite batches at one fixed environment, ideal energy and excess both add.
A batch meets the ideal bound iff every member has zero excess; no positive
excess can cancel another member's overhead. Remaining mass subtracts **all**
erasure energy divided by c², with conservation counting spent energy as well.
Nonnegative remaining mass is equivalent to aggregate affordability. Sequential
payment matches concatenated payment, and every prefix of an affordable batch
is affordable. These are accounting identities and inequalities, not a physical
conversion mechanism or a microscopic derivation of thermodynamic inequalities.

The existing ideal information-mass/density/scale coupling is deliberately
unchanged. Twelve new Lean regression examples cover zero/positive overhead,
ideal and mixed batches, empty input, exact exhaustion and two jobs that each
fit separately but exceed the budget together. They remain unexecuted locally.
The 35 Python/tooling tests and lexical/style checks pass; the kernel inventory
has 76 selected declarations. The actual kernel gate still exits 2 (Lake absent),
so all six passes remain pending compiler elaboration and kernel inspection.

## Compiler-validation pass: actual diagnostics and repairs

Opened draft PR #45 and enabled the existing Lean CI workflow on this exact
session branch. The PR conflicts with newer main changes, so validation ran on
branch pushes, not a synthetic merged revision. Nothing was merged into main.

The first compiler run found errors in six modules. Repairs preserved the
mathematical claims and did not add proof escapes:

- Vol12: normalize conjugation of complex numerals explicitly with `map_ofNat`.
- Vol11: normalize negated quotients before linear arithmetic on Landau bounds.
- CBwK: prove the headroom subtraction/addition identity explicitly; remove dead
  branches already closed by `split_ifs`.
- Vol32: expose the sequence lambda before comparing limit statements.
- Vol49: use the supplied scalar action and norm law on the wrapped StateSpace;
  give the nontrivial witness an explicit `unitState` instead of requiring a
  missing numeral instance on that wrapper.
- Vol26: keep bundled graph instances consistent, type finite vertex witnesses
  explicitly, and prove the finite-cardinality equality by a calculation chain.

The successful run also elaborated downstream protocol exports and all Lean
regression examples. The workflow's error annotations now filter long runner
command lines so real diagnostics are not truncated by those lines.

**Remaining integration work:** reconcile this draft branch with newer main
changes, then rerun the full build and selected axiom gate on the reconciled
revision. Existing nonfatal linter warnings, 13 counted Unit types and 96 legacy
statement aliases remain; a passing build does not eliminate those limitations.
