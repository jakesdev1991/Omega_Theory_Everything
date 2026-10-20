<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE. -->

# Omega Theory v4.0: exploratory information-geometric technical note

**Author:** Jacob See
**Status (2026-10-11):** exploratory research program; not a completed or
empirically validated Theory of Everything.

> **v4.1 catch-up note (2026-10-20, draft on the toe-catchup-lean-citations
> branch).** This edit adds theorem-level Lean 4 citations to the sections
> below, a map of the cross-volume structural-consistency theorems, a
> numbered statement of the Q-region benchmark negative finding, and a
> clearly labelled speculative Part II summarising extension material held
> outside this repository. It changes no claim's status: the status banner
> above and the epistemic ledger in §7 stand unchanged in spirit and, except
> for one added ledger row pointing at §9, in text. The claim-by-claim
> formal anchor for every citation added here is
> [`lean_proofs/PROOF_MAP.md`](lean_proofs/PROOF_MAP.md). Newer ToE drafts
> held in Google Drive (the 5.0 and OmegaAlpha lines) have not yet been
> read into this repository and may supersede parts of this note; nothing
> here should be read as a sync from them.

## Scope and claim level

This note records a proposed route from correlations between subsystems to
spatial geometry, plus a candidate scalar-field model. It does **not** establish
that spacetime, gravity, matter, or cosmic acceleration emerge from quantum
information. The map from Q-region data to the field \(\Phi\), the macroscopic
metric, the field equations, and a novel data-supported prediction are open.

The document distinguishes:

- **Definition/model choice:** stipulated mathematical objects or functions.
- **Conditional result:** follows only after the displayed assumptions.
- **Identified dictionary:** a variable is assigned a form that reproduces a
  known solution; this is not a derivation of that solution from Q-regions.
- **Open:** the required physical construction or evidence is absent.

The current concrete tests are in
[`docs/research/QREGION_MARKOV_MODEL.md`](docs/research/QREGION_MARKOV_MODEL.md),
[`docs/research/ENTANGLED_TFIM_BENCHMARK.md`](docs/research/ENTANGLED_TFIM_BENCHMARK.md),
[`docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md`](docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md),
and [`docs/research/GAUSS_BONNET_SCALARIZATION.md`](docs/research/GAUSS_BONNET_SCALARIZATION.md).

## 1. Q-regions, mutual information, and \(\Phi\)

A Q-region may be represented by a subsystem with Hilbert space \(\mathcal H_i\)
and state \(\rho_i\). For two regions, their quantum mutual information is

\[
I(i:j)=S(\rho_i)+S(\rho_j)-S(\rho_{ij}),
\qquad S(\rho)=-\operatorname{Tr}(\rho\ln\rho).
\]

This definition is standard, but the proposed scalar field \(\Phi(x)\) is not
yet an operationally specified functional of the full set of \(I(i:j)\). The
convention \(0\leq\Phi\leq1\), with \(\Phi=0\) called the vacuum and
\(\Phi\to1\) the high-overlap end, is a model convention in the current
simulation/Lean corpus. In particular, “no shared state” at \(\Phi=0\) is not
established for a physical quantum vacuum.

The explicit diagonal Markov-chain benchmark uses qubit subsystems and
calculates \(I(i:j)\) exactly, but its states are classical mixtures with no
bipartite entanglement. The separate transverse-field Ising-chain benchmark
uses ground states with positive two-spin negativity, so it tests the proposed
mutual-information map in a genuinely entangled family. Both define \(\Phi\)
explicitly for their own state families; neither supplies a universal
operational definition or a physical-vacuum baseline.

> **Formal anchor (Lean 4).** The foundation-level statements this section
> rests on are formalised in two places, with two different strengths.
> In lean_proofs/OmegaUnifiedFoundation.lean, the concrete foundation model
> gives `concreteRelativeEntropy_mono`, `concrete_kms_holds`,
> `fisherMetric_symm`, `fisherMetric_pos`, and
> `klSecondOrder_hessian_eq_fisher` (the Fisher metric as the Hessian of the
> second-order KL expansion). In lean_proofs/OmegaAxioms.lean, the Q-region
> apparatus — `mutualInformation_nonneg`, `mutualInformation_le_quarter`,
> `mutualInformation_eq_zero_iff`, `pairDistance_ultrametric`,
> `distance_pos_iff_mutualInformation_pos`, and
> `bridge_metric_triangle_from_DPI` — is checked only inside the concrete
> minimal model that lean_proofs/PROOF_AUDIT.md classifies as degenerate
> (zero-information / singleton). Those names, and every bridge_-prefixed
> name, are structural consistency checks of the definitions, not physical
> support. The information-to-energy bookkeeping used later in this note is
> formalised separately in lean_proofs/InformationPhysics.lean:
> `information_mass_energy_equivalence`,
> `energy_conservation_during_state_conversion`, `erasureCost_lower_bound`,
> `batchCost_decomposition`, and `batch_affordable_iff`. That file
> formalises the stated model; it does not derive gravity from information.

## 2. Conditional logarithmic chain distance

Let \(0<K_{ij}\leq1\) be a dimensionless **correlation kernel**. If, along a specified
chain,

\[
K_{AC}=K_{AB}K_{BC},
\]

and the chain length is additive and continuous (or monotone) in the link kernel,

\[
d_{AC}=d_{AB}+d_{BC},
\]

then the positive logarithmic solution is

\[
\boxed{d_{ij}=-\ell_P\ln K_{ij}},\qquad \ell_P>0.
\]

The continuity-or-monotonicity hypothesis is required: without it, the additive
Cauchy-type equation admits non-logarithmic solutions.

This is a conditional functional-equation result. It does not by itself prove
that \(d\) is a metric on a spatial manifold, that \(K\) is a physical
correlation observable, or that the construction supplies time and causal
structure.

An earlier version identified \(K\) with \(I(i:j)/I_{\max}\). In the explicit
Markov-chain benchmark, mutual information does not multiply along the chain:
\(I_{i,k}/I_{\max}\ne(I_{i,j}/I_{\max})(I_{j,k}/I_{\max})\). Moreover, zero
mutual information makes \(-\ln K\) infinite. The logarithmic kernel and the
mutual-information statistic must therefore be distinguished; any relation
between them has to be derived from a specified Q-region state family.

The radial notes derive a two-factor spatial expression under additional
locality and regularity assumptions,

\[
h_{ij}=\ell_P(\Phi)^2\gamma_{ij},\qquad
h_{rr}=\ell_P(\Phi)^2\kappa^2,
\]

where \(\gamma\) is the metric of the correlation-decay coordinate. That result
does not determine \(\gamma\) or \(\kappa(\Phi)\). The diagonal Markov toy
model supplies one explicit \(\kappa(\Phi)\); the entangled Ising benchmark
instead fits a finite-range \(\kappa_{\rm eff}\) to a two-point correlator.
Both selected diagnostics show a decay-rate trend opposite to the
Schwarzschild-required trend. Neither establishes a continuum metric or a
universal no-go.

> **Formal anchor (Lean 4).** The conditional statement above is the prose
> form of lean_proofs/LogCorrelationMetric.lean, which proves
> `neg_log_triangle_of_mul_le`, `logDistance_triangle`,
> `logDistance_nonneg`, `logDistance_symm`, and
> `correlation_supermultiplicative` under the multiplicative-kernel
> hypothesis stated in that file. The same file also proves the limit of
> the claim: `upper_multiplicative_bound_fails` is a formal negative
> result showing the stronger universal multiplicative bound does not
> hold. An older prose argument (held outside this repository) claimed the
> triangle inequality follows from the Data Processing Inequality alone;
> it does not — DPI does not yield the multiplicative bound for arbitrary
> triples, which is exactly why the Lean version carries the bound as a
> hypothesis. That prose sketch must not be cited as a proof.

## 3. Chosen operational-scale profile

The profile used by the current Lean model is

\[
\ell_P(\Phi)=\ell_{P0}\sqrt{1-\Phi^2},\qquad 0\leq\Phi\leq1.
\]

The Lean development checks algebraic properties of this chosen function,
such as its endpoint values and monotonicity on the stated interval. Those
proofs do not derive the profile from mutual information or show that it is a
physical Planck length. Under uniform correlation decay it contracts the
spatial radial factor; the radial notes show that it cannot alone reproduce
Schwarzschild. An additional, independently derived correlation metric would
be needed.

> **Formal anchor (Lean 4) — and two models that must not be conflated.**
> The profile above is lean_proofs/DynamicCODScale.lean: the formally
> checked properties include `dynamicPlanckCOD_zero`,
> `dynamicPlanckCOD_one`, `dynamicPlanckCOD_pos`,
> `dynamicPlanckCOD_strictAntiOn`, `dynamicPlanckCOD_zero_iff`,
> `cod_scale_contraction`, and `cod_squared_recovery`. A second, different
> parametrisation also exists and is *not* the profile above:
> lean_proofs/DynamicPlanckScale.lean defines a density-driven Planck
> length, with `dynamicPlanckLength_pos`, `dynamicPlanckLength_le`,
> `planck_scale_contraction_iff`, `informationDensity_pos`, and
> `information_contracts_planck_scale`. Earlier prose presented the two as
> if they were one model. They are separate model choices with separate
> drivers (the field value Φ versus information density); neither is
> derived from the other, and results proved for one do not transfer to
> the other.

**Finding (numbered, Python-only).** Both explicit Q-region diagnostics —
the diagonal Markov-chain benchmark and the genuinely entangled TFIM
benchmark — show a correlation-decay trend *opposite* to the trend the
Schwarzschild dictionary requires. This was previously recorded only in
the ledger below and in docs/research/. It is a computational result about
those two benchmark models: not a Lean theorem, and not a universal no-go
for richer Q-region state families. Its correct status is the one already
given in §7: the macroscopic metric remains Open, with a negative
consistency result against the two models so far made explicit.

## 4. Candidate covariant scalar-matter model

For signature \((-+++ )\), a canonical candidate action is

\[
S=\int d^4x\sqrt{-g}\left[
\frac{R}{16\pi G}-\frac12Z(\Phi)(\nabla\Phi)^2-V(\Phi)\right]
+S_m[g,\psi].
\]

Here the spacetime metric is varied independently and \(\Phi\) is a matter
scalar minimally coupled to it. Variation yields

\[
G_{\mu\nu}=8\pi G\left(T^{(m)}_{\mu\nu}+T^{(\Phi)}_{\mu\nu}\right),
\]

\[
T^{(\Phi)}_{\mu\nu}=Z\nabla_\mu\Phi\nabla_\nu\Phi
-g_{\mu\nu}\left[\tfrac12 Z(\nabla\Phi)^2+V\right],
\]

\[
Z\Box\Phi+\tfrac12Z'(\nabla\Phi)^2-V'=0.
\]

For \(Z(0)>0\), \(V(0)=V'(0)=0\), the constant solution \(\Phi=0\) has
zero scalar stress-energy and the metric equation reduces exactly to Einstein's
equation. This is a **conditional GR limit of the selected action**, not a
derivation of that action from information. The scalar-as-matter interpretation
also differs from the radial Schwarzschild dictionary, where \(\Phi\) is a
coordinate-dependent label of the metric. Under regularity and convex-potential
conditions, the minimally coupled scalar has no nonconstant static black-hole
hair; the two interpretations cannot be identified without additional theory.

A separate, unselected curvature-coupled gravitational-scalar alternative is
also documented there; it has an exact GR branch but does not yet derive a
Q-region source or a stable black-hole profile. The two interpretations remain
distinct rather than being combined by notation.

The full variation, conservation laws, linear stability conditions, static
no-hair argument, FLRW equations, and limits of these candidate branches are
recorded in
[`docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md`](docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md).

## 5. Black-hole information loss and cosmology: underived ansatz

The depletion equation

\[
\dot I=-\gamma A_{BH}^{\kappa} I
\]

is a phenomenological assumption in the simulation suite. The classical
non-decrease of horizon area does not by itself prove that a global mutual
information measure decreases at a specified rate. The further relation
\(H\propto A_{BH}^{\kappa}\) requires a separate definition connecting
\(I\) to the scale factor and a covariant account of stress-energy and
conservation; it does not follow from the depletion equation alone.

Accordingly, descriptions such as “information shredding proves expansion,”
“smoking gun,” or “no cosmological constant is required” are not established
results. `sim6_v14_depletion.py` calibrates \(\gamma\) to a selected value of
\(H_0\); reproducing that calibration target is not an independent prediction
or validation. A distinct prediction and a pre-registered evaluation against
independent data are not yet available.

## 6. Other proposed applications

Biological “active shredding,” life-as-information-processing, and similar
extensions are speculative analogies in this program. A computational
population model, including one inspired by the LTEE, cannot on its own verify
a physical theory of life or establish that a measured overlap change is a
cause of adaptation. These ideas are not evidence for the gravitational or
cosmological claims.

Agent software, token pilots, and the proposed economy are likewise separate
engineering projects; their implementation status is not evidence for the
physics.

## 7. Current epistemic ledger

| Claim or component | Current status |
|---|---|
| Q-region mutual information | Well-defined for a specified quantum state; the physical Q-region family is unspecified |
| \(\Phi\) as a functional of Q-region data | Open in general; two model-specific toy definitions are tested in separate benchmarks |
| Logarithmic distance | Conditional on positive multiplicative kernel and regular additive length; the kernel is not generally pairwise mutual information |
| \(\ell_P=\ell_{P0}\sqrt{1-\Phi^2}\) | Chosen profile with formally checked mathematical properties; physical derivation open |
| Macroscopic spatial metric | Open; the explicit Markov and entangled-chain diagnostics both show a decay-rate trend opposite to the Schwarzschild-required trend; neither is a continuum metric derivation |
| Scalar stress-energy and field equation | Derived from the stated candidate action; the action is not derived from Q-region axioms |
| GR limit | Recovered on the constant-scalar branch of the stated actions, under their separate assumptions |
| Curvature-coupled gravitational scalar | Provisional next-test branch with a linear static Schwarzschild zero mode; nonlinear hair, stability, constraints, and Q-region origin open; fundamental ontology unsettled |
| Black-hole depletion and \(H(A_{BH})\) | Phenomenological ansatz, not a covariant derivation or independent observation |
| Theory of Everything claim | Not supported by the current results |
| Formal unification layer (§9, added 2026-10-20) | Structural-consistency results inside the shared formal corpus; not a physical unification and not evidence for any row above |

## 8. Reproduction and audit trail

```bash
python research/qregion_markov.py
python -m pytest tests/test_qregion_markov.py -q
python research/quantum_ising_chain.py
python -m pytest tests/test_quantum_ising_chain.py -q
python research/gauss_bonnet_scalarization.py
python -m pytest tests/test_gauss_bonnet_scalarization.py -q
python Sim7_Radial_Metric.py --no-plots
python research/verify_derivations.py
python -m pytest tests/test_verify_derivations.py -q
python research/verify_covariant.py
python research/verify_tfim.py
python research/check_latex_citations.py
python research/check_latex_structure.py
python research/consistency_gate.py
```

`check_latex_citations.py` verifies that each LaTeX volume's status box points
at a Lean module that exists and cites declarations that exist, so the volumes
cannot drift away from the code they describe. `check_latex_structure.py`
checks brace balance, environment nesting and math delimiters in the same
files, since no TeX toolchain is available here to compile them. `consistency_gate.py` runs every
check above in one command and reports which ones passed, failed or were
skipped; the Lean build, the kernel audit and `lean_proofs/check_mathlib_names.py`
(the cited-name check against the pinned mathlib and Lean sources) are CI-only
and it says so.

The Lean source and CI audit are documented in
[`lean_proofs/README.md`](lean_proofs/README.md) and
[`lean_proofs/PROOF_AUDIT.md`](lean_proofs/PROOF_AUDIT.md). A successful kernel
check establishes the formal theorem under its assumptions; it does not
establish that the assumptions describe nature.

## 9. The formal unification layer: cross-volume theorems (added 2026-10-20)

lean_proofs/OmegaProtocol.lean contains the corpus's only explicitly
unifying formal statements: theorems that restate results from two
different volumes inside the shared formal corpus and check that the
restatements agree. Their correct label is **structural-consistency
results**. They show the formal corpus is internally coherent where
claimed; they do not derive one physical domain from another, and they
are not evidence for the Open rows in §7.

- Holography ↔ ER=EPR: `cross_vol09_vol22_er_epr_dictionary` and
  `cross_vol09_vol22_er_epr_dictionary_iff`.
- General relativity ↔ cosmology: `cross_vol05_vol07_flrw`.
- Electromagnetism ↔ general relativity: `cross_vol02_vol05_em_stress_energy`.
- Thermodynamics ↔ black-hole thermodynamics: `cross_vol03_vol08_bh_thermo`.
- ER=EPR ↔ black-hole information: `cross_vol22_vol25_er_epr_bh_info`.
- Consciousness ↔ quantum reference frames: `cross_vol27_vol49_consciousness_qrf`
  and `cross_vol49_all_observer_dependent`.
- Capstones: `omega_protocol_structural_bundle` and
  `omega_theory_structural_consistency`.

Several supporting components in that file are bridge_-prefixed and carry
the minimal-model caveat of §1: they are structural checks only. The full
claim-by-claim list, including those components, is in
lean_proofs/PROOF_MAP.md §2.

## 10. Part II — Extensions: Life & Mind (speculative; added 2026-10-20)

**Status of everything in this section: speculative / analogical.** Nothing
here is evidence for the gravitational or cosmological claims above, and
§6's quarantine stands. This section exists because the broader ToE
material held outside this repository (Drive, ToE-V.Omega 4.4 §§6–7)
develops these extensions at length while this note previously mentioned
them only to set them aside; a reader of this note should be able to see
what they are and exactly how little formal support they currently have.

- **Life.** The Drive material proposes an *Active Shredding Operator* for
  living systems and a *Metabolic Inequality* (an information-processing
  cost bound for observation and adaptation), and presents the E. coli
  long-term evolution experiment's Cit+ innovation as a claimed
  verification case, with RCOD-style redundancy pruning as the proposed
  rule for living systems. None of these three items is formalised in
  Lean as stated, and the LTEE case is an interpretation of published
  experimental history, not a result derived from the model.
- **Mind.** The same material proposes a *Q-Systemic Self*: a self-model
  described in overlap/entropy terms — anxiety as a low-overlap,
  high-entropy mode, coherence as a high-overlap state, and psychological
  "reboots" as crossings of an innovation horizon. This is a case-study
  narrative, not a formalised or tested model.
- **The ambition, in its own words.** The Drive evidence map summarises
  the intended scope as: a single scalar informational field Φ underlies
  geometry, gravity, cosmology, life, and mind. In this repository that
  sentence is a research ambition, not a result; §7's ledger is unchanged
  by it.

**Neighbouring formal work — not formalisation of the above.** Toy models
in adjacent domains do exist in the Lean corpus and are worth naming
precisely so they are not over-read: lean_proofs/Vol27_Consciousness.lean
formalises an integrated-information toy model (`integrated_info_nonneg`,
`phi_nonneg`, `experiences_of_strict_integration`, `cortical_phi_pos`);
lean_proofs/Vol28_EvolutionaryAlgorithms.lean formalises a simple
selection model (`fitness_bounded_by_length`, `selection_never_decreases`);
lean_proofs/Vol29_EcosystemDynamics.lean formalises Lotka–Volterra
equilibria (`lotka_volterra_equilibrium`, `coexistence_positive`); and
lean_proofs/Vol49_QuantumReferenceFrames.lean formalises frame-change
facts (`frame_preserves_norm`, `frame_round_trip`) used by the §9
consciousness ↔ reference-frames structural result. These are formal
statements inside their own toy models. They do not formalise the Active
Shredding Operator, the Metabolic Inequality, or the Q-Systemic Self.
