<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE. -->

# PROOF_MAP — claim → Lean 4 theorem map (v4.1 catch-up draft, 2026-10-20)

This map connects the claims made in the ToE-facing documents
(Omega_Theory_v4.0_Technical.md, Omega_Theory_v4.0_Radial_Metric.md,
Omega_Theory_Laymans_Guide.md, README.md) to the Lean 4 statements that
actually formalise them — or records, explicitly, that no such statement
exists. It was drafted on 2026-10-20 to close the citation gap found by a
cross-reference scan: the core ToE documents previously named exactly one
real theorem identifier, while the Lean corpus contains 923 theorem/lemma
names across 69 files.

**How to read the Status column**

- **Formalised** — a Lean theorem/lemma in the named file states this, under
  its encoded assumptions. Kernel checking establishes the formal statement,
  not that its assumptions describe nature.
- **Conditional** — formalised only after extra hypotheses that are stated in
  the Lean file or the prose document (e.g. a multiplicative kernel).
- **Structural-check-only** — the statement lives in, or is proved relative
  to, the concrete minimal model of lean_proofs/OmegaAxioms.lean, which the
  repo's own audit (lean_proofs/PROOF_AUDIT.md) classifies as a degenerate
  zero-information / singleton model. Names with the bridge_ prefix are, by
  the repo's proof-honesty policy, structural consistency checks inside that
  model. They are never physical or empirical support for a claim.
- **Python-only** — checked or computed only by Python code in research/;
  no Lean theorem exists.
- **Open** — no formal or computational result settles this.
- **Ansatz** — a phenomenological assumption written down for simulation;
  not derived.

Theorem names below are verbatim identifiers from the named Lean files.
Where a prose source and a Lean file disagree, the Lean file wins.

---

## 1. Tier 1 — core mechanism files

### 1.1 Information ↔ mass/energy (Landauer–Einstein bridge)

| Claim (prose) | Lean file | Theorem(s) | Status |
|---|---|---|---|
| Information has an associated mass/energy in the stated model | lean_proofs/InformationPhysics.lean | `information_mass_energy_equivalence` | Formalised (model-internal) |
| State conversion conserves total energy in the model | lean_proofs/InformationPhysics.lean | `energy_conservation_during_state_conversion` | Formalised (model-internal) |
| Information mass behaves linearly / additively / monotonically in its argument | lean_proofs/InformationPhysics.lean | `informationMass_linear`, `informationMass_add`, `informationMass_strictMono` | Formalised |
| Erasure costs at least the Landauer energy; saturation and strictness are characterised | lean_proofs/InformationPhysics.lean | `landauerEnergy_pos`, `erasureCost_lower_bound`, `erasureCost_saturates_iff`, `erasureCost_strict_iff`, `idealErasure_cost`, `not_every_erasure_is_ideal` | Formalised |
| Batch erasure cost decomposes, and affordability of a batch is decidable from its parts | lean_proofs/InformationPhysics.lean | `batchCost_decomposition`, `batchCost_lower_bound`, `batchCost_saturates_iff`, `batchCost_append`, `erasure_energy_bookkeeping`, `batch_affordable_iff`, `affordable_batch_has_affordable_prefix` | Formalised |

Note: these results formalise the bookkeeping of the stated
information-physics model. They do not by themselves derive gravity, and
the Technical note continues to label the step from information to a
macroscopic metric as Open.

### 1.2 Logarithmic correlation distance — and its formal limit

| Claim (prose) | Lean file | Theorem(s) | Status |
|---|---|---|---|
| Negative log of a multiplicative kernel is additive / satisfies the triangle inequality | lean_proofs/LogCorrelationMetric.lean | `neg_log_triangle_of_mul_le`, `logDistance_triangle` | Conditional (multiplicative-kernel hypothesis) |
| The log distance is a well-behaved distance on distinct points | lean_proofs/LogCorrelationMetric.lean | `logDistance_eq_abs`, `logDistance_nonneg`, `logDistance_eq_zero_iff`, `logDistance_symm`, `distinct_points_distance` | Conditional |
| Correlation is positive, bounded by one, and supermultiplicative in the model | lean_proofs/LogCorrelationMetric.lean | `correlation_pos`, `correlation_le_one`, `correlation_supermultiplicative` | Formalised (model-internal) |
| The stronger universal multiplicative upper bound does **not** hold | lean_proofs/LogCorrelationMetric.lean | `upper_multiplicative_bound_fails` | Formalised **negative result** |

This file is the honest formal replacement for the prose "Formal Proofs"
argument (held outside this repo) that claimed the triangle inequality
follows from the Data Processing Inequality alone. It does not: DPI bounds
mutual information through a Markov chain and does not by itself yield the
multiplicative bound for arbitrary triples. The Lean version states the
bound as a hypothesis, proves the conditional result, and separately proves
that the stronger bound fails. The prose sketch should not be cited as a
proof.

### 1.3 Foundations: modular theory, KMS, Fisher information geometry

| Claim (prose) | Lean file | Theorem(s) | Status |
|---|---|---|---|
| Relative entropy is monotone in the concrete foundation model | lean_proofs/OmegaUnifiedFoundation.lean | `concreteRelativeEntropy_mono` | Formalised (concrete model) |
| The KMS condition holds in the concrete foundation model | lean_proofs/OmegaUnifiedFoundation.lean | `concrete_kms_holds` | Formalised (concrete model) |
| The Fisher metric is symmetric and positive in the model | lean_proofs/OmegaUnifiedFoundation.lean | `fisherMetric_symm`, `fisherMetric_pos` | Formalised |
| The second-order KL expansion has the Fisher metric as its Hessian | lean_proofs/OmegaUnifiedFoundation.lean | `klSecondOrder_first_deriv`, `klSecondOrder_hessian_eq_fisher` | Formalised |

### 1.4 Q-region apparatus in the minimal model (OmegaAxioms)

**Standing caveat for this whole subsection.** lean_proofs/OmegaAxioms.lean
is a degenerate zero-information / singleton model. Everything below is a
structural consistency check of the definitions inside that model. None of
it is evidence that a physical Q-region family behaves this way, and the
repo's audits say so explicitly.

| Claim (prose) | Lean file | Theorem(s) | Status |
|---|---|---|---|
| Mutual information is nonnegative, symmetric, self-zero, and bounded by a quarter in the model | lean_proofs/OmegaAxioms.lean | `mutualInformation_nonneg`, `mutualInformation_symm`, `mutualInformation_self`, `mutualInformation_le_quarter`, `mutualInformation_eq_zero_iff`, `mutualInformation_pos_iff` | Structural-check-only |
| Pair distance is an ultrametric-style quantity tied to mutual information | lean_proofs/OmegaAxioms.lean | `pairDistance_eq`, `pairDistance_nonneg`, `pairDistance_antitone`, `pairDistance_pos_iff`, `pairDistance_ultrametric`, `distance_eq_zero_iff`, `distance_pos_iff_mutualInformation_pos` | Structural-check-only |
| The Planck factor in the model equals the COD profile and is antitone | lean_proofs/OmegaAxioms.lean | `planckFactor_eq_dynamicPlanckCOD`, `planckFactor_antitone` | Structural-check-only |
| A metric triangle statement can be recovered from a DPI-style hypothesis **inside the model** | lean_proofs/OmegaAxioms.lean | `bridge_metric_triangle_from_DPI` | Structural-check-only — see §1.2 for why this does not rescue the prose DPI argument in general |
| Novelty / processing-speed bookkeeping in the model | lean_proofs/OmegaAxioms.lean | `informationNovelty_nonneg`, `informationNovelty_le_one`, `processingSpeed_le_baseRate`, `speed_reduction_of_novelty` | Structural-check-only |
| Freezing / impedance characterisations in the model | lean_proofs/OmegaAxioms.lean | `informationalImpedance_pos_iff`, `bridge_isFrozen_iff`, `asymmetryTensor_eq_forward_of_reverseFlux_zero` | Structural-check-only |
| Algebra-level facts about the model's operator algebra | lean_proofs/OmegaAxioms.lean | `omegaAlgebra_commutative`, `every_functional_is_tracial`, `connes_III1_ne_I1`, `no_finite_trace_of_infinite_identity`, `no_pure_qregion_states` | Structural-check-only |
| Witness constructions exist inside the model | lean_proofs/OmegaAxioms.lean | `witness_mutualInformation_eq`, `witness_distance_pos`, `witness_codProfile_pos`, `bridge_mutualInformation_le_max`, `chainOverlapDensity_nonneg` | Structural-check-only |

### 1.5 The two Planck-scale models — distinct, not interchangeable

The prose documents previously presented only the first of these. They are
different parametrisations and must not be conflated.

| Model | Lean file | Theorem(s) | Status |
|---|---|---|---|
| Φ-driven (COD) profile: the chosen profile ℓ_P(Φ) = ℓ_P0·sqrt(1 − Φ²) | lean_proofs/DynamicCODScale.lean | `dynamicPlanckCOD_zero`, `dynamicPlanckCOD_one`, `dynamicPlanckCOD_nonneg`, `dynamicPlanckCOD_pos`, `dynamicPlanckCOD_le`, `dynamicPlanckCOD_strictAntiOn`, `dynamicPlanckCOD_injOn`, `dynamicPlanckCOD_zero_iff`, `dynamicPlanckCOD_baseline_iff`, `dynamicPlanckCOD_even`, `dynamicPlanckCOD_not_globally_injective`, `cod_scale_contraction`, `cod_monotonic_contraction`, `cod_squared_recovery` | Formalised properties of a **chosen** profile; the profile itself is a model choice, not derived |
| Density-driven Planck length: scale contracts as information density rises | lean_proofs/DynamicPlanckScale.lean | `dynamicPlanckLength_pos`, `dynamicPlanckLength_le`, `dynamicPlanckLength_strictAntiOn`, `dynamicPlanckLength_injOn`, `planck_scale_contraction`, `planck_scale_contraction_iff`, `planck_scale_baseline_iff`, `informationDensity_pos`, `information_contracts_planck_scale` | Formalised (separate model; not the Φ profile) |

### 1.6 Radial metric and dilation

| Claim (prose) | Lean file | Theorem(s) | Status |
|---|---|---|---|
| The ℓ_P-only radial factor contracts: g_RR ≤ 1 | lean_proofs/RadialDilation.lean | `gRR_lpOnly_le_one`, `gRR_lpOnly_eq_one_imp`, `gRR_lpOnly_zero` | Formalised |
| The ℓ_P-only Misner–Sharp mass is negative (zero only in the trivial case) | lean_proofs/RadialDilation.lean | `misnerSharpLPOnly_neg`, `misnerSharpLPOnly_zero` | Formalised |
| The κ required by the Schwarzschild dictionary is positive and outruns the inverse ℓ_P factor | lean_proofs/RadialDilation.lean | `kappaRequired_pos`, `kappaRequired_zero`, `kappa_lp_product`, `kappa_outruns_inverse_lp`, `conformalFactor_strictMono` | Formalised (dictionary-dependent) |
| Isotropic conformal reproduction and areal reciprocity hold for the dictionary | lean_proofs/RadialDilation.lean | `isotropic_conformal_reproduction`, `areal_reciprocity`, `gRR_schwarzschild_gt_one`, `gRR_schwarzschild_zero` | Formalised (Identified dictionary) |
| The dilation picture reconciles with the COD ratio | lean_proofs/RadialDilation.lean | `lpRatio_quadratic_deviation`, `lpRatio_eq_cod_ratio`, `dilation_reconciliation` | Formalised |
| The golden-ratio bottleneck: location, gap, and bound | lean_proofs/RadialMetric.lean | `phiStar_pos`, `phiStar_golden`, `golden_bottleneck_gap`, `golden_bottleneck_bound`, `golden_bottleneck_eq_iff` | Formalised |
| Finite-chain length behaviour under the two-factor law | lean_proofs/RadialMetric.lean | `chainLength_mono`, `chainLength_const_scale`, `chainLength_error`, `chainLength_factorization_iff` | Conditional |
| Areal radius at vacuum / horizon, lapse identity, COD profile matching the lapse | lean_proofs/RadialMetric.lean | `arealRadius_at_vacuum`, `arealRadius_at_horizon`, `lapse_conformal_identity`, `cod_profile_matches_lapse` | Formalised (Identified dictionary) |

---

## 2. Tier 2 — cross-volume unification theorems (OmegaProtocol)

These are the repo's only explicitly unifying formal statements. Their
correct label is **structural-consistency results**: they show that the
restatements of the individual volumes, inside the shared formal corpus,
are mutually consistent and inter-derivable where claimed. They are not
derivations of one physical theory from another, and several are built
from bridge_-prefixed components, which carry the §1.4 caveat.

| Linked domains | Lean file | Theorem(s) | Status |
|---|---|---|---|
| Electromagnetism ↔ general relativity (stress-energy) | lean_proofs/OmegaProtocol.lean | `cross_vol02_vol05_em_stress_energy` | Structural-consistency |
| Thermodynamics ↔ black-hole thermodynamics | lean_proofs/OmegaProtocol.lean | `cross_vol03_vol08_bh_thermo` | Structural-consistency |
| General relativity ↔ cosmology (FLRW) | lean_proofs/OmegaProtocol.lean | `cross_vol05_vol07_flrw` | Structural-consistency |
| Holography ↔ ER=EPR dictionary | lean_proofs/OmegaProtocol.lean | `cross_vol09_vol22_er_epr_dictionary`, `cross_vol09_vol22_er_epr_dictionary_iff` | Structural-consistency |
| ER=EPR ↔ black-hole information | lean_proofs/OmegaProtocol.lean | `cross_vol22_vol25_er_epr_bh_info` | Structural-consistency |
| Consciousness ↔ quantum reference frames | lean_proofs/OmegaProtocol.lean | `cross_vol27_vol49_consciousness_qrf`, `cross_vol49_all_observer_dependent` | Structural-consistency |
| Bridge cross-checks (coupling / entropy / metric) | lean_proofs/OmegaProtocol.lean | `bridge_cross_v01_v04_coupling_symm`, `bridge_cross_v04_v06_entropy_nonneg`, `bridge_cross_v08_v09_mutual_info_nonneg`, `bridge_cross_v10_v13_metric_self_zero`, `bridge_cross_v53_entropy_nonneg`, `bridge_cross_v54_metric_self_zero` | Structural-check-only |
| Capstones: the corpus bundles consistently | lean_proofs/OmegaProtocol.lean | `omega_protocol_structural_bundle`, `omega_theory_structural_consistency` | Structural-consistency |
| Supporting identities used by the cross-volume layer | lean_proofs/OmegaProtocol.lean | `fisher_metric_eq_kl_hessian`, `weyl_commutator_poisson`, `twoLevel_kms_temperature_unique`, `real_conservation_of_energy_momentum`, `page_curve` | Formalised (model-internal) |

---

## 3. Representative volume highlights (Vol01–Vol54)

Selection principle: the audited, non-degenerate restatements surfaced in
lean_proofs/README.md and the proof audits. The full per-volume tables stay
in lean_proofs/README.md; this is the ToE-facing short list. Inclusion here
does not change any volume's scope: several volumes are exploratory or toy
models, and their theorems are statements inside those models.

| Volume | Lean file | Highlight theorem(s) | Status |
|---|---|---|---|
| Vol01 Classical Mechanics | lean_proofs/Vol01_ClassicalMechanics.lean | `conservation_of_energy` | Formalised |
| Vol03 Thermodynamics | lean_proofs/Vol03_Thermodynamics.lean | `twoLevel_kms_iff_eq`, `twoLevel_kms_zeroth_law`, `carnot_efficiency_lt_one`, `gibbsState_not_tracial` | Formalised |
| Vol05 General Relativity | lean_proofs/Vol05_GeneralRelativity.lean | `zeroCurvature_einstein_field_equations`, `zeroCurvature_bianchi_identity`, `real_conservation_of_energy_momentum` | Formalised (zero-curvature model; not a variation of the Technical §4 action) |
| Vol07 Cosmology | lean_proofs/Vol07_Cosmology.lean | `flat_friedmann_critical_density_eq`, `redshift_positive_of_expansion` | Formalised |
| Vol08 Black Hole Thermodynamics | lean_proofs/Vol08_BlackHoleThermodynamics.lean | `full_first_law_equivalence`, `kms_beta_hawking`, `kms_exterior_vacuum` | Formalised |
| Vol09 Holographic Principle | lean_proofs/Vol09_HolographicPrinciple.lean | `entangled_iff_correlation_det_ne_zero`, `isProduct_iff_correlation_det`, `er_epr_iff`, `bridge_entanglement_entropy_eq_bh_entropy` | Formalised; last item structural-check-only |
| Vol12 Quantum Information | lean_proofs/Vol12_QuantumInformation.lean | `no_cloning_theorem`, `bellPair_strictly_below_capacity` | Formalised |
| Vol13 Quantum Gravity | lean_proofs/Vol13_QuantumGravity.lean | `wheeler_dewitt_balance`, `constraint_not_automatic` | Formalised |
| Vol23 Measurement Problem | lean_proofs/Vol23_MeasurementProblem.lean | `dephase_coherence_decay`, `complete_decoherence_vanishes`, `plusState_decohered_purity` | Formalised |
| Vol24 Non-Locality | lean_proofs/Vol24_NonLocality.lean | `singlet_chsh_attains_tsirelson`, `singlet_violates_classical_bound` | Formalised |
| Vol25 Black Hole Information | lean_proofs/Vol25_BlackHoleInformation.lean | `pure_marginal_entropies_equal` | Formalised |
| Vol27 Consciousness | lean_proofs/Vol27_Consciousness.lean | `integrated_info_nonneg`, `phi_nonneg`, `experiences_of_strict_integration`, `cortical_phi_pos` | Formalised (integrated-information toy model) |
| Vol28 Evolutionary Algorithms | lean_proofs/Vol28_EvolutionaryAlgorithms.lean | `fitness_bounded_by_length`, `selection_never_decreases` | Formalised (toy selection model) |
| Vol29 Ecosystem Dynamics | lean_proofs/Vol29_EcosystemDynamics.lean | `lotka_volterra_equilibrium`, `coexistence_positive` | Formalised |
| Vol33 AGI | lean_proofs/Vol33_AGI.lean | `general_dominates_max_demand`, `specialization_gap` | Formalised (abstract model) |
| Vol34 Quantum Computing | lean_proofs/Vol34_QuantumComputing.lean | `matH_unitary`, `matH_preserves_norm`, `hadamard_conjugates_X_to_Z`, `interpGate_injective` | Formalised |
| Vol36 Topos Theory | lean_proofs/Vol36_ToposTheory.lean | `skel_subobject_classifier`, `skel_subobject_pullback_universal`, `skel_has_products` | Formalised |
| Vol38 Economics | lean_proofs/Vol38_Economics.lean | `market_clearing`, `market_equilibrium_exists_unique` | Formalised (linear market model) |
| Vol49 Quantum Reference Frames | lean_proofs/Vol49_QuantumReferenceFrames.lean | `frame_preserves_norm`, `frame_round_trip` | Formalised |
| Vol53 Universal Compiler | lean_proofs/Vol53_UniversalCompiler.lean | `compile_correct`, `compile_correct_stack`, `identity_not_representable` | Formalised; last item is a **negative** expressiveness result |

The remaining volumes (Vol02, Vol04, Vol06, Vol10, Vol11, Vol14–Vol22,
Vol26, Vol30–Vol32, Vol35, Vol37, Vol39–Vol48, Vol50–Vol52, Vol54) are
inventoried, with their theorem lists, in lean_proofs/README.md and the
§1a inventory of the catch-up analysis. No theorem from them is cited in
the core ToE documents yet; that is a documented gap, not an oversight of
this map.

Out-of-physics formal work, kept separate by design: the security and
budget-pacer files lean_proofs/APPA_Context_Branching.lean and
lean_proofs/CBwK_Budget_Pacer.lean. Their invariants are formalised, but
they are protocol/engineering results, not physics, and no ToE claim below
rests on them.

---

## 4. Tier 4 — reverse gaps: claims with NO Lean theorem

These rows are the most important honesty feature of this map. No citation
effort can close them; they need new Lean work or continued explicit
labelling.

| Claim | Where it appears | Lean theorem | Status |
|---|---|---|---|
| Black-hole information depletion causes / drives cosmic expansion | Technical §5; Sim2_Cosmology.py; sim6_v14_depletion.py | **None exists in any Lean file** | Ansatz |
| H₀ = 73.5 km/s/Mpc as an Omega prediction | sim6_v14_depletion.py | **None** — the value is a calibration target fitted by the simulation, not a prediction | Ansatz / calibration |
| The covariant scalar-matter action, its GR limit, and the static no-hair argument | Technical §4; docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md | **None for the action as stated** — verification is by research/verify_covariant.py; the Vol05 zero-curvature results are a different model statement | Python-only |
| Curvature-coupled (scalar–Gauss–Bonnet) branch and its linear zero mode at η/M² ≈ 0.725613 | docs/research/GAUSS_BONNET_SCALARIZATION.md | **None** — no Lean file formalises Gauss–Bonnet scalarization | Python-only |
| Q-region Markov-chain benchmark behaviour | docs/research/QREGION_MARKOV_MODEL.md; research/qregion_markov.py | **None** — and the Lean minimal model of §1.4 is a different, degenerate model; citing it here would be incorrect | Python-only |
| Q-region entangled TFIM benchmark behaviour | docs/research/ENTANGLED_TFIM_BENCHMARK.md; research/quantum_ising_chain.py | **None** | Python-only |
| **Negative finding:** both Q-region diagnostics show a correlation-decay trend **opposite** to the Schwarzschild-required trend | Technical §§1–2 ledger; docs/research/ (both benchmarks) | **None** — this is a computational result about the two benchmark models, integrated into the Technical note as a numbered finding by the v4.1 catch-up edit | Python-only (negative result) |
| Φ as an operational functional of a physical Q-region state family | Technical §1 | **None** | Open — an adopted external candidate route now exists (Route B, Technical §1.1: EDG's relative-entropy-density construction, Ðulović, Zenodo 17227471, Sep 2025), but no Φ↔s dictionary has been constructed, so this status is unchanged. See §5 below |
| Macroscopic metric derived from Q-region data | Technical §§2–4; Radial Metric notes | **None** | Open |
| Life / mind extensions (Active Shredding Operator, Metabolic Inequality, Q-Systemic Self) | Drive ToE 4.4 §§6–7; summarised in Technical Part II by the v4.1 catch-up edit | **None for the extensions as stated** — neighbouring formal toy models exist (Vol27, Vol28, Vol29, Vol49 rows above) but they do not formalise the Drive claims | Speculative / analogical |

---

## 5. External results adopted as building blocks (added 2026-10-20)

Per this programme's stated principle (unification over originality;
build on published work with full credit), the results below are
**adopted** — used, not merely cited — in the ToE-facing documents.
They are not Lean results, not results of this programme, and not
validated physics; each carries its status in its own literature.
Adoption does not change any status elsewhere in this map.

| External result | Source | Where used in Omega | Status |
|---|---|---|---|
| Pointwise scalar defined from quantum information: s(x) := (4πGℏ/c³)·∂²_λ S_rel(ρ‖σ) per unit transverse area, s ≥ 0 from relative-entropy monotonicity/convexity (data processing) | Ðulović, *Entanglement Density Gravity*, Zenodo record 17227471, 30 Sep 2025, §§4–4.1, Thm 4.1 (sketch-level) | Technical §1.1 "Route B" candidate for specifying the Φ-functional; Technical §11 item 1 | Adopted candidate route (external preprint, not peer-reviewed). Φ-functional remains **Open**: no Φ↔s dictionary exists |
| Constraint checklist for an information-defined gravitational scalar: WEP/fifth-force safety, QNEC compatibility, GW170817 luminality (c_T = 1), Cassini PPN γ = 1, generalised second law | Ðulović, EDG, §§7–9, 14 | Adopted as the criteria of the covariant test gate, Technical §4 | Adopted criteria (external preprint; EDG's compliance is its own self-assessment; EDG's dynamics are postulated, not derived) |
| Thawing dark energy with w_s = −1 + αṡ²/ρ_s ≥ −1, never phantom-crossing; bound α ≳ λ_s²/(3ϵ_w) | Ðulović, EDG, §§10–12, Eqs. 23, 26 | Discriminating test against the depletion ansatz, Technical §5; registered as PRED-WZ-01 in docs/PREDICTIONS_REGISTER.md | Adopted discriminating test (conditional; no fit performed in EDG or here) |
| Laboratory channel: atom-interferometer phase ΔΦ from an engineered entanglement-gradient slab, ~10⁻¹⁰–10⁻⁶ rad over EDG's α benchmarks; null result bounds α | Ðulović, EDG, §§11.2, 13, Eqs. 30, 32–33 | Registered as PRED-EDG-LAB-01 in docs/PREDICTIONS_REGISTER.md | External / conditional. Blocking: no physical realisation or achievable value of the gradient g_s is known in EDG or elsewhere |
| Coarse-grained mutual information defines a geodesic distance in an emergent hyperbolic-like geometry: M(y) ∝ A·e^(−α·d_geo(y)) in a disordered zigzag graphene nanoribbon (fractional e/2 charges) | Le, Lee & Yang, *Quantum Entanglement of Anyonic Charges and Emergent Spacetime Geometry*, arXiv:2512.15256, submitted 17 Dec 2025 (v2, 28 Jan 2026), §V, Eq. 16 | Building block for the MI-to-distance strategy of Technical §2; Technical §11 item 2 | Adopted building block (arXiv preprint). Concrete material model only; supplies no gravitational dynamics |
| Field-theory version of ER=EPR: Gaussian disorder average yields a wormhole; large-N equivalence of quenched and annealed averages in the spatially random Yukawa–SYK model | Sin & Wang, *ER=EPR and Strange Metals from Quantum Entanglement: Disorder theory vs quantum gravity*, arXiv:2503.19431, submitted 25 Mar 2025 (v3, 7 Aug 2025) | Comparison model cross-referenced at the ER=EPR dictionary discussion, Technical §§9, 11 item 3 | Adopted comparison model (arXiv preprint). Not support for the §2 structural dictionary itself |
| Geometric ER=EPR realisation: regular wormholes from non-local gravitational self-energy; only the zero-throat geometry is compatible with an entanglement-induced ER bridge under non-traversability | Jusufi, Lobo, Saridakis & Singleton, *Emergence of ER=EPR from non-local gravitational energy*, arXiv:2512.05022, submitted 4 Dec 2025 | Geometric-side comparison model, Technical §§9, 11 item 4; external counterpart adjacent to the §5 depletion thread (replica wormholes, entanglement-driven dark energy) | Adopted comparison model (arXiv preprint). Transition dynamics out of its own stated scope |

---

## 6. Maintenance rule

When a ToE-facing document cites a formal result, add or update its row
here in the same change. When a claim has no row, it has no formal anchor
and must carry its prose status tag (Definition / Conditional / Identified /
Open / Ansatz / Speculative) in the document itself.
