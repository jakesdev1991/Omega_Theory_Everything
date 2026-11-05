# Q-region data and evolution: next formal target

This is a research specification, not a proposed fundamental evolution law.
The distribution below is the two-site case of the existing diagonal benchmark
in `research/qregion_markov.py`, with correlation `1 − 2q`; the proposed Lean
milestone formalises its data and a specified noise channel rather than inventing
an unrelated simulation. The Planck-profile construction in
`lean_proofs/PlanckProfileMatching.lean` passed Lean and its transitive axiom
audit in CI run 42126178975 at commit `2d7e70e`. It relates two chosen profiles
conditionally; it does not determine physical Φ from a Q-region.

`lean_proofs/QRegionChannel.lean` now proposes the joint distribution and
bit-flip channel, including normalization, unbiased marginals, channel action,
bounds, composition and a nontrivial witness. These declarations passed the
Lean build and transitive kernel-axiom audit in CI run 42127214338 at commit
`e202364`. The mutual-information identity, bounds, endpoints and monotonicity
under the bit-flip channel passed Lean build and kernel audit in CI run
42128816079 at commit `ab6ad24`. Further candidate declarations add normalized
overlap and expose its independent endpoint's incompatibility with the current
strictly positive `QRegion` overlap; these additions passed Lean build and kernel
axiom audit in CI run 42132144272 at commit `c770dce`. A general bridge to
physical Q-region data remains open.

## What the present model actually supplies

`OmegaProtocol.QRegion` in `OmegaAxioms.lean` is a record with two independently supplied real numbers,
`entropy` and `density`, both in `(0, 1]`; it has distinct witnesses. Its `Φ`
is defined as `1` for equal records and as the minimum of the two supplied
`density` fields otherwise. The model's `mutualInformation` is then a function
of those fields and `Φ`; it is not calculated from a joint density matrix. The
structure alone has no transition law or clock. Accordingly, non-constant
model values do not establish a microscopic quantum construction or an
independently derived dynamics.

Two consequences need to be respected in any bridge: `Φ R R = 1`, so the
chosen dynamic scale `ℓ_P0 √(1 − Φ²)` is zero on a self-pair, and `Φ` is
strictly positive even for distinct regions. Thus the model cannot realize a
zero-overlap vacuum pair, and a self-pair is not automatically a physical
position. These are properties of the current definitions, not empirical
predictions. Mapping a pairwise Φ to a local spacetime field requires an explicit
coarse-graining or other justified prescription; it cannot be obtained merely by
setting `R₁ = R₂`.

## First falsifiable formal milestone: a finite-state benchmark

1. Specify a two-bit **classical** joint distribution with unbiased marginals
   and crossover probability `q ∈ [0, 1/2]`: equal outcomes each have
   probability `(1-q)/2`, unequal outcomes each have probability `q/2`.
   Specify the endpoints using the continuous `0 log 0 = 0` convention.
2. Define its mutual information from this joint distribution and marginals,
   and a dimensionless overlap `Φ = I / ln 2`. Prove `0 ≤ Φ ≤ 1`, `Φ = 1`
   at `q = 0`, `Φ = 0` at `q = 1/2`, symmetry under exchanging the bits,
   and at least one non-endpoint witness. Prove or explicitly scope the
   endpoint converses. This is a model-specific choice; do not call it a
   universal COD functional or an entangled quantum-state model.
3. Specify a local stochastic bit-flip channel with noise parameter
   `ε ∈ [0, 1/2]`, yielding `q' = q + ε(1 − 2q)`. Prove that the channel
   preserves normalization and the specified domain, composes as claimed, and
   does not increase this benchmark's mutual information. If a continuous-time
   trajectory is added, its semigroup and rate must be specified explicitly;
   the monotonicity of ℓ_P then follows only for that trajectory and direction.
4. Compare the benchmark to the existing `QRegion` interface. State an explicit
   mapping and its domain, or prove a no-go under proposed compatibility
   requirements. In particular, do not silently identify the benchmark's
   independent-pair zero with the existing `Φ`, which is strictly positive.

**Acceptance:** Lean compiles the construction, endpoint and channel results;
the kernel audit finds no new non-foundational axioms; tests include the
uncorrelated and correlated endpoints and a genuinely changing channel. No
existing model law is removed or ledger status changed merely because a
benchmark exists. Any mismatch with the existing `QRegion` definitions is
reported, not hidden behind a new assumption.

This benchmark makes data and an evolution rule explicit and checkable. It
still leaves the quantum generalization, physical vacuum, spatial/temporal
coarse-graining, and evidence for the choice of channel open. The proposed
Choi-state route in Technical §1.1 is a separate, more ambitious quantum task.
