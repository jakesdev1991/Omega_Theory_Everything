<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE. -->

# Omega Theory v4.0: exploratory information-geometric technical note

**Author:** Jacob See
**Status (2026-10-11):** exploratory research program; not a completed or
empirically validated Theory of Everything.

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
