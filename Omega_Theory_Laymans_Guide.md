<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE. -->

# An exploratory information-geometric research program
## A cautious reader's guide to the Omega v4.0 ideas

**Jacob See**
**Status:** revised 2026-10-12
**Companions:** [`Omega_Theory_v4.0_Technical.md`](Omega_Theory_v4.0_Technical.md),
[`Omega_Theory_v4.0_Radial_Metric.md`](Omega_Theory_v4.0_Radial_Metric.md),
[`docs/research/QREGION_MARKOV_MODEL.md`](docs/research/QREGION_MARKOV_MODEL.md),
[`docs/research/ENTANGLED_TFIM_BENCHMARK.md`](docs/research/ENTANGLED_TFIM_BENCHMARK.md),
[`docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md`](docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md), and
[`docs/research/GAUSS_BONNET_SCALARIZATION.md`](docs/research/GAUSS_BONNET_SCALARIZATION.md).

> **Bottom line:** Omega is not currently a complete or experimentally
> established Theory of Everything. It is a collection of information-geometric
> hypotheses, mathematical models, simulations, and open questions. A proof
> about a chosen model is not evidence that the model describes the universe.

## 1. The starting idea: relationships between subsystems

A **Q-region** is proposed to be a bounded quantum subsystem with a state. For
two specified subsystems, quantum mutual information measures how much
information their joint state contains beyond their separate states. It is a
well-defined quantity once the subsystems and their quantum state have been
specified.

The larger physical proposal is that patterns of these correlations might
help characterize geometry. That proposal is not yet a derivation: the
repository does not identify a physical family of Q-region states or an
operational rule that turns all their mutual informations into a local field
\(\Phi(x)\).

## 2. What \(\Phi\) means so far

The current simulations use a number \(\Phi\) between zero and one. They call
zero the vacuum end and one the high-overlap end. This is a convention in the
model, not yet a measured physical quantity. In particular, the phrase “no
shared state at \(\Phi=0\)” has not been established for a quantum vacuum.

Two reproducible benchmarks make particular choices explicit. The diagonal
Markov-chain calculation is a classical probability model encoded in qubits;
it is not entangled. The separate transverse-field Ising calculation uses
quantum ground states and verifies pairwise entanglement with a positive
negativity measure. It normalizes the nearest-neighbour mutual information by
\(2\ln2\), the general two-qubit upper bound. In that finite-chain sweep, the
fitted correlation-decay rate decreases as this \(\Phi\) rises, while the
Schwarzschild comparison requires the opposite trend. This is a negative
consistency result for two selected models, not a refutation of every possible
information-geometric model.

## 3. When a logarithmic distance follows

Suppose a positive, dimensionless correlation kernel \(K\) multiplies along a
specified chain, while lengths add. Under a regularity condition, the
mathematical map that turns multiplication into addition is a logarithm:

\[
d=-\ell_P\ln K.
\]

This is a conditional statement about a kernel and a chain. It does not say
that pairwise mutual information itself multiplies. In the explicit classical
Markov benchmark it does not: at every nonzero link strength in that model the
two-link mutual information differs from the square of the one-link value, and
the two agree only in the limit of zero correlation (machine-checked by
`research/verify_derivations.py`, which scales the difference by the fourth
power of the link strength so the comparison is stable as the link goes to
zero). The entangled Ising benchmark also
does not assume a multiplicative QMI kernel; it fits an effective decay rate
to a finite set of two-point correlations instead. And if the chosen kernel is
zero, its negative logarithm is infinite.

So the theory needs to specify which correlation quantity is the kernel, how
it is measured, and why its chain composition rule holds. That step is not
finished.

## 4. The changing length scale is a model choice

The current formal model uses

\[
\ell_P(\Phi)=\ell_{P0}\sqrt{1-\Phi^2}.
\]

Lean checks algebraic properties of this function, including its endpoint
values and monotonicity on its stated domain. Those checks establish properties
of the chosen formula; they do not derive the formula from mutual information
or demonstrate that it is a physical Planck length.

The radial notes show that this scale alone makes the spatial factor contract
rather than reproduce the Schwarzschild radial geometry. A further
correlation-decay metric is required, and it must be derived from a specified
Q-region model rather than selected to match the desired answer.

One clarification (added 2026-10-20): the formal corpus contains a *second*,
different shrinking-length model that older writing blurred into this one.
In the second model the length shrinks because information is denser, not
because \(\Phi\) is larger. Both models have been checked as mathematics;
they are different choices, results about one do not carry over to the
other, and neither has been shown to be the physical Planck length.

## 5. Gravity: distinct candidate roles for \(\Phi\)

One proposed route treats \(\Phi\) as an ordinary scalar field that carries
energy and sources an independently existing spacetime metric. For a chosen
canonical action, varying the action gives a stress-energy tensor and a scalar
field equation. On the constant \(\Phi=0\) branch, with the stated vacuum
conditions, the metric equation reduces to Einstein's equation. This is a
useful consistency check, but it does not derive the action from Q-regions.

A different route uses \(\Phi\) as a label for the Schwarzschild metric itself.
Those are not currently the same theory. Under standard regularity and
convex-potential conditions, a minimally coupled scalar does not provide the
nonconstant static black-hole profile used by that metric dictionary.

A third candidate makes \(\Phi\) a gravitational scalar coupled to spacetime
curvature through a scalar--Gauss--Bonnet term. The constant \(\Phi=0\) branch
reduces to GR under the same vacuum conditions, a hand derivation from that
action; a linear static calculation finds a possible zero-mode onset near
\(\eta/M^2=0.725613\pm10^{-6}\), the value two independent numerical methods
agree on (the \(0.725609\) of a truncated run is not it). This is not yet a nonlinear, stable black-hole
solution or a derivation from quantum information. It is the provisional
working branch for the next consistency pass, because it permits a curvature
source while retaining a GR branch; it does not settle the fundamental ontology.

The program must select one role or derive a consistent map between distinct
roles without double counting. The assumptions and branch comparison are in
the [covariant-completion note](docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md);
the numerical diagnostic is in the
[scalarization note](docs/research/GAUSS_BONNET_SCALARIZATION.md).

## 6. Black holes and cosmic expansion: an unverified proposal

The repository contains a phenomenological model in which an information-like
quantity decays at a rate depending on black-hole horizon area. A simulation
can show what follows after that decay law is assumed. The area theorem alone
does not establish that global mutual information decays that way, and the
decay equation does not by itself derive a covariant Friedmann equation.

The existing cosmology script calibrates a parameter to a chosen value of
\(H_0\). It therefore does not independently predict that calibration value.
There is no frozen, parameter-fixed Omega prediction that has been evaluated
against independent cosmological data. Such an analysis must wait until the
field equations and the map from the model to observables are fixed.

## 7. Biology and other extensions

The repository also contains biological and information-processing ideas.
They are speculative analogies, not established consequences of the gravity
model. A digital evolution simulation can test its own programmed rules; it
cannot by itself verify a physical theory of life or show that an overlap
measure causes adaptation.

*(Added 2026-10-20.)* The fuller version of these ideas, held outside this
repository, gives them names: living things are described as running an
"active shredding" process that keeps their internal information organised,
at a cost stated as a "Metabolic Inequality"; the long-running E. coli
evolution experiment (and its famous Cit+ moment, when the bacteria gained
a new food ability) is presented as an example of the pattern; and the mind
is described as a "Q-Systemic Self," with anxiety pictured as a state where
the parts of the self-model barely overlap and coherence as the opposite.
Please read all of that exactly as labelled here: a speculative story told
by analogy. A few nearby toy models have been formally checked — a simple
model of selection in evolution, a toy model of integrated information,
and a model of ecosystems — but they check their own small mathematical
worlds. They do not check the active-shredding story, the metabolic claim,
or the picture of the mind, and those remain unproven ideas, not results.

Agent software, token pilots, and economic proposals are separate engineering
projects. They do not validate the physics.

## 8. What has actually been checked

- Some mathematical statements in the Lean corpus are kernel-checked under
  their formal assumptions. The proof audit records important limitations,
  including model degeneracy and the gap between a theorem and its physical
  interpretation; the adversarial audit is the harsher of the two records, and
  its executive summary is the pre-remediation scoreboard, kept as history.
- (Added 2026-10-20.) There is now a plain map, `lean_proofs/PROOF_MAP.md`,
  saying which named Lean results stand behind which claims in the technical
  notes — and, just as important, which claims have *no* Lean result behind
  them at all. Newly pointed out there: the information-and-energy
  bookkeeping has been formally checked as a model; the logarithmic-distance
  idea has been checked in a conditional form, together with a checked
  *negative* result showing the stronger version of it fails; the radial
  geometry results in the companion notes are now named individually; and a
  set of "cross-volume" theorems checks that the formal restatements of
  different subject volumes agree with each other. That last kind of check
  is about the internal consistency of the mathematics, not about the
  universe: a consistent story is not automatically a true one. Some named
  results were also checked only inside a deliberately tiny, simplified
  model; the map flags those, and they should not be quoted as physics.
- The identities quoted in the core notes are re-derived by code, not only
  asserted: twelve checks in `research/verify_derivations.py` (the Schwarzschild
  dictionary, the required correlation-decay rate, the golden-ratio bound, the
  Markov and Gauss--Bonnet numbers used above), four in
  `research/verify_covariant.py`, two in `research/verify_tfim.py`, and a
  citation check that every LaTeX volume names declarations that exist
  (`research/check_latex_citations.py`). One command runs all of them:
  `python research/consistency_gate.py`. Not every number this guide quotes is
  inside that set -- the two-qubit bound \(2\ln2\) is standard, not repo
  content -- so "checked" here means the check named beside it, no more.
- The radial-metric code checks algebraic and numerical identities for
  specified candidate metrics. It identifies the correlation-decay rate that
  would reproduce Schwarzschild, but does not derive that rate from Q-regions.
- The classical Markov-chain code tests one explicit mutual-information map;
  the entangled Ising-chain code tests a second map and directly verifies
  positive two-spin negativity. The opposite decay-rate trend persists across
  \(N=8,10,12,14\) for a common fitting window; neither benchmark derives a
  continuum metric.
- The minimal covariant scalar-action note verifies an exact GR branch. A
  separate curvature-coupled gravitational-scalar model has a numerical
  linear zero-mode diagnostic, but no nonlinear solution or data-tested
  prediction.
- No physical result is confirmed by the fact that a simulation reproduces
  behavior built into its own equations.

For the detailed distinction between derived, identified, and open claims, see
the [technical note](Omega_Theory_v4.0_Technical.md) and the
[Lean proof audit](lean_proofs/PROOF_AUDIT.md).

## 9. The short version

- Q-region mutual information is well-defined for a specified state; the
  physical state family is still unspecified.
- A logarithmic distance follows only for a positive kernel that composes
  multiplicatively; mutual information generally is not that kernel.
- The \(\ell_P(\Phi)\) profile is chosen, not derived from the Q-region data.
- The minimal matter-scalar action and geometric metric label remain distinct.
  A curvature-coupled gravitational scalar is the provisional next-test branch,
  but is neither derived from Q-regions nor established as the fundamental role.
- The black-hole depletion and cosmic-expansion links remain hypotheses.
- The research program has not yet produced an independently tested,
  parameter-fixed prediction, much less a complete Theory of Everything.

```bash
python research/qregion_markov.py
python -m pytest tests/test_qregion_markov.py -q
python research/quantum_ising_chain.py
python -m pytest tests/test_quantum_ising_chain.py -q
python research/gauss_bonnet_scalarization.py
python -m pytest tests/test_gauss_bonnet_scalarization.py -q
python Sim7_Radial_Metric.py --no-plots
python research/consistency_gate.py   # every local check, one command
```
