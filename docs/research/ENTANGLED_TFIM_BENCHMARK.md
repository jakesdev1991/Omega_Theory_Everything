<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT -->

# Entangled Q-region benchmark: transverse-field Ising chain

**Status (2026-10-11):** finite-size quantum consistency check. Unlike the
diagonal Markov-chain benchmark, this model has pairwise entanglement. It still
does not produce a spatial metric or test gravity against nature.

## 1. State family and Q-regions

Use an open chain of \(N\) spin-\(1/2\) Q-regions with

\[
H/J=-\sum_{i=1}^{N-1}Z_iZ_{i+1}
-(h/J)\sum_{i=1}^{N}X_i,
\qquad h/J>1.
\]

The calculation obtains the lowest two eigenvalues and the ground state by
sparse exact diagonalization. In this finite chain the ground state is
non-degenerate (positive first excitation gap). Each Q-region is one spin. For
sites \(i,j\), calculate their reduced density matrices and quantum mutual
information

\[
I(i:j)=S(\rho_i)+S(\rho_j)-S(\rho_{ij}),
\qquad S(\rho)=-\operatorname{Tr}(\rho\ln\rho).
\]

Unlike the classical-diagonal family, a general two-qubit pair can have
\(I_{\max}=2\ln2\) (for example, a pure Bell pair). The dimensionless overlap
proxy in this benchmark is therefore

\[
\Phi_{\rm Ising}=\frac{I(i:i+1)}{2\ln2}.
\]

This is an explicit normalization, not a universal physical definition of
\(\Phi\). The nearest-neighbour negativity is also calculated. For two qubits,
the positive-partial-transpose (PPT) criterion is necessary and sufficient for
separability; thus a negative partial-transpose eigenvalue (positive
negativity) certifies that the two-spin reduced state is entangled.

## 2. Decay-rate estimator

For the selected central site (zero-based \(i=5\) when \(N=12\)), calculate
\(C_r=\langle Z_iZ_{i+r}\rangle\) and pair mutual information for separations
\(r=1,\ldots,5\). Define the finite-range estimator

\[
\kappa_{\rm eff}:=-\operatorname{slope}_{r=1,\ldots,5}
\left[\ln|C_r|\right].
\]

Its units are inverse lattice spacings. It is an empirical exponential-fit
summary for this finite numerical sample, not an exact continuum
\(\kappa(\Phi)\), not proof that \(C_{r+s}=C_rC_s\), and not a metric. The
comparison column below is the dimensionless Schwarzschild-required ratio from
the separate radial dictionary. It is used only to compare trends: the two
columns have different normalizations and must not be read as a fitted
quantitative match.

## 3. Fixed sweep, no gravity fit

Run the fixed field sweep with no fitted parameter:

```bash
python research/quantum_ising_chain.py
python -m pytest tests/test_quantum_ising_chain.py -q
```

Output for \(N=12\), central-site separations 1–5, and \(J=1\):

| \(h/J\) | \(\Phi_{\rm Ising}\) | \(I_1\) (nats) | nearest-pair negativity | \(\kappa_{\rm eff}\) (site\(^{-1}\)) | fit \(R^2\) | Schwarzschild-required \(\kappa/\kappa_0\) |
|---:|---:|---:|---:|---:|---:|---:|
| 4.00 | 0.029007 | 0.0402124 | 0.0580078 | 1.5584825 | 0.999 | 1.0593 |
| 2.50 | 0.060613 | 0.0840272 | 0.0873185 | 1.0869665 | 0.999 | 1.1270 |
| 1.80 | 0.098801 | 0.1369671 | 0.1114708 | 0.7570605 | 0.998 | 1.2133 |
| 1.40 | 0.141755 | 0.1965145 | 0.1267818 | 0.5061548 | 0.997 | 1.3169 |
| 1.20 | 0.177420 | 0.2459564 | 0.1293611 | 0.3558353 | 0.997 | 1.4087 |
| 1.10 | 0.202782 | 0.2811157 | 0.1257215 | 0.2746502 | 0.997 | 1.4774 |
| 1.05 | 0.218571 | 0.3030033 | 0.1215383 | 0.2333800 | 0.997 | 1.5217 |

The positive nearest-pair negativities show this is not merely a classical
diagonal state. Across this fixed sweep, larger nearest-neighbour \(\Phi\)
comes with a **smaller** fitted correlator-decay rate. The Schwarzschild-required
ratio in the radial notes instead increases with \(\Phi\). The two trends
therefore disagree in this selected entangled chain as well as in the simpler
Markov-chain benchmark.

### Finite-size and fitting-window check

The same comparison with one fixed fitting window, \(r=1,2,3\), gives:

| \(N\) | \(\Phi\) at \(h/J=4\) | \(\kappa_{\rm eff}\) at \(h/J=4\) | \(\Phi\) at \(h/J=1.05\) | \(\kappa_{\rm eff}\) at \(h/J=1.05\) |
|---:|---:|---:|---:|---:|
| 8 | 0.029007 | 1.619504 | 0.214500 | 0.300279 |
| 10 | 0.029007 | 1.619280 | 0.216883 | 0.269199 |
| 12 | 0.029007 | 1.619271 | 0.218571 | 0.252830 |
| 14 | 0.029007 | 1.619271 | 0.219785 | 0.242889 |

The opposite trend persists across these sizes using the identical short-range
window, and the test suite checks it for all four sizes. This reduces (but does
not eliminate) concern that the sign is a particular \(N=12\), \(r=1\ldots5\)
artifact. These are still small finite chains, not a thermodynamic-limit
calculation.

One possible missing ingredient is a physical site spacing \(a(\Phi)\), since
the reported \(\kappa_{\rm eff}\) is in inverse sites. If
\(\kappa_{\rm phys}=\kappa_{\rm eff}/a(\Phi)\) is to increase with \(\Phi\)
while \(\kappa_{\rm eff}\) decreases, then necessarily
\(d\ln a/d\Phi<d\ln\kappa_{\rm eff}/d\Phi<0\) over the tested range. A spacing
law chosen by solving for the Schwarzschild target would be circular; this
benchmark supplies no independent derivation of \(a(\Phi)\). Thus variable
spacing is a concrete next hypothesis, not a repair already demonstrated.

As \(h/J\to\infty\), the ground state approaches a product state,
\(\Phi_{\rm Ising}\to0\), and the correlator decay rate grows without bound.
Thus the literal product-state end again does not supply a finite vacuum
\(\kappa_0\) for the logarithmic-distance construction. Choosing a nonzero
reference state could make a finite normalization, but would change the
meaning of \(\Phi=0\) and would require independent justification.

## 4. Reproducibility and limits

Implementation: [`research/quantum_ising_chain.py`](../../research/quantum_ising_chain.py).
The tests check the finite Hamiltonian, a positive ground-state gap, Hermitian,
normalized positive reduced density matrices, a negative partial-transpose
eigenvalue, and the opposite decay-rate trend on \(N=8,10,12,14\) chains with
one common fitting window.

This is a finite, one-dimensional, gapped spin model. Its field ratio is a
global model parameter, not a spatially varying gravitational field. The
correlation-rate fit depends on system size, site choice, and the finite fitting
window. It provides no transverse/angular metric, no continuum limit, no
covariant field equation, and no independent observational prediction. It
rejects neither all Q-region constructions nor general relativity; it shows
that adding pairwise entanglement alone does not repair the current
fixed-spacing identification.

The next geometric benchmark must specify how a two- or three-dimensional
Q-region arrangement, its physical spacing, and its full correlation tensor
are obtained without fitting them to Schwarzschild. The separate ontology
choice—whether \(\Phi\) is a geometric order parameter or a matter scalar—is
still required before selecting a canonical covariant action.
