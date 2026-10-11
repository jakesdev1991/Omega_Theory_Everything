<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT -->

# Q-region Markov-chain benchmark: mutual information and correlation decay

**Status (2026-10-11):** an explicit, reproducible toy-model result. It is not a
model of quantum gravity, an empirical result, or a derivation of general
relativity. The purpose is to test whether the proposed local mutual-information
variable determines a correlation-decay rate in one simple, non-degenerate
Q-region family.

## 1. Model and state

Take a one-dimensional chain of qubit subsystems, one Q-region per site. In the
computational (`Z`) basis, define the density operator

\[
\rho_N(\lambda)=\frac12\sum_{s_1=\pm1}\sum_{s_2,\ldots,s_N=\pm1}
\left[\prod_{k=1}^{N-1}\frac{1+\lambda s_k s_{k+1}}2\right]
|s_1\cdots s_N\rangle\langle s_1\cdots s_N|,
\qquad 0\leq\lambda<1.
\]

This is a normalized, full-rank density operator for \(\lambda<1\). It is
diagonal, so it is a **classical** stationary binary Markov chain encoded in
qubits; it contains no bipartite entanglement. Calling the sites Q-regions does
not make this family a quantum-entanglement model. The endpoint \(\lambda=1\)
is a singular, perfectly correlated limit.

For sites separated by \(n\) links,

\[
C_n:=\langle Z_i Z_{i+n}\rangle=\lambda^n,
\qquad
\rho_{i,i+n}=\frac14\bigl(\mathbb{1}+\lambda^n Z\otimes Z\bigr).
\]

The pair eigenvalues are \((1+\lambda^n)/4\) and \((1-\lambda^n)/4\), each with
multiplicity two. Each one-site marginal is maximally mixed. In nats, the pair
mutual information is therefore

\[
I_n(\lambda)=\ln 2-h_2\!\left(\frac{1+\lambda^n}{2}\right),\qquad
h_2(p)=-p\ln p-(1-p)\ln(1-p).
\]

In particular, for a nearest-neighbour pair, \(I_1(0)=0\),
\(I_1(1)=\ln2\), and

\[
\frac{d I_1}{d\lambda}=\operatorname{atanh}(\lambda)>0
\quad (0<\lambda<1).
\]

Thus \(I_1\) determines \(\lambda\) uniquely within this specified family.
For a pair at separation \(n\), the inferred one-link value is
\(\lambda=[I_1^{-1}(I_n)]^{1/n}\); this inference requires knowing the chain
family and the separation \(n\).

## 2. What composes along the chain

The Markov-composing object is the two-point correlator \(C_n=\lambda^n\):

\[
C_{n+m}=C_n C_m.
\]

It is **not** the mutual information. For example, at \(\lambda=0.5\),

\[
I_1/\ln2=0.1887218755,\qquad
I_2/\ln2=0.0455659971,\qquad
(I_1/\ln2)^2=0.0356159463.
\]

So \(I_2/\ln2\ne(I_1/\ln2)^2\). The expression
\(d=-\ell_P\ln(I/I_{\max})\) cannot use these pairwise mutual informations as
the multiplicative kernel in the logarithmic-additivity argument. Also,
\(I=0\) gives \(-\ln(I/I_{\max})=+\infty\), not a finite distance to an
asymptotically flat, uncorrelated vacuum.

A logarithmic chain coordinate can instead be defined from the correlator:

\[
\sigma_n=-\ln C_n=-n\ln\lambda,
\qquad
\kappa=-\frac{d\ln C}{dr}= -\frac{\ln\lambda}{a},
\]

where \(a>0\) is the fixed coordinate spacing per link. This is a conditional
kinematic definition. It does not establish that \(\sigma\) is physical proper
distance.

## 3. Two explicit choices of overlap variable

### 3.1 Literal normalization used in the earlier technical note

For this **classical-diagonal family**, the pairwise mutual-information ceiling is \(I_{\max}=\ln2\) because each single-spin marginal has entropy \(\ln2\). This is not the generic two-qubit quantum ceiling, which is \(2\ln2\) (see the entangled benchmark in [`ENTANGLED_TFIM_BENCHMARK.md`](ENTANGLED_TFIM_BENCHMARK.md)). The literal choice in this family is

\[
\Phi_{\rm raw}=I_1/\ln2.
\]

It maps the product-state limit to \(\Phi_{\rm raw}=0\), but then
\(\lambda=0\) and \(\kappa=+\infty\). The finite-vacuum metric required by the
radial-metric notes is not obtained. At the other endpoint,
\(\Phi_{\rm raw}\to1\) implies \(\lambda\to1\) and \(\kappa\to0\). The raw
mapping therefore fails both endpoint requirements if the fixed-spacing
correlator is used in the logarithmic metric.

### 3.2 A finite-baseline benchmark

To test the *most favorable finite-baseline version* of this chain ansatz, set
an explicit, arbitrary background correlation \(\lambda_v=1/2\) and define
\(I_v=I_1(\lambda_v)\). Normalize only the excess mutual information:

\[
\Phi=\frac{I_1-I_v}{\ln2-I_v},\qquad
I_1=I_v+\Phi(\ln2-I_v),\qquad 0\leq\Phi\leq1.
\]

This reinterprets \(\Phi=0\) as a **correlated background state**, not as zero
mutual information. The baseline is a declared toy parameter, not a derived
property of the physical vacuum. The implied one-link correlation and decay
rate are

\[
\lambda(\Phi)=I_1^{-1}\!\left[I_v+\Phi(\ln2-I_v)\right],
\qquad
\boxed{\;\kappa(\Phi)=-\frac{1}{a}\ln\lambda(\Phi)\;}.
\]

Let \(D=\ln2-I_v>0\). Differentiating the defining equation gives

\[
\frac{d\kappa}{d\Phi}
=-\frac{D}{a\,\lambda(\Phi)\operatorname{atanh}(\lambda(\Phi))}<0.
\]

Hence \(\kappa\) is finite at the chosen background and **strictly decreases**
as shared mutual information increases; it tends to zero as \(\Phi\to1\).
This sign is independent of the particular nonzero baseline \(0<\lambda_v<1\).

The executable implementation uses bisection for \(I_1^{-1}\), so no fitted
interpolation is hidden in the table. For the comparison below, assume the
fixed-spacing chain coordinate is the Schwarzschild isotropic radial coordinate;
that embedding/gauge identification is itself an extra assumption, not derived
by the chain model. With \(a=1\), \(\lambda_v=1/2\), and the already-proposed
\(\ell_P/\ell_{P0}=\sqrt{1-\Phi^2}\), the two-factor radial ansatz gives

\[
\frac{h_{rr}(\Phi)}{h_{rr}(0)}
=(1-\Phi^2)\left[\frac{\kappa(\Phi)}{\kappa(0)}\right]^2.
\]

| \(\Phi\) | \(\lambda(\Phi)\) | \(\kappa/\kappa_0\) | toy \(h_{rr}/h_{rr}(0)\) | isotropic Schwarzschild \(h_{rr}\) |
|---:|---:|---:|---:|---:|
| 0 | 0.5000000 | 1.0000000 | 1.0000000 | 1.0000 |
| 0.1 | 0.5917005 | 0.7570609 | 0.5674098 | 1.4641 |
| 0.3 | 0.7323062 | 0.4494810 | 0.1838502 | 2.8561 |
| 0.5 | 0.8380308 | 0.2549249 | 0.0487400 | 5.0625 |
| 0.618034 (= Φ\*) | 0.8887238 | 0.1701929 | 0.0179017 | 6.8541 |
| 0.9 | 0.9798990 | 0.0292951 | 0.0001631 | 13.0321 |
| 0.99 | 0.9986444 | 0.0019571 | 0.0000001 | 15.6824 |

The row marked \(\Phi^*=(\sqrt5-1)/2\approx0.618034\) is evaluated at that exact
golden value. Evaluating at the rounded label \(0.618\) instead gives
\(\lambda=0.8887103\), \(\kappa/\kappa_0=0.1702148\); the difference is in the
fifth significant figure and does not change any conclusion. The table values
are checked by `research/verify_derivations.py` (`check_markov_kappa_table`).

The comparison column is the radial spatial coefficient
\(h_{rr}=(1+\Phi)^4\) in the isotropic Schwarzschild dictionary
\(\Phi=M/(2r)\) used in the radial-metric notes. Reproducing it with the chosen
\(\ell_P\) requires

\[
\frac{\kappa_{\rm GR}}{\kappa_0}
=\frac{(1+\Phi)^2}{\sqrt{1-\Phi^2}},
\]

which increases above one and diverges at the horizon. The independently
specified Markov chain instead drives \(\kappa/\kappa_0\) toward zero and its
radial coefficient toward zero. It does **not** recover the Schwarzschild
radial coefficient. This is a model-level inconsistency, not an observational
falsification of every possible Q-region model.

## 4. Reproduction and limits

Run:

```bash
python research/qregion_markov.py
python -m pytest tests/test_qregion_markov.py -q
```

The tests cover the mutual-information formula and inverse, Markov composition
of \(C_n\), failure of mutual-information composition, monotonicity of
\(\kappa(\Phi)\), and the endpoint/Schwarzschild comparison.

**What this establishes:** one explicit finite-dimensional Q-region family
makes the mapping from adjacent-pair mutual information to \(\Phi\) and
\(\kappa\) computable and testable. It exposes a conflict between this simple
correlation family and the required Schwarzschild behavior.

**What it does not establish:** a universal \(\kappa(\Phi)\), a continuum or
transverse metric, a quantum-entangled vacuum, the physical identity of
\(\Phi\), or a no-go theorem for richer state families. A next model would need
to state independently how Q-region spacing, state family, and transverse
correlations change; it must not choose them by fitting the Schwarzschild
answer and then call the result a derivation.
