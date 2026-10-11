<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT -->

# Curvature-coupled gravitational-scalar zero mode

**Status (2026-10-11):** candidate-model calculation only. This tests whether a
scalar in the gravitational sector can have a curvature-sourced static mode on
a Schwarzschild background. It is not a Q-region derivation, a nonlinear
black-hole solution, or evidence that Omega describes nature.

## 1. Candidate and exact GR branch

Take the scalar-Gauss--Bonnet coupling

\[
S=\int d^4x\sqrt{-g}\left[
\frac{R}{16\pi G}-\frac12Z_0(\nabla\Phi)^2
+\frac12Z_0\eta\Phi^2\mathcal G
\right]+S_m[g,\psi],
\qquad
\mathcal G=R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}
-4R_{\mu\nu}R^{\mu\nu}+R^2.
\]

Take \(\Phi\) dimensionless and \(Z_0>0\) as its kinetic normalization.
Then \(\eta=f''(0)/Z_0\) has dimensions of length squared; the numerical
coupling reported below is the dimensionless ratio \(\eta/M^2\).

Here \(\Phi\) is an additional **gravitational scalar degree of freedom**,
not the minimally coupled matter scalar of the earlier candidate and not the
coordinate-dependent label in the radial Schwarzschild dictionary. At
\(\Phi=0\), the scalar and its first derivative of the coupling vanish, so
Schwarzschild is an exact GR-branch solution. The action is a new model choice;
\(\eta\) is not derived from the Q-region benchmarks.

The linearized scalar equation on that branch is

\[
\Box\delta\Phi+\eta\mathcal G_{\rm Schw}\delta\Phi=0,
\qquad
\mathcal G_{\rm Schw}=\frac{48M^2}{r^6},
\]

where \(r\) is areal radius. Unlike the minimally coupled convex-potential
model, the curvature invariant is nonzero in the Ricci-flat exterior and can
change the scalar's effective mass. This supplies a *possible* vacuum-curvature
source; it does not by itself prove that a regular, stable nonlinear hairy
black hole exists.

## 2. Static s-wave boundary-value problem

Set \(x=r/(2M)\) and \(\lambda=\eta/M^2\). A static spherical mode satisfies

\[
\frac{d}{dx}\left[x(x-1)\frac{d\phi}{dx}\right]
+\frac{3\lambda}{x^4}\phi=0,\qquad x\geq1.
\]

Normalize the regular horizon value to \(\phi(1)=1\). Regularity fixes
\(\phi'(1)=-3\lambda\). At large \(x\), a massless static solution has
\(\phi(x)=A+B/x+o(x^{-1})\); a mode that decays at infinity requires \(A=0\).

`research/gauss_bonnet_scalarization.py` integrates outward from a short
Frobenius expansion at the horizon. At a finite outer radius \(x_{\max}\), it
estimates \(A\) by \(\phi+x\phi'\), scans a fixed interval for the first
sign-changing bracket, and refines that zero with Brent's method. No
Schwarzschild metric coefficient is fitted.

## 3. Numerical result

With the default \(x_{\max}=250\), the first bracketed s-wave zero is

\[
\boxed{\eta/M^2\simeq0.725609167},
\qquad A_{\rm estimated}\simeq-1.2\times10^{-11}.
\]

The inferred threshold varies from about \(0.72558965\) at \(x_{\max}=100\)
to \(0.72561262\) at \(x_{\max}=1000\), consistent with convergence of this
shooting estimate. Reproduce with:

```bash
python research/gauss_bonnet_scalarization.py
python -m pytest tests/test_gauss_bonnet_scalarization.py -q
```

## 4. Interpretation and remaining gates

This is a linear static zero mode, a candidate onset diagnostic for curvature-
induced scalarization. It is **not** a computed nonlinear branch, a proof of
its stability, an observationally allowed coupling, or a prediction of
\(\Phi\) from mutual information. In particular, matching the radial
\(\Phi=M/(2\rho)\) dictionary would be a separate boundary-value problem and
must not be imposed as the target of this calculation.

The candidate advances the geometric-versus-matter discussion by adding a
third, explicitly named possibility: \(\Phi\) can be a degree of freedom in
the gravitational action, distinct from ordinary matter and from a
post-hoc metric label. This is the **provisional working branch for the next
consistency pass**, not a settled fundamental ontology. A complete next stage
would solve the nonlinear metric-scalar equations, check horizon regularity
and stability along the branch, constrain \(\eta\) against independent gravity
tests, and derive (rather than stipulate) the coupling and its connection to
Q-region observables.
