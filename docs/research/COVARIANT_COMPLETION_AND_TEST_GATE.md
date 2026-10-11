<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT -->

# Covariant completion, GR limit, and empirical-test gate

**Status (2026-10-11):** one minimally coupled scalar-matter completion has
been varied explicitly and its general-relativistic limit identified. For the
next consistency pass, the curvature-coupled gravitational-scalar branch below
is the **provisional working model** because it can couple to vacuum curvature
while retaining a GR branch. This is a modeling choice, not a resolution of
the fundamental ontology and not a derivation from Q-region information. The
nonlinear branch and independent observational prediction remain unestablished.

## 1. Conventions and candidate action

Use signature \((-+++ )\), set \(c=\hbar=1\), and take \(\Phi\) dimensionless.
Let \(Z(\Phi)>0\) and \(V(\Phi)\) be differentiable functions. Matter fields
\(\psi\) couple minimally to the metric and do not couple directly to \(\Phi\):

\[
S[g,\Phi,\psi]=\int d^4x\sqrt{-g}\left[
\frac{R}{16\pi G}
-\frac12 Z(\Phi)g^{\mu\nu}\nabla_\mu\Phi\nabla_\nu\Phi
-V(\Phi)\right]+S_m[g,\psi].
\]

This sign convention gives positive kinetic energy when \(Z>0\). It makes
explicit the interpretation implicit in the earlier scalar action: \(\Phi\) is
an ordinary matter field on an independently varied spacetime metric. It is
**not** the emergent-metric interpretation of the Q-region proposal.

Define
\(T^{(m)}_{\mu\nu}:=-2(-g)^{-1/2}\delta S_m/\delta g^{\mu\nu}\). Varying with
respect to \(g^{\mu\nu}\) gives

\[
G_{\mu\nu}=8\pi G\left(T^{(m)}_{\mu\nu}+T^{(\Phi)}_{\mu\nu}\right),
\]

\[
T^{(\Phi)}_{\mu\nu}
=Z(\Phi)\nabla_\mu\Phi\nabla_\nu\Phi
-g_{\mu\nu}\left[\frac12Z(\Phi)(\nabla\Phi)^2+V(\Phi)\right].
\]

Varying with respect to \(\Phi\), integrating the kinetic variation by parts,
and assuming fixed boundary data gives

\[
\nabla_\mu\!\left(Z\nabla^\mu\Phi\right)
-\frac12 Z'(\nabla\Phi)^2-V'=0,
\]

or equivalently

\[
\boxed{\;Z\Box\Phi+\frac12Z'(\nabla\Phi)^2-V'=0.\;}
\]

Minimal matter coupling implies \(\nabla^\mu T^{(m)}_{\mu\nu}=0\) on the
matter equations of motion. The scalar equation similarly makes
\(\nabla^\mu T^{(\Phi)}_{\mu\nu}=0\). These are consequences of this action;
they do not identify \(Z\) or \(V\) from information theory.

## 2. Exact GR limit and linear stability conditions

A flat-space vacuum at \(\Phi=0\) requires at least

\[
V(0)=0,\qquad V'(0)=0,\qquad Z(0)>0.
\]

For \(\Phi\equiv0\), \(T^{(\Phi)}_{\mu\nu}=0\), so the metric equation is
exactly Einstein's equation with the same matter source. In particular, in
vacuum the Schwarzschild metric is an exact solution, and asymptotically flat
weak-field observables reduce to their GR values. With boundary condition
\(\Phi\to0\) and no direct matter coupling, the scalar is not sourced by
ordinary matter at linear order; the metric's standard PPN values are
\(\gamma=\beta=1\) on this branch.

Linearizing the scalar equation about the vacuum gives

\[
\left(\Box-m_{\rm eff}^2\right)\delta\Phi=0,
\qquad
m_{\rm eff}^2=\frac{V''(0)}{Z(0)}.
\]

Thus \(Z(0)>0\) avoids a wrong-sign kinetic term and
\(V''(0)\geq0\) avoids a tachyonic linear instability. These checks are
necessary conditions for this candidate; they are not a proof of full
nonlinear stability.

### The geometry/matter fork remains real

In the radial-metric notes, \(\Phi=M/(2\rho)\) is a coordinate-dependent
label used to write the isotropic Schwarzschild metric. It is not thereby a
solution for a scalar field that sources that metric. For the minimally
coupled scalar above, assume a regular static black-hole exterior,
\(\Phi\to0\) at infinity, and a potential convex toward the vacuum. With the
canonical field \(\varphi=\int_0^\Phi\sqrt{Z(u)}\,du\) and
\(U(\varphi)=V(\Phi(\varphi))\), the static scalar equation implies, after
multiplication by \(\varphi\) and integration over a static slice,

\[
\int_\Sigma N\left[D_i\varphi D^i\varphi
+\varphi\,U_{,\varphi}\right]d\mathrm{vol}_h=0,
\]

provided the regular-horizon and asymptotic boundary terms vanish. If
\(\varphi U_{,\varphi}\geq0\), both terms are nonnegative and the only such
solution is \(\varphi=0\). In particular, the quadratic potential used in the
current model does not supply the nonconstant \(M/(2\rho)\) profile as
minimally coupled scalar hair. The Schwarzschild dictionary treats \(\Phi\) as
a label of geometry; the scalar action treats it as matter. They are not yet
one theory.

### Provisional next-test branch: a curvature-coupled gravitational scalar

If the goal is for \(\Phi\) to participate in vacuum geometry without calling
it an ordinary minimally coupled matter field or merely a coordinate label, a
covariant candidate is a scalar coupled to the four-dimensional
Gauss--Bonnet invariant:

\[
S_{\rm GB}=\int d^4x\sqrt{-g}\left[
\frac{R}{16\pi G}-\frac12Z(\Phi)(\nabla\Phi)^2-V(\Phi)
+f(\Phi)\,\mathcal G\right]+S_m[g,\psi],
\qquad
\mathcal G=R_{\mu\nu\rho\sigma}R^{\mu\nu\rho\sigma}
-4R_{\mu\nu}R^{\mu\nu}+R^2.
\]

Here \(\Phi\) is a **gravitational scalar degree of freedom**, distinct from
both the minimally coupled matter branch above and the radial coordinate label.
For differentiable functions with
\(Z(0)>0\), \(V(0)=V'(0)=0\), and \(f'(0)=0\), the constant branch
\(\Phi=0\) solves the scalar equation and leaves the exact Einstein equation.
The additive constant \(f(0)\) multiplies a topological term in four
spacetime dimensions and can be set to zero.

The scalar equation is

\[
Z\Box\Phi+\frac12 Z'(\nabla\Phi)^2-V'+f'(\Phi)\mathcal G=0.
\]

On a Schwarzschild background, \(R=R_{\mu\nu}=0\) but
\(\mathcal G=48M^2/R_{\rm areal}^6\neq0\). If also \(f'(0)=0\), the linearized
scalar equation contains the curvature-dependent effective mass

\[
m_{\rm eff}^2(R_{\rm areal})=
\frac{V''(0)-f''(0)\mathcal G}{Z(0)}.
\]

This supplies a possible curvature source/instability channel that the
minimal matter action lacks. For the massless quadratic-coupling case, a
reproducible static s-wave zero-mode calculation finds a threshold near
\(\eta/M^2=0.72561\); see [`GAUSS_BONNET_SCALARIZATION.md`](GAUSS_BONNET_SCALARIZATION.md).
That is only a linear onset diagnostic. It does **not** establish that a
regular, stable nonlinear hairy solution exists, that such a solution
reproduces the radial dictionary, or that \(f\) follows from Q-region data.
The coupling and potential require independent motivation; solving them to
match Schwarzschild would be circular. This is the provisional branch used for
the next consistency tests, not an established ontology or completed theory.

The three constructions must remain distinct:

| Construction | Role of \(\Phi\) | GR branch | What remains open |
|---|---|---|---|
| Minimal action above | Ordinary matter scalar on an independent metric | Exact at \(\Phi=0\) under the stated vacuum conditions | No nonconstant static hair for the stated convex-potential case; no Q-region derivation |
| Curvature-coupled action here | Additional gravitational scalar degree of freedom | Exact at \(\Phi=0\) when \(f'(0)=0\) | Hair existence, nonlinear stability, parameter constraints, Q-region origin |
| Radial Schwarzschild dictionary | Coordinate-dependent scalar label assigned after choosing a metric | Reproduces the selected GR solution by construction | Not a covariant dynamics or a derivation from Q-regions |

For the next modeling pass, the curvature-coupled gravitational-scalar row is
selected provisionally because it keeps an exact GR branch and permits a
curvature source. The fundamental ontology remains unresolved: this workflow
choice is not evidence that \(\Phi\) is a physical gravitational scalar. The
matter-scalar action and post-hoc metric label remain separate comparators; a
unified theory must derive any map between them without double counting.

## 3. Homogeneous cosmology under this action

For a spatially flat FLRW metric and minimally coupled perfect-fluid matter,

\[
3H^2=8\pi G\left(\rho_m+\frac12Z\dot\Phi^2+V\right),
\]

\[
-2\dot H=8\pi G\left(\rho_m+p_m+Z\dot\Phi^2\right),
\]

\[
Z\left(\ddot\Phi+3H\dot\Phi\right)
+\frac12Z'\dot\Phi^2+V'=0,
\qquad
\dot\rho_m+3H(\rho_m+p_m)=0.
\]

These equations follow from the chosen covariant action. They do **not** imply
\(H(t)=\alpha\gamma A_{BH}^{\kappa}(t)\). The black-hole-area depletion law
in the older technical note is a phenomenological ansatz: no covariant source,
exchange term, or stress-energy conservation argument connecting that law to
the equations above has been supplied. A simulation that imposes the
phenomenological law and calibrates its parameter to a chosen \(H_0\) does not
independently predict that \(H_0\), nor does it test the law against withheld
data.

## 4. What is and is not recovered

| Check | Result for the stated minimal action | Status |
|---|---|---|
| Covariant metric and scalar equations | Explicit equations above | Derived from the chosen action |
| Minkowski/GR branch at \(\Phi=0\) | Einstein equation with ordinary matter | Recovered conditionally on \(V(0)=V'(0)=0\) |
| Linear scalar stability | \(Z(0)>0\), \(V''(0)\geq0\) | Necessary local conditions |
| Standard weak-field GR on the zero-scalar branch | \(\gamma=\beta=1\) | Exact on that branch; no modified prediction |
| Nonconstant scalar profile around a regular static hole | Excluded for the minimally coupled convex-potential case above | Does not derive the radial-metric dictionary |
| Black-hole-area-driven expansion | Not obtained from this action | Open; current depletion law is an ansatz |
| A distinct, parameter-fixed observable prediction | None yet | Cannot honestly preregister or evaluate yet |

## 5. Empirical preregistration gate (not yet a registration)

**No independent-data analysis is claimed here.** A proper empirical test is
blocked until the theory has a unique, covariant model and a unique map from
Q-region observables to measured quantities. In particular, do not count a
fit to the same \(H_0\) used to calibrate \(\gamma\) as confirmation. Before
registering a test, freeze all of the following:

1. **Theory choice:** whether \(\Phi\) is a matter scalar, a geometric variable,
or part of a scalar-tensor theory; specify the metric frame and matter coupling.
2. **Equations and parameters:** full action/source terms, units, boundary
   conditions, parameter count, allowed priors/ranges, and stability domain.
3. **Observable map:** derive how the frozen model predicts a specified
   observable without importing the target curve into the definition of
   \(\Phi\), \(\kappa\), or a fit function.
4. **Data and comparator:** name immutable dataset release(s), record file
   hashes and selection cuts, preselect the independent holdout, and specify a
   GR/\(\Lambda\)CDM baseline with the same nuisance treatment.
5. **Decision rule:** publish the likelihood/statistic, systematic covariance,
   exclusion or support threshold, missing-data policy, and all code hashes
   before examining the holdout.
6. **Reproduction:** require a second clean run from the frozen inputs and
   report negative, null, and inconclusive outcomes as well as positive ones.

The existing \(H_0\)-calibrated depletion simulation is not suitable as the
primary registered prediction in its current form: its field equations are
not derived from the covariant action above, and the calibration consumes the
headline datum. A distinct prediction and an independent empirical evaluation
are deferred rather than simulated or described as complete. The current classical and entangled finite-chain diagnostics and their
negative Schwarzschild-trend comparisons are in
[`QREGION_MARKOV_MODEL.md`](QREGION_MARKOV_MODEL.md) and
[`ENTANGLED_TFIM_BENCHMARK.md`](ENTANGLED_TFIM_BENCHMARK.md).
