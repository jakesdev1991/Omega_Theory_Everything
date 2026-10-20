<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT -->

# Toward the Macroscopic Radial Metric $g_{rr}(\Phi)$

*Working notes, companion to §2.2 of [`Omega_Theory_v4.0_Technical.md`](Omega_Theory_v4.0_Technical.md).*

**Status: exploratory.** Nothing here changes a canonical claim. Every result is tagged:

- **[Derived]**: follows from the stated axioms and assumptions by the argument given;
- **[Identified]**: a dictionary or postulate chosen to match known physics, *not* derived from the Omega axioms;
- **[Open]**: not settled.

Every formula is checked numerically, and where noted symbolically (SymPy), by [`Sim7_Radial_Metric.py`](Sim7_Radial_Metric.py). Its `self_check()` runs on every invocation: `python Sim7_Radial_Metric.py --no-plots`. Units are $G=c=1$ and $\ell_{P0}=1$ unless stated.

## 0. Summary

1. **Notation: fixed; physical meaning: open.** The repository now uses $\Phi=0$ for the vacuum end and $\Phi\to1$ for the high-overlap end. This resolves an internal sign convention, not the physical definition of $\Phi$. The profile in `DynamicCODScale.lean` and the choice $V=\tfrac12m^2\Phi^2$ are model choices. A macroscopic metric must be asymptotically flat at the vacuum; a horizon condition is coordinate-dependent and should be stated for a specified metric component and chart. The opposite convention in earlier drafts is not physically excluded by the Lean check.
2. **$g_{\mu\nu}=\delta S_{\rm ent}/\delta\rho$ is not a well-formed tensor equation** (§2). This shorthand has been removed from the README. The logarithmic chain-length result below is conditional on a positive multiplicative correlation kernel; the classical Markov benchmark shows why that kernel cannot simply be identified with pairwise mutual information, while the entangled-chain benchmark tests a separate finite-correlation-rate proxy.
3. **[Conditional] Two-factor chain law** (§3). Markov factorization, additive length, locality, and regularity give $ds=\ell_P(\Phi)\,d\sigma$ with $d\sigma=-d\ln K$ along a chain. If a smooth positive-definite continuum metric for $d\sigma$ exists, then $h_{ij}=\ell_P(\Phi)^2\gamma_{ij}$ and along a radial chain $h_{rr}=\ell_P(\Phi)^2\kappa^2$. The continuum metric and its derivation from Q-region data are additional open assumptions; the two-factor identity alone does not establish a spatial geometry.
4. **[Derived] Obstruction** (§4). Suppose $\ell_P(\Phi)$ were the only $\Phi$-dependence of the spatial metric, with a harmonic profile $\Phi=A/\rho$. Then $g_{RR}=(1-\Phi^2)^2\le1$ (contraction), the ADM mass is zero, the Misner–Sharp mass is negative, and the "horizon" has zero area. The canonical profile cannot by itself produce gravitational attraction.
5. **[Identified] Schwarzschild written in $\Phi$** (§5). With $\Phi:=\psi-1$ (the isotropic conformal factor minus one), the exact vacuum solution is
   $$ds^2=-\Big(\frac{1-\Phi}{1+\Phi}\Big)^2dt^2+(1+\Phi)^4\big(d\rho^2+\rho^2d\Omega^2\big),\qquad \Phi=\frac{M}{2\rho}.$$
   In areal gauge, $g_{RR}(\Phi)=\big(\frac{1+\Phi}{1-\Phi}\big)^2$. $\Phi$ is harmonic, $\Phi\in[0,1]$ is exactly one exterior, and $\Phi=1$ is the horizon. By the two-factor law this requires $\kappa/\kappa_0=(1+\Phi)^2/\sqrt{1-\Phi^2}$.
6. **$g_{rr}\propto1/\Phi$** (§6) is exactly Schwarzschild in the *other* convention ($\Phi=-g_{tt}$, vacuum at 1). In the Lean convention it diverges at the vacuum. It cannot coexist with $\ell_P=\ell_{P0}\sqrt{1-\Phi^2}$ for the same $\Phi$.
7. **[Open]** Two explicit Q-region diagnostics are now available: the diagonal-qubit Markov-chain benchmark derives one $\Phi(I_{ij})$ and $\kappa(\Phi)$ in [`docs/research/QREGION_MARKOV_MODEL.md`](docs/research/QREGION_MARKOV_MODEL.md), while the genuinely entangled finite Ising-chain sweep fits a finite-range correlation-decay rate in [`docs/research/ENTANGLED_TFIM_BENCHMARK.md`](docs/research/ENTANGLED_TFIM_BENCHMARK.md). Both selected models show a decay-rate trend opposite to the Schwarzschild-required trend; the entangled-chain sign persists across $N=8,10,12,14$ at a common fit window. These are model-level consistency results, not a universal no-go. A covariant metric derivation and the ontology choice remain open; a separate curvature-coupled gravitational-scalar candidate is tested only at linear zero-mode level in [`docs/research/GAUSS_BONNET_SCALARIZATION.md`](docs/research/GAUSS_BONNET_SCALARIZATION.md).

## 1. Working convention and its limits

| Location | Working vacuum convention | High-overlap end |
|---|---|---|
| This note and current technical note | $\Phi=0$; its operational physical meaning is open | $\Phi\to1$ by convention |
| `DynamicCODScale.lean` and `Sim3_Dynamic_Scale.py` | $\Phi=0$ for the chosen profile/potential | $\Phi\to1$ in the model domain |
| `Sim5_Emergent_Gravity.py` | `PHI_VACUUM = 0.0` | selected simulation values 0.4 and 0.9 |

The repository uses one convention to avoid internal sign flips. A Lean-checked formula or a stable simulation does not establish that this orientation describes physical vacuum or a horizon. The profile $\ell_P=\ell_{P0}\sqrt{1-\Phi^2}$ and the potential $V=\tfrac12m^2\Phi^2$ remain choices. The opposite parametrization is not ruled out by formal verification; the two conventions would need to be translated consistently if a model motivated one.

The calculations below use the current notation only. They do not treat $\Phi$ as an observed scalar or identify the mathematical endpoint $\Phi=1$ with a physical event horizon unless the stated metric dictionary is separately assumed.

## 2. Why not $g_{\mu\nu}=\delta S_{\rm ent}/\delta\rho$

The left side is a symmetric rank-2 spacetime tensor field. The right side is a functional derivative of a scalar functional. With respect to a scalar density it is a scalar. With respect to a density operator it is the operator $-\ln\rho-\mathbb{1}$ on Hilbert space. Neither carries spacetime indices, so no choice of $\rho$ turns it into a metric without extra structure. The shorthand has been removed from the repository README; any replacement needs an explicit tensor-valued construction.

Well-defined routes that capture the intent:

- **Kinematics: conditional chain length.** A negative logarithm gives an additive length only when the chosen positive kernel composes multiplicatively. The Q-region benchmark in [`docs/research/QREGION_MARKOV_MODEL.md`](docs/research/QREGION_MARKOV_MODEL.md) shows that pairwise mutual information is not itself such a kernel for a simple Markov chain. Any proposed map from mutual information to the multiplicative kernel needs its own derivation.
- **Dynamics: entanglement equilibrium.** Stationarity of small-ball entanglement entropy at fixed volume is equivalent to the Einstein equation (Jacobson 1995, 2016). This yields field equations *for* $g$, not $g$ itself.
- **Information geometry.** Fisher and Bures metrics live on state (parameter) space. They become spatial metrics only if the parameters are positions.

## 3. [Conditional] The two-factor chain law

**Assumptions.** From §2.1:

- (M) a dimensionless link kernel $0<K\leq1$ with Markov factorization, $K_{AC}=K_{AB}K_{BC}$ along a chain;
- (A) additivity, $d_{AC}=d_{AB}+d_{BC}$;
- $d=-\ell_P\ln K$ when $\Phi$ is constant.

Added here:

- (L) locality: the length of a short link depends only on that link's correlation and its local $\Phi$;
- (R) regularity: that length is continuous or monotone in the link's correlation, excluding pathological Cauchy-equation solutions;
- (G), only for a continuum spatial interpretation: the local correlation deficit has a smooth, positive-definite quadratic length form. This is not implied by (M), (A), (L), or (R).

**Statement.** Let $\sigma=-\ln K$ (nats of correlation deficit). By (M), $\sigma$ is additive along chains. If an additional continuum limit satisfying (G) exists, denote its local quadratic form by $\gamma_{ij}$, so that $d\sigma^2=\gamma_{ij}\,dx^i dx^j$. Along a differentiable radial chain $d\sigma=\kappa(r)\,dr$, with $\kappa=-\partial_{r'}\ln K(r,r')\big|_{r'=r}$. Under these assumptions,

$$\boxed{\;h_{ij}=\ell_P(\Phi)^2\,\gamma_{ij},\qquad g_{rr}=\ell_P(\Phi)^2\,\kappa^2\;}$$

**Proof.** Write the length of a link as $F(\Phi,\sigma)$. Splitting a link at fixed $\Phi$ multiplies its correlations (M) and adds its lengths (A), so $F(\Phi,\sigma_1+\sigma_2)=F(\Phi,\sigma_1)+F(\Phi,\sigma_2)$. With (R), this Cauchy equation forces $F=c(\Phi)\,\sigma$, and matching §2.1 gives $c=\ell_P$. Summing over links (L) and refining gives $d=\int\ell_P(\Phi)\,\kappa\,dr$. This is the §2.1 uniqueness argument applied link by link. ∎

**Corollaries.**

1. **The finite form needs constant $\Phi$.** §2.1's $d=-\ell_P(\Phi)\ln K$ is exact only when $\Phi$ is constant. For a smooth profile, using a single $\ell_P$ per pair errs by 6–52 %, depending on where $\Phi$ is evaluated (Sim7 check 1). Chain sums converge to the integral at $O(n^{-2})$.
2. **Reconciliation.** $\ell_P\to0$ and $g_{rr}\to\infty$ are compatible if and only if $\kappa$ grows faster than $1/\ell_P$. With uniform decay ($\kappa$ constant) the canonical profile can only contract. Uniform decay is the implicit assumption in Sim5's `physical_x = cumsum(local_l_p)`. *(Formal anchor, added 2026-10-20: this reconciliation, and the identification of the dilation ratio with the COD ratio, are formalised in lean_proofs/RadialDilation.lean as `dilation_reconciliation` and `lpRatio_eq_cod_ratio`, with the deviation of the ratio from its quadratic approximation in `lpRatio_quadratic_deviation`.)*
3. **Local dilation criterion.** For a conformally flat metric $h=\Omega^2(d\rho^2+\rho^2d\Omega^2)$ with $\Omega=\ell_P\kappa/(\ell_{P0}\kappa_0)$, the areal radius is $R=\Omega\rho$ and $g_{RR}=(1+d\ln\Omega/d\ln\rho)^{-2}$. With $\Phi$ decreasing outward, areal-gauge dilation ($g_{RR}>1$) needs
   $$\frac{d\ln\kappa}{d\Phi}>-\frac{d\ln\ell_P}{d\Phi}=\frac{\Phi}{1-\Phi^2},$$
   together with $d\ln\Omega/d\ln\rho>-2$ (no throat). The same function $\Phi/(1-\Phi^2)$ sets the golden-ratio bottleneck (§8), where it equals 1.

## 4. [Derived] $\ell_P(\Phi)$ alone cannot produce attraction

**Setting.** Correlation decay is uniform, so $\gamma$ is flat, as for Sim1's kernel $K=e^{-|i-j|/\xi}$. The profile is $\Phi=A/\rho$, harmonic on that flat background: the static massless form of the §3.1 equation with $Z=1$. Then $h=\Omega^2\delta$ with $\Omega=\sqrt{1-\Phi^2}$, and the following hold exactly (checked in SymPy and numerically):

| Quantity | Value | Meaning |
|---|---|---|
| areal radius | $R=\sqrt{\rho^2-A^2}$ | the $\Phi=1$ surface has zero area: a point, not a horizon |
| $g_{RR}$ | $(1-\Phi^2)^2\le1$ | contraction everywhere (0.036 at $\Phi=0.9$) |
| ADM mass | $0$, since $\Omega=1-A^2/2\rho^2+\dots$ has no $1/\rho$ term | no Newtonian far field |
| Misner–Sharp mass | $\tfrac R2\big(1-g_{RR}^{-1}\big)<0$ for all $R$ | negative quasi-local mass, $\to-\infty$ as $\Phi\to1$ |
| energy density (Hamiltonian constraint) | $\dfrac{A^2(A^2+2\rho^2)}{8\pi(\rho^2-A^2)^3}>0$ | a negative-mass point dressed by a positive-energy cloud |

**General form.** For $\ell_P=\ell_{P0}f(\Phi)$, $M_{\rm ADM}=f'(0)\,A$.

- The canonical $f=\sqrt{1-\Phi^2}$ is even, so $f'(0)=0$.
- The superseded exponential profile, rewritten in this convention as $\ell_{P0}e^{\Phi/\phi_c}$, has $f'(0)=1/\phi_c>0$, i.e. positive mass.

This is the quantitative content of the §2.2 remark that the two forms "point in opposite directions".

**Reading.** $\sqrt{1-\Phi^2}$ can serve as a *microscopic resolution scale*, but not as the macroscopic metric. The information metric $\gamma$ has to supply the mass.

**Formal anchor (Lean 4, added 2026-10-20).** Every row of the table above is formalised in lean_proofs/RadialDilation.lean: the contraction is `gRR_lpOnly_le_one` (with `gRR_lpOnly_eq_one_imp` and `gRR_lpOnly_zero` characterising the boundary cases), and the negative Misner–Sharp mass is `misnerSharpLPOnly_neg` (with `misnerSharpLPOnly_zero`). These are formal statements under the section's stated setting; they do not derive the required information metric.

## 5. [Identified] What GR demands: Schwarzschild written in $\Phi$

**The dictionary.** Consider a static vacuum, a time-symmetric slice, and a conformally flat metric $h=\psi^4\delta$. The Hamiltonian constraint is then $\nabla^2\psi=0$ (flat Laplacian). **Identify $\Phi:=\psi-1$**, which vanishes at infinity (Lean convention). The static equation for $\Phi$ is then the flat-background Laplace equation. That is the static massless form of the §3.1 equation with $Z=1$ and $V=0$, posed on the flat background rather than on the emergent metric. The single-centre solution $\Phi=M/2\rho$ gives exact Schwarzschild (4D Ricci-flatness checked symbolically):

$$ds^2=-\Big(\frac{1-\Phi}{1+\Phi}\Big)^2dt^2+(1+\Phi)^4\big(d\rho^2+\rho^2d\Omega^2\big).$$

Properties (all checked in Sim7):

- **Range.** $\Phi\in[0,1]$ covers exactly one exterior. $\Phi=1$ is the horizon: $\rho=M/2$, $R=2M$, area $16\pi M^2$. The throat isometry $\rho\mapsto M^2/4\rho$ acts as $\Phi\mapsto1/\Phi$.
- **Areal gauge.** $R=\rho(1+\Phi)^2$, $g_{RR}(\Phi)=\big(\frac{1+\Phi}{1-\Phi}\big)^2$, $g_{tt}=-1/g_{RR}$, and the Misner–Sharp mass is $M$.
- **Lapse.** $N=(1-\Phi)/(1+\Phi)$ solves the static equation $\nabla\cdot(\psi^2\nabla N)=0$ whenever $\nabla^2\Phi=0$.
- **Weak field.** $-g_{tt}\simeq1-4\Phi$, so $\Phi\simeq-\Phi_N/2c^2$. For scale, with $\Phi=GM/(2rc^2)$: Earth's surface $\approx3.5\times10^{-10}$, the Sun's surface $\approx1.1\times10^{-6}$, a $1.4\,M_\odot$ neutron star of radius 10 km $\approx0.10$, and a horizon $1$.
- **With matter** (time-symmetric): $\nabla^2\Phi=-2\pi(1+\Phi)^5\varepsilon$ (Lichnerowicz). This is what the static limit of an Omega field equation would have to reduce to: no potential term on these scales, and a source weighted by $(1+\Phi)^5$.
- **Several centres.** $h=(1+\Phi)^4\delta$ with $\Phi=\sum_i m_i/2|x-x_i|$ is valid (Brill–Lindquist) initial data, but it is not static. The lapse formula holds for one centre only.

**Required information metric.** By the two-factor law, $\gamma=h/\ell_P^2$, which requires

$$\frac{\kappa}{\kappa_0}=\frac{(1+\Phi)^2}{\sqrt{1-\Phi^2}}.$$

This satisfies the dilation criterion of §3 with margin $2/(1+\Phi)$.

*(Formal anchor, added 2026-10-20: the required κ of this dictionary is formalised in lean_proofs/RadialDilation.lean as `kappaRequired_pos` and `kappaRequired_zero`, its product with the ℓ_P factor as `kappa_lp_product`, the fact that it outruns the inverse ℓ_P factor as `kappa_outruns_inverse_lp`, and the conformal side as `isotropic_conformal_reproduction`, `conformalFactor_strictMono`, and `areal_reciprocity`. The Schwarzschild-side radial values are `gRR_schwarzschild_gt_one` and `gRR_schwarzschild_zero`. In lean_proofs/RadialMetric.lean the matching areal/lapse statements are `arealRadius_at_vacuum`, `arealRadius_at_horizon`, `lapse_conformal_identity`, and `cod_profile_matches_lapse`.)*

| $\Phi$ | 0 | 0.1 | 0.3 | $\Phi^*\approx0.618$ | 0.9 | 0.99 |
|---|---|---|---|---|---|---|
| $\ell_P/\ell_{P0}$ | 1 | 0.995 | 0.954 | 0.786 | 0.436 | 0.141 |
| required $\kappa/\kappa_0$ | 1 | 1.216 | 1.772 | 3.330 | 8.282 | 28.07 |
| $g_{RR}$, Schwarzschild | 1 | 1.494 | 3.449 | 17.94 | 361 | 39601 |
| $g_{RR}$, $\ell_P$ alone (§4) | 1 | 0.980 | 0.828 | 0.382 | 0.036 | 0.0004 |

$\kappa$ diverges like $(1-\Phi)^{-1/2}$, but the singularity is integrable, so the information distance to the horizon is finite. A chain built from these link correlations reproduces the Schwarzschild proper radial distance to a relative error of $4\times10^{-11}$.

The same $\gamma$ has two readings, depending on how Q-regions are placed:

- **Fixed coordinate spacing.** Neighbouring regions must *decorrelate* as $\Phi\to1$. If $\Phi$ measures pairwise shared correlation between neighbours, that is anti-monotone, hence a contradiction.
- **Fixed correlation per link.** The number of Q-regions per coordinate length grows as $\Phi\to1$: more, smaller cells.

Which reading holds depends on what $\Phi$ counts. At present $\Phi$ is defined only verbally ("local density of shared correlations"). Defining it as a functional of the same $I_{ij}$ that define $K$ would fix $\kappa(\Phi)$ and turn this section into a derivation.

**Remark: a coordinate-dependent identity.** In this dictionary $N\psi^2=1-\Phi^2=(\ell_P/\ell_{P0})^2$ exactly. Equivalently, the canonical profile is $\ell_{P0}\,\psi\sqrt N$, and $(\ell_P/\ell_{P0})^4=-g_{tt}\,g_{\rho\rho}$ in isotropic coordinates. In areal coordinates the corresponding product is identically 1. So this is at most a hint about which radial labelling a Q-region lattice might realise; it is not evidence for the profile.

**Other dictionaries.** The functional form of $g_{rr}(\Phi)$ depends on the dictionary, which is exactly the part not yet derived:

- areal, Lean convention: $\Phi=r_s/R$ gives $g_{RR}=1/(1-\Phi)$;
- areal, §1.2 convention: $\Phi=-g_{tt}=1-r_s/R$ gives $g_{RR}=1/\Phi$.

The isotropic choice is preferred here because only there is the static field equation exactly the flat Laplace equation. On Schwarzschild, $\Box(r_s/R)=-r_s^2/R^4\neq0$; the harmonic combination is $\ln(-g_{tt})$.

## 6. About $g_{rr}\propto1/\Phi$

This form is exactly Schwarzschild in areal gauge, but *in the §1.2 convention* ($\Phi=-g_{tt}=1-r_s/R$: vacuum at 1, horizon at 0). In the Lean convention it diverges at the vacuum ($\Phi\to0$), so it is not asymptotically flat, and it stays finite at the horizon. Sim7 check 4 tabulates both conditions for every candidate form.

Hence $g_{rr}\propto1/\Phi$ and $\ell_P=\ell_{P0}\sqrt{1-\Phi^2}$ cannot both hold for the same $\Phi$. Keeping it out of §2.2 was right, and it stays out: with the convention now fixed at $\Phi=0$ vacuum, $g_{rr}\propto1/\Phi$ is not asymptotically flat and is not a candidate. The form that *is* Schwarzschild in this convention is the isotropic dictionary of §5.

## 7. No-hair caveat: geometry or matter?

The candidate action in §4 of [`Omega_Theory_v4.0_Technical.md`](Omega_Theory_v4.0_Technical.md) varies an independent metric and therefore treats $\Phi$ as a minimally coupled scalar that sources gravity. Under regularity, asymptotic-flatness, and convex-potential conditions, the integral no-hair argument in [`docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md`](docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md) leaves only the constant scalar on a static black-hole exterior. This is a conditional result for that scalar model, not a theorem about every possible Q-region theory.

The Schwarzschild dictionary of §5 instead treats $\Phi$ as a coordinate-dependent label of the metric. Thus the matter-scalar action and the geometric dictionary are different constructions. A curvature-coupled gravitational-scalar candidate is now the provisional next-test branch; its linear zero mode is not a nonlinear solution and it is not derived from Q-regions. The fundamental ontology remains open, and any future map between these constructions must avoid double counting; the present calculations do not establish that "$\Phi\to1$ at the horizon" is a physical scalar boundary condition.

## 8. Side result: the golden-ratio bottleneck

**[Derived] Elementary proof of the global minimum.** Let $\Phi^*=(\sqrt5-1)/2$, so $1-\Phi^{*2}=\Phi^*$, and put $t=\Phi-\Phi^*$. Then

$$1-\Phi^2=\Phi^*(1-2t)-t^2\;\le\;\Phi^*(1-2t)\;\le\;\Phi^*e^{-2t},$$

using $e^x\ge1+x$ in the last step. Hence $e^{2\Phi}(1-\Phi^2)\le\Phi^*e^{2\Phi^*}$ for every real $\Phi$, with equality if and only if $\Phi=\Phi^*$.

So the disformal bound $e^{-\Phi}/(\sqrt\beta\sqrt{1-\Phi^2})$ has its unique global minimum on $(-1,1)$ at $\Phi^*$. The minimum value is $e^{-\Phi^*}/\sqrt{\beta\Phi^*}\approx0.6856/\sqrt\beta$. The proof needs no calculus, and it is now formalized: `RadialMetric.golden_bottleneck_bound` in `lean_proofs/RadialMetric.lean` proves the inequality via `Real.add_one_le_exp` and is kernel-checked by the Lean CI job. *(Added 2026-10-20: the companion statements in the same file are `phiStar_pos` and `phiStar_golden`, locating the bottleneck value; `golden_bottleneck_gap`, quantifying it; and `golden_bottleneck_eq_iff`, characterising equality. The finite-chain statements behind §3 are `chainLength_mono`, `chainLength_const_scale`, `chainLength_error`, and `chainLength_factorization_iff`.)*

**Robustness.** For $C=e^{-2a\Phi}$ and $D\propto(1-\Phi^2)^p$, the tightest point is

$$\Phi^*(a,p)=\frac{\sqrt{p^2+4a^2}-p}{2a},$$

which is golden if and only if $a=p$. For example, $a=\tfrac12,\,p=1$ gives $\sqrt2-1\approx0.414$, and $a=2,\,p=1$ gives $0.781$. The golden value is a property of the unit-rate choice, not a structural prediction.

**Where it would sit around a hole.** In the isotropic dictionary $R^*=\tfrac{2+\sqrt5}{4}\,r_s\approx1.059\,r_s$. In the areal dictionary $R^*=\tfrac{1+\sqrt5}{2}\,r_s\approx1.618\,r_s$.

Caveat: the bound constrains $\dot\Phi$ only. For the metric $Cg+D\,d\Phi\,d\Phi$ with $C=e^{-2\Phi}$ and $D=\beta(1-\Phi^2)$ on a flat background, the $(t,r)$ determinant is $C\,(D\dot\Phi^2-C-D\Phi'^2)$, so the exact signature condition is $D\dot\Phi^2<C+D\Phi'^2$ and the quoted bound $e^{-\Phi}/(\sqrt\beta\sqrt{1-\Phi^2})$ is the $\Phi'=0$ case of it. It is therefore sufficient for every profile and *conservative* rather than tight once the gradient is nonzero — the published number understates the room available. For static profiles the disformal term adds $D\,\Phi'^2\ge0$ to $g_{rr}$ and never threatens the signature, at any gradient, so this matters only for collapse or ringdown. Both statements are machine-checked by `research/verify_derivations.py` (`check_disformal_bound_depends_on_phidot`), for this $C$, $D$ and background only.

## 9. What would turn this into a derivation

1. **Notation fixed; physical interpretation open.** The current notes use $\Phi=0$ for the vacuum end. This resolves an internal convention clash, not the operational definition of $\Phi$.
2. **Two explicit Q-region benchmarks completed, each with a negative consistency result.** [`docs/research/QREGION_MARKOV_MODEL.md`](docs/research/QREGION_MARKOV_MODEL.md) defines $\Phi$ from adjacent-pair mutual information in a non-degenerate, classical-diagonal qubit Markov chain and derives $\kappa(\Phi)$. [`docs/research/ENTANGLED_TFIM_BENCHMARK.md`](docs/research/ENTANGLED_TFIM_BENCHMARK.md) uses pairwise-entangled ground states, verified by positive negativity, and fits a finite-range correlation-decay rate. In both fixed model families the decay rate decreases as $\Phi$ rises, opposite to the Schwarzschild-required trend. The entangled-chain sign persists across $N=8,10,12,14$ at a common fit window. This is not a no-go for richer Q-region models or a Schwarzschild fit.
3. Derive the transverse/areal structure from Q-region data, e.g. a specified state family and correlations across shells. The Schwarzschild identities are comparison targets; they must not be imposed as the answer to the derivation.
4. Keep three roles for $\Phi$ distinct: a minimally coupled matter scalar, a coordinate-dependent metric label, and a gravitational scalar degree of freedom. The first has a conditional GR branch but no nonconstant static hair under the stated convex-potential assumptions. The curvature-coupled Gauss–Bonnet branch is the provisional next-test model and has a linear s-wave zero-mode test, but no nonlinear solution or Q-region derivation; see [`docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md`](docs/research/COVARIANT_COMPLETION_AND_TEST_GATE.md) and [`docs/research/GAUSS_BONNET_SCALARIZATION.md`](docs/research/GAUSS_BONNET_SCALARIZATION.md). This is a workflow choice only; the fundamental ontology and any non-double-counting map remain open.
5. **Formalization completed, subject to the Lean audit.** `lean_proofs/RadialMetric.lean` contains the golden-ratio inequality, conditional finite-chain result, and $\Phi$-form Schwarzschild identities. These are mathematical results under their encoded assumptions, not a physical derivation.
6. `lean_proofs/RadialDilation.lean` formalizes the $\ell_P$-only obstruction and consequences of imposing the Schwarzschild-required $\kappa$. Kernel checking verifies those formal consequences; it does not derive the required $\kappa$ from Q-regions.
7. Before claiming an observational prediction, choose a covariant model, recover GR in its tested regime, derive a parameter-fixed observable, and preregister an independent-data test. No such Omega prediction is currently ready; the gate and required registration fields are in the covariant-completion note.

## References

- T. Jacobson, *Thermodynamics of spacetime: the Einstein equation of state*, Phys. Rev. Lett. 75, 1260 (1995), arXiv:gr-qc/9504004.
- T. Jacobson, *Entanglement equilibrium and the Einstein equation*, Phys. Rev. Lett. 116, 201101 (2016), arXiv:1505.04753.
- C. Cao, S. M. Carroll, S. Michalakis, *Space from Hilbert space: recovering geometry from bulk entanglement*, Phys. Rev. D 95, 024031 (2017), arXiv:1606.08444.
- J. D. Bekenstein, *Nonexistence of baryon number for static black holes*, Phys. Rev. D 5, 1239 (1972); *Novel "no-scalar-hair" theorem for black holes*, Phys. Rev. D 51, R6608 (1995).
- A. I. Janis, E. T. Newman, J. Winicour, *Reality of the Schwarzschild singularity*, Phys. Rev. Lett. 20, 878 (1968).
- D. R. Brill, R. W. Lindquist, *Interaction energy in geometrostatics*, Phys. Rev. 131, 471 (1963).
- C. W. Misner, D. H. Sharp, *Relativistic equations for adiabatic, spherically symmetric gravitational collapse*, Phys. Rev. 136, B571 (1964).
- R. Arnowitt, S. Deser, C. W. Misner, *The dynamics of general relativity* (1962), reprinted as arXiv:gr-qc/0405109.
- J. D. Bekenstein, *The relation between physical and gravitational geometry*, Phys. Rev. D 48, 3641 (1993), arXiv:gr-qc/9211017 (disformal metrics).
