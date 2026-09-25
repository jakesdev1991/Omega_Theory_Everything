<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT -->

# Toward the Macroscopic Radial Metric $g_{rr}(\Phi)$

*Working notes, companion to §2.2 of [`Omega_Theory_v4.0_Technical.md`](Omega_Theory_v4.0_Technical.md).*

**Status: exploratory.** Nothing here changes a canonical claim. Every result is tagged:

- **[Derived]**: follows from the stated axioms and assumptions by the argument given;
- **[Identified]**: a dictionary or postulate chosen to match known physics, *not* derived from the Omega axioms;
- **[Open]**: not settled.

Every formula is checked numerically, and where noted symbolically (SymPy), by [`Sim7_Radial_Metric.py`](Sim7_Radial_Metric.py). Its `self_check()` runs on every invocation: `python Sim7_Radial_Metric.py --no-plots`. Units are $G=c=1$ and $\ell_{P0}=1$ unless stated.

## 0. Summary

1. **Convention: settled.** The repository used to carry two opposite readings of $\Phi$ (§1); they are now reconciled on the reading of the kernel-checked profile (`DynamicCODScale.lean`), $\Phi=0$ being the vacuum and $\Phi\to1$ the bound-state/horizon end. A metric needs exactly one reading, because $g_{rr}\to1$ must hold at the vacuum end and $g_{rr}\to\infty$ at a horizon. The consensus potential is now $V=\tfrac12m^2\Phi^2$, so the vacuum is a static solution ($V(0)=V'(0)=0$) and the horizon end is not an attractor. Under the *other* reading ($\Phi=1$ vacuum, as in earlier drafts of §1.2), $\ell_P(\text{vacuum})=\ell_P(1)=0$, which no metric can use.
2. **$g_{\mu\nu}=\delta S_{\rm ent}/\delta\rho$ is not a well-formed tensor equation** (§2). The Omega axioms already contain a well-defined route: the §2.1 logarithmic metric.
3. **[Derived] Two-factor law** (§3). Markov factorization, additivity and locality give $ds=\ell_P(\Phi)\,d\sigma$ with $d\sigma=-d\ln K$. Hence $h_{ij}=\ell_P(\Phi)^2\gamma_{ij}$, and along a radial chain $g_{rr}=\ell_P(\Phi)^2\kappa^2$. Microscopic contraction and macroscopic dilation are compatible. The open item reduces to one unknown: the correlation-decay metric $\gamma$ (radial rate $\kappa$).
4. **[Derived] Obstruction** (§4). Suppose $\ell_P(\Phi)$ were the only $\Phi$-dependence of the spatial metric, with a harmonic profile $\Phi=A/\rho$. Then $g_{RR}=(1-\Phi^2)^2\le1$ (contraction), the ADM mass is zero, the Misner–Sharp mass is negative, and the "horizon" has zero area. The canonical profile cannot by itself produce gravitational attraction.
5. **[Identified] Schwarzschild written in $\Phi$** (§5). With $\Phi:=\psi-1$ (the isotropic conformal factor minus one), the exact vacuum solution is
   $$ds^2=-\Big(\frac{1-\Phi}{1+\Phi}\Big)^2dt^2+(1+\Phi)^4\big(d\rho^2+\rho^2d\Omega^2\big),\qquad \Phi=\frac{M}{2\rho}.$$
   In areal gauge, $g_{RR}(\Phi)=\big(\frac{1+\Phi}{1-\Phi}\big)^2$. $\Phi$ is harmonic, $\Phi\in[0,1]$ is exactly one exterior, and $\Phi=1$ is the horizon. By the two-factor law this requires $\kappa/\kappa_0=(1+\Phi)^2/\sqrt{1-\Phi^2}$.
6. **$g_{rr}\propto1/\Phi$** (§6) is exactly Schwarzschild in the *other* convention ($\Phi=-g_{tt}$, vacuum at 1). In the Lean convention it diverges at the vacuum. It cannot coexist with $\ell_P=\ell_{P0}\sqrt{1-\Phi^2}$ for the same $\Phi$.
7. **[Open]** Derive $\gamma$ from a microscopic Q-region model, and decide whether $\Phi$ is geometry or matter (no-hair caveat, §7).

## 1. Two conventions for $\Phi$ are in the repository

| Location | Vacuum | Bound matter / horizon |
|---|---|---|
| Technical note §1.2 | $\Phi=0$ ("no shared state") | $0<\Phi<1$ matter; $\Phi\to1$ horizon ("states merged") |
| Technical note §3.1 | $V=\tfrac12m^2\Phi^2$, minimum at $\Phi=0$ | — |
| `Sim5_Emergent_Gravity.py` | `PHI_VACUUM = 0.0` | matter 0.4, black hole 0.9 |
| Technical note §2.2, `DynamicCODScale.lean`, Sim3's $\ell_P$ | $\Phi=0$ ("unshared vacuum") | $\Phi\to1$ ("bound-mass core or horizon") |

The repository now reads $\Phi$ one way only. The change that resolved it: §1.2 and §3.1 of the technical note were flipped to the Lean reading, Sim3's consensus potential became $V=\tfrac12m^2\Phi^2$, and Sim5 was re-based on the COD profile with `PHI_VACUUM = 0.0`, matter at $\Phi>0$ and the black hole at $\Phi=0.9$. Sim5's earlier `exp((1-\Phi)/\phi_c)` profile was dropped for the Lean-proven $\sqrt{1-\Phi^2}$, which is bounded and so keeps the black hole visible.

Why this reading and not the other one:

- **It is the one that is kernel-checked.** $\ell_P=\ell_{P0}\sqrt{1-\Phi^2}$ is proved in `DynamicCODScale.lean` with the vacuum at $\Phi=0$; the opposite reading would make $\ell_P(\text{vacuum})=0$.
- **The potential then works.** $V=\tfrac12m^2\Phi^2$ has $V(0)=V'(0)=0$, so $\Phi=0$ is a static, asymptotically flat vacuum, and it is Yukawa-screened ($\Phi\propto e^{-m\rho}/\rho$). Sim3's default run relaxes from its initial kick back toward $\Phi=0$ (peak 0.153, still 0.138 at $t=40$ under Hubble damping) and never reaches the horizon.
- **The other reading would need the Lean profile relabelled**, e.g. $\ell_P=\ell_{P0}\sqrt{1-(1-\Phi)^2}$, and would then make $g_{rr}\propto1/\Phi$ natural. It was rejected because it discards a kernel-checked module to save an unverified guess.

The formulas below use this convention throughout; §6 gives the §1.2 translation where it matters.

## 2. Why not $g_{\mu\nu}=\delta S_{\rm ent}/\delta\rho$

The left side is a symmetric rank-2 spacetime tensor field. The right side is a functional derivative of a scalar functional. With respect to a scalar density it is a scalar. With respect to a density operator it is the operator $-\ln\rho-\mathbb{1}$ on Hilbert space. Neither carries spacetime indices, so no choice of $\rho$ turns it into a metric without extra structure. The formula also appears as the "Theory Foundation" line of `README.md`.

Well-defined routes that capture the intent:

- **Kinematics: distance from correlations.** The §2.1 logarithmic metric, a special case of defining distance as a decreasing function of mutual information (Cao, Carroll, Michalakis 2017). Used below.
- **Dynamics: entanglement equilibrium.** Stationarity of small-ball entanglement entropy at fixed volume is equivalent to the Einstein equation (Jacobson 1995, 2016). This yields field equations *for* $g$, not $g$ itself.
- **Information geometry.** Fisher and Bures metrics live on state (parameter) space. They become spatial metrics only if the parameters are positions.

## 3. [Derived] The two-factor law

**Assumptions.** From §2.1:

- (M) Markov factorization, $K_{AC}=K_{AB}K_{BC}$ along a chain;
- (A) additivity, $d_{AC}=d_{AB}+d_{BC}$;
- $d=-\ell_P\ln K$ when $\Phi$ is constant.

Added here:

- (L) locality: the length of a short link depends only on that link's correlation and its local $\Phi$;
- (R) regularity: that length is continuous or monotone in the link's correlation.

**Statement.** Let $\sigma=-\ln K$ (nats of correlation deficit). By (M), $\sigma$ is additive along chains, so in the continuum it defines an *information metric* $d\sigma^2=\gamma_{ij}\,dx^i dx^j$. Along a radial chain $d\sigma=\kappa(r)\,dr$, with $\kappa=-\partial_{r'}\ln K(r,r')\big|_{r'=r}$. Then

$$\boxed{\;h_{ij}=\ell_P(\Phi)^2\,\gamma_{ij},\qquad g_{rr}=\ell_P(\Phi)^2\,\kappa^2\;}$$

**Proof.** Write the length of a link as $F(\Phi,\sigma)$. Splitting a link at fixed $\Phi$ multiplies its correlations (M) and adds its lengths (A), so $F(\Phi,\sigma_1+\sigma_2)=F(\Phi,\sigma_1)+F(\Phi,\sigma_2)$. With (R), this Cauchy equation forces $F=c(\Phi)\,\sigma$, and matching §2.1 gives $c=\ell_P$. Summing over links (L) and refining gives $d=\int\ell_P(\Phi)\,\kappa\,dr$. This is the §2.1 uniqueness argument applied link by link. ∎

**Corollaries.**

1. **The finite form needs constant $\Phi$.** §2.1's $d=-\ell_P(\Phi)\ln K$ is exact only when $\Phi$ is constant. For a smooth profile, using a single $\ell_P$ per pair errs by 6–52 %, depending on where $\Phi$ is evaluated (Sim7 check 1). Chain sums converge to the integral at $O(n^{-2})$.
2. **Reconciliation.** $\ell_P\to0$ and $g_{rr}\to\infty$ are compatible if and only if $\kappa$ grows faster than $1/\ell_P$. With uniform decay ($\kappa$ constant) the canonical profile can only contract. Uniform decay is the implicit assumption in Sim5's `physical_x = cumsum(local_l_p)`.
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

## 5. [Identified] What GR demands: Schwarzschild written in $\Phi$

**The dictionary.** Consider a static vacuum, a time-symmetric slice, and a conformally flat metric $h=\psi^4\delta$. The Hamiltonian constraint is then $\nabla^2\psi=0$ (flat Laplacian). **Identify $\Phi:=\psi-1$**, which vanishes at infinity (Lean convention). The static equation for $\Phi$ is then the flat-background Laplace equation. That is the static massless form of the §3.1 equation with $Z=1$ and $V=0$, posed on the flat background rather than on the emergent metric. The single-centre solution $\Phi=M/2\rho$ gives exact Schwarzschild (4D Ricci-flatness checked symbolically):

$$ds^2=-\Big(\frac{1-\Phi}{1+\Phi}\Big)^2dt^2+(1+\Phi)^4\big(d\rho^2+\rho^2d\Omega^2\big).$$

Properties (all checked in Sim7):

- **Range.** $\Phi\in[0,1]$ covers exactly one exterior. $\Phi=1$ is the horizon: $\rho=M/2$, $R=2M$, area $16\pi M^2$. The throat isometry $\rho\mapsto M^2/4\rho$ acts as $\Phi\mapsto1/\Phi$.
- **Areal gauge.** $R=\rho(1+\Phi)^2$, $g_{RR}(\Phi)=\big(\frac{1+\Phi}{1-\Phi}\big)^2$, $g_{tt}=-1/g_{RR}$, and the Misner–Sharp mass is $M$.
- **Lapse.** $N=(1-\Phi)/(1+\Phi)$ solves the static equation $\nabla\cdot(\psi^2\nabla N)=0$ whenever $\nabla^2\Phi=0$.
- **Weak field.** $-g_{tt}\simeq1-4\Phi$, so $\Phi\simeq-\Phi_N/2c^2$. For scale: Earth's surface $\approx3.5\times10^{-10}$, the Sun's surface $\approx1\times10^{-6}$, a $1.4\,M_\odot$ neutron star $\approx0.1$, a horizon $1$.
- **With matter** (time-symmetric): $\nabla^2\Phi=-2\pi(1+\Phi)^5\varepsilon$ (Lichnerowicz). This is what the static limit of an Omega field equation would have to reduce to: no potential term on these scales, and a source weighted by $(1+\Phi)^5$.
- **Several centres.** $h=(1+\Phi)^4\delta$ with $\Phi=\sum_i m_i/2|x-x_i|$ is valid (Brill–Lindquist) initial data, but it is not static. The lapse formula holds for one centre only.

**Required information metric.** By the two-factor law, $\gamma=h/\ell_P^2$, which requires

$$\frac{\kappa}{\kappa_0}=\frac{(1+\Phi)^2}{\sqrt{1-\Phi^2}}.$$

This satisfies the dilation criterion of §3 with margin $2/(1+\Phi)$.

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

§3.2 derives $T^\Phi_{\mu\nu}$ by varying $S_\Phi$ with respect to an independent metric. It therefore treats $\Phi$ as a minimally coupled scalar that *sources* gravity. In that reading:

- Bekenstein's no-scalar-hair theorems (1972; 1995) force $\Phi$ to sit at its potential minimum everywhere outside a static black hole. They cover the massless case and the quadratic consensus potentials used here.
- Static massless solutions with non-constant $\Phi$ are the Fisher/Janis–Newman–Winicour family, which have a naked singularity and no horizon.

So "$\Phi\to1$ at the horizon" is impossible if $\Phi$ is matter. The dictionary of §5 treats $\Phi$ as part of the metric, the §2 reading. §2 and §3.2 therefore describe different theories. The framework has to choose, or show how $\Phi$ can be both without double counting.

## 8. Side result: the golden-ratio bottleneck

**[Derived] Elementary proof of the global minimum.** Let $\Phi^*=(\sqrt5-1)/2$, so $1-\Phi^{*2}=\Phi^*$, and put $t=\Phi-\Phi^*$. Then

$$1-\Phi^2=\Phi^*(1-2t)-t^2\;\le\;\Phi^*(1-2t)\;\le\;\Phi^*e^{-2t},$$

using $e^x\ge1+x$ in the last step. Hence $e^{2\Phi}(1-\Phi^2)\le\Phi^*e^{2\Phi^*}$ for every real $\Phi$, with equality if and only if $\Phi=\Phi^*$.

So the disformal bound $e^{-\Phi}/(\sqrt\beta\sqrt{1-\Phi^2})$ has its unique global minimum on $(-1,1)$ at $\Phi^*$. The minimum value is $e^{-\Phi^*}/\sqrt{\beta\Phi^*}\approx0.6856/\sqrt\beta$. The proof needs no calculus, and it is now formalized: `RadialMetric.golden_bottleneck_bound` in `lean_proofs/RadialMetric.lean` proves the inequality via `Real.add_one_le_exp` and is kernel-checked by the Lean CI job.

**Robustness.** For $C=e^{-2a\Phi}$ and $D\propto(1-\Phi^2)^p$, the tightest point is

$$\Phi^*(a,p)=\frac{\sqrt{p^2+4a^2}-p}{2a},$$

which is golden if and only if $a=p$. For example, $a=\tfrac12,\,p=1$ gives $\sqrt2-1\approx0.414$, and $a=2,\,p=1$ gives $0.781$. The golden value is a property of the unit-rate choice, not a structural prediction.

**Where it would sit around a hole.** In the isotropic dictionary $R^*=\tfrac{2+\sqrt5}{4}\,r_s\approx1.059\,r_s$. In the areal dictionary $R^*=\tfrac{1+\sqrt5}{2}\,r_s\approx1.618\,r_s$.

Caveat: the bound constrains $\dot\Phi$ only. For static profiles the disformal term adds $D\,\Phi'^2\ge0$ to $g_{rr}$ and never threatens the signature, so this matters only for collapse or ringdown.

## 9. What would turn this into a derivation

1. ~~Fix the convention across §1.2, §2.2 and §3.1 of the technical note, Sim3's potential and Sim5, plus the Lean docstrings if the convention flips.~~ **Done:** the whole repository now uses $\Phi=0$ as the vacuum (§1).
2. Build an explicit Q-region model in which $\Phi$ is defined from the $I_{ij}$, and test $\kappa(\Phi)$ against $(1+\Phi)^2/\sqrt{1-\Phi^2}$. A Gaussian or free-fermion lattice, where mutual information is computable, is one option.
3. Obtain the transverse/areal structure from Q-region data, e.g. area-law entanglement across shells, so that $g_{tt}g_{RR}=-1$ is *tested* rather than imposed.
4. Decide $\Phi$'s role (§7) and re-derive §3.2 accordingly. Its static limit should reduce to the Lichnerowicz form of §5.
5. ~~Lean: formalize the golden-ratio minimum (§8), the finite-chain two-factor law (§3) and the $\Phi$-form Schwarzschild identities (§5).~~ **Done:** `lean_proofs/RadialMetric.lean` proves all three and is kernel-checked by the Lean CI job.

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
