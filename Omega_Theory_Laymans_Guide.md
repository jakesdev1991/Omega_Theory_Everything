<!-- Copyright (c) 2025-2026 Jacob See. Licensed under Apache-2.0. -->

# Emergent Reality from Quantum Correlation: Life IS Quantum
## A Layman's Guide to the Omega Theory

**Jacob See**

**Version:** 4.0 (2026-09-25)  
**Companion documents:** [`Omega_Theory_v4.0_Technical.md`](Omega_Theory_v4.0_Technical.md) for the formal statement, [`Omega_Theory_v4.0_Radial_Metric.md`](Omega_Theory_v4.0_Radial_Metric.md) for the open geometry work, and [`lean_proofs/README.md`](lean_proofs/README.md) for which claims are kernel-checked by the build.

> **What changed since v3.4.** This guide previously described the theory with the parameter $\Phi$ read one way; v4.0 fixes the opposite reading: **$\Phi = 0$ is the vacuum**, and $\Phi \to 1$ is maximal overlap. The theory also stopped treating mass as "the non-overlapping part" and now derives it from the energy stored in *gradients* of a single field. §1 of the radial-metric notes records the reconciliation; the technical paper's §1.2 and §3.1 carry the corrected convention.

### Abstract

Reality does not begin with matter or spacetime — it begins with correlation. Every piece of the universe is a quantum system that shares information with others. Where those correlations thin, space stretches; where they tighten, gravity deepens. There is one dial that measures how much shared information a region carries, and almost everything else in this guide is what happens when that dial moves.

---

## 1. Building Reality from Views, Not Stuff

### The Q-region: the smallest observer

The universe is not made of particles or points. It is made of **Q-regions** — bounded quantum systems that can hold a state and pass information along. A Q-region is what "an observer" means here: not a person, not a mind, just something that can retain and communicate information.

### The only thing there is to measure

Between any two Q-regions, the only fundamental quantity is **mutual information**: how much knowing about one tells you about the other. If two regions share nothing, nothing relates them. If they share everything, they stop being two things.

That is the whole ontology. Space, time, and matter all have to be built out of that one number.

---

## 2. The Master Dial: Chain Overlap Density ($\Phi$)

To turn a web of relationships into a continuous field, the theory defines **Chain Overlap Density**, written $\Phi$ (phi). It measures how much shared correlation a region carries, on a scale from 0 to 1:

| Value | Meaning |
|---|---|
| $\Phi = 0$ | **Vacuum.** No shared state. Nothing is being rendered here, and the operational scale sits at its relaxed baseline. |
| $0 < \Phi < 1$ | **Matter.** A "knot" of correlated information, partly overlapping its surroundings. The mismatch between the knot and the vacuum around it is what we call mass. |
| $\Phi \to 1$ | **Horizon.** Total overlap. The states have merged, the information that distinguished them drops to zero, and the network is severed. |

$\Phi$ is the master variable. Once you know it everywhere, you have the geometry, the gravity, and the expansion history.

> **A note on direction, because it trips people up.** Earlier drafts of this theory read $\Phi$ backwards, with $\Phi \approx 1$ as vacuum. The repository now uses one reading everywhere — $\Phi = 0$ vacuum, $\Phi \to 1$ horizon — because the kernel-checked profile and the simulation code both use it, and because a metric needs exactly one reading: space must flatten at the vacuum end and blow up at a horizon. If you are reading an older description, flip the dial in your head.

---

## 3. Distance Is Missing Information

How does a number — a correlation between two regions — become a physical distance?

The theory imposes two requirements. Correlations multiply along a chain (information combine multiplicatively), and distances add along a chain (length adds up). Only one kind of function turns multiplication into addition: a logarithm.

That gives the metric:

$$d = -\ell_P(\Phi)\,\ln\!\left(\frac{I}{I_{\max}}\right)$$

Read it plainly: **distance is the deficit of correlation.** Two regions that share almost everything are close; two regions that share almost nothing are far apart. Space is not a container — it is the bookkeeping cost of lost information.

---

## 4. The Ruler Itself Changes Size

The symbol $\ell_P(\Phi)$ in that equation is not a constant. It is the **informational stiffness of the vacuum** — the operational scale at which distances get rendered — and it depends on how much overlap is present:

$$\ell_P(\Phi) = \ell_{P0}\sqrt{1-\Phi^2}$$

At the vacuum ($\Phi = 0$) you recover the familiar baseline $\ell_{P0}$. As overlap approaches the maximum ($\Phi \to 1$, a bound-mass core or a horizon), the operational distance between overlapping states **collapses toward zero**. The variation of the ruler from place to place *is* curvature.

Two honest caveats, because they matter:

- This is a **model choice, not a derivation.** The axioms of the theory do not force this particular square-root profile.
- An earlier version of the theory used an exponential profile that pointed the *other* way — the ruler stretching in low-overlap regions. That intuition came from describing matter regions at macroscopic scales, and how the microscopic rendering scale relates to that picture is still an open problem. The working notes and numerical checks live in [`Omega_Theory_v4.0_Radial_Metric.md`](Omega_Theory_v4.0_Radial_Metric.md).

---

## 5. Gravity Without a Force

To make $\Phi$ a real field, the theory gives it an action — the same machinery physicists use for every other field — with a kinetic term (the cost of gradients) and a potential:

$$V(\Phi) = \tfrac{1}{2}m^2\Phi^2$$

That potential has its minimum at $\Phi = 0$: the universe prefers to relax toward the unshared vacuum, and it does so with Yukawa screening.

Varying the action produces an ordinary stress-energy tensor. The crucial move is what that tensor is made of: **mass is not "stuff." Mass is the energy stored in the gradients of $\Phi$.** Where the field changes steeply — high informational asymmetry — gravity is strong. Nothing pulls anything; regions simply fall toward wherever the field is steeper. The theory also derives an effective sound speed for information waves and requires it to be positive, which keeps updates inside a light cone instead of allowing them to outrun cause and effect.

---

## 6. Black Holes Are Shredders, and That Is Why the Universe Expands

A black hole is not merely a heavy object. Its horizon is a one-way membrane that **deletes mutual information from the outside universe**.

Because a horizon's area never shrinks, the universe's total shared information must fall at a rate tied to the total black-hole area:

$$\frac{dI}{dt} \le -c\,A_{BH}(t)$$

Turn that into a model of information as a finite resource and you get the **Depletion Law**, an exponential decay whose rate is set by total black-hole area. Since distance is the inverse of correlation, the expansion rate follows directly:

$$H(t) = \alpha\gamma\,A_{BH}^{\kappa}(t)$$

This is the theory's headline prediction, and it comes with a straight face: **the expansion rate of the universe is proportional to the total area of black-hole horizons.** Black holes grow, information thins, space stretches. No cosmological constant is needed. Because the decay rate slows as information runs out, the universe avoids a "Big Rip" and settles toward a stable de Sitter attractor with an effective equation of state $w_{\rm eff} \to -1$.

The prediction has history it can be checked against: during the quasar peak around $z \approx 2$, black holes grew super-exponentially, and the theory expects a phantom phase ($w_{\rm eff} < -1$) followed by today's approach to $-1$.

---

## 7. Life: The Master of the Shredder

The universe shreds information passively. Life shreds it **actively**.

Define a living system by what it does to the depletion law: it is a region that locally inverts it. A cell takes in highly correlated matter, shreds those correlations through metabolism, and captures the released energy to build new internal structure — DNA repair, protein folding, neural patterns. Input: high correlation. Output: low correlation. The capture is what makes it alive, because a black hole just loses that energy to the vacuum.

For a living system to persist, it must satisfy the **Omega Metabolic Inequality**:

$$\frac{dI_{\rm internal}}{dt} > \left|\frac{dI_{\rm env}}{dt}\right|_{\rm shred}$$

It must build internal correlations faster than it shreds external ones. Life is the only mechanism we know of that uses the universe's own expansion engine — information destruction — to power local assembly.

### The experiment behind that claim

The Long-Term Evolution Experiment (LTEE) is a decades-long real-world study of *E. coli*. The repository contains a computational replication of it (75,000 generations of digital populations), and the result is genuinely counter-intuitive:

- A complex new trait — the ability to eat citrate, called **Cit+** — emerges at about **generation 31,500**, as it does in the real experiment.
- Immediately afterwards, the population's internal overlap **drops**, from a pre-innovation average of **0.100** to about **0.050**.

$\Phi_{\rm post} < \Phi_{\rm pre}$ — advanced life does not maximize correlation. It *optimizes* it. By pruning redundant internal overlap, the population lowers its informational inertia and becomes more adaptable. In the theory's language, evolution is a trajectory toward informational efficiency: shred externally, construct internally, prune redundancy.

---

## 8. What Has Actually Been Checked

This guide is a design document's companion, not a textbook, so here is the honest ledger.

**Supporting the theory:**

- The logarithmic metric's uniqueness follows from two stated axioms and is a mathematical statement, not a physical claim.
- The dynamic-Planck-length profile $\ell_P = \ell_{P0}\sqrt{1-\Phi^2}$ is **formally verified in Lean** as a model: the build checks that the vacuum value is the baseline, that the horizon value is zero, that the scale never exceeds the baseline, and that it decreases strictly with overlap. That is a statement about the model, not about nature.
- Three simulation classes in the repository reproduce the expected behaviours: correlation kernels recover linear geometry, the depletion law produces a stable $w = -1$ attractor, and inertia defined as latency conserves momentum without Newton's laws being hard-coded.

**Not yet established:**

- The theory's assumptions about $\Phi$ are **model choices**. Neither the square-root profile nor the exponential one is derived from the axioms.
- Simulation agreement is **not** empirical confirmation. A simulation that reproduces a behaviour shows the model is self-consistent, not that the universe works that way.
- The macroscopic geometry is unfinished: reproducing Schwarzschild exactly, and relating the microscopic rendering scale to coordinate distances, are active open items with their own working notes.
- Nothing here licenses a claim about consciousness, economics, or human value. $\Phi$ is not a moral quantity, and the theory does not have one.

---

## 9. The Short Version

- Reality is a network of **Q-regions** whose only relation is shared information.
- **$\Phi$** measures that sharing: 0 is vacuum, 1 is a horizon.
- Distance is the **logarithm** of missing correlation.
- The ruler $\ell_P$ **shrinks** as overlap grows; its variation is curvature.
- Mass is the **energy in gradients** of $\Phi$, not a substance.
- Black holes **shred** information, and the shredding drives cosmic expansion.
- Life **inverts** the shredding locally to build order — and prunes its own redundancy to adapt.

Every one of those sentences is a hypothesis with a stated scope. The point of writing them down this plainly is that you can now ask exactly which piece you doubt, and go check it.

---

*Copyright (C) 2025-2026 Jacob See. Licensed under Apache-2.0; see [`LICENSE`](LICENSE).*
