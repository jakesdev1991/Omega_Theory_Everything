<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE. -->

# PREDICTIONS REGISTER (draft, 2026-10-20)

Tests and discriminators this programme has explicitly registered, with
their sources and epistemic status. An entry here is a commitment to a
stated test and a stated interpretation of its outcomes — it is **not**
a claim that the effect exists. Entries adopted from external work are
labelled with their source and remain the cited authors' results, not
results of this programme. Physics in this repository is exploratory
and unvalidated; nothing in this register changes that.

---

## PRED-EDG-LAB-01 — Atom-interferometer response to engineered entanglement gradients

**[External / Conditional — attributed to EDG (Ðulović, Zenodo,
Sep 2025), not derived within Omega]**

*Claim:* A slab (thickness d) carrying a near-constant gradient
g_s = ∂_x s of an EDG-type entanglement density acts as a mass density
ρ_s = αg_s²/2c² (+ V(s₀)/c²), producing a differential acceleration
g_EDG = 2πGdρ_s and an atom-interferometer phase
ΔΦ = (m_at/ħ)(πGd/c²)(αg_s² + 2V(s₀))ΔxT (EDG Eqs. 32–33), spanning
~10⁻¹⁰–10⁻⁶ rad across EDG's α benchmarks (10⁻²–10²) for a 10 m / 1 s
geometry.

*Test:* Large-momentum-transfer atom interferometry across a
state-engineered slab, with the entanglement gradient modulated at
**fixed ordinary mass distribution**.

*Null-result value:* ΔΦ < ΔΦ_min implies
α ≲ ħΔΦ_min c²/(πGd g_s² m_at Δx T) (EDG Eq. 30).

*Open feasibility items (blocking):* (i) no physical protocol or
achievable value for g_s is known — the entry is void until one exists;
(ii) the ordinary-mass systematic must be shown controllable by
modulation; (iii) Omega-side status: no Φ→s dictionary exists, so this
tests EDG's scalar, not Φ, until the Route-B candidate of Technical
§1.1 is worked out.

*Status:* Registered 2026-10-20. Do not cite as an Omega prediction.

*Source:* Damir Ðulović, *Entanglement Density Gravity: A Dynamical
Theory of Emergent Spacetime from Quantum Information Constraints*,
Zenodo record 17227471, 30 September 2025. Preprint, not peer-reviewed.
EDG's own caveats apply in full: its scalar dynamics are postulated
(an explicitly labelled "EFT hypothesis"), and the phase range above
is a parameterised sensitivity estimate over the free coefficient α
with g_s left symbolic, not a parameter-free prediction.

---

## PRED-WZ-01 — The w(z) phantom-crossing discriminator

**[Adopted discriminating test — between EDG's canonical thawing
cosmology and Omega's depletion ansatz. Conditional; no numerical fit
exists in either programme.]**

*Claim:* The two information-based dark-energy mechanisms in this
programme's comparison set make opposite statements about the phantom
regime. EDG's canonical slow roll gives
w_s = −1 + αṡ²/ρ_s ≥ −1 at all times (EDG Eq. 23): it approaches −1
strictly from above and cannot cross. The depletion cosmology of
Technical §5, as developed in the Drive ToE 4.4 line, permits epochs
with w_eff < −1 (phantom epochs).

*Test:* Percent-level determinations of the dark-energy equation of
state w(z) at z ≲ 1 from DESI-class and successor SNe Ia / BAO
programmes, with the phantom-crossing question pre-registered as the
discriminating observable.

*What each outcome does:*
- **Confirmed phantom crossing (w < −1 at any probed epoch):** excludes
  the EDG baseline as formulated (canonical kinetic term, minimal
  coupling). Does not by itself confirm the depletion ansatz, which
  remains an underived phenomenological assumption with calibrated
  freedom.
- **Persistent w ≥ −1 at percent precision:** tightens EDG's slow-roll
  bound α ≳ λ_s²/(3ϵ_w) (EDG Eq. 26, with 1 + w₀ ≤ ϵ_w). Leaves the
  depletion ansatz's calibration freedom untouched — i.e. constrains
  EDG, does not confirm depletion.
- **w indistinguishable from −1 at all probed precision:** squeezes
  both mechanisms' natural parameter space (EDG's falsification
  criterion in its own FAQ couples this outcome with laboratory nulls
  under PRED-EDG-LAB-01).

*Status:* Registered 2026-10-20. Both sides of the comparison are at
ansatz / slow-roll level; this entry registers a discriminator, not a
prediction by either programme.

*Sources:* EDG as above (Eqs. 23, 26); depletion cosmology as stated in
Omega_Theory_v4.0_Technical.md §5.
