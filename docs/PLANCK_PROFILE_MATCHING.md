# Matching the two Planck-scale profiles

This analytic calculation is **not a Lean-checked construction** and is not a
physical derivation of either profile. The conditional results described below
have been Lean-checked. The definitions being compared are
[`dynamicPlanckLength`](../lean_proofs/DynamicPlanckScale.lean) and
[`dynamicPlanckCOD`](../lean_proofs/DynamicCODScale.lean).

## 1. Matching on physical densities

Let `L > 0` be the **same** baseline in both environments, `ρ₀ > 0`, and
`ρ ≥ 0`. Require `0 ≤ Φ ≤ 1` and equality of the two lengths:

```text
L / √(1 + ρ/ρ₀) = L √(1 − Φ²).
```

At `ρ = 0` the density definition takes its nonpositive branch, but the
left-hand formula above still equals `L`. For positive `ρ` it is the positive
branch. Since both sides divided by `L` are nonnegative, squaring preserves
and reflects equality. Writing `x = ρ/ρ₀ ≥ 0`, equality is equivalent to

```text
1/(1 + x) = 1 − Φ²,
Φ² = x/(1 + x),
Φ = √(x/(1 + x)).
```

The last step uses `Φ ≥ 0`. Conversely this formula lies in `[0, 1)` because
`0 ≤ x/(1+x) < 1`, and substituting it gives equality of the nonnegative
lengths. Thus there is **exactly one physical overlap for each finite
nonnegative density** that matches the profiles. No monotonicity assumption is
needed for uniqueness; it follows pointwise from the range restriction.

The dictionary is strictly increasing: for `0 ≤ x < y`, positive denominators
and cross multiplication give `x/(1+x) < y/(1+y)`; the square root is strictly
increasing on nonnegative arguments. The inverse, for `0 ≤ Φ < 1`, is

```text
ρ = ρ₀ Φ²/(1 − Φ²).
```

Consequently it is a bijection from finite `ρ ∈ [0, ∞)` to `Φ ∈ [0, 1)`.
At `Φ = 1`, the COD length is zero, whereas the density length is strictly
positive at every finite real density. There is no finite-density match for
that endpoint; `Φ → 1` only as `ρ → ∞`. A claimed bijection onto the closed
interval `[0, 1]` would therefore be false.

## 2. What happens outside those assumptions

- For `ρ ≤ 0`, the density profile is clamped to `L`. Under the same-baseline
  and physical-range assumptions, matching forces `Φ = 0`. Extending the
  dictionary this way gives a nondecreasing function on all reals, but it is
  **not** strictly increasing or injective there.
- Without `0 ≤ Φ ≤ 1`, uniqueness fails: the COD profile is even, so positive
  and negative overlaps yield the same length. Its total real definition also
  uses the truncated real square root outside the physical range.
- The common baseline is an explicit identification, not a theorem relating
  the two environment structures. If their baselines differ, the vacuum
  requirements `ρ = 0`, `Φ = 0` and length matching force them to be equal.
- `ρ₀` remains free. Most importantly, equality of lengths does not establish
  that the physical COD functional and information-mass density actually obey
  this dictionary. It is a necessary and sufficient **matching condition** for
  the chosen profiles, not an independent derivation of either. The two chosen
  profile rows and the missing microscopic Φ functional remain open.

## 3. Formalisation acceptance criterion

A Lean bridge should import the existing two modules and prove, without new
axioms, `sorry`, or added carried laws: for a shared positive baseline and
`ρ₀ > 0`, the displayed dictionary maps nonnegative finite densities into
`[0, 1)`, matches the existing definitions including `ρ = 0`, is the unique
matching value in `[0, 1]`, and is strictly increasing there. It should also
prove the inverse on `[0, 1)` and the impossibility of a finite match at `Φ = 1`.
Separately state the clamped extension for negative densities if used. The
statement must not assert that the dictionary follows from Q-region physics.

`lean_proofs/PlanckProfileMatching.lean` contains an initial Lean bridge for
conditional uniqueness, the maximal-overlap obstruction, the clamped branch,
and strict monotonicity on positive densities for **any** physical-range matching
function. The latter is conditional and does not construct such a function.
These declarations passed the Lean build and transitive kernel-axiom audit in
[Lean CI run 42114531942](https://github.com/jakesdev1991/Omega_Theory_Everything/actions/runs/42114531942)
at commit `b8048ae`. The full formalisation remains open until existence,
monotonicity including the zero boundary, and the inverse are also proved and
the build and axiom audit pass. A failure to formalise this particular proof is
not evidence that no dictionary exists.
