# What would move this programme toward a theory of everything

This note answers a single question: *what can be done to make this closer to a
theory of everything?* It does not restate the aspiration. It records where the
programme actually stands against the obligations a theory of everything has to
discharge, and it ranks the work that would move the number, including the work
that would honestly move it the other way.

The ledger it measures is `docs/theory_ledger.json`, checked by
`research/toe_census.py` and run by the consistency gate. That file is a list of
judgements, not a proof; what is enforced is only that the list exists, that
every evidence pointer resolves, and that the two counts cannot quietly grow.

## 1. The four obligations, and the position against each

A candidate theory of everything has to do four things. Not four volumes' worth —
four things.

| Obligation | Where this programme stands | Ledger rows |
| --- | --- | --- |
| Specify its own basic object without a free function | **Not met.** Φ is a functional of Q-region data with no construction. Two candidate routes are recorded, neither adopted. | `phi-functional`, `phi-s-dictionary` |
| Derive the spacetime metric from that object | **Not met.** Two inequivalent Planck-scale laws coexist; the macroscopic-metric diagnostics trend *opposite* to the Schwarzschild-required trend. | `lp-phi-profile`, `lp-density-profile`, `kappa-dilation`, `macroscopic-metric` |
| Derive, rather than assume, the standard-model and gravity bridges | **Not met.** 16 `law_*` fields carry Einstein's equations, the cosmological constant, Bekenstein–Hawking, the modular structure and `d⋆F = ⋆J` into the formal layer. | 16 carried `law_*` fields |
| Produce at least one number, computable before measurement | **Not met.** Two registered predictions exist; both are external or conditional, neither carries a number fitted on this side. | `depletion-rate`, `depletion-exponent`, `nonminimal-coupling`, `rho-critical`, `phi-star`, `freeze-threshold` |

The epistemic ledger in `Omega_Theory_v4.0_Technical.md` §7 already says this in
one line: *Theory of Everything claim — Not supported by the current results*.
Nothing below contradicts that line. The point of this note is that the line is
also the starting position, and it can be measured.

## 2. The measured distance

```
python research/toe_census.py --list
```

> 17 tracked items — 9 undetermined, 4 chosen, 1 identified, 2 postulate,
> 1 external; plus 16 carried `law_*` fields in the Lean corpus.
> open distance: 14 (ratchet 14) | carried laws: 16 (ratchet 16)

Fourteen open choices and sixteen carried laws is the distance. It is not a
large number, and that is the important observation: the corpus is not short of
material, it is short of *decisions*. Each of the fourteen is a place where a
form was picked or a gap was left, and each has a named evidence pointer, so
each can be worked on or closed off. What cannot be done is to make the number
smaller by writing another volume.

## 3. The moves, in order of value per unit of work

**1. Reconcile the two Planck-scale laws — or prove they cannot be reconciled.**
`DynamicPlanckScale.dynamicPlanckLength` makes ℓ_P a function of density;
`DynamicCODScale.dynamicPlanckCOD` makes it a function of Φ. Both are proved
positive and strictly monotone. Nothing relates them, which means the corpus
currently contains two inequivalent theories of the same quantity.

This is the highest-value formal target in the repository, because either answer
is progress. Exhibit the unique monotone dictionary Φ(ρ) that makes
`dynamicPlanckCOD (Φ ρ) = dynamicPlanckLength ρ` hold for every ρ and both laws
collapse into one — a derived statement, not a carried one, and the first
genuinely new bridge in the corpus. Or prove no such dictionary exists, which
retires one of the two profiles and removes a degree of freedom. Both outcomes
are Lean-sized, both land inside existing modules, and neither touches the
`carried_cap` in `statement_audit.json` (33, at the limit — do not raise it).

**2. Audit the two literal constants before strengthening anything else.**
`freezeBoundaryThreshold : ℝ := 0.012` in `OmegaAxioms.lean` is the most
falsifiable-looking number in the formal corpus, and the theorems near it are
`Iff.rfl` because the threshold *is* the definition. The zero-valued primitives
next to physical names (`QFIM`, `Hessian`, `state_at`, `tangent_at`, `vacuum_at`)
need the same treatment: each is either a witness placeholder — fine, but then
say so in the docstring and in `statement_audit.json` — or it is load-bearing in
a physics statement, in which case the statement is about the zero, not the
physics. Until that audit is done, any claim strengthened elsewhere rests on an
unaudited floor.

**3. Choose one side of Open Problem 1 and confine the other.**
The JBD identification takes non-minimal coupling seriously, which is what makes
the radial dilation and the conformal factor work; the adopted EDG route
suppresses it. Both are recorded as live, so `ξ` is simultaneously a free
parameter and a fixed constant, depending on which section a reader is in. A
theory cannot carry both completions of itself. Pick one, state the scale at
which the other is asserted to be negligible, and record the choice as a ledger
row with a status — that converts an unresolved fork into a testable
approximation.

**4. Make the metric question decidable, not merely open.**
The Markov and entangled-chain diagnostics both show the decay-rate trend
running opposite to the Schwarzschild-required trend. That is not a gap waiting
for a bigger simulation; it is a disagreement that will survive one. The next
step is a *stated continuum limit* — which Q-region geometry, which scaling of
region size and number, which error control — written down narrowly enough that
its Schwarzschild limit can be checked and, if necessary, refuted now. If the
limit cannot be stated, that is itself the finding, and it should be recorded in
§7 as such rather than as "open".

**5. Ship one number per year, through `onset_eval`.**
The only part of this repository that already behaves like science is
`packages/onset_eval`: pre-registered protocols, three-outcome verdicts
(`PASS` / `KILL` / `UNDECIDED`), a false-alarm budget per thousand, and a test
that cannot return a p-value when it has one seed. Any empirical arm — including
the RCOD comparison against the EDG functional — should go through that harness
with the prediction registered in `docs/PREDICTIONS_REGISTER.md` *before* the
data, in the same format as `PRED-WZ-01`. A registered prediction with a number
and a kill criterion is worth more to this programme than the rest of the
corpus, because it is the only kind of statement the corpus cannot absorb.

## 4. Kill criteria

Intellectual honesty in a programme this size is mostly a matter of naming in
advance what would end it. These are the observations that should count:

- **The dictionary of move 1 does not exist, and neither profile is derivable
  from anything.** Then Φ is a parameter, the Planck-scale profile is a family of
  models, and the programme is a framework — which is a legitimate thing to be,
  just not a theory of everything, and should be described as such everywhere.
- **The stated continuum limit of move 4 does not reproduce Schwarzschild even
  under generous error control.** Then the macroscopic-metric claim is not merely
  open, it is false for the model as constructed, and §7 should say so in those
  words.
- **The 0.012-style literals cannot be derived or bounded.** Then the formal
  layer proves theorems about chosen constants, and the ratio of derived to
  chosen content should be reported alongside any formal-coverage number.

## 5. What not to do

Do not add volumes. The corpus is 87 proof-map rows, 163 audited statements and
33 carried axioms deep; its unification layer is *structural*, by its own
description in §9, and structure does not accumulate into physics by
accumulation. Sixteen carried laws do not become fifteen by writing a
seventeenth volume; they become fifteen by deriving one.

The honest deliverable here is a smaller set of sharper claims, and the ratchet
in `docs/theory_ledger.json` exists so that the smaller set is the only direction
the number can move. One open row closed and one prediction with a number in it
would do more for this programme's standing than any further expansion — and,
unusually for this literature, it is a result that can be checked by
`python research/toe_census.py`.
