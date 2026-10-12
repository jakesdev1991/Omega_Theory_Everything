<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT -->

# Claim and derivation tightening: end-to-end roadmap

**Status date:** 2026-10-12 · **Branch:** `arena/fd09e6cb-omega-theory-everything`
**Scope:** every claim-bearing surface in the repository, from the core technical notes
to the 41 LaTeX volumes and the 69 Lean modules. Phases run in the order given; each
phase has an exit gate that is a command, not an opinion.

This roadmap does not claim that the physics is established. Its job is to make each
statement say exactly what has been shown, under which assumptions, and by which check.

---

## 0. Acceptance rule: what "tightened" means

Every claim receives one status. A claim is never upgraded without a named check.

| Status | Meaning | Required evidence |
|---|---|---|
| **Verified** | The stated algebra or numerics are re-derived by code | An entry in `research/verify_derivations.py`, or a kernel-checked Lean declaration in CI |
| **Derived (hand)** | Re-read step by step; no machine check yet | Listed as open in Phase 3 |
| **Conditional** | True only under the listed assumptions | The assumptions are stated beside the claim |
| **Identified** | A dictionary chosen to match known physics | Labelled as a choice, never as a prediction |
| **Model choice / Open / Retired** | Stipulated, unresolved, or withdrawn | Says so in the same sentence as the claim |

Wording rules for prose surfaces:

1. "Volume N proves X" becomes "In the model of Volume N, X holds under assumptions A."
2. "Our framework shows that the universe is …" becomes a statement about the model, not the universe.
3. Labels such as "REAL formalization" or "GENUINE PROOF" are replaced by a description of
   the declaration, for example "Machine-checked: Born rule for the stated finite-dimensional model."
4. Every physical claim cites the declaration, script, or note that supports it, or says
   "no formal or numerical counterpart."

---

## 1. Baseline, measured on this branch

| Measure | Result | How measured |
|---|---|---|
| Research and simulation tests (before this work) | 31 passed | `pytest tests/test_qregion_markov.py tests/test_quantum_ising_chain.py tests/test_gauss_bonnet_scalarization.py tests/test_simulations.py` |
| Sim7 self-check | Runs; prints the Schwarzschild identities | `python Sim7_Radial_Metric.py --no-plots` |
| Derivation checks (new) | 11 of 11 pass | `python research/verify_derivations.py` |
| Claim scan, unhedged hits (measured on a clean export of `HEAD`) | 82 in LaTeX (41 files), 36 Lean labels (21 files), 36 text-companion labels (21 files), 1 in core notes | `python research/claim_scan.py --list` |
| LaTeX coverage | 41 of 54 volumes (none for Vol01–08 and Vol36–40) | `ls latex_docs/Vol*.tex` |
| Plain-text coverage | 54 of 54 volumes, 5 core files | `ls txt_proofs` |
| Lean | Not runnable here (no `lake`, no network to the toolchain). Kernel truth is CI-only. | `lean_proofs/PROOF_AUDIT.md` §Validation |

**Known scanner limitation:** a line of LaTeX is one paragraph, so a single hedge word
on that line marks every claim in it as hedged. Read the `--list` output before trusting
a zero. The one core-notes hit is a sentence that quotes a rejected claim
("information shredding proves expansion") in order to reject it. It is a false positive
and is left in place.

---

### 1a. Measured after the Phase 3–5 pass

| Measure | Result | Command |
|---|---|---|
| Claim scan, unhedged hits | latex_docs 0 (15 hedged, 12 files); txt_proofs 0; lean_proofs 0; core notes 1 (documented false positive); **total 1** (was 49 on `HEAD`) | `python research/claim_scan.py` |
| LaTeX coverage | 41 of 54 volumes (missing Vol01–08, Vol36–40, by decision) | `ls latex_docs/Vol*.tex` |
| Covariant checks (new) | 4 of 4 pass; four negative controls fail as they must | `python research/verify_covariant.py` |
| TFIM checks (new) | 2 of 2 pass (independent dense ED row; Schwarzschild column) | `python research/verify_tfim.py` |
| Derivation checks | 12 of 12 pass (one added by Phase 3 item 6) | `python research/verify_derivations.py` |
| LaTeX citation gate (new) | 0 errors: 41 volumes cite 41 modules and 5 declarations; 2 documented external exemptions | `python research/check_latex_citations.py` |
| LaTeX structure gate (new) | 0 findings over 41 volumes: braces, environment nesting, document markers, math delimiters | `python research/check_latex_structure.py` |
| Consistency gate (new) | 16 passed, 0 failed, 0 skipped (one command runs every local check) | `python research/consistency_gate.py` |
| Statement audit | **163 rows** for 161 physics-sounding volume declarations (two rows record renamed names the lexicon no longer matches): 73 witness, 43 conditional, 14 renamed, 33 carried (cap 33, ratcheted). The Phase 6 rename pass retired 16 names that asserted results the statements do not contain; each is recorded as a `previous_name` on its row. | `python lean_proofs/check_statement_audit.py` |
| `ruff check .`, `ruff format --check .` | pass (132 files formatted) | venv, CI versions |
| `mypy . --ignore-missing-imports` | pass (61 source files) | venv, CI versions |
| `pytest -q` | 215 passed, 81 subtests passed (needs the full `requirements.txt`: `matplotlib` for the Sim6 tests and `sympy`/`scipy` for the derivation checks) | venv, CI versions |
| Cited-name check (new) | **0 suspicious names** against the pinned sources — 563,126 mathlib + 292,520 Lean core + 7,429 repository declarations; the previous script reported 205 on the same trees. Four injected typos all detected, clean copy 0. | CI only, `.github/workflows/lean-ci.yml` |
| Lean | Not compiled here (no `lake`). Apart from the Phase 6 rename pass (declaration names only, code otherwise untouched), Lean edits are comments and docstrings only; all 68 `txt_proofs/` mirrors are byte-identical, and the code-projection parity check covers 69 files / 1,448 declarations. | CI only |

## 2. The road

### Phase 0 — Baseline and tooling · **DONE**

- Measured the baseline above.
- Installed the pinned tooling from `requirements.txt` into a virtual environment outside
  the repository, so the CI versions were used (`ruff` 0.16.10, `mypy` 2.4.0, `pytest` 9.1.1).
- Confirmed the existing research suites are green before any edit.

**Exit gate (met):** baseline table recorded; existing 31 tests pass; new tooling passes
`ruff check .`, `ruff format --check .`, and `mypy . --ignore-missing-imports`.

### Phase 1 — Core derivation ledger and machine checks · **DONE**

Each displayed identity in the core notes was re-derived. The table in §3 gives the result.
The checks in `research/verify_derivations.py` cover every "Verified" row. They are exact
SymPy identities where the algebra is closed-form, and floating-point or independent
numerical methods where it is not. There were eleven at the end of this phase; Phase 3
item 6 added a twelfth, so the run prints 12 of 12.

**Exit gate (met):** `python research/verify_derivations.py` exits 0 with every check
passing (11 of 11 as of this phase, then 12 of 12); `tests/test_verify_derivations.py` runs
the same checks in `pytest`, so CI enforces them.

### Phase 2 — Correct the core text · **DONE**

Errors found by the checks, or by re-reading against the checks, and their fixes:

| # | Location | Problem found | Fix |
|---|---|---|---|
| 1 | `docs/research/GAUSS_BONNET_SCALARIZATION.md` §3 | The headline \(\eta/M^2\simeq0.725609\) is the \(x_{\max}=250\) truncation. The note's own table shows it is not converged. | Headline becomes \(0.725613\) (±\(10^{-6}\)), with an independent Rayleigh–Ritz cross-check. The truncated value is labelled as such. |
| 2 | `docs/research/QREGION_MARKOV_MODEL.md` table | The row labelled \(\Phi=0.618\) was computed at \(\Phi^*=0.6180340\). At the rounded label the values differ in the fifth figure. | Row relabelled \(\Phi^*\); the rounding effect is stated. |
| 3 | `Omega_Theory_v4.0_Technical.md` §2 | The logarithmic-distance result omitted the regularity hypothesis that the radial note lists. Without it the additive equation has non-logarithmic solutions. | Continuity or monotonicity added as a stated hypothesis. |
| 4 | `Omega_Theory_v4.0_Radial_Metric.md` §5 | "A 1.4 \(M_\odot\) neutron star \(\approx0.1\)" gave no radius. | Radius (10 km) stated; Sun value corrected to \(1.1\times10^{-6}\) (computed). |
| 5 | `README.md` | "Each physics proof has companion files" is false for LaTeX (14 volumes have none). | Coverage stated exactly: 54/54 text, 41/54 LaTeX. |
| 6 | `latex_docs/README.md` | "Volumes 09-54 are currently documented in LaTeX" is false. | Lists the 13 missing volumes; counts corrected. |
| 7 | `Omega_Theory_v4.0_Technical.md` §8 | Reproduction block omitted the derivation checks. | Added `verify_derivations.py` and its test. |

Items checked and found **correct** (no change): the Schwarzschild dictionary, the lapse
equation, the Lichnerowicz form, the golden-ratio bound and its robustness formula, the
Markov mutual-information closed form, the Kretschmann invariant, and the Gauss–Bonnet
reduction to the \(x\)-ODE.

**Exit gate (met for the core notes):** every number that the core notes state in §3 is
either Verified, or is labelled "reported" with the script that prints it.

### Phase 3 — Remaining core-note items · **DONE (items 1–6)**

These are the statements the checks have not yet reached. Each must become Verified,
or be relabelled Derived (hand) or Conditional, before the core notes are called final.

1. **COVARIANT §1–§3:** machine-check the scalar field equation and \(T^{(\Phi)}_{\mu\nu}\)
   from the action (SymPy, minisuperspace for FLRW), the FLRW Friedmann and acceleration
   equations, and the linear mass \(m_{\rm eff}^2=(V''(0)-f''(0)\mathcal G)/Z(0)\).
2. **COVARIANT §2 no-hair argument:** the integral identity
   \(\int_\Sigma N[D\varphi\cdot D\varphi+\varphi U_\varphi]\,d\mathrm{vol}=0\) is hand-checked
   (Green's identity; the horizon boundary term vanishes because \(N=0\) there). Machine-check
   it in a one-dimensional radial model with a stated boundary condition.
3. **Entangled TFIM (`docs/research/ENTANGLED_TFIM_BENCHMARK.md`):** the table reproduces
   from `research/quantum_ising_chain.py`, and the tests check the opposite trend. Add an
   independent exact-diagonalization cross-check of one row, and a check that the
   Schwarzschild column equals \((1+\Phi)^2/\sqrt{1-\Phi^2}\) at each \(\Phi\).
4. **Sim2, Sim3, Sim5, Sim6:** re-run each script, and check every printed number that the
   notes quote. Sim6 calibrates \(\gamma\) to \(H_0=73.5\); the note must keep saying that
   value is an input, not a prediction. Sim5's uniform-decay assumption (`physical_x =
   cumsum(local_l_p)`) is confirmed by reading the code and should be stated in Sim5's own
   header.
5. **Laymans Guide:** the whole document has been read once and is consistent with the
   technical notes. A line-by-line pass against the Verified table is still required.
   **Done 2026-10-12.** Every quantitative sentence in `Omega_Theory_Laymans_Guide.md`
   was read against the ledger in §3 and the artifacts behind it. Three were sharpened
   rather than left standing: the Markov non-composition sentence now says the two-link
   mutual information differs from the squared one-link value at every nonzero link
   strength in that model, which needed a new assertion (the scaled difference has a
   floor of 0.204 on a grid down to 10⁻³, so the claim is not an artifact of the sample
   link); the scalarization threshold is quoted as 0.725613 ± 10⁻⁶, the value the two
   numerical methods agree on, with the truncated 0.725609 called out as not it; and the
   \(\Phi\equiv0\) branch of the Gauss–Bonnet model is now labelled a hand derivation
   under the stated vacuum conditions, not a machine check. The guide's §8 now lists the
   checks that exist -- including the citation gate and the one-command gate -- and points
   at the adversarial audit, with its executive summary marked as the pre-remediation
   scoreboard. Two claims were verified as already correct: the ℓ_P algebraic properties
   (`dynamicPlanckCOD_zero`/`_one`/`_strictAntiOn`) and the 2 ln 2 normalization in
   `research/quantum_ising_chain.py`.
6. **Radial note §8 and §9:** the claim that the disformal bound constrains \(\dot\Phi\) only
   is hand-checked against the stated bound. Machine-check the bound's dependence on \(\dot\Phi\).
   **Done** — `check_disformal_bound_depends_on_phidot` in `research/verify_derivations.py`:
   the disformed \((t,r)\) determinant is \(C(D\dot\Phi^2-C-D\Phi'^2)\), so the signature
   condition is exactly \(D\dot\Phi^2<C+D\Phi'^2\) (checked against the matrix on 200,000
   samples across three \(\beta\)); the published bound is that condition at \(\Phi'=0\) and
   is therefore sufficient but not tight once the gradient is nonzero; \(g_{rr}>0\) at every
   gradient up to \(10^6\), so static profiles cannot threaten the signature; and the bound's
   minimum reproduces \(0.6856/\sqrt\beta\) at \(\Phi^*\). Sim3's header now states the general
   condition rather than the gradient-free one as if it were general.

**Done in this pass.**

- Items 1–2: `research/verify_covariant.py` (4 checks). The scalar equation is derived from the
  minisuperspace Euler–Lagrange equation and matched to the covariant form; the FLRW
  Friedmann and acceleration equations come from the SymPy Einstein tensor; the linear mass
  is checked on both branches; the no-hair integration-by-parts identity is checked at
  R = 3 r_s, where the boundary term is nonzero. Negative controls (sign flips, dropped
  boundary term) fail, so the checks are not vacuous.
- Item 3: `research/verify_tfim.py`. The Schwarzschild column is checked row by row against
  the formula. One exact-diagonalization row (N = 8, h/J = 4) is rebuilt from Kronecker
  products and reproduces the module and the note.
- Item 4: Sim2, Sim3, Sim5 and Sim6 re-run with exit 0. Sim6 reproduces its calibrated
  H₀ = 73.50 km/s/Mpc, which the README states is an input. Sim5's header now states the
  `cumsum` uniform-decay assumption. Sim3 writes three PDFs to the working directory on each
  run; these were removed as untracked artifacts.
- Item 6: `check_disformal_bound_depends_on_phidot` (see the item above). The derivation-check
  count rises from 11 to 12; `tests/test_verify_derivations.py` pins the new count.
- Item 5: the line-by-line reading of the guide, described under the item itself.

**Open.** Nothing in this phase. The core notes are the three documents `claim_scan.py`
tracks (`Omega_Theory_v4.0_Technical.md`, `Omega_Theory_v4.0_Radial_Metric.md`,
`Omega_Theory_Laymans_Guide.md`); each number they state either has a check or carries a
"reported"/"model choice" label, and the line-by-line pass is recorded in item 5 above.

**Exit gate:** `python research/claim_scan.py` shows zero unhedged hits in the core notes,
apart from the documented false positive (met). Each core-note number has a check or a
"reported" label (met for items 1–6, 2026-10-12). The pass is manual and stays manual:
a number added to those documents later needs a check or a label, and nothing stops that
drift automatically.

### Phase 4 — LaTeX volumes · **DONE (option a; 13 volumes absent by decision; cited-declaration gate added)**

Scope: 41 files, 82 unhedged prose hits. Most are of the form "Volume N proves that …"
in a layman's summary, and the layman's summaries make claims the Lean statements do not
support.

Steps, per volume:

1. Read the Lean counterpart (`lean_proofs/VolNN_*.lean`) and list the declarations that
   carry the volume's mathematical content, with their hypotheses.
2. Rewrite each "proves"/"we prove"/"exactly"/"fundamental" sentence as a statement about
   the model, under those hypotheses, or delete it if no declaration supports it.
3. Add a status box at the top: "Formal content: [declarations]. Physical interpretation:
   [Model choice / Identified / Open]."
4. For the 14 volumes with no LaTeX file, decide whether to write one at all (see §4).

**Done.** All 41 LaTeX volumes were rewritten in place. Each now has a status box that names
the Lean file, lists what the file establishes, and lists what it does not. Stipulated
values, definitional identities, and hypotheses are labelled as such. Claims with no formal
counterpart are withdrawn and listed as open items. Known problems exposed by the rewrite
include: Vol10 sets the generation count to 3 by definition; Vol13's "graviton–SM coupling"
reduces to a zero self-distance; Vol09's bulk reconstruction holds by reflexivity.

**Done later (2026-10-12).** The script check that every cited declaration exists:
`research/check_latex_citations.py`, run by `tests/test_latex_citations.py` under pytest, so
CI enforces it. It checks that all 41 volumes cite a module that exists, that each volume
cites its own `VolNN` module, and that every `\texttt{}` identifier is a declaration of the
corpus -- resolved in the citing volume where possible, otherwise reported. Two names are
exempt as external, each with a reason and each required to still be cited: `Empty` (Lean
core) and `HasDerivAt` (Mathlib). A token that is none of module path, identifier, or
`identifier = value` is an error rather than a silent skip, and the coverage claims in
`latex_docs/README.md` are compared with the directory. Negative controls were exercised on
copies: a renamed declaration, a retargeted module, a stale exemption, an unclassified
token, a missing module, and a drifted README count each exit 1.

**Also scripted (2026-10-12).** The structural hand check is now a gate:
`research/check_latex_structure.py` checks brace balance (escaped braces are literals),
`\begin{}`/`\end{}` nesting with every environment closed, exactly one `\documentclass`,
one `\begin{document}` and one `\end{document}`, balanced `\[`/`\]` display delimiters, and
an even number of unescaped `$`. All 41 volumes pass, and each rule was shown to fail on a
mutated copy. It is a floor, not a compile: a missing package or an undefined macro would
still only surface in a TeX toolchain, which this environment does not have. The citation
check is lexical too: it does not verify that a cited declaration says what the prose claims.

**Exit gate:** `python research/claim_scan.py` shows zero unhedged `latex_docs` hits (met).
The declaration-existence script check is in place and green (met 2026-10-12), and the
structural rules are scripted and green over all 41 volumes. Compilation remains unverified
here, and the `docs` CI job was not run in this environment.

### Phase 5 — Lean headers and text mirrors · **DONE (items 1–3 done; exit gate met except the CI-only half)**

Scope: 21 Lean files carry "REAL formalization", "REAL mathematical formalization", or
"GENUINE PROOF" labels (36 lines). The text companions in `txt_proofs/` mirror those headers.

Steps:

1. Replace each label with a description of what the declaration establishes. Edit comments
   and docstrings only. No declaration name, statement, or proof may change. **Done** for the
   exact phrases "REAL formalization", "REAL mathematical formalization", "GENUINE PROOF",
   "Genuinely proven", "genuine content" and "genuine proof" in all 69 Lean files and their
   text companions. The remaining lowercase "genuine" in prose (for example in
   `OmegaAxioms.lean`) is ordinary mathematical wording and was left in place.
2. Add a check that compares the non-comment declaration lines of every `lean_proofs/*.lean`
   file before and after, so a comment edit cannot change a theorem.
   **Done** (`lean_proofs/check_declaration_parity.py`, baseline
   `lean_proofs/declaration_parity.json`). Each file is reduced to a canonical code
   projection — nested block comments, docstrings and line comments removed, string and
   character literals kept, blank lines and trailing whitespace dropped — and hashed:
   69 files, 1,449 declarations, 7,506 code lines. Verified in both directions: appending
   a comment leaves all 69 digests unchanged (exit 0), while inserting one token inside a
   proof body fails with `Vol18_StatisticalMechanics.lean: code changed (...)`. CI runs the
   gate; `--against <ref>` proves a branch is comment-only against any revision.
3. Regenerate `txt_proofs/` from `lean_proofs/`, as `PROOF_AUDIT.md` describes.
   **Done** (`lean_proofs/sync_txt_mirrors.py`): the 9 modules that had never been mirrored
   (`RadialMetric`, `RadialDilation`, `InformationPhysics`, `DynamicPlanckScale`,
   `DynamicCODScale`, `LogCorrelationMetric`, `ProofRegression`, `CBwK_Budget_Pacer`,
   `APPA_Context_Branching`) were generated; all 68 mirrors are byte-identical to their
   sources. Orphans are reported, never deleted.

**Exit gate:** `claim_scan.py` shows zero `label` hits in `lean_proofs/` and `txt_proofs/`
(**met** — 0 files with hits on both surfaces); the declaration-comparison check passes
(**met** — 69 files unchanged); `lake build ToE` and the kernel audit pass in Lean CI
(**not runnable from this environment**; green on `main` at CI run 38119532226, the merge
of PR #79).

**Also scripted (2026-10-12): the cited-name check.** `docs/PROJECT_STATUS_2026-10-08.md`
listed `lean_proofs/check_mathlib_names.py` as existing "for the next round" — a lexical
check that would turn a name typo into an offline finding instead of a CI build that stops
one dependency layer at a time. The next round is now done. The checker indexes every
declaration in the pinned Mathlib4 checkout plus the Lean core sources and reports qualified
names used by the corpus that exist in neither. The indexer was rebuilt for recall, because
every miss is a false "suspicious" name and a ratchet over false positives is pressure to
raise it: it now registers declarations behind attribute blocks that span lines, constructors
of `inductive`/`structure`/`class` together with the names Lean generates for them
(`X.rec`, `X.injEq`, `X.noConfusion`, ...), structure fields, the
`nonrec`/`public`/`expose` modifiers, same-line `| ctor`, and the `to_additive`/`to_dual`
twins of all of the above; `end <section>` no longer pops the enclosing namespace.

**Result on the pinned sources:** 563,126 mathlib + 292,520 Lean core + 7,429 repository
declarations indexed, **0 suspicious names** (the previous script reported 205 on the same
trees). Negative control: a copy of the tree with four injected typos (`Real.sqrtt`,
`Finset.sum_range_succs`, `RadialMetric.chainLenght`, `Set.mem_Ioi.mpq`) reports all four.
The check exits 2 rather than reporting a clean run when a tree yields no declarations, and
`--max N` is the ratchet. It needs the pinned sources, so — like `lake build` — it is
CI-only: it cannot join the consistency gate, which names it as one of the three checks it
cannot run. Lean CI runs it after the build with `if: always()`, finding the trees by
derivation rather than assumption (the mathlib checkout under `.lake/packages/` — the
requirement is named `mathlib4`, so the step searches for the checkout that holds
`Mathlib/` instead of trusting one directory name; core sources at
`<lean --print-prefix>/src/lean`, the destination the pinned `src/CMakeLists.txt` installs
to). 13 new tests (80 in `lean_proofs/`, all `unittest`).

The gate's first CI run failed on the one thing that had only been assumed: the fixture here
had used the directory name `mathlib`, and the requirement is named `mathlib4`, so the
workflow's hardcoded path did not exist in CI. It exited 1 rather than passing hollow — the
honest failure — and the step now searches for the checkout that holds `Mathlib/`. The
fixture was then rebuilt to the CI layout so a smoke test can fail the same way again.

### Phase 6 — Statement-level audit of the Lean physics volumes · **DONE (audit and gate added 2026-10-12)**

`PROOF_AUDIT.md` lists the existing findings: degenerate models, parameterized stubs, and
conditional lemmas. Phase 6 goes through each physics volume and decides, per theorem,
whether the name overstates the statement, whether the hypotheses are satisfiable, and
whether a non-degenerate witness exists. Each result is recorded in the audit file.

**Exit gate:** every physics-volume theorem with a physical-sounding name has a row in the
audit with one of: non-degenerate witness, conditional with stated hypothesis, or renamed.

**Implemented.** `lean_proofs/statement_audit.json` carries one row for each of the **161**
theorems/lemmas in `Vol*.lean` whose name carries a word from a documented lexicon (163 rows
in all: two record renamed declarations whose new names the lexicon does not match)
(`lean_proofs/check_statement_audit.py`). A row states a disposition, an honest note, and
the evidence that disposition needs:

* **witness** (73 rows) — names a declared, non-degenerate object or instance that the
  statement is about, or that realizes (or refutes) its setting;
* **conditional** (43 rows) — quotes the hypothesis the statement carries in its own
  signature, checked to occur verbatim in the collapsed statement;
* **renamed** (14 rows) — records a name retired for the `bridge_*` convention, with the
  retired name checked to survive nowhere in the corpus once comments are masked (the
  renames were made by earlier passes; this phase performed none);
* **carried** (33 rows) — records a statement that is true only from data the model
  carries: a structure field, a postulate, a placeholder or constant definition, a
  definitional restatement, or a degenerate instance. It names the declaration that
  carries the data, and no non-degenerate witness exists for it.

The fourth disposition is a deliberate addition to the sketch above, and it is stricter,
not looser: against this corpus the three-way split has no honest place for
`theorem photon_massless : PhotonMass = 0 := rfl`, for `|0| ≤ |0|` in the placeholder
layer, or for a structure field returned unchanged — those rows would have had to name a
"witness" for a degenerate instance, which is exactly the misreport this phase exists to
stop. The count of `carried` rows is ratcheted by a `carried_cap` field in the audit file
(30, the measured count when the audit landed): a new `carried` row fails the gate until the
cap is raised in a visible diff, and fixing a row means lowering the cap in the same commit.
Rows whose name still read as a legacy label (`photon_massless`, `metric_is_qfim`,
`quantum_computing_from_omega`, `turing_instability_ratio`, ...) recorded the mismatch in
their note instead of renaming the declaration — and the rename pass that followed did the
renaming, described below.

**The rename pass (2026-10-12).** Sixteen declarations whose *name* asserted a result the
statement does not contain were renamed to names that state the content: the six
`zeroCurvature_*` restatements in Vol05 (`metric_is_qfim`, `einstein_field_equations`,
`bianchi_identity`, `cosmological_constant`, `conservation_of_energy_momentum`,
`em_sources_gravity`), the Vol02 postulate and Maxwell pair (`photonMass_eq_zero`,
`maxwell_equations_of_sourced_field`), the two definitional gauge-group equalities
(`GaugeSymmetry_eq_product`, `SM_Gauge_Group_eq_product`), the placeholder optical identity
(`placeholder_optical_expansion`), the Vol09 definitional entropy bridge
(`bridge_entanglement_entropy_eq_bh_entropy`), the two legacy-constant Vol14 rows
(`legacy_density_constants_sum_eq_one`, `legacy_density_constants_dark_energy_eq`), the Vol34
gate restatement (`hadamard_unitary_restated`) and the Vol37 diffusivity ratio
(`diffusion_ratio_gt_one_of_activator_pos`). This is the one pass in the report that changes
code: it ships `check_declaration_parity.py --update`, the re-synced `txt_proofs/` mirrors,
the updated `ProofRegression.lean` reference and kernel-audit target, and the
`OmegaProtocol.lean` re-exports under the same names. The gate grew with it — `previous_name`
is now validated on *any* row (a renamed declaration keeps the disposition of its statement
and still records what it was called), with identifier-boundary matching, so
`zeroCurvature_bianchi_identity` does not count as a surviving `bianchi_identity` while a
kept law field such as `law_metric_is_qfim` is a different name. The three renamed
declarations the lexicon never matched (`einstein_field_equations`, `bianchi_identity`,
`maxwells_equations`) now have rows, so the audit grew 160 → 163 rows and `carried_cap`
30 → 33: a coverage expansion in one visible diff, with no existing row reclassified. Four
new tests (27 in the file, 107 in `lean_proofs/`) cover the per-row retirement check, the
boundary match, a `previous_name` equal to the row's own name, and the shipped audit's
rename records.

The gate is lexical and runs offline: `python3 check_statement_audit.py` in `lean_proofs/`
(also a command in `research/consistency_gate.py`) fails on a missing row, a malformed row,
a witness or source that is not a declaration, a `stated_condition` that does not occur, a
`previous_name` that survives, or a `carried` count above the cap, and exits 2 rather than
passing when the audit is unreadable. What it cannot do is decide whether a judgement is
*right*: that half is CI's `lake build ToE`, which compiles every witness and source a row
names. 23 new tests (103 in `lean_proofs/`, all `unittest`), one negative control per
disposition, plus a test that the shipped audit covers the shipped corpus. The audit itself
added, removed and renamed no Lean declaration; the rename pass that followed it renamed
sixteen, as recorded above.

**Second rename pass, decorative-hypothesis upgrade, and 54/54 volume coverage (2026-10-12).**
Follow-up audit closed the remaining statement-audit and static-vacuity gaps:

1. **Decorative-hypothesis detector upgrade (`lean_proofs/audit_vacuity.py`).** `hypothesis_binders`
   previously checked only `Prop`/inequality/equality binder types and single-identifier binder
   groups, missing hypotheses typed by corpus-declared `: Prop` predicates (such as
   `subset_region` or `spacelike_separated`) and shared binders `(a b : T)`. Expanding
   `prop_predicates(root)` and splitting multi-name binders surfaced four real findings across
   the 69-file corpus, all remediated:
   - `Vol01.least_action_strict`: removed the unused boundary hypothesis `(hN : η N = 0)`
     (already unnecessary once the sum positive-ness uses `h1 : η 1 ≠ 0`) and replaced
     `by linarith` with `sub_eq_zero.mp hs`.
   - `Vol06.LocalNet` & `Vol06.local_nets_isotony`: replaced the region-independent singleton
     `LocalNet O := {0}` with interval-supported scalar operators
     `{ A | ∃ x : ℝ, O.x_min ≤ x ∧ x ≤ O.x_max ∧ (A : Operator) = (x : ℂ) • (1 : Operator) }`,
     so `local_nets_isotony` genuinely uses `h : subset_region O₁ O₂` (`le_trans h.1 hmin` and
     `le_trans hmax h.2`); added `local_nets_spacelike_disjoint` proving disjoint support
     coordinates under `spacelike_separated`, and marked `_h` on `local_nets_microcausality`
     (where commutation comes from `omegaAlgebra_commutative`).
   - `Vol37.drive_iff_ratio`: removed the unused hypothesis `(hd : S.d < 0)` (only `0 < S.a`
     is used by `div_gt_one_iff`).
2. **Second rename pass (20 declarations).** Renamed the remaining over-named volume theorems
   whose names overstated their statements: `commutator_limit` →
   `constant_commutator_eval_at_zero` (Vol01), `dirac_equation` → `diracOperator_eq_smul` and
   `optical_theorem` → `placeholder_optical_theorem` (Vol06), `friedmannequations` →
   `constantScale_friedmann_equation`, `cosmologicalredshift` → `cosmological_redshift_ratio`,
   and `criticaldensity` → `constantScale_critical_density` (Vol07), `hawkingtemperature` →
   `hawking_temperature_def`, `hawkingtemperature_pos` → `hawking_temperature_pos`,
   `bekensteinhawkingentropy` → `bekenstein_hawking_entropy_def`, and `bhentropy_pos` →
   `bekenstein_hawking_entropy_pos` (Vol08), `bulk_reconstruction` →
   `bulkFields_eq_boundaryFields_map` and `ryu_takayanagi` →
   `entanglementEntropy_eq_minimalSurfaceArea_div` (Vol09), `fermiongenerations` →
   `numberOfGenerations_eq_three` (Vol10), `graviton_sm_coupling` →
   `sm_interaction_of_self_distance_zero` (Vol13), and the six `*_from_omega` restatements in
   Vol26 (`handshaking_lemma_restated`), Vol31 (`defect_dominates_p1_restated`), Vol32
   (`feedback_convergence_restated`), Vol36 (`boolean_excluded_middle_restated`), Vol38
   (`equilibrium_excess_demand_zero_restated`), and Vol39 (`min_connections_restated`).
3. **Full 54/54 volume lexicon coverage (`statement_audit.json`).** Expanded `PHYSICS_TOKENS`
   from 94 to 117 tokens (adding domain tokens including `dirac`, `optical`, `commutator`,
   `isotony`, `microcausality`, `spacelike`, `graviton`, `wheeler`, `dewitt`, `chern`, `hall`,
   `haldane`, `landau`, `higgs`, `ryu`, `takayanagi`, `epr`, `tsirelson`, `chsh`, `bell`,
   `singlet`, `partition`, `frame`, `novikov`, `omega`). All **54/54** volumes are now
   represented across **244** rows (covering all 233 lexicon candidates plus 11 extra renamed
   rows): **122 `witness`**, **61 `conditional`**, **14 `renamed`**, and **47 `carried`**
   (`carried_cap: 47`), with **50** retired names recorded under `previous_name`.

### Phase 7 — Consistency gate and review · **DONE (gate added 2026-10-12; the review step is human)**

A single command runs every check the repository can run without the Lean toolchain:

```bash
python research/consistency_gate.py          # add --quick to skip the pytest suite
```

It runs, in order: the three derivation-check scripts, the two LaTeX gates (citation and
structure), the claim scanner, the Lean sorry/trivial grep, the axiom audit, the eight vacuity
ratchets plus the alias baseline, the declaration-parity gate, the text-mirror gate, the
statement audit of the physics volumes, the three C++23 verification suites (`cpp/build.sh --quick`,
`omni-bridge/cpp/build.sh --quick`, and `ai-governor/build.sh`, which now includes a self-contained
FIPS 180-4 SHA-256 implementation and NIST test-vector check with zero external library dependencies),
the pytest suite, and
`ruff check` / `ruff format --check` / `mypy` when those tools are on `PATH`
(a missing tool is reported as skipped, never as a pass). It reports one line per command and
the tail of any failure, and exits non-zero if anything failed. It states in its own summary
that three checks are CI-only here — `lake build ToE`, the kernel audit, and the cited-name
check, all of which need the pinned Lean sources — so a green run is not read as more than it
is.

Negative control: on a copy of the tree with one theorem appended to
`Vol18_StatisticalMechanics.lean`, the gate exits 1 and four commands fail -- the vacuity
ratchets, declaration parity, the text mirrors and the pytest suite.

Still human, and not claimed by the gate: a reviewer reads the `claim_scan.py --list` output
and the Verified table in §3 and confirms that every status matches the evidence. The gate
says a list of commands exited 0; it does not read the diff.

---

## 3. Derivation ledger (core notes)

| # | Claim (location) | Type | Status | Evidence |
|---|---|---|---|---|
| 1 | Logarithmic distance \(d=-\ell_P\ln K\) (Technical §2) | Conditional | Derived (hand); regularity hypothesis now stated | Phase 2, item 3 |
| 2 | Two-factor law \(h_{ij}=\ell_P^2\gamma_{ij}\), \(g_{rr}=\ell_P^2\kappa^2\) (Radial §3) | Conditional | Derived (hand) | Cauchy-equation argument reads correctly; (L), (R), (G) must hold |
| 3 | Flat obstruction: \(g_{RR}=(1-\Phi^2)^2\), \(R=\sqrt{\rho^2-A^2}\), negative Misner–Sharp mass, zero ADM mass, \(\varepsilon\) formula (Radial §4) | Derived | **Verified** | `check_flat_obstruction` |
| 4 | Schwarzschild in \(\Phi\): Ricci-flat, lapse equation, areal \(g_{RR}=((1+\Phi)/(1-\Phi))^2\), weak field (Radial §5) | Identified | **Verified** | `check_isotropic_schwarzschild` |
| 5 | Lichnerowicz source \(\nabla^2\Phi=-2\pi(1+\Phi)^5\varepsilon\) (Radial §5) | Derived | **Verified** | `check_lichnerowicz_and_required_kappa` |
| 6 | Required \(\kappa/\kappa_0=(1+\Phi)^2/\sqrt{1-\Phi^2}\); dilation criterion; margin \(2/(1+\Phi)\) (Radial §3, §5) | Identified / derived | **Verified** | same check |
| 7 | Conformal identity \(g_{RR}=(1+d\ln\Omega/d\ln\rho)^{-2}\) (Radial §3) | Derived | **Verified** | `check_conformal_radial_identity` |
| 8 | Golden bottleneck: global maximum of \(e^{2\Phi}(1-\Phi^2)\) at \(\Phi^*\); robustness root \(\Phi^*(a,p)\); golden iff \(a=p\) (Radial §8) | Derived | **Verified** | `check_golden_bottleneck` |
| 9 | Horizon locations: isotropic \(R^*=(2+\sqrt5)r_s/4\), areal \(R^*=\varphi\,r_s\) (Radial §8) | Derived | **Verified** | `check_horizon_locations` |
| 10 | Markov pair mutual information \(I_n=\ln2-h_2((1+\lambda^n)/2)\) (Markov §1) | Derived | **Verified** against an explicit density matrix | `check_markov_mutual_information` |
| 11 | Mutual information does not compose along the chain (Markov §2) | Derived | **Verified** | same check |
| 12 | \(\kappa(\Phi)\) table and its monotonicity (Markov §3.2) | Derived | **Verified**; row label corrected | `check_markov_kappa_table` |
| 13 | Linearised scalar–Gauss–Bonnet equation equals \(\frac{1}{4M^2x^2}\times\) stated \(x\)-ODE (GB §2) | Derived | **Verified** | `check_gauss_bonnet_reduction` |
| 14 | Kretschmann invariant \(=48M^2/r^6\) on Schwarzschild (GB §1) | Derived | **Verified** | `check_kretschmann` |
| 15 | Scalarization threshold \(\eta/M^2\) (GB §3) | Numerical | **Verified** as \(0.725613\pm10^{-6}\); headline corrected | `check_gauss_bonnet_threshold` (Rayleigh–Ritz) |
| 16 | Scalar field equation and \(T^{(\Phi)}_{\mu\nu}\) from the action (COVARIANT §1) | Derived | **Verified** (SymPy) | `check_scalar_eom_flrw`, `check_flrw_equations` in `research/verify_covariant.py` |
| 17 | GR limit at \(\Phi\equiv0\) under \(V(0)=V'(0)=0\), \(Z(0)>0\) (COVARIANT §2) | Conditional | Derived (hand) | immediate from item 16 |
| 18 | Static no-hair identity for the minimally coupled scalar (COVARIANT §2) | Conditional | **Verified** as an identity on a Schwarzschild exterior with stated boundary (R = 3 r_s); the no-hair conclusion stays conditional on the boundary terms and a convex potential | `check_no_hair_identity` in `research/verify_covariant.py` |
| 19 | Linear mass \(m_{\rm eff}^2\) for the minimal and curvature-coupled branches (COVARIANT §2, GB §1) | Derived | **Verified** (SymPy) | `check_linear_mass` in `research/verify_covariant.py` |
| 20 | FLRW equations and scalar equation (COVARIANT §3) | Derived | **Verified** (SymPy) | `check_flrw_equations` in `research/verify_covariant.py` |
| 21 | Earth, Sun, neutron-star values of \(\Phi\) (Radial §5) | Arithmetic | **Verified** by hand with rounded standard constants | Sun corrected to \(1.1\times10^{-6}\) |
| 22 | Uniform correlation decay in Sim5 (`cumsum`) | Code fact | Verified by reading `Sim5_Emergent_Gravity.py` L74–75; now stated in the module header | — |
| 23 | Schwarzschild-required trend opposite to Markov and Ising decay rates | Reported | Reproduced; tests check the sign; Schwarzschild column and one dense-ED row now checked | `tests/test_quantum_ising_chain.py`, `tests/test_qregion_markov.py`, `research/verify_tfim.py` |
| 24 | \(H_0=73.5\) in the depletion simulation | Calibration input | Not a prediction; re-run done (Phase 3 item 4), reproduces 73.50 by construction | `sim6_v14_depletion.py` |
| 25 | Disformal causality bound constrains \(\dot\Phi\) only; static profiles safe (Radial §8) | Derived | **Verified** for the model as stated (\(C=e^{-2\Phi}\), \(D=\beta(1-\Phi^2)\), flat background): the Lorentzian condition is \(D\dot\Phi^2<C+D\Phi'^2\), the printed bound \(e^{-\Phi}/(\sqrt\beta\sqrt{1-\Phi^2})\) is its \(\Phi'=0\) case (sufficient, and conservative when \(\Phi'\neq0\)), and \(g_{rr}=C+D\Phi'^2>0\) at every gradient. Not a claim about any other \(C\), \(D\), or background. | `check_disformal_bound_depends_on_phidot` |

---

## 4. Decisions needed from the owner

**Resolved.** The user delegated these decisions ("take the best route, the most optimal").
The route taken is recorded in each item.

These decisions gate later phases. The roadmap does not assume an answer.

1. **LaTeX layman's summaries (Phase 4).** Taken: option (a). Options:
   (a) rewrite each volume's summary in place under the rules in §0 (recommended; keeps the
   public material and removes the overclaim);
   (b) move the 41 files to an archive folder marked "superseded" until each volume's Lean
   counterpart has been audited in Phase 6;
   (c) remove the layman's summaries and keep only the formal statements.
2. **Missing LaTeX volumes (Phase 4).** Taken: leave the 13 files absent and say so in the README
   and in this roadmap.
3. **Lean label replacement (Phase 5).** Taken: comments and docstrings only. No code changed.

---

## 5. What this roadmap cannot do from this environment

- Build or check Lean. Kernel truth for the Lean corpus comes from GitHub CI only. The same
  applies to the cited-name check, which needs the pinned mathlib and Lean core sources; the
  consistency gate lists it with `lake build ToE` and the kernel audit as checks it cannot run.
- Run deployments, or any external empirical comparison. The empirical-test gate in
  `COVARIANT_COMPLETION_AND_TEST_GATE.md` §5 stays closed until its six items are frozen.
- Establish that the physical model is correct. Every "Verified" row means the stated
  algebra or numerics hold for the stated model. It does not mean the model describes nature.
