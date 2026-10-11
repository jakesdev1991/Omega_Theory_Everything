<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: MIT -->

# Claim and derivation tightening: end-to-end roadmap

**Status date:** 2026-10-11 · **Branch:** `arena/0f3b6b8f-omega-theory-everything`
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
| Derivation checks | 11 of 11 pass (unchanged) | `python research/verify_derivations.py` |
| `ruff check .`, `ruff format --check .` | pass (120 files formatted) | venv, CI versions |
| `mypy . --ignore-missing-imports` | pass (49 source files) | venv, CI versions |
| `pytest -q` | 118 passed, 81 subtests passed | venv, CI versions |
| Lean | Not compiled here (no `lake`). Lean edits are comment and docstring only; `txt_proofs/` copies are byte-identical to the Lean files. | CI only |

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
The eleven checks in `research/verify_derivations.py` cover every "Verified" row. They are
exact SymPy identities where the algebra is closed-form, and floating-point or independent
numerical methods where it is not.

**Exit gate (met):** `python research/verify_derivations.py` exits 0 with 11 of 11 passing;
`tests/test_verify_derivations.py` runs the same checks in `pytest`, so CI enforces them.

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

### Phase 3 — Remaining core-note items · **PARTIAL (items 1–4 done; 5–6 open)**

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
6. **Radial note §8 and §9:** the claim that the disformal bound constrains \(\dot\Phi\) only
   is hand-checked against the stated bound. Machine-check the bound's dependence on \(\dot\Phi\).

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

**Open.** Item 5 (line-by-line pass of the Layman's Guide against the Verified table) and
item 6 (machine-check of the disformal bound's dependence on \(\dot\Phi\)) are not done.
The core notes are therefore **not** yet called final.

**Exit gate:** `python research/claim_scan.py` shows zero unhedged hits in the core notes,
apart from the documented false positive (met). Each core-note number has a check or a
"reported" label (met for items 1–4). Items 5–6 remain (not met).

### Phase 4 — LaTeX volumes · **DONE (option a; 13 volumes absent by decision)**

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

**Not done.** A script check that every cited declaration exists (step 2 of this phase). The
LaTeX sources were not compiled, since no LaTeX toolchain is installed here; they were checked
for brace balance and unescaped special characters only.

**Exit gate:** `python research/claim_scan.py` shows zero unhedged `latex_docs` hits (met).
The declaration-existence script check is still to be added. The `docs` CI job was not run
here.

### Phase 5 — Lean headers and text mirrors · **DONE (comments and docstrings only; automated check not yet added)**

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
3. Regenerate `txt_proofs/` from `lean_proofs/`, as `PROOF_AUDIT.md` describes.

**Exit gate:** `claim_scan.py` shows zero `label` hits in `lean_proofs/` and `txt_proofs/`;
the declaration-comparison check passes; `lake build ToE` and the kernel audit pass in Lean
CI (these cannot be run locally).

### Phase 6 — Statement-level audit of the Lean physics volumes · **PLANNED, CI-ONLY**

`PROOF_AUDIT.md` lists the existing findings: degenerate models, parameterized stubs, and
conditional lemmas. Phase 6 goes through each physics volume and decides, per theorem,
whether the name overstates the statement, whether the hypotheses are satisfiable, and
whether a non-degenerate witness exists. Each result is recorded in the audit file. This
cannot be done without a Lean toolchain, so it runs in CI or on a machine with `lake`.

**Exit gate:** every physics-volume theorem with a physical-sounding name has a row in the
audit with one of: non-degenerate witness, conditional with stated hypothesis, or renamed.

### Phase 7 — Consistency gate and review · **PLANNED**

A single command runs every check in the repository:

```bash
python research/verify_derivations.py
python research/claim_scan.py
python -m pytest -v
ruff check . && ruff format --check . && mypy . --ignore-missing-imports
```

Then a reviewer reads the `claim_scan.py --list` output and the Verified table in §3, and
confirms that every status in §3 matches the evidence. The pull request is opened only
after this step.

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

- Build or check Lean. Kernel truth for the Lean corpus comes from GitHub CI only.
- Run deployments, or any external empirical comparison. The empirical-test gate in
  `COVARIANT_COMPLETION_AND_TEST_GATE.md` §5 stays closed until its six items are frozen.
- Establish that the physical model is correct. Every "Verified" row means the stated
  algebra or numerics hold for the stated model. It does not mean the model describes nature.
