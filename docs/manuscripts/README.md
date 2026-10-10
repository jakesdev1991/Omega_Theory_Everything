<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../../LICENSE and ../../LICENSES/MIT.txt. -->

# Manuscripts

Prose documents that spent time in the repository root with a `.py` extension,
where the README presented them as runnable simulations and CI's smoke step
papered over the fact that they were not (`... --help || true`).

| Document | Was | Why it is not a script |
|---|---|---|
| [`sim1_emergent_geometry.md`](sim1_emergent_geometry.md) | `Sim1_Emergent_Geometry.py` | The v4.0 paste of this paper passed the text through a Markdown round-trip that consumed underscores and `**`. The appendix listing lost every underscore in its identifiers (`build_mutual_info` → `buildmutualinfo`, `D**2` → `D  2`) and is truncated mid-expression, so the file never parsed. The prose is intact; the listing is marked damaged in place. |
| [`sim4_evolution.md`](sim4_evolution.md) | `Sim4_Evolution.py` | A bundle: prose header, the v2.3.0 Wright-Fisher script and its run transcript, and a complete standalone HTML simulator (Tailwind + Chart.js). The prose sections are not Python, and the code imports `jax` plus an `omega` package that is not part of this repository (recorded in [`../PROVENANCE.md`](../PROVENANCE.md)). |

Two other files in this family were *fixed* rather than moved:

- `Sim2_Cosmology.py` — its prose header is now the module docstring, so the
  script parses and runs (verified headless).
- `sim6_v14_depletion.py` — stayed a script, but its γ-calibration was broken
  (a units error made every run raise inside SciPy). See the regression tests in
  [`../../tests/test_simulations.py`](../../tests/test_simulations.py).

If you have the original Sim1 or Sim4 script sources, drop them in at the
repository root under their old names: the hygiene tests in
`tests/test_simulations.py` require every root `Sim*.py` to parse as Python, so a
restored script is checked automatically, and the README quick start is checked
against the files that actually exist.
