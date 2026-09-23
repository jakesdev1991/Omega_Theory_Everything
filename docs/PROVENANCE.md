# Provenance and rights register

> This is a working inventory, not proof of authorship, ownership, originality, or patent priority. It deliberately records uncertainties that must be checked before licensing or commercialization.

## Current checkout facts

- Repository: `jakesdev1991/Omega_Theory_Everything`.
- Baseline commit in this checkout: `c717dc53bd20db4a6290734f09a3db5bb93fa287` (`Merge tri-token, app, and governance work`), dated 2026-09-23 14:45:41 UTC.
- The checkout reports itself as a shallow repository and exposes only that grafted baseline commit. It does not contain the complete earlier commit graph needed to establish first authorship, contribution dates, or first public-disclosure dates for every file.
- The baseline root `LICENSE` stated MIT License and `Copyright (c) 2025-2026 Jacob See`. A copy of that text is kept as `LICENSES/MIT-legacy.txt`.
- The baseline commit includes scientific materials and product materials (`app/`, `rust/`, `whitepapers/`, and the Tri-Token blueprint) under that root MIT notice. This record does not determine whether every historical contributor had authority to license every file.
- No signed contributor license agreement or assignment is recorded among the files in this checkout.
- Direct dependency declarations include Python packages in `requirements.txt`, Lean/Mathlib tooling in `lean_proofs/`, and Rust workspace crates. Observed external references also include Google Fonts in `app/styles.css`, Tailwind CSS and Chart.js CDN links in `Sim4_Evolution.py`, and GitHub Actions in `.github/workflows/ci.yml`; the `Sim4` file also imports an `omega` package not present in this checkout. This is not a complete software bill of materials or third-party content audit.

## Current file-scope map

The intended current licensing split is recorded in [`../LICENSE`](../LICENSE) and [`LICENSING.md`](LICENSING.md):

- Scientific materials: root simulations and theory files, `lean_proofs/`, `latex_docs/`, `txt_proofs/`, and listed tooling — Apache-2.0.
- Product materials: `app/`, `rust/`, `whitepapers/`, and `tri_token_sovereign_economy_blueprint.md` — all rights reserved / `LicenseRef-Omega-Product-Proprietary`.
- Other project-administration and legal notices — no separate grant to product assets.

The scope map is an intent; it does not establish that the rights holder owns every item or cure a third-party licensing issue.

## Audit status and open items

The following checks remain outstanding unless separately documented with evidence:

- Recover and review the full GitHub commit/PR/release history and identify the first public date for each protected work or important technical disclosure.
- Review all contributors, employment/contractor obligations, joint authorship, assignments, and authority to relicense. Confirm the rights holder's legal identity and ownership for each asset.
- Identify copied/adapted text, figures, diagrams, datasets, code, generated content, fonts, icons, and other third-party material. Record source, author, license, required attribution, and whether redistribution is permitted.
- Produce a dependency inventory/SBOM and review license notices for Python, Lean/Mathlib, Rust, and any future browser/runtime dependencies.
- Confirm that all science files marked Apache-2.0 are eligible for that grant, including its patent terms; obtain written consent for any contribution not owned by the rights holder.
- Confirm product materials intended for royalty-bearing licensing are not subject to employer, contractor, co-author, or prior-license restrictions.

## Evidence to retain

Maintain dated copies/links of signed assignments and contributor agreements, source drafts and notebooks, citations, dependency lockfiles/SBOMs, third-party permissions, GitHub export/history, release archives, and public disclosure records. Restrict access to confidential evidence and personal data; do not place secrets in this public repository.

When an item is verified, record the file/path, rightsholder(s), source/creation date, license, evidence location, reviewer, and review date here or in a controlled companion register. Do not convert an unknown into an assertion of sole ownership by silence.
