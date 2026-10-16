# Provenance and rights register

> This is a working inventory, not proof of authorship, ownership, originality, or patent priority. It deliberately records uncertainties that must be checked before licensing or commercialization.

## Current checkout facts

- Repository: `jakesdev1991/Omega_Theory_Everything`.
- Review checkout HEAD: `e362afe8e138db5e936f14ca501f85a5c58f0fc6` (`Merge pull request #78 from jakesdev1991/arena/0fe1264e-omega-theory-everything`), dated 2026-10-11. This is the source revision reviewed before the working-tree changes in this session.
- The checkout reports itself as shallow. Its local object graph contains only the two grafted commits `e362afe` and `a764f34`; it does not contain the complete earlier graph needed to establish first authorship, contribution dates, or first-public-disclosure dates for every file.
- The earlier public commit `c717dc53bd20db4a6290734f09a3db5bb93fa287` (`Merge tri-token, app, and governance work`) is not present in the local Git object database. A GitHub API lookup confirmed its SHA/date (2026-09-23 14:45:41 UTC) and its tree includes `LICENSE`, `app/`, `rust/`, `whitepapers/`, and `tri_token_sovereign_economy_blueprint.md`. The fetched `LICENSE` text matches `LICENSES/MIT-legacy.txt` apart from a trailing-newline difference. This verifies the archived license text against that public commit, but does not reconstruct the missing history or prove that every historical contributor had authority to license every file.
- The historical root `LICENSE` stated MIT License and `Copyright (c) 2025-2026 Jacob See`. A copy of that text is kept as `LICENSES/MIT-legacy.txt`.
- No signed contributor license agreement or assignment is recorded among the files in this checkout.
- Direct dependency declarations include Python packages in `requirements.txt`, Lean/Mathlib tooling in `lean_proofs/`, Rust workspace crates, and per-project npm lockfiles. Observed external references also include Google Fonts in `app/styles.css`, Tailwind CSS and Chart.js CDN links in `Sim4_Evolution.py`, and GitHub Actions in `.github/workflows/ci.yml`; the `Sim4` manuscript (now `docs/manuscripts/sim4_evolution.md`) also imports an `omega` package not present in this checkout, and its embedded HTML simulator loads Tailwind CSS and Chart.js from CDNs. This remains an incomplete software bill of materials and third-party content audit.

## Current file-scope map

The intended current licensing split is recorded in [`../LICENSE`](../LICENSE) and [`LICENSING.md`](LICENSING.md):

- Scientific materials: root simulations and theory files, `lean_proofs/`, `latex_docs/`, `txt_proofs/`, and `rcod/` — MIT.
- Public novel text: `web/public/book/full-book.md` and its notice — CC BY 4.0, applied by Jacob See on 2026-10-10 to the extent he owns or is authorized to license the relevant rights. The full draft is already public in Git and the web app's static asset directory. This grant does not clear third-party material or establish ownership beyond the rights the licensor controls.
- Product software and staging tools: `web/**` except the CC BY book files and all-rights-reserved store legal texts; `mobile-node/`, `app/`, `desktop/`, `omni-bridge/`, `cpp/`, `mcp/`, `launch/`, `novel/`, and `docs/store/` — PolyForm Noncommercial 1.0.0.
- Crypto and C.A.R.E. materials: `rust/`, `evm/`, `solana/`, `amity/`, `whitepapers/`, the Tri-Token blueprint, and specified C.A.R.E. docs — read-only / `LicenseRef-Omega-ReadOnly`.
- Other project-administration and legal notices — no separate grant to product assets.

The scope map is an intent; it does not establish that the rights holder owns every item or cure a third-party licensing issue.

## Audit status and open items

The following checks remain outstanding unless separately documented with evidence:

- Recover and review the full GitHub commit/PR/release history and identify the first public date for each protected work or important technical disclosure.
- Review all contributors, employment/contractor obligations, joint authorship, assignments, and authority to relicense. Confirm the rights holder's legal identity and ownership for each asset.
- Identify copied/adapted text, figures, diagrams, datasets, code, generated content, fonts, icons, and other third-party material. Record source, author, license, required attribution, and whether redistribution is permitted.
- Produce a complete dependency inventory/SBOM and review dependency-license notices across Python, Lean/Mathlib, Rust, and browser/runtime packages. The 2026-10-16 security review records clean npm audits (including the migrated EVM toolchain) and Python audits, a direct Tauri advisory remediation with Cargo transitive audits still pending, and one manually triaged Lean `import-graph` advisory exception. These checks are not a complete SBOM or license review; see [`SECURITY-REVIEW-2026-10-16.md`](SECURITY-REVIEW-2026-10-16.md).
- Confirm that all science files marked MIT are eligible for that grant; obtain written consent for any contribution not owned by the rights holder.
- Confirm product materials intended for royalty-bearing licensing are not subject to employer, contractor, co-author, or prior-license restrictions.

## Evidence to retain

Maintain dated copies/links of signed assignments and contributor agreements, source drafts and notebooks, citations, dependency lockfiles/SBOMs, third-party permissions, GitHub export/history, release archives, and public disclosure records. Restrict access to confidential evidence and personal data; do not place secrets in this public repository.

When an item is verified, record the file/path, rightsholder(s), source/creation date, license, evidence location, reviewer, and review date here or in a controlled companion register. Do not convert an unknown into an assertion of sole ownership by silence.
