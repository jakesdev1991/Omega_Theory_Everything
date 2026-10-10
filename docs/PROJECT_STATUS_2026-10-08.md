<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../LICENSE and ../LICENSES/MIT.txt. -->

# Project status inventory — 2026-10-08

**Revision inventoried:** `5ebe7851b215cc90d2713a0baa6b53b2017aab14` (working branch
`arena/e789c885-omega-theory-everything`, identical to `main` HEAD, the merge of PR #49).
**Method:** every suite in the repository was executed locally in this workspace today,
read-only toward GitHub, plus a review of CI history via `gh`. Results below are measured,
not quoted from the repo's own documents.

---

## 1. Bottom line

**The merged tree is green.** Every check that CI runs, plus five suites that CI does *not*
run, pass locally on this revision: 164 Python/Node tests, 25 C++ acceptance groups, the MCP
hub smoke test, a production web build, and the full offline economy scenario suite.

**But "green" is not "honest at the statement level," and the work that fixes that is not
merged.** The adversarial audit of the Lean corpus
([`lean_proofs/ADVERSARIAL_AUDIT.md`](../lean_proofs/ADVERSARIAL_AUDIT.md)) found the kernel
sound but the *honesty layer* porous (audit-evading `_Stmt` indirection, a division-by-zero-
masked theorem, contradictory Planck profiles, vacuous hypotheses). The tenth/eleventh/twelfth
passes that remediate exactly those findings live on **PR #56, which is open and whose Lean
build is red** on one unsolved goal. Nothing from that work is in `main`.

So: **the repository is ready for offline testing today; it is not ready for the "honesty
remediation complete" milestone, and not ready for any on-chain test because nothing has ever
been deployed.**

---

## 2. Evidence — what was executed today

| Component | Command run | Result |
|---|---|---|
| Lean source audits | `python3 -m unittest discover -s lean_proofs -p 'test_*.py'` | **41 passed** |
| Lean axiom policy | `python3 audit_axioms.py --max 0` | **0** axiom/opaque declarations |
| Lean vacuity ratchet | `audit_vacuity.py --max-misleading 0 --max-axiom-names 0 --max-unit-stubs 0 --max-unit-types 0 --max-trivial-proofs 0` | **all metrics 0, OK** |
| Python lint/format/types | `ruff check .`, `ruff format --check .`, `mypy .` | **clean** (91 files formatted, 31 typed) |
| EVM pilot | `npm run check` (compile → size → hardhat test) | **66 contracts compile, 5 size-checked OK, 6/6 tests pass** |
| Solana pilot | `npm test` | **13/13 pass** |
| AMITY scaffold | `npm test` *(not in CI)* | **17/17 pass** |
| Web app | `sync:wallet` → `typecheck` → `npm test` → `npm run build` | **31/31 tests, build OK (54 pages)** |
| Economy console | live `GET /api/economy/scenarios?run=all` | **10 groups / 42 steps, all PASS** |
| Economy readiness | live `GET /api/economy/health` | `ready-with-warnings` (rails unconfigured — expected) |
| Mobile node (NIP-90) | `npm test` | **19/19 pass** |
| Nostr client | `npm test` *(not in CI)* | **12 pass, 1 skipped, 0 fail** |
| Conjecture pilot | `pytest conjecture_pilot` *(not in CI)* | **24/24 pass** |
| Omni-Bridge C++23 | `omni-bridge/cpp/build.sh --quick` *(not in CI)* | **18/18 acceptance groups OK** |
| CBwK shadow pacer | `cpp/build.sh` (strict + TSan + ASan/UBSan): **all pass**, 7 groups |
| MCP hub | `mcp/smoke_test.py` (with `mcp>=1.7,<2`) | **22 tools, stdio roundtrip OK** (see D4) |
| AI governor | `g++ -std=c++23 -Iinclude governor_train.cpp` | **build fails here**: `openssl/sha.h` missing (sandbox has no libssl-dev) — environment, not repo, defect |
| Simulations | `python Sim3/Sim5/Sim6/Sim7` | **run clean**; Sim1/Sim2/Sim4 do not (see D2) |

A live instance of the built site is running at port 3000 in this workspace
(`/testnet`, `/wallet`, `/store` and the API routes all return 200).

### CI history (read-only, via `gh`)

- Revision `5ebe785` (this branch = `main`): **Lean CI green** (run 36305576303, full
  `lake build ToE` + 164-declaration transitive kernel audit), **CI green** on all eight jobs.
- The only failing check on `main` is a **scheduled Security Scan** job (run 37186553206);
  PR #51 fixes it and is still open after 11 days.
- **PR #56** (`arena/ccde3398-…`, 14 commits, "Tenth and eleventh pass"): all non-Lean jobs
  green; the Lean job fails in `leanprover/lean-action@v1`.

---

## 3. Lean 4 proof status (the detailed one)

**Corpus size at this revision:** 69 `.lean` files, 796 `theorem` declarations, 12 `abbrev`,
214 `def`, 68 `structure`, 2 `instance`.

**Kernel evidence.** `main`'s last Lean CI run passed the complete configured target
(`lake build ToE`, which includes all 54 volumes plus `ProofRegression`) and the
`audit_kernel.py` gate over **164 selected declarations**, permitting only `propext`,
`Classical.choice`, `Quot.sound`. That is real kernel verification, performed on GitHub's
runners.

**Honesty gates (all at zero, locally reproducible without Lean):**

| Gate | Value |
|---|---|
| `sorry` / `admit` / `native_decide` / `sorryAx` / `@[extern]` / `implemented_by` / `opaque` / `unsafe` / `partial` | 0 |
| Declaration-level `axiom` commands | 0 |
| Declarations *named* `axiom_*` / `*_axiom` | 0 |
| Degenerate-model tautologies outside `bridge_*` | 0 |
| `Unit` stubs / `Unit`-typed volume defs | 0 / 0 |
| Legacy `*_Stmt` alias exports | 0 |
| Trivial-only proofs | 0 |
| Python regression tests over the corpus | 41 |

**What this does *not* mean** (stated by the repo itself, and I agree):
`kernel-checked ≠ physically derived`, `weak model ≠ intended claim`, and a passing lexical
audit is **not** a semantic non-vacuity certificate. The adversarial audit's own scorecard is
~25 files genuinely substantive and honestly presented, ~20 files with valid proofs over
vacuous or mislabeled statements, and 3 files flagged "must fix or retire."

**P1 items from that audit were remediated** in the eighth/ninth passes (now merged): `law_*`
renames, vacuous-hypothesis retirements, real Vol04 Robertson uncertainty, an exact 2×2
ER=EPR dictionary (`IsProduct ↔ det = 0`), and genuine Schrödinger dynamics.

### The frontier is blocked on one line of Lean

PR #56 carries the tenth/eleventh/twelfth passes — honest bridges, `_Stmt`-indirection
detection, degenerate-model redesigns, an all-zero ratchet (109 files, +7336/−2649). Its Lean
job fails at build step 5 of 8 with, verbatim:

```
error: OmegaAxioms.lean:120:36: unsolved goals
A : Operator
v : StateSpace
⊢ A 1 * v = A 1 • v
error: Lean exited with code 1
Some required targets logged failures: - OmegaAxioms
```

A scalar-vs-·-action mismatch in the redesigned non-degenerate Q-Region model. The kernel
audit step never ran (skipped after the build failure). The last 14 pushes to that branch are
a fix-one-error-see-the-next loop on this same model. **That branch is one compile cycle away
from being evaluable, and it is the single highest-value open item in the repository.**

---

## 4. Defects found during this inventory

| # | Sev | Item |
|---|---|---|
| D1 | **Blocker (frontier)** | PR #56 Lean build fails on `OmegaAxioms.lean:120` (above). Not on this branch — this branch's Lean state is the last *fully verified* one. |
| D2 | High | `Sim1_Emergent_Geometry.py`, `Sim2_Cosmology.py`, `Sim4_Evolution.py` are **prose manuscripts with a `.py` extension** — appended paper text plus a markdown-fenced code dump with mangled identifiers (`psdproject`, `Kpsd`, `D2 = D  2` where `**` was eaten). They are documented as such in `ruff.toml`/`mypy.ini`, but **`README.md` Quick Start still says `python Sim1_Emergent_Geometry.py`**, and CI's smoke step hides the failure with `\|\| true`. `Sim4` additionally imports `jax` (absent from `requirements.txt`) and an `omega` package that is not in the checkout (already noted in `docs/PROVENANCE.md`). |
| D3 | Medium | CI does not run the suites for `amity/`, `nostr-client/`, `omni-bridge/`, `cpp/`, `mcp/`, `conjecture_pilot/`, or `rcod/`. All of the ones I could run pass (17, 12, 18 groups, 7 groups, smoke, 24) — so these are unguarded, not broken. PR #51 claims to close this; it is unmerged. |
| D4 | Medium | `mcp/smoke_test.py` hardcodes `cwd="/tmp/omwga/mcp"`; it fails with `FileNotFoundError` on any machine without that path. Through a symlink, the hub itself works (22 tools, 5 planes, roundtrip OK). The committed `smoke_test_result.json` is therefore not reproducible elsewhere. |
| D5 | Medium | `update_discovery.sh` **overwrites `README.md`** with stale v3.5 "The Asymmetry Discovery" copy that contradicts the current README and asserts "validated by OPAE". Running it once destroys the current README. |
| D6 | Low | README counts have drifted from the tree: claims 58 Lean formalizations (69 `.lean` files), 104 plain-text companions (61), 46 LaTeX docs (42) — the last two are also stated in the README structure block. |
| D7 | Low | `main`'s scheduled CI job has been failing on Security Scan for 4 days (PR #51 open). |
| D8 | Low | `ai-governor/` has no build script and is in no CI job; it needs OpenSSL headers. It cannot be built or verified anywhere today. |
| D9 | Low | No GitHub release and no `wallet-v*` tag has ever been cut, so `.github/workflows/wallet-desktop.yml` (Tauri installers) has **never run**; `desktop/releases.json` is aspirational. |

---

## 5. What cannot be verified in this environment

- **Lean itself.** `lean`/`lake`/`elan` are absent and unobtainable: elan and lean4 release
  assets 302-redirect to `release-assets.githubusercontent.com`, which this sandbox cannot
  reach, and the mathlib cache host is unreachable too. 2 cores / 3 GB RAM would rule out a
  from-source mathlib build regardless. **Kernel truth for the Lean corpus rests entirely on
  GitHub-hosted CI runs** — which is why the #56 red build matters so much.
- **Rust** (`cargo` absent) — `rust/` is verified only by CI, where it is green.
- **Deployment paths.** No Sepolia deployment, no Devnet `tTWC` mint, no AMITY/testnet node,
  no manifests (`evm/deployments/sepolia.json`, `solana/deployments/twc-devnet.json` are
  gitignored and absent). Both unlock rails correctly **fail closed** and report `warn`.

---

## 6. Readiness assessment by pillar

| Pillar | State | Honest read |
|---|---|---|
| Lean corpus (merged) | Kernel-green, honesty holes documented | **Verified but not yet remediated**; remediation blocked on #56 |
| Lean corpus (frontier) | +7336 lines of remediation | **Red build** — not evaluable |
| Web app / economy console | 54 pages build; 31 tests; 42 scenario steps pass live | **Testable today, offline-only** |
| Wallet GUI | Syncs into the site, offline bundle + SHA-256 manifest served | **Testable today**; desktop installers never built |
| EVM pilot | 66 contracts compile, 6 tests, size-checked | **Not deployed** — needs treasury/guardian addresses, test ETH, audit |
| Solana pilot | 13 tests, atomic fixed-supply plan | **Not deployed** — needs keypair, treasury, metadata URI/hash |
| AMITY | 17 tests, fail-closed scaffold | **Not deployed** — longest-lead infra item |
| Mobile node / Nostr | 19 + 12 tests, NIP-90 daemon | **Testable today**; needs a real relay to matter |
| Omni-Bridge / CBwK / MCP | 18 + 7 groups + 22 tools, sanitizer-clean | **Testable today**; unguarded by CI |
| Novel | 17 chapters served, 428 KB plaintext committed | **Open release decision**: seal or accept public plaintext |
| Legal/rights register | `docs/PROVENANCE.md` lists open items | Unresolved: authorship/assignment, trademark (R11), counsel review (R5) |

---

## 7. Next actions, in priority order

1. **Unblock PR #56** — fix `OmegaAxioms.lean:120` (`A 1 * v = A 1 • v`: use `•` consistently
   / make `Operator 1` act via `SMul`), get `lake build ToE` + the kernel audit green, then
   merge. Until then the repository's honesty remediation is invisible to the world.
2. **Merge PR #51** — fixes the failing Security Scan and puts the Python suite and the
   uncovered suites under CI (D3, D7).
3. **Fix the Quick Start lie (D2)** — either rename Sim1/2/4 to `.md` (and fix inbound links)
   or wrap them in triple-quoted docstrings with the real code restored; drop the `|| true`
   from the CI smoke step so this class of failure can never hide again.
4. **Disarm `update_discovery.sh` (D5)** — move it to a dated archive or make it write
   `README_v3.5_archive.md` instead of `README.md`.
5. **Repair the MCP smoke test path (D4)** and regenerate its committed result.
6. **Fix the count drift (D6)** and cut the first `wallet-v*` tag so the desktop workflow
   proves itself.
7. **Then** the deployment steps from [`launch/novel_day_one_plan.md`](../launch/novel_day_one_plan.md) §11:
   Sepolia pilot preflight → valueless `tOMEGA` deploy → drill → independent audit; the
   manuscript sealing decision; the §10 risk register (R2 mainnet venue, R3 IP clearance,
   R5 counsel, R11 trademark).

**Definition of "ready for testing," stated plainly:** the offline half is already testable —
clone, `npm ci`, `npm test`, `npm run build`, open `/testnet`. The on-chain half becomes
testable only after actions 1–3 plus one Sepolia and one Devnet deployment; until then every
rail is *correctly* refusing to pretend.

---

## 8. Remediation status — end of day, 2026-10-08

Everything in §4 and §7 was worked in one session on the branch
`arena/e789c885-omega-theory-everything`, now open as **PR #57** (which contains
PRs #51 and #56 and supersedes both). Every claim below is a command result or a
CI run, not an intention.

| Defect | State | Evidence |
|---|---|---|
| D1 Lean frontier red build | **Fixed.** `lake build ToE` is green in CI. | Three distinct elaboration failures closed in order (`OmegaAxioms.lean:120`, `OmegaUnifiedFoundation.lean:144`/`:150` — the last was an operator-level rewrite that could never match, reduced pointwise). Lean CI run 3781848738 and the full 14-job CI run on the same commit both pass. A machine-checkable lemma-name checker (`lean_proofs/check_mathlib_names.py`, offline, against the pinned Mathlib + core sources) exists for the next round. |
| D2 Sim1/2/4 with a `.py` extension | **Fixed.** Sim1/Sim4 are manuscripts under `docs/manuscripts/` with the damage documented in place; Sim2 is a real script with its prose as the module docstring and runs headless; the CI smoke step has no `\|\| true`; `tests/test_simulations.py` (10 tests) locks all of it in, including "every `python file.py` in the README exists". | CI job `Python Lint & Type Check` green (ruff, ruff format, mypy, 81 pytest tests, five simulation runs). |
| D3–D5, D7 (PR #51 items) | **Fixed.** Merged into this branch; the suites the report listed as uncovered now run in CI. | `CI` run 3781848738: 14/14 jobs. |
| D6 README count drift | **Fixed.** Lean 69 modules, LaTeX 41, companions 59 — each stated with its breakdown. | Recounted against the tree. |
| D8 `ai-governor/` had no build path and no CI | **Fixed.** `ai-governor/build.sh` (strict build, OpenSSL check, overridable `LDLIBS`, governed+baseline smoke run requiring receipts from both arms) plus a CI job. | CI job `AI Governor Build & Smoke Run (PoUW trainer)` green; the run reports the governor's own verdict ("hurt at this scale, prototype scale") rather than a flattering number. |
| D9 no wallet release | **Fixed — and the workflow was broken in six ways, all found and fixed while cutting the first tag.** | See below. |

### D9, in full, because it is the most instructive

`wallet-v0.1.0` was cut, and the workflow it triggered had never run. It failed,
then failed again, for reasons no amount of reading had caught:

1. `npx tauri build --target-dir target` — `--target-dir` is a **v1 flag**; the
   pinned v2 CLI rejects it before compiling anything.
2. `beforeBuildCommand` used a path one directory too deep (Tauri runs build
   hooks with the frontend directory as cwd).
3. Windows cannot build without a real `.ico`; the config listed only PNGs. The
   icon generator now emits one (classic DIB entries, pixel-identical to the
   PNGs) and the wrapper check validates its signature.
4. The publish job had no `permissions:` block and assumed a release existed —
   a tag push creates a tag, not a release.
5. The artifact glob swept up nine bundler scratch files (`template.applescript`,
   `debian-binary`, …) while the actual installers were not uploaded at all.
6. The upload ran every asset through one `gh release upload` invocation; one
   transient failure aborted the rest, silently dropping the `.AppImage`, `.deb`,
   `.msi` and the checksum file.

After fixing all six, the re-cut tag built **all three platforms** and published
exactly: `Omega.Wallet_0.1.0_amd64.AppImage`, `…_amd64.deb`, `…_aarch64.dmg`,
`…_x64_en-US.msi`, `SHA256SUMS.txt` — as a **prerelease**, marked as such in the
release notes, with the assets' digests recorded in `desktop/releases.json`, and
with `/api/wallet/manifest` now reporting `desktop.status: "published"` on the
live site. The installers are unsigned (no Apple/Microsoft credentials exist for
this project) and the download page now says so instead of promising signed
builds.

### Defects found beyond the nine

- `web/.env.example` was missing eleven variables the code reads; a new test
  (`web/src/lib/env-example.test.ts`) now fails if code and documentation drift.
- `evm/scripts/preflight-sepolia.js` never checked that the deployer can pay for
  the deployment — the failure mode that leaves a six-contract deployment half
  done. It now estimates the cost from the compiled artifacts and compares it to
  the live balance, and `npm run preflight:rehearsal` exercises the whole
  preflight against the local network (pinned to Sepolia's chain ID) so its
  passing path is tested in CI.
- CI installed lint/type/test tooling unpinned, so a tool release could flip a
  verdict without a commit; `requirements.txt` now pins them and CI installs the
  same file the README tells readers to install.

### What still cannot be verified from here

Unchanged from §5: no Lean toolchain, no Rust, no deployment paths. The Lean
build is verified only by CI; the desktop installers are verified by CI building
them (nobody has launched one); no Sepolia or Devnet deployment exists, and
`launch/novel_day_one_plan.md` §11 items 1, 3, 4, 5, 6 and 8 remain decisions or
infrastructure for the maintainer — action 7's tooling is now tested, but the
deployment itself is still deliberately not done.
