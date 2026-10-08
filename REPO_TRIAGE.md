<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: LicenseRef-Omega-ReadOnly -->

# Repository triage — what needs doing

**Date:** 2026-09-27 · **Assessed at:** `5ebe785` (branch `arena/01a0e211-omega-theory-everything`,
branched from `main`) · **Scope:** 532 tracked files, 27 top-level directories, 78 MB.

Every number and status below was produced by a command run against this checkout on this date.
Items marked **[unchecked]** could not be verified from here and say so explicitly.

---

## Resolution status

Fixed on this branch (see the commit message and `git diff` for details):

| Item | Status |
|---|---|
| §1 Security Scan fails on schedule | **Fixed** — both URI fixtures tagged, lockfiles excluded, Box hit localized to `evm/package-lock.json:719` (`"license": "MIT",`; a false positive on the package names `hardhat-toolbox`/`boxen`/`cli-boxes`). `workflow_dispatch` added so the full-history scan is re-runnable on demand. |
| §2 `pytest` at root does not run | **Fixed** — `pytest.ini` excludes `mcp/`; `pytest` now exits 0 with 65 passed. |
| §3 CI never runs pytest | **Fixed** — `Python tests` step added to `python-checks`. |
| §4 Suites with no CI job | **Fixed** — new `amity-pilot`, `nostr-client`, `cpp-suites` (full sanitizer matrix) and `mcp-hub` jobs. `ai-governor` still has no build script at all and is **not** covered. Adding `cpp-suites` immediately surfaced a latent flake — see §18. |
| §5 Dependabot coverage | **Fixed** — 4 → 11 ecosystems. `amity/` and `desktop/` given lockfiles so they can `npm ci`. |
| §6 `update_discovery.sh` clobbers README | **Fixed** — deleted, README reference removed. |
| §7 `wallet-desktop.yml` artifact path | **Re-fixed 2026-10-08** — the 2026-09-27 "fix" relied on a flag the Tauri v2 CLI does not have, so that workflow would have failed before building anything. See the correction at the end of §7. |
| §8 `mcp/` machine-specific paths | **Fixed** — `smoke_test.py` derives its path from `__file__`; docs de-hardcoded. Also found and fixed a real bug: `mcp/pyproject.toml` declared an unbounded `mcp>=1.7.0`, and MCP SDK 2.x renamed `FastMCP` → `MCPServer`, so a fresh install broke the hub at import. Now `mcp>=1.7.0,<2`. |
| §9 Dead `lean-ci.yml` branch triggers | **Fixed** — reduced to `main`. |
| §10 Floating action refs | **Partly fixed** — `trivy-action@master` → `@0.36.0`, `trufflehog@main` → `@v3.97.9`. The `actions/*` Node-20 majors (§11) are left to Dependabot PRs #2/#16 to avoid conflicting with them. |
| §11–§17 PR backlog, book sealing, product/legal decisions, pilot freeze, RCOD, omni-bridge scope | **Open** — these need a decision from the repository owner, not a code change. |

---

## 0. Baseline: what is green right now

Ran locally in this sandbox:

| Check | Command | Result |
|---|---|---|
| Python lint | `ruff check .` | `All checks passed!` (ruff 0.16.9) |
| Python format | `ruff format --check .` | `91 files already formatted` |
| Python types | `mypy . --ignore-missing-imports` | `Success: no issues found in 31 source files` |
| Lean source audits | `python3 -m unittest discover -s lean_proofs -p 'test_*.py'` | `Ran 41 tests … OK` |
| conjecture_pilot | `python3 -m pytest conjecture_pilot -q` | `24 passed` |
| Web | `npm run sync:wallet` / `typecheck` / `test` | synced 11 files · tsc clean · `31 pass / 0 fail` |
| EVM | `npm run check` | contract sizes OK · `6 passing` |
| Solana | `npm test` | `13 pass / 0 fail` |
| AMITY | `npm test` | `17 pass / 0 fail` |
| mobile-node | `npm test` | `19 pass / 0 fail` |
| nostr-client | `npm test` | `13 tests, 12 pass, 0 fail` (1 skipped — the live-relay suite) |
| CBwK pacer | `bash cpp/build.sh --quick` | `ALL BUILDS AND RUNS PASSED` |
| Omni-Bridge | `bash omni-bridge/cpp/build.sh --quick` | `ALL BUILDS AND RUNS PASSED` (16 checks, M1 = 951 ns/op) |

GitHub `main` status: latest push CI run `36305576343` **success**, latest Lean CI run
`36305576303` **success**. `git status` clean.

So the code is healthy. The work below is CI plumbing, backlog, and decisions — not broken code.

---

## P0 — actively failing right now

### 1. Security Scan fails on every scheduled run of `main`

Run `36303302037` (Sunday cron, ~1 h before this assessment): job `Security Scan`, step
**"Check for secrets"**, `Process completed with exit code 183`. Job annotations returned by
the API:

```
warning: Found unverified URI result 🐷🔑
warning: Found unverified Box result 🐷🔑
warning: Found unverified URI result 🐷🔑
```

Push and PR runs pass; only `schedule` fails, because Trufflehog derives a commit range from
the event and scans **full history** on schedule.

**Two of the three are localized.** Exactly two embedded-credential URLs exist in the working
tree (find them with a grep for a scheme, then `user:password`, then `@` before the host):

- `amity/test/holder-verification.test.mjs:152` — a credentialed `https://` universe URL,
  username `user`, password `password`, host `universe.example.testnet`, path `/amity`
- `solana/test/config.test.mjs:36` — a credentialed `https://` metadata URI, username `user`,
  password `pass`, host `arweave.net`, path `/twc.json`

Both are negative-test fixtures (they assert the config parser *rejects* credentials in URLs).
They are not real secrets, but Trufflehog's URI detector fires on the shape.

> The literals are deliberately **not** reproduced here. TruffleHog scans commit content *and*
> commit messages, and a quoted credential-shaped URL in a document or a commit message is
> itself a finding — a results-from-a-commit-message hit has no file path, so GitHub's
> annotation layer attributes it to `.github`. Quoting the strings in this file and in the
> commit message is what produced four new findings on the first run of this fix.

**Fix options:** rewrite the fixtures to assemble the URL from parts so no literal credential
URL appears in source; or add a `.trufflehogignore` / allowlist entry; or set
`only_verified: true` on the Trufflehog step (weakest — it also silences real findings).

**The "Box" result is [unchecked].** No JWT-shaped (`a.b.c`) or ≥200-char base64 string exists
anywhere in the tracked tree, and I could not retrieve the job log —
`gh run view --job 108574948144 --log` returned `EOF` on three attempts. Getting that log is
the first step. Note that `Omega_Theory_v4.0_Radial_Metric.md:148` contains the literal string
`\Box(r_s/R)` (the d'Alembertian), which is a plausible keyword trigger, but I could not confirm
the matched fragment.

### 2. `pytest` at the repo root does not run

`README.md` tells contributors to run `pytest -v`. It aborts at collection:

```
ERROR mcp/smoke_test.py
mcp/omega_mcp/__init__.py:27: in <module>
    from mcp.server import FastMCP
E   ModuleNotFoundError: No module named 'mcp.server'
1 error in 0.19s
```

Two causes: the MCP SDK is declared only in `mcp/pyproject.toml` (not `requirements.txt`), and
the top-level directory named `mcp/` shadows the PyPI `mcp` package as a namespace package.
With `--ignore=mcp` the suite is **65 passed, 81 subtests passed** (24 conjecture_pilot + 41
lean_proofs).

**Fix:** add a `[tool.pytest.ini_options]` block with `testpaths`/`norecursedirs = mcp` (or
exclude `smoke_test.py` by name — it is a script, not a pytest module), and keep the MCP hub on
its own `uv`/venv boundary as `mcp/README_MCP_HUB.md` already describes.

### 3. CI never runs the Python test suite

`.github/workflows/ci.yml:29` installs `pytest` and never invokes it — the `python-checks`
job's steps are Ruff lint, Ruff format check, MyPy, and a simulation smoke test. `lean-ci.yml:34`
runs `unittest discover -s lean_proofs`, but that workflow is path-filtered to `lean_proofs/**`.

Net effect: **`conjecture_pilot`'s 24 tests run in no workflow anywhere.** They pass locally, so
this is a one-line addition (`pytest --ignore=mcp`) to `python-checks`.

---

## P1 — CI coverage gaps (green locally, enforced nowhere)

### 4. Test suites with no CI job

| Subproject | Suite | Local result |
|---|---|---|
| `amity/` | 5 files, `npm run check` | 17 pass |
| `nostr-client/` | 3 files, `node --test` | 12 pass, 1 skip |
| `omni-bridge/cpp/` | `build.sh`, 3-config sanitizer build | all pass |
| `cpp/` | `build.sh`, 4-config sanitizer build | all pass |
| `ai-governor/` | `governor_train.cpp` — **no build script at all** | **[unchecked]** |
| `mcp/` | `smoke_test.py` | blocked by P0 §2 |

`ci.yml` has exactly 8 jobs: `python-checks`, `rust-checks`, `evm-contracts`, `solana-pilot`,
`web-app`, `mobile-node`, `security-scan`, `docs`. `grep -rn 'omni-bridge\|cpp/\|amity\|nostr-client\|mcp' .github/workflows/` returns nothing.

Note the sanitizer builds are the expensive part — consider a `--quick` CI job and a nightly
full-sanitizer job.

### 5. Dependabot covers 4 of 9 dependency manifests

`.github/dependabot.yml` declares: `github-actions` `/`, `pip` `/`, `npm` `/evm`, `npm` `/solana`.

Uncovered: `npm` `/web`, `/amity`, `/mobile-node`, `/nostr-client`, `/desktop`; `cargo` `/rust`;
and `mcp/pyproject.toml` (pip). Also `amity/` has **no `package-lock.json`** (`npm ci` there
fails) — it has zero runtime dependencies, so a lockfile is optional, but it breaks the
`npm ci` pattern every other JS workspace uses.

---

## P2 — latent bugs and hazards

### 6. `update_discovery.sh` destroys `README.md`

Line 7 is `cat << 'EOF' > README.md` (heredoc through line 67, then
`echo ">> README Updated. Ready to Push."`). Verified by copying both files to a scratch
directory and running it: the result is a stale **v3.5** README ("The Omega Theory: Emergent
Reality from Quantum Information (v3.5)", "The Asymmetry Discovery") — a **222-line diff** from
the current README. The script is still listed in `README.md`'s Structure block.

**Fix:** delete it, or redirect it at an archive file, and drop the README reference.

### 7. `wallet-desktop.yml` uploads from the wrong path

The build step runs `npx tauri build --target-dir target` with `working-directory: desktop`, so
bundles land at `desktop/target/...`. The upload step uses
`path: desktop/src-tauri/target/release/bundle/**/*` with `if-no-files-found: error`.
`desktop/package.json`'s own `build:*` scripts agree with `desktop/target`, not `desktop/src-tauri/target`.

`git tag -l` is **empty**, so no `wallet-v*` tag exists and this workflow has never executed —
the mismatch is latent, not yet observed. **[unchecked in a real run]**

**Correction (2026-10-08).** The "Fixed — verified against Tauri source" claim above was wrong,
and it was wrong in a way only a real build could expose: `--target-dir` is a Tauri **v1** flag, and
the v2 CLI rejects it outright (`error: unexpected argument '--target-dir' found`), so the build
failed at argument parsing on all three platforms before compiling anything. Verified two ways: the
pinned CLI that `desktop/package-lock.json` installs (`@tauri-apps/cli` 2.12.0) refuses the flag
locally, and `tauri-cli` `src/build.rs` has no such option. Reviewing the same file against the CLI
source found three more hard failures, all now fixed:

- `beforeBuildCommand` said `npm --prefix ../../web ...`, but Tauri runs build hooks with
  `current_dir` set to the frontend directory it resolves from the invocation directory
  (`helpers::run_hook`, called from `build.rs::setup` with `dirs.frontend`), i.e. `desktop/` — where
  the correct relative path is `../web`. From `desktop/`, `../../web` resolves outside the checkout
  and the hook dies immediately.
- Windows cannot build without a real `.ico`: `tauri-build/src/lib.rs` searches `bundle > icon` for
  an entry ending in `.ico`, falls back to `src-tauri/icons/icon.ico`, and hard-errors when neither
  exists. The wrapper listed only PNGs. `web/scripts/sync-wallet.mjs` now also generates
  `icon.ico` (classic DIB entries, 16–256 px) and the config lists it.
- The publish job called `gh release upload` for a release nobody had created, with no
  `permissions:` block — a tag push creates the tag, not the release, and the default token cannot
  upload assets. It now creates a prerelease for the tag when one is missing, and writes a
  `SHA256SUMS.txt` asset.

Whether the bundles really land in `desktop/src-tauri/target/release/bundle` (the crate default, with
no ancestor cargo workspace) is still **[unchecked in a real run]** — that requires the first
`wallet-v*` tag. Note the artifact path in the upload step points there now, and
`desktop/package.json`'s `build:*` scripts were changed to match.

### 8. `mcp/` is pinned to one developer's machine

- `mcp/smoke_test.py:12` → `sys.path.insert(0, "/tmp/omwga/mcp")`
- `mcp/smoke_test.py:58` → `cwd="/tmp/omwga/mcp"`
- `mcp/README_MCP_HUB.md:37,57` and `mcp/DEVELOPMENT.md:39` → `/home/jake/.venvs/omwga-mcp`, `/home/jake/Omega_Theory_Everything/mcp`

Also a toolchain split: `mcp/pyproject.toml` requires `>=3.12`, `ci.yml` pins Python `3.11`.
And `mcp/smoke_test_result.json` / `smoke_test.log` are committed evidence dated
`2026-09-23T18:22:00Z` — worth re-running once the import path is fixed.

### 9. Dead branch triggers in `lean-ci.yml`

`.github/workflows/lean-ci.yml:9` still lists
`arena/01a0dc87-omega-theory-everything, arena/01a0df96-omega-theory-everything` in
`on.push.branches`. Both were merged (PRs #45 and #49). The comment above it says the list
exists to "validate this review snapshot when PR merge conflicts block PR checks" — that
snapshot no longer needs it.

### 10. Floating action refs and runner deprecations

- `ci.yml:189` `aquasecurity/trivy-action@master` and `ci.yml:210` `trufflesecurity/trufflehog@main` are unpinned. Pin to commit SHAs.
- Every job emits: *"Node.js 20 is deprecated … actions/checkout@v4, actions/setup-node@v4 … forced to run on Node.js 24."* Dependabot PRs #2 (`checkout` → v7) and #16 (`setup-node` → v7) address this.
- *"The ubuntu-latest label will migrate to Ubuntu 26 beginning October 19, 2026"* — three weeks out. Worth pinning `ubuntu-24.04` deliberately rather than inheriting the switch.

---

## P3 — backlog and decisions

### 11. 24 open PRs; the 3 substantive ones are all unmergeable

`gh pr list --state open` → 24 open, 21 of them Dependabot. `gh issue list --state open` → **zero** open issues (nothing is being tracked outside PRs).

All three human PRs report `mergeable: CONFLICTING`, `mergeStateStatus: DIRTY`:

| PR | Title | Size | Recommended |
|---|---|---|---|
| **#14** | CI: fix Security Scan on main — permissions + upload-sarif v4 | +8/−1 | **Close.** Its change is already on `main` in stronger form: `ci.yml:176` has the `permissions` block (with `actions: read` added), `ci.yml:202` is `upload-sarif@v4`, plus a fork-PR guard the PR lacks. |
| **#34** | docs: align theory, whitepapers, and sovereign economy vision | 16 files, +1139/−61 | **Rebase or close.** `whitepapers/twc_whitepaper.md` and `docs/sovereign_economy_manifesto.md` already exist on `main` (landed via #43), so much of it may be redundant — diff before rebasing. |
| **#27** | Align simulation physics with Omega Theory v4.0 (Jules) | 69 files, +3472/−2395 | **Decide.** Three days stale against 8 merged PRs; a rewrite of the sim suite this size needs a human read before rebase. |

Dependabot's 21 PRs are mostly major bumps: `hardhat` 2→3 (#19), `@nomicfoundation/hardhat-toolbox` 5→7 (#23), `pandas` 2→3 (#9), `numpy` →2.5 (#11), `dotenv` 16→18 (#38, #40), plus the `actions/*` majors. They should be triaged in a batch, not left to accumulate.

### 12. Launch-blocking decision stated in the README itself

`README.md` calls this out verbatim: *"**Open release decision:** the manuscript the web app
reads, `web/public/book/full-book.md`, is currently committed in plaintext, while the
sealed-staging tooling in `novel/` (`seal.sh`; ciphertext + SHA-256 commitment) is not yet
applied."*

Confirmed on disk: `web/public/book/full-book.md` is **428,834 bytes of plaintext**, and
`novel/` contains only `README.md`, `seal.sh`, and `.gitignore` — **no `manuscript.md.enc`, no
`manuscript.sha256`**. Either seal it or accept the public plaintext and rewrite
`novel/README.md`, which currently asserts "The manuscript is committed **encrypted**" — a
statement that is not true of the repo today.

### 13. Open product/legal decisions in `launch/novel_day_one_plan.md`

- **Mainnet venue still open** (§3.1, flagged "highest-priority remaining decision"): Ethereum L1 vs Base/Arbitrum/OP Mainnet.
- The unlock-threshold parameter is *"currently … parameterized — it must be pinned"* (§3.1).
- Trademark and securities/compliance clearance open (§1.1, §4).
- Nothing is deployed: EVM Sepolia pilot, Solana Devnet pilot, and the AMITY testnet scaffold are all local-only and *"no pilot nor scaffold has been deployed, independently audited, or approved for mainnet/value-bearing use."* No Taproot asset exists.

### 14. `conjecture_pilot` is a draft that has not started

`PILOT_PREREGISTRATION.md` header: *"**Status:** draft. The numeric thresholds in §7 are
proposals. They become frozen when this file is tagged `pilot-prereg-v1`."* `git tag -l` is
empty — **the freeze tag does not exist**, so nothing is pre-registered yet and the two-week
pilot has not run. `DRY_RUN_2026-09-26.md` and `heldout_freeze.json` are the committed dry-run
artifacts. Sign-off → tag → run is the next step.

### 15. `rcod/` carries a published negative result

`rcod/RESULTS.md`: *"the designed claim — RCOD governance improves loss recovery after a
distribution shock — is **not supported** on this workload."* The README still presents RCOD as
a live research prototype. Decide whether to re-scope the claim, keep it as an honest negative,
or drop it from the roadmap.

### 16. Documented-but-unbuilt scope in `omni-bridge/README.md:67` ("Not implemented")

- **T1/T2/T3 enforcement is host-side** — the core selects the tier but does not spawn Wasmtime/container/microVM sandboxes. `scripts/omni_host_audit.py` is the first step, and it needs the Gentoo host.
- **The governance gate's "human" is a process-internal call** in this single-process build; production needs a separate service/account the LLM cannot reach.
- **`framework_complete bridge.json` was never supplied** — sections gated on it stay gated.
- **L0/L1/L2 progressive-disclosure loader** for the Hermes skill store is not built.
- **Fisher-Rao, RCOD, Swarm Dissonance, ToM diagnostics (§71)** are not wired in; audit tiers are not truth oracles.

### 17. Smaller known gaps

- `web/src/lib/amity-server.ts:391` — *"live holder verification is still not implemented. The wallet/web unlock flow remains two rails only: `$OMEGA` and `TWC`."* AMITY is visibility-only at `GET /api/amity/status`.
- `lean_proofs/Vol12_QuantumInformation.lean:9` — *"The separate entropy-bound stub remains legacy."*
- `desktop/releases.json` must be hand-updated after each `wallet-v*` tag (the workflow's own last step prints this as a reminder) — and it has never happened, because no tag exists.
- `docs/PATENT-POSTURE.md`, `docs/TRADEMARKS.md`, `docs/TRADE-SECRETS.md` are posture documents; nothing in them is enforced by code.

---

## Found while fixing the above

### 18. `cpp/test_cbwk_shadow_pacer.cpp` T5 was flaky (fixed)

Adding the `cpp-suites` CI job turned it red intermittently. Reproduced locally:
**2 of 8 runs failed**, always identically:

```
CHECK FAILED test_cbwk_shadow_pacer.cpp:275: reads.load() > 1000
T5 seqlock prices .......... OK  [0 concurrent reads]
```

T5 spins up a reader thread that samples `p.prices()` and flags torn reads, then
races it against a fixed 2000-iteration writer loop and asserts the reader got
more than 1000 reads in. That second assertion is a **wall-clock race**, not a
correctness property: on a loaded 2-vCPU runner the reader thread can be starved
for the entire writer loop and report 0. The real assertion — `!bad`, i.e. no
torn price vectors — was never the problem.

Fixed by yielding to a starved reader until it has made progress, bounded by a
10-second deadline so a pathological scheduler fails loudly instead of hanging.
The writer deliberately stops calling `on_interval()` during that wait: driving
lambda for ten more seconds could push it to infinity and trip the reader's
`isfinite()` check, which would be a new false failure.

**After: 15/15 runs pass**, read counts 2,075–49,679. `omni-bridge`'s suite was
stress-tested alongside it: **6/6 pass**, no equivalent flake.

---

## Suggested order

Items 1–6 below are **done** on this branch; what remains is the owner-decision
track.

1. ~~**Fix Security Scan** (§1)~~ — done. Confirmed green on CI run 36311119579.
2. ~~**Add a `pytest` step + pytest config** (§2, §3)~~ — done. 65 tests enforced.
3. ~~**Delete `update_discovery.sh`** (§6)~~ — done.
4. **Close #14, decide #34 and #27, batch-triage the 21 Dependabot PRs** (§11).
5. ~~**Extend CI to amity / nostr-client / C++ / mcp, and extend Dependabot** (§4, §5)~~ — done. `ai-governor/` still has no build script and no coverage.
6. ~~**Fix `wallet-desktop.yml`'s artifact path**~~ — done, before any `wallet-v*` tag exists.
7. **Make the book decision** (§12) — the only remaining item that is externally visible and irreversible.
8. Then the product/legal track (§13), the pilot freeze (§14), and the research-scope calls (§15–§17).
9. **Bump the `actions/*` majors** (§10) via Dependabot PRs #2/#16 before `ubuntu-latest` becomes Ubuntu 26 on 2026-10-19.
