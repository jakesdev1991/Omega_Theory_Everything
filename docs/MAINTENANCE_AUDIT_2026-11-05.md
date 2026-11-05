<!-- Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see ../LICENSE and ../LICENSES/MIT.txt. -->

# Maintenance audit — 2026-11-05

**Assessed at:** `fee4b96c` (`main` HEAD). **Scope:** 703 tracked files, 44 MB, 14 CI
jobs, 7 npm manifests, one Python manifest. **Method:** every claim below was
produced by a command run against a fresh clone of `main` on this date; the
numbers are counts of what the tools reported, not estimates.

This is the successor to [`../REPO_TRIAGE.md`](../REPO_TRIAGE.md) (2026-09-27).
Every engineering item in that document's §1–§10 is now closed; what remains
there is owner decisions. This document records the pass that closed the last
of them indirectly, and what is still open.

---

## 1. What this pass changed

| # | Change | Why |
|---|---|---|
| 1 | CI: workflow-level `permissions: contents: read`, `concurrency` that cancels superseded PR runs but never `main`, and `timeout-minutes` on all 14 jobs | Least privilege by default; four whole CI runs were queued behind each other on 2026-11-05, and a hung step could occupy a runner for the 6 h default |
| 2 | Six floating action refs pinned to commit SHAs (`Swatinem/rust-cache`, `dtolnay/rust-toolchain`, `gaurav-nelson/…markdown-link-check`, `aquasecurity/trivy-action`, `trufflesecurity/trufflehog`, `github/codeql-action/upload-sarif`, `leanprover/lean-action`) | The core `actions/*` refs were already pinned; these were the remaining `@vN` tags, which a maintainer of each action can repoint at any time |
| 3 | `mlc_config.json`: retries, a 20 s timeout, and 403/429 accepted as "alive" | The Documentation Check job was failing on bot-blocked hosts, not on dead links — all 325 relative links resolve (see §3) |
| 4 | `tests/test_markdown_links.py` | The network-dependent action is now backed by a deterministic offline check of the same markdown set |
| 5 | Advisory gates in the web and wallet jobs now test the **production** tree (`npm audit --omit=dev --audit-level=high`) and print the full tree for information | `web/` has 0 production advisories; every open finding is in the build-only `@opennextjs/cloudflare → @opennextjs/aws → @node-minify/core → glob` chain. The wallet job's copy of the same audit over the same lockfile was removed |
| 6 | `web`: `next` 16.4.2 → 16.4.5, `@opennextjs/cloudflare` ^1.20.15 → ^1.20.16, `@noble/hashes` removed | Patch updates; and `@noble/hashes` was added to `web/package.json` but imported nowhere (only `@noble/curves` is) |
| 7 | `mypy.ini`: `check_untyped_defs`, `warn_unused_ignores`, `warn_redundant_casts`, `strict_equality`; five resulting errors fixed | Without `check_untyped_defs` mypy visited only annotated functions — a small fraction of the tree — so the "MyPy type check" step was reporting success while skipping the code it guards. It now checks 93 files |
| 8 | `tests/test_shared_nostr_modules.py` | `mobile-node/lib/{events,bech32}.mjs` and `nostr-client/lib/{events,bech32}.mjs` are byte-identical copies (NIP-01 id recomputation and strict schnorr verification). Duplication is deliberate — each package deploys standalone — but nothing prevented a one-sided fix. It is now a red check |

## 2. Where the four red jobs came from

On the last `main` run (`42048165901`), 10 of 14 jobs were green. The four reds,
with the exact failing step:

| Job | Failing step | Cause |
|---|---|---|
| Documentation Check | Validate markdown links | Network/bot-blocking, not content — see §3 |
| Web App Build & Economy Tests | Check dependency advisories | `npm audit` over the dev tree; production tree is clean |
| Wallet Desktop Wrapper Checks | Check wallet web dependency advisories | The same `npm audit` over the same `web/package-lock.json` |
| EVM Contract Build & Test | Audit the full Hardhat development and runtime dependency tree | Intentional full-tree gate; see §4 |

Rows 1–3 are addressed above. Row 4 is a policy choice, not a defect.

## 3. Links are fine

`git ls-files '*.md'` → 82 files, 325 relative links, **0 missing**. The 44
"missing" links a naive sweep reports come from `web/node_modules` markdown that
is never tracked. The new `tests/test_markdown_links.py` enforces the same thing
offline. No document added in the last month contains an external URL without a
scheme that markdown-link-check can reach, and the two hosts it choked on
(`solana.com`, `v2.tauri.app`) both serve the linked pages to a browser.

## 4. Still open

**P1 — worth doing next**

1. **EVM full-tree audit** (`ci.yml`, "Audit the full Hardhat … tree") can never
   go green: `zod@3.25.76` arrives via Hardhat and has no non-breaking upgrade.
   It needs the thing the repo does not have — an explicit allowlist of accepted
   advisories with a reason and a review date — or it will keep this job red
   forever, which teaches people to ignore a red CI.
2. **`ruff.toml` `target-version = "py38"`** while the project runs 3.11 and
   `pyproject.toml` says `requires-python = ">=3.10"`. Raising it enables 95
   modernization findings (76 auto-fixable: `datetime.UTC`, `TimeoutError`,
   `list`/`tuple`/`X | None` annotations). Mechanical, but a 30-file diff.
3. **Solana kit 8** remains blocked: `@solana-program/system` 0.16 and
   `@solana-program/token` 0.18 both require `@solana/kit@8`, and
   `@metaplex-foundation/mpl-token-metadata-kit` still peers `^6.5.0`.

**P2 — housekeeping**

4. `mcp/smoke_test.log` and `mcp/smoke_test_result.json` are committed evidence
   dated 2026-09-23, produced by a script that hard-coded `/tmp/omwga/mcp`;
   `docs/PROJECT_STATUS_2026-10-08.md` already notes the result is not
   reproducible elsewhere. Either regenerate or drop them.
5. `@solana-program/*` and the web adapter are worth a dependabot `groups` entry
   so the Metaplex-blocked trio is not proposed one package at a time.
6. `REPO_TRIAGE.md` is dated 2026-09-27 and its engineering sections are closed;
   it is retained as the record of that pass, and this document supersedes it.

**Owner decisions — no engineering content**

7. The book is committed in plaintext (`web/public/book/full-book.md`, 428,834
   bytes) while `novel/README.md` describes the manuscript as encrypted; the
   sealing work in `REPO_TRIAGE.md` §11 has no code yet.
8. Mainnet venue, the `pilot-prereg-v1` tag, and the rcod negative result
   (`rcod/SECTION5_RESULTS.md`) are unchanged decisions from the last audit.

---

*No test suite in this document was assumed to pass: `pytest` (218 tests), the
web job's full sequence (`sync:wallet`, `typecheck`, `lint`, 43 tests, 14 wallet
tests, `build`), and the three simulation entry points were executed on this
date against the change set in §1.*
