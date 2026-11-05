# Plan: Make the Lean MCP Stack 100% Local

This is the implementation and operations plan for replacing the cloud-backed AXLE adapter with
an offline, AXLE-compatible Lean 4 MCP server. It starts from the existing
`lean-local-verifier` project, which already provides sandboxed compilation and strict
formal-statement/candidate verification through SafeVerify.

---

## 1. Define “100% local” precisely

The finished system must satisfy all of the following:

1. Runtime proof checking, proof-state extraction, search, transformation, and optional model
   inference work with the internet physically disconnected.
2. No request is sent to AXLE, LeanSearch, LeanFinder, hosted Loogle, Hugging Face, telemetry,
   package indexes, or any other remote service.
3. Every Lean, Mathlib, Python, model, tokenizer, index, and native binary artifact is present on
   local storage and pinned by version/checksum.
4. Untrusted Lean compilation runs in containers with no network, a read-only root filesystem,
   no Linux capabilities, resource limits, and an ephemeral writable directory.
5. The statement being proved is immutable during candidate generation and verification.
6. Final success requires kernel replay and an allowed-axiom check, not merely a successful
   `lake build`.
7. Unknown environment/version names fail closed; the server never downloads a toolchain on
   demand.

There are two legitimate forms of an offline installation:

- **Disconnected runtime:** a connected staging machine downloads/builds everything, then the
  final runtime operates with no external network.
- **Air-gapped machine:** artifacts are built elsewhere, checksummed, transferred by removable
  media, verified, and imported. The target machine is never connected.

Local loopback or Unix-domain-socket communication between Hermes, a model process, and the MCP
server is still local. Proof-worker containers themselves should use `--network none`.

---

## 2. Target architecture

```text
Hermes Agent
  |
  +-- MCP server named "axle" (local compatibility API)
  |     |
  |     +-- environment registry (fixed Lean/Mathlib versions)
  |     +-- isolated compile workers
  |     +-- SafeVerify + lean4checker
  |     +-- LeanInteract or Pantograph state workers
  |     +-- semantic source transformation worker
  |     +-- local declaration/search database
  |     +-- deterministic tactic/search engine
  |     `-- optional local prover-model controller
  |
  `-- lean_lsp MCP (entirely local)
        +-- lake serve / Lean LSP
        +-- local project search
        +-- local Loogle
        `-- remote search tools disabled
```

Keep compilation/verifier workers stateless and disposable. Keep expensive read-only services
(Mathlib index, local model, warm LeanInteract/Pantograph processes) persistent.

The Hermes MCP entry can continue to be named `axle`. Hermes will therefore expose tools as
`mcp_axle_check`, `mcp_axle_verify_proof`, etc., even though no Axiom service is involved.

---

## 3. Freeze the trusted Lean environment

Do this before implementing more tools. A moving Mathlib target makes extraction, transformations,
model prompting, and regression tests unstable.

### 3.1 Record the Volumes project environment

From the Volumes repository:

```bash
cd /absolute/path/to/Volumes
cat lean-toolchain
sha256sum lean-toolchain lake-manifest.json
lake --version
lean --version
lake build
```

Commit these files:

```text
lean-toolchain
lakefile.toml or lakefile.lean
lake-manifest.json
```

Do not use `master`, `nightly`, or an unpinned Git dependency in production.

### 3.2 Create an environment registry

Add a local file such as:

```json
{
  "lean-4.27.0": {
    "lean_toolchain": "leanprover/lean4:v4.27.0",
    "mathlib_revision": "a3a10db0e9d66acbebf76c5e6a135066525ac900",
    "worker_image": "offline-lean-env:4.27.0-a3a10db0",
    "allowed_axioms": ["propext", "Quot.sound", "Classical.choice"]
  }
}
```

The MCP `environment` argument must be looked up in this registry. Never interpolate it directly
into a Docker image name or shell command.

### 3.3 One immutable image per environment

Build one image for each supported Lean/Mathlib pairing. The image contains:

- exact Lean toolchain,
- exact Mathlib checkout and `.olean` cache,
- SafeVerify,
- lean4checker,
- the custom extraction/transformation executable,
- Lean REPL/LeanInteract or Pantograph support,
- no API keys.

Tag the image with both Lean version and a shortened Mathlib commit.

---

## 4. Split the implementation into services

Do not put every feature in one giant Python function.

Recommended source layout:

```text
src/offline_lean/
  server.py                 # MCP schemas and tool registration
  config.py                 # immutable paths, limits, environment registry
  jobs.py                   # job IDs, timeouts, concurrency, cancellation
  workers.py                # networkless container execution
  messages.py               # normalized diagnostics/result schemas
  tools/
    check.py
    verify.py
    proof_states.py
    declarations.py
    transformations.py
    merge.py
    simplify.py
    repair.py
    disprove.py
  search/
    declarations.py
    loogle.py
    vector_index.py
  prover/
    tactics.py
    model_client.py
    self_correction.py
  security/
    source_policy.py
    sandbox.py
    axiom_policy.py
lean/OfflineLean/
  ExtractDecls.lean
  ExtractProofStates.lean
  Transform.lean
  Merge.lean
  Verify.lean
```

Python should orchestrate jobs and return MCP JSON. Lean code should perform operations that need
Lean syntax, elaboration, expressions, environments, constants, or pretty printing.

Never use `shell=True`. Pass every subprocess argument as a separate array element.

---

## 5. Implement an AXLE-compatible local tool surface

Use the same public tool names and broadly compatible input/output shapes. Every response should
include:

```json
{
  "okay": true,
  "content": "processed Lean source",
  "lean_messages": {
    "errors": [],
    "warnings": [],
    "infos": []
  },
  "tool_messages": {
    "errors": [],
    "warnings": [],
    "infos": []
  },
  "failed_declarations": [],
  "timings": {"total_ms": 0},
  "environment": "lean-4.27.0"
}
```

Add endpoint-specific fields without removing the stable common envelope.

### 5.1 `check`

Input:

```json
{"content": "import Mathlib\n...", "environment": "lean-4.27.0"}
```

Implementation:

1. Check source-size and timeout limits.
2. Create a fresh job directory.
3. Write source under a random module name.
4. Run `lake env lean --json` in the selected networkless worker image.
5. Parse every JSON diagnostic line.
6. Distinguish compilation success from proof completeness. A file using `sorry` may compile but
   must be listed under `failed_declarations`.
7. Delete the job directory.

Acceptance tests:

- valid theorem,
- syntax error,
- type mismatch,
- `sorry`,
- warning-only file,
- timeout,
- oversized input.

### 5.2 `verify_proof`

Keep the current SafeVerify design:

1. Compile trusted formal statement and candidate separately.
2. Keep target and candidate module directories isolated.
3. Replay both through fresh environments.
4. Compare names, declaration kinds, universe parameters, types, and required fixed values.
5. Reject `sorryAx`, custom axioms, unsafe and configured partial declarations.
6. Run an additional kernel/olean recheck with lean4checker for the production quality gate.
7. Return per-declaration failure information.

Never let the candidate choose or modify the trusted specification file.

Add a file-based tool such as:

```text
verify_against_spec(spec_name, candidate)
```

where `spec_name` resolves only underneath an immutable specs directory.

### 5.3 `extract_proof_states`

Use LeanInteract first; migrate to Pantograph if tree-search requirements grow.

Required output per hole/error:

```json
{
  "declaration": "Volume1.some_theorem",
  "start": {"line": 10, "column": 4},
  "end": {"line": 10, "column": 9},
  "local_context": ["n : Nat", "h : n > 0"],
  "goal": "n + 1 > 1",
  "pretty_state": "n : Nat\nh : n > 0\n⊢ n + 1 > 1",
  "proof_state_id": 17
}
```

Implementation:

1. Keep a warm Lean process per environment.
2. Send source containing its existing `sorry` holes.
3. Obtain source spans, goals, and proof-state IDs.
4. Return structured context as well as pretty text.
5. Expire states by age and memory limit.
6. Never reuse one user's mutable proof state for another request.

For persistent proof search, support:

```text
open_state
apply_tactic
fork_state
close_state
```

These exceed AXLE's stateless behavior but are valuable for local proof automation.

### 5.4 `extract_decls` and `extract_theorems`

Write a Lean executable that reads the compiled module and iterates only declarations introduced
by that module.

For each declaration emit:

```json
{
  "name": "Volume1.foo",
  "kind": "theorem",
  "type": "∀ n : Nat, ...",
  "value_available": true,
  "axioms": ["propext"],
  "dependencies": ["Nat.add_comm"],
  "source_range": {"start": {}, "end": {}},
  "docstring": "...",
  "module": "Volume1.Chapter1"
}
```

Use `ConstantInfo`, module constant names, environment extension data, and axiom/dependency
collection. Do not infer declaration boundaries with regular expressions.

`extract_theorems` is the same pipeline filtered to theorem/lemma declarations, with enough imports
and dependencies to compile each standalone result.

### 5.5 `theorem2sorry` and `theorem2lemma`

Use Lean parser source ranges:

- `theorem2sorry`: preserve declaration header and replace only the body with `by sorry`.
- `theorem2lemma`: replace the command keyword while preserving attributes, docstrings, namespace,
  declaration name, binders, type, and body.

Reparse and compile the transformed result before returning it.

Do not rewrite comments, strings, or identifiers containing the word `theorem`.

### 5.6 `sorry2lemma`, `have2lemma`, and `have2sorry`

Mirror the `extract_goal` pattern:

1. Elaborate to the selected `sorry`, error, or `have` position.
2. Obtain the goal expression and `LocalContext`.
3. Remove implementation-only and inaccessible local declarations.
4. Revert/generalize the remaining context in dependency order.
5. Generate a fresh collision-free top-level lemma name.
6. Pretty-print the generalized type.
7. Replace the original location with a call to the new lemma.
8. Compile both the extracted lemma and modified source.

A generated lemma must explicitly bind every variable and typeclass instance it uses.

### 5.7 `rename`

Never perform global text replacement.

Use one of:

- Lean LSP references/definitions and exact source edits, or
- elaboration info trees mapping syntax identifiers to resolved `Name` values.

Update only references resolving to the target declaration. Reparse and compile after applying
edits from the end of the file toward the beginning.

Test namespace shadowing, open namespaces, qualified names, notation, comments, strings, and local
variables with the same spelling.

### 5.8 `merge`

1. Parse and elaborate all input fragments.
2. Build a declaration dependency graph.
3. Deduplicate imports and identical declarations.
4. Detect same-name/non-equivalent conflicts.
5. Topologically sort declarations.
6. Preserve required namespace/section context or emit fully qualified declarations.
7. Compile the final source.

Return conflicts instead of silently choosing one declaration.

### 5.9 `normalize`

Use Lean parse/delaboration/pretty-printing APIs. Define explicitly whether comments and exact
formatting are preserved. A first version may normalize declarations while preserving the import
header verbatim.

The key invariant is semantic preservation:

```text
original declaration types/values == normalized declaration types/values
```

Verify this by compiling and comparing declaration metadata.

### 5.10 `simplify_theorems`

Implement verification-guided delta debugging:

1. Split a tactic sequence using syntax nodes, not lines.
2. Remove unreachable tactics after goals close.
3. Attempt removal of one tactic/sub-block at a time.
4. Recompile and require the theorem type to remain identical.
5. Iterate to a fixed point.
6. Optionally remove unused hypotheses only if the requested mode allows statement changes; default
   must preserve theorem statements.
7. Run strict verification on final output.

Keep a maximum attempt count because minimization can become quadratic.

### 5.11 `repair_proofs`

Start with deterministic, auditable passes:

- remove tactics after `no goals to be solved`,
- fix common obsolete tactic/option spellings using a versioned table,
- close exact assumptions,
- try `rfl`, `simp`, `simpa`, `aesop`, `grind`, `omega`, `norm_num`, `ring`, `linarith`,
- use `exact?`, `apply?`, and `rw?` suggestions,
- repair missing imports from an allowlisted local import map,
- retain unresolved goals as explicit output failures, not false success.

Every repair must be compiled and verified. Do not claim success because diagnostics became shorter.

### 5.12 `disprove`

Define a conservative contract. Success means one of:

- a kernel-checked proof of the theorem's negation, or
- a concrete counterexample whose falsification is kernel checked.

Use bounded finite-domain enumeration, `decide`, `norm_num`, `omega`, SMT/ATP integrations, and a
local model only as proposal mechanisms. Return `unknown` rather than `false` when search is
inconclusive.

---

## 6. Add an entirely local search stack

Use search layers in this order:

1. local project text/declaration search (`ripgrep` + parsed declaration DB),
2. Lean tactics `exact?`, `apply?`, `rw?`, `simp?`,
3. local Loogle,
4. optional local embedding index,
5. optional local premise selector/hammer.

### 6.1 Configure local lean-lsp-mcp

Install it into a fixed virtual environment during staging. At runtime invoke the executable
itself, not `uvx`.

Recommended environment:

```yaml
LEAN_PROJECT_PATH: "/absolute/path/to/Volumes"
LEAN_LOG_LEVEL: "NONE"
LEAN_LOOGLE_LOCAL: "true"
LEAN_MCP_DISABLED_TOOLS: "lean_leansearch,lean_leanfinder,lean_state_search,lean_hammer_premise"
```

Remote search tools must be disabled, not merely avoided by prompt instructions.

Use WSL/Linux for local Loogle if native Windows support is unavailable. Budget substantial RAM
for the Mathlib index. On low-memory machines, use exact/apply/rw suggestions and the declaration
SQLite database first.

### 6.2 Build a local declaration database

Create SQLite tables for:

```text
declarations(name, module, kind, type, docstring, source_file, line, axioms)
dependencies(source_name, target_name)
constants(declaration_name, constant_name)
embeddings(declaration_name, vector_id)
```

Rebuild the index whenever the Lean toolchain, Mathlib revision, or project source hash changes.
Store those hashes in index metadata and refuse stale results.

### 6.3 Optional local semantic search

Download an embedding model during staging, pin its exact revision, and build a local FAISS/Qdrant
index over names, types, and docstrings. Set all model libraries to offline mode at runtime.

Semantic results are suggestions only. Hermes must confirm names and types through Lean before use.

---

## 7. Add optional local proof generation

Verification and AXLE-compatible transformations do not automatically prove all of Volumes 1–54.
For automated proof generation, add a local Lean-specialized model after deterministic tooling is
stable.

Practical choices include a small Goedel-Prover-V2 or DeepSeek-Prover-V2 model. Pin the model
revision and tokenizer. Account for model/toolchain mismatch: many published Lean models target
older Lean/Mathlib snapshots, so local retrieval and compiler-guided correction are mandatory.

Approximate practical GPU tiers for quantized models:

```text
7B/8B:  12–16 GB VRAM preferred
14B:    16–24 GB VRAM
32B:    32–48 GB VRAM
```

CPU-only inference is possible but high-sample proof search will be slow.

### 7.1 Model loop

```text
immutable theorem statement
  -> exact goal extraction
  -> retrieve local premises
  -> generate N candidate proofs
  -> compile candidates in parallel networkless workers
  -> rank by verified progress/diagnostics
  -> feed exact Lean errors and remaining goals back to model
  -> repeat within fixed budget
  -> SafeVerify + lean4checker final gate
```

Never accept natural-language confidence. Only a strictly verified candidate succeeds.

### 7.2 Keep model inference local

Run the model through vLLM, llama.cpp, Ollama, or Transformers bound only to `127.0.0.1`, or use a
Unix-domain socket. Apply an OS firewall rule denying external egress from the model/controller
user.

Set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

These variables prevent accidental library fetches but do not replace firewall/network isolation.

---

## 8. Build the connected staging bundle

This section is executed on a connected build machine.

### 8.1 Lock Python dependencies

From this repository:

```bash
uv lock
uv sync --frozen
uv build
uv export --frozen --format requirements-txt -o requirements-offline.txt
python -m pip download -r requirements-offline.txt -d wheelhouse
cp dist/*.whl wheelhouse/
```

Test installation without indexes:

```bash
python -m venv /tmp/offline-venv-test
/tmp/offline-venv-test/bin/pip install \
  --no-index --find-links wheelhouse lean-local-verifier
```

Do the same for lean-lsp-mcp, LeanInteract/Pantograph Python packages, local model runtime, and every
other Python component.

### 8.2 Build and export images

```bash
docker build -t offline-lean-env:4.27.0-a3a10db0 .
docker tag offline-lean-env:4.27.0-a3a10db0 local-lean-verifier:4.27.0

./scripts/test.sh

docker save \
  offline-lean-env:4.27.0-a3a10db0 \
  local-lean-verifier:4.27.0 \
  -o offline-lean-images.tar
```

### 8.3 Download model artifacts

Use the model vendor's official download command with an exact revision/commit. Store:

```text
models/<model>/config.json
models/<model>/tokenizer files
models/<model>/weight shards
models/<model>/generation config
models/<model>/LICENSE
models/<model>/REVISION
```

Never rely on a mutable `main` revision.

### 8.4 Package source and indexes

Include:

```text
lean-local-verifier source archive
offline Lean/Mathlib source bundle
wheelhouse/
offline-lean-images.tar
models/
indexes/
environments.json
licenses/
SBOM/checksum manifest
```

Create checksums:

```bash
find offline-bundle -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > offline-bundle/SHA256SUMS

tar -czf offline-lean-bundle.tar.gz offline-bundle
sha256sum offline-lean-bundle.tar.gz > offline-lean-bundle.tar.gz.sha256
```

---

## 9. Install on the offline machine

Verify before installing:

```bash
sha256sum -c offline-lean-bundle.tar.gz.sha256
tar -xzf offline-lean-bundle.tar.gz
cd offline-bundle
sha256sum -c SHA256SUMS
```

Load images:

```bash
docker load -i offline-lean-images.tar
docker image inspect offline-lean-env:4.27.0-a3a10db0 >/dev/null
```

Create fixed virtual environments without an index:

```bash
python3 -m venv /opt/offline-lean/venvs/mcp
/opt/offline-lean/venvs/mcp/bin/pip install \
  --no-index --find-links wheelhouse lean-local-verifier

python3 -m venv /opt/offline-lean/venvs/lsp
/opt/offline-lean/venvs/lsp/bin/pip install \
  --no-index --find-links wheelhouse lean-lsp-mcp
```

Use direct executable paths at runtime. Do not run `uvx`, `pip install`, `lake update`,
`elan toolchain install`, or model download commands on the offline machine.

Set the source/index/model directories read-only for the service account. Only job, log, and cache
directories should be writable.

---

## 10. Cut Hermes over from cloud AXLE to local AXLE-compatible MCP

Back up the active profile:

```bash
hermes config path
hermes config env-path
hermes backup --quick --label pre-offline-lean
```

Remove the cloud AXLE command (`uvx --from axiom-axle-mcp ...`) and replace the `axle` entry with
the local executable. A template is in `hermes-offline-config.example.yaml`.

Core entry:

```yaml
mcp_servers:
  axle:
    command: "/opt/offline-lean/venvs/mcp/bin/lean-verifier-mcp"
    args: []
    env:
      LEAN_VERIFIER_IMAGE: "offline-lean-env:4.27.0-a3a10db0"
      LEAN_VERIFIER_SPECS_DIR: "/absolute/read-only/path/to/specs"
      HF_HUB_OFFLINE: "1"
      TRANSFORMERS_OFFLINE: "1"
      HF_DATASETS_OFFLINE: "1"
    enabled: true
    timeout: 320
    connect_timeout: 60
    trust: full
    supports_parallel_tool_calls: false
```

Because the Hermes server key remains `axle`, tools remain under `mcp_axle_*`.

Add the local LSP entry from the template and point it at Volumes. Ensure remote search tools are
explicitly disabled.

The cloud key is no longer required. Remove `AXLE_API_KEY` from the active profile `.env` if no
other process needs it.

Test:

```bash
hermes mcp test axle
hermes mcp test lean_lsp
hermes mcp list
```

Then disconnect networking and run the same tests again.

---

## 11. Enforce runtime isolation

Every untrusted Lean job should use controls equivalent to:

```bash
docker run --rm -i \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=768m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --memory 2g --memory-swap 2g \
  --cpus 2 \
  --pids-limit 256 \
  --user 10001:10001 \
  offline-lean-env:4.27.0-a3a10db0
```

Additional production controls:

- default Docker seccomp profile or a tighter tested profile,
- AppArmor/SELinux policy,
- no Docker socket mounted into proof workers,
- no host project mounted writable,
- maximum request/output size,
- process-group termination on timeout/cancel,
- concurrency semaphore,
- disk quota and stale-job cleanup,
- secret-free worker environment,
- structured audit log containing hashes, not full private source unless explicitly enabled.

Trusted project macros may require a less restrictive source policy than hostile submissions. Use two
modes:

- `strict_untrusted`: rejects custom elaborators, macros, FFI, initializers, unsafe/partial, etc.
- `trusted_project`: allows project metaprogramming inside the networkless sandbox but retains kernel
  replay and axiom checks.

Do not weaken the final verification policy merely to make generated code compile.

---

## 12. Build a parity and adversarial test suite

Before permanently removing AXLE access, create golden tests for every public endpoint. Compare
semantic outcomes, not only formatting.

Required suites:

```text
tests/parity/check/
tests/parity/verify_proof/
tests/parity/proof_states/
tests/parity/extract_decls/
tests/parity/extract_theorems/
tests/parity/theorem2sorry/
tests/parity/theorem2lemma/
tests/parity/have2lemma/
tests/parity/sorry2lemma/
tests/parity/rename/
tests/parity/merge/
tests/parity/normalize/
tests/parity/simplify/
tests/parity/repair/
tests/parity/disprove/
tests/adversarial/
```

Adversarial cases:

- changed theorem type,
- theorem replaced by definition or axiom,
- explicit and imported `sorryAx`,
- custom axiom hidden behind helper declarations,
- unsafe and partial declarations,
- `unsafeCast`, environment mutation, malicious elaborator,
- `prelude` and core-name redefinition,
- omitted imports and type redefinition,
- source path traversal,
- oversized output, infinite elaboration, memory exhaustion,
- namespace shadowing in rename,
- duplicate/conflicting declarations in merge,
- model candidate that edits the statement.

Acceptance for a tool release:

1. all valid fixtures pass,
2. all invalid/adversarial fixtures fail for the expected reason,
3. repeated runs are deterministic except documented timing/job IDs,
4. no external packets are observed during the full suite,
5. no worker survives cancellation or timeout,
6. version/environment hashes appear in every result.

Use a physically disconnected test and an OS network monitor/firewall audit. Environment variables
alone do not prove offline behavior.

---

## 13. Volumes 1–54 operating procedure

For every placeholder theorem:

1. Create a branch and immutable spec file.
2. If the declaration is `: True := True.intro`, reconstruct the actual proposition from the volume
   text first; the statement itself is currently a stub.
3. Have a human review the formal statement independently of its proof.
4. Hash/freeze the spec.
5. Extract exact local proof states.
6. Search local project declarations, exact/apply/rw suggestions, and local Loogle.
7. Run deterministic tactics before using the local model.
8. Generate multiple candidates without allowing statement edits.
9. Compile and iterate using exact Lean errors.
10. Require local `verify_proof` and lean4checker success.
11. Reject any custom axiom, `sorry`, unsafe, or partial declaration.
12. Commit one logically reviewable theorem or small group at a time.
13. Record environment hash, verifier version, and result in the commit/CI artifact.

Correct verification proves that Lean accepts the formal proposition. It does not prove that the
formal proposition faithfully represents the volume's prose; that translation requires review.

---

## 14. Implementation milestones and gates

### Milestone A — offline verification (already mostly complete)

Tools:

```text
verifier_health
list_verification_specs
verify_against_spec
verify_proof
check
```

Gate: works with network disconnected; SafeVerify and lean4checker pass/fail corpus.

### Milestone B — analysis parity

Tools:

```text
extract_proof_states
extract_decls
extract_theorems
read_docs (local documentation)
list_environments
```

Gate: source spans and declaration metadata match hand-reviewed fixtures.

### Milestone C — safe transformations

Tools:

```text
theorem2sorry
theorem2lemma
have2sorry
sorry2lemma
have2lemma
rename
normalize
merge
```

Gate: every transformed result compiles and preserves required declaration semantics.

### Milestone D — deterministic optimization/repair

Tools:

```text
simplify_theorems
repair_proofs
disprove
```

Gate: no false-success result in adversarial corpus; fixed effort/time budgets.

### Milestone E — local proof automation

Components:

```text
local declaration retrieval
local Loogle
local model
candidate sampling
compiler-guided self-correction
bounded proof-state search
```

Gate: benchmark proof success, median latency, GPU/RAM usage, and zero-network audit are recorded.

### Milestone F — air-gap release

Artifacts:

```text
signed/checksummed bundle
image tar files
wheelhouse
model files
indexes
environment registry
licenses/SBOM
restore and upgrade runbooks
```

Gate: clean offline machine installs and passes the entire suite without package resolution.

---

## 15. Suggested immediate next work

Implement this exact sequence in the existing repository:

1. Add `check` to the current Docker worker and MCP server.
2. Add a LeanInteract-backed `extract_proof_states` worker.
3. Add a Lean executable for `extract_decls` and `extract_theorems`.
4. Add `list_environments` backed by a fixed registry.
5. Add parser-based `theorem2sorry` and `theorem2lemma`.
6. Add lean4checker to the final verification gate.
7. Produce `uv.lock`, wheelhouse, and exported Docker image.
8. Replace Hermes's cloud `axle` entry with the direct local venv executable.
9. Disconnect networking and run all regression tests.
10. Only then start lemma extraction, repair, and local-model work.

This order delivers useful offline parity early while keeping correctness and isolation ahead of
proof-generation complexity.
