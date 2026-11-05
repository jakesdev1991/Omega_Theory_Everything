# Local Lean 4 Verifier

A complete local Lean 4 verification service with:

- a command-line interface (`lean-verify`),
- a stdio MCP server (`lean-verifier-mcp`),
- pinned Lean 4.27.0 and Mathlib,
- declaration/signature comparison,
- proof-term replay through Lean's kernel,
- rejection of `sorryAx`, custom axioms, `unsafe`, and `partial`, and
- per-request Docker isolation with no network.

The verifier uses the vendored [SafeVerify](https://github.com/GasStationManager/SafeVerify)
backend. Its Apache-2.0 license is retained at
`vendor/SafeVerify/LICENSE.txt`; see `NOTICE`.

## What is being verified?

The verifier receives two separate Lean files:

1. **Formal statement/specification** — trusted declarations whose bodies may contain
   `sorry`.
2. **Candidate** — untrusted code that must implement declarations with the same names,
   declaration kinds, universe parameters, and types.

Example specification:

```lean
import Mathlib

theorem local_add_comm (a b : Nat) : a + b = b + a := by
  sorry
```

Accepted candidate:

```lean
import Mathlib

theorem local_add_comm (a b : Nat) : a + b = b + a := by
  exact Nat.add_comm a b
```

A candidate that changes the theorem to `a + b = a + b` compiles by itself but is
rejected because its type differs from the trusted specification.

## Security and checking pipeline

For every request the host starts a fresh container with:

- `--network none`
- a read-only root filesystem
- a temporary `/tmp` filesystem
- all Linux capabilities removed
- `no-new-privileges`
- CPU, memory, PID, source-size, output-size, and time limits

Inside the container:

1. A source-policy pass rejects direct metaprogramming and dangerous commands such as
   `axiom`, `unsafe`, `partial`, `unsafeCast`, `extern`, `implemented_by`, `run_tac`,
   custom elaborators/macros/syntax, initializers, and `#eval`.
2. The formal statement is compiled to a fresh `.olean` file.
3. The candidate is compiled separately to another fresh `.olean` file.
4. SafeVerify replays both modules into fresh Lean environments.
5. Required declaration names, kinds, types, universe parameters, and applicable values
   are compared.
6. Candidate proof terms are checked through the Lean kernel.
7. Candidate declarations may depend only on Lean's standard allowed axioms:
   `propext`, `Quot.sound`, and `Classical.choice`.
8. `sorryAx`, custom axioms, unsafe declarations, and partial definitions fail.

The runtime container has no network access. The initial Docker image build needs network
access to download Lean and the pinned Mathlib dependencies.

## Prerequisites

Install and start:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine
- [uv](https://docs.astral.sh/uv/)

You do **not** need a separate host installation of Lean; Lean is installed inside the
Docker image.

The first image build downloads Mathlib and may take 10–30 minutes and several gigabytes,
depending on your machine and network.

## One-command setup

### macOS, Linux, or WSL

```bash
cd /path/to/lean-local-verifier
chmod +x scripts/*.sh
./scripts/setup.sh
```

### Windows PowerShell

Run Docker Desktop, then:

```powershell
cd C:\path\to\lean-local-verifier
.\scripts\setup.ps1
```

Equivalent manual commands:

```bash
docker build -t local-lean-verifier:4.27.0 .
uv sync
uv run lean-verify health
```

A successful health response has `"okay": true` and reports Lean 4.27.0.

## Command-line use

Verify the included valid candidate:

```bash
uv run lean-verify verify \
  --spec examples/specs/add_comm.lean \
  --candidate examples/candidates/add_comm_good.lean
```

This should return `"okay": true` and exit with status 0.

Try the rejection cases:

```bash
# Changes the requested theorem statement
uv run lean-verify verify \
  --spec examples/specs/add_comm.lean \
  --candidate examples/candidates/add_comm_bad_statement.lean

# Uses sorry
uv run lean-verify verify \
  --spec examples/specs/add_comm.lean \
  --candidate examples/candidates/add_comm_bad_sorry.lean

# Declares a custom axiom
uv run lean-verify verify \
  --spec examples/specs/add_comm.lean \
  --candidate examples/candidates/add_comm_bad_axiom.lean
```

Each rejection exits with status 1 and returns structured JSON explaining whether failure
occurred during policy checking, compilation, or SafeVerify.

Run all tests:

```bash
./scripts/test.sh
```

## Trusted specifications

Put immutable challenge files under `specs/`, for example:

```text
specs/
  algebra/add_comm.lean
  number_theory/example.lean
```

The MCP tool then uses a relative name such as `algebra/add_comm.lean`.
Path traversal and absolute paths are rejected.

You can use another directory:

```bash
export LEAN_VERIFIER_SPECS_DIR=/absolute/path/to/specs
```

The `verify_against_spec` MCP tool is preferable to passing both strings dynamically.
If an AI can modify both the specification and candidate, it can weaken the specification.
Verification can only prove that a candidate matches the specific statement it was given.

## Deployment profiles

### Hybrid scholarly research + local verification—recommended

The target architecture keeps the Lean/SafeVerify worker local and networkless while separate,
untrusted research MCPs access allowlisted scholarly services. See
**`HYBRID_RESEARCH_PLAN.md`** and **`hermes-hybrid-research-config.example.yaml`**.

This repository now includes a `research-evidence-mcp` entry point for official ERIC,
OpenAlex, and Crossref metadata. It keeps provenance, publication type, peer-review evidence,
open-access status, correction/retraction results, document versions, and claim-level quotations
separate. Research records are evidence for proof development—not formal Lean results.

### Optional fully offline profile

For a completely disconnected deployment, endpoint-by-endpoint implementation plan, artifact
bundling procedure, offline Hermes cutover, security controls, parity tests, and Volumes 1–54
runbook, see **`OFFLINE_SERVER_PLAN.md`**. A no-cloud Hermes template is provided at
**`hermes-offline-config.example.yaml`**. This remains an optional strict profile, not the
system-wide network policy of the recommended hybrid setup.

## Hermes + cloud AXLE setup

To register the cloud-backed AXLE stdio adapter, this local verifier, and optionally a local
Lean LSP server in Hermes Agent, follow **`HERMES_AXLE_SETUP.md`**.

Automated setup:

```bash
uv sync
uv run python scripts/configure_hermes_axle.py \
  --lean-project /absolute/path/to/your/lean/project
```

Hermes uses the top-level YAML key `mcp_servers`, not Claude's JSON key `mcpServers`. The
script discovers the active Hermes profile, creates backups, stores the AXLE key in the
profile `.env`, uses absolute executable paths, and runs `hermes mcp test`.

## Other MCP clients

Copy `mcp-config.example.json` into your MCP client's configuration and replace both
`/ABSOLUTE/PATH/TO/...` placeholders.

Typical configuration:

```json
{
  "mcpServers": {
    "lean-local-verifier": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/lean-local-verifier",
        "run",
        "lean-verifier-mcp"
      ],
      "env": {
        "LEAN_VERIFIER_IMAGE": "local-lean-verifier:4.27.0",
        "LEAN_VERIFIER_SPECS_DIR": "/absolute/path/to/lean-local-verifier/specs",
        "LEAN_VERIFIER_TIMEOUT": "120"
      }
    }
  }
}
```

Some clients, particularly VS Code configurations, call the top-level object `servers`
instead of `mcpServers`. Use the form required by your client.

Restart the MCP client after changing its configuration. Starting
`uv run lean-verifier-mcp` manually and seeing no text is normal: a stdio MCP server waits
for JSON-RPC input and must not print logs to stdout.

### MCP tools

#### `verifier_health()`

Checks Docker, the image, Lean, Lake, and the SafeVerify executable.

#### `list_verification_specs()`

Lists `.lean` files under the trusted specifications directory.

#### `verify_against_spec(spec_name, candidate, timeout_seconds=120)`

The recommended interface. It loads an immutable local spec and verifies candidate code.

#### `verify_proof(formal_statement, candidate, timeout_seconds=120)`

Verifies two dynamically supplied source strings. This is useful for experimentation, but
the caller is responsible for ensuring `formal_statement` is the intended theorem.

### Scholarly evidence MCP

Run the separate connected research gateway with:

```bash
export SCHOLAR_CONTACT_EMAIL=researcher@example.org
uv run research-evidence-mcp
```

It exposes:

- `search_eric` and `lookup_eric` for official U.S. Department of Education ERIC records;
- `search_openalex` for broad discovery with explicit OA and retraction metadata;
- `lookup_doi` for Crossref/OpenAlex metadata consistency plus Crossref update checks; and
- `make_claim_evidence` for a hashed claim, exact quotation, locator, version, and status record.

The gateway is intentionally outside the verifier container. It uses fixed official API hosts,
does not follow redirects, and has no arbitrary URL-fetch tool. Its records explicitly state
that metadata, peer-review status, and quotations are not Lean verification results. See
`HYBRID_RESEARCH_PLAN.md` for the trust and source policy.

## Result format

Successful result, abbreviated:

```json
{
  "okay": true,
  "stage": "verified",
  "formal_statement_result": {
    "exit_code": 0,
    "diagnostics": []
  },
  "candidate_result": {
    "exit_code": 0,
    "diagnostics": []
  },
  "verification_result": {
    "exit_code": 0,
    "timed_out": false
  },
  "guarantees": {
    "declaration_names_kinds_and_types_match": true,
    "proof_terms_kernel_replayed": true,
    "only_standard_axioms_allowed": true,
    "unsafe_rejected": true,
    "partial_rejected": true,
    "runtime_network_disabled": true
  }
}
```

Failure stages:

- `input` — invalid request or source too large
- `policy` — banned source construct
- `compile_formal_statement` — invalid specification
- `compile_candidate` — Lean syntax, elaboration, tactic, or type error
- `safe_verify` — changed declaration, `sorry`, custom axiom, kernel replay failure, etc.
- `timeout` — the total job deadline expired
- `infrastructure` — Docker or image problem

Lean's JSON diagnostics are returned under each compilation result.

## Configuration

Environment variables:

| Variable | Default | Meaning |
|---|---:|---|
| `LEAN_VERIFIER_IMAGE` | `local-lean-verifier:4.27.0` | Docker image |
| `LEAN_VERIFIER_SPECS_DIR` | repository `specs/` | Trusted spec directory |
| `LEAN_VERIFIER_TIMEOUT` | `120` | Default job timeout in seconds |
| `LEAN_VERIFIER_MAX_TIMEOUT` | `300` | Maximum caller-selected timeout |
| `LEAN_VERIFIER_MAX_SOURCE_BYTES` | `250000` | Maximum bytes per source string |
| `LEAN_VERIFIER_MEMORY` | `2g` | Container memory limit |
| `LEAN_VERIFIER_CPUS` | `2` | Container CPU quota |
| `LEAN_VERIFIER_PIDS` | `256` | Container process limit |
| `LEAN_VERIFIER_TMPFS_SIZE` | `768m` | Writable temporary filesystem size |
| `LEAN_VERIFIER_DOCKER_BIN` | `docker` | Docker executable/path |

If large Mathlib proofs time out or run out of memory, try:

```bash
export LEAN_VERIFIER_TIMEOUT=300
export LEAN_VERIFIER_MEMORY=4g
export LEAN_VERIFIER_CPUS=4
```

## Intentional limitations

- `native_decide` is normally rejected because it introduces a non-kernel proof trust path
  that is not in the allowed axiom set. Use kernel-producing tactics such as `decide`,
  `omega`, `linarith`, `ring`, or explicit lemmas when appropriate.
- Compilation-time metaprogram definitions and execution are rejected by source policy.
  Ordinary trusted Mathlib tactics remain available.
- Imported Lean/Mathlib modules are trusted and are not replayed from source on every job.
  They are pinned in the image.
- The sandbox materially reduces risk, but no container runtime or theorem prover should be
  treated as an absolute security boundary against a hostile kernel-level attacker. Keep
  Docker and the pinned Lean release updated when security fixes are released.
- The project verifies formal Lean declarations. It does not determine whether a formal
  statement correctly represents an informal English claim.

## Updating Lean or Mathlib

The current image intentionally pins Lean and Mathlib to 4.27.0 because SafeVerify's
vendored source targets that version. To upgrade:

1. update `vendor/SafeVerify` to a revision supporting the desired Lean release,
2. retain its license and update `NOTICE`,
3. update `lean-toolchain`, `lakefile.lean`, and `lake-manifest.json`,
4. change `LEAN_TOOLCHAIN` and the image tag in `Dockerfile`, and
5. rebuild and rerun every valid and invalid regression test.

Do not silently use `master` or a nightly toolchain for a verification service.
