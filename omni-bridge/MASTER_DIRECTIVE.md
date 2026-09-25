<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

# Master Installation and Integration Directive — Preservation Copy

> **Provenance note (added at commit time, not part of the directive):**
> This is the verbatim master directive for the Lucifer–Hermes Omni-Bridge
> Prime integration, as supplied 2026-09-23 by the rights holder. It is the
> architectural source of truth for the agentic framework alongside the
> [Lucifer–Hermes whitepaper](../whitepapers/lucifer_hermes_omni_bridge_whitepaper.md).
> The directive references a `framework_complete bridge.json` as a
> co-equal source of truth; **that file has not been supplied to this
> repository** — items that cite it are marked in
> [`README.md`](README.md) as pending cross-check.
> The directive targets a specific Gentoo host (`nvme0n1p7`) and Btrfs
> data volume (`nvme0n1p3`). Those steps are host-side; see
> [`scripts/omni_host_audit.py`](scripts/omni_host_audit.py) and the
> status map in [`README.md`](README.md) for what runs where.

---

# MASTER INSTALLATION AND INTEGRATION DIRECTIVE

## Lucifer–Hermes Omni-Bridge Prime

### Existing Hermes/Lucifer Codebase → Btrfs `nvme0n1p3` → Gentoo Runtime `nvme0n1p7`

---

## 0. Mission

You are the implementation, integration, systems-engineering, verification, security, benchmarking, and deployment agent responsible for turning the existing unfinished Hermes and Lucifer implementations on this computer into a coherent, production-grade Lucifer–Hermes Omni-Bridge Prime system based on the supplied `framework_complete bridge.json`.

The objective is **not** to blindly recreate the architecture from scratch.

The objective is:

1. discover the existing Hermes implementation;
2. discover the existing Lucifer implementation;
3. preserve useful existing work;
4. determine what is actually implemented;
5. determine what is incomplete, experimental, broken, unsafe, duplicated, or obsolete;
6. create a dedicated Btrfs framework volume on `nvme0n1p3` if and only if that partition is confirmed safe to format;
7. migrate/integrate the existing implementations into that volume;
8. make the system execute using the existing Gentoo installation and kernel on `nvme0n1p7`;
9. implement the missing Omni-Bridge control boundary;
10. establish deterministic security, authorization, provenance, verification, rollback, telemetry, and testing;
11. benchmark every performance claim;
12. only then introduce advanced optimization;
13. never fake implementation, verification, security, benchmarks, or scientific validity.

The supplied architecture document is the architectural source of truth, but **actual implementation evidence, tests, measurements, security boundaries, and verified behavior take precedence over aspirational claims in the document**.

The architecture itself explicitly identifies several mechanisms as uncertain, research-oriented, or requiring empirical validation. Do not convert those claims into facts merely because they appear in the architecture.

---

# 1. CRITICAL STORAGE AND OPERATING-SYSTEM RULE

## 1.1 Existing host

The existing Gentoo installation on:

```text
/dev/nvme0n1p7
```

is the host operating system.

It supplies:

* Linux kernel
* boot environment
* system libraries
* compiler toolchain
* CPU scheduling
* device access
* networking
* systemd/OpenRC as actually configured
* GPU/NPU drivers
* eBPF facilities
* filesystem drivers
* userspace environment

Do not replace it.

Do not create a second Linux installation.

Do not create a second kernel.

Do not create a second initramfs.

Do not create a second bootloader.

Do not make `nvme0n1p3` independently bootable.

The framework must run as an application/system of services under the kernel already running from the Gentoo installation.

---

# 2. TARGET PARTITION

The intended framework storage target is:

```text
/dev/nvme0n1p3
```

The intended execution host is:

```text
/dev/nvme0n1p7
```

The desired filesystem for p3 is:

```text
Btrfs
```

if the partition is confirmed unused.

---

# 3. NEVER FORMAT BEFORE VERIFICATION

Before modifying p3, execute a complete storage audit.

Collect:

```bash
lsblk -e7 -o NAME,PATH,SIZE,FSTYPE,FSVER,LABEL,UUID,MOUNTPOINTS,PARTUUID
findmnt
blkid
cat /etc/fstab
cat /etc/os-release
uname -a
```

Also inspect:

```bash
sudo fdisk -l /dev/nvme0n1
sudo parted /dev/nvme0n1 print
```

Do not assume p3 is blank from its lack of a filesystem label.

Determine whether:

* a filesystem exists;
* partitions contain recognizable signatures;
* the partition is mounted;
* it appears in `/etc/fstab`;
* it is used by another service;
* it contains data;
* it has LVM, RAID, encryption, or other metadata;
* another mount mechanism references it.

If p3 contains meaningful data or its purpose cannot be established with high confidence:

**STOP.**

Do not format it.

Report:

* partition state;
* filesystem;
* mount status;
* signatures;
* suspected usage;
* exact reason for stopping.

---

# 4. IF AND ONLY IF p3 IS CONFIRMED EMPTY

Create Btrfs on p3.

Use the system's installed Btrfs tooling.

Do not blindly copy a filesystem command from documentation without first determining the installed `btrfs-progs` version and available features.

After formatting:

```bash
blkid /dev/nvme0n1p3
lsblk -f
```

Record:

```text
filesystem type
UUID
label
size
sector information
mount options
```

The partition must remain a data/application filesystem.

It must NOT contain:

```text
/boot
kernel
initramfs
bootloader
EFI system
```

---

# 5. BTRFS MOUNT DESIGN

Choose a stable framework mountpoint.

Preferred:

```text
/opt/omni-bridge
```

or, if the existing Gentoo filesystem layout strongly favors it:

```text
/var/lib/omni-bridge
```

Use one canonical mountpoint and document the decision.

The canonical framework root becomes:

```text
/opt/omni-bridge
```

unless host inspection demonstrates a compelling reason otherwise.

The mount must be persistent through `/etc/fstab` using the filesystem UUID rather than assuming the device path will never change.

Example conceptual entry:

```text
UUID=<P3-UUID>  /opt/omni-bridge  btrfs  <validated-options>  0  0
```

Do not invent mount options.

Determine which options are appropriate for this actual hardware, workload, kernel, and Btrfs version.

---

# 6. BTRFS SUBVOLUME ARCHITECTURE

Do not put everything into one undifferentiated directory.

Create Btrfs subvolumes with separate lifecycle characteristics.

Recommended:

```text
/opt/omni-bridge/
├── @root
├── @source
├── @build
├── @runtime
├── @skills
├── @memory
├── @retrieval
├── @audit
├── @telemetry
├── @models
├── @cache
├── @snapshots
└── @backup-staging
```

Use actual mountpoints such as:

```text
/opt/omni-bridge
/opt/omni-bridge/source
/opt/omni-bridge/build
/opt/omni-bridge/runtime
/opt/omni-bridge/skills
/opt/omni-bridge/memory
/opt/omni-bridge/retrieval
/opt/omni-bridge/audit
/opt/omni-bridge/telemetry
/opt/omni-bridge/models
/opt/omni-bridge/cache
/opt/omni-bridge/snapshots
/opt/omni-bridge/backup-staging
```

Do not blindly create all subvolumes if they complicate the system without providing a measurable benefit.

Every Btrfs design choice must have a reason.

---

# 7. IMPORTANT BTRFS SAFETY RULES

Do not enable aggressive compression, CoW disabling, nodatacow, special mount flags, or unusual Btrfs tuning merely because they sound faster.

Benchmark them.

For databases, append-only audit logs, VM images, model files, build trees, and high-write telemetry, determine the appropriate policy separately.

Do not disable CoW globally merely for performance.

Do not trade filesystem integrity for speculative latency.

Snapshots must not be treated as backups.

A snapshot is useful for rollback; a real backup must exist independently.

---

# 8. HOST KERNEL RULE

The kernel is owned by the Gentoo installation on p7.

Verify:

```bash
uname -r
uname -a
```

Determine whether the currently running kernel provides required facilities.

Inspect:

```text
Btrfs
namespaces
cgroups v2
seccomp
user namespaces
network namespaces
mount namespaces
eBPF
BPF LSM where available
io_uring where useful
KVM if required
IOMMU if relevant
hugepages
perf
futex
epoll
inotify
fanotify
Landlock if available
```

Do not build or install a new kernel unless the current system actually lacks a required capability.

If a feature is unavailable:

1. identify it;
2. determine whether it is genuinely required;
3. provide a userspace fallback if possible;
4. otherwise produce a clearly documented kernel prerequisite.

The framework must not assume that a second kernel belongs on p3.

---

# 9. EXISTING HERMES AND LUCIFER DISCOVERY

Before creating new code, find the existing implementations.

Search the existing Gentoo environment and user-owned source directories.

Look for:

```text
hermes
lucifer
omni
bridge
agents
skills
SOUL.md
AGENTS.md
pyproject.toml
Cargo.toml
CMakeLists.txt
Makefile
package manifests
service definitions
configuration files
database files
model adapters
API gateways
tool adapters
retrieval indexes
memory stores
logs
test suites
```

Do not assume their locations.

Do not move them immediately.

Create:

```text
build/existing-system-inventory.json
```

Record for every discovered component:

```text
absolute path
language
build system
version
git repository
git revision
dependencies
configuration
runtime dependencies
database dependencies
network dependencies
GPU/NPU dependencies
current status
tests
known failures
security concerns
license
owner
migration risk
```

---

# 10. PRESERVE EXISTING WORK

Before modification:

1. calculate hashes where appropriate;
2. create a source inventory;
3. create a backup;
4. if the existing projects are Git repositories, record:

   * branch;
   * commit;
   * dirty state;
   * remotes;
   * tags;
5. never delete unfinished implementations merely because they are incomplete.

Create:

```text
build/pre-migration-manifest.json
```

and:

```text
build/pre-migration-hashes.json
```

The migration must be reversible.

---

# 11. CLASSIFY EXISTING COMPONENTS

Every existing Hermes/Lucifer component must be classified:

```text
IMPLEMENTED
PARTIALLY_IMPLEMENTED
BROKEN
EXPERIMENTAL
RESEARCH
UNSAFE_FOR_PRODUCTION
DUPLICATE
OBSOLETE
DEFERRED
REJECTED
```

Do not infer implementation from filenames.

A component is implemented only if it actually works and has tests appropriate to its function.

---

# 12. DO NOT REWRITE HERMES OR LUCIFER FOR THE SAKE OF PURITY

Prefer:

```text
existing implementation
        ↓
audit
        ↓
adapter
        ↓
integration
        ↓
refactor where necessary
```

over:

```text
delete existing code
        ↓
rewrite everything
```

Existing working code is an asset.

Existing unsafe assumptions are liabilities.

Preserve assets.

Remove liabilities only after establishing replacements.

---

# 13. TARGET DIRECTORY

The canonical installation becomes:

```text
/opt/omni-bridge/
```

with:

```text
/opt/omni-bridge/
├── source/
│   ├── hermes/
│   ├── lucifer/
│   └── omni-bridge/
│
├── build/
├── runtime/
├── config/
├── skills/
├── memory/
├── retrieval/
├── audit/
├── telemetry/
├── models/
├── cache/
├── snapshots/
├── backup-staging/
└── docs/
```

The exact arrangement may be refined after inspection.

---

# 14. SOURCE REPOSITORY

Create a unified integration repository where appropriate:

```text
/opt/omni-bridge/source/omni-bridge/
```

The repository should contain:

```text
docs/
schemas/
config/
crates/
cpp/
python/
lean/
wasm/
bpf/
skills/
tests/
deploy/
scripts/
```

Use Hermes and Lucifer as real source components rather than pretending they do not exist.

---

# 15. LANGUAGE ARCHITECTURE

Use the language appropriate to the responsibility.

## C++20/23

Prefer for:

* latency-sensitive routing;
* zero-copy structures;
* typed execution frames;
* shared-memory mechanisms;
* high-throughput telemetry;
* low-level systems integration;
* SIMD where demonstrated beneficial;
* hardware interfaces where justified.

Use the existing Gentoo C++ optimization/toolchain environment rather than replacing it unnecessarily.

Determine:

```bash
gcc --version
g++ --version
clang --version
cmake --version
ninja --version
```

and inspect the actual compiler flags already used by the existing implementation.

Do not discard useful existing Gentoo optimizations.

Do not add unsafe flags merely because they improve a benchmark.

## 16. RUST

Use Rust for:

* memory-sensitive services;
* concurrency;
* Wasmtime integration;
* security-sensitive components;
* FFI boundaries;
* services where memory safety materially reduces risk.

Install/use the existing Rust toolchain if present.

Otherwise establish a reproducible Rust toolchain.

Pin it.

Record:

```text
rustc version
cargo version
toolchain
target
build profile
dependency lock state
```

## 17. PYTHON

Python is allowed for:

* offline research;
* evaluation;
* benchmarking;
* evolution experiments;
* data analysis;
* tooling;
* model experimentation.

Do not put Python into a latency-critical control path merely because it is easier.

Existing Python Hermes/Lucifer functionality must not automatically be rewritten.

Instead measure where Python actually matters.

## 18. LEAN

Lean is used for formal verification of selected invariants.

Do not pretend Lean proves the entire agent correct.

Prioritize:

```text
authorization invariants
capability intersection
taint propagation
declassification conditions
promotion conditions
provenance invariants
rollback invariants
state-machine safety properties
```

Formalize the boundaries where mathematical certainty is actually useful.

---

# 19. OMNI-BRIDGE CONTROL BOUNDARY

Create the missing integration layer between Hermes and Lucifer.

Conceptually:

```text
                  USER / EXTERNAL INPUT
                          │
                          ▼
                 TASK NORMALIZATION
                          │
                          ▼
              INTENT / STATE HYPOTHESES
                          │
                          ▼
                 SECURITY CLASSIFIER
                          │
                          ▼
                 CAPABILITY CHECK
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
          HERMES                    LUCIFER
       GOVERNANCE                  EXPLORATION
             │                         │
             └────────────┬────────────┘
                          ▼
                  ROUTE GENERATION
                          │
                          ▼
                   ROUTE RANKING
                          │
                          ▼
                    BUDGET CHECK
                          │
                          ▼
                 SANDBOX SELECTION
                          │
                          ▼
                     EXECUTION
                          │
                          ▼
                     VERIFICATION
                          │
                          ▼
                   PROVENANCE
                          │
                          ▼
                       AUDIT
                          │
                          ▼
                  OUTCOME / TELEMETRY
                          │
                          ▼
                OFFLINE EVOLUTION
```

Lucifer may propose.

Hermes may govern.

The Omni-Bridge decides whether the proposed action is permitted.

No LLM may directly override the control boundary.

---

# 20. HERMES RESPONSIBILITIES

Hermes remains responsible for:

```text
typed expertise
retrieval
knowledge
procedures
constraints
anti-skills
tool adapters
provenance
risk validation
artifact lifecycle
regression
promotion
canary
rollback
retirement
```

The framework specification explicitly describes Hermes as the governance/expertise layer containing versioned knowledge, procedures, constraints, tool adapters, and composites.

Implement progressive disclosure:

```text
Level 0
catalog / summary

Level 1
core procedure

Level 2
supporting files / templates / scripts
```

Do not inject the entire skill database into every LLM request.

---

# 21. LUCIFER RESPONSIBILITIES

Lucifer remains responsible for exploration and synthesis.

Implement the four principal roles:

```text
CONQUEST
WAR
FAMINE
DEATH
```

Conquest:

```text
planning
synthesis
architecture
evidence integration
```

War:

```text
red teaming
threat modeling
contradiction search
failure discovery
```

Famine:

```text
low-budget reasoning
latency-constrained execution
minimal sufficient computation
```

Death:

```text
invalid assumption detection
branch pruning
contradiction analysis
termination
```

The supplied architecture explicitly defines these roles and states that Lucifer is not an unrestricted authority.

---

# 22. LUCIFER MUST NOT GOVERN ITSELF

Lucifer must not be able to:

```text
grant itself permissions
modify authorization policy
expand capabilities
promote its own skills
disable Hermes
disable mandatory verification
remove audit records
change production security floors
manufacture credentials
```

Lucifer generates candidates and routes.

The control plane enforces authority.

---

# 23. HERMES MUST NOT BECOME AN UNCHECKED BOTTLENECK

Hermes should govern without becoming an unnecessarily centralized computational bottleneck.

Use:

```text
immutable policy floors
cached metadata
typed decisions
bounded retrieval
deterministic validators
precomputed indexes
efficient route candidates
```

Benchmark Hermes overhead.

---

# 24. APPA

Implement the permission algebra.

Effective permissions are the intersection of applicable authority sets.

Conceptually:

```text
Effective =
System
∩ Declared
∩ Tenant
∩ Tool
∩ Context
∩ Risk
```

Risk may remove authority.

Risk must never manufacture authority.

This follows the architecture's explicit APPA model.

---

# 25. SECURITY LABELS

Every significant artifact and execution context should carry:

```text
integrity
confidentiality
origin
purpose
provenance
```

Never allow an untrusted artifact to silently become trusted merely because an LLM summarized it.

---

# 26. PROSPECTIVE ACQUISITION

Before consuming external content:

```text
source
↓
classify
↓
label
↓
isolate
↓
sanitize
↓
derive bounded information
↓
validate
↓
permit insertion into trusted context
```

Do not allow web pages, documents, tool descriptions, or retrieved text to acquire authority merely by being retrieved.

---

# 27. CONTEXT BRANCHING

Untrusted external material must be processed in restricted child contexts when necessary.

Child branches must have:

```text
restricted capabilities
no unnecessary credentials
restricted filesystem
restricted network
bounded execution
sanitization
provenance
```

A child context cannot upgrade itself.

---

# 28. WASM SANDBOXING

Use Wasmtime/WASI where it is an appropriate sandbox tier.

Suggested execution tiers:

```text
T0 trusted deterministic in-process
T1 Wasmtime/WASI
T2 hardened container
T3 microVM
T4 human-approved irreversible action
T5 prohibited
```

Do not claim Wasmtime is sufficient for every untrusted workload.

Native binaries and high-risk operations may require stronger isolation.

---

# 29. TOOL ABI

Every tool must expose a manifest describing:

```text
tool ID
version
input schema
output schema
required capabilities
security requirements
network requirements
filesystem requirements
resource limits
timeout
determinism
side effects
rollback
verification requirements
```

Execution pipeline:

```text
agent proposal
↓
schema validation
↓
APPA authorization
↓
risk evaluation
↓
sandbox selection
↓
resource limits
↓
execution
↓
output validation
↓
verification
↓
audit
```

---

# 30. ROUTING

Route complete execution plans rather than merely selecting a model.

A route should describe:

```text
models
roles
skills
retrieval
tools
verification
sandbox
latency budget
cost budget
risk
expected confidence
resource requirements
fallbacks
```

Rank using measured evidence involving:

```text
quality
risk
latency
cost
capability fit
verification burden
historical outcomes
resource usage
```

Never lower mandatory security merely to reduce cost.

---

# 31. FAST AND SLOW PATHS

Implement two operational paths.

Fast path:

```text
classification
known route
known expertise
bounded execution
light verification
```

Slow path:

```text
uncertainty
novel task
multiple hypotheses
War review
Death pruning
additional verification
human review where necessary
```

The slow path must be available whenever the fast path cannot establish sufficient confidence or safety.

---

# 32. RETRIEVAL

Begin with:

```text
lexical retrieval
+
dense retrieval
+
metadata filtering
```

Use SQLite FTS5 where appropriate.

Use Qdrant or another vector store only where its measured benefit justifies it.

Graph retrieval remains gated.

The architecture explicitly requires graph retrieval to beat lexical+dense retrieval on held-out recall, utility, false positives, cost, latency, and poisoning robustness before production adoption.

---

# 33. MEMORY

Separate:

```text
working memory
episodic memory
semantic memory
procedural memory
policy memory
audit memory
experimental memory
```

Each persistent memory item should record:

```text
source
confidence
provenance
time
security classification
validation status
expiry
dependencies
```

Similarity is not truth.

Embedding distance is not proof.

---

# 34. EVIDENCE LEDGER

Implement an append-only evidence system.

Record, where applicable:

```text
timestamp
trace ID
actor
route
model
skill
tool
input hash
output hash
security labels
capabilities
budget
latency
verification result
outcome
previous hash
current hash
```

Use BLAKE3 where appropriate.

The architecture specifies a tamper-evident evidence chain and asynchronous telemetry architecture.

Never put raw credentials into the ledger.

Never log secrets.

---

# 35. REPLAY

Implement replay classifications:

```text
EXACT_REPLAY
DETERMINISTIC_REPLAY
SIMULATED_COUNTERFACTUAL
APPROXIMATE_COUNTERFACTUAL
NON_REPLAYABLE
```

Never label a nondeterministic simulation as an exact replay.

---

# 36. THREE-TIER AUDIT

Implement:

## Tier 1 — Engine

Mandatory deterministic checks:

```text
schema
syntax
compile
resource limits
sandbox result
security
capabilities
deterministic invariants
```

## Tier 2 — Scrutiny

Check:

```text
known expertise
constraints
versions
regressions
contradictions
dependency status
```

## Tier 3 — Meta-scrutiny

Research diagnostics may include:

```text
Fisher-Rao distance
embedding distance
novelty
curvature
gradient-related diagnostics
```

But these are not truth or usefulness oracles.

The supplied framework explicitly warns against treating these mathematical measures as universal truth validators.

Do not implement:

```text
Fisher-Rao distance >= X
therefore truth
```

as an unconditional production theorem.

Benchmark and validate the metric first.

---

# 37. RCOD / SWARM SIGNALS

If the existing Lucifer implementation contains RCOD/Swarm Dissonance mechanisms, preserve them as experimental/operational telemetry unless empirical validation establishes a stronger status.

They may be used for:

```text
anomaly detection
throttling
early warning
branch instability
```

Do not treat them as proven physical laws or infallible reasoning-collapse detectors.

Every automated freeze must have:

```text
false-positive measurement
false-negative measurement
operator override
fallback
audit trail
```

---

# 38. EVOLUTION

Production must never directly mutate itself.

Use:

```text
production
↓
telemetry
↓
failure trace
↓
offline Dream State
↓
candidate
↓
Tier 1
↓
Tier 2
↓
Tier 3
↓
shadow
↓
canary
↓
promotion gate
↓
production
```

The architecture explicitly requires controlled offline evolution and reversible promotion.

---

# 39. GEPA / MIPRO / TEXTGRAD / EVOLUTION

These mechanisms are research/evolution tools.

They must not directly modify production policy.

A failure must produce:

```text
failure
↓
root-cause hypothesis
↓
candidate mutation
↓
new tests
↓
verification
↓
shadow evaluation
↓
promotion decision
```

Never:

```text
failure
↓
automatically rewrite production
```

---

# 40. SKILL LIFECYCLE

Implement:

```text
DISCOVERED
↓
CANDIDATE
↓
TESTING
↓
AUDITED
↓
SHADOW
↓
CANARY
↓
PROMOTED
↓
MONITORED
↓
DEPRECATED
↓
RETIRED
```

Every transition must be auditable.

---

# 41. ROLLBACK

Every production artifact must be versioned.

Support:

```text
version pinning
atomic promotion
health checks
shadow deployment
canary
rollback
dependency rollback
configuration rollback
database migration strategy
```

Use Btrfs snapshots as an additional rollback mechanism where appropriate, but do not make snapshots the sole recovery mechanism.

---

# 42. SERVICES

Create separate services only where separation provides a real security or operational benefit.

Possible architecture:

```text
omni-gateway
omni-router
hermes
lucifer
omni-runtime
omni-retrieval
omni-memory
omni-audit
omni-verifier
omni-evolution
omni-observability
```

Do not create twelve services simply because the architecture diagram contains twelve boxes.

Measure the cost of process boundaries.

---

# 43. SERVICE ACCOUNTS

Each service should have the minimum privileges required.

Prefer:

```text
dedicated user
restricted filesystem
restricted network
cgroup limits
resource limits
seccomp where appropriate
capability reduction
read-only mounts where possible
```

No service receives root merely because it is convenient.

---

# 44. CONFIGURATION

Create:

```text
config/development/
config/staging/
config/production/
```

Security-sensitive defaults must fail closed.

Validate configuration at startup.

Invalid configuration must prevent unsafe execution.

---

# 45. SECRETS

Never store secrets in:

```text
Git
source
prompt
skills
audit ledger
telemetry
logs
benchmark output
container image
Btrfs snapshot
```

Use the existing Gentoo/system secret-management mechanism where available.

Separate Hermes and Lucifer credential namespaces.

The framework architecture explicitly calls for provider allowlists, credential separation, timeouts, cost tracking, and secret IDs rather than embedded credentials.

---

# 46. DATABASES

If Hermes already has databases:

1. identify them;
2. determine schema;
3. back them up;
4. migrate carefully;
5. test restoration;
6. benchmark.

Do not destroy existing Hermes memory.

Do not automatically merge incompatible databases.

---

# 47. BUILD SYSTEM

Create reproducible build commands:

```bash
./scripts/bootstrap.sh
./scripts/doctor.sh
./scripts/build.sh
./scripts/test.sh
./scripts/security-audit.sh
./scripts/benchmark.sh
```

Also provide:

```bash
omni doctor
omni status
omni build
omni test
omni verify
omni benchmark
omni audit
omni replay
omni skills
omni routes
omni rollback
```

---

# 48. FIRST BUILD

The first successful build must prioritize correctness.

Do not initially enable:

```text
DPDK
custom kernel modifications
eBPF acceleration
hugepage optimization
NUMA pinning
custom NPU drivers
mirrored virtual-memory buffers
aggressive SIMD
lock-free everything
```

unless the existing code already depends on them and they can be safely isolated.

---

# 49. PERFORMANCE OPTIMIZATION ORDER

Use:

```text
correctness
↓
security
↓
profiling
↓
benchmark
↓
optimization
↓
benchmark again
```

Only keep an optimization if:

```text
performance improves
AND
correctness remains intact
AND
security remains intact
AND
operational complexity is justified
```

---

# 50. ZERO-COPY

After the system works, investigate:

```text
std::span
fixed binary frames
shared memory
ring buffers
SIMD
cache alignment
NUMA
hugepages
```

Do not use zero-copy merely because it sounds sophisticated.

Measure:

```text
p50
p90
p95
p99
p99.9
CPU cycles
allocations
cache misses
memory bandwidth
tail latency
```

---

# 51. TELEMETRY

The critical path should avoid:

```text
dynamic formatting
blocking disk writes
unbounded queues
unbounded memory allocation
```

Use bounded asynchronous telemetry.

Stress-test overflow behavior.

Telemetry failure must never silently corrupt authorization or execution state.

---

# 52. OBSERVABILITY

Implement trace correlation across:

```text
input
classification
Hermes retrieval
Lucifer exploration
route generation
route selection
tool invocation
sandbox
verification
audit
outcome
evolution
```

Every major operation should be reconstructable.

Never record secrets merely for observability.

---

# 53. ADVERSARIAL TESTING

Test:

```text
prompt injection
indirect prompt injection
malicious documents
malicious skills
malicious tool descriptions
schema attacks
Unicode attacks
path traversal
symlink attacks
TOCTOU
resource exhaustion
Wasm traps
oversized inputs
integer overflow
buffer overflow
race conditions
audit tampering
rollback corruption
retrieval poisoning
memory poisoning
capability escalation
credential leakage
```

---

# 54. FUZZING

Fuzz:

```text
binary parsers
JSON
tool manifests
route objects
capability objects
sandbox configuration
audit records
memory records
Wasm adapters
FFI boundaries
```

Use appropriate sanitizers:

```text
ASan
UBSan
TSan
MSan
```

where supported and practical.

---

# 55. TEST HERMES

Create tests for:

```text
artifact loading
artifact versioning
retrieval
progressive disclosure
constraints
anti-skills
provenance
promotion
retirement
rollback
contamination control
```

---

# 56. TEST LUCIFER

Test:

```text
Conquest
War
Famine
Death
branching
budget exhaustion
role switching
termination
model failure
provider failure
contradictory outputs
```

---

# 57. TEST THE BRIDGE

The most important tests are cross-system tests:

```text
Lucifer proposes forbidden action
→ Hermes/Omni-Bridge rejects

Lucifer receives malicious document
→ branch isolation prevents privilege escalation

Hermes retrieves obsolete skill
→ eligibility check rejects it

tool output violates schema
→ execution is rejected

verification disagrees
→ system escalates/abstains

provider fails
→ deterministic fallback

router becomes unstable
→ baseline route

audit subsystem fails
→ high-risk autonomous execution stops

evolution produces unsafe candidate
→ candidate rejected

rollback requested
→ previous known-good artifact restored
```

---

# 58. SMOKE TEST

Create a complete end-to-end smoke test:

```text
user task
↓
normalization
↓
security classification
↓
Hermes retrieval
↓
Lucifer exploration
↓
route generation
↓
route authorization
↓
tool invocation
↓
sandbox
↓
verification
↓
audit
↓
result
```

Every stage must leave traceable evidence.

---

# 59. PROMPT-INJECTION ACCEPTANCE TEST

Inject malicious instructions into:

```text
web page
PDF
skill
retrieved document
tool description
model output
memory
```

Verify:

```text
untrusted text remains data
untrusted text cannot grant capability
untrusted text cannot change policy
untrusted text cannot access credentials
untrusted text cannot modify trusted memory
```


# 60. CAPABILITY-ESCALATION ACCEPTANCE TEST

Attempt:

```text
Lucifer → privilege escalation
Hermes skill → privilege escalation
tool output → privilege escalation
retrieved document → privilege escalation
LLM response → privilege escalation
evolution candidate → privilege escalation
```

Every attempt must fail closed.

---

# 61. EVOLUTION ACCEPTANCE TEST

Create three candidates:

```text
bad
redundant
genuinely useful
```

Verify:

```text
bad → rejected
redundant → rejected/deferred
useful → shadow/canary
```

Do not automatically promote merely because a candidate appears novel.

---

# 62. ROLLBACK ACCEPTANCE TEST

Break a known-good component deliberately in staging.

Verify:

```text
health failure
↓
promotion stop
↓
rollback
↓
known-good version
↓
verification
↓
service recovery
```

---

# 63. RECOVERY

Create documented procedures for:

```text
database loss
audit corruption
retrieval corruption
skill poisoning
model outage
provider outage
filesystem failure
Btrfs corruption
bad deployment
bad evolution
credential compromise
security incident
```

---

# 64. BTRFS BACKUP POLICY

Implement:

```text
local snapshot
+
independent backup
+
restore verification
```

Never say "we have backups" until restoration has actually been tested.

---

# 65. KERNEL / P7 INTEGRATION

All runtime processes must execute under the kernel currently provided by p7.

When a framework component requests kernel functionality:

```text
inspect current kernel
↓
verify availability
↓
use it
```

If unavailable:

```text
fallback
or
document kernel requirement
```

Do not automatically replace the kernel.

---

# 66. HARDWARE DISCOVERY

Inspect:

```bash
lscpu
lspci -nn
lsusb
free -h
numactl --hardware
cat /proc/meminfo
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_driver
```

If applicable inspect:

```text
GPU
NPU
IOMMU
hugepages
eBPF
BPF LSM
KVM
```

Do not assume the machine has hardware merely because the architecture document discusses it.

---

# 67. NPU

If AMD XDNA/NPU functionality exists on this specific machine:

1. identify hardware;
2. identify kernel driver;
3. identify firmware;
4. identify userspace runtime;
5. determine whether supported;
6. benchmark;
7. isolate the integration.

Do not reverse-engineer or install an experimental NPU stack merely to satisfy the architecture document.

The architecture itself distinguishes validated mechanisms from hardware-specific experimental mechanisms.

---

# 68. EBBF / EBPF

eBPF should initially be an observability/control mechanism.

Only introduce execution-path acceleration after proving that it provides a measurable benefit.

Never make the system dependent on speculative eBPF optimizations before the userspace implementation is correct.

---

# 69. DPDK

DPDK is deferred until benchmarks demonstrate that conventional networking is a bottleneck.

Do not install DPDK merely because the framework mentions high-performance networking.

---

# 70. GRAPH RETRIEVAL

Graph retrieval remains experimental until benchmarked.

Required comparison:

```text
FTS5
vs
dense
vs
FTS5 + dense
vs
FTS5 + dense + graph
```

Measure:

```text
recall
precision
utility
latency
memory
CPU
false positives
poisoning robustness
maintenance cost
```

Remove graph retrieval if it does not pay for itself.

---

# 71. RESEARCH CLAIMS

Treat these as hypotheses unless experimentally validated:

```text
Fisher-Rao as novelty detector
PCRB as hallucination detector
RCOD as reasoning-collapse predictor
Swarm Dissonance as universal instability detector
ToM inference
provider-side KV cache behavior
NPU performance claims
sub-millisecond end-to-end routing
"immunity" to prompt injection
mathematical elimination of hallucinations
```

Never write a report claiming proof when the test only demonstrated correlation.

---

# 72. KILL SWITCHES

Implement:

```text
OMNI_DYNAMIC_EXPERTISE_OFF=1
OMNI_LUCIFER_OFF=1
OMNI_HERMES_RETRIEVAL_OFF=1
OMNI_GENERATION_OFF=1
OMNI_LEARNING_OFF=1
OMNI_AUDIT_READONLY=1
OMNI_MODEL_PIN=<approved-set>
OMNI_ARTIFACT_PIN=<approved-set>
```

Every switch must leave a functional degraded mode whenever safely possible.

The supplied architecture explicitly specifies this philosophy and fallback requirement.

---

# 73. CI/CD

Every change should pass, as appropriate:

```text
format
lint
compile
unit tests
integration tests
property tests
fuzz tests
security tests
Lean verification
Wasm tests
benchmark regression
dependency checks
container checks
SBOM
```

Do not permit a failing security gate to be bypassed merely to obtain a green build.

---

# 74. SUPPLY CHAIN

Pin:

```text
compiler
Rust toolchain
dependencies
container images
Python dependencies
Wasm dependencies
model adapters
```

Generate:

```text
SBOM
dependency lockfiles
build provenance
```

Scan for:

```text
vulnerabilities
license conflicts
malicious dependencies
unmaintained dependencies
```

Do not add arbitrary GitHub dependencies simply because they appear in an example.

---

# 75. SYSTEM SERVICE DEPLOYMENT

Once the application is proven manually, create appropriate Gentoo service integration.

Determine whether this machine uses:

```text
OpenRC
systemd
```

Do not assume systemd.

Use the host's actual service manager.

Services must:

```text
run as dedicated users
have resource limits
have appropriate filesystem access
have appropriate network access
restart safely
expose health checks
produce structured logs
```

---

# 76. INSTALLATION ORDER

Execute in this exact broad order:

```text
1. Audit host
2. Audit partitions
3. Confirm p7 as host
4. Confirm p3 is safe to format
5. Create Btrfs
6. Mount p3
7. Create subvolume structure
8. Snapshot initial state
9. Discover Hermes
10. Discover Lucifer
11. Back up existing implementations
12. Inventory implementations
13. Build existing projects independently
14. Record failures
15. Create integration repository
16. Establish typed schemas
17. Integrate Hermes
18. Integrate Lucifer
19. Build Omni-Bridge
20. Implement APPA
21. Implement tool ABI
22. Implement sandbox layer
23. Implement provenance
24. Implement audit
25. Implement verification
26. Implement routing
27. Implement retrieval
28. Implement memory
29. Implement replay
30. Implement observability
31. Run unit tests
32. Run integration tests
33. Run adversarial tests
34. Run fuzzing
35. Run benchmarks
36. Fix correctness issues
37. Fix security issues
38. Profile
39. Optimize measured bottlenecks
40. Implement offline evolution
41. Shadow evolution
42. Canary evolution
43. Validate rollback
44. Produce readiness report
45. Only then consider production autonomous operation
```

---

# 77. DO NOT SKIP THE EXISTING-CODE AUDIT

Because Hermes and Lucifer already exist, the most important first engineering question is:

> What is already real?

Do not recreate something that already works.

Do not trust something merely because it exists.

Do not delete something merely because it is incomplete.

---

# 78. FINAL DIRECTORY MODEL

The desired result should resemble:

```text
nvme0n1p7
└── Gentoo Linux
    └── Linux kernel
        │
        │ executes
        ▼
nvme0n1p3
└── Btrfs
    └── /opt/omni-bridge
        │
        ├── Hermes
        ├── Lucifer
        ├── Omni-Bridge
        ├── Runtime
        ├── Skills
        ├── Retrieval
        ├── Memory
        ├── Audit
        ├── Telemetry
        ├── Models
        ├── Cache
        ├── Snapshots
        └── Backups
```

There is exactly **one operating system kernel**.

There is exactly **one boot environment**.

p3 is application/data storage.

p7 is the host.

---

# 79. FINAL ACCEPTANCE CRITERIA

Do not declare the system complete until all of the following are true:

### Storage

```text
p3 verified as Btrfs
p3 mounted correctly
p3 persistent across reboot
p7 remains the sole OS
no second kernel on p3
```

### Existing implementation

```text
Hermes discovered
Lucifer discovered
existing functionality inventoried
existing code backed up
existing tests executed
existing failures documented
```

### Architecture

```text
Hermes integrated
Lucifer integrated
Omni-Bridge operational
control/data separation operational
typed interfaces operational
```

### Security

```text
APPA operational
capability isolation operational
sandbox selection operational
prompt injection tests pass
capability escalation tests pass
secrets absent from logs
```

### Verification

```text
Tier 1 operational
Tier 2 operational
Tier 3 research diagnostics separated from truth claims
Lean invariants verified where implemented
```

### Evidence

```text
append-only audit
BLAKE3 integrity
provenance
replay classification
trace correlation
```

### Evolution

```text
offline generation
candidate evaluation
shadow
canary
promotion
rollback
```

### Reliability

```text
provider fallback
retrieval fallback
Hermes fallback
Lucifer fallback
router fallback
audit failure behavior
kill switches
```

### Performance

```text
baseline measured
p50 measured
p95 measured
p99 measured
p99.9 measured where useful
CPU measured
memory measured
allocation behavior measured
```

### Operations

```text
doctor command
build command
test command
audit command
benchmark command
status command
rollback command
backup procedure
restore procedure
```

---

# 80. FINAL READINESS REPORT

Produce:

```text
reports/final-readiness.md
reports/final-readiness.json
```

The report must contain:

```text
Host
Kernel
CPU
Memory
GPU/NPU
Filesystem
Partition layout
Existing Hermes status
Existing Lucifer status
Omni-Bridge status
Components implemented
Components partial
Components experimental
Components rejected
Security results
Fuzz results
Formal verification results
Benchmark results
Evolution results
Rollback results
Backup/restore results
Known limitations
Deferred research
Known vulnerabilities
Operational risks
Recommended next steps
```

Do not use words such as:

```text
complete
secure
proven
immune
verified
production-ready
optimal
world-class
```

unless the corresponding claim is backed by explicit evidence.

---

# 81. ABSOLUTE RULES

Never:

```text
format an uncertain partition
destroy existing Hermes/Lucifer code
install a second kernel on p3
modify boot configuration unnecessarily
disable host security for convenience
run everything as root
expose secrets
fake benchmarks
fake formal proofs
fake test results
fake successful builds
allow LLM output to become authorization
allow untrusted content to gain privilege
allow production self-modification
remove mandatory verification to improve latency
treat embeddings as truth
treat Fisher-Rao as truth
treat model confidence as truth
treat RCOD as proven law
add complexity without measured benefit
```

The framework's own governing principle is that every mechanism must have a narrow responsibility, measurable benefit, clear failure mode, and safe removal path.

---

# 82. DEFINITION OF "BEST"

The goal is not the framework containing the most technologies.

The goal is:

```text
maximum verified capability
+
minimum unnecessary complexity
+
strong security
+
efficient resource use
+
reproducibility
+
observability
+
formal verification where useful
+
safe autonomous exploration
+
controlled self-improvement
+
reversible deployment
+
excellent failure behavior
```

A feature that makes the architecture more impressive but makes it less reliable must be removed.

A feature that makes it slower without measurable value must be removed.

A feature that weakens the security boundary must be removed.

A feature that cannot be tested must remain experimental.

---

# 83. OPERATING PRINCIPLE

Work iteratively:

```text
INSPECT
↓
PLAN
↓
BACK UP
↓
IMPLEMENT
↓
COMPILE
↓
TEST
↓
VERIFY
↓
FUZZ
↓
RED TEAM
↓
BENCHMARK
↓
PROFILE
↓
OPTIMIZE
↓
RETEST
↓
DOCUMENT
↓
PROMOTE
```

Never skip directly from:

```text
IMPLEMENT
```

to:

```text
PRODUCTION
```

---

# 84. FIRST ACTION

Before making any destructive filesystem change or moving any existing code, perform the complete host/partition/source audit.

The first objective is to establish an accurate map of:

```text
what p7 currently runs
what kernel p7 currently runs
what p3 currently contains
where Hermes currently lives
where Lucifer currently lives
what each currently does
what dependencies they have
what is already working
what is broken
```

Then present that inventory and proceed with the implementation plan.

Do not format p3 until the storage audit proves it is safe.

Do not rewrite Hermes or Lucifer until the source audit proves rewriting is necessary.

Do not install a second operating system.

Do not install a second kernel.

The target architecture is:

**Gentoo on p7 = host and kernel**

**Btrfs on p3 = dedicated Omni-Bridge/Hermes/Lucifer application and data volume**

**Hermes + Lucifer existing code = starting implementation**

**Omni-Bridge = integration/control boundary**

**Verification + security + measurement = authority over architectural speculation**
