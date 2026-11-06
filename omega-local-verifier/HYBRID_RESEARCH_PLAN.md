# Hybrid Scholarly Research + Local Lean Verification

**Status:** implemented baseline and production-hardening plan  
**Reviewed:** 2026-08-18  
**Supersedes:** the system-wide zero-network policy in `OFFLINE_SERVER_PLAN.md`  

The offline plan remains the correct design for the Lean trust plane and for an optional fully
disconnected profile. It is **not** the network policy for this hybrid profile.

## 1. Target architecture

```text
                                 connected, untrusted research plane
                    +------------------------------------------------------+
                    | arXiv MCP      scholarly evidence MCP     refcheck  |
                    | arXiv/S2       ERIC/OpenAlex/Crossref      DOI/cites |
                    | optional PubMed/PMC/Europe PMC/Unpaywall             |
                    +--------------------------+---------------------------+
                                               |
                                               | normalized metadata,
                                               | quotations, provenance
                                               v
+---------+       candidate Lean       +-------+--------+
| Hermes  |--------------------------->| local `axle`  |
| Agent   |<---------------------------| MCP process   |
+---------+     structured result      +-------+--------+
                                               |
                                               | source strings only
                                               v
                    +------------------------------------------------------+
                    | fresh Lean verifier container                        |
                    | --network none, read-only root, tmpfs, no caps       |
                    | pinned Lean/Mathlib -> separate compile -> SafeVerify|
                    | immutable spec match -> kernel replay -> axiom check |
                    +------------------------------------------------------+
```

The research processes may access only approved scholarly services. The Docker proof worker has
no network. Retrieved prose may help formulate a specification or proof candidate, but cannot
change the verifier policy or turn an informal claim into a verified Lean theorem.

## 2. Implemented baseline

### Local trust plane

The existing `lean-verifier-mcp` remains the final checker. In the hybrid Hermes profile it is
registered under the server key `axle`, preserving `mcp_axle_*` tool names while local
AXLE-compatible proof-engineering tools are developed.

Current guarantees remain:

- fixed or immutable local specifications;
- pinned Lean 4.27.0, Mathlib, and vendored SafeVerify environment;
- separate specification and candidate compilation;
- SafeVerify replay in fresh Lean environments;
- declaration name, kind, universe, and type matching;
- kernel checking of candidate values;
- only `propext`, `Quot.sound`, and `Classical.choice` allowed;
- rejection of `sorryAx`, custom axioms, unsafe/partial declarations, and dangerous source forms;
- a new Docker container per request with `--network none` and resource limits.

### Connected scholarly evidence gateway

This repository now installs a separate `research-evidence-mcp` entry point. It exposes:

| Tool | Official upstream | Purpose |
|---|---|---|
| `search_eric` | IES ERIC API | Education research, optional ERIC `peerreviewed:T`, publication-type filtering |
| `lookup_eric` | IES ERIC API | Resolve one stable ED/EJ identifier; check DOI integrity when a DOI is present |
| `search_openalex` | OpenAlex API | Broad scholarly discovery with OA and OpenAlex retraction fields |
| `lookup_doi` | Crossref + Crossref updates + OpenAlex | DOI consistency plus correction/retraction evidence |
| `make_claim_evidence` | local only | Preserve a claim, exact quotation, locator, version/status fields, and SHA-256 hashes |

The gateway has no arbitrary-URL fetch tool. Its network code contains fixed official API base
URLs. It normalizes status fields instead of returning a misleading `verified: true` Boolean.

Important limitations are explicit in every relevant result:

- ERIC's `peerreviewed` field is bibliographic metadata, not proof that a conclusion is true.
- A Crossref DOI confirms deposited metadata, not peer review or claim correctness.
- OpenAlex work/venue types are not used as a proxy for peer review.
- “No retraction found” means no flag was found in the checked sources at lookup time; it is not
  a guarantee that no concern exists.
- `make_claim_evidence` records and hashes a supplied quotation but does not independently prove
  that the quote occurs in the document.
- None of these tools returns a formal-verification result.

### Dedicated community MCPs in the example profile

`hermes-hybrid-research-config.example.yaml` also includes exact-version examples for:

| Hermes key | Package/version reviewed 2026-08-18 | Role | Default |
|---|---|---|---|
| `arxiv` | `arxiv-mcp-server==0.6.3` | arXiv search, bounded full text/LaTeX, citation graph | enabled |
| `refcheck` | `refcheck-mcp==0.1.0` | bibliographic matching through Crossref, Semantic Scholar, arXiv | enabled |
| `pubmed` | `@cyanheads/pubmed-mcp-server@2.10.2` | PubMed/PMC/Europe PMC and lawful OA retrieval | disabled/optional |
| `lean_lsp` | local Lean LSP/Loogle | proof states and local search | disabled/optional |

These are community packages, not formally trusted proof components. Version pinning is a
starting control, not a security certification. Review and lock their source and transitive
dependencies before production deployment.

## 3. Source policy

### Tier 1 — canonical metadata and integrity

Prefer these to establish identity and status:

- Crossref for DOI registration, publisher metadata, relations, corrections, and retractions;
- OpenAlex for cross-database IDs, versions/locations, citation graphs, OA metadata, and
  `is_retracted`;
- ERIC for education-specific metadata and its explicit `peerreviewed` field;
- PubMed for PMID/MeSH/publication type and PMC/Europe PMC for lawful biomedical full text.

### Tier 2 — curated repositories and legal full text

- arXiv for versioned preprints and author-submitted sources;
- PMC and Europe PMC for biomedical full text;
- Unpaywall, CORE, DOAJ, OpenAIRE, Zenodo, HAL, and publisher OA URLs when added through reviewed
  connectors.

### Tier 3 — discovery only

Semantic Scholar recommendations and citation graphs may find candidates. Reconcile stable IDs
and metadata with a canonical source before citing consequential claims.

### Excluded from the trusted profile

- Sci-Hub and other copyright-bypassing services;
- arbitrary PDF URL download tools;
- unrestricted browser/web-scraping MCPs;
- Google Scholar scraping as a canonical metadata source;
- ResearchGate scraping or automated use that violates access terms;
- model-generated citations that were not resolved against an external record.

## 4. “Verified source” is not one Boolean

Every research output should keep these dimensions separate:

```yaml
stable_identifiers:
  doi: "10.x/..."       # or arXiv/ERIC/PMID/PMCID/OpenAlex ID
provenance:
  database: "Crossref"
  record_url: "https://doi.org/..."
  retrieved_at: "2026-08-18T...Z"
document_version: "arXiv v2 / accepted manuscript / version of record / unknown"
publication_type: "preprint / journal-article / report / systematic review / ..."
peer_review:
  status: "confirmed_by_explicit_metadata / unknown / not_established"
  evidence: "ERIC peerreviewed:T / publisher policy URL / none"
open_access:
  status: "gold / green / hybrid / bronze / closed / unknown"
integrity:
  retraction_status: "retracted / not_flagged_at_lookup_time / not_checked"
  correction_status: "correction found / none found / not checked"
claim_evidence:
  claim: "the proposition being supported"
  exact_quote: "verbatim text"
  locator: "page, section, theorem, table, or paragraph"
  quote_hash: "sha256:..."
formal_verification_status: "not a Lean verification result"
```

Recommended language:

- **bibliographically matched** — identifier/title/author metadata agree;
- **peer review explicitly reported** — a source supplies a direct field or policy;
- **not flagged as retracted at lookup time** — integrity sources had no flag;
- **formally verified** — reserved exclusively for a successful local Lean/SafeVerify result.

Never call an arXiv paper peer-reviewed solely because it has an arXiv record. arXiv provides
moderation, authenticity, timestamps, and versions; it does not provide journal peer review.

## 5. Required workflow in Hermes

1. **Discover** through `mcp_arxiv_search_papers`, `mcp_scholar_search_eric`, or
   `mcp_scholar_search_openalex`.
2. **Resolve identity** using a DOI, arXiv ID/version, ERIC ID, PMID/PMCID, or OpenAlex ID.
3. **Cross-check bibliography** with `mcp_refcheck_verify_reference` or
   `mcp_scholar_lookup_doi`.
4. **Check integrity** for every DOI used in a consequential claim. Preserve the lookup time and
   all correction/retraction warnings.
5. **Retrieve full text lawfully** from arXiv, ERIC-provided links, PMC/Europe PMC, Unpaywall, or
   a publisher OA URL. Never silently substitute a different document version.
6. **Capture claim evidence**: exact quote, page/section/theorem/table locator, identifier,
   document version, publication type, peer-review evidence, OA status, and integrity status.
   Use `mcp_scholar_make_claim_evidence` to produce a hashed record.
7. **Review the informal-to-formal translation**. A human or trusted specification process must
   confirm that the Lean statement faithfully represents the intended claim.
8. **Develop a candidate proof** with local Lean diagnostics and future local AXLE-compatible
   extraction/transformation/repair/search tools.
9. **Verify only against the immutable local specification** with
   `mcp_axle_verify_against_spec`. Dynamic `verify_proof` is exploratory, not equivalent to an
   immutable challenge.
10. **Accept only the local result** when compilation, declaration comparison, kernel replay,
    and axiom policy all pass.

## 6. Prompt-injection and tool-isolation policy

Scholarly text is untrusted content. A PDF, HTML page, abstract, or LaTeX source can contain text
that looks like agent instructions.

- Mark every connected research MCP `trust: untrusted` in Hermes.
- Disable MCP resources and prompts unless specifically reviewed; use tools only.
- Do not grant research MCP processes shell, messaging, credential-management, or unrestricted
  filesystem tools.
- Never execute code, scripts, macros, links, or tool instructions found in a document.
- Cache downloads as non-executable data under a dedicated research directory.
- Keep credentials in the active Hermes profile `.env`, not YAML, prompts, or paper caches.
- Do not pass secrets to `arxiv` or other servers that do not require them.
- Research text may become comments or proof-planning notes, but only Lean source reaches the
  verifier request, and the verifier worker remains networkless.

## 7. Network and deployment controls

The example configuration separates processes but does not by itself implement host firewall
rules. For a hardened deployment:

1. Run research MCPs under a dedicated unprivileged OS account or in separate rootless
   containers.
2. Permit DNS/TLS egress only to reviewed hosts needed by enabled tools, such as:
   `api.ies.ed.gov`, `api.openalex.org`, `api.crossref.org`, `arxiv.org`,
   `export.arxiv.org`, `api.semanticscholar.org`, NCBI E-utilities/PMC, Europe PMC, and
   `api.unpaywall.org`.
3. Deny private-address, metadata-service, and loopback destinations from the research sandbox
   unless a specific local service is required.
4. Set response-size, result-count, redirect, and time limits. The in-repository gateway limits
   search calls to 25 records, uses an HTTPS-only fixed API set, and does not follow redirects.
5. Pin source commits or package versions and lock hashes. Mirror approved artifacts internally
   if reproducibility matters.
6. Keep the Lean container's `--network none` control; do not rely only on application policy.

## 8. Installation and activation

Build and install the repository environment:

```bash
cd /absolute/path/to/lean-local-verifier
uv sync
```

Build the Lean verifier image as described in `README.md`, then prepare a research cache:

```bash
mkdir -p /absolute/path/to/research-cache/arxiv
```

Find absolute executable paths:

```bash
command -v uvx
command -v npx       # needed only if the optional PubMed MCP is enabled
hermes config path
hermes config env-path
```

Put a real contact address in the active Hermes profile `.env`:

```dotenv
SCHOLAR_CONTACT_EMAIL=researcher@example.org
```

Merge `hermes-hybrid-research-config.example.yaml` into the active top-level Hermes config,
replace placeholders, then test:

```bash
hermes mcp test axle
hermes mcp test scholar
hermes mcp test arxiv
hermes mcp test refcheck
hermes mcp list
```

If biomedical tools are needed, install Node.js 24 or newer, set `pubmed.enabled: true`, and run:

```bash
hermes mcp test pubmed
```

Start a fresh Hermes session after changing MCP registrations.

## 9. Validation performed in this workspace

On 2026-08-18:

- all 15 unit tests passed, including six verifier-policy tests, two runner tests, and seven
  evidence-normalization/integrity tests;
- Python compilation succeeded;
- live calls to the official ERIC, OpenAlex, and Crossref APIs succeeded through the new client;
- a known retracted DOI was identified as retracted through Crossref update data/OpenAlex even
  when one concurrent Crossref request was rate-limited;
- Docker is unavailable in this workspace, so no claim is made that the container image or
  end-to-end Hermes/Docker runtime was tested here.

## 10. Next proof-engineering work

The stable Hermes server key `axle` is intentionally reserved for local tools. Add capabilities
incrementally behind that endpoint:

1. check/elaborate snippets and return structured diagnostics;
2. extract declarations and proof states;
3. semantic source transformations;
4. repair and simplification;
5. local theorem search and premise selection;
6. bounded proof search;
7. strict final handoff to the existing immutable-spec/SafeVerify path.

Each new tool should have network disabled, pinned Lean dependencies, explicit time/memory limits,
and adversarial parity tests. AXLE API parity remains a development target, not a current claim.
