# Omega MCP Hub

Sovereign MCP server for the Omega tri-token economy. It is the main operating hub for the
three-token model described in
[`tri_token_sovereign_economy_blueprint.md`](../tri_token_sovereign_economy_blueprint.md).

## Planes

| Plane | Role | On-ledger? |
|-------|------|------------|
| **SOV** | Accounting and settlement unit for budgets, invoices, grants, internal accounting. | Yes — balances, mint/burn/transfer events |
| **USE** | Non-transferable proof-of-useful-work contribution receipts. | Yes — receipts keyed by ID, attestation counts, revocation state |
| **CARE** | Stewardship, governance, protected care coordination. | Yes — claims, attestations, appeals |
| **AMITY** | Exchange-eligible representation of approved care value, compliance-gated. | Yes — balances, supply, entitlements |
| **OMEGA** | Scarce macro-governance asset for parameter changes, treasury, lock positions. | Yes — balances, supply, locks, proposals, votes |

## Status

This hub is a **deterministic in-memory simulation** for prototyping and agent integration. It
does **not** connect to a real chain, hold real keys, or represent real value. Every state
mutation is recorded in an append-only event log so the ledger is replayable and auditable.

The same event-log discipline is intended to carry into any later on-chain or custody-backed
deployment, where the hub would become a scoped NWC / agent access layer over real settlement.

## Quick start

```bash
# From the repo root:
uv run --directory mcp python -m omega_mcp
```

Or against the venv directly:

```bash
/home/jake/.venvs/omwga-mcp/bin/python -m omega_mcp
```

Default transport is stdio. For HTTP mode:

```bash
uv run --directory mcp python -m omega_mcp --transport http --port 8029
```

## MCP client config (Hermes Agent)

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  omega-hub:
    command: "uv"
    args:
      - "run"
      - "--directory"
      - "/home/jake/Omega_Theory_Everything/mcp"
      - "python"
      - "-m"
      - "omega_mcp"
```

Restart Hermes Agent and the Omega tools appear as `mcp_omega_hub_*`.

## Tool map

- **SOV:** `sov_mint`, `sov_burn`, `sov_transfer`, `sov_snapshot`
- **USE:** `use_issue`, `use_revoke`, `use_snapshot`
- **CARE:** `care_submit`, `care_attest`, `care_appeal`, `care_snapshot`
- **AMITY:** `amity_entitle`, `amity_transfer`, `amity_snapshot`
- **OMEGA:** `omega_mint`, `omega_lock`, `omega_proposal`, `omega_vote`, `omega_snapshot`
- **House:** `ledger`, `events_since`, `full_event_log`

## Architecture notes

- `omega_mcp/__init__.py` owns the `OmegaState` ledger and all `@mcp.tool()` definitions.
- State is split into five plane namespaces inside `OmegaState` but shares one versioned
  event log and snapshot mechanism.
- `server.json` is the MCP manifest describing the server to clients and marketplaces.
- `omega_mcp/__main__.py` is the CLI entry point; `pyproject.toml` makes it installable as
  `omega-hub`.

## Connection to the rest of the repo

- Deterministic state machine: the hub is the runtime expression of the ledger modules called
  for in the blueprint's Rust workspace spec and smart-contract spec.
- Event log is the same append-only design the blueprint's acceptance criteria require before
  any real deployment.
- For the agentic layer around this hub, see
  [`whitepapers/lucifer_hermes_omni_bridge_whitepaper.md](../whitepapers/lucifer_hermes_omni_bridge_whitepaper.md`).

## License

MIT — see [`LICENSE`](../LICENSE).
