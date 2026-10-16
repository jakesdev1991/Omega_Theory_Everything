<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->
# Omega MCP Hub — in-memory prototype

MCP server for exploring a prototype of the Omega tri-token economy concepts. It is not the
operating hub for a deployed economy and is not connected to the contracts or other code in the
blueprint at [`tri_token_sovereign_economy_blueprint.md`](../tri_token_sovereign_economy_blueprint.md).

## Planes

| Plane | Role | Simulation state tracked? |
|-------|------|------------|
| **SOV** | Accounting and settlement unit for budgets, invoices, grants, internal accounting. | Yes, in memory — balances and simulated mint/burn/transfer events |
| **USE** | Non-transferable proof-of-useful-work contribution receipts. | Yes, in memory — receipts, attestation counts, revocation state |
| **CARE** | Stewardship, governance, protected care coordination. | Yes, in memory — claims, attestations, appeals |
| **AMITY** | Exchange-eligible representation of approved care value, compliance-gated. | Yes, in memory — balances, supply, entitlements |
| **OMEGA** | Scarce macro-governance asset for parameter changes, treasury, lock positions. | Yes, in memory — balances, supply, locks, proposals, votes |

## Status

This hub is an **in-memory simulation** for prototyping and agent integration. It does **not**
connect to a real chain, hold real keys, or represent real value. State-changing operations emit
versioned audit events, and snapshots expose the current state. Event payloads are not yet a
complete replay log, so consumers must not treat the event stream as sufficient to rebuild every
snapshot field. Streamable HTTP has no built-in authentication and binds to loopback by default;
do not expose it to untrusted networks without a separately secured boundary.

The append-only event discipline is intended to inform later on-chain or custody-backed designs;
that future architecture is not implemented here.

## Quick start

```bash
# From the repo root:
uv run --directory mcp python -m omega_mcp
```

Or against the venv directly (substitute your own venv path):

```bash
"$OMEGA_MCP_VENV"/bin/python -m omega_mcp
```

Default transport is stdio. For Streamable HTTP on the local machine:

```bash
uv run --directory mcp python -m omega_mcp --transport streamable-http --host 127.0.0.1 --port 8029
```

The HTTP server binds to loopback by default. Use `--host 0.0.0.0` only when you intend to
expose it on a network and have added the appropriate firewall and authentication controls.

## MCP client config (Hermes Agent)

Add to `~/.hermes/config.yaml`, replacing the `--directory` value with the
absolute path to `mcp/` in your own checkout:

```yaml
mcp_servers:
  omega-hub:
    command: "uv"
    args:
      - "run"
      - "--directory"
      - "/absolute/path/to/Omega_Theory_Everything/mcp"
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
- **House:** `ledger`, `events_since`, `full_event_log` (inspection and audit; not full state replay)

## Architecture notes

- `omega_mcp/__init__.py` owns the in-memory `OmegaState` model and all documented MCP tools.
- State is split into five plane namespaces inside `OmegaState` but shares one versioned
  event log and snapshot mechanism.
- `server.json` is the MCP manifest describing the server to clients and marketplaces.
- `omega_mcp/__main__.py` is the CLI entry point; `pyproject.toml` makes it installable as
  `omega-hub`.

## Connection to the rest of the repo

- The in-memory hub is a prototype of the ledger concepts described in the blueprint; it is not
  connected to the Rust workspace or smart contracts.
- Versioned event metadata and current-state snapshots provide a test surface. Full event-sourced
  reconstruction and production controls remain future work.
- For the agentic layer around this hub, see
  [`whitepapers/lucifer_hermes_omni_bridge_whitepaper.md](../whitepapers/lucifer_hermes_omni_bridge_whitepaper.md).

## License

PolyForm Noncommercial 1.0.0 — see [`LICENSES/PolyForm-Noncommercial-1.0.0.txt`](../LICENSES/PolyForm-Noncommercial-1.0.0.txt) and the path-specific scope in [`docs/LICENSING.md`](../docs/LICENSING.md).
