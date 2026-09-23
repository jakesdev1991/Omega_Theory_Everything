# Omega MCP Hub — agent integration wiring

This directory wires the Omega tri-token economy into an MCP server that any MCP client can
call as a set of tools.

## What lives here

- `omega_mcp/__init__.py` — server implementation: `OmegaState` ledger + 22 MCP tools.
- `omega_mcp/__main__.py` — CLI entry point for `uv run -m omega_mcp` and `omega-hub`.
- `server.json` — MCP manifest describing the server and its tools.
- `pyproject.toml` — installable Python package (`omega-mcp-hub`).
- `README_MCP_HUB.md` — hub docs.

## Top-level layout

After this lands, the repo root will carry:

```text
Omega_Theory_Everything/
├── mcp/                       # NEW: MCP hub package
│   ├── omega_mcp/
│   │   ├── __init__.py
│   │   └── __main__.py
│   ├── pyproject.toml
│   ├── server.json
│   └── README_MCP_HUB.md
├── app/                       # existing local social prototype (offline, valueless)
├── lean_proofs/               # Lean 4 formalizations
├── whitepapers/               # $OMEGA, TOKAMAK, C.A.R.E./AMITY, Omni-Bridge
├── tri_token_sovereign_economy_blueprint.md
└── ...
```

## Development

```bash
# Install in editable mode into the omwga-mcp venv
uv pip install --python /home/jake/.venvs/omwga-mcp/bin/python -e mcp/

# Run the stdio server
uv run --directory mcp python -m omega_mcp

# Build a wheel
uv build mcp/
```

## Notes

- The hub deliberately does not touch real keys, real chains, or real value in this form.
- The event log is the invariant: every tool that changes state emits an append-only event
  with a version number. Consumers can replay from any version via `events_since`.
- This is the layer the Lucifer/Hermes agentic stack would call when it needs to reason about
  the economy in a grounded, auditable way rather than through ad-hoc state in prompts.
