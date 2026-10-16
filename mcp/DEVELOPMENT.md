<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->
# Omega MCP Hub — agent integration wiring

This directory contains an MCP prototype that exposes simulated Omega tri-token economy
operations to compatible clients. All balances and claims live only in process memory.

## What lives here

- `omega_mcp/__init__.py` — server implementation: `OmegaState` ledger + 22 MCP tools.
- `omega_mcp/__main__.py` — CLI entry point for `uv run -m omega_mcp` and `omega-hub`.
- `server.json` — MCP manifest describing the server and its tools.
- `pyproject.toml` — installable Python package (`omega-mcp-hub`).
- `README_MCP_HUB.md` — hub docs.

## Top-level layout

The current repository layout includes:

```text
Omega_Theory_Everything/
├── mcp/                       # MCP simulation prototype package
│   ├── omega_mcp/
│   │   ├── __init__.py
│   │   └── __main__.py
│   ├── pyproject.toml
│   ├── server.json
│   └── README_MCP_HUB.md
├── app/                       # standalone wallet GUI source
├── lean_proofs/               # Lean 4 formalizations
├── whitepapers/               # $OMEGA, TWC, TOKAMAK (historical draft), C.A.R.E./AMITY, Omni-Bridge
├── tri_token_sovereign_economy_blueprint.md
└── ...
```

## Development

```bash
# Install in editable mode into your hub venv (create it first, e.g.
#   uv venv "$HOME/.venvs/omwga-mcp" --python 3.12
# — mcp/pyproject.toml requires Python >=3.12).
uv pip install --python "$OMEGA_MCP_VENV/bin/python" -e mcp/

# Run the stdio server
uv run --directory mcp python -m omega_mcp

# Optional Streamable HTTP on loopback
uv run --directory mcp python -m omega_mcp --transport streamable-http --host 127.0.0.1 --port 8029

# Build a wheel
uv build mcp/
```

## Notes

- The hub deliberately does not touch real keys, real chains, or real value in this form.
- Every state-changing tool emits a versioned append-only audit event. `events_since` supports
  inspection, but the payloads are not complete enough to reconstruct every snapshot field; use
  snapshots as the current-state view.
- This prototype is a local integration/test surface for an eventual agent stack; it is not an
  independent verification layer or a production operating service.
