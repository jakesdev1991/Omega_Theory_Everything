#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

command -v docker >/dev/null || { echo "Docker is required" >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

docker build --tag local-lean-verifier:4.27.0 .
uv sync
uv run lean-verify health

echo
echo "Setup complete. Try:"
echo "  uv run lean-verify verify --spec examples/specs/add_comm.lean --candidate examples/candidates/add_comm_good.lean"
