#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run python -m unittest discover -s tests -v
uv run lean-verify health
uv run lean-verify verify \
  --spec examples/specs/add_comm.lean \
  --candidate examples/candidates/add_comm_good.lean

if uv run lean-verify verify \
  --spec examples/specs/add_comm.lean \
  --candidate examples/candidates/add_comm_bad_statement.lean; then
  echo "ERROR: changed theorem statement was accepted" >&2
  exit 1
fi

if uv run lean-verify verify \
  --spec examples/specs/add_comm.lean \
  --candidate examples/candidates/add_comm_bad_sorry.lean; then
  echo "ERROR: sorry was accepted" >&2
  exit 1
fi

echo "All expected checks passed."
