#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Build and smoke-run the AI governor (Proof-of-Useful-Work trainer).
#
# Zero external dependencies: FIPS 180-4 SHA-256 for work-receipt digests is
# implemented in-header (`include/omega/omega_governor.hpp`), matching
# `omni-bridge/cpp/omni_bridge_core.hpp`.
#
# Usage: ./build.sh [--quick]
#   default : strict build, then a 300-step governed+baseline run (seconds)
#   --quick : build only
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

CXX=${CXX:-g++}
STD=${STD:--std=c++23}
OUT=${OUT:-/tmp/omega_governor_train}
LDLIBS=${LDLIBS:-}
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

echo "=== strict build (-Wall -Wextra -Werror -O2) ==="
"$CXX" $STD -O2 -Wall -Wextra -Werror -Iinclude governor_train.cpp $LDLIBS -o "$OUT"
echo "built $OUT"

if [ "$QUICK" -eq 1 ]; then
  echo "=== build only (--quick) ==="
  exit 0
fi

# The trainer takes the corpus glob and a step count.  Run from the repository
# root so the glob resolves; 300 steps is a smoke test, not a result.
cd "$REPO_ROOT"
echo
echo "=== smoke run (300 steps, governed + baseline) ==="
output=$("$OUT" "lean_proofs/*.lean" 300)
echo "$output" | tail -12
echo "$output" | grep -q "governed receipts: [1-9]" || { echo "!! no governed receipts produced" >&2; exit 1; }
echo "$output" | grep -q "baseline final receipt"    || { echo "!! no baseline receipt produced" >&2; exit 1; }
echo "ALL CHECKS PASSED"
