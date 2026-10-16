#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Build + run the CorePolicyAdapter unification test. The vendored core
# copy (namespace omni_core) sits in vendor/ — identical to the repo's
# canonical omni-bridge/cpp/omni_bridge_core.hpp except for the namespace
# rename, which avoids colliding with the daemon's own omni::bridge types.
set -euo pipefail
HOST="$(cd "$(dirname "$0")" && pwd)"
OUT="/tmp/test_core_policy_adapter"

g++ -std=c++23 -O2 -Wall -Wextra \
    -I"$HOST/include" \
    "$HOST/tests/test_core_policy_adapter.cpp" \
    -o "$OUT"

"$OUT"
