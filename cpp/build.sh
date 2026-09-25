#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Build and run the CBwK shadow-pacer validation suite:
#   1. strict -Wall -Wextra -Werror build, run
#   2. ThreadSanitizer build, run        (validates the race-freedom claim)
#   3. AddressSanitizer + UBSan, run     (validates memory safety)
#   4. forced expected-fallback build    (validates the pre-C++23 path)
#
# Usage: ./build.sh [--quick]   (--quick skips sanitizers)

set -euo pipefail
cd "$(dirname "$0")"

CXX=${CXX:-g++}
STD=${STD:--std=c++23}
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

run_one() {
  local name="$1"; shift
  echo "=== $name ==="
  "$CXX" $STD "$@" -pthread test_cbwk_shadow_pacer.cpp -o /tmp/test_pacer_bin
  /tmp/test_pacer_bin
  echo
}

run_one "strict build (-Wall -Wextra -Werror -O2)" \
  -O2 -Wall -Wextra -Werror

if [ "$QUICK" -ne 1 ]; then
  run_one "ThreadSanitizer" -O1 -g -fsanitize=thread -fno-omit-frame-pointer
  run_one "AddressSanitizer + UBSan" -O1 -g \
    -fsanitize=address,undefined -fno-omit-frame-pointer
  run_one "expected-fallback (pre-C++23 toolchain path)" \
    -O2 -Wall -Wextra -Werror -DPAC_NO_STD_EXPECTED
fi

echo "ALL BUILDS AND RUNS PASSED"
