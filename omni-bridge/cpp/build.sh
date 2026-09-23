#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
#
# Build and run the Omni-Bridge control-boundary acceptance suite:
#   1. strict -Wall -Wextra -Werror build, run
#   2. ThreadSanitizer build, run
#   3. AddressSanitizer + UBSan build, run
#
# Usage: ./build.sh [--quick]   (--quick = strict build + run only)

set -euo pipefail
cd "$(dirname "$0")"

CXX=${CXX:-g++}
STD=${STD:--std=c++23}
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

run_one() {
  local name="$1"; shift
  echo "=== $name ==="
  "$CXX" $STD "$@" test_omni_bridge.cpp -o /tmp/test_omni_bin
  /tmp/test_omni_bin
  echo
}

run_one "strict build (-Wall -Wextra -Werror -O2)" \
  -O2 -Wall -Wextra -Werror

if [ "$QUICK" -ne 1 ]; then
  run_one "ThreadSanitizer" -O1 -g -fsanitize=thread -fno-omit-frame-pointer
  run_one "AddressSanitizer + UBSan" -O1 -g \
    -fsanitize=address,undefined -fno-omit-frame-pointer
fi

echo "ALL BUILDS AND RUNS PASSED"
