#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

# build.sh - Reproducible build for all Omni-Bridge components
# Per Section 47 of agentic_instruction.txt

set -euo pipefail

PROJECT_ROOT="/opt/omni-bridge"
SOURCE_ROOT="$PROJECT_ROOT/source"

echo "=========================================="
echo "Omni-Bridge Build"
echo "=========================================="

BUILD_MODE="${1:-Release}"

# 1. Build Hermes
echo "[1/4] Building Hermes (C++)..."
cd "$SOURCE_ROOT/hermes"
cmake -B build -DCMAKE_BUILD_TYPE="$BUILD_MODE"
cmake --build build --parallel
echo "  ✓ Hermes built"

# 2. Build Omni-Bridge C++
echo "[2/4] Building Omni-Bridge C++..."
cd "$SOURCE_ROOT/omni-bridge/cpp"
cmake -B build -DCMAKE_BUILD_TYPE="$BUILD_MODE"
cmake --build build --parallel
echo "  ✓ Omni-Bridge C++ built"

# 3. Verify Python prototypes (no build needed)
echo "[3/4] Verifying Python prototypes..."
cd "$SOURCE_ROOT/omni-bridge/offline_python_prototypes"
# Check if packages are available (Gentoo externally managed - don't pip install)
python3 -c "import blake3; import wasmtime; import numpy; import sqlite3" 2>/dev/null || echo "  ⚠ Some Python packages missing (install via emerge)"
python3 -m pytest -q --tb=short
echo "  ✓ Python prototypes verified"

# 4. Build Lucifer C++ (if dependencies available)
echo "[4/4] Checking Lucifer C++ build..."
if [[ -f "$SOURCE_ROOT/lucifer/CMakeLists.txt" ]]; then
    if cmake -B "$SOURCE_ROOT/lucifer/build" -DCMAKE_BUILD_TYPE="$BUILD_MODE" "$SOURCE_ROOT/lucifer" 2>&1 | grep -q "Configuring done"; then
        cmake --build "$SOURCE_ROOT/lucifer/build" --parallel
        echo "  ✓ Lucifer C++ built"
    else
        echo "  ⚠ Lucifer C++ dependencies missing (spdlog, ftxui, grpc, protobuf, pybind11) - skipping"
    fi
else
    echo "  ⚠ No Lucifer CMakeLists.txt - skipping"
fi

echo ""
echo "=========================================="
echo "Build complete"
echo "=========================================="