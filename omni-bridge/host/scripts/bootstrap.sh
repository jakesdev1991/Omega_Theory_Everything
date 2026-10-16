#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; CRITICAL_FAIL=1; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

CRITICAL_FAIL=0

echo "Running bootstrap checks..."

# OS check
if grep -q "Gentoo" /etc/os-release 2>/dev/null; then
    pass "OS: Gentoo Linux"
else
    fail "OS: Gentoo Linux not detected"
fi

# Kernel check
KERNEL_VER=$(uname -r)
pass "Kernel: $KERNEL_VER"

# Toolchain check
for cmd in gcc clang cmake rustc cargo python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        pass "Toolchain: $cmd found"
    else
        fail "Toolchain: $cmd missing"
    fi
done

# Check GCC version (>=15)
if command -v gcc >/dev/null 2>&1; then
    GCC_VER=$(gcc -dumpversion | cut -d. -f1)
    if [ "$GCC_VER" -ge 15 ]; then pass "gcc version $GCC_VER (>=15)"; else fail "gcc version $GCC_VER (<15)"; fi
fi

# Check Clang version (>=22)
if command -v clang >/dev/null 2>&1; then
    CLANG_VER=$(clang -dumpversion | cut -d. -f1)
    if [ "$CLANG_VER" -ge 22 ]; then pass "clang version $CLANG_VER (>=22)"; else fail "clang version $CLANG_VER (<22)"; fi
fi

# Check CMake version (>=4)
if command -v cmake >/dev/null 2>&1; then
    CMAKE_VER=$(cmake --version | head -n1 | grep -Eo "[0-9]+\.[0-9]+" | head -n1 | cut -d. -f1)
    if [ "$CMAKE_VER" -ge 4 ]; then pass "cmake version $CMAKE_VER (>=4)"; else fail "cmake version $CMAKE_VER (<4)"; fi
fi

# Check Rustc version (>=1.97)
if command -v rustc >/dev/null 2>&1; then
    RUSTC_VER=$(rustc --version | awk '{print $2}' | cut -d. -f2)
    if [ "$RUSTC_VER" -ge 97 ]; then pass "rustc minor version $RUSTC_VER (>=97)"; else fail "rustc version < 1.97"; fi
fi

# Python version (>=3.13)
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' | cut -d. -f2)
    if [ "$PYTHON_VER" -ge 13 ]; then pass "python3 minor version $PYTHON_VER (>=13)"; else fail "python3 version < 3.13"; fi
fi

# Libraries check (basic binary check or pkg-config)
for lib in sqlite3 curl; do
    if command -v "$lib" >/dev/null 2>&1; then pass "Library/Tool: $lib found"; else fail "Library/Tool: $lib missing"; fi
done

# btrfs-progs
if command -v btrfs >/dev/null 2>&1; then pass "btrfs-progs installed"; else fail "btrfs-progs missing"; fi

# btrfs mount check
if mount -l -t btrfs | grep -q "/opt/omni-bridge "; then pass "/opt/omni-bridge is mounted as btrfs"; else fail "/opt/omni-bridge not mounted as btrfs"; fi

# 13 subvolumes mounted (use findmnt, works without root)
SUBVOL_COUNT=$(findmnt -t btrfs --noheadings | grep -c "omni-bridge" 2>/dev/null || echo 0)
if [ "$SUBVOL_COUNT" -ge 13 ]; then
    pass "All $SUBVOL_COUNT Btrfs subvolume mounts present"
else
    fail "Only $SUBVOL_COUNT of 13 expected Btrfs subvolume mounts found"
fi

# cgroups v2
if grep -q cgroup2 /proc/filesystems && mount | grep -q cgroup2; then pass "cgroups v2 available"; else fail "cgroups v2 not available"; fi

# systemd
if command -v systemctl >/dev/null 2>&1; then pass "systemd present"; else fail "systemd missing"; fi

# Qdrant reachable (optional)
if curl -s http://localhost:6333 >/dev/null 2>&1; then
    pass "Qdrant server reachable"
else
    warn "Qdrant server not reachable"
fi

if [ $CRITICAL_FAIL -eq 1 ]; then
    echo -e "${RED}Bootstrap checks failed!${NC}"
    exit 1
else
    echo -e "${GREEN}Bootstrap checks passed!${NC}"
    exit 0
fi