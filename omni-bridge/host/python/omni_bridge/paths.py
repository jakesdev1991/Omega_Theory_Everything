# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Omni-Bridge canonical IPC socket paths.

Sockets live under /opt/omni-bridge/runtime/ (the @runtime Btrfs subvolume),
NOT /tmp. Reason: the systemd unit runs with PrivateTmp=true, which gives the
service a private /tmp namespace — sockets created there are unreachable from
outside the unit. The @runtime subvolume is the designed home for runtime
state (§6 subvolume architecture) and is shared across all contexts.
"""

RUNTIME_DIR = "/opt/omni-bridge/runtime"

BRIDGE_SOCKET = f"{RUNTIME_DIR}/omni_bridge.sock"
HERMES_SOCKET = f"{RUNTIME_DIR}/hermes.sock"
LUCIFER_SOCKET = f"{RUNTIME_DIR}/lucifer.sock"
