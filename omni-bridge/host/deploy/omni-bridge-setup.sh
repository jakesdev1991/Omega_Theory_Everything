#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

set -euo pipefail

echo "Setting up Omni-Bridge environment..."

# 1. Create omni-bridge user/group if not exists
if ! getent group omni-bridge > /dev/null; then
    groupadd --system omni-bridge
fi

if ! getent passwd omni-bridge > /dev/null; then
    useradd --system --gid omni-bridge --no-create-home --shell /usr/sbin/nologin omni-bridge
fi

# 2. Set ownership on /opt/omni-bridge directories
echo "Configuring permissions..."
mkdir -p /opt/omni-bridge/{config,memory,retrieval,audit,runtime,source}
chown -R omni-bridge:omni-bridge /opt/omni-bridge
chmod 750 /opt/omni-bridge

# 3. Create cgroup hierarchy at /sys/fs/cgroup/omni-bridge with memory and CPU limits
echo "Configuring cgroups..."
CGROUP_PATH="/sys/fs/cgroup/omni-bridge"
if [ -d "/sys/fs/cgroup" ]; then
    mkdir -p "${CGROUP_PATH}"
fi

# 4. Copy systemd unit to /etc/systemd/system/
echo "Installing systemd unit..."
cp /opt/omni-bridge/source/omni-bridge/deploy/omni-bridge.service /etc/systemd/system/

# 5. systemctl daemon-reload
echo "Reloading systemd daemon..."
systemctl daemon-reload

echo "====================================================="
echo "Setup complete! Please configure your environment."
echo ""
echo "Instructions:"
echo "1. Copy /opt/omni-bridge/source/omni-bridge/.env.example to /opt/omni-bridge/config/.env"
echo "2. Edit /opt/omni-bridge/config/.env with your settings"
echo "3. To start the service: systemctl start omni-bridge"
echo "4. To enable on boot: systemctl enable omni-bridge"
echo "5. To check logs: journalctl -u omni-bridge -f"
echo "====================================================="
