#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

set -euo pipefail

echo "Starting Omni-Bridge..."

# Source config from /opt/omni-bridge/config/.env if exists
if [ -f "/opt/omni-bridge/config/.env" ]; then
    echo "Sourcing environment from /opt/omni-bridge/config/.env"
    set -a
    source "/opt/omni-bridge/config/.env"
    set +a
fi

# Ensure runtime directory exists
mkdir -p /opt/omni-bridge/runtime
PID_FILE="/opt/omni-bridge/runtime/bridge.pid"
echo $$ > "$PID_FILE"

# Track children
PIDS=()

# Cleanup function on signal
cleanup() {
    echo "Received signal. Stopping Omni-Bridge components..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" || true
        fi
    done
    wait "${PIDS[@]}" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "Omni-Bridge stopped cleanly."
    exit 0
}
trap cleanup SIGTERM SIGINT

# Check kill switches before starting components (normalized §72 semantics)
START_LUCIFER=true
case "${OMNI_LUCIFER_OFF:-}" in
    1|true|TRUE|yes|on) echo "[KILL SWITCH] OMNI_LUCIFER_OFF active. Lucifer will not be started."
        START_LUCIFER=false ;;
esac

# 1. Start Hermes IPC adapter
echo "Starting Hermes IPC Adapter..."
declare -A LAST_CMD RESTART_IDX
if [ -f "/opt/omni-bridge/source/omni-bridge/bin/hermes_adapter" ]; then
    /opt/omni-bridge/source/omni-bridge/bin/hermes_adapter &
    PIDS+=($!)
    LAST_CMD[$!]="hermes_adapter"
fi

# 2. Start Lucifer IPC adapter
if [ "$START_LUCIFER" = true ]; then
    echo "Starting Lucifer IPC Adapter..."
    if [ -f "/opt/omni-bridge/source/omni-bridge/bin/lucifer_adapter" ]; then
        /opt/omni-bridge/source/omni-bridge/bin/lucifer_adapter &
        PIDS+=($!)
        LAST_CMD[$!]="lucifer_adapter"
    else
        echo "Warning: lucifer_adapter not found."
    fi
fi

# 3. Start C++ daemon with LD_LIBRARY_PATH
echo "Starting Omni-Bridge C++ Daemon..."
export LD_LIBRARY_PATH="/opt/omni-bridge/source/omni-bridge/lib:${LD_LIBRARY_PATH:-}"
if [ -f "/opt/omni-bridge/source/omni-bridge/bin/omni_bridge_daemon" ]; then
    /opt/omni-bridge/source/omni-bridge/bin/omni_bridge_daemon &
    PIDS+=($!)
    LAST_CMD[$!]="omni_bridge_daemon"
else
    echo "Error: omni_bridge_daemon not found. Proceeding without it for testing."
fi

# Health check loop after startup
echo "Omni-Bridge started. Monitoring health..."
declare -A RESTART_COUNT
while true; do
    sleep 10
    for pid in "${PIDS[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            idx=${RESTART_IDX[$pid]:-0}
            count=${RESTART_COUNT[$pid]:-0}
            # Identify which component died so we can restart just it.
            cmd=$(ps -p "$pid" -o args= 2>/dev/null)
            # $cmd is empty for a dead pid; recover identity from LAST_CMD map
            cmd=${LAST_CMD[$pid]:-unknown}
            echo "[watchdog] component (PID $pid, $cmd) died; restart #$((count+1))"
            if [ "$count" -ge 3 ]; then
                echo "[watchdog] component $cmd exceeded 3 restarts. Shutting down stack for manual review."
                cleanup
            fi
            RESTART_COUNT[$pid]=$((count+1))
            # Restart ONLY the dead component
            case "$cmd" in
                *hermes_adapter*|*hermes_ipc_adapter*)
                    /opt/omni-bridge/source/omni-bridge/bin/hermes_adapter &
                    newpid=$! ;;
                *lucifer_adapter*|*lucifer_ipc_adapter*)
                    if [ "$START_LUCIFER" = true ]; then
                        /opt/omni-bridge/source/omni-bridge/bin/lucifer_adapter &
                        newpid=$!
                    else
                        newpid=""
                    fi ;;
                *omni_bridge_daemon*)
                    /opt/omni-bridge/source/omni-bridge/bin/omni_bridge_daemon &
                    newpid=$! ;;
                *)
                    echo "[watchdog] unknown component; full restart"
                    /opt/omni-bridge/source/omni-bridge/scripts/start_bridge.sh &
                    exit 0 ;;
            esac
            if [ -n "$newpid" ]; then
                # Replace the dead pid's slot so cleanup and future checks track it
                for i in "${!PIDS[@]}"; do
                    if [ "${PIDS[$i]}" = "$pid" ]; then
                        PIDS[$i]="$newpid"
                        LAST_CMD[$newpid]="$cmd"
                        RESTART_IDX[$newpid]=$idx
                    fi
                done
            fi
        fi
    done
done
