#!/data/data/com.termux/files/usr/bin/bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Termux installer for the Lucifer mobile node (Pixel 8a sovereign executor).
#
# What it does:
#   1. installs Node.js (and optionally python for the bundled algorithms)
#   2. installs this workspace's mobile-node dependencies
#   3. writes ~/.lucifer/env.sh with your relays/keys (chmod 600)
#   4. prints the wake-lock + launcher recipe for background persistence
#
# It never disables TLS verification and never stores keys in this repository.

set -euo pipefail

cd "$(dirname "$0")"

pkg install -y nodejs-lts python wget
npm install --omit=dev

ENV_FILE="${HOME}/.lucifer/env.sh"
mkdir -p "$(dirname "${ENV_FILE}")"

if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<'EOF'
# Lucifer mobile node environment — keep private (chmod 600).
export LUCIFER_RELAYS="wss://nos.lol,wss://relay.damus.io"
# Server/root key (hex or nsec). This is the pubkey the store addresses jobs to.
export LUCIFER_SECRET="replace-with-nsec-or-hex"
# Comma separated operator pubkeys allowed to request execution (npub or hex).
export LUCIFER_OPERATORS="replace-with-your-npub"
# Debian sandbox prefix (Droid@Debian / proot-distro). Empty = local execution.
# export LUCIFER_SANDBOX_NOTE="set sandboxArgv in algorithms.json instead"
EOF
  chmod 600 "${ENV_FILE}"
  echo "created ${ENV_FILE} — edit it with your keys before starting"
fi

cat <<'EOF'

Next steps on the device:
  1. Edit ~/.lucifer/env.sh (relays, LUCIFER_SECRET, LUCIFER_OPERATORS).
  2. Optionally enable the Debian sandbox: set "sandboxArgv" in
     mobile-node/algorithms.json to ["proot-distro", "run", "debian", "--"]
     after `proot-distro install debian`.
  3. Publish the store directory once:
       source ~/.lucifer/env.sh && node publish-directory.mjs
  4. Start the daemon with wake lock (background persistence):
       source ~/.lucifer/env.sh
       termux-wake-lock
       nohup node lucifer-daemon.mjs >> ~/lucifer.log 2>&1 &
     Or wire it into your custom launcher's boot sequence.
  5. Open /store on the website, set the root npub, and press Run.

Security reminders:
  - LUCIFER_OPERATORS is the execution allowlist. Keep it tight.
  - algorithms.json is the code allowlist. Nothing outside it can run.
  - Relays are always TLS-verified; loopback ws:// exists only for tests.
EOF
