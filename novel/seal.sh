#!/usr/bin/env bash
# Copyright (c) 2025-2026 Jacob See.
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Seal the novel manuscript for the token-gated day-one launch.
#   ./seal.sh            seal .plaintext/manuscript.md -> manuscript.md.enc + manuscript.sha256
#   ./seal.sh --unseal   decrypt manuscript.md.enc -> .plaintext/manuscript.decrypted.md
#
# The key (.release-key.txt) and plaintext (.plaintext/) are gitignored and must
# never be committed. Keep offline backups of both.

set -euo pipefail
cd "$(dirname "$0")"

SRC=".plaintext/manuscript.md"
ENC="manuscript.md.enc"
SUM="manuscript.sha256"
KEYFILE=".release-key.txt"

need_openssl() {
  command -v openssl >/dev/null 2>&1 || {
    echo "error: openssl is required but not installed." >&2
    exit 1
  }
}

if [ "${1:-}" = "--unseal" ]; then
  need_openssl
  [ -f "$ENC" ] || { echo "error: $ENC not found (nothing to unseal)." >&2; exit 1; }
  [ -f "$KEYFILE" ] || { echo "error: $KEYFILE not found; restore it from your offline backup." >&2; exit 1; }
  mkdir -p .plaintext
  openssl enc -d -aes-256-cbc -pbkdf2 -iter 600000 \
    -in "$ENC" -out ".plaintext/manuscript.decrypted.md" \
    -pass file:"$KEYFILE"
  echo "Unsealed to .plaintext/manuscript.decrypted.md"
  echo "Verify:  sha256sum .plaintext/manuscript.decrypted.md   (expect $(cut -d' ' -f1 "$SUM"))"
  exit 0
fi

need_openssl
[ -f "$SRC" ] || {
  echo "error: place the manuscript at $SRC first (see README.md)." >&2
  exit 1
}

if [ ! -f "$KEYFILE" ]; then
  umask 077
  openssl rand -hex 32 > "$KEYFILE"
  echo "Generated new content key: $KEYFILE"
  echo "IMPORTANT: back this file up offline. Losing it before launch means re-sealing."
fi

openssl enc -aes-256-cbc -pbkdf2 -iter 600000 -salt \
  -in "$SRC" -out "$ENC" \
  -pass file:"$KEYFILE"

sha256sum "$SRC" > "$SUM"

echo "Sealed:"
echo "  ciphertext : $ENC   (commit this)"
echo "  commitment : $SUM   (commit this)"
echo "  key        : $KEYFILE   (NEVER commit; back up offline)"
echo ""
echo "Verify roundtrip with:  ./seal.sh --unseal"
