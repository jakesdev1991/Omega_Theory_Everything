# Copyright (c) 2025-2026 Jacob See. Licensed under MIT; see LICENSE.
"""Keep the two copies of the NIP-01 primitives from drifting apart.

``mobile-node/`` and ``nostr-client/`` each carry their own ``lib/events.mjs``
and ``lib/bech32.mjs``. That duplication is deliberate: the mobile node is
deployed on its own to a phone (``termux-install.sh`` runs ``npm install`` in
the package directory), so neither package can import across the repository
boundary.

Deliberate does not mean safe. These two modules implement event-id
recomputation and strict schnorr verification, which is exactly the code where
a one-sided fix silently leaves one deployment checking less than the other.
Nothing in CI noticed that before, because both copies also carry their own
test files. This test makes a divergence a red check instead of a surprise.

If a divergence is genuinely wanted, update the expected map below *and* say in
the commit message why the two deployments should differ.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Shared by design. ``policy.mjs`` is not in this list: the two packages
# enforce different spending/authority policies on purpose.
MUST_MATCH = ("events.mjs", "bech32.mjs")


def test_shared_nostr_modules_are_identical() -> None:
    for name in MUST_MATCH:
        mobile = ROOT / "mobile-node" / "lib" / name
        client = ROOT / "nostr-client" / "lib" / name
        assert mobile.is_file(), f"{mobile.relative_to(ROOT)} is missing"
        assert client.is_file(), f"{client.relative_to(ROOT)} is missing"
        assert mobile.read_bytes() == client.read_bytes(), (
            f"mobile-node/lib/{name} and nostr-client/lib/{name} have diverged. "
            "They are duplicated because each package deploys on its own; port the "
            "fix to both copies (or change this test deliberately)."
        )
