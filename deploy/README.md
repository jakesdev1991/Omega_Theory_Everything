<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

# Omega Nostr backplane — deployment (Ryzenvoid, systemd user units)

Three user services make up the backplane:

| Unit | Role |
|---|---|
| `omega-nostr-relay.service` | Chorus v2.0.2 relay (the event backplane) |
| `omega-nostr-client.service` | Sovereign publisher (store directory kind 31990, economy events) |
| `omega-lucifer.service` | NIP-90 executor node (answers /store Run requests; moves to the Pixel 8a later) |

## Bootstrap order (one-time)

The unit files in this directory carry **absolute paths from the machine they
were first deployed on** (`/home/jake/...`) — systemd wants absolute paths, and
the alternative, `%h`, behaves differently enough between user and system units
that a wrong guess would break the deployment silently. Rewrite them for
whoever is installing instead; this is also a no-op on that first machine:

```bash
sed "s|/home/jake|$HOME|g" deploy/omega-*.service > /tmp/omega-units/   # then inspect
```

The examples below use `~` and assume the repo at `$HOME/Omega_Theory_Everything`
and the relay build at `$HOME/omega_nostr_build/chorus/target/release/chorus`.

1. **Keys (GATED — requires Jake's explicit approval).**
   ```bash
   node ~/.config/omega-nostr/keys/keygen.mjs   # root + operator npubs, 0600
   ```
   Writes `~/.config/omega-nostr/keys/env.sh` (root/operator secrets,
   LUCIFER_* aliases, relay URLs). The nsec values are never printed to
   chat; back up `secrets.json` offline.

2. **TLS cert for the relay** (local CA; replaced by Let's Encrypt at
   public launch):
   ```bash
   sh ~/.config/omega-nostr/tls/make-certs.sh
   ```
   Creates `~/.config/omega-nostr/tls/ryzenvoid-fullchain.pem` +
   privkey (SANs: ryzenvoid, localhost, ryzenvoid.attlocal.net, LAN IPs).
   LAN devices that talk to the relay add the CA to their trust store
   (`NODE_EXTRA_CA_CERTS=...omega-ca-cert.pem` for Node clients).

3. **Relay user registration** (personal-relay phase):
   ```bash
   ~/omega_nostr_build/chorus/target/release/chorus_cmd \
     ~/.config/omega-nostr/chorus.conf add_user $(grep OMEGA_NOSTR_ROOT_HEX= \
     ~/.config/omega-nostr/keys/env.sh | cut -d'"' -f2) 0
   ```
   (During the current LAN phase `open_relay=true` in chorus.conf, so
   this is optional; it becomes mandatory before flipping to
   `open_relay=false`.)

4. **Install + enable the units:**
   ```bash
   mkdir -p ~/.config/systemd/user
   # Rewrite the absolute paths baked into the units for this machine; on the
   # machine they were written for, the substitution changes nothing. The grep
   # that follows fails loudly if any path was missed.
   for unit in deploy/omega-*.service; do
     sed "s|/home/jake|$HOME|g" "$unit" > ~/.config/systemd/user/"$(basename "$unit")"
   done
   grep -l '/home/jake' ~/.config/systemd/user/omega-*.service 2>/dev/null \
     && echo 'STILL MACHINE-SPECIFIC — fix before enabling' || echo 'unit paths rewritten'
   systemctl --user daemon-reload
   systemctl --user enable --now omega-nostr-relay
   systemctl --user enable --now omega-nostr-client
   systemctl --user enable --now omega-lucifer
   ```

5. **Fill the web env**: replace the `FILL_AFTER_KEYGEN` placeholders in
   `web/.env.local` with the root npub from step 1 (the npub IS public).

6. **Persist across reboots (needs root once):**
   ```bash
   sudo loginctl enable-linger jake
   ```

## Verify

```bash
systemctl --user status omega-nostr-relay   # listening wss://ryzenvoid:8080
curl -s https://ryzenvoid:8080 -H 'Accept: application/nostr+json' | jq .
journalctl --user -u omega-nostr-client -f  # directory publish OKs
cd nostr-client && npm test                  # unit + integration + live smoke
```

The full e2e (directory visible → kind-5001 job → 6001 result) is
`~/omega_nostr_build/e2e-live.mjs`, runnable once keys exist:
`node ~/omega_nostr_build/e2e-live.mjs`.

## Public-launch TODOs (do NOT do these in the LAN phase)

- [ ] `open_relay=false` in `~/.config/omega-nostr/chorus.conf` (NIP-42
      AUTH + registered users only; strangers can still post replies to
      authorized users, per chorus personal-relay rules)
- [ ] Real domain → Let's Encrypt (chorus systemd copies certs on start),
      update `hostname`, SANs, and `NOSTR_RELAYS` everywhere
- [ ] `NOSTR_NIP05_DOMAIN` + `NOSTR_NIP05_NAMES` in web env →
      `/.well-known/nostr.json` goes live
- [ ] Publish the relay on nostr.wine / relay registries AFTER the above
- [ ] Move `omega-lucifer.service` to the Pixel 8a (same env file, same
      key — identity is portable by design)

## Failure modes (documented gotchas)

- **NIP-42 AUTH "relay is wrong"**: the AUTH event's `relay` tag must equal
  the exact URL dialed (no canonicalization). `nostr-client/lib/relay.mjs`
  handles this; any new client must too.
- **`ryzenvoid` resolves IPv6 only** on this LAN (attlocal.net): the relay
  binds `ip_address = "::"` (dual-stack). IPv4-only clients use the LAN IP
  `192.168.1.184` (in the cert SANs).
- **chorus bans IPs for ~1-2s after disconnect** — reconnect loops must
  back off (the pool does: 1s → 30s exponential).
- **Self-signed CA**: Node clients need `NODE_EXTRA_CA_CERTS`; browsers
  need the CA imported once.
