<!-- Copyright (c) 2025-2026 Jacob See. SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->

# Lucifer mobile node — sovereign NIP-90 executor

The phone half of the Omega app-store architecture. The website's `/store` page
is a **static frontend**: it reads its directory from Nostr and signs NIP-90 job
requests. This daemon is the **backend**: a single-digit-megabyte WebSocket
process that runs on the Pixel 8a (Termux → Droid@Debian), verifies incoming
jobs, executes allowlisted algorithms in the sandbox, and answers with signed
job results.

```
STATIC FRONTEND (/store)          PUBLIC RELAY POOL            PIXEL 8a (this daemon)
  REQ kinds 31990/30017  ──read──▶  event backplane  ──push──▶  REQ filter {kinds:[5001…], "#p":[server]}
  EVENT kind 5001…      ──write──▶                 ──push──▶  verify → policy → sandbox exec
  EVENT kind 6001…      ◀──read──                  ◀──write──  result (kind+1000, e/p tags)
```

## Corrections applied to the original sketch

| Sketch issue | What this implementation does |
|---|---|
| `{"kinds":, "#p": [SERVER_PUBKEY]}` (invalid JSON) | Proper filter: `{"kinds": [5001, 5002, 5003, 5099], "#p": [serverHex]}` |
| `sslopt={"cert_reqs": ssl.CERT_NONE}` | TLS is always verified. Plain `ws://` is accepted **only** for loopback hosts, for the integration suite. |
| "Construct and sign the response" (undefined) | Full NIP-01 serialization → sha256 id → BIP-340 schnorr signature (`lib/events.mjs`), result kind = request kind + 1000, tagged `["e", requestId]`, `["p", requester]`, `["status", …]` |
| No verification of incoming events | **Strict** verification: the event id is recomputed from `[0, pubkey, created_at, kind, tags, content]` *and* the signature checked. (nostr-tools' `verifyEvent` alone does not recompute the id — see `web/src/lib/nostr-store.ts`.) |
| Parameters piped "straight into" the sandbox | Five-gate policy (`lib/policy.mjs`): strict verify → operator allowlist → kind allowlist → algorithm id allowlist → params object + size cap. Execution uses argv arrays and stdin JSON, never a shell string (`lib/executor.mjs`), with timeouts and bounded output. |
| Directory kind ambiguity | 31990 (NIP-89 handler announcement) preferred, 30017 (NIP-99 listing) accepted; deduped per `d` tag, newest wins. |

## Layout

```
mobile-node/
├── lucifer-daemon.mjs        relay loop, policy gate, sandbox exec, result publish
├── publish-directory.mjs     publishes kind 31990 listings for algorithms.json (root key)
├── issue-license.mjs         issue / renew / revoke kind 31335 app licenses (root key)
├── algorithms.json           THE code allowlist: id, kind, argv, timeout, output cap
├── algorithms/               bundled allowlisted algorithms (stdin JSON → stdout JSON)
│   ├── radial_metric.py      g_rr(Φ) finite-chain toy evaluation
│   ├── lean_audit.py         sorry/axiom inventory over lean_proofs/
│   └── rcod_benchmark.py     honest noise-recovery verdict
├── lib/                      bech32, NIP-01 events (strict), policy, license, executor
├── test/                     unit + daemon↔relay integration suite (node --test)
└── termux-install.sh         Termux bootstrap + key env file + wake-lock recipe
```

## Run it

```bash
# on the device (or anywhere with Node 18+):
./termux-install.sh                     # Termux: deps + ~/.lucifer/env.sh
source ~/.lucifer/env.sh                # LUCIFER_RELAYS / LUCIFER_SECRET / LUCIFER_OPERATORS
node publish-directory.mjs              # once: announce the store directory (kind 31990)
node lucifer-daemon.mjs                 # forever: answer job requests
```

Desktop/bench run without Termux:

```bash
npm install
LUCIFER_RELAYS=wss://nos.lol \
LUCIFER_SECRET=<nsec-or-hex> \
LUCIFER_OPERATORS=<your-npub> \
npm start
npm test                                # unit + licensing + daemon↔relay integration
```

## Pairing with the website

1. `web/.env.local`: `NOSTR_RELAYS=…` and `NOSTR_STORE_ROOT_NPUB=<npub of LUCIFER_SECRET>`.
2. `/store` → mode **live relays** → Connect → the directory cards render from your 31990 events.
3. Press **Run**: the page signs a kind-5001… request with its local operator key; this daemon
   executes it (operator key must be in `LUCIFER_OPERATORS`) and answers; the page correlates by
   `["e", requestId]`.
4. **Settle result as TWC work** posts the verified job into the economy ledger as an audited
   work receipt (`/testnet` shows it), keeping the ledger's double-entry invariants intact.

## Licensing apps

By default every algorithm is **operator-only**. To sell or share an app, add a `license` block
to it in `algorithms.json` (the bundled `echo` self-test is licensed as an example):

```json
"license": { "access": "licensed", "terms": "omega-store-eula-1.0", "tiers": ["trial", "standard", "pro"] }
```

Re-run `publish-directory.mjs` so the listing advertises `access`/`terms`/`license` tags, then
issue licenses to buyers' npubs (the key stays on the device):

```bash
node issue-license.mjs issue  --app echo --to npub1buyer… --tier standard --days 30 --payment manual
node issue-license.mjs revoke --app echo --to npub1buyer…
```

The daemon honours licenses signed by `LUCIFER_LICENSE_ISSUERS` (default: its own key), follows
revocations live from relays, and answers unlicensed requests with NIP-90 `payment-required`.
Licenses record the Store Terms version the buyer accepted (`/store/terms` on the website).
Full spec: [`../docs/store/LICENSE-PROTOCOL.md`](../docs/store/LICENSE-PROTOCOL.md). Third-party
apps may only be listed under a signed Publisher Agreement (`web/public/legal/publisher-agreement.md`).

## Security model

- Relays are untrusted transport. Nothing executes without passing all five policy gates.
- Non-operators can run only algorithms explicitly marked `licensed`, and only with a valid,
  unexpired, unrevoked license from a trusted issuer. Per-licensee rate limits are not yet
  implemented; keep licensed algorithms cheap until they are.
- `algorithms.json` is the only code path: adding capability means editing the allowlist in git,
  never relay content.
- Parameters arrive on stdin as JSON; argv is a fixed array; no `sh -c` anywhere; env is scrubbed.
- Timeouts SIGKILL; stdout/stderr are capped; processed-request ids are deduped.
- Keys live in `~/.lucifer/env.sh` (chmod 600) on the device, or browser-local storage on the
  storefront. Neither is ever transmitted.
