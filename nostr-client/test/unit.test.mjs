// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadPolicy, isAcceptableRelayUrl, parseRelayList } from "../lib/policy.mjs";
import { getPublicKeyHex, signEvent, verifyEventStrict } from "../lib/events.mjs";
import { hexToNpub, npubToHex, nsecToHex, bech32Encode } from "../lib/bech32.mjs";
import { RelayConnection } from "../lib/relay.mjs";
import { ReplyLedger, isMention, buildReply, PITCH } from "../bot.mjs";
import { buildDirectoryEvents, ECONOMY_KINDS } from "../client-daemon.mjs";

const SECRET = "07".repeat(32);
const OTHER = "0b".repeat(32);

test("policy: the reply-ledger state file follows HOME, not one machine", () => {
  // The default used to be the literal string "/home/jake/.local/state/..." —
  // the developer's home directory. On any other machine the daemon would write
  // its state to a path it does not own. It must derive the path from HOME (or
  // the OS home directory) instead.
  const policy = loadPolicy({
    OMEGA_NOSTR_RELAYS: "wss://relay.example.org",
    OMEGA_NOSTR_ROOT_SECRET: SECRET,
    HOME: "/tmp/omega-home-check",
  });
  assert.equal(policy.stateFile, "/tmp/omega-home-check/.local/state/omega-nostr/bot-state.json");

  // And an explicit override wins, as documented.
  const overridden = loadPolicy({
    OMEGA_NOSTR_RELAYS: "wss://relay.example.org",
    OMEGA_NOSTR_ROOT_SECRET: SECRET,
    HOME: "/tmp/omega-home-check",
    OMEGA_NOSTR_STATE_FILE: "/var/lib/omega/state.json",
  });
  assert.equal(overridden.stateFile, "/var/lib/omega/state.json");
});

test("policy: fails closed on missing env", () => {
  const saved = { ...process.env };
  try {
    for (const k of Object.keys(process.env)) {
      if (k.startsWith("OMEGA_") || k.startsWith("LUCIFER_")) delete process.env[k];
    }
    assert.throws(() => loadPolicy({}), /missing required env/);
  } finally {
    process.env = saved;
  }
});

test("policy: fails closed on empty relay list", () => {
  assert.throws(
    () => loadPolicy({ OMEGA_NOSTR_RELAYS: "", OMEGA_NOSTR_ROOT_SECRET: SECRET }),
    /no acceptable relay/,
  );
});

test("policy: rejects ws:// off-loopback (web contract requires wss://)", () => {
  assert.equal(isAcceptableRelayUrl("ws://example.com"), false);
  assert.equal(isAcceptableRelayUrl("ws://192.168.1.50:8080"), false);
  assert.equal(isAcceptableRelayUrl("wss://ryzenvoid:8080"), true);
  assert.equal(isAcceptableRelayUrl("ws://127.0.0.1:18080"), true);
  assert.equal(parseRelayList("wss://a.example, ws://b.example, ws://127.0.0.1:1").length, 2);
});

test("policy: accepts nsec or hex root secret", () => {
  const p = loadPolicy({ OMEGA_NOSTR_RELAYS: "wss://r.example", OMEGA_NOSTR_ROOT_SECRET: SECRET });
  assert.equal(p.rootPubkeyHex, getPublicKeyHex(SECRET));
  assert.throws(() =>
    loadPolicy({ OMEGA_NOSTR_RELAYS: "wss://r.example", OMEGA_NOSTR_ROOT_SECRET: "zz" }),
  );
});

test("events: sign/verify roundtrip + strict verification catches tampering", () => {
  const ev = signEvent({ kind: 1, created_at: 1, tags: [], content: "omega" }, SECRET);
  assert.equal(verifyEventStrict(ev), true);
  const bad = { ...ev, content: "tampered" };
  assert.equal(verifyEventStrict(bad), false);
});

test("bech32: npub/nsec roundtrip", () => {
  const pub = getPublicKeyHex(SECRET);
  const npub = hexToNpub(pub);
  assert.equal(npub.startsWith("npub1"), true);
  assert.equal(npubToHex(npub), pub);
  // nsec roundtrip through our own encoder/decoder
  const nsec = bech32Encode("nsec", Uint8Array.from(Buffer.from(SECRET, "hex")));
  assert.equal(nsec.startsWith("nsec1"), true);
  assert.equal(nsecToHex(nsec), SECRET);
});

test("bot: mention detection requires #p tag or npub in content", () => {
  const root = getPublicKeyHex(SECRET);
  const rootNpub = hexToNpub(root);
  const tagged = signEvent({ kind: 1, created_at: 1, tags: [["p", root]], content: "hey" }, OTHER);
  const inContent = signEvent({ kind: 1, created_at: 1, tags: [], content: `ping ${rootNpub}` }, OTHER);
  const unrelated = signEvent({ kind: 1, created_at: 1, tags: [], content: "hello world" }, OTHER);
  const notKind1 = signEvent({ kind: 3, created_at: 1, tags: [["p", root]], content: "" }, OTHER);
  assert.equal(isMention(tagged, root, rootNpub), true);
  assert.equal(isMention(inContent, root, rootNpub), true);
  assert.equal(isMention(unrelated, root, rootNpub), false);
  assert.equal(isMention(notKind1, root, rootNpub), false);
});

test("bot: reply includes p/e/r tags, lnurl only when configured", () => {
  const root = getPublicKeyHex(SECRET);
  const mention = signEvent({ kind: 1, created_at: 1, tags: [["p", root]], content: "hey" }, OTHER);
  const withLn = buildReply(mention, SECRET, "jake@getalby.com");
  assert.deepEqual(withLn.tags[0], ["p", mention.pubkey]);
  assert.ok(withLn.tags.some((t) => t[0] === "e" && t[1] === mention.id));
  assert.ok(withLn.tags.some((t) => t[0] === "r" && t[1].includes("github.com/jakesdev1991")));
  assert.ok(withLn.tags.some((t) => t[0] === "lnurl"));
  assert.ok(withLn.content.startsWith("The Omega Theory of Everything"));
  const noLn = buildReply(mention, SECRET, null);
  assert.ok(!noLn.tags.some((t) => t[0] === "lnurl"));
  assert.equal(verifyEventStrict(withLn), true);
});

test("bot: ReplyLedger enforces 24h cooldown with injectable clock", () => {
  const dir = mkdtempSync(join(tmpdir(), "omega-bot-"));
  const stateFile = join(dir, "state.json");
  let t = Date.now(); // realistic wall-clock base for the fake clock
  const ledger = new ReplyLedger(stateFile, { now: () => t });
  const pubkey = getPublicKeyHex(OTHER);
  assert.equal(ledger.mayReply(pubkey), true);
  ledger.markReplied(pubkey);
  assert.equal(ledger.mayReply(pubkey), false);
  t += 23 * 3600 * 1000;
  assert.equal(ledger.mayReply(pubkey), false);
  t += 2 * 3600 * 1000; // 25h total
  assert.equal(ledger.mayReply(pubkey), true);
  // persistence across reload
  const ledger2 = new ReplyLedger(stateFile, { now: () => t });
  assert.equal(ledger2.mayReply(pubkey), true); // 25h > cooldown
  rmSync(dir, { recursive: true, force: true });
});

test("daemon: directory events are signed kind 31990 with d/k tags", () => {
  const algorithms = {
    algorithms: [
      { id: "echo", kind: 5099, name: "Echo", workClass: "engineering_protocol" },
      { id: "lean-audit", kind: 5002, name: "Lean Axiom Audit", workClass: "lean_formalization" },
    ],
  };
  const events = buildDirectoryEvents(algorithms, SECRET);
  assert.equal(events.length, 2);
  for (const ev of events) {
    assert.equal(ev.kind, ECONOMY_KINDS.storeHandler);
    assert.equal(verifyEventStrict(ev), true);
    assert.equal(ev.pubkey, getPublicKeyHex(SECRET));
  }
  assert.ok(events[0].tags.some((t) => t[0] === "d" && t[1] === "echo"));
  assert.ok(events[0].tags.some((t) => t[0] === "k" && t[1] === "5099"));
});
