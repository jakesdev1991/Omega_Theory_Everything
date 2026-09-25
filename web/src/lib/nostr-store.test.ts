import test from "node:test";
import assert from "node:assert/strict";

import { generateSecretKey, getPublicKey } from "nostr-tools";

import {
  buildJobRequestTemplate,
  dedupeListings,
  demoJobResult,
  demoListings,
  hexToBytes,
  jobResultFilter,
  parseJobResult,
  parseListingEvent,
  signJobRequest,
  DEMO_ROOT_SECRET_HEX,
} from "./nostr-store";

const demoSecret = hexToBytes(DEMO_ROOT_SECRET_HEX);
const demoRoot = getPublicKey(demoSecret);

test("Directory events parse into listings, and foreign kinds are ignored", () => {
  const events = demoListings(demoRoot);
  assert.equal(events.length, 3);

  const listings = events.map(parseListingEvent).filter((listing) => listing !== null);
  assert.equal(listings.length, 3);

  const byId = Object.fromEntries(listings.map((listing) => [listing.appId, listing]));
  assert.equal(byId["radial-metric-sim"].jobKind, 5001, "k tag wins");
  assert.equal(byId["lean-audit"].jobKind, 5002);
  assert.equal(byId["rcod-benchmark"].jobKind, 5003, "content jobKind fallback for 30017");
  assert.equal(byId["radial-metric-sim"].sourceKind, 31990);
  assert.equal(byId["rcod-benchmark"].sourceKind, 30017);
  assert.deepEqual(byId["radial-metric-sim"].paramsTemplate, { phi: 0.618, chainLength: 64 });

  const notAListing = parseListingEvent({ ...events[0], kind: 1 });
  assert.equal(notAListing, null);
});

test("Tampered directory events fail signature verification and are dropped", () => {
  const [event] = demoListings(demoRoot);
  const tampered = { ...event, content: `${event.content} // injected` };
  assert.equal(parseListingEvent(tampered), null);
});

test("Dedupe keeps the newest parameterized-replaceable entry per publisher", () => {
  const events = demoListings(demoRoot);
  const listings = events.map(parseListingEvent).filter((l) => l !== null);
  const older = { ...listings[0], updatedAt: listings[0].updatedAt - 500, name: "Stale name" };
  const deduped = dedupeListings([...listings, older]);
  const radial = deduped.find((listing) => listing.appId === "radial-metric-sim");
  assert.equal(radial?.name, "Radial Metric Simulator");
  assert.equal(deduped.length, 3);
});

test("Job requests carry the NIP-90 correlation tags and sign verifiably", () => {
  const [listing] = demoListings(demoRoot).map(parseListingEvent).filter((l) => l !== null);
  const operatorSecret = generateSecretKey();
  const operatorPubkey = getPublicKey(operatorSecret);

  const request = signJobRequest(
    buildJobRequestTemplate({
      listing,
      serverPubkey: demoRoot,
      params: { phi: 0.5 },
      inputs: ["pkg-head-commit-2026"],
    }),
    operatorSecret,
  );

  assert.equal(request.jobKind, 5001);
  assert.deepEqual(request.event.tags.find((tag) => tag[0] === "p"), ["p", demoRoot]);
  assert.deepEqual(request.event.tags.find((tag) => tag[0] === "a"), ["a", listing.appId]);
  assert.deepEqual(request.event.tags.find((tag) => tag[0] === "i"), ["i", "pkg-head-commit-2026"]);
  assert.equal(request.event.pubkey, operatorPubkey);
  assert.match(request.event.content, /"phi":0\.5/);
});

test("Job results correlate by request id and reject unrelated or tampered events", () => {
  const [listing] = demoListings(demoRoot).map(parseListingEvent).filter((l) => l !== null);
  const operatorSecret = generateSecretKey();
  const operatorPubkey = getPublicKey(operatorSecret);

  const request = signJobRequest(
    buildJobRequestTemplate({ listing, serverPubkey: demoRoot, params: {} }),
    operatorSecret,
  );
  const result = demoJobResult(request.event, operatorPubkey);

  assert.equal(result.kind, 6001, "result kind = request kind + 1000");
  const parsed = parseJobResult(result, request.event.id);
  assert.ok(parsed);
  assert.equal(parsed.status, "success");
  assert.match(parsed.content, /stable_convergence/);

  // The filter must target exactly this request and requester.
  const filter = jobResultFilter(request.event.id, operatorPubkey, 5001);
  assert.deepEqual(filter, { kinds: [6001], "#e": [request.event.id], "#p": [operatorPubkey] });

  // A result for a different request id must not correlate.
  assert.equal(parseJobResult(result, "0".repeat(64)), null);
  // Tampered result content fails verification.
  assert.equal(parseJobResult({ ...result, content: "forged" }, request.event.id), null);
});
