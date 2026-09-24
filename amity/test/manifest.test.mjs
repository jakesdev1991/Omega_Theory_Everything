import test from "node:test";
import assert from "node:assert/strict";

import { assertValidManifest, createManifest } from "../lib/manifest.mjs";

function sampleConfig() {
  return {
    network: "testnet",
    assetId: "a".repeat(64),
    universeUrl: "https://universe.example.testnet/amity",
    assetName: "AMITY Test Asset",
    unitName: "AMITY",
    litdRpcHost: "127.0.0.1:8443",
    tapdRpcHost: "127.0.0.1:10029",
    litdTlsCertPath: "/tmp/litd.cert",
    litdMacaroonPath: "/tmp/litd.macaroon",
    tapdTlsCertPath: undefined,
    tapdMacaroonPath: undefined,
  };
}

function sampleOperatorFiles() {
  return [
    { label: "LITD_TLS_CERT_PATH", configured: true, exists: true },
    { label: "LITD_MACAROON_PATH", configured: true, exists: true },
    { label: "TAPD_TLS_CERT_PATH", configured: false, exists: false },
    { label: "TAPD_MACAROON_PATH", configured: false, exists: false },
  ];
}

test("AMITY manifest captures scaffold state without unlock wiring", () => {
  const manifest = createManifest({ config: sampleConfig(), operatorFiles: sampleOperatorFiles() });

  assert.equal(manifest.pilot.network, "testnet");
  assert.equal(manifest.pilot.wiredIntoWebUnlock, false);
  assert.equal(manifest.pilot.liveHolderVerificationImplemented, false);
  assert.equal(manifest.readiness.requiredConfigComplete, true);
  assert.equal(manifest.readiness.operatorFilesPresent, false);
  assert.equal(manifest.readiness.holderVerificationReady, false);
  assert.equal(manifest.services.litd.tlsCertConfigured, true);
  assert.equal(manifest.services.tapd.tlsCertConfigured, false);
  assert.doesNotThrow(() => assertValidManifest(manifest));
});

test("AMITY manifest rejects invalid integration claims", () => {
  const manifest = createManifest({ config: sampleConfig(), operatorFiles: sampleOperatorFiles() });

  assert.throws(
    () => assertValidManifest({
      ...manifest,
      pilot: {
        ...manifest.pilot,
        wiredIntoWebUnlock: true,
      },
    }),
    /wiredIntoWebUnlock/,
  );

  assert.throws(
    () => assertValidManifest({
      ...manifest,
      readiness: {
        ...manifest.readiness,
        holderVerificationReady: true,
      },
    }),
    /holderVerificationReady/,
  );
});
