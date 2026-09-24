import assert from "node:assert/strict";
import test from "node:test";

import {
  CARE_AMITY_POLICY_VERSION,
  calculateCareAmityConversion,
  getCareAmityProfile,
} from "../lib/care-amity-policy.mjs";

test("initial CARE to AMITY conversion docks five percent", () => {
  const result = calculateCareAmityConversion({
    careAmount: 1000,
    conversionRateBps: 10_000,
    profile: "initial",
  });

  assert.equal(result.policyVersion, CARE_AMITY_POLICY_VERSION);
  assert.equal(result.grossAmity, "1000");
  assert.equal(result.reserveAmity, "50");
  assert.equal(result.burnedAmity, "0");
  assert.equal(result.netAmity, "950");
  assert.equal(result.totalReductionBps, 500);
  assert.equal(result.usesPriceOracle, false);
  assert.equal(result.appliedAtConversion, true);
});

test("extreme profile keeps participant dock at twenty percent and burns fifteen percent", () => {
  const result = calculateCareAmityConversion({
    careAmount: 10_000,
    conversionRateBps: 10_000,
    profile: "extreme",
  });

  assert.equal(result.grossAmity, "10000");
  assert.equal(result.reserveAmity, "2000");
  assert.equal(result.burnedAmity, "1500");
  assert.equal(result.netAmity, "6500");
  assert.equal(result.participantDockBps, 2_000);
  assert.equal(result.burnBps, 1_500);
  assert.equal(result.totalReductionBps, 3_500);
});

test("profile schedule exposes the intended bounded rates", () => {
  assert.deepEqual(getCareAmityProfile("initial"), {
    name: "initial",
    participantDockBps: 500,
    burnBps: 0,
  });
  assert.deepEqual(getCareAmityProfile("maximum"), {
    name: "maximum",
    participantDockBps: 2_000,
    burnBps: 0,
  });
});

test("policy rejects rates above the twenty plus fifteen percent ceiling", () => {
  assert.throws(
    () =>
      calculateCareAmityConversion({
        careAmount: 1000,
        participantDockBps: 2_001,
      }),
    /participantDockBps cannot exceed/,
  );

  assert.throws(
    () =>
      calculateCareAmityConversion({
        careAmount: 1000,
        burnBps: 1_501,
      }),
    /burnBps cannot exceed/,
  );

  assert.throws(
    () =>
      calculateCareAmityConversion({
        careAmount: 1000,
        participantDockBps: 2_000,
        burnBps: 1_501,
      }),
    /burnBps cannot exceed/,
  );
});

test("policy cannot be bypassed by changing later transaction context", () => {
  const result = calculateCareAmityConversion({
    careAmount: 1234,
    conversionRateBps: 10_000,
    profile: "extreme",
  });

  assert.equal(result.appliedAtConversion, true);
  assert.equal(result.totalReductionBps, 3_500);
  assert.equal(result.netAmity, "803");
});
