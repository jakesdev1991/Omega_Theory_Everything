export const BASIS_POINTS = 10_000n;
export const MAX_PARTICIPANT_DOCK_BPS = 2_000;
export const MAX_BURN_BPS = 1_500;
export const MAX_TOTAL_REDUCTION_BPS = 3_500;
export const CARE_AMITY_POLICY_VERSION = "care-amity-bootstrap-v0.1";

const PROFILES = Object.freeze({
  initial: Object.freeze({ participantDockBps: 500, burnBps: 0 }),
  elevated: Object.freeze({ participantDockBps: 1_000, burnBps: 0 }),
  maximum: Object.freeze({ participantDockBps: 2_000, burnBps: 0 }),
  extreme: Object.freeze({ participantDockBps: 2_000, burnBps: 1_500 }),
});

function assert(condition, message) {
  if (!condition) throw new Error(`CARE/AMITY policy error: ${message}`);
}

function integerValue(value, label) {
  const normalized = typeof value === "bigint" ? value : BigInt(value);
  assert(normalized >= 0n, `${label} must not be negative.`);
  return normalized;
}

function positiveIntegerValue(value, label) {
  const normalized = integerValue(value, label);
  assert(normalized > 0n, `${label} must be positive.`);
  return normalized;
}

function basisPointValue(value, label) {
  assert(Number.isInteger(value), `${label} must be an integer number of basis points.`);
  assert(value >= 0 && value <= Number(BASIS_POINTS), `${label} must be between 0 and 10000 basis points.`);
  return value;
}

export function getCareAmityProfile(name = "initial") {
  assert(typeof name === "string" && name in PROFILES, `unknown policy profile: ${name}.`);
  return Object.freeze({ name, ...PROFILES[name] });
}

/**
 * Calculate a CARE -> AMITY entitlement without a price oracle or minting.
 *
 * The participant dock is a reserve/protection allocation. The burn is a
 * supply sink. Both are calculated at conversion time, before any AMITY can
 * be transferred, so later barter or item purchases cannot bypass them.
 */
export function calculateCareAmityConversion({
  careAmount,
  conversionRateBps = 10_000,
  profile = "initial",
  participantDockBps = undefined,
  burnBps = undefined,
  policyVersion = CARE_AMITY_POLICY_VERSION,
}) {
  const care = positiveIntegerValue(careAmount, "careAmount");
  const selected = getCareAmityProfile(profile);
  const rate = basisPointValue(conversionRateBps, "conversionRateBps");
  const dock = basisPointValue(participantDockBps ?? selected.participantDockBps, "participantDockBps");
  const burn = basisPointValue(burnBps ?? selected.burnBps, "burnBps");

  assert(dock <= MAX_PARTICIPANT_DOCK_BPS, `participantDockBps cannot exceed ${MAX_PARTICIPANT_DOCK_BPS}.`);
  assert(burn <= MAX_BURN_BPS, `burnBps cannot exceed ${MAX_BURN_BPS}.`);
  assert(dock + burn <= MAX_TOTAL_REDUCTION_BPS, `participant dock plus burn cannot exceed ${MAX_TOTAL_REDUCTION_BPS}.`);
  assert(typeof policyVersion === "string" && policyVersion.trim().length > 0, "policyVersion is required.");

  const gross = (care * BigInt(rate)) / BASIS_POINTS;
  assert(gross > 0n, "conversion produced no AMITY units; increase the CARE amount or conversion rate.");

  const reserve = (gross * BigInt(dock)) / BASIS_POINTS;
  const burned = (gross * BigInt(burn)) / BASIS_POINTS;
  const net = gross - reserve - burned;
  assert(net > 0n, "conversion would produce no participant AMITY.");
  assert(net * BASIS_POINTS >= gross * BigInt(BASIS_POINTS - BigInt(MAX_TOTAL_REDUCTION_BPS)), "conversion would reduce the gross entitlement by more than 35%.");

  return Object.freeze({
    policyVersion,
    profile: selected.name,
    careAmount: care.toString(),
    conversionRateBps: rate,
    grossAmity: gross.toString(),
    participantDockBps: dock,
    reserveAmity: reserve.toString(),
    burnBps: burn,
    burnedAmity: burned.toString(),
    totalReductionBps: dock + burn,
    netAmity: net.toString(),
    usesPriceOracle: false,
    appliedAtConversion: true,
  });
}
