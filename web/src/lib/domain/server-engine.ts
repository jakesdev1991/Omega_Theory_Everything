// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import "server-only";

import { TriTokenEngine } from "./engine";

/**
 * The shared in-memory economy engine used by every /api/economy route.
 *
 * It survives hot reloads through globalThis so the Test Console sees one
 * consistent ledger while developing, and it is created in test mode whenever
 * the deployment is not a production build (or ECONOMY_TEST_MODE=1 opts in
 * explicitly). A production build without ECONOMY_TEST_MODE therefore gets a
 * fail-closed engine: no faucet, no bootstrap voting power, no test credits.
 */

const globalForEngine = globalThis as unknown as {
  omegaEconomyEngine?: TriTokenEngine;
};

export function economyTestModeEnabled(): boolean {
  if (process.env.ECONOMY_TEST_MODE === "1") return true;
  if (process.env.ECONOMY_TEST_MODE === "0") return false;
  return process.env.NODE_ENV !== "production";
}

function createEngine(): TriTokenEngine {
  return new TriTokenEngine({
    testMode: economyTestModeEnabled(),
    // The shared dev engine keeps the historical bootstrap so the economy page
    // workbench stays usable; scenarios and governance checks that must fail
    // closed construct their own engine with bootstrapVotingPower: false.
    bootstrapVotingPower: true,
  });
}

export function getEngine(): TriTokenEngine {
  if (!globalForEngine.omegaEconomyEngine) {
    globalForEngine.omegaEconomyEngine = createEngine();
  }
  return globalForEngine.omegaEconomyEngine;
}

export function resetEngine(): TriTokenEngine {
  globalForEngine.omegaEconomyEngine = createEngine();
  return globalForEngine.omegaEconomyEngine;
}

/** Guards destructive or minting routes outside test deployments. */
export function assertEconomyTestMode(): void {
  if (!economyTestModeEnabled()) {
    throw new Error(
      "Economy test controls are disabled in this deployment. Set ECONOMY_TEST_MODE=1 to enable the faucet, credits, and reset."
    );
  }
}
