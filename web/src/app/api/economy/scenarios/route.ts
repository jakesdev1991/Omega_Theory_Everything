import { NextRequest, NextResponse } from "next/server";

import { runAllScenarios, runScenario, SCENARIOS, scenarioSummaries } from "@/lib/domain/scenarios";

/**
 * GET  /api/economy/scenarios          — list the scenario suite
 * GET  /api/economy/scenarios?run=all  — run everything server-side
 * POST /api/economy/scenarios          — run selected scenarios: { ids?: string[] }
 *
 * Each run uses fresh engine instances, so results are deterministic and never
 * mutate the shared workbench ledger.
 */
export async function GET(request: NextRequest) {
  const run = request.nextUrl.searchParams.get("run");
  if (run === "all") {
    return NextResponse.json({ ok: true, suite: runAllScenarios() });
  }
  return NextResponse.json({ ok: true, scenarios: scenarioSummaries(), count: SCENARIOS.length });
}

export async function POST(req: NextRequest) {
  let body: { ids?: string[] };
  try {
    body = await req.json();
  } catch {
    body = {};
  }

  const ids = Array.isArray(body.ids) ? body.ids : [];
  if (ids.length === 0) {
    return NextResponse.json({ ok: true, suite: runAllScenarios() });
  }

  const results = [];
  for (const id of ids) {
    const definition = SCENARIOS.find((scenario) => scenario.id === id);
    if (!definition) {
      return NextResponse.json({ ok: false, error: `Unknown scenario: ${id}` }, { status: 404 });
    }
    results.push(runScenario(definition));
  }

  const stepResults = results.flatMap((result) => result.steps);
  return NextResponse.json({
    ok: true,
    suite: {
      ok: results.every((result) => result.passed),
      ranAt: new Date().toISOString(),
      total: results.length,
      passed: results.filter((result) => result.passed).length,
      failed: results.filter((result) => !result.passed).length,
      stepTotal: stepResults.length,
      stepPassed: stepResults.filter((step) => step.passed).length,
      results,
    },
  });
}
