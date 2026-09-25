import test from "node:test";
import assert from "node:assert/strict";

import { runAllScenarios, runScenario, SCENARIOS, scenarioSummaries } from "./scenarios";

test("The economy scenario suite covers every documented integration slice", () => {
  const slices = scenarioSummaries().map((summary) => summary.slice).sort();
  for (const expected of ["A", "B", "C", "D", "E", "F", "G", "R1", "R2", "X"]) {
    assert.ok(slices.includes(expected), `slice ${expected} is covered`);
  }
});

test("Every scenario passes, step by step", () => {
  for (const definition of SCENARIOS) {
    const result = runScenario(definition);
    for (const step of result.steps) {
      assert.ok(step.passed, `${step.id}: ${step.actual}`);
    }
    assert.ok(result.passed, `${result.id} failed`);
    assert.ok(result.steps.length > 0, `${result.id} has steps`);
  }
});

test("The full suite reports green with a complete step count", () => {
  const suite = runAllScenarios();
  assert.equal(suite.failed, 0, JSON.stringify(suite.results.filter((r) => !r.passed), null, 2));
  assert.ok(suite.stepTotal >= 30, `expected a comprehensive suite, got ${suite.stepTotal} steps`);
  assert.equal(suite.stepPassed, suite.stepTotal);
  assert.ok(suite.ok);
});
