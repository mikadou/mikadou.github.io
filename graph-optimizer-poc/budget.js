'use strict';

// The optimizer still needs a stopping condition, but the user—not the POC—
// chooses the action-budget multiplier. The original implementation clamps the
// multiplier to 12 in runComparison() and runBenchmark(); this compatibility
// wrapper removes only that specific upper clamp.
const pocOriginalClamp = clamp;
clamp = function userControlledPolicyBudget(value, min, max) {
  if (min === 1 && max === 12) {
    return Math.max(min, Number.isFinite(value) ? Math.floor(value) : min);
  }
  return pocOriginalClamp(value, min, max);
};

ui.policyBudgetMultiplier.removeAttribute('max');
ui.policyBudgetMultiplier.disabled = false;

const pocBudgetLabel = ui.policyBudgetMultiplier.closest('label');
if (pocBudgetLabel) {
  const help = pocBudgetLabel.querySelector('small');
  if (help) {
    help.textContent = 'User-selected multiplier × N; no hardcoded upper limit';
  }
}

// Keep long runs interruptible. This replaces the policy runner from fix.js
// with the same behavior plus a periodic browser yield.
runLearnedPolicy = async function runUserBudgetLearnedPolicy(problem, requestedMaxSteps) {
  const maxSteps = Math.max(1, Number.isFinite(requestedMaxSteps)
    ? Math.floor(requestedMaxSteps)
    : problem.n);
  const values = problem.initialValues.slice();
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let bestValues = values.slice();
  const history = [currentEnergy];
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let lastNodeId = null;
  let steps = 0;
  const started = performance.now();

  for (let step = 0; step < maxSteps && currentEnergy > 0; step++) {
    if (state.stopRequested) break;

    const output = tf.tidy(() =>
      state.policy.forward(problem, values, hidden, step, maxSteps)
    );
    const logitsData = output.logits.dataSync();
    let actionIndex = 0;
    let bestLogit = Number.NEGATIVE_INFINITY;

    for (let i = 0; i < logitsData.length; i++) {
      if (logitsData[i] > bestLogit) {
        bestLogit = logitsData[i];
        actionIndex = i;
      }
    }

    hidden.dispose();
    hidden = output.hidden;
    output.logits.dispose();

    const action = applyAction(problem, values, actionIndex);
    lastNodeId = action.nodeId;
    steps++;
    currentEnergy = energy(problem, values);

    if (currentEnergy < bestEnergy) {
      bestEnergy = currentEnergy;
      bestValues = values.slice();
    }

    history.push(currentEnergy);

    // Yield often enough that rendering and the Stop button remain responsive
    // when the user chooses a very large budget.
    if (steps % 20 === 0) await tf.nextFrame();
  }

  const hiddenNorms = hiddenNormsFromTensor(hidden, problem.n);
  hidden.dispose();

  return {
    values: bestValues,
    energy: bestEnergy,
    steps,
    runtimeMs: performance.now() - started,
    success: bestEnergy === 0,
    history,
    hiddenNorms,
    lastNodeId,
    distinctValues: new Set(bestValues).size
  };
};
