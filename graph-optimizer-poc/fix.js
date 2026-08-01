'use strict';

// Corrective patch for the first POC implementation.
// The learned optimizer may revisit nodes, reuse values, make mistakes, and
// correct them later. Only exact no-op assignments are masked.

const POC_NEGATIVE_MASK = -1e9;

currentValueMask = function maskOnlyExactNoOps(problem, values) {
  const mask = new Float32Array(problem.n * problem.domainSize);
  for (let node = 0; node < problem.n; node++) {
    mask[node * problem.domainSize + values[node]] = POC_NEGATIVE_MASK;
  }
  return mask;
};

function pocWrongValue(problem, targetValue, rng) {
  let value = rng.int(problem.domainSize);
  if (value === targetValue) value = (value + 1) % problem.domainSize;
  return value;
}

function pocCorrectionExamples(problem) {
  const rng = new SeededRandom(problem.seed ^ 0xC0FFEE);
  const values = problem.initialValues.slice();
  const examples = [];
  let perturbationsRemaining = Math.max(1, Math.ceil(problem.n / 3));
  const maxExamples = problem.n * 2 + perturbationsRemaining;
  let guard = 0;

  while (examples.length < maxExamples && guard < maxExamples * 5) {
    guard++;
    const incorrectDepths = [];
    for (let depth = 0; depth < problem.n; depth++) {
      const storageIndex = problem.indexById.get(problem.chain[depth]);
      if (values[storageIndex] !== targetValueAtDepth(problem, depth)) {
        incorrectDepths.push(depth);
      }
    }

    if (incorrectDepths.length === 0) {
      if (perturbationsRemaining === 0) break;
      const depth = rng.int(problem.n);
      const storageIndex = problem.indexById.get(problem.chain[depth]);
      const targetValue = targetValueAtDepth(problem, depth);
      values[storageIndex] = pocWrongValue(problem, targetValue, rng);
      perturbationsRemaining--;
      continue;
    }

    // Usually repair the first incorrect node along the chain, but sometimes
    // repair another incorrect node so the policy does not depend on one rigid
    // traversal order.
    const chosenDepth = rng.next() < 0.25
      ? incorrectDepths[rng.int(incorrectDepths.length)]
      : incorrectDepths[0];
    const storageIndex = problem.indexById.get(problem.chain[chosenDepth]);
    const targetValue = targetValueAtDepth(problem, chosenDepth);
    examples.push({ values: values.slice(), action: storageIndex * problem.domainSize + targetValue });
    values[storageIndex] = targetValue;

    // Corrupt an already-correct node occasionally. The next teacher actions
    // demonstrate revisiting and repairing previously assigned variables.
    if (perturbationsRemaining > 0 && examples.length > 1 && rng.next() < 0.35) {
      const correctDepths = [];
      for (let depth = 0; depth < problem.n; depth++) {
        const index = problem.indexById.get(problem.chain[depth]);
        if (values[index] === targetValueAtDepth(problem, depth)) correctDepths.push(depth);
      }
      if (correctDepths.length > 0) {
        const depth = correctDepths[rng.int(correctDepths.length)];
        const index = problem.indexById.get(problem.chain[depth]);
        const target = targetValueAtDepth(problem, depth);
        values[index] = pocWrongValue(problem, target, rng);
        perturbationsRemaining--;
      }
    }
  }

  return examples;
}

RecurrentGraphPolicy.prototype.trainEpisode = function trainCorrectionEpisode(problem, optimizer) {
  const examples = pocCorrectionExamples(problem);
  if (examples.length === 0) return 0;

  const result = tf.tidy(() => tf.variableGrads(() => {
    let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
    let totalLoss = tf.scalar(0);
    const maxSteps = Math.max(problem.n * 2, examples.length);

    for (let step = 0; step < examples.length; step++) {
      const example = examples[step];
      const output = this.forward(problem, example.values, hidden, step, maxSteps);
      hidden = output.hidden;

      const storageIndex = Math.floor(example.action / problem.domainSize);
      const targetValue = example.action % problem.domainSize;
      const actionMatrix = output.logits.reshape([problem.n, problem.domainSize]);
      const nodeLogits = tf.logSumExp(actionMatrix, 1);
      const selectedValueLogits = actionMatrix.gather([storageIndex]).reshape([problem.domainSize]);

      const jointNll = tf.logSoftmax(output.logits).gather([example.action]).squeeze().neg();
      const nodeNll = tf.logSoftmax(nodeLogits).gather([storageIndex]).squeeze().neg();
      const valueNll = tf.logSoftmax(selectedValueLogits).gather([targetValue]).squeeze().neg();
      totalLoss = totalLoss.add(jointNll).add(nodeNll.mul(0.5)).add(valueNll.mul(0.5));
    }

    return totalLoss.div(examples.length * 2);
  }, this.vars));

  const lossValue = result.value.dataSync()[0];
  const clipped = {};
  for (const [name, gradient] of Object.entries(result.grads)) {
    clipped[name] = tf.clipByValue(gradient, -5, 5);
  }
  optimizer.applyGradients(clipped);
  result.value.dispose();
  Object.values(result.grads).forEach(tensor => tensor.dispose());
  Object.values(clipped).forEach(tensor => tensor.dispose());
  return lossValue;
};

runLearnedPolicy = async function runRevisableLearnedPolicy(problem, maxSteps) {
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
    const output = tf.tidy(() => state.policy.forward(problem, values, hidden, step, maxSteps));
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
    // Plot the actual current energy so temporary mistakes and later repairs
    // remain visible instead of being hidden by a best-so-far curve.
    history.push(currentEnergy);
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

const policyBudgetLabel = ui.policyBudgetMultiplier.closest('label');
if (policyBudgetLabel) {
  const firstTextNode = Array.from(policyBudgetLabel.childNodes)
    .find(node => node.nodeType === Node.TEXT_NODE);
  if (firstTextNode) firstTextNode.textContent = 'Policy action budget ';
  const help = policyBudgetLabel.querySelector('small');
  if (help) help.textContent = 'Configurable multiplier × N; revisits and corrections are allowed';
}
ui.policyBudgetMultiplier.value = '6';
ui.policyBudgetMultiplier.min = '1';
ui.policyBudgetMultiplier.max = '12';
ui.policyBudgetMultiplier.disabled = false;

if (ui.trainEpisodes.value === '800') ui.trainEpisodes.value = '1600';
if (ui.trainMaxN.value === '12') ui.trainMaxN.value = '16';

const policyCaption = ui.policyGraph.closest('.result-panel')?.querySelector('.caption');
if (policyCaption) {
  policyCaption.textContent = 'Nodes may be revisited and values may be reused. The policy can make an early mistake and revise it during later optimization steps.';
}

const trainingBlock = Array.from(document.querySelectorAll('.explain-grid > div'))
  .find(block => block.querySelector('h3')?.textContent === 'Training signal');
if (trainingBlock) {
  trainingBlock.querySelector('p').innerHTML = 'The teacher repairs incorrect variables toward <code>2N − 1 − depth</code>. Training also injects corrupted previously-correct nodes, so later steps demonstrate revisiting and correcting earlier assignments. Inference masks only exact no-op assignments.';
}
