'use strict';

// Corrective patch for the first POC implementation.
// The teacher demonstrates a one-pass constructive algorithm, so training and
// inference now use the same one-visit-per-node, one-reserved-value action space.

const POC_NEGATIVE_MASK = -1e9;

function pocActionMask(problem, visited, reservedValues) {
  const mask = new Float32Array(problem.n * problem.domainSize);
  for (let node = 0; node < problem.n; node++) {
    for (let value = 0; value < problem.domainSize; value++) {
      if (visited[node] || reservedValues.has(value)) {
        mask[node * problem.domainSize + value] = POC_NEGATIVE_MASK;
      }
    }
  }
  return mask;
}

// No-op assignments are valid teacher actions. The old mask made an already
// correct teacher target impossible to learn.
currentValueMask = function currentValueMaskWithoutNoOpBan(problem) {
  return new Float32Array(problem.n * problem.domainSize);
};

RecurrentGraphPolicy.prototype.trainEpisode = function trainConstructiveEpisode(problem, optimizer) {
  const values = problem.initialValues.slice();
  const visited = new Uint8Array(problem.n);
  const reservedValues = new Set();

  const result = tf.tidy(() => tf.variableGrads(() => {
    let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
    let totalLoss = tf.scalar(0);

    for (let depth = 0; depth < problem.n; depth++) {
      const storageIndex = problem.indexById.get(problem.chain[depth]);
      const targetValue = targetValueAtDepth(problem, depth);
      const targetAction = storageIndex * problem.domainSize + targetValue;
      const output = this.forward(problem, values, hidden, depth, problem.n);
      hidden = output.hidden;

      const maskedLogits = output.logits.add(tf.tensor1d(
        pocActionMask(problem, visited, reservedValues)
      ));
      const actionMatrix = maskedLogits.reshape([problem.n, problem.domainSize]);
      const nodeLogits = tf.logSumExp(actionMatrix, 1);
      const selectedValueLogits = actionMatrix.gather([storageIndex])
        .reshape([problem.domainSize]);

      const jointNll = tf.logSoftmax(maskedLogits).gather([targetAction]).squeeze().neg();
      const nodeNll = tf.logSoftmax(nodeLogits).gather([storageIndex]).squeeze().neg();
      const valueNll = tf.logSoftmax(selectedValueLogits).gather([targetValue]).squeeze().neg();
      totalLoss = totalLoss.add(jointNll).add(nodeNll.mul(0.5)).add(valueNll.mul(0.5));

      values[storageIndex] = targetValue;
      visited[storageIndex] = 1;
      reservedValues.add(targetValue);
    }

    return totalLoss.div(problem.n * 2);
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

runLearnedPolicy = async function runConstructiveLearnedPolicy(problem, requestedMaxSteps) {
  const values = problem.initialValues.slice();
  let currentEnergy = energy(problem, values);
  const history = [currentEnergy];
  const visited = new Uint8Array(problem.n);
  const reservedValues = new Set();
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let lastNodeId = null;
  let steps = 0;
  const maxSteps = Math.min(problem.n, requestedMaxSteps);
  const started = performance.now();

  for (let step = 0; step < maxSteps && currentEnergy > 0; step++) {
    if (state.stopRequested) break;
    const mask = pocActionMask(problem, visited, reservedValues);
    const output = tf.tidy(() => {
      const base = state.policy.forward(problem, values, hidden, step, maxSteps);
      return {
        hidden: base.hidden,
        logits: base.logits.add(tf.tensor1d(mask))
      };
    });

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
    visited[action.storageIndex] = 1;
    reservedValues.add(action.value);
    lastNodeId = action.nodeId;
    steps++;
    currentEnergy = energy(problem, values);
    history.push(currentEnergy);
  }

  const hiddenNorms = hiddenNormsFromTensor(hidden, problem.n);
  hidden.dispose();
  return {
    values,
    energy: currentEnergy,
    steps,
    runtimeMs: performance.now() - started,
    success: currentEnergy === 0,
    history,
    hiddenNorms,
    lastNodeId,
    distinctValues: new Set(values).size
  };
};

// Make the budget explicit. The learned policy is now a single constructive
// pass, while SA keeps its much larger proposal budget.
const policyBudgetLabel = ui.policyBudgetMultiplier.closest('label');
if (policyBudgetLabel) {
  const firstTextNode = Array.from(policyBudgetLabel.childNodes)
    .find(node => node.nodeType === Node.TEXT_NODE);
  if (firstTextNode) firstTextNode.textContent = 'Policy action budget ';
  const help = policyBudgetLabel.querySelector('small');
  if (help) help.textContent = 'Exactly 1 × N: each node can be selected once';
}
ui.policyBudgetMultiplier.value = '1';
ui.policyBudgetMultiplier.min = '1';
ui.policyBudgetMultiplier.max = '1';
ui.policyBudgetMultiplier.disabled = true;

if (ui.trainEpisodes.value === '800') ui.trainEpisodes.value = '1600';
if (ui.trainMaxN.value === '12') ui.trainMaxN.value = '16';

const policyCaption = ui.policyGraph.closest('.result-panel')?.querySelector('.caption');
if (policyCaption) {
  policyCaption.textContent = 'Each node is visited at most once; selected values are reserved, so the learned pass cannot assign the same chosen value to several nodes.';
}

const trainingBlock = Array.from(document.querySelectorAll('.explain-grid > div'))
  .find(block => block.querySelector('h3')?.textContent === 'Training signal');
if (trainingBlock) {
  trainingBlock.querySelector('p').innerHTML = 'The teacher always performs one complete head-to-tail pass and assigns <code>2N − 1 − depth</code>. Training uses joint, node-selection, and value-selection losses under the same masks used at inference.';
}
