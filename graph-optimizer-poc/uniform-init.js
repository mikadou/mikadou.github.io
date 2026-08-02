'use strict';

// Make the untrained neural proposer exactly equivalent to uniform random
// mutation. The final action projection starts at zero, so all legal mutations
// have equal logits. Both SA variants also use the same one-draw action mapping,
// which makes their untrained trajectories identical for a shared seed.

function uniformMutationFromUnit(problem, values, unit) {
  const alternativesPerNode = problem.domainSize - 1;
  const legalActionCount = problem.n * alternativesPerNode;
  const ordinal = Math.min(
    legalActionCount - 1,
    Math.floor(unit * legalActionCount)
  );
  const storageIndex = Math.floor(ordinal / alternativesPerNode);
  const valueOrdinal = ordinal % alternativesPerNode;
  const currentValue = values[storageIndex];
  const value = valueOrdinal >= currentValue
    ? valueOrdinal + 1
    : valueOrdinal;
  return storageIndex * problem.domainSize + value;
}

function sampleUniformMutationAction(problem, values, rng) {
  return uniformMutationFromUnit(problem, values, rng.next());
}

function sampleNeuralMutationAction(problem, values, logits, temperature, rng) {
  // This explicit branch is both a safety invariant and an exact baseline test:
  // reset + no training must reproduce random-mutation SA proposal-for-proposal.
  if (state.trainedEpisodes === 0) {
    return sampleUniformMutationAction(problem, values, rng);
  }

  const data = logits.dataSync();
  const safeTemperature = Math.max(0.05, temperature);
  let maxLogit = Number.NEGATIVE_INFINITY;

  for (let node = 0; node < problem.n; node++) {
    const offset = node * problem.domainSize;
    for (let value = 0; value < problem.domainSize; value++) {
      if (value === values[node]) continue;
      const logit = data[offset + value];
      if (Number.isFinite(logit) && logit > maxLogit) maxLogit = logit;
    }
  }

  if (!Number.isFinite(maxLogit)) {
    return sampleUniformMutationAction(problem, values, rng);
  }

  let totalWeight = 0;
  for (let node = 0; node < problem.n; node++) {
    const offset = node * problem.domainSize;
    for (let value = 0; value < problem.domainSize; value++) {
      if (value === values[node]) continue;
      const scaled = (data[offset + value] - maxLogit) / safeTemperature;
      const weight = Math.exp(Math.max(-80, Math.min(0, scaled)));
      if (Number.isFinite(weight)) totalWeight += weight;
    }
  }

  if (!(totalWeight > 0) || !Number.isFinite(totalWeight)) {
    return sampleUniformMutationAction(problem, values, rng);
  }

  const threshold = rng.next() * totalWeight;
  let cumulative = 0;
  let fallback = 0;

  for (let node = 0; node < problem.n; node++) {
    const offset = node * problem.domainSize;
    for (let value = 0; value < problem.domainSize; value++) {
      if (value === values[node]) continue;
      fallback = offset + value;
      const scaled = (data[offset + value] - maxLogit) / safeTemperature;
      const weight = Math.exp(Math.max(-80, Math.min(0, scaled)));
      if (!Number.isFinite(weight)) continue;
      cumulative += weight;
      if (cumulative > threshold) return offset + value;
    }
  }

  return fallback;
}

function zeroInitialActionProjection(policy) {
  tf.tidy(() => {
    policy.wAction2.assign(tf.zerosLike(policy.wAction2));
    policy.bAction2.assign(tf.zerosLike(policy.bAction2));
  });
  policy.rewardBaseline = Number.NaN;
}

const uniformOriginalResetModel = resetModel;
resetModel = function resetUniformProposer() {
  uniformOriginalResetModel();
  zeroInitialActionProjection(state.policy);
  state.trainedEpisodes = 0;
  ui.policyStateBadge.textContent = 'Uniform init';
  ui.policyStateBadge.classList.add('muted');
  ui.compareBtn.disabled = false;
  ui.benchmarkBtn.disabled = false;
  ui.statusText.textContent =
    'Model reset to uniform mutation. Without training, both SA runs use identical proposals and acceptance decisions.';
};

const uniformOriginalClearResults = clearResults;
clearResults = function clearUniformResults() {
  uniformOriginalClearResults();
  ui.benchmarkSummary.textContent =
    'Benchmark immediately to verify untrained parity, or train the neural proposer first.';
};

runNeuralAnnealing = async function runUniformInitializedNeuralAnnealing(
  problem,
  requestedMaxSteps,
  seed = problem.seed
) {
  const maxSteps = Math.max(1, Math.floor(requestedMaxSteps));
  const proposalRng = new SeededRandom(seed ^ 0x13579BDF);
  const acceptanceRng = new SeededRandom(seed ^ 0x2468ACE0);
  const values = problem.initialValues.slice();
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let bestValues = values.slice();
  const history = [currentEnergy];
  const actionHistory = [];
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let lastNodeId = null;
  let lastAccepted = true;
  let lastDelta = 0;
  let acceptedCount = 0;
  let steps = 0;
  const started = performance.now();

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    if (state.stopRequested) break;

    const temperature = annealingTemperature(problem, step, maxSteps);
    const context = { temperature, lastAccepted, lastDelta };
    const output = tf.tidy(() =>
      state.policy.forward(problem, values, hidden, step, maxSteps, context)
    );
    const actionIndex = sampleNeuralMutationAction(
      problem,
      values,
      output.logits,
      EVAL_PROPOSAL_TEMPERATURE,
      proposalRng
    );

    hidden.dispose();
    hidden = output.hidden;
    output.logits.dispose();

    actionHistory.push(actionIndex);
    const action = decodeAction(problem, actionIndex);
    const oldValue = values[action.storageIndex];
    values[action.storageIndex] = action.value;
    const candidateEnergy = energy(problem, values);
    const delta = candidateEnergy - currentEnergy;
    const acceptanceProbability = annealingAcceptanceProbability(delta, temperature);
    const accepted = acceptanceRng.next() < acceptanceProbability;

    if (accepted) {
      currentEnergy = candidateEnergy;
      acceptedCount++;
      lastNodeId = action.nodeId;
      if (currentEnergy < bestEnergy) {
        bestEnergy = currentEnergy;
        bestValues = values.slice();
      }
    } else {
      values[action.storageIndex] = oldValue;
    }

    lastAccepted = accepted;
    lastDelta = delta;
    steps++;
    history.push(currentEnergy);
    if (steps % 20 === 0) await tf.nextFrame();
  }

  const hiddenNorms = hiddenNormsFromTensor(hidden, problem.n);
  hidden.dispose();

  const result = {
    values: bestValues,
    energy: bestEnergy,
    steps,
    accepted: acceptedCount,
    acceptanceRate: steps ? acceptedCount / steps : 0,
    runtimeMs: performance.now() - started,
    success: bestEnergy === 0,
    history,
    actionHistory,
    hiddenNorms,
    lastNodeId
  };
  uniformLastNeuralResult = result;
  return result;
};

runRandomAnnealing = function runSharedUniformAnnealing(
  problem,
  requestedMaxSteps,
  seed = problem.seed
) {
  const maxSteps = Math.max(1, Math.floor(requestedMaxSteps));
  const proposalRng = new SeededRandom(seed ^ 0x13579BDF);
  const acceptanceRng = new SeededRandom(seed ^ 0x2468ACE0);
  const values = problem.initialValues.slice();
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let bestValues = values.slice();
  const history = [currentEnergy];
  const actionHistory = [];
  let lastNodeId = null;
  let acceptedCount = 0;
  let steps = 0;
  const started = performance.now();

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    if (state.stopRequested) break;

    const actionIndex = sampleUniformMutationAction(problem, values, proposalRng);
    actionHistory.push(actionIndex);
    const action = decodeAction(problem, actionIndex);
    const oldValue = values[action.storageIndex];
    values[action.storageIndex] = action.value;
    const candidateEnergy = energy(problem, values);
    const delta = candidateEnergy - currentEnergy;
    const temperature = annealingTemperature(problem, step, maxSteps);
    const acceptanceProbability = annealingAcceptanceProbability(delta, temperature);
    const accepted = acceptanceRng.next() < acceptanceProbability;

    if (accepted) {
      currentEnergy = candidateEnergy;
      acceptedCount++;
      lastNodeId = action.nodeId;
      if (currentEnergy < bestEnergy) {
        bestEnergy = currentEnergy;
        bestValues = values.slice();
      }
    } else {
      values[action.storageIndex] = oldValue;
    }

    steps++;
    history.push(currentEnergy);
  }

  const result = {
    values: bestValues,
    energy: bestEnergy,
    steps,
    accepted: acceptedCount,
    acceptanceRate: steps ? acceptedCount / steps : 0,
    runtimeMs: performance.now() - started,
    success: bestEnergy === 0,
    history,
    actionHistory,
    lastNodeId
  };
  uniformLastRandomResult = result;
  return result;
};

let uniformLastNeuralResult = null;
let uniformLastRandomResult = null;

function arraysEqual(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

const uniformOriginalRunComparison = runComparison;
runComparison = async function runComparisonWithoutTrainingRequirement() {
  const wasUntrained = state.trainedEpisodes === 0;
  const savedEpisodes = state.trainedEpisodes;
  if (wasUntrained) state.trainedEpisodes = 1;
  uniformLastNeuralResult = null;
  uniformLastRandomResult = null;
  try {
    await uniformOriginalRunComparison();
  } finally {
    state.trainedEpisodes = savedEpisodes;
  }

  if (wasUntrained && uniformLastNeuralResult && uniformLastRandomResult) {
    const identical =
      arraysEqual(uniformLastNeuralResult.actionHistory, uniformLastRandomResult.actionHistory) &&
      arraysEqual(uniformLastNeuralResult.history, uniformLastRandomResult.history) &&
      arraysEqual(uniformLastNeuralResult.values, uniformLastRandomResult.values) &&
      uniformLastNeuralResult.accepted === uniformLastRandomResult.accepted &&
      uniformLastNeuralResult.steps === uniformLastRandomResult.steps;
    setStatus(
      identical
        ? 'Untrained parity verified: proposals, accept/reject decisions, energy trajectory, and final assignment are identical. Runtime differs because the neural forward pass still runs.'
        : 'Untrained parity check failed. Reset the model and rerun; this indicates a reproducibility bug.',
      !identical
    );
  }
};

const uniformOriginalRunBenchmark = runBenchmark;
runBenchmark = async function runBenchmarkWithoutTrainingRequirement() {
  const savedEpisodes = state.trainedEpisodes;
  if (savedEpisodes === 0) state.trainedEpisodes = 1;
  try {
    await uniformOriginalRunBenchmark();
  } finally {
    state.trainedEpisodes = savedEpisodes;
  }
};

setBusy = function setBusyWithUntrainedEvaluation(busy) {
  state.busy = busy;
  ui.newProblemBtn.disabled = busy;
  ui.trainBtn.disabled = busy;
  ui.compareBtn.disabled = busy;
  ui.benchmarkBtn.disabled = busy;
  ui.resetModelBtn.disabled = busy;
  ui.stopBtn.disabled = !busy;
};

// runtime.js registered listeners with the original function objects. Replace
// those listeners so reset and untrained evaluation use the parity-preserving
// implementations above.
ui.compareBtn.removeEventListener('click', uniformOriginalRunComparison);
ui.compareBtn.addEventListener('click', runComparison);
ui.benchmarkBtn.removeEventListener('click', uniformOriginalRunBenchmark);
ui.benchmarkBtn.addEventListener('click', runBenchmark);
ui.resetModelBtn.removeEventListener('click', uniformOriginalResetModel);
ui.resetModelBtn.addEventListener('click', resetModel);

// initialize() may still be awaiting tf.ready(), in which case it will call the
// overridden resetModel. If it already completed, reset once more now.
if (state.policy) resetModel();
