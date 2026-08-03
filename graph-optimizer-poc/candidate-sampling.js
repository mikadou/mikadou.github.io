'use strict';

// Scalable candidate approximation for the mutation policy. Candidate actions
// are sampled before action features are built, so training and inference avoid
// constructing the full N x domain action tensor.

const SAMPLED_ACTION_COUNT = 32;

function sampleCandidateActions(problem, values, rng, requestedCount = SAMPLED_ACTION_COUNT) {
  const legalActionCount = problem.n * (problem.domainSize - 1);
  const targetCount = Math.min(legalActionCount, Math.max(1, requestedCount));

  if (targetCount === legalActionCount) {
    const all = [];
    for (let node = 0; node < problem.n; node++) {
      for (let value = 0; value < problem.domainSize; value++) {
        if (value !== values[node]) all.push(node * problem.domainSize + value);
      }
    }
    return all;
  }

  const selected = [];
  const seen = new Set();
  while (selected.length < targetCount) {
    const actionIndex = sampleUniformMutationAction(problem, values, rng);
    if (seen.has(actionIndex)) continue;
    seen.add(actionIndex);
    selected.push(actionIndex);
  }
  return selected;
}

function buildSampledActionFeatures(problem, values, searchContext, candidateActions) {
  const features = new Float32Array(candidateActions.length * ACTION_FEATURE_DIM);
  const valueDenom = Math.max(1, problem.domainMax);
  const energyDenom = Math.max(1, problem.n);

  candidateActions.forEach((actionIndex, position) => {
    const action = decodeAction(problem, actionIndex);
    const currentValue = values[action.storageIndex];
    const diagnostics = localMutationDiagnostics(
      problem,
      values,
      action.storageIndex,
      action.value,
      searchContext
    );
    const difference = action.value - currentValue;
    const offset = position * ACTION_FEATURE_DIM;

    features[offset] = action.value / valueDenom;
    features[offset + 1] = difference / valueDenom;
    features[offset + 2] = Math.abs(difference) / valueDenom;
    features[offset + 3] = clamp(diagnostics.newLeftViolation / valueDenom, 0, 2);
    features[offset + 4] = clamp(diagnostics.newRightViolation / valueDenom, 0, 2);
    features[offset + 5] = clamp(diagnostics.deltaEnergy / energyDenom, -2, 2);
    features[offset + 6] = clamp(diagnostics.candidateEnergy / energyDenom, 0, 4);
    features[offset + 7] = diagnostics.acceptanceProbability;
    features[offset + 8] = clamp(diagnostics.feasibleDistance / valueDenom, 0, 2);
    features[offset + 9] = clamp(diagnostics.bestImprovement / energyDenom, 0, 2);
  });

  return features;
}

function sampleWeightedCandidatePosition(logits, temperature, rng) {
  const data = logits.dataSync();
  const safeTemperature = Math.max(0.05, temperature);
  let maxLogit = Number.NEGATIVE_INFINITY;

  for (const logit of data) {
    if (Number.isFinite(logit) && logit > maxLogit) maxLogit = logit;
  }
  if (!Number.isFinite(maxLogit)) return rng.int(data.length);

  let totalWeight = 0;
  const weights = new Float64Array(data.length);
  for (let index = 0; index < data.length; index++) {
    const scaled = (data[index] - maxLogit) / safeTemperature;
    const weight = Math.exp(clamp(scaled, -80, 0));
    weights[index] = Number.isFinite(weight) ? weight : 0;
    totalWeight += weights[index];
  }
  if (!(totalWeight > 0) || !Number.isFinite(totalWeight)) return rng.int(data.length);

  const threshold = rng.next() * totalWeight;
  let cumulative = 0;
  for (let index = 0; index < weights.length; index++) {
    cumulative += weights[index];
    if (cumulative > threshold) return index;
  }
  return weights.length - 1;
}

function buildSampledCounterfactualUtilities(
  problem,
  values,
  searchContext,
  candidateActions
) {
  const utilities = new Float32Array(candidateActions.length);
  const energyDenom = Math.max(1, problem.n);
  const valueDenom = Math.max(1, problem.domainMax);

  candidateActions.forEach((actionIndex, position) => {
    const action = decodeAction(problem, actionIndex);
    const diagnostics = localMutationDiagnostics(
      problem,
      values,
      action.storageIndex,
      action.value,
      searchContext
    );
    const acceptedCurrentChange =
      diagnostics.acceptanceProbability * diagnostics.deltaEnergy;
    const expectedBestImprovement =
      diagnostics.acceptanceProbability * diagnostics.bestImprovement;
    const expectedSuccess = diagnostics.candidateEnergy === 0
      ? diagnostics.acceptanceProbability * UTILITY_SUCCESS_BONUS
      : 0;
    const rejectionCost =
      (1 - diagnostics.acceptanceProbability) * UTILITY_REJECTION_COST;
    const distanceCost =
      UTILITY_DISTANCE_COST * diagnostics.feasibleDistance / valueDenom;

    utilities[position] =
      -acceptedCurrentChange / energyDenom +
      UTILITY_BEST_IMPROVEMENT_WEIGHT * expectedBestImprovement / energyDenom +
      expectedSuccess -
      rejectionCost -
      distanceCost;
  });

  const mean = utilities.reduce((sum, value) => sum + value, 0) / utilities.length;
  const variance = utilities.reduce((sum, value) => {
    const centered = value - mean;
    return sum + centered * centered;
  }, 0) / utilities.length;
  const scale = Math.max(0.05, Math.sqrt(variance));

  for (let index = 0; index < utilities.length; index++) {
    utilities[index] = clamp((utilities[index] - mean) / scale, -5, 5);
  }
  return utilities;
}

RecurrentGraphPolicy.prototype.forwardCandidates = function forwardCandidates(
  problem,
  values,
  hidden,
  step,
  maxSteps,
  candidateActions,
  saContext = {}
) {
  if (!candidateActions.length) throw new Error('At least one candidate action is required.');

  const n = problem.n;
  const searchContext = resolveSearchContext(
    problem,
    values,
    step,
    maxSteps,
    saContext
  );
  const nodeFeatureData = buildNodeFeatures(
    problem,
    values,
    step,
    maxSteps,
    searchContext
  );
  const actionFeatureData = buildSampledActionFeatures(
    problem,
    values,
    searchContext,
    candidateActions
  );
  const features = tf.tensor2d(nodeFeatureData, [n, FEATURE_DIM]);
  const actionFeatures = tf.tensor2d(
    actionFeatureData,
    [candidateActions.length, ACTION_FEATURE_DIM]
  );
  const predAdj = tf.tensor2d(problem.predMatrix, [n, n]);
  const succAdj = tf.tensor2d(problem.succMatrix, [n, n]);
  const projected = tf.tanh(features.matMul(this.wInput).add(this.bInput));
  let nextHidden = hidden;

  for (let pass = 0; pass < MESSAGE_PASSES; pass++) {
    const predMessage = predAdj.matMul(nextHidden);
    const succMessage = succAdj.matMul(nextHidden);
    const gate = tf.sigmoid(projected.matMul(this.wGateInput)
      .add(predMessage.matMul(this.wGatePred))
      .add(succMessage.matMul(this.wGateSucc))
      .add(nextHidden.matMul(this.wGateSelf))
      .add(this.bGate));
    const candidate = tf.tanh(projected.matMul(this.wCandInput)
      .add(predMessage.matMul(this.wCandPred))
      .add(succMessage.matMul(this.wCandSucc))
      .add(nextHidden.matMul(this.wCandSelf))
      .add(this.bCand));
    nextHidden = gate.mul(candidate)
      .add(tf.scalar(1).sub(gate).mul(nextHidden));
  }

  const nodeContext = tf.concat([nextHidden, projected, features], 1);
  const candidateNodeData = Int32Array.from(
    candidateActions,
    actionIndex => Math.floor(actionIndex / problem.domainSize)
  );
  const candidateNodes = tf.tensor1d(candidateNodeData, 'int32');
  const selectedNodeContext = nodeContext.gather(candidateNodes);
  const actionInput = tf.concat([selectedNodeContext, actionFeatures], 1);
  const actionHidden = tf.relu(actionInput.matMul(this.wAction1).add(this.bAction1));
  const logits = actionHidden.matMul(this.wAction2).add(this.bAction2)
    .reshape([candidateActions.length]);
  return { hidden: nextHidden, logits };
};

RecurrentGraphPolicy.prototype.trainEpisode = function trainSampledCounterfactualSaEpisode(
  problem,
  optimizer
) {
  const maxSteps = readPositiveInt(ui.trainBudgetMultiplier, 8) * problem.n;
  const proposalTemperature = proposalTrainingTemperature();
  const klCoefficient = uniformKlCoefficient();
  const candidateRng = new SeededRandom(problem.seed ^ 0x0C0FFEE0);
  const proposalRng = new SeededRandom(problem.seed ^ 0x1A2B3C4D);
  const acceptanceRng = new SeededRandom(problem.seed ^ 0x5EEDBEEF);
  const values = problem.initialValues.slice();
  const valueStates = [];
  const contexts = [];
  const candidateSets = [];
  const utilities = [];
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let lastAccepted = true;
  let lastDelta = 0;
  let recentAcceptance = 1;
  let stagnation = 0;
  let acceptedCount = 0;
  let selectedUtilitySum = 0;

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    if (step > 0 && step % POLICY_MEMORY_WINDOW === 0) {
      hidden.dispose();
      hidden = tf.zeros([problem.n, HIDDEN_DIM]);
    }

    const temperature = annealingTemperature(problem, step, maxSteps);
    const context = {
      temperature,
      currentEnergy,
      bestEnergy,
      violationCount: violationCount(problem, values),
      lastAccepted,
      lastDelta,
      recentAcceptance,
      stagnation
    };
    const resolvedContext = resolveSearchContext(
      problem,
      values,
      step,
      maxSteps,
      context
    );
    const candidates = sampleCandidateActions(problem, values, candidateRng);
    const utilityData = buildSampledCounterfactualUtilities(
      problem,
      values,
      resolvedContext,
      candidates
    );
    const output = tf.tidy(() => this.forwardCandidates(
      problem,
      values,
      hidden,
      step,
      maxSteps,
      candidates,
      context
    ));
    const selectedPosition = proposalRng.next() < TRAIN_UNIFORM_EXPLORATION
      ? proposalRng.int(candidates.length)
      : sampleWeightedCandidatePosition(
        output.logits,
        proposalTemperature,
        proposalRng
      );
    const actionIndex = candidates[selectedPosition];

    valueStates.push(values.slice());
    contexts.push({ ...context });
    candidateSets.push(candidates.slice());
    utilities.push(utilityData);
    selectedUtilitySum += utilityData[selectedPosition];

    hidden.dispose();
    hidden = output.hidden;
    output.logits.dispose();

    const action = decodeAction(problem, actionIndex);
    const oldValue = values[action.storageIndex];
    values[action.storageIndex] = action.value;
    const candidateEnergy = energy(problem, values);
    const delta = candidateEnergy - currentEnergy;
    const acceptanceProbability = annealingAcceptanceProbability(delta, temperature);
    const accepted = acceptanceRng.next() < acceptanceProbability;
    const previousBest = bestEnergy;

    if (accepted) {
      currentEnergy = candidateEnergy;
      acceptedCount++;
      if (currentEnergy < bestEnergy) bestEnergy = currentEnergy;
    } else {
      values[action.storageIndex] = oldValue;
    }

    lastAccepted = accepted;
    lastDelta = delta;
    recentAcceptance = 0.90 * recentAcceptance + 0.10 * (accepted ? 1 : 0);
    stagnation = bestEnergy < previousBest ? 0 : stagnation + 1;
  }

  hidden.dispose();

  if (!valueStates.length) {
    this.lastRewardStats = {
      episodeReturn: 0,
      finalEnergy: 0,
      success: true,
      steps: 0,
      acceptanceRate: 0,
      selectedUtility: 0,
      candidateCount: 0
    };
    return 0;
  }

  const result = tf.tidy(() => tf.variableGrads(() => {
    let replayHidden = tf.zeros([problem.n, HIDDEN_DIM]);
    let totalLoss = tf.scalar(0);

    for (let step = 0; step < valueStates.length; step++) {
      if (step > 0 && step % POLICY_MEMORY_WINDOW === 0) {
        replayHidden = tf.zeros([problem.n, HIDDEN_DIM]);
      }
      const candidates = candidateSets[step];
      const output = this.forwardCandidates(
        problem,
        valueStates[step],
        replayHidden,
        step,
        maxSteps,
        candidates,
        contexts[step]
      );
      replayHidden = output.hidden;
      const scaledLogits = output.logits.div(proposalTemperature);
      const logProbabilities = tf.logSoftmax(scaledLogits);
      const probabilities = tf.softmax(scaledLogits);
      const utilityTensor = tf.tensor1d(utilities[step]);
      const expectedUtility = probabilities.mul(utilityTensor).sum();
      const entropy = probabilities.mul(logProbabilities).sum().neg();
      const klToCandidateUniform = tf.scalar(Math.log(candidates.length)).sub(entropy);
      const stepLoss = expectedUtility.neg()
        .add(klToCandidateUniform.mul(klCoefficient));
      totalLoss = totalLoss.add(stepLoss);
    }

    return totalLoss.div(valueStates.length);
  }, this.vars));

  const lossValue = result.value.dataSync()[0];
  const clipped = {};
  for (const [name, gradient] of Object.entries(result.grads)) {
    clipped[name] = tf.clipByValue(gradient, -1, 1);
  }
  optimizer.applyGradients(clipped);
  result.value.dispose();
  Object.values(result.grads).forEach(tensor => tensor.dispose());
  Object.values(clipped).forEach(tensor => tensor.dispose());

  this.lastRewardStats = {
    episodeReturn: selectedUtilitySum / valueStates.length,
    finalEnergy: bestEnergy,
    success: bestEnergy === 0,
    steps: valueStates.length,
    acceptanceRate: acceptedCount / valueStates.length,
    selectedUtility: selectedUtilitySum / valueStates.length,
    candidateCount: candidateSets[0].length
  };

  return lossValue;
};

runNeuralAnnealing = async function runSampledNeuralAnnealing(
  problem,
  requestedMaxSteps,
  seed = problem.seed
) {
  const maxSteps = Math.max(1, Math.floor(requestedMaxSteps));
  const candidateRng = new SeededRandom(seed ^ 0x0C0FFEE0);
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
  let recentAcceptance = 1;
  let stagnation = 0;
  let acceptedCount = 0;
  let steps = 0;
  const started = performance.now();

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    if (state.stopRequested) break;
    if (step > 0 && step % POLICY_MEMORY_WINDOW === 0) {
      hidden.dispose();
      hidden = tf.zeros([problem.n, HIDDEN_DIM]);
    }

    const temperature = annealingTemperature(problem, step, maxSteps);
    const context = {
      temperature,
      currentEnergy,
      bestEnergy,
      violationCount: violationCount(problem, values),
      lastAccepted,
      lastDelta,
      recentAcceptance,
      stagnation
    };
    const candidates = sampleCandidateActions(problem, values, candidateRng);
    const output = tf.tidy(() => state.policy.forwardCandidates(
      problem,
      values,
      hidden,
      step,
      maxSteps,
      candidates,
      context
    ));
    const actionIndex = state.trainedEpisodes === 0
      ? sampleUniformMutationAction(problem, values, proposalRng)
      : candidates[sampleWeightedCandidatePosition(
        output.logits,
        EVAL_PROPOSAL_TEMPERATURE,
        proposalRng
      )];

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
    const previousBest = bestEnergy;

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
    recentAcceptance = 0.90 * recentAcceptance + 0.10 * (accepted ? 1 : 0);
    stagnation = bestEnergy < previousBest ? 0 : stagnation + 1;
    steps++;
    history.push(currentEnergy);
    if (steps % 20 === 0) await tf.nextFrame();
  }

  const hiddenNorms = hiddenNormsFromTensor(hidden, problem.n);
  hidden.dispose();

  return {
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
    lastNodeId,
    candidateCount: Math.min(
      SAMPLED_ACTION_COUNT,
      problem.n * (problem.domainSize - 1)
    )
  };
};
