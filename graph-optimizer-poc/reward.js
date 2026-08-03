'use strict';

// Objective-derived counterfactual training for the neural mutation proposer.
// Every visited SA state evaluates all legal mutations with the real energy
// function and fixed Metropolis rule. There are no target assignments or
// teacher trajectories.

const TRAIN_UNIFORM_EXPLORATION = 0.10;
const UTILITY_BEST_IMPROVEMENT_WEIGHT = 1.5;
const UTILITY_SUCCESS_BONUS = 3.0;
const UTILITY_REJECTION_COST = 0.01;
const UTILITY_DISTANCE_COST = 0.03;
const EVAL_PROPOSAL_TEMPERATURE = 0.85;

function proposalTrainingTemperature() {
  const progress = Math.min(1, state.trainedEpisodes / 3000);
  return 1.15 - 0.30 * progress;
}

function uniformKlCoefficient() {
  const progress = Math.min(1, state.trainedEpisodes / 3000);
  return 0.08 - 0.06 * progress;
}

function buildCounterfactualUtilities(problem, values, searchContext) {
  const actionCount = problem.n * problem.domainSize;
  const utilities = new Float32Array(actionCount);
  const legal = [];
  const energyDenom = Math.max(1, problem.n);
  const valueDenom = Math.max(1, problem.domainMax);

  for (let node = 0; node < problem.n; node++) {
    for (let candidate = 0; candidate < problem.domainSize; candidate++) {
      const actionIndex = node * problem.domainSize + candidate;
      if (candidate === values[node]) continue;

      const diagnostics = localMutationDiagnostics(
        problem,
        values,
        node,
        candidate,
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

      const utility =
        -acceptedCurrentChange / energyDenom +
        UTILITY_BEST_IMPROVEMENT_WEIGHT * expectedBestImprovement / energyDenom +
        expectedSuccess -
        rejectionCost -
        distanceCost;
      utilities[actionIndex] = utility;
      legal.push(actionIndex);
    }
  }

  if (!legal.length) return utilities;

  const mean = legal.reduce((sum, index) => sum + utilities[index], 0) / legal.length;
  const variance = legal.reduce((sum, index) => {
    const centered = utilities[index] - mean;
    return sum + centered * centered;
  }, 0) / legal.length;
  const scale = Math.max(0.05, Math.sqrt(variance));

  for (const index of legal) {
    utilities[index] = clamp((utilities[index] - mean) / scale, -5, 5);
  }
  return utilities;
}

RecurrentGraphPolicy.prototype.trainEpisode = function trainCounterfactualSaEpisode(
  problem,
  optimizer
) {
  const maxSteps = readPositiveInt(ui.trainBudgetMultiplier, 8) * problem.n;
  const proposalTemperature = proposalTrainingTemperature();
  const klCoefficient = uniformKlCoefficient();
  const proposalRng = new SeededRandom(problem.seed ^ 0x1A2B3C4D);
  const acceptanceRng = new SeededRandom(problem.seed ^ 0x5EEDBEEF);
  const values = problem.initialValues.slice();
  const valueStates = [];
  const contexts = [];
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
    const utilityData = buildCounterfactualUtilities(
      problem,
      values,
      resolvedContext
    );
    const output = tf.tidy(() =>
      this.forward(problem, values, hidden, step, maxSteps, context)
    );
    const actionIndex = proposalRng.next() < TRAIN_UNIFORM_EXPLORATION
      ? sampleUniformMutationAction(problem, values, proposalRng)
      : sampleWeightedMutationAction(
        problem,
        values,
        output.logits,
        proposalTemperature,
        proposalRng
      );

    valueStates.push(values.slice());
    contexts.push({ ...context });
    utilities.push(utilityData);
    selectedUtilitySum += utilityData[actionIndex];

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
      selectedUtility: 0
    };
    return 0;
  }

  const legalActionCount = problem.n * (problem.domainSize - 1);
  const logLegalActionCount = Math.log(Math.max(1, legalActionCount));
  const result = tf.tidy(() => tf.variableGrads(() => {
    let replayHidden = tf.zeros([problem.n, HIDDEN_DIM]);
    let totalLoss = tf.scalar(0);

    for (let step = 0; step < valueStates.length; step++) {
      if (step > 0 && step % POLICY_MEMORY_WINDOW === 0) {
        replayHidden = tf.zeros([problem.n, HIDDEN_DIM]);
      }
      const output = this.forward(
        problem,
        valueStates[step],
        replayHidden,
        step,
        maxSteps,
        contexts[step]
      );
      replayHidden = output.hidden;
      const scaledLogits = output.logits.div(proposalTemperature);
      const logProbabilities = tf.logSoftmax(scaledLogits);
      const probabilities = tf.softmax(scaledLogits);
      const utilityTensor = tf.tensor1d(utilities[step]);
      const expectedUtility = probabilities.mul(utilityTensor).sum();
      const entropy = probabilities.mul(logProbabilities).sum().neg();
      const klToUniform = tf.scalar(logLegalActionCount).sub(entropy);
      const stepLoss = expectedUtility.neg().add(klToUniform.mul(klCoefficient));
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
    selectedUtility: selectedUtilitySum / valueStates.length
  };

  return lossValue;
};
