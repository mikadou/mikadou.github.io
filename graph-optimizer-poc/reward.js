'use strict';

// Reward-only training for a neural mutation proposer embedded inside a fixed
// simulated-annealing loop. The acceptance rule and temperature schedule are
// not learned.

const RL_DISCOUNT = 0.97;
const RL_ENTROPY_COEF = 0.01;
const RL_PROPOSAL_COST = 0.005;
const RL_REJECTION_COST = 0.01;
const RL_BEST_IMPROVEMENT_WEIGHT = 1.5;
const RL_SUCCESS_BONUS = 3.0;
const RL_TERMINAL_PENALTY = 0.25;
const RL_BASELINE_RATE = 0.05;
const EVAL_PROPOSAL_TEMPERATURE = 0.85;

function proposalTrainingTemperature() {
  const progress = Math.min(1, state.trainedEpisodes / 4000);
  return 1.4 - 0.55 * progress;
}

function samplePolicyAction(logits, temperature, seed) {
  const safeTemperature = Math.max(0.05, temperature);
  return tf.tidy(() =>
    tf.multinomial(logits.div(safeTemperature), 1, seed).dataSync()[0]
  );
}

function discountedReturns(rewards) {
  const returns = new Array(rewards.length);
  let running = 0;
  for (let index = rewards.length - 1; index >= 0; index--) {
    running = rewards[index] + RL_DISCOUNT * running;
    returns[index] = running;
  }
  return returns;
}

RecurrentGraphPolicy.prototype.trainEpisode = function trainNeuralSaEpisode(problem, optimizer) {
  const maxSteps = readPositiveInt(ui.trainBudgetMultiplier, 8) * problem.n;
  const proposalTemperature = proposalTrainingTemperature();
  const proposalRng = new SeededRandom(problem.seed ^ 0x1A2B3C4D);
  const acceptanceRng = new SeededRandom(problem.seed ^ 0x5EEDBEEF);
  const values = problem.initialValues.slice();
  const valueStates = [];
  const contexts = [];
  const actions = [];
  const rewards = [];
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let lastAccepted = true;
  let lastDelta = 0;
  let acceptedCount = 0;

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    const temperature = annealingTemperature(problem, step, maxSteps);
    const context = { temperature, lastAccepted, lastDelta };
    const output = tf.tidy(() =>
      this.forward(problem, values, hidden, step, maxSteps, context)
    );
    const actionIndex = samplePolicyAction(
      output.logits,
      proposalTemperature,
      proposalRng.int(0x7fffffff)
    );

    valueStates.push(values.slice());
    contexts.push({ ...context });
    actions.push(actionIndex);

    hidden.dispose();
    hidden = output.hidden;
    output.logits.dispose();

    const previousEnergy = currentEnergy;
    const previousBestEnergy = bestEnergy;
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
      if (currentEnergy < bestEnergy) bestEnergy = currentEnergy;
    } else {
      values[action.storageIndex] = oldValue;
    }

    const currentImprovement = clamp(
      (previousEnergy - currentEnergy) / Math.max(1, problem.n),
      -2,
      2
    );
    const bestImprovement = clamp(
      (previousBestEnergy - bestEnergy) / Math.max(1, problem.n),
      0,
      2
    );
    let reward = currentImprovement +
      RL_BEST_IMPROVEMENT_WEIGHT * bestImprovement -
      RL_PROPOSAL_COST;
    if (!accepted) reward -= RL_REJECTION_COST;
    if (bestEnergy === 0) reward += RL_SUCCESS_BONUS;
    rewards.push(reward);

    lastAccepted = accepted;
    lastDelta = delta;
  }

  hidden.dispose();

  if (actions.length === 0) {
    this.lastRewardStats = {
      episodeReturn: RL_SUCCESS_BONUS,
      finalEnergy: 0,
      success: true,
      steps: 0,
      acceptanceRate: 0
    };
    return 0;
  }

  if (bestEnergy > 0) {
    const normalizedRemaining = Math.min(4, bestEnergy / Math.max(1, problem.n));
    rewards[rewards.length - 1] -= RL_TERMINAL_PENALTY * normalizedRemaining;
  }

  const returns = discountedReturns(rewards);
  const meanReturn = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const variance = returns.reduce(
    (sum, value) => sum + (value - meanReturn) ** 2,
    0
  ) / returns.length;
  const scale = Math.max(0.25, Math.sqrt(variance));

  if (!Number.isFinite(this.rewardBaseline)) this.rewardBaseline = meanReturn;
  const baseline = this.rewardBaseline;
  const advantages = returns.map(value =>
    clamp((value - baseline) / scale, -5, 5)
  );
  this.rewardBaseline =
    (1 - RL_BASELINE_RATE) * this.rewardBaseline +
    RL_BASELINE_RATE * meanReturn;

  const result = tf.tidy(() => tf.variableGrads(() => {
    let replayHidden = tf.zeros([problem.n, HIDDEN_DIM]);
    let totalLoss = tf.scalar(0);

    for (let step = 0; step < actions.length; step++) {
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
      const selectedLogProbability = logProbabilities
        .gather([actions[step]]).squeeze();
      const probabilities = tf.softmax(scaledLogits);
      const entropy = probabilities.mul(logProbabilities).sum().neg();
      const advantage = tf.scalar(advantages[step]);
      const stepLoss = selectedLogProbability.mul(advantage).neg()
        .sub(entropy.mul(RL_ENTROPY_COEF));
      totalLoss = totalLoss.add(stepLoss);
    }

    return totalLoss.div(actions.length);
  }, this.vars));

  const lossValue = result.value.dataSync()[0];
  const clipped = {};
  for (const [name, gradient] of Object.entries(result.grads)) {
    clipped[name] = tf.clipByValue(gradient, -3, 3);
  }
  optimizer.applyGradients(clipped);
  result.value.dispose();
  Object.values(result.grads).forEach(tensor => tensor.dispose());
  Object.values(clipped).forEach(tensor => tensor.dispose());

  this.lastRewardStats = {
    episodeReturn: rewards.reduce((sum, reward) => sum + reward, 0),
    finalEnergy: bestEnergy,
    success: bestEnergy === 0,
    steps: actions.length,
    acceptanceRate: acceptedCount / actions.length
  };

  return lossValue;
};
