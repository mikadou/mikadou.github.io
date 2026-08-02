'use strict';

// Reward-only training override. This replaces the legacy supervised
// RecurrentGraphPolicy.trainEpisode method before the user can start training.
// No target assignment, node depth, constructive solution, or teacher action is
// used by this training path.

const RL_DISCOUNT = 0.97;
const RL_ENTROPY_COEF = 0.01;
const RL_STEP_COST = 0.01;
const RL_SUCCESS_BONUS = 3.0;
const RL_TERMINAL_PENALTY = 0.25;
const RL_BASELINE_RATE = 0.05;

const rlTrainBudgetInput = document.getElementById('trainBudgetMultiplier');

function rlPositiveInt(input, fallback) {
  const parsed = Number.parseInt(input?.value, 10);
  const value = Math.max(1, Number.isFinite(parsed) ? parsed : fallback);
  if (input) input.value = String(value);
  return value;
}

function rlTemperature() {
  const progress = Math.min(1, state.trainedEpisodes / 3000);
  return 1.5 - 0.6 * progress;
}

function rlSampleAction(logits, temperature) {
  return tf.tidy(() => tf.multinomial(logits.div(temperature), 1).dataSync()[0]);
}

function rlDiscountedReturns(rewards) {
  const returns = new Array(rewards.length);
  let running = 0;
  for (let index = rewards.length - 1; index >= 0; index--) {
    running = rewards[index] + RL_DISCOUNT * running;
    returns[index] = running;
  }
  return returns;
}

RecurrentGraphPolicy.prototype.trainEpisode = function trainRewardEpisode(problem, optimizer) {
  const maxSteps = rlPositiveInt(rlTrainBudgetInput, 4) * problem.n;
  const temperature = rlTemperature();
  const values = problem.initialValues.slice();
  const valueStates = [];
  const actions = [];
  const rewards = [];
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let currentEnergy = energy(problem, values);

  for (let step = 0; step < maxSteps && currentEnergy > 0; step++) {
    const output = tf.tidy(() => this.forward(problem, values, hidden, step, maxSteps));
    const actionIndex = rlSampleAction(output.logits, temperature);
    valueStates.push(values.slice());
    actions.push(actionIndex);

    hidden.dispose();
    hidden = output.hidden;
    output.logits.dispose();

    const previousEnergy = currentEnergy;
    applyAction(problem, values, actionIndex);
    currentEnergy = energy(problem, values);

    const improvement = clamp(
      (previousEnergy - currentEnergy) / Math.max(1, problem.n),
      -2,
      2
    );
    let reward = improvement - RL_STEP_COST;
    if (currentEnergy === 0) reward += RL_SUCCESS_BONUS;
    rewards.push(reward);
  }

  hidden.dispose();

  if (actions.length === 0) {
    this.lastRewardStats = {
      episodeReturn: RL_SUCCESS_BONUS,
      finalEnergy: 0,
      success: true,
      steps: 0
    };
    return 0;
  }

  if (currentEnergy > 0) {
    const normalizedRemaining = Math.min(4, currentEnergy / Math.max(1, problem.n));
    rewards[rewards.length - 1] -= RL_TERMINAL_PENALTY * normalizedRemaining;
  }

  const returns = rlDiscountedReturns(rewards);
  const meanReturn = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const variance = returns.reduce(
    (sum, value) => sum + (value - meanReturn) ** 2,
    0
  ) / returns.length;
  const scale = Math.max(0.25, Math.sqrt(variance));

  if (!Number.isFinite(this.rewardBaseline)) this.rewardBaseline = meanReturn;
  const baseline = this.rewardBaseline;
  const advantages = returns.map(value => clamp((value - baseline) / scale, -5, 5));
  this.rewardBaseline =
    (1 - RL_BASELINE_RATE) * this.rewardBaseline + RL_BASELINE_RATE * meanReturn;

  const result = tf.tidy(() => tf.variableGrads(() => {
    let replayHidden = tf.zeros([problem.n, HIDDEN_DIM]);
    let totalLoss = tf.scalar(0);

    for (let step = 0; step < actions.length; step++) {
      const output = this.forward(
        problem,
        valueStates[step],
        replayHidden,
        step,
        maxSteps
      );
      replayHidden = output.hidden;
      const scaledLogits = output.logits.div(temperature);
      const logProbabilities = tf.logSoftmax(scaledLogits);
      const selectedLogProbability = logProbabilities.gather([actions[step]]).squeeze();
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
    finalEnergy: currentEnergy,
    success: currentEnergy === 0,
    steps: actions.length
  };

  return lossValue;
};

if (ui.trainEpisodes.value === '800') ui.trainEpisodes.value = '2000';
if (ui.trainMaxN.value === '12') ui.trainMaxN.value = '8';
ui.trainBtn.textContent = 'Train from rewards';

const policyCaption = ui.policyGraph.closest('.result-panel')?.querySelector('.caption');
if (policyCaption) {
  policyCaption.textContent = 'Nodes may be revisited and values may be reused. The policy can make mistakes and correct them later.';
}

const trainingBlock = Array.from(document.querySelectorAll('.explain-grid > div'))
  .find(block => block.querySelector('h3')?.textContent === 'Reward-only learning');
if (trainingBlock) {
  trainingBlock.querySelector('p').innerHTML = 'REINFORCE uses discounted returns and an entropy bonus. Reward comes only from violation-energy change, a small action cost, a feasibility bonus, and a terminal penalty for remaining violations. No target assignment or constructive rule is supplied.';
}
