'use strict';

// Learning-stability patch for the direct actor-critic search.
//
// 1. The critic evaluates a fixed-size summary of the actual post-action state.
// 2. The actor target uses exact rewards plus realized continuation, not the
//    critic's current guesses.
// 3. The critic is trained repeatedly from replayed Monte Carlo returns.

const STABLE_CRITIC_POOL_STATS = 4;
const STABLE_CRITIC_STATE_DIM = FEATURE_DIM * STABLE_CRITIC_POOL_STATS;
const STABLE_REPLAY_CAPACITY = 1024;
const STABLE_REPLAY_BATCH_SIZE = 32;
const STABLE_CRITIC_UPDATES = 4;

function ensureStableActorCritic(policy) {
  ensureActorCriticParameters(policy);
  if (policy.stableActorCriticReady) return;

  policy.wStateCritic1 = policy.add(initVariable(
    [STABLE_CRITIC_STATE_DIM, ACTION_DIM],
    `${policy.prefix}_wStateCritic1`,
    STABLE_CRITIC_STATE_DIM
  ));
  policy.bStateCritic1 = policy.add(tf.variable(
    tf.zeros([ACTION_DIM]),
    true,
    `${policy.prefix}_bStateCritic1`
  ));
  policy.wStateCritic2 = policy.add(tf.variable(
    tf.zeros([ACTION_DIM, 1]),
    true,
    `${policy.prefix}_wStateCritic2`
  ));
  policy.bStateCritic2 = policy.add(tf.variable(
    tf.zeros([1]),
    true,
    `${policy.prefix}_bStateCritic2`
  ));
  policy.stateCriticVars = [
    policy.wStateCritic1,
    policy.bStateCritic1,
    policy.wStateCritic2,
    policy.bStateCritic2
  ];
  policy.stateCriticReplay = [];
  policy.stableActorCriticReady = true;
}

function stableCriticStateFeatures(problem, values, step, maxSteps, rawContext = {}) {
  const context = actorCriticContext(problem, values, step, maxSteps, rawContext);
  const nodeFeatures = buildActorCriticNodeFeatures(
    problem,
    values,
    step,
    maxSteps,
    context
  );
  const pooled = new Float32Array(STABLE_CRITIC_STATE_DIM);

  for (let feature = 0; feature < FEATURE_DIM; feature++) {
    let sum = 0;
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (let node = 0; node < problem.n; node++) {
      const value = nodeFeatures[node * FEATURE_DIM + feature];
      sum += value;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    const mean = sum / Math.max(1, problem.n);
    let variance = 0;
    for (let node = 0; node < problem.n; node++) {
      const centered = nodeFeatures[node * FEATURE_DIM + feature] - mean;
      variance += centered * centered;
    }
    pooled[feature] = mean;
    pooled[FEATURE_DIM + feature] = Number.isFinite(min) ? min : 0;
    pooled[2 * FEATURE_DIM + feature] = Number.isFinite(max) ? max : 0;
    pooled[3 * FEATURE_DIM + feature] = Math.sqrt(
      variance / Math.max(1, problem.n)
    );
  }
  return pooled;
}

function stablePostActionState(
  problem,
  values,
  context,
  actionIndex,
  diagnostics,
  reward,
  nextStep,
  maxSteps
) {
  const nextValues = values.slice();
  const action = decodeAction(problem, actionIndex);
  nextValues[action.storageIndex] = action.value;
  const nextBestEnergy = Math.min(context.bestEnergy, diagnostics.candidateEnergy);
  const improvedBest = nextBestEnergy < context.bestEnergy;
  return {
    nextValues,
    nextContext: {
      currentEnergy: diagnostics.candidateEnergy,
      bestEnergy: nextBestEnergy,
      violationCount: violationCount(problem, nextValues),
      lastDelta: diagnostics.deltaEnergy,
      lastReward: reward,
      recentImprovement: 0.90 * context.recentImprovement + 0.10 * reward,
      stagnation: improvedBest ? 0 : context.stagnation + 1,
      step: nextStep,
      maxSteps
    }
  };
}

RecurrentGraphPolicy.prototype.stateCriticValues = function stateCriticValues(features) {
  ensureStableActorCritic(this);
  const hidden = tf.relu(features.matMul(this.wStateCritic1).add(this.bStateCritic1));
  return hidden.matMul(this.wStateCritic2).add(this.bStateCritic2)
    .reshape([features.shape[0]]);
};

RecurrentGraphPolicy.prototype.scoreActorCriticCandidates = function scorePostActionCandidates(
  problem,
  values,
  encoded,
  candidateActions,
  prepared = null
) {
  ensureStableActorCritic(this);
  if (!candidateActions.length) throw new Error('At least one candidate action is required.');

  const actionData = prepared || candidateDiagnostics(
    problem,
    values,
    encoded.context,
    candidateActions
  );
  const actorOutput = this.actorCandidateLogits(problem, encoded, candidateActions);
  const featureData = new Float32Array(
    candidateActions.length * STABLE_CRITIC_STATE_DIM
  );
  const activeMask = new Float32Array(candidateActions.length);
  const nextStep = encoded.context.step + 1;

  candidateActions.forEach((actionIndex, position) => {
    const postAction = stablePostActionState(
      problem,
      values,
      encoded.context,
      actionIndex,
      actionData.diagnostics[position],
      actionData.immediateRewards[position],
      nextStep,
      encoded.context.maxSteps
    );
    featureData.set(stableCriticStateFeatures(
      problem,
      postAction.nextValues,
      nextStep,
      encoded.context.maxSteps,
      postAction.nextContext
    ), position * STABLE_CRITIC_STATE_DIM);
    const terminal = actionData.diagnostics[position].candidateEnergy === 0 ||
      nextStep >= encoded.context.maxSteps;
    activeMask[position] = terminal ? 0 : 1;
  });

  const features = tf.tensor2d(
    featureData,
    [candidateActions.length, STABLE_CRITIC_STATE_DIM]
  );
  const continuationValues = this.stateCriticValues(features)
    .mul(tf.tensor1d(activeMask));
  return {
    actorLogits: actorOutput.logits,
    continuationValues,
    immediateRewards: actionData.immediateRewards,
    diagnostics: actionData.diagnostics
  };
};

function stableAddReplay(policy, items) {
  for (const item of items) policy.stateCriticReplay.push(item);
  if (policy.stateCriticReplay.length > STABLE_REPLAY_CAPACITY) {
    policy.stateCriticReplay.splice(
      0,
      policy.stateCriticReplay.length - STABLE_REPLAY_CAPACITY
    );
  }
}

function stableSampleReplay(policy, rng) {
  const count = Math.min(STABLE_REPLAY_BATCH_SIZE, policy.stateCriticReplay.length);
  const selected = [];
  const seen = new Set();
  while (selected.length < count) {
    const index = rng.int(policy.stateCriticReplay.length);
    if (seen.has(index)) continue;
    seen.add(index);
    selected.push(policy.stateCriticReplay[index]);
  }
  return selected;
}

function stableHuber(errors) {
  const absolute = errors.abs();
  const quadratic = tf.minimum(absolute, tf.scalar(1));
  return quadratic.square().mul(0.5).add(absolute.sub(quadratic));
}

function stableApplyGradients(optimizer, result) {
  const loss = result.value.dataSync()[0];
  const clipped = {};
  for (const [name, gradient] of Object.entries(result.grads)) {
    clipped[name] = tf.clipByValue(gradient, -1, 1);
  }
  optimizer.applyGradients(clipped);
  disposeGradientResult(result, clipped);
  return loss;
}

RecurrentGraphPolicy.prototype.trainEpisode = function trainStableActorCriticEpisode(
  problem,
  optimizer
) {
  ensureStableActorCritic(this);
  const maxSteps = readPositiveInt(ui.trainBudgetMultiplier, 8) * problem.n;
  const actorRng = new SeededRandom(problem.seed ^ 0xA17C0A11);
  const randomRng = new SeededRandom(problem.seed ^ 0xBADC0FFE);
  const selectionRng = new SeededRandom(problem.seed ^ 0xC011EC70);
  const replayRng = new SeededRandom(problem.seed ^ 0xC71C1C00);
  const values = problem.initialValues.slice();
  const trajectory = [];
  const rewards = [];
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let lastDelta = 0;
  let lastReward = 0;
  let recentImprovement = 0;
  let stagnation = 0;
  let selectedScoreSum = 0;

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    if (step > 0 && step % POLICY_MEMORY_WINDOW === 0) {
      hidden.dispose();
      hidden = tf.zeros([problem.n, HIDDEN_DIM]);
    }
    const context = {
      currentEnergy,
      bestEnergy,
      violationCount: violationCount(problem, values),
      lastDelta,
      lastReward,
      recentImprovement,
      stagnation
    };
    const actorStateValues = values.slice();
    const encoded = tf.tidy(() => this.encodeActorCriticState(
      problem,
      values,
      hidden,
      step,
      maxSteps,
      context
    ));
    const generated = generateActorCriticCandidates(
      this,
      problem,
      values,
      encoded,
      actorRng,
      randomRng
    );
    const prepared = candidateDiagnostics(
      problem,
      values,
      encoded.context,
      generated.candidates
    );
    const output = tf.tidy(() => this.scoreActorCriticCandidates(
      problem,
      values,
      encoded,
      generated.candidates,
      prepared
    ));
    const actorLogits = Array.from(output.actorLogits.dataSync());
    const continuationValues = Array.from(output.continuationValues.dataSync());
    const totalScores = continuationValues.map((continuation, index) =>
      prepared.immediateRewards[index] + ACTOR_CRITIC_DISCOUNT * continuation
    );
    const selectedPosition = selectionRng.next() < TRAIN_ACTION_EXPLORATION
      ? selectionRng.int(generated.candidates.length)
      : sampleFromScores(totalScores, TRAIN_SELECTION_TEMPERATURE, selectionRng);
    const actionIndex = generated.candidates[selectedPosition];
    const reward = prepared.immediateRewards[selectedPosition];
    selectedScoreSum += totalScores[selectedPosition];

    hidden.dispose();
    hidden = encoded.hidden;
    encoded.nodeContext.dispose();
    output.actorLogits.dispose();
    output.continuationValues.dispose();

    const action = decodeAction(problem, actionIndex);
    const previousBest = bestEnergy;
    values[action.storageIndex] = action.value;
    const candidateEnergy = energy(problem, values);
    const delta = candidateEnergy - currentEnergy;
    currentEnergy = candidateEnergy;
    if (currentEnergy < bestEnergy) bestEnergy = currentEnergy;
    lastDelta = delta;
    lastReward = reward;
    recentImprovement = 0.90 * recentImprovement + 0.10 * reward;
    stagnation = bestEnergy < previousBest ? 0 : stagnation + 1;

    const nextContext = {
      currentEnergy,
      bestEnergy,
      violationCount: violationCount(problem, values),
      lastDelta,
      lastReward,
      recentImprovement,
      stagnation
    };
    trajectory.push({
      actorValues: actorStateValues,
      context: { ...context },
      candidates: generated.candidates.slice(),
      actorLogits,
      immediateRewards: Float32Array.from(prepared.immediateRewards),
      selectedPosition,
      selectedCriticPrediction: continuationValues[selectedPosition],
      nextCriticFeatures: stableCriticStateFeatures(
        problem,
        values,
        step + 1,
        maxSteps,
        nextContext
      )
    });
    rewards.push(reward);
  }

  hidden.dispose();
  if (!trajectory.length) {
    this.lastRewardStats = {
      episodeReturn: 0,
      finalEnergy: 0,
      success: true,
      steps: 0,
      acceptanceRate: 1,
      selectedUtility: 0,
      candidateCount: 0,
      actorLoss: 0,
      criticLoss: 0,
      replaySize: this.stateCriticReplay.length
    };
    return 0;
  }

  const continuationTargets = computeContinuationTargets(rewards);
  const replayItems = [];
  let actorLossEstimate = 0;

  trajectory.forEach((item, step) => {
    const targetScores = Array.from(item.immediateRewards);
    targetScores[item.selectedPosition] = clamp(
      rewards[step] + ACTOR_CRITIC_DISCOUNT * continuationTargets[step],
      -RETURN_CLIP,
      RETURN_CLIP
    );
    item.targetPolicy = Float32Array.from(
      softmaxArray(targetScores, ACTOR_TARGET_TEMPERATURE)
    );
    actorLossEstimate += crossEntropy(
      item.targetPolicy,
      item.actorLogits,
      ACTOR_CANDIDATE_TEMPERATURE
    );
    replayItems.push({
      features: item.nextCriticFeatures,
      target: continuationTargets[step]
    });
  });
  actorLossEstimate /= trajectory.length;
  stableAddReplay(this, replayItems);

  const actorResult = tf.tidy(() => tf.variableGrads(() => {
    let replayHidden = tf.zeros([problem.n, HIDDEN_DIM]);
    let totalLoss = tf.scalar(0);
    trajectory.forEach((item, step) => {
      if (step > 0 && step % POLICY_MEMORY_WINDOW === 0) {
        replayHidden = tf.zeros([problem.n, HIDDEN_DIM]);
      }
      const encoded = this.encodeActorCriticState(
        problem,
        item.actorValues,
        replayHidden,
        step,
        maxSteps,
        item.context
      );
      replayHidden = encoded.hidden;
      const actor = this.actorCandidateLogits(problem, encoded, item.candidates);
      const logProbabilities = tf.logSoftmax(
        actor.logits.div(ACTOR_CANDIDATE_TEMPERATURE)
      );
      const probabilities = tf.softmax(
        actor.logits.div(ACTOR_CANDIDATE_TEMPERATURE)
      );
      const target = tf.tensor1d(item.targetPolicy);
      const crossEntropyLoss = target.mul(logProbabilities).sum().neg();
      const entropy = probabilities.mul(logProbabilities).sum().neg();
      totalLoss = totalLoss.add(
        crossEntropyLoss.sub(entropy.mul(ACTOR_ENTROPY_WEIGHT))
      );
    });
    return totalLoss.div(trajectory.length);
  }, this.actorVars));
  const actorTrainingLoss = stableApplyGradients(optimizer, actorResult);

  let criticTrainingLoss = 0;
  let criticUpdates = 0;
  for (let update = 0; update < STABLE_CRITIC_UPDATES; update++) {
    const batch = stableSampleReplay(this, replayRng);
    if (!batch.length) break;
    const featureData = new Float32Array(batch.length * STABLE_CRITIC_STATE_DIM);
    const targetData = new Float32Array(batch.length);
    batch.forEach((item, index) => {
      featureData.set(item.features, index * STABLE_CRITIC_STATE_DIM);
      targetData[index] = item.target;
    });
    const criticResult = tf.tidy(() => tf.variableGrads(() => {
      const features = tf.tensor2d(
        featureData,
        [batch.length, STABLE_CRITIC_STATE_DIM]
      );
      const targets = tf.tensor1d(targetData);
      return stableHuber(this.stateCriticValues(features).sub(targets)).mean();
    }, this.stateCriticVars));
    criticTrainingLoss += stableApplyGradients(optimizer, criticResult);
    criticUpdates++;
  }
  criticTrainingLoss = criticUpdates
    ? criticTrainingLoss / criticUpdates
    : 0;

  const episodeReturn = rewards.reduce((sum, reward) => sum + reward, 0);
  this.lastRewardStats = {
    episodeReturn,
    finalEnergy: bestEnergy,
    success: bestEnergy === 0,
    steps: trajectory.length,
    acceptanceRate: 1,
    selectedUtility: selectedScoreSum / trajectory.length,
    candidateCount: trajectory[0].candidates.length,
    actorLoss: actorLossEstimate,
    criticLoss: criticTrainingLoss,
    meanReward: episodeReturn / trajectory.length,
    replaySize: this.stateCriticReplay.length
  };
  return actorTrainingLoss + criticTrainingLoss;
};

function replaceButtonListener(key, handler) {
  const current = ui[key];
  const replacement = current.cloneNode(true);
  current.replaceWith(replacement);
  ui[key] = replacement;
  replacement.addEventListener('click', handler);
}

const resetBaseModel = resetModel;

function resetActorCriticModel() {
  resetBaseModel();
  ui.backendNotice.textContent =
    `TensorFlow.js ready · backend: ${tf.getBackend()} · post-action actor–critic training`;
  ui.policyStateBadge.textContent = 'Untrained actor–critic';
  ui.policyStateBadge.classList.add('muted');
  ui.statusText.textContent =
    'Actor starts uniform and the post-action critic starts at zero. Untrained selection is exact one-step hill climbing over the sampled candidates.';
}

const createProblemBase = createCurrentProblem;

function createActorCriticProblem() {
  createProblemBase();
  setStatus(
    `New N=${state.currentProblem.n} problem created. Actor–critic search and random SA share this initial assignment.`
  );
  ui.benchmarkSummary.className = 'benchmark-empty';
  ui.benchmarkSummary.textContent =
    'Train the actor–critic, compare once, or benchmark 30 paired instances.';
}

resetModel = resetActorCriticModel;
createCurrentProblem = createActorCriticProblem;

replaceButtonListener('newProblemBtn', createActorCriticProblem);
replaceButtonListener('trainBtn', trainActorCriticPolicy);
replaceButtonListener('compareBtn', runActorCriticComparison);
replaceButtonListener('benchmarkBtn', runActorCriticBenchmark);
replaceButtonListener('resetModelBtn', resetActorCriticModel);

const currentTestN = ui.testN;
const replacementTestN = currentTestN.cloneNode(true);
currentTestN.replaceWith(replacementTestN);
ui.testN = replacementTestN;
replacementTestN.addEventListener('change', createActorCriticProblem);

ui.policyStateBadge.textContent = state.trainedEpisodes > 0
  ? `${state.trainedEpisodes} actor–critic episodes`
  : 'Untrained actor–critic';
ui.backendNotice.textContent = tf && tf.getBackend
  ? `TensorFlow.js ready · backend: ${tf.getBackend()} · post-action actor–critic training`
  : ui.backendNotice.textContent;
if (state.trainedEpisodes === 0) {
  ui.statusText.textContent =
    'Actor starts uniform and the post-action critic starts at zero. Train the actor–critic or compare its hill-climb bootstrap against random SA.';
}
