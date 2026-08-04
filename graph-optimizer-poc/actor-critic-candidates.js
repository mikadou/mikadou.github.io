'use strict';

// Candidate actor-critic graph search.
//
// The actor proposes actions directly from a factorized distribution: first a
// variable node, then a numeric replacement value for that node. It never
// constructs or ranks an intermediate list of joint (node, value) actions.
// Eight uniformly random legal actions are added for exploration, and the
// critic evaluates only the final candidate set.

const ACTOR_CANDIDATE_COUNT = 24;
const RANDOM_CANDIDATE_COUNT = 8;
const TOTAL_CANDIDATE_COUNT = ACTOR_CANDIDATE_COUNT + RANDOM_CANDIDATE_COUNT;
const ACTOR_CRITIC_DISCOUNT = 0.97;
const TERMINAL_SUCCESS_REWARD = 1.0;
const TRAIN_ACTION_EXPLORATION = 0.10;
const ACTOR_CANDIDATE_TEMPERATURE = 1.0;
const TRAIN_SELECTION_TEMPERATURE = 0.70;
const EVAL_SELECTION_TEMPERATURE = 0.20;
const ACTOR_TARGET_TEMPERATURE = 0.35;
const ACTOR_ENTROPY_WEIGHT = 0.01;
const CRITIC_LOSS_WEIGHT = 1.0;
const RETURN_CLIP = 5.0;
const MIN_VALUE_SCALE = 0.05;
const MAX_VALUE_SCALE = 0.75;

function actorTrainingTemperature() {
  const progress = Math.min(1, state.trainedEpisodes / 3000);
  return 1.15 - 0.35 * progress;
}

function actorCriticContext(problem, values, step, maxSteps, context = {}) {
  const currentEnergy = Number.isFinite(context.currentEnergy)
    ? context.currentEnergy
    : energy(problem, values);
  const bestEnergy = Number.isFinite(context.bestEnergy)
    ? context.bestEnergy
    : currentEnergy;
  return {
    currentEnergy,
    bestEnergy,
    violationCount: Number.isFinite(context.violationCount)
      ? context.violationCount
      : violationCount(problem, values),
    lastDelta: Number.isFinite(context.lastDelta) ? context.lastDelta : 0,
    lastReward: Number.isFinite(context.lastReward) ? context.lastReward : 0,
    recentImprovement: clamp(
      Number.isFinite(context.recentImprovement) ? context.recentImprovement : 0,
      -1,
      1
    ),
    stagnation: Math.max(0, Number.isFinite(context.stagnation) ? context.stagnation : 0),
    step,
    maxSteps
  };
}

function buildActorCriticNodeFeatures(problem, values, step, maxSteps, context) {
  const features = new Float32Array(problem.n * FEATURE_DIM);
  const valueDenom = Math.max(1, problem.domainMax);
  const energyDenom = Math.max(1, problem.n);
  const stepProgress = maxSteps > 0 ? step / maxSteps : 0;
  const remainingProgress = 1 - stepProgress;
  const normalizedCurrentEnergy = clamp(context.currentEnergy / energyDenom, 0, 4);
  const normalizedBestEnergy = clamp(context.bestEnergy / energyDenom, 0, 4);
  const normalizedGap = clamp(
    (context.currentEnergy - context.bestEnergy) / energyDenom,
    0,
    4
  );
  const violationFraction = context.violationCount / Math.max(1, problem.n - 1);
  const stagnationRatio = clamp(context.stagnation / Math.max(1, maxSteps), 0, 1);

  for (let storageIndex = 0; storageIndex < problem.n; storageIndex++) {
    const id = problem.nodeIds[storageIndex];
    const depth = problem.chainIndexById.get(id);
    const value = values[storageIndex];
    const hasPred = depth > 0;
    const hasSucc = depth < problem.n - 1;
    const predValue = hasPred
      ? values[problem.indexById.get(problem.chain[depth - 1])]
      : problem.domainMax;
    const succValue = hasSucc
      ? values[problem.indexById.get(problem.chain[depth + 1])]
      : 0;
    const leftMargin = hasPred ? (predValue - value - 1) / valueDenom : 1;
    const rightMargin = hasSucc ? (value - succValue - 1) / valueDenom : 1;
    const incidentViolation =
      (hasPred ? edgeViolation(predValue, value) : 0) +
      (hasSucc ? edgeViolation(value, succValue) : 0);
    const offset = storageIndex * FEATURE_DIM;

    features[offset] = value / valueDenom;
    features[offset + 1] = clamp(leftMargin, -1, 1);
    features[offset + 2] = clamp(rightMargin, -1, 1);
    features[offset + 3] = hasPred ? 1 : 0;
    features[offset + 4] = hasSucc ? 1 : 0;
    features[offset + 5] = clamp(incidentViolation / valueDenom, 0, 2);
    features[offset + 6] = stepProgress;
    features[offset + 7] = remainingProgress;
    features[offset + 8] = 1 / valueDenom;
    features[offset + 9] = 1;
    features[offset + 10] = normalizedCurrentEnergy;
    features[offset + 11] = normalizedBestEnergy;
    features[offset + 12] = normalizedGap;
    features[offset + 13] = violationFraction;
    features[offset + 14] = clamp(context.lastDelta / energyDenom, -2, 2);
    features[offset + 15] = clamp(context.lastReward, -2, 2);
    features[offset + 16] = context.recentImprovement;
    features[offset + 17] = stagnationRatio;
    features[offset + 18] = context.bestEnergy === 0 ? 1 : 0;
  }
  return features;
}

function candidateDiagnostics(problem, values, context, candidateActions) {
  const diagnostics = new Array(candidateActions.length);
  const immediateRewards = new Float32Array(candidateActions.length);
  const energyDenom = Math.max(1, problem.n);

  candidateActions.forEach((actionIndex, position) => {
    const action = decodeAction(problem, actionIndex);
    const local = localMutationDiagnostics(
      problem,
      values,
      action.storageIndex,
      action.value,
      {
        currentEnergy: context.currentEnergy,
        bestEnergy: context.bestEnergy,
        temperature: 1
      }
    );
    let reward = -local.deltaEnergy / energyDenom;
    if (context.currentEnergy > 0 && local.candidateEnergy === 0) {
      reward += TERMINAL_SUCCESS_REWARD;
    }
    immediateRewards[position] = reward;
    diagnostics[position] = local;
  });

  return { diagnostics, immediateRewards };
}

function buildActorCriticActionFeatures(
  problem,
  values,
  context,
  candidateActions,
  prepared = null
) {
  const features = new Float32Array(candidateActions.length * ACTION_FEATURE_DIM);
  const valueDenom = Math.max(1, problem.domainMax);
  const energyDenom = Math.max(1, problem.n);
  const candidateData = prepared || candidateDiagnostics(
    problem,
    values,
    context,
    candidateActions
  );

  candidateActions.forEach((actionIndex, position) => {
    const action = decodeAction(problem, actionIndex);
    const currentValue = values[action.storageIndex];
    const local = candidateData.diagnostics[position];
    const difference = action.value - currentValue;
    const offset = position * ACTION_FEATURE_DIM;

    features[offset] = action.value / valueDenom;
    features[offset + 1] = difference / valueDenom;
    features[offset + 2] = Math.abs(difference) / valueDenom;
    features[offset + 3] = clamp(local.newLeftViolation / valueDenom, 0, 2);
    features[offset + 4] = clamp(local.newRightViolation / valueDenom, 0, 2);
    features[offset + 5] = clamp(local.deltaEnergy / energyDenom, -2, 2);
    features[offset + 6] = clamp(local.candidateEnergy / energyDenom, 0, 4);
    features[offset + 7] = clamp(candidateData.immediateRewards[position], -2, 2);
    features[offset + 8] = clamp(local.feasibleDistance / valueDenom, 0, 2);
    features[offset + 9] = clamp(local.bestImprovement / energyDenom, 0, 2);
  });

  return { features, ...candidateData };
}

function legalActionCount(problem) {
  return problem.n * (problem.domainSize - 1);
}

function allLegalActions(problem, values) {
  const actions = [];
  for (let node = 0; node < problem.n; node++) {
    for (let value = 0; value < problem.domainSize; value++) {
      if (value !== values[node]) actions.push(node * problem.domainSize + value);
    }
  }
  return actions;
}

function sampleUniqueUniformActions(problem, values, rng, requestedCount, excluded = null) {
  const excludedSet = excluded || new Set();
  const available = legalActionCount(problem) - excludedSet.size;
  const targetCount = Math.min(Math.max(0, requestedCount), Math.max(0, available));
  if (!targetCount) return [];

  if (targetCount === available) {
    return allLegalActions(problem, values).filter(actionIndex => !excludedSet.has(actionIndex));
  }

  const selected = [];
  const seen = new Set(excludedSet);
  while (selected.length < targetCount) {
    const actionIndex = sampleUniformMutationAction(problem, values, rng);
    if (seen.has(actionIndex)) continue;
    seen.add(actionIndex);
    selected.push(actionIndex);
  }
  return selected;
}

function softmaxArray(values, temperature = 1) {
  if (!values.length) return [];
  const safeTemperature = Math.max(0.05, temperature);
  let maxValue = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (Number.isFinite(value) && value > maxValue) maxValue = value;
  }
  if (!Number.isFinite(maxValue)) return values.map(() => 1 / values.length);

  const weights = values.map(value => {
    const scaled = (value - maxValue) / safeTemperature;
    return Number.isFinite(scaled) ? Math.exp(clamp(scaled, -80, 0)) : 0;
  });
  const total = weights.reduce((sum, value) => sum + value, 0);
  if (!(total > 0) || !Number.isFinite(total)) return values.map(() => 1 / values.length);
  return weights.map(value => value / total);
}

function logSoftmaxArray(values, temperature = 1) {
  const probabilities = softmaxArray(values, temperature);
  return probabilities.map(probability => Math.log(Math.max(1e-12, probability)));
}

function sampleFromProbabilities(probabilities, rng) {
  let threshold = rng.next();
  for (let index = 0; index < probabilities.length; index++) {
    threshold -= probabilities[index];
    if (threshold <= 0) return index;
  }
  return probabilities.length - 1;
}

function sampleFromScores(scores, temperature, rng) {
  return sampleFromProbabilities(softmaxArray(scores, temperature), rng);
}

function argMax(values) {
  let bestIndex = 0;
  for (let index = 1; index < values.length; index++) {
    if (values[index] > values[bestIndex]) bestIndex = index;
  }
  return bestIndex;
}

function sampleStandardNormal(rng) {
  const u1 = clamp(rng.next(), 1e-9, 1 - 1e-9);
  const u2 = rng.next();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function sampleActorActionsDirectly(
  problem,
  values,
  actorData,
  rng,
  requestedCount,
  excluded = null
) {
  const excludedSet = excluded || new Set();
  const targetCount = Math.min(
    Math.max(0, requestedCount),
    Math.max(0, legalActionCount(problem) - excludedSet.size)
  );
  if (!targetCount) return [];

  const nodeProbabilities = softmaxArray(
    Array.from(actorData.nodeLogits),
    actorTrainingTemperature()
  );
  const selected = [];
  const seen = new Set(excludedSet);
  const maxAttempts = Math.max(100, targetCount * 80);

  for (let attempt = 0; attempt < maxAttempts && selected.length < targetCount; attempt++) {
    const node = sampleFromProbabilities(nodeProbabilities, rng);
    const mean = actorData.valueMeans[node];
    const scale = actorData.valueScales[node];
    const normalizedValue = clamp(mean + scale * sampleStandardNormal(rng), 0, 1);
    const value = Math.round(normalizedValue * problem.domainMax);
    if (value === values[node]) continue;
    const actionIndex = node * problem.domainSize + value;
    if (seen.has(actionIndex)) continue;
    seen.add(actionIndex);
    selected.push(actionIndex);
  }

  if (selected.length < targetCount) {
    selected.push(...sampleUniqueUniformActions(
      problem,
      values,
      rng,
      targetCount - selected.length,
      seen
    ));
  }
  return selected;
}
