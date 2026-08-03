'use strict';

const FEATURE_DIM = 19;
const ACTION_FEATURE_DIM = 10;
const HIDDEN_DIM = 32;
const MESSAGE_PASSES = 2;
const ACTION_DIM = 48;
const POLICY_MEMORY_WINDOW = 32;
const DEFAULT_LEARNING_RATE = 0.00075;
const NEGATIVE_MASK = -1e9;

const ui = {
  testN: document.getElementById('testN'),
  trainEpisodes: document.getElementById('trainEpisodes'),
  trainMinN: document.getElementById('trainMinN'),
  trainMaxN: document.getElementById('trainMaxN'),
  trainBudgetMultiplier: document.getElementById('trainBudgetMultiplier'),
  proposalBudgetMultiplier: document.getElementById('proposalBudgetMultiplier'),
  newProblemBtn: document.getElementById('newProblemBtn'),
  trainBtn: document.getElementById('trainBtn'),
  compareBtn: document.getElementById('compareBtn'),
  benchmarkBtn: document.getElementById('benchmarkBtn'),
  stopBtn: document.getElementById('stopBtn'),
  resetModelBtn: document.getElementById('resetModelBtn'),
  progressBar: document.getElementById('progressBar'),
  statusText: document.getElementById('statusText'),
  backendNotice: document.getElementById('backendNotice'),
  domainLabel: document.getElementById('domainLabel'),
  storageLabel: document.getElementById('storageLabel'),
  initialGraph: document.getElementById('initialGraph'),
  policyGraph: document.getElementById('policyGraph'),
  saGraph: document.getElementById('saGraph'),
  policyMetrics: document.getElementById('policyMetrics'),
  saMetrics: document.getElementById('saMetrics'),
  policyStateBadge: document.getElementById('policyStateBadge'),
  energyChart: document.getElementById('energyChart'),
  benchmarkSummary: document.getElementById('benchmarkSummary')
};

const state = {
  policy: null,
  optimizer: null,
  currentProblem: null,
  trainedEpisodes: 0,
  busy: false,
  stopRequested: false,
  seedCounter: 1
};

class SeededRandom {
  constructor(seed) { this.seed = seed >>> 0; }
  next() {
    let t = this.seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  int(maxExclusive) { return Math.floor(this.next() * maxExclusive); }
}

function shuffle(items, rng) {
  const result = items.slice();
  for (let i = result.length - 1; i > 0; i--) {
    const j = rng.int(i + 1);
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function readInt(input, fallback) {
  const parsed = Number.parseInt(input.value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function readPositiveInt(input, fallback) {
  const value = Math.max(1, readInt(input, fallback));
  input.value = String(value);
  return value;
}

function makeProblem(n, seed = Date.now() + state.seedCounter++) {
  const rng = new SeededRandom(seed);
  const ids = Array.from({ length: n }, (_, i) => `x${i + 1}`);
  const chain = shuffle(ids, rng);
  const nodeIds = shuffle(ids, rng);
  const indexById = new Map(nodeIds.map((id, index) => [id, index]));
  const chainIndexById = new Map(chain.map((id, index) => [id, index]));
  const domainSize = 2 * n;
  const domainMax = domainSize - 1;
  const initialValues = Array.from({ length: n }, () => rng.int(domainSize));
  const predMatrix = new Float32Array(n * n);
  const succMatrix = new Float32Array(n * n);

  for (let depth = 0; depth < n - 1; depth++) {
    const greaterIndex = indexById.get(chain[depth]);
    const lesserIndex = indexById.get(chain[depth + 1]);
    predMatrix[lesserIndex * n + greaterIndex] = 1;
    succMatrix[greaterIndex * n + lesserIndex] = 1;
  }

  return {
    n, seed, chain, nodeIds, indexById, chainIndexById,
    domainSize, domainMax, initialValues, predMatrix, succMatrix
  };
}

function edgeViolation(leftValue, rightValue) {
  return Math.max(0, rightValue - leftValue + 1);
}

function energy(problem, values) {
  let total = 0;
  for (let depth = 0; depth < problem.n - 1; depth++) {
    const left = values[problem.indexById.get(problem.chain[depth])];
    const right = values[problem.indexById.get(problem.chain[depth + 1])];
    total += edgeViolation(left, right);
  }
  return total;
}

function violationCount(problem, values) {
  let count = 0;
  for (let depth = 0; depth < problem.n - 1; depth++) {
    const left = values[problem.indexById.get(problem.chain[depth])];
    const right = values[problem.indexById.get(problem.chain[depth + 1])];
    if (left <= right) count++;
  }
  return count;
}

function decodeAction(problem, actionIndex) {
  const storageIndex = Math.floor(actionIndex / problem.domainSize);
  const value = actionIndex % problem.domainSize;
  return { storageIndex, value, nodeId: problem.nodeIds[storageIndex] };
}

function applyAction(problem, values, actionIndex) {
  const action = decodeAction(problem, actionIndex);
  values[action.storageIndex] = action.value;
  return action;
}

function annealingTemperature(problem, step, maxSteps) {
  const initialTemperature = Math.max(1, problem.n / 2);
  const progress = maxSteps > 1 ? step / (maxSteps - 1) : 1;
  return initialTemperature * Math.pow(0.001, progress);
}

function annealingAcceptanceProbability(delta, temperature) {
  if (delta <= 0) return 1;
  return Math.exp(-delta / Math.max(1e-6, temperature));
}

function resolveSearchContext(problem, values, step, maxSteps, saContext = {}) {
  const currentEnergy = Number.isFinite(saContext.currentEnergy)
    ? saContext.currentEnergy
    : energy(problem, values);
  const bestEnergy = Number.isFinite(saContext.bestEnergy)
    ? saContext.bestEnergy
    : currentEnergy;
  const violations = Number.isFinite(saContext.violationCount)
    ? saContext.violationCount
    : violationCount(problem, values);
  const temperature = Number.isFinite(saContext.temperature)
    ? saContext.temperature
    : annealingTemperature(problem, step, maxSteps);

  return {
    currentEnergy,
    bestEnergy,
    violationCount: violations,
    temperature,
    lastAccepted: saContext.lastAccepted === false ? 0 : 1,
    lastDelta: Number.isFinite(saContext.lastDelta) ? saContext.lastDelta : 0,
    recentAcceptance: clamp(
      Number.isFinite(saContext.recentAcceptance) ? saContext.recentAcceptance : 1,
      0,
      1
    ),
    stagnation: Math.max(0, Number.isFinite(saContext.stagnation) ? saContext.stagnation : 0)
  };
}

function localMutationDiagnostics(problem, values, storageIndex, candidateValue, searchContext) {
  const id = problem.nodeIds[storageIndex];
  const depth = problem.chainIndexById.get(id);
  const currentValue = values[storageIndex];
  const hasPred = depth > 0;
  const hasSucc = depth < problem.n - 1;
  const predValue = hasPred
    ? values[problem.indexById.get(problem.chain[depth - 1])]
    : problem.domainMax;
  const succValue = hasSucc
    ? values[problem.indexById.get(problem.chain[depth + 1])]
    : 0;

  const oldLeftViolation = hasPred ? edgeViolation(predValue, currentValue) : 0;
  const oldRightViolation = hasSucc ? edgeViolation(currentValue, succValue) : 0;
  const newLeftViolation = hasPred ? edgeViolation(predValue, candidateValue) : 0;
  const newRightViolation = hasSucc ? edgeViolation(candidateValue, succValue) : 0;
  const deltaEnergy =
    newLeftViolation + newRightViolation - oldLeftViolation - oldRightViolation;
  const candidateEnergy = Math.max(0, searchContext.currentEnergy + deltaEnergy);
  const acceptanceProbability = annealingAcceptanceProbability(
    deltaEnergy,
    searchContext.temperature
  );

  const lowerBound = hasSucc ? succValue + 1 : 0;
  const upperBound = hasPred ? predValue - 1 : problem.domainMax;
  let feasibleDistance = 0;
  if (lowerBound <= upperBound) {
    if (candidateValue < lowerBound) feasibleDistance = lowerBound - candidateValue;
    else if (candidateValue > upperBound) feasibleDistance = candidateValue - upperBound;
  } else {
    feasibleDistance = Math.min(
      Math.abs(candidateValue - lowerBound),
      Math.abs(candidateValue - upperBound)
    ) + (lowerBound - upperBound);
  }

  return {
    currentValue,
    newLeftViolation,
    newRightViolation,
    deltaEnergy,
    candidateEnergy,
    acceptanceProbability,
    feasibleDistance,
    bestImprovement: Math.max(0, searchContext.bestEnergy - candidateEnergy)
  };
}

function buildNodeFeatures(problem, values, step, maxSteps, searchContext) {
  const features = new Float32Array(problem.n * FEATURE_DIM);
  const denom = Math.max(1, problem.domainMax);
  const energyDenom = Math.max(1, problem.n);
  const initialTemperature = Math.max(1, problem.n / 2);
  const temperatureRatio = clamp(searchContext.temperature / initialTemperature, 0, 1);
  const stepProgress = maxSteps > 0 ? step / maxSteps : 0;
  const remainingProgress = 1 - stepProgress;
  const normalizedCurrentEnergy = clamp(searchContext.currentEnergy / energyDenom, 0, 4);
  const normalizedBestEnergy = clamp(searchContext.bestEnergy / energyDenom, 0, 4);
  const normalizedGap = clamp(
    (searchContext.currentEnergy - searchContext.bestEnergy) / energyDenom,
    0,
    4
  );
  const violationFraction = searchContext.violationCount / Math.max(1, problem.n - 1);
  const stagnationRatio = clamp(searchContext.stagnation / Math.max(1, maxSteps), 0, 1);

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
    const leftMargin = hasPred ? (predValue - value - 1) / denom : 1;
    const rightMargin = hasSucc ? (value - succValue - 1) / denom : 1;
    const incidentViolation =
      (hasPred ? edgeViolation(predValue, value) : 0) +
      (hasSucc ? edgeViolation(value, succValue) : 0);
    const offset = storageIndex * FEATURE_DIM;

    features[offset] = value / denom;
    features[offset + 1] = clamp(leftMargin, -1, 1);
    features[offset + 2] = clamp(rightMargin, -1, 1);
    features[offset + 3] = hasPred ? 1 : 0;
    features[offset + 4] = hasSucc ? 1 : 0;
    features[offset + 5] = clamp(incidentViolation / denom, 0, 2);
    features[offset + 6] = stepProgress;
    features[offset + 7] = remainingProgress;
    features[offset + 8] = 1 / denom;
    features[offset + 9] = 1;
    features[offset + 10] = temperatureRatio;
    features[offset + 11] = searchContext.lastAccepted;
    features[offset + 12] = clamp(searchContext.lastDelta / energyDenom, -2, 2);
    features[offset + 13] = normalizedCurrentEnergy;
    features[offset + 14] = normalizedBestEnergy;
    features[offset + 15] = normalizedGap;
    features[offset + 16] = violationFraction;
    features[offset + 17] = searchContext.recentAcceptance;
    features[offset + 18] = stagnationRatio;
  }
  return features;
}

function buildActionFeatures(problem, values, searchContext) {
  const features = new Float32Array(
    problem.n * problem.domainSize * ACTION_FEATURE_DIM
  );
  const valueDenom = Math.max(1, problem.domainMax);
  const energyDenom = Math.max(1, problem.n);

  for (let node = 0; node < problem.n; node++) {
    const currentValue = values[node];
    for (let candidate = 0; candidate < problem.domainSize; candidate++) {
      const diagnostics = localMutationDiagnostics(
        problem,
        values,
        node,
        candidate,
        searchContext
      );
      const offset = (node * problem.domainSize + candidate) * ACTION_FEATURE_DIM;
      const difference = candidate - currentValue;

      features[offset] = candidate / valueDenom;
      features[offset + 1] = difference / valueDenom;
      features[offset + 2] = Math.abs(difference) / valueDenom;
      features[offset + 3] = clamp(diagnostics.newLeftViolation / valueDenom, 0, 2);
      features[offset + 4] = clamp(diagnostics.newRightViolation / valueDenom, 0, 2);
      features[offset + 5] = clamp(diagnostics.deltaEnergy / energyDenom, -2, 2);
      features[offset + 6] = clamp(diagnostics.candidateEnergy / energyDenom, 0, 4);
      features[offset + 7] = diagnostics.acceptanceProbability;
      features[offset + 8] = clamp(diagnostics.feasibleDistance / valueDenom, 0, 2);
      features[offset + 9] = clamp(diagnostics.bestImprovement / energyDenom, 0, 2);
    }
  }

  return features;
}

function currentValueMask(problem, values) {
  const mask = new Float32Array(problem.n * problem.domainSize);
  for (let node = 0; node < problem.n; node++) {
    mask[node * problem.domainSize + values[node]] = NEGATIVE_MASK;
  }
  return mask;
}

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

function sampleWeightedMutationAction(problem, values, logits, temperature, rng) {
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
      const weight = Math.exp(clamp(scaled, -80, 0));
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
      const weight = Math.exp(clamp(scaled, -80, 0));
      if (!Number.isFinite(weight)) continue;
      cumulative += weight;
      if (cumulative > threshold) return offset + value;
    }
  }
  return fallback;
}

function initVariable(shape, name, fanIn = shape[0] || 1) {
  const std = Math.sqrt(2 / Math.max(1, fanIn));
  return tf.variable(tf.randomNormal(shape, 0, std), true, name);
}

class RecurrentGraphPolicy {
  constructor() {
    this.prefix = `rgp_${Math.random().toString(36).slice(2, 9)}`;
    this.vars = [];
    this.wInput = this.add(initVariable(
      [FEATURE_DIM, HIDDEN_DIM], `${this.prefix}_wInput`, FEATURE_DIM
    ));
    this.bInput = this.add(tf.variable(
      tf.zeros([HIDDEN_DIM]), true, `${this.prefix}_bInput`
    ));
    this.wGateInput = this.add(initVariable(
      [HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wGateInput`, HIDDEN_DIM
    ));
    this.wGatePred = this.add(initVariable(
      [HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wGatePred`, HIDDEN_DIM
    ));
    this.wGateSucc = this.add(initVariable(
      [HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wGateSucc`, HIDDEN_DIM
    ));
    this.wGateSelf = this.add(initVariable(
      [HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wGateSelf`, HIDDEN_DIM
    ));
    this.bGate = this.add(tf.variable(
      tf.zeros([HIDDEN_DIM]), true, `${this.prefix}_bGate`
    ));
    this.wCandInput = this.add(initVariable(
      [HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wCandInput`, HIDDEN_DIM
    ));
    this.wCandPred = this.add(initVariable(
      [HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wCandPred`, HIDDEN_DIM
    ));
    this.wCandSucc = this.add(initVariable(
      [HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wCandSucc`, HIDDEN_DIM
    ));
    this.wCandSelf = this.add(initVariable(
      [HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wCandSelf`, HIDDEN_DIM
    ));
    this.bCand = this.add(tf.variable(
      tf.zeros([HIDDEN_DIM]), true, `${this.prefix}_bCand`
    ));

    const nodeContextDim = HIDDEN_DIM * 2 + FEATURE_DIM;
    this.wAction1 = this.add(initVariable(
      [nodeContextDim + ACTION_FEATURE_DIM, ACTION_DIM],
      `${this.prefix}_wAction1`,
      nodeContextDim + ACTION_FEATURE_DIM
    ));
    this.bAction1 = this.add(tf.variable(
      tf.zeros([ACTION_DIM]), true, `${this.prefix}_bAction1`
    ));
    this.wAction2 = this.add(tf.variable(
      tf.zeros([ACTION_DIM, 1]), true, `${this.prefix}_wAction2`
    ));
    this.bAction2 = this.add(tf.variable(
      tf.zeros([1]), true, `${this.prefix}_bAction2`
    ));
  }

  add(variable) {
    this.vars.push(variable);
    return variable;
  }

  forward(problem, values, hidden, step, maxSteps, saContext = {}) {
    const n = problem.n;
    const d = problem.domainSize;
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
    const actionFeatureData = buildActionFeatures(problem, values, searchContext);
    const features = tf.tensor2d(nodeFeatureData, [n, FEATURE_DIM]);
    const actionFeatures = tf.tensor2d(
      actionFeatureData,
      [n * d, ACTION_FEATURE_DIM]
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
    const contextDim = HIDDEN_DIM * 2 + FEATURE_DIM;
    const nodeExpanded = nodeContext.expandDims(1)
      .tile([1, d, 1])
      .reshape([n * d, contextDim]);
    const actionInput = tf.concat([nodeExpanded, actionFeatures], 1);
    const actionHidden = tf.relu(actionInput.matMul(this.wAction1).add(this.bAction1));
    let logits = actionHidden.matMul(this.wAction2).add(this.bAction2)
      .reshape([n * d]);
    logits = logits.add(tf.tensor1d(currentValueMask(problem, values)));
    return { hidden: nextHidden, logits };
  }

  dispose() {
    this.vars.forEach(variable => variable.dispose());
  }
}
