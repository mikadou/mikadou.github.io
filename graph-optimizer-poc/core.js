'use strict';

const FEATURE_DIM = 9;
const HIDDEN_DIM = 32;
const MESSAGE_PASSES = 2;
const ACTION_DIM = 48;
const DEFAULT_LEARNING_RATE = 0.001;
const NEGATIVE_MASK = -1e9;

const ui = {
  testN: document.getElementById('testN'),
  trainEpisodes: document.getElementById('trainEpisodes'),
  trainMinN: document.getElementById('trainMinN'),
  trainMaxN: document.getElementById('trainMaxN'),
  trainBudgetMultiplier: document.getElementById('trainBudgetMultiplier'),
  policyBudgetMultiplier: document.getElementById('policyBudgetMultiplier'),
  saBudgetMultiplier: document.getElementById('saBudgetMultiplier'),
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

function energy(problem, values) {
  let total = 0;
  for (let depth = 0; depth < problem.n - 1; depth++) {
    const left = values[problem.indexById.get(problem.chain[depth])];
    const right = values[problem.indexById.get(problem.chain[depth + 1])];
    total += Math.max(0, right - left + 1);
  }
  return total;
}

function applyAction(problem, values, actionIndex) {
  const storageIndex = Math.floor(actionIndex / problem.domainSize);
  const value = actionIndex % problem.domainSize;
  values[storageIndex] = value;
  return { storageIndex, value, nodeId: problem.nodeIds[storageIndex] };
}

function buildNodeFeatures(problem, values, step, maxSteps) {
  const features = new Float32Array(problem.n * FEATURE_DIM);
  const denom = Math.max(1, problem.domainMax);

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
      (hasPred ? Math.max(0, -leftMargin) : 0) +
      (hasSucc ? Math.max(0, -rightMargin) : 0);
    const offset = storageIndex * FEATURE_DIM;

    features[offset] = value / denom;
    features[offset + 1] = clamp(leftMargin, -1, 1);
    features[offset + 2] = clamp(rightMargin, -1, 1);
    features[offset + 3] = hasPred ? 1 : 0;
    features[offset + 4] = hasSucc ? 1 : 0;
    features[offset + 5] = clamp(incidentViolation, 0, 2);
    features[offset + 6] = maxSteps > 0 ? step / maxSteps : 0;
    features[offset + 7] = 1 / denom;
    features[offset + 8] = 1;
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

function initVariable(shape, name, fanIn = shape[0] || 1) {
  const std = Math.sqrt(2 / Math.max(1, fanIn));
  return tf.variable(tf.randomNormal(shape, 0, std), true, name);
}

class RecurrentGraphPolicy {
  constructor() {
    this.prefix = `rgp_${Math.random().toString(36).slice(2, 9)}`;
    this.vars = [];
    this.wInput = this.add(initVariable([FEATURE_DIM, HIDDEN_DIM], `${this.prefix}_wInput`, FEATURE_DIM));
    this.bInput = this.add(tf.variable(tf.zeros([HIDDEN_DIM]), true, `${this.prefix}_bInput`));
    this.wGateInput = this.add(initVariable([HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wGateInput`, HIDDEN_DIM));
    this.wGatePred = this.add(initVariable([HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wGatePred`, HIDDEN_DIM));
    this.wGateSucc = this.add(initVariable([HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wGateSucc`, HIDDEN_DIM));
    this.wGateSelf = this.add(initVariable([HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wGateSelf`, HIDDEN_DIM));
    this.bGate = this.add(tf.variable(tf.zeros([HIDDEN_DIM]), true, `${this.prefix}_bGate`));
    this.wCandInput = this.add(initVariable([HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wCandInput`, HIDDEN_DIM));
    this.wCandPred = this.add(initVariable([HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wCandPred`, HIDDEN_DIM));
    this.wCandSucc = this.add(initVariable([HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wCandSucc`, HIDDEN_DIM));
    this.wCandSelf = this.add(initVariable([HIDDEN_DIM, HIDDEN_DIM], `${this.prefix}_wCandSelf`, HIDDEN_DIM));
    this.bCand = this.add(tf.variable(tf.zeros([HIDDEN_DIM]), true, `${this.prefix}_bCand`));
    const nodeContextDim = HIDDEN_DIM * 2 + FEATURE_DIM;
    this.wAction1 = this.add(initVariable([nodeContextDim + 3, ACTION_DIM], `${this.prefix}_wAction1`, nodeContextDim + 3));
    this.bAction1 = this.add(tf.variable(tf.zeros([ACTION_DIM]), true, `${this.prefix}_bAction1`));
    this.wAction2 = this.add(initVariable([ACTION_DIM, 1], `${this.prefix}_wAction2`, ACTION_DIM));
    this.bAction2 = this.add(tf.variable(tf.zeros([1]), true, `${this.prefix}_bAction2`));
  }

  add(variable) { this.vars.push(variable); return variable; }

  forward(problem, values, hidden, step, maxSteps) {
    const n = problem.n;
    const d = problem.domainSize;
    const features = tf.tensor2d(buildNodeFeatures(problem, values, step, maxSteps), [n, FEATURE_DIM]);
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
      nextHidden = gate.mul(candidate).add(tf.scalar(1).sub(gate).mul(nextHidden));
    }

    const nodeContext = tf.concat([nextHidden, projected, features], 1);
    const contextDim = HIDDEN_DIM * 2 + FEATURE_DIM;
    const nodeExpanded = nodeContext.expandDims(1).tile([1, d, 1]);
    const candidates = tf.linspace(0, problem.domainMax, d)
      .div(Math.max(1, problem.domainMax))
      .reshape([1, d, 1]).tile([n, 1, 1]);
    const currentValues = features.slice([0, 0], [n, 1])
      .reshape([n, 1, 1]).tile([1, d, 1]);
    const difference = candidates.sub(currentValues);
    const actionInput = tf.concat(
      [nodeExpanded, candidates, difference, difference.abs()], 2
    ).reshape([n * d, contextDim + 3]);
    const actionHidden = tf.relu(actionInput.matMul(this.wAction1).add(this.bAction1));
    let logits = actionHidden.matMul(this.wAction2).add(this.bAction2).reshape([n * d]);
    logits = logits.add(tf.tensor1d(currentValueMask(problem, values)));
    return { hidden: nextHidden, logits };
  }

  dispose() { this.vars.forEach(variable => variable.dispose()); }
}
