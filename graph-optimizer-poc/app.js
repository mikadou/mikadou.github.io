'use strict';

const FEATURE_DIM = 9;
const HIDDEN_DIM = 32;
const MESSAGE_PASSES = 2;
const ACTION_DIM = 48;
const DEFAULT_LEARNING_RATE = 0.0025;

const ui = {
  testN: document.getElementById('testN'),
  trainEpisodes: document.getElementById('trainEpisodes'),
  trainMinN: document.getElementById('trainMinN'),
  trainMaxN: document.getElementById('trainMaxN'),
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

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function readInt(input, fallback) {
  const parsed = Number.parseInt(input.value, 10);
  return Number.isFinite(parsed) ? parsed : fallback;
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

  return { n, seed, chain, nodeIds, indexById, chainIndexById, domainSize, domainMax, initialValues, predMatrix, succMatrix };
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

function targetValueAtDepth(problem, depth) { return problem.domainMax - depth; }

function teacherTrajectory(problem) {
  const values = problem.initialValues.slice();
  const actions = [];
  for (let depth = 0; depth < problem.n; depth++) {
    const storageIndex = problem.indexById.get(problem.chain[depth]);
    const targetValue = targetValueAtDepth(problem, depth);
    if (values[storageIndex] !== targetValue) {
      actions.push(storageIndex * problem.domainSize + targetValue);
      values[storageIndex] = targetValue;
    }
  }
  return actions;
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
    const predValue = hasPred ? values[problem.indexById.get(problem.chain[depth - 1])] : problem.domainMax;
    const succValue = hasSucc ? values[problem.indexById.get(problem.chain[depth + 1])] : 0;
    const leftMargin = hasPred ? (predValue - value - 1) / denom : 1;
    const rightMargin = hasSucc ? (value - succValue - 1) / denom : 1;
    const incidentViolation = (hasPred ? Math.max(0, -leftMargin) : 0) + (hasSucc ? Math.max(0, -rightMargin) : 0);
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
  for (let i = 0; i < problem.n; i++) mask[i * problem.domainSize + values[i]] = -1e9;
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
    const candidates = tf.linspace(0, problem.domainMax, d).div(Math.max(1, problem.domainMax)).reshape([1, d, 1]).tile([n, 1, 1]);
    const currentValues = features.slice([0, 0], [n, 1]).reshape([n, 1, 1]).tile([1, d, 1]);
    const difference = candidates.sub(currentValues);
    const actionInput = tf.concat([nodeExpanded, candidates, difference, difference.abs()], 2).reshape([n * d, contextDim + 3]);
    const actionHidden = tf.relu(actionInput.matMul(this.wAction1).add(this.bAction1));
    let logits = actionHidden.matMul(this.wAction2).add(this.bAction2).reshape([n * d]);
    logits = logits.add(tf.tensor1d(currentValueMask(problem, values)));
    return { hidden: nextHidden, logits };
  }

  trainEpisode(problem, optimizer) {
    const actions = teacherTrajectory(problem);
    if (actions.length === 0) return 0;
    const values = problem.initialValues.slice();
    const result = tf.tidy(() => tf.variableGrads(() => {
      let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
      let totalLoss = tf.scalar(0);
      const maxSteps = Math.max(problem.n, actions.length);
      for (let step = 0; step < actions.length; step++) {
        const output = this.forward(problem, values, hidden, step, maxSteps);
        hidden = output.hidden;
        const selectedLogProbability = tf.logSoftmax(output.logits).gather([actions[step]]).squeeze();
        totalLoss = totalLoss.sub(selectedLogProbability);
        applyAction(problem, values, actions[step]);
      }
      return totalLoss.div(actions.length);
    }, this.vars));

    const lossValue = result.value.dataSync()[0];
    const clipped = {};
    for (const [name, gradient] of Object.entries(result.grads)) clipped[name] = tf.clipByValue(gradient, -5, 5);
    optimizer.applyGradients(clipped);
    result.value.dispose();
    Object.values(result.grads).forEach(tensor => tensor.dispose());
    Object.values(clipped).forEach(tensor => tensor.dispose());
    return lossValue;
  }
  dispose() { this.vars.forEach(variable => variable.dispose()); }
}

function resetModel() {
  if (state.policy) state.policy.dispose();
  if (state.optimizer && typeof state.optimizer.dispose === 'function') state.optimizer.dispose();
  state.policy = new RecurrentGraphPolicy();
  state.optimizer = tf.train.adam(DEFAULT_LEARNING_RATE);
  state.trainedEpisodes = 0;
  ui.policyStateBadge.textContent = 'Untrained';
  ui.policyStateBadge.classList.add('muted');
  ui.compareBtn.disabled = true;
  ui.benchmarkBtn.disabled = true;
  ui.progressBar.style.width = '0%';
  ui.statusText.textContent = 'Model reset. Train the policy before comparison.';
  clearResults();
}

function createCurrentProblem() {
  const n = clamp(readInt(ui.testN, 12), 4, 64);
  ui.testN.value = String(n);
  state.currentProblem = makeProblem(n);
  renderProblem(state.currentProblem);
  clearResults();
  setStatus(`New N=${n} problem created. Hidden state will start at zero for each optimizer run.`);
}

function renderProblem(problem) {
  ui.domainLabel.textContent = `0…${problem.domainMax}`;
  ui.storageLabel.textContent = problem.nodeIds.join(', ');
  renderChain(ui.initialGraph, problem, problem.initialValues);
  renderChain(ui.policyGraph, problem, problem.initialValues);
  renderChain(ui.saGraph, problem, problem.initialValues);
}

function renderChain(container, problem, values, hiddenNorms = null, recentNodeId = null) {
  container.replaceChildren();
  const maxNorm = hiddenNorms && hiddenNorms.length ? Math.max(1e-6, ...hiddenNorms) : 1;
  for (let depth = 0; depth < problem.n; depth++) {
    const id = problem.chain[depth];
    const storageIndex = problem.indexById.get(id);
    const node = document.createElement('div');
    node.className = `chain-node${recentNodeId === id ? ' recent' : ''}`;
    const idLabel = document.createElement('span');
    idLabel.className = 'node-id'; idLabel.textContent = id;
    const valueLabel = document.createElement('span');
    valueLabel.className = 'node-value'; valueLabel.textContent = String(values[storageIndex]);
    const depthLabel = document.createElement('span');
    depthLabel.className = 'node-depth'; depthLabel.textContent = `depth ${depth}`;
    node.append(idLabel, valueLabel, depthLabel);
    if (hiddenNorms) {
      const track = document.createElement('div'); track.className = 'hidden-track';
      const bar = document.createElement('div'); bar.className = 'hidden-bar';
      bar.style.width = `${Math.min(100, (hiddenNorms[storageIndex] / maxNorm) * 100)}%`;
      track.append(bar); node.append(track);
    }
    container.append(node);
    if (depth < problem.n - 1) {
      const nextValue = values[problem.indexById.get(problem.chain[depth + 1])];
      const ok = values[storageIndex] > nextValue;
      const edge = document.createElement('div');
      edge.className = `chain-edge ${ok ? 'ok' : 'bad'}`;
      edge.innerHTML = `&gt;<small>${ok ? 'ok' : 'violation'}</small>`;
      container.append(edge);
    }
  }
}

function updateMetrics(container, data, actionLabel) {
  const items = container.querySelectorAll('div');
  const values = [data.energy, data.steps, `${data.runtimeMs.toFixed(1)} ms`, data.success ? 'Yes' : 'No'];
  items[1].querySelector('span').textContent = actionLabel;
  items.forEach((item, index) => { item.querySelector('strong').textContent = String(values[index]); });
}

function clearResults() {
  if (!state.currentProblem) return;
  renderChain(ui.policyGraph, state.currentProblem, state.currentProblem.initialValues);
  renderChain(ui.saGraph, state.currentProblem, state.currentProblem.initialValues);
  resetMetricPanel(ui.policyMetrics, 'Actions');
  resetMetricPanel(ui.saMetrics, 'Proposals');
  drawEnergyChart([], []);
  ui.benchmarkSummary.className = 'benchmark-empty';
  ui.benchmarkSummary.textContent = 'Run a 30-instance benchmark after training.';
}

function resetMetricPanel(container, actionLabel) {
  const labels = ['Energy', actionLabel, 'Runtime', 'Success'];
  container.querySelectorAll('div').forEach((item, index) => {
    item.querySelector('span').textContent = labels[index];
    item.querySelector('strong').textContent = '–';
  });
}

function hiddenNormsFromTensor(hiddenTensor, n) {
  const data = hiddenTensor.dataSync();
  const norms = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < HIDDEN_DIM; j++) {
      const value = data[i * HIDDEN_DIM + j];
      sum += value * value;
    }
    norms[i] = Math.sqrt(sum);
  }
  return norms;
}

async function runLearnedPolicy(problem, maxSteps) {
  const values = problem.initialValues.slice();
  let bestValues = values.slice();
  let bestEnergy = energy(problem, values);
  const history = [bestEnergy];
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let lastNodeId = null;
  let steps = 0;
  const started = performance.now();

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    if (state.stopRequested) break;
    const output = tf.tidy(() => state.policy.forward(problem, values, hidden, step, maxSteps));
    const logitsData = output.logits.dataSync();
    let actionIndex = 0;
    let bestLogit = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < logitsData.length; i++) {
      if (logitsData[i] > bestLogit) { bestLogit = logitsData[i]; actionIndex = i; }
    }
    hidden.dispose();
    hidden = output.hidden;
    output.logits.dispose();
    const action = applyAction(problem, values, actionIndex);
    lastNodeId = action.nodeId;
    steps++;
    const currentEnergy = energy(problem, values);
    if (currentEnergy < bestEnergy) { bestEnergy = currentEnergy; bestValues = values.slice(); }
    history.push(bestEnergy);
  }

  const hiddenNorms = hiddenNormsFromTensor(hidden, problem.n);
  hidden.dispose();
  return { values: bestValues, energy: bestEnergy, steps, runtimeMs: performance.now() - started, success: bestEnergy === 0, history, hiddenNorms, lastNodeId };
}

function runSimulatedAnnealing(problem, maxSteps, seed = problem.seed ^ 0xA5A5A5A5) {
  const rng = new SeededRandom(seed);
  const values = problem.initialValues.slice();
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let bestValues = values.slice();
  const history = [bestEnergy];
  const initialTemperature = Math.max(1, problem.n / 2);
  let steps = 0;
  let lastNodeId = null;
  const started = performance.now();

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    if (state.stopRequested) break;
    const storageIndex = rng.int(problem.n);
    let candidateValue = rng.int(problem.domainSize);
    if (candidateValue === values[storageIndex]) candidateValue = (candidateValue + 1 + rng.int(problem.domainSize - 1)) % problem.domainSize;
    const oldValue = values[storageIndex];
    values[storageIndex] = candidateValue;
    const candidateEnergy = energy(problem, values);
    const delta = candidateEnergy - currentEnergy;
    const progress = maxSteps > 1 ? step / (maxSteps - 1) : 1;
    const temperature = initialTemperature * Math.pow(0.001, progress);
    const accept = delta <= 0 || rng.next() < Math.exp(-delta / Math.max(1e-6, temperature));
    if (accept) {
      currentEnergy = candidateEnergy;
      lastNodeId = problem.nodeIds[storageIndex];
      if (currentEnergy < bestEnergy) { bestEnergy = currentEnergy; bestValues = values.slice(); }
    } else values[storageIndex] = oldValue;
    steps++;
    history.push(bestEnergy);
  }

  return { values: bestValues, energy: bestEnergy, steps, runtimeMs: performance.now() - started, success: bestEnergy === 0, history, lastNodeId };
}

async function trainPolicy() {
  if (state.busy) return;
  const episodes = clamp(readInt(ui.trainEpisodes, 800), 50, 10000);
  let minN = clamp(readInt(ui.trainMinN, 4), 3, 32);
  let maxN = clamp(readInt(ui.trainMaxN, 12), 4, 40);
  if (minN > maxN) [minN, maxN] = [maxN, minN];
  ui.trainMinN.value = String(minN);
  ui.trainMaxN.value = String(maxN);
  setBusy(true);
  state.stopRequested = false;
  let movingLoss = null;

  try {
    for (let episode = 1; episode <= episodes; episode++) {
      if (state.stopRequested) break;
      const n = minN + Math.floor(Math.random() * (maxN - minN + 1));
      const loss = state.policy.trainEpisode(makeProblem(n), state.optimizer);
      movingLoss = movingLoss === null ? loss : movingLoss * 0.95 + loss * 0.05;
      state.trainedEpisodes++;
      if (episode === 1 || episode % 10 === 0 || episode === episodes) {
        ui.progressBar.style.width = `${(episode / episodes) * 100}%`;
        setStatus(`Training ${episode}/${episodes} · moving loss ${movingLoss.toFixed(3)} · total episodes ${state.trainedEpisodes}`);
        await tf.nextFrame();
      }
    }
    if (state.trainedEpisodes > 0) {
      ui.policyStateBadge.textContent = `${state.trainedEpisodes} episodes`;
      ui.policyStateBadge.classList.remove('muted');
      ui.compareBtn.disabled = false;
      ui.benchmarkBtn.disabled = false;
    }
    setStatus(state.stopRequested ? `Training stopped after ${state.trainedEpisodes} total episodes.` : `Training complete: ${state.trainedEpisodes} total episodes. Run a paired comparison.`);
  } catch (error) {
    console.error(error);
    setStatus(`Training failed: ${error.message}`, true);
  } finally {
    setBusy(false);
    state.stopRequested = false;
  }
}

async function runComparison() {
  if (state.busy || !state.currentProblem || state.trainedEpisodes === 0) return;
  setBusy(true);
  state.stopRequested = false;
  try {
    const n = state.currentProblem.n;
    const policyBudget = clamp(readInt(ui.policyBudgetMultiplier, 3), 1, 12) * n;
    const saBudget = clamp(readInt(ui.saBudgetMultiplier, 250), 10, 2000) * n;
    setStatus(`Running policy (${policyBudget} actions) and SA (${saBudget} proposals) from the same assignment…`);
    const policyResult = await runLearnedPolicy(state.currentProblem, policyBudget);
    await tf.nextFrame();
    const saResult = runSimulatedAnnealing(state.currentProblem, saBudget);
    renderChain(ui.policyGraph, state.currentProblem, policyResult.values, policyResult.hiddenNorms, policyResult.lastNodeId);
    renderChain(ui.saGraph, state.currentProblem, saResult.values, null, saResult.lastNodeId);
    updateMetrics(ui.policyMetrics, policyResult, 'Actions');
    updateMetrics(ui.saMetrics, saResult, 'Proposals');
    drawEnergyChart(policyResult.history, saResult.history);
    setStatus(`Comparison complete. Policy: ${policyResult.success ? 'feasible' : 'not feasible'} in ${policyResult.steps} actions; SA: ${saResult.success ? 'feasible' : 'not feasible'} in ${saResult.steps} proposals.`);
  } catch (error) {
    console.error(error);
    setStatus(`Comparison failed: ${error.message}`, true);
  } finally {
    setBusy(false);
    state.stopRequested = false;
  }
}

function median(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

async function runBenchmark() {
  if (state.busy || state.trainedEpisodes === 0) return;
  setBusy(true);
  state.stopRequested = false;
  const count = 30;
  const n = clamp(readInt(ui.testN, 12), 4, 64);
  const policyBudget = clamp(readInt(ui.policyBudgetMultiplier, 3), 1, 12) * n;
  const saBudget = clamp(readInt(ui.saBudgetMultiplier, 250), 10, 2000) * n;
  const policyResults = [];
  const saResults = [];

  try {
    for (let i = 0; i < count; i++) {
      if (state.stopRequested) break;
      const problem = makeProblem(n, 500000 + i * 7919 + state.trainedEpisodes);
      policyResults.push(await runLearnedPolicy(problem, policyBudget));
      saResults.push(runSimulatedAnnealing(problem, saBudget, problem.seed ^ 0x55AA55AA));
      ui.progressBar.style.width = `${((i + 1) / count) * 100}%`;
      setStatus(`Benchmarking ${i + 1}/${count} instances at N=${n}…`);
      if ((i + 1) % 3 === 0) await tf.nextFrame();
    }
    renderBenchmark(n, policyResults, saResults, policyBudget, saBudget);
    setStatus(`Benchmark complete on ${policyResults.length} instances at N=${n}.`);
  } catch (error) {
    console.error(error);
    setStatus(`Benchmark failed: ${error.message}`, true);
  } finally {
    setBusy(false);
    state.stopRequested = false;
  }
}

function summarizeResults(results) {
  const successes = results.filter(result => result.success);
  return {
    successRate: results.length ? successes.length / results.length : 0,
    medianSteps: median(successes.map(result => result.steps)),
    medianRuntime: median(results.map(result => result.runtimeMs)),
    medianFinalEnergy: median(results.map(result => result.energy))
  };
}

function renderBenchmark(n, policyResults, saResults, policyBudget, saBudget) {
  const policy = summarizeResults(policyResults);
  const sa = summarizeResults(saResults);
  ui.benchmarkSummary.className = '';
  ui.benchmarkSummary.innerHTML = `
    <div class="benchmark-grid">
      <div class="benchmark-card"><span>Policy success</span><strong>${(policy.successRate * 100).toFixed(0)}%</strong></div>
      <div class="benchmark-card"><span>SA success</span><strong>${(sa.successRate * 100).toFixed(0)}%</strong></div>
      <div class="benchmark-card"><span>Policy median actions</span><strong>${policy.medianSteps ?? '–'}</strong></div>
      <div class="benchmark-card"><span>SA median proposals</span><strong>${sa.medianSteps ?? '–'}</strong></div>
    </div>
    <table class="benchmark-table">
      <thead><tr><th>Method</th><th>Budget</th><th>Success</th><th>Median steps</th><th>Median runtime</th><th>Median final energy</th></tr></thead>
      <tbody>
        <tr><td>Learned recurrent policy</td><td>${policyBudget}</td><td>${(policy.successRate * 100).toFixed(1)}%</td><td>${policy.medianSteps ?? '–'}</td><td>${policy.medianRuntime?.toFixed(2) ?? '–'} ms</td><td>${policy.medianFinalEnergy ?? '–'}</td></tr>
        <tr><td>Simulated annealing</td><td>${saBudget}</td><td>${(sa.successRate * 100).toFixed(1)}%</td><td>${sa.medianSteps ?? '–'}</td><td>${sa.medianRuntime?.toFixed(2) ?? '–'} ms</td><td>${sa.medianFinalEnergy ?? '–'}</td></tr>
      </tbody>
    </table>
    <p class="caption">${policyResults.length} paired instances, N=${n}. Median step counts use successful runs only.</p>`;
}

function drawEnergyChart(policyHistory, saHistory) {
  const canvas = ui.energyChart;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const styles = getComputedStyle(document.documentElement);
  const textColor = styles.getPropertyValue('--muted').trim() || '#677085';
  const gridColor = styles.getPropertyValue('--border').trim() || '#dfe4ef';
  const policyColor = styles.getPropertyValue('--accent').trim() || '#4f46e5';
  const saColor = styles.getPropertyValue('--warning').trim() || '#9a6700';
  const pad = { left: 56, right: 18, top: 22, bottom: 42 };
  ctx.clearRect(0, 0, width, height);
  ctx.font = '14px system-ui, sans-serif';
  ctx.strokeStyle = gridColor;
  ctx.fillStyle = textColor;
  ctx.lineWidth = 1;
  const maxX = Math.max(1, policyHistory.length - 1, saHistory.length - 1);
  const maxY = Math.max(1, ...policyHistory, ...saHistory);
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  for (let tick = 0; tick <= 4; tick++) {
    const y = pad.top + (plotHeight * tick) / 4;
    const value = Math.round(maxY * (1 - tick / 4));
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(width - pad.right, y); ctx.stroke();
    ctx.fillText(String(value), 10, y + 5);
  }
  ctx.fillText('0', pad.left - 4, height - 12);
  ctx.fillText(String(maxX), width - pad.right - 28, height - 12);
  ctx.fillText('Iteration', width / 2 - 24, height - 10);
  function plot(history, color) {
    if (history.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath();
    history.forEach((value, index) => {
      const x = pad.left + (index / maxX) * plotWidth;
      const y = pad.top + (1 - value / maxY) * plotHeight;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  plot(saHistory, saColor);
  plot(policyHistory, policyColor);
  if (!policyHistory.length && !saHistory.length) {
    ctx.fillStyle = textColor; ctx.textAlign = 'center';
    ctx.fillText('Run a comparison to plot search progress.', width / 2, height / 2);
    ctx.textAlign = 'start';
  }
}

function setStatus(message, isError = false) {
  ui.statusText.textContent = message;
  ui.statusText.style.color = isError ? 'var(--danger)' : '';
}

function setBusy(busy) {
  state.busy = busy;
  ui.newProblemBtn.disabled = busy;
  ui.trainBtn.disabled = busy;
  ui.compareBtn.disabled = busy || state.trainedEpisodes === 0;
  ui.benchmarkBtn.disabled = busy || state.trainedEpisodes === 0;
  ui.resetModelBtn.disabled = busy;
  ui.stopBtn.disabled = !busy;
}

async function initialize() {
  try {
    await tf.ready();
    ui.backendNotice.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}`;
    resetModel();
    createCurrentProblem();
  } catch (error) {
    console.error(error);
    ui.backendNotice.textContent = `TensorFlow.js failed to initialize: ${error.message}`;
    ui.backendNotice.classList.add('error');
    ui.trainBtn.disabled = true;
  }
}

ui.newProblemBtn.addEventListener('click', createCurrentProblem);
ui.trainBtn.addEventListener('click', trainPolicy);
ui.compareBtn.addEventListener('click', runComparison);
ui.benchmarkBtn.addEventListener('click', runBenchmark);
ui.stopBtn.addEventListener('click', () => { state.stopRequested = true; setStatus('Stopping after the current operation…'); });
ui.resetModelBtn.addEventListener('click', resetModel);
ui.testN.addEventListener('change', createCurrentProblem);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => drawEnergyChart([], []));

initialize();
