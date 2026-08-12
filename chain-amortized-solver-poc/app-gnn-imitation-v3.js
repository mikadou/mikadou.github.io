const DOMAIN_MAX = 1000;
const MAX_TEST_N = 100;
const NODE_FEATURE_DIM = 9;
const EMBED_DIM = 16;
const EDGE_TYPES = 4;
const MOVE_MULTIPLIER = 3;
const TEACHER_SIGMA = 1.0;
const VALUE_SCALE = 2 * MAX_TEST_N;
const BATCH_STATES = 16;
const ENDPOINT_EPOCHS = 4;
const ENDPOINT_THRESHOLD = 0.5;
const ENDPOINT_MARGIN = 2.0;
const VALUE_PRETRAIN_STEPS = 180;

let MESSAGE_ROUNDS = 1;
let params = null;
let endpointOptimizer = null;
let valueOptimizer = null;
let trained = false;
let trainedRange = { min: 2, max: 20 };
let graphCache = new Map();
let paramGeneration = 0;
let diagnosticLines = [];

const $ = id => document.getElementById(id);
const trainMinN = $('trainMinN');
const trainMaxN = $('trainMaxN');
const episodes = $('episodes');
const messageRounds = $('messageRounds');
const trainBtn = $('trainBtn');
const benchmarkBtn = $('benchmarkBtn');
const statusEl = $('status');

function installDiagnosticsUI() {
  if ($('diagnosticLog')) return;
  const section = document.createElement('section');
  section.className = 'card explanation';
  section.innerHTML = `
    <h2>Copyable diagnostic log</h2>
    <p>After training and benchmark finish, tap <strong>Copy diagnostics</strong> and paste the block into ChatGPT.</p>
    <div class="buttons" style="margin:12px 0">
      <button id="copyLogBtn" type="button">Copy diagnostics</button>
      <button id="clearLogBtn" type="button" class="secondary">Clear log</button>
    </div>
    <textarea id="diagnosticLog" readonly spellcheck="false" style="width:100%;min-height:340px;resize:vertical;border:1px solid #dfe4ee;border-radius:10px;padding:12px;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#fbfcff;color:#20283a"></textarea>`;
  const footer = document.querySelector('footer');
  footer.parentNode.insertBefore(section, footer);
  $('copyLogBtn').addEventListener('click', async () => {
    const text = diagnosticLines.join('\n');
    try {
      await navigator.clipboard.writeText(text);
      $('copyLogBtn').textContent = 'Copied';
      setTimeout(() => { $('copyLogBtn').textContent = 'Copy diagnostics'; }, 1000);
    } catch {
      const el = $('diagnosticLog'); el.focus(); el.select(); document.execCommand('copy');
    }
  });
  $('clearLogBtn').addEventListener('click', () => { diagnosticLines = []; syncLog(); });
}
function syncLog() { const el = $('diagnosticLog'); if (el) { el.value = diagnosticLines.join('\n'); el.scrollTop = el.scrollHeight; } }
function logLine(line) { diagnosticLines.push(String(line)); syncLog(); }
function resetLog() { diagnosticLines = []; syncLog(); }
installDiagnosticsUI();

function bindRange(input, labelId) {
  const update = () => $(labelId).textContent = input.value;
  input.addEventListener('input', update); update();
}
bindRange(trainMinN, 'trainMinNLabel');
bindRange(trainMaxN, 'trainMaxNLabel');
bindRange(episodes, 'episodesLabel');
trainMinN.addEventListener('input', () => {
  if (+trainMinN.value > +trainMaxN.value) { trainMaxN.value = trainMinN.value; $('trainMaxNLabel').textContent = trainMaxN.value; }
});
trainMaxN.addEventListener('input', () => {
  if (+trainMaxN.value < +trainMinN.value) { trainMinN.value = trainMaxN.value; $('trainMinNLabel').textContent = trainMinN.value; }
});
messageRounds.addEventListener('change', () => {
  if (trained) { trained = false; benchmarkBtn.disabled = true; statusEl.textContent = 'Message-passing depth changed. Retrain before benchmarking.'; }
});

function mulberry32(seed) {
  return function () {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussian(mean, sigma, rng) {
  const u1 = Math.max(1e-12, rng()), u2 = rng();
  return mean + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
function sigmoidScalar(x) { return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); }

function emptyState(n) {
  return { n, assigned: new Uint8Array(n), values: Int16Array.from({ length: n }, () => -1), assignedCount: 0 };
}
function fullStateFromValues(values) {
  const n = values.length;
  return { n, assigned: Uint8Array.from({ length: n }, () => 1), values: Int16Array.from(values), assignedCount: n };
}
function applyAction(state, i, v) {
  if (!state.assigned[i]) { state.assigned[i] = 1; state.assignedCount++; }
  state.values[i] = v;
}
function isStrictChain(x) { for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) return false; return true; }
function violationEnergy(x) { let e = 0; for (let i = 0; i + 1 < x.length; i++) e += Math.max(0, x[i] - x[i + 1] + 1); return e; }
function violatedConstraintCount(x) { let c = 0; for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) c++; return c; }
function violatedEndpointIds(x) {
  const marked = new Uint8Array(x.length);
  for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) { marked[i] = 1; marked[i + 1] = 1; }
  const ids = []; for (let i = 0; i < x.length; i++) if (marked[i]) ids.push(i); return ids;
}
function teacherMeanValue(i) { return Math.max(0, Math.min(DOMAIN_MAX, 2 * i)); }
function teacherSampleValue(i, rng) { return Math.max(0, Math.min(DOMAIN_MAX, Math.round(gaussian(teacherMeanValue(i), TEACHER_SIGMA, rng)))); }

function makeRepairTrainingState(n, rng) {
  const values = new Int16Array(n);
  let current = Math.floor(rng() * 12);
  for (let i = 0; i < n; i++) {
    if (i > 0) current += 1 + Math.floor(rng() * 4);
    values[i] = Math.min(900, current);
  }
  const maxInject = Math.max(1, Math.min(4, Math.ceil(n / 7)));
  const injectCount = 1 + Math.floor(rng() * maxInject);
  const chosen = new Set();
  while (chosen.size < Math.min(injectCount, Math.max(1, n - 1))) chosen.add(Math.floor(rng() * Math.max(1, n - 1)));
  for (const k of chosen) {
    if (k + 1 >= n) continue;
    if (rng() < 0.5) values[k + 1] = Math.max(0, values[k] - Math.floor(3 * rng()));
    else values[k] = Math.min(DOMAIN_MAX, values[k + 1] + Math.floor(3 * rng()));
  }
  if (isStrictChain(values) && n > 1) {
    const k = Math.floor(rng() * (n - 1)); values[k + 1] = values[k];
  }
  const state = fullStateFromValues(values);
  return { state, candidateIds: Int32Array.from(violatedEndpointIds(values)) };
}

function makeVariable(shape, scale, name, mean = 0) {
  const init = tf.randomNormal(shape, mean, scale), v = tf.variable(init, true, `${name}_${paramGeneration}`); init.dispose(); return v;
}
function makeBias(size, name) { const init = tf.zeros([size]), v = tf.variable(init, true, `${name}_${paramGeneration}`); init.dispose(); return v; }
function endpointParams() {
  return [params.Winit, params.binit, params.Wself, params.brel, ...params.Wrel, params.Wendpoint, params.bendpoint];
}
function valueParams() { return [params.valueSlope, params.valueBias, params.logSigma]; }
function allParams() { return [...endpointParams(), ...valueParams()]; }
function disposeModel() {
  if (params) for (const p of allParams()) p.dispose();
  if (endpointOptimizer?.dispose) endpointOptimizer.dispose();
  if (valueOptimizer?.dispose) valueOptimizer.dispose();
  params = null; endpointOptimizer = null; valueOptimizer = null;
}
function initModel() {
  disposeModel(); paramGeneration++;
  const s = 0.08;
  params = {
    Winit: makeVariable([NODE_FEATURE_DIM, EMBED_DIM], s, 'Winit'), binit: makeBias(EMBED_DIM, 'binit'),
    Wself: makeVariable([EMBED_DIM, EMBED_DIM], s, 'Wself'), brel: makeBias(EMBED_DIM, 'brel'),
    Wrel: Array.from({ length: EDGE_TYPES }, (_, r) => makeVariable([EMBED_DIM, EMBED_DIM], s, `Wrel${r}`)),
    Wendpoint: makeVariable([EMBED_DIM, 1], s, 'Wendpoint'), bendpoint: makeBias(1, 'bendpoint'),
    valueSlope: makeVariable([1], 0.18, 'valueSlope', 0.1),
    valueBias: makeVariable([1], 0.05, 'valueBias', 0),
    logSigma: makeVariable([1], 0.15, 'logSigma', 0)
  };
  endpointOptimizer = tf.train.adam(0.003);
  valueOptimizer = tf.train.adam(0.03);
}

function graphSpec(n) {
  if (graphCache.has(n)) return graphCache.get(n);
  const constraintCount = n - 1, nodeCount = n + constraintCount;
  const baseFeatures = new Float32Array(nodeCount * NODE_FEATURE_DIM);
  const absScale = Math.max(1, MAX_TEST_N - 1), sizeFeature = n / MAX_TEST_N;
  for (let i = 0; i < n; i++) {
    const p = i * NODE_FEATURE_DIM;
    baseFeatures[p] = 1;
    baseFeatures[p + 2] = i / absScale;
    baseFeatures[p + 3] = n <= 1 ? 0 : i / (n - 1);
    baseFeatures[p + 4] = sizeFeature;
  }
  for (let k = 0; k < constraintCount; k++) {
    const p = (n + k) * NODE_FEATURE_DIM;
    baseFeatures[p + 1] = 1;
    baseFeatures[p + 2] = k / absScale;
    baseFeatures[p + 3] = constraintCount <= 1 ? 0 : k / (constraintCount - 1);
    baseFeatures[p + 4] = sizeFeature;
  }
  const adjs = Array.from({ length: EDGE_TYPES }, () => new Float32Array(nodeCount * nodeCount));
  const setEdge = (type, target, source) => { adjs[type][target * nodeCount + source] = 1; };
  for (let k = 0; k < constraintCount; k++) {
    const c = n + k;
    setEdge(0, c, k); setEdge(1, c, k + 1); setEdge(2, k, c); setEdge(3, k + 1, c);
  }
  const spec = { n, constraintCount, nodeCount, baseFeatures, adjs }; graphCache.set(n, spec); return spec;
}
function nodeFeatureData(state) {
  const spec = graphSpec(state.n), data = Float32Array.from(spec.baseFeatures);
  for (let i = 0; i < state.n; i++) {
    const p = i * NODE_FEATURE_DIM;
    data[p + 5] = state.assigned[i] ? 1 : 0;
    data[p + 6] = state.assigned[i] ? state.values[i] / DOMAIN_MAX : 0;
  }
  for (let k = 0; k + 1 < state.n; k++) {
    const p = (state.n + k) * NODE_FEATURE_DIM;
    if (state.assigned[k] && state.assigned[k + 1]) {
      const violated = state.values[k] >= state.values[k + 1];
      data[p + 7] = violated ? 1 : 0;
      data[p + 8] = violated ? 0 : 1;
    }
  }
  return data;
}
function graphEmbeddings(state) {
  const spec = graphSpec(state.n);
  const nodeX = tf.tensor2d(nodeFeatureData(state), [spec.nodeCount, NODE_FEATURE_DIM]);
  const adj = spec.adjs.map(a => tf.tensor2d(a, [spec.nodeCount, spec.nodeCount]));
  let h = tf.relu(tf.matMul(nodeX, params.Winit).add(params.binit));
  for (let round = 0; round < MESSAGE_ROUNDS; round++) {
    let z = tf.matMul(h, params.Wself);
    for (let r = 0; r < EDGE_TYPES; r++) z = z.add(tf.matMul(adj[r], tf.matMul(h, params.Wrel[r])));
    h = tf.relu(h.add(z.add(params.brel)));
  }
  return h;
}
function endpointTensors(state) {
  const h = graphEmbeddings(state), variableH = h.slice([0, 0], [state.n, EMBED_DIM]);
  const endpointLogits = tf.matMul(variableH, params.Wendpoint).add(params.bendpoint).reshape([state.n]);
  return { h, variableH, endpointLogits };
}
function endpointSnapshot(state) {
  return tf.tidy(() => {
    const p = endpointTensors(state);
    return { endpointLogits: Float32Array.from(p.endpointLogits.dataSync()) };
  });
}
function valueSnapshot() {
  return { slope: params.valueSlope.dataSync()[0], bias: params.valueBias.dataSync()[0], sigma: Math.exp(params.logSigma.dataSync()[0]) };
}
function learnedMeanForIndex(i) {
  const v = valueSnapshot();
  const absPos = i / Math.max(1, MAX_TEST_N - 1);
  return Math.max(0, Math.min(DOMAIN_MAX, VALUE_SCALE * (v.slope * absPos + v.bias)));
}
function learnedValueForIndex(i, rng, stochastic = true, oracleValue = false) {
  if (oracleValue) {
    const meanValue = teacherMeanValue(i), sigmaValue = TEACHER_SIGMA;
    const raw = stochastic ? gaussian(meanValue, sigmaValue, rng) : meanValue;
    return { v: Math.max(0, Math.min(DOMAIN_MAX, Math.round(raw))), meanValue, sigmaValue };
  }
  const snap = valueSnapshot();
  const absPos = i / Math.max(1, MAX_TEST_N - 1);
  const meanValue = Math.max(0, Math.min(DOMAIN_MAX, VALUE_SCALE * (snap.slope * absPos + snap.bias)));
  const sigmaValue = snap.sigma;
  const raw = stochastic ? gaussian(meanValue, sigmaValue, rng) : meanValue;
  return { v: Math.max(0, Math.min(DOMAIN_MAX, Math.round(raw))), meanValue, sigmaValue };
}

function endpointLoss(sample) {
  const p = endpointTensors(sample.state), n = sample.state.n;
  const labelsArray = new Float32Array(n); for (const i of sample.candidateIds) labelsArray[i] = 1;
  const labels = tf.tensor1d(labelsArray), negatives = tf.onesLike(labels).sub(labels);
  const posCount = sample.candidateIds.length, negCount = n - posCount;
  const posBce = tf.softplus(p.endpointLogits.neg()).mul(labels).sum().div(Math.max(1, posCount));
  const negBce = tf.softplus(p.endpointLogits).mul(negatives).sum().div(Math.max(1, negCount));
  const bce = posCount && negCount ? posBce.add(negBce).mul(0.5) : (posCount ? posBce : negBce);
  const posMargin = tf.relu(tf.scalar(ENDPOINT_MARGIN).sub(p.endpointLogits)).mul(labels).sum().div(Math.max(1, posCount));
  const negMargin = tf.relu(tf.scalar(ENDPOINT_MARGIN).add(p.endpointLogits)).mul(negatives).sum().div(Math.max(1, negCount));
  const margin = posCount && negCount ? posMargin.add(negMargin).mul(0.5) : (posCount ? posMargin : negMargin);
  return bce.add(margin.mul(0.25));
}
async function endpointUpdate(samples) {
  let last = NaN;
  for (let epoch = 0; epoch < ENDPOINT_EPOCHS; epoch++) {
    const cost = endpointOptimizer.minimize(() => tf.tidy(() => {
      let total = tf.scalar(0);
      for (const s of samples) total = total.add(endpointLoss(s));
      return total.div(samples.length);
    }), true, endpointParams());
    last = cost.dataSync()[0]; cost.dispose();
  }
  await tf.nextFrame();
  return last;
}
async function pretrainValueHead(maxN) {
  for (let step = 0; step < VALUE_PRETRAIN_STEPS; step++) {
    const cost = valueOptimizer.minimize(() => tf.tidy(() => {
      const ids = tf.range(0, maxN, 1, 'float32');
      const absPos = ids.div(Math.max(1, MAX_TEST_N - 1));
      const targetNorm = ids.mul(2 / VALUE_SCALE);
      const predNorm = absPos.mul(params.valueSlope).add(params.valueBias);
      const meanLoss = predNorm.sub(targetNorm).square().mean();
      const sigmaLoss = params.logSigma.square().mean().mul(0.1);
      return meanLoss.add(sigmaLoss);
    }), true, valueParams());
    cost.dispose();
    if ((step + 1) % 30 === 0) await tf.nextFrame();
  }
}
function valueMae(n) {
  let err = 0;
  for (let i = 0; i < n; i++) err += Math.abs(learnedMeanForIndex(i) - teacherMeanValue(i));
  return err / Math.max(1, n);
}

function endpointDiagnostics(n, rng, states = 48) {
  let tp = 0, fp = 0, fn = 0, posScore = 0, negScore = 0, posN = 0, negN = 0;
  for (let s = 0; s < states; s++) {
    const sample = makeRepairTrainingState(n, rng), positive = new Set(sample.candidateIds), snap = endpointSnapshot(sample.state);
    for (let i = 0; i < n; i++) {
      const score = sigmoidScalar(snap.endpointLogits[i]), isPos = positive.has(i), pred = score >= ENDPOINT_THRESHOLD;
      if (isPos) { posScore += score; posN++; } else { negScore += score; negN++; }
      if (pred && isPos) tp++; else if (pred) fp++; else if (isPos) fn++;
    }
  }
  const precision = tp / Math.max(1, tp + fp), recall = tp / Math.max(1, tp + fn), f1 = 2 * precision * recall / Math.max(1e-9, precision + recall);
  return { precision, recall, f1, posScore: posScore / Math.max(1, posN), negScore: negScore / Math.max(1, negN) };
}
function chooseRepairVariable(state, snap, rng, stochastic) {
  const predicted = [], scores = [];
  for (let i = 0; i < state.n; i++) {
    const score = sigmoidScalar(snap.endpointLogits[i]); scores.push(score); if (score >= ENDPOINT_THRESHOLD) predicted.push(i);
  }
  if (predicted.length) {
    if (stochastic) return predicted[Math.floor(rng() * predicted.length)];
    return predicted.reduce((best, i) => scores[i] > scores[best] ? i : best, predicted[0]);
  }
  return scores.indexOf(Math.max(...scores));
}
function runImitation(n, rng, stochastic = true, oracleEndpoints = false, oracleValue = false) {
  const state = emptyState(n), maxMoves = MOVE_MULTIPLIER * n; let moves = 0;
  while (state.assignedCount < n && moves < maxMoves) {
    const remaining = []; for (let i = 0; i < n; i++) if (!state.assigned[i]) remaining.push(i);
    const i = stochastic ? remaining[Math.floor(rng() * remaining.length)] : remaining[0];
    const val = learnedValueForIndex(i, rng, stochastic, oracleValue); applyAction(state, i, val.v); moves++;
  }
  while (moves < maxMoves && !isStrictChain(state.values)) {
    let i;
    if (oracleEndpoints) {
      const ids = violatedEndpointIds(state.values); if (!ids.length) break;
      i = stochastic ? ids[Math.floor(rng() * ids.length)] : ids[0];
    } else {
      const snap = endpointSnapshot(state); i = chooseRepairVariable(state, snap, rng, stochastic);
    }
    const val = learnedValueForIndex(i, rng, stochastic, oracleValue); applyAction(state, i, val.v); moves++;
  }
  return { x: Int16Array.from(state.values), solved: isStrictChain(state.values), moves };
}
function runHandcrafted(n, rng) {
  const state = emptyState(n), maxMoves = MOVE_MULTIPLIER * n; let moves = 0;
  for (let i = 0; i < n; i++) { applyAction(state, i, teacherSampleValue(i, rng)); moves++; }
  while (moves < maxMoves && !isStrictChain(state.values)) {
    const ids = violatedEndpointIds(state.values); if (!ids.length) break;
    const i = ids[Math.floor(rng() * ids.length)]; applyAction(state, i, teacherSampleValue(i, rng)); moves++;
  }
  return { x: Int16Array.from(state.values), solved: isStrictChain(state.values), moves };
}
function randomAssignment(n, rng) { return Int16Array.from({ length: n }, () => Math.floor(rng() * (DOMAIN_MAX + 1))); }
function runSA(n, evaluations, rng) {
  const x = randomAssignment(n, rng); let e = violationEnergy(x), best = Int16Array.from(x), bestE = e;
  const T0 = 80, Tend = 0.02;
  for (let step = 0; step < evaluations && bestE > 0; step++) {
    const i = Math.floor(rng() * n), old = x[i]; x[i] = Math.floor(rng() * (DOMAIN_MAX + 1));
    const next = violationEnergy(x), delta = next - e, t = step / Math.max(1, evaluations - 1), temp = T0 * Math.pow(Tend / T0, t);
    if (delta <= 0 || rng() < Math.exp(-delta / Math.max(1e-6, temp))) { e = next; if (e < bestE) { bestE = e; best = Int16Array.from(x); } } else x[i] = old;
  }
  return { x: best, solved: bestE === 0 };
}
function solveRate(fn, n, seed, trials = 10) {
  const rng = mulberry32(seed); let solved = 0;
  for (let t = 0; t < trials; t++) if (fn(n, rng, true).solved) solved++;
  return solved / trials;
}

async function trainImitation() {
  if (!window.tf) throw new Error('TensorFlow.js did not load.');
  MESSAGE_ROUNDS = +messageRounds.value;
  const startN = +trainMinN.value, maxN = +trainMaxN.value, totalStates = +episodes.value;
  trainedRange = { min: startN, max: maxN }; trained = false; trainBtn.disabled = true; benchmarkBtn.disabled = true;
  initModel(); resetLog();
  logLine('GNN_ENDPOINT_DIAGNOSTIC v3');
  logLine(`CONFIG backend=${tf.getBackend()} train=${startN}..${maxN} states=${totalStates} rounds=${MESSAGE_ROUNDS} embed=${EMBED_DIM} nodeFeatures=${NODE_FEATURE_DIM} batch=${BATCH_STATES} endpointEpochs=${ENDPOINT_EPOCHS} margin=${ENDPOINT_MARGIN} threshold=${ENDPOINT_THRESHOLD} budget=${MOVE_MULTIPLIER}n`);

  statusEl.textContent = 'Pretraining static positional value head…';
  await pretrainValueHead(maxN);
  const value = valueSnapshot();
  logLine(`VALUE slope=${value.slope.toFixed(4)} bias=${value.bias.toFixed(4)} sigma=${value.sigma.toFixed(3)} trainMAE=${valueMae(maxN).toFixed(3)} extrapMAE100=${valueMae(MAX_TEST_N).toFixed(3)}`);

  const rng = mulberry32(20260816); let completed = 0, updates = 0, lastLoss = NaN;
  while (completed < totalStates) {
    const batch = [], count = Math.min(BATCH_STATES, totalStates - completed);
    for (let b = 0; b < count; b++) {
      const n = startN + Math.floor(rng() * (maxN - startN + 1)); batch.push(makeRepairTrainingState(n, rng)); completed++;
    }
    lastLoss = await endpointUpdate(batch); updates++;
    if (updates % 8 === 0 || completed >= totalStates) {
      const diag = endpointDiagnostics(maxN, mulberry32(800000 + updates), 40);
      const solve = solveRate((n, r) => runImitation(n, r, true, false, false), maxN, 810000 + updates, 6);
      statusEl.textContent = `Repair states ${completed}/${totalStates} · update ${updates} · endpoint P/R/F1 ${(100*diag.precision).toFixed(0)}/${(100*diag.recall).toFixed(0)}/${(100*diag.f1).toFixed(0)}% · pos/neg ${(100*diag.posScore).toFixed(0)}/${(100*diag.negScore).toFixed(0)}% · solve@${maxN} ${(100*solve).toFixed(0)}%`;
      logLine(`TRAIN states=${completed} update=${updates} loss=${lastLoss.toFixed(4)} P=${(100*diag.precision).toFixed(1)}% R=${(100*diag.recall).toFixed(1)}% F1=${(100*diag.f1).toFixed(1)}% posScore=${(100*diag.posScore).toFixed(1)}% negScore=${(100*diag.negScore).toFixed(1)}% solve@n${maxN}=${(100*solve).toFixed(0)}%`);
      await tf.nextFrame();
    }
  }
  trained = true; trainBtn.disabled = false; benchmarkBtn.disabled = false;
  statusEl.textContent = 'Two-branch training complete. Running benchmark…';
  await runBenchmark();
}

function benchmarkLengths() {
  const m = trainedRange.max;
  return [...new Set([trainedRange.min, m, Math.min(MAX_TEST_N, m + 1), Math.min(MAX_TEST_N, Math.round(m * 1.5)), Math.min(MAX_TEST_N, m * 2), MAX_TEST_N])].sort((a,b)=>a-b);
}
function median(values) { if (!values.length) return NaN; const a=[...values].sort((x,y)=>x-y); return a[Math.floor(a.length/2)]; }
async function runBenchmark() {
  if (!trained || !params) return;
  trainBtn.disabled = true; benchmarkBtn.disabled = true;
  const rows = [], rng = mulberry32(20260817), trials = 12;
  for (const n of benchmarkLengths()) {
    statusEl.textContent = `Benchmarking two-branch GNN, teacher, and SA at n=${n}…`;
    let pSolved=0,hSolved=0,saSolved=0,oracleEndpointSolved=0,oracleValueSolved=0,pViol=0,hViol=0;
    const pMoves=[],hMoves=[];
    for (let t=0;t<trials;t++) {
      const p=runImitation(n,rng,true,false,false); if(p.solved){pSolved++;pMoves.push(p.moves);} pViol += violatedConstraintCount(p.x);
      if(runImitation(n,rng,true,true,false).solved) oracleEndpointSolved++;
      if(runImitation(n,rng,true,false,true).solved) oracleValueSolved++;
      const h=runHandcrafted(n,rng); if(h.solved){hSolved++;hMoves.push(h.moves);} hViol += violatedConstraintCount(h.x);
      if(runSA(n,120*n,rng).solved) saSolved++;
      await tf.nextFrame();
    }
    rows.push({n,policySuccess:pSolved/trials,heuristicSuccess:hSolved/trials,saSuccess:saSolved/trials,policyMoves:median(pMoves),heuristicMoves:median(hMoves)});
    logLine(`BENCH n=${n} gnn=${Math.round(100*pSolved/trials)}% oracleEndpoints=${Math.round(100*oracleEndpointSolved/trials)}% oracleValue=${Math.round(100*oracleValueSolved/trials)}% teacher=${Math.round(100*hSolved/trials)}% sa=${Math.round(100*saSolved/trials)}% gnnAvgViol=${(pViol/trials).toFixed(2)} teacherAvgViol=${(hViol/trials).toFixed(2)}`);
  }
  drawScaling(rows);
  const longest=rows[rows.length-1], n=longest.n;
  const p=runImitation(n,mulberry32(9101),true,false,false), h=runHandcrafted(n,mulberry32(9102)), sa=runSA(n,120*n,mulberry32(9103));
  drawAssignment(p.x,h.x,sa.x);
  const diag=endpointDiagnostics(trainedRange.max,mulberry32(9200),64);
  $('policySuccess').textContent=`${Math.round(100*longest.policySuccess)}%`;
  $('heuristicSuccess').textContent=`${Math.round(100*longest.heuristicSuccess)}%`;
  $('saSuccess').textContent=`${Math.round(100*longest.saSuccess)}%`;
  $('candidateMass').textContent=`${Math.round(100*diag.f1)}%`;
  $('meanMae').textContent=valueMae(trainedRange.max).toFixed(2);
  $('policyMoves').textContent=Number.isFinite(longest.policyMoves)?Math.round(longest.policyMoves):'—';
  $('heuristicMoves').textContent=Number.isFinite(longest.heuristicMoves)?Math.round(longest.heuristicMoves):'—';
  $('saEffort').textContent=(120*n).toLocaleString();
  $('metricLength').textContent=`n=${n} · separated static value head + repair GNN · GNN/teacher ${MOVE_MULTIPLIER}n · SA 120n · train ${trainedRange.min}…${trainedRange.max} · depth ${MESSAGE_ROUNDS}`;
  appendProbe(trainedRange.max); appendProbe(Math.min(MAX_TEST_N, Math.max(trainedRange.max + 1, trainedRange.max * 2))); appendTrace(trainedRange.max); logLine('END');
  statusEl.textContent='Done. Copy diagnostics below if either branch still misses its teacher subproblem.';
  trainBtn.disabled=false; benchmarkBtn.disabled=false;
}

function appendProbe(n) {
  const sample=makeRepairTrainingState(n,mulberry32(9300+n)), positives=new Set(sample.candidateIds), snap=endpointSnapshot(sample.state);
  const scored=Array.from({length:n},(_,i)=>[i,sigmoidScalar(snap.endpointLogits[i])]).sort((a,b)=>b[1]-a[1]);
  const pred=scored.filter(([,p])=>p>=ENDPOINT_THRESHOLD).map(([i])=>i);
  logLine(`PROBE_REPAIR n=${n} violations=${violatedConstraintCount(sample.state.values)} candidates=[${[...positives].join(',')}] predicted=[${pred.join(',')}] top=[${scored.slice(0,12).map(([i,p])=>`${i}:${(100*p).toFixed(1)}%${positives.has(i)?'*':''}`).join(',')}]`);
  const ids=[...new Set([0,Math.floor((n-1)/4),Math.floor((n-1)/2),Math.floor(3*(n-1)/4),n-1])];
  const v=valueSnapshot();
  logLine(`PROBE_VALUES n=${n} slope=${v.slope.toFixed(4)} bias=${v.bias.toFixed(4)} sigma=${v.sigma.toFixed(3)} ${ids.map(i=>`${i}:pred=${learnedMeanForIndex(i).toFixed(2)}/target=${teacherMeanValue(i)}`).join(' | ')}`);
}
function appendTrace(n) {
  const rng=mulberry32(9400+n), state=emptyState(n), maxMoves=MOVE_MULTIPLIER*n; let moves=0;
  while(state.assignedCount<n&&moves<maxMoves){const remaining=[];for(let i=0;i<n;i++)if(!state.assigned[i])remaining.push(i);const i=remaining[Math.floor(rng()*remaining.length)],val=learnedValueForIndex(i,rng,true,false);applyAction(state,i,val.v);moves++;}
  logLine(`TRACE_START n=${n} violations=${violatedConstraintCount(state.values)} energy=${violationEnergy(state.values)}`);
  for(let step=0;step<16&&moves<maxMoves&&!isStrictChain(state.values);step++){
    const valid=new Set(violatedEndpointIds(state.values)),before=violatedConstraintCount(state.values),snap=endpointSnapshot(state),i=chooseRepairVariable(state,snap,rng,true),val=learnedValueForIndex(i,rng,true,false);applyAction(state,i,val.v);moves++;
    logLine(`TRACE_REPAIR step=${step} choose=${i} valid=${valid.has(i)?1:0} score=${(100*sigmoidScalar(snap.endpointLogits[i])).toFixed(1)}% v=${val.v} mean=${val.meanValue.toFixed(2)} viol=${before}->${violatedConstraintCount(state.values)} energy=${violationEnergy(state.values)}`);
  }
  logLine(`TRACE_END solved=${isStrictChain(state.values)?1:0} moves=${moves} violations=${violatedConstraintCount(state.values)} energy=${violationEnergy(state.values)}`);
}

function setupCanvas(canvas){const dpr=window.devicePixelRatio||1,rect=canvas.getBoundingClientRect();canvas.width=Math.max(300,Math.floor(rect.width*dpr));canvas.height=Math.floor(Math.max(260,rect.width*.46)*dpr);const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return{ctx,w:canvas.width/dpr,h:canvas.height/dpr};}
function axes(ctx,w,h,pad,xl,yl){ctx.clearRect(0,0,w,h);ctx.strokeStyle='#dbe1eb';ctx.beginPath();ctx.moveTo(pad,14);ctx.lineTo(pad,h-pad);ctx.lineTo(w-12,h-pad);ctx.stroke();ctx.fillStyle='#697386';ctx.font='12px system-ui';ctx.fillText(yl,pad+6,24);ctx.textAlign='right';ctx.fillText(xl,w-12,h-10);ctx.textAlign='left';}
function line(ctx,pts,color,width=2.4){if(!pts.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();ctx.fillStyle=color;for(const p of pts){ctx.beginPath();ctx.arc(p[0],p[1],2.6,0,Math.PI*2);ctx.fill();}}
function drawScaling(rows){const{ctx,w,h}=setupCanvas($('scalingChart')),pad=44;axes(ctx,w,h,pad,'chain length n','solve rate');const lo=Math.min(...rows.map(r=>r.n)),hi=Math.max(...rows.map(r=>r.n)),X=n=>pad+(n-lo)/Math.max(1,hi-lo)*(w-pad-20),Y=p=>h-pad-p*(h-pad-28);ctx.fillStyle='#7b8495';ctx.font='11px system-ui';for(const r of rows)ctx.fillText(String(r.n),X(r.n)-6,h-pad+17);for(let k=0;k<=4;k++)ctx.fillText(`${25*k}%`,5,Y(k/4)+4);const bx=X(trainedRange.max);ctx.save();ctx.setLineDash([5,5]);ctx.strokeStyle='#a9b1c2';ctx.beginPath();ctx.moveTo(bx,18);ctx.lineTo(bx,h-pad);ctx.stroke();ctx.restore();ctx.fillText('train max',Math.min(w-80,bx+5),28);line(ctx,rows.map(r=>[X(r.n),Y(r.policySuccess)]),'#5b67d6');line(ctx,rows.map(r=>[X(r.n),Y(r.heuristicSuccess)]),'#2e8b72');line(ctx,rows.map(r=>[X(r.n),Y(r.saSuccess)]),'#dd6b55');}
function drawAssignment(policy,teacher,sa){const{ctx,w,h}=setupCanvas($('instanceChart')),pad=44;axes(ctx,w,h,pad,'variable node id i','assigned value');const n=policy.length,X=i=>pad+i/Math.max(1,n-1)*(w-pad-20),Y=v=>h-pad-Math.max(0,v)/DOMAIN_MAX*(h-pad-28);line(ctx,Array.from(policy,(v,i)=>[X(i),Y(v)]),'#5b67d6',2.2);line(ctx,Array.from(teacher,(v,i)=>[X(i),Y(v)]),'#2e8b72',2);line(ctx,Array.from(sa,(v,i)=>[X(i),Y(v)]),'#dd6b55',1.8);}

trainBtn.addEventListener('click',()=>trainImitation().catch(err=>{console.error(err);statusEl.textContent=`Error: ${err.message}`;logLine(`ERROR ${err.stack||err.message}`);trainBtn.disabled=false;benchmarkBtn.disabled=!trained;}));
benchmarkBtn.addEventListener('click',()=>runBenchmark().catch(err=>{console.error(err);statusEl.textContent=`Error: ${err.message}`;logLine(`ERROR ${err.stack||err.message}`);trainBtn.disabled=false;benchmarkBtn.disabled=false;}));
statusEl.textContent=`TensorFlow.js ready · backend: ${tf.getBackend()}. Two-branch imitation: static learned value rule + message-passing repair classifier.`;