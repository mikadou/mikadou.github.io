const DOMAIN_MIN = 0;
const DOMAIN_MAX = 1000;
const MAX_TEST_N = 100;
const NODE_FEATURE_DIM = 9;
const EMBED_DIM = 24;
const EDGE_TYPES = 4;
const MOVE_MULTIPLIER = 3;
const TEACHER_SIGMA = 1.0;
const BATCH_STATES = 16;
const UPDATE_EPOCHS = 4;
const VALUE_NEGATIVES = 48;
const VARIABLE_TEMPERATURE = 0.30;
const VALUE_TEMPERATURE = 0.80;

let MESSAGE_ROUNDS = 1;
let params = null;
let optimizer = null;
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

// Teacher exists only to generate imitation labels. The learned architecture does not encode this formula.
function teacherMeanValue(i) { return Math.max(DOMAIN_MIN, Math.min(DOMAIN_MAX, 2 * i)); }
function teacherSampleValue(i, rng) { return Math.max(DOMAIN_MIN, Math.min(DOMAIN_MAX, Math.round(gaussian(teacherMeanValue(i), TEACHER_SIGMA, rng)))); }

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
    if (rng() < 0.5) values[k + 1] = Math.max(DOMAIN_MIN, values[k] - Math.floor(3 * rng()));
    else values[k] = Math.min(DOMAIN_MAX, values[k + 1] + Math.floor(3 * rng()));
  }
  if (isStrictChain(values) && n > 1) {
    const k = Math.floor(rng() * (n - 1)); values[k + 1] = values[k];
  }
  const state = fullStateFromValues(values);
  return { state, candidateIds: Int32Array.from(violatedEndpointIds(values)) };
}

function buildValueCandidateSet(positiveValue, rng) {
  const values = [positiveValue];
  const seen = new Set(values);
  while (values.length < VALUE_NEGATIVES + 1) {
    let v;
    if (rng() < 0.6) {
      const radius = 1 + Math.floor(rng() * 32);
      const sign = rng() < 0.5 ? -1 : 1;
      v = positiveValue + sign * radius;
    } else v = DOMAIN_MIN + Math.floor(rng() * (DOMAIN_MAX - DOMAIN_MIN + 1));
    v = Math.max(DOMAIN_MIN, Math.min(DOMAIN_MAX, v));
    if (!seen.has(v)) { seen.add(v); values.push(v); }
  }
  for (let i = values.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [values[i], values[j]] = [values[j], values[i]];
  }
  return { values: Int32Array.from(values), positiveIndex: values.indexOf(positiveValue) };
}

function makeTrainingSample(n, rng) {
  const base = makeRepairTrainingState(n, rng);
  const ids = Array.from(base.candidateIds);
  const teacherVar = ids[Math.floor(rng() * ids.length)];
  const teacherValue = teacherSampleValue(teacherVar, rng);
  const candidates = buildValueCandidateSet(teacherValue, rng);
  return { ...base, teacherVar, teacherValue, valueCandidates: candidates.values, positiveIndex: candidates.positiveIndex };
}

function makeVariable(shape, scale, name, mean = 0) {
  const init = tf.randomNormal(shape, mean, scale), v = tf.variable(init, true, `${name}_${paramGeneration}`); init.dispose(); return v;
}
function makeBias(size, name) { const init = tf.zeros([size]), v = tf.variable(init, true, `${name}_${paramGeneration}`); init.dispose(); return v; }
function trainableParams() {
  if (!params) return [];
  return [params.Winit, params.binit, params.Wself, params.brel, ...params.Wrel,
    params.Wvar, params.bvar, params.Wvalue1, params.bvalue1, params.Wvalue2, params.bvalue2];
}
function disposeModel() {
  if (params) for (const p of trainableParams()) p.dispose();
  if (optimizer?.dispose) optimizer.dispose();
  params = null; optimizer = null;
}
function initModel() {
  disposeModel(); paramGeneration++;
  const s = 0.08;
  params = {
    Winit: makeVariable([NODE_FEATURE_DIM, EMBED_DIM], s, 'Winit'), binit: makeBias(EMBED_DIM, 'binit'),
    Wself: makeVariable([EMBED_DIM, EMBED_DIM], s, 'Wself'), brel: makeBias(EMBED_DIM, 'brel'),
    Wrel: Array.from({ length: EDGE_TYPES }, (_, r) => makeVariable([EMBED_DIM, EMBED_DIM], s, `Wrel${r}`)),
    Wvar: makeVariable([EMBED_DIM, 1], s, 'Wvar'), bvar: makeBias(1, 'bvar'),
    Wvalue1: makeVariable([3 * EMBED_DIM + 3, 48], s, 'Wvalue1'), bvalue1: makeBias(48, 'bvalue1'),
    Wvalue2: makeVariable([48, 1], s * 0.6, 'Wvalue2'), bvalue2: makeBias(1, 'bvalue2')
  };
  optimizer = tf.train.adam(0.002);
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
    data[p + 6] = state.assigned[i] ? (state.values[i] - DOMAIN_MIN) / (DOMAIN_MAX - DOMAIN_MIN) : 0;
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
  const h0 = tf.relu(tf.matMul(nodeX, params.Winit).add(params.binit));
  let h = h0;
  for (let round = 0; round < MESSAGE_ROUNDS; round++) {
    let z = tf.matMul(h, params.Wself);
    for (let r = 0; r < EDGE_TYPES; r++) z = z.add(tf.matMul(adj[r], tf.matMul(h, params.Wrel[r])));
    h = tf.relu(h.add(z.add(params.brel)));
  }
  return { h0, h };
}
function variableLogitsFromRep(rep, state) {
  const variableH = rep.h.slice([0, 0], [state.n, EMBED_DIM]);
  return tf.matMul(variableH, params.Wvar).add(params.bvar).reshape([state.n]);
}
function candidateFeatureData(state, variableId, candidateValues) {
  const out = new Float32Array(candidateValues.length * 3);
  const assigned = !!state.assigned[variableId];
  const current = assigned ? state.values[variableId] : DOMAIN_MIN;
  const span = Math.max(1, DOMAIN_MAX - DOMAIN_MIN);
  for (let k = 0; k < candidateValues.length; k++) {
    const v = candidateValues[k];
    out[3 * k] = (v - DOMAIN_MIN) / span;
    out[3 * k + 1] = assigned ? (v - current) / span : 0;
    out[3 * k + 2] = assigned && v === current ? 1 : 0;
  }
  return out;
}
function valueLogitsFromRep(rep, state, variableId, candidateValues) {
  const k = candidateValues.length;
  const local0 = rep.h0.slice([variableId, 0], [1, EMBED_DIM]).tile([k, 1]);
  const local = rep.h.slice([variableId, 0], [1, EMBED_DIM]).tile([k, 1]);
  const global = rep.h.mean(0).reshape([1, EMBED_DIM]).tile([k, 1]);
  const candidateX = tf.tensor2d(candidateFeatureData(state, variableId, candidateValues), [k, 3]);
  const x = tf.concat([local0, local, global, candidateX], 1);
  const hidden = tf.relu(tf.matMul(x, params.Wvalue1).add(params.bvalue1));
  return tf.matMul(hidden, params.Wvalue2).add(params.bvalue2).reshape([k]);
}
function variableSnapshot(state) {
  return tf.tidy(() => {
    const rep = graphEmbeddings(state);
    return Float32Array.from(variableLogitsFromRep(rep, state).dataSync());
  });
}
function valueSnapshot(state, variableId, candidateValues) {
  return tf.tidy(() => {
    const rep = graphEmbeddings(state);
    return Float32Array.from(valueLogitsFromRep(rep, state, variableId, candidateValues).dataSync());
  });
}
function endpointLossFromLogits(logits, candidateIds, n) {
  const labelsArray = new Float32Array(n); for (const i of candidateIds) labelsArray[i] = 1;
  const labels = tf.tensor1d(labelsArray), negatives = tf.onesLike(labels).sub(labels);
  const posCount = candidateIds.length, negCount = n - posCount;
  const posLoss = tf.softplus(logits.neg()).mul(labels).sum().div(Math.max(1, posCount));
  const negLoss = tf.softplus(logits).mul(negatives).sum().div(Math.max(1, negCount));
  return posCount && negCount ? posLoss.add(negLoss).mul(0.5) : (posCount ? posLoss : negLoss);
}
function trainingSampleLoss(sample) {
  const rep = graphEmbeddings(sample.state);
  const varLogits = variableLogitsFromRep(rep, sample.state);
  const variableLoss = endpointLossFromLogits(varLogits, sample.candidateIds, sample.state.n);
  const valueLogits = valueLogitsFromRep(rep, sample.state, sample.teacherVar, sample.valueCandidates);
  const logPolicy = tf.logSoftmax(valueLogits);
  const valueLoss = logPolicy.gather(tf.tensor1d([sample.positiveIndex], 'int32')).mean().neg();
  return variableLoss.add(valueLoss.mul(0.35));
}
async function trainingUpdate(samples) {
  let last = NaN;
  for (let epoch = 0; epoch < UPDATE_EPOCHS; epoch++) {
    const cost = optimizer.minimize(() => tf.tidy(() => {
      let total = tf.scalar(0);
      for (const s of samples) total = total.add(trainingSampleLoss(s));
      return total.div(samples.length);
    }), true, trainableParams());
    last = cost.dataSync()[0]; cost.dispose();
  }
  await tf.nextFrame();
  return last;
}
function softmaxDistribution(logits, ids, temperature = 1) {
  const temp = Math.max(1e-3, temperature);
  let max = -Infinity;
  for (const i of ids) max = Math.max(max, logits[i] / temp);
  let sum = 0;
  const out = [];
  for (const i of ids) { const w = Math.exp(logits[i] / temp - max); out.push([i, w]); sum += w; }
  return out.map(([i, w]) => [i, w / sum]);
}
function sampleDistribution(entries, rng) {
  let r = rng();
  for (const [id, p] of entries) { r -= p; if (r <= 0) return id; }
  return entries[entries.length - 1][0];
}
function chooseRepairVariable(state, rng, stochastic = true) {
  const logits = variableSnapshot(state);
  const ids = Array.from({ length: state.n }, (_, i) => i);
  if (!stochastic) return ids.reduce((best, i) => logits[i] > logits[best] ? i : best, 0);
  return sampleDistribution(softmaxDistribution(logits, ids, VARIABLE_TEMPERATURE), rng);
}
const FULL_DOMAIN_VALUES = Int32Array.from({ length: DOMAIN_MAX - DOMAIN_MIN + 1 }, (_, k) => DOMAIN_MIN + k);
function chooseValue(state, variableId, rng, stochastic = true, oracleValue = false) {
  if (oracleValue) return teacherSampleValue(variableId, rng);
  const logits = valueSnapshot(state, variableId, FULL_DOMAIN_VALUES);
  if (!stochastic) {
    let best = 0; for (let k = 1; k < logits.length; k++) if (logits[k] > logits[best]) best = k;
    return FULL_DOMAIN_VALUES[best];
  }
  const ids = Array.from({ length: logits.length }, (_, i) => i);
  const picked = sampleDistribution(softmaxDistribution(logits, ids, VALUE_TEMPERATURE), rng);
  return FULL_DOMAIN_VALUES[picked];
}
function runLearned(n, rng, stochastic = true, oracleVariable = false, oracleValue = false) {
  const state = emptyState(n), maxMoves = MOVE_MULTIPLIER * n; let moves = 0;
  while (state.assignedCount < n && moves < maxMoves) {
    const remaining = []; for (let i = 0; i < n; i++) if (!state.assigned[i]) remaining.push(i);
    const i = stochastic ? remaining[Math.floor(rng() * remaining.length)] : remaining[0];
    const v = chooseValue(state, i, rng, stochastic, oracleValue);
    applyAction(state, i, v); moves++;
  }
  while (moves < maxMoves && !isStrictChain(state.values)) {
    let i;
    if (oracleVariable) {
      const ids = violatedEndpointIds(state.values); if (!ids.length) break;
      i = stochastic ? ids[Math.floor(rng() * ids.length)] : ids[0];
    } else i = chooseRepairVariable(state, rng, stochastic);
    const v = chooseValue(state, i, rng, stochastic, oracleValue);
    applyAction(state, i, v); moves++;
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
function randomAssignment(n, rng) { return Int16Array.from({ length: n }, () => DOMAIN_MIN + Math.floor(rng() * (DOMAIN_MAX - DOMAIN_MIN + 1))); }
function runSA(n, evaluations, rng) {
  const x = randomAssignment(n, rng); let e = violationEnergy(x), best = Int16Array.from(x), bestE = e;
  const T0 = 80, Tend = 0.02;
  for (let step = 0; step < evaluations && bestE > 0; step++) {
    const i = Math.floor(rng() * n), old = x[i]; x[i] = DOMAIN_MIN + Math.floor(rng() * (DOMAIN_MAX - DOMAIN_MIN + 1));
    const next = violationEnergy(x), delta = next - e, t = step / Math.max(1, evaluations - 1), temp = T0 * Math.pow(Tend / T0, t);
    if (delta <= 0 || rng() < Math.exp(-delta / Math.max(1e-6, temp))) { e = next; if (e < bestE) { bestE = e; best = Int16Array.from(x); } } else x[i] = old;
  }
  return { x: best, solved: bestE === 0 };
}
function endpointDiagnostics(n, rng, states = 32) {
  let tp = 0, fp = 0, fn = 0, mass = 0;
  for (let s = 0; s < states; s++) {
    const sample = makeRepairTrainingState(n, rng), positive = new Set(sample.candidateIds), logits = variableSnapshot(sample.state);
    const probs = softmaxDistribution(logits, Array.from({ length: n }, (_, i) => i), VARIABLE_TEMPERATURE);
    for (let i = 0; i < n; i++) {
      const pred = sigmoidScalar(logits[i]) >= 0.5;
      if (pred && positive.has(i)) tp++; else if (pred) fp++; else if (positive.has(i)) fn++;
    }
    for (const [i, p] of probs) if (positive.has(i)) mass += p;
  }
  const precision = tp / Math.max(1, tp + fp), recall = tp / Math.max(1, tp + fn), f1 = 2 * precision * recall / Math.max(1e-9, precision + recall);
  return { precision, recall, f1, candidateMass: mass / states };
}
function valueDiagnostics(n, rng, states = 12) {
  let mae = 0, count = 0, teacherProb = 0;
  for (let s = 0; s < states; s++) {
    const sample = makeRepairTrainingState(n, rng), ids = Array.from(sample.candidateIds);
    const i = ids[Math.floor(rng() * ids.length)];
    const logits = valueSnapshot(sample.state, i, FULL_DOMAIN_VALUES);
    let best = 0; for (let k = 1; k < logits.length; k++) if (logits[k] > logits[best]) best = k;
    mae += Math.abs(FULL_DOMAIN_VALUES[best] - teacherMeanValue(i)); count++;
    const probs = softmaxDistribution(logits, Array.from({ length: logits.length }, (_, k) => k), VALUE_TEMPERATURE);
    const target = teacherMeanValue(i) - DOMAIN_MIN;
    teacherProb += probs[target]?.[1] || 0;
  }
  return { mae: mae / Math.max(1, count), teacherProb: teacherProb / Math.max(1, count) };
}
function solveRate(fn, n, seed, trials = 8) {
  const rng = mulberry32(seed); let solved = 0;
  for (let t = 0; t < trials; t++) if (fn(n, rng, true).solved) solved++;
  return solved / trials;
}
async function trainGeneric() {
  if (!window.tf) throw new Error('TensorFlow.js did not load.');
  MESSAGE_ROUNDS = +messageRounds.value;
  const startN = +trainMinN.value, maxN = +trainMaxN.value, totalStates = +episodes.value;
  trainedRange = { min: startN, max: maxN }; trained = false; trainBtn.disabled = true; benchmarkBtn.disabled = true; initModel(); resetLog();
  logLine('GENERIC_GNN_ACTION_DIAGNOSTIC v1');
  logLine(`CONFIG backend=${tf.getBackend()} train=${startN}..${maxN} states=${totalStates} rounds=${MESSAGE_ROUNDS} embed=${EMBED_DIM} valueNegatives=${VALUE_NEGATIVES} varTemp=${VARIABLE_TEMPERATURE} valueTemp=${VALUE_TEMPERATURE} budget=${MOVE_MULTIPLIER}n`);
  const rng = mulberry32(20260816); let completed = 0, updates = 0, lastLoss = NaN;
  while (completed < totalStates) {
    const batch = [], count = Math.min(BATCH_STATES, totalStates - completed);
    for (let b = 0; b < count; b++) {
      const n = startN + Math.floor(rng() * (maxN - startN + 1)); batch.push(makeTrainingSample(n, rng)); completed++;
    }
    lastLoss = await trainingUpdate(batch); updates++;
    if (updates % 4 === 0 || completed >= totalStates) {
      const ed = endpointDiagnostics(maxN, mulberry32(810000 + updates), 24);
      const vd = valueDiagnostics(maxN, mulberry32(820000 + updates), 8);
      const solve = solveRate(runLearned, maxN, 830000 + updates, 6);
      statusEl.textContent = `States ${completed}/${totalStates} · loss ${lastLoss.toFixed(3)} · endpoint F1 ${(100*ed.f1).toFixed(0)}% · candidate mass ${(100*ed.candidateMass).toFixed(0)}% · value argmax MAE ${vd.mae.toFixed(1)} · solve@${maxN} ${(100*solve).toFixed(0)}%`;
      logLine(`TRAIN states=${completed} update=${updates} loss=${lastLoss.toFixed(4)} P=${(100*ed.precision).toFixed(1)}% R=${(100*ed.recall).toFixed(1)}% F1=${(100*ed.f1).toFixed(1)}% candMass=${(100*ed.candidateMass).toFixed(1)}% valueMAE=${vd.mae.toFixed(2)} teacherValueProb=${(100*vd.teacherProb).toFixed(2)}% solve@n${maxN}=${(100*solve).toFixed(0)}%`);
      await tf.nextFrame();
    }
  }
  trained = true; trainBtn.disabled = false; benchmarkBtn.disabled = false;
  statusEl.textContent = 'Generic GNN action training complete. Running benchmark…'; await runBenchmark();
}
function benchmarkLengths() {
  const m = trainedRange.max;
  return [...new Set([trainedRange.min, m, Math.min(MAX_TEST_N, m + 1), Math.min(MAX_TEST_N, Math.round(m * 1.5)), Math.min(MAX_TEST_N, m * 2), MAX_TEST_N])].sort((a,b)=>a-b);
}
function median(values) { if (!values.length) return NaN; const a=[...values].sort((x,y)=>x-y); return a[Math.floor(a.length/2)]; }
async function runBenchmark() {
  if (!trained || !params) return;
  trainBtn.disabled = true; benchmarkBtn.disabled = true;
  const rows = [], rng = mulberry32(20260817), trials = 8;
  for (const n of benchmarkLengths()) {
    statusEl.textContent = `Benchmarking generic GNN action policy at n=${n}…`;
    let pSolved=0,hSolved=0,saSolved=0,oracleVarSolved=0,oracleValueSolved=0,pViol=0,hViol=0;
    const pMoves=[],hMoves=[];
    for (let t=0;t<trials;t++) {
      const p=runLearned(n,rng,true,false,false); if(p.solved){pSolved++;pMoves.push(p.moves);} pViol+=violatedConstraintCount(p.x);
      if(runLearned(n,rng,true,true,false).solved) oracleVarSolved++;
      if(runLearned(n,rng,true,false,true).solved) oracleValueSolved++;
      const h=runHandcrafted(n,rng); if(h.solved){hSolved++;hMoves.push(h.moves);} hViol+=violatedConstraintCount(h.x);
      if(runSA(n,120*n,rng).solved) saSolved++;
      await tf.nextFrame();
    }
    rows.push({n,policySuccess:pSolved/trials,heuristicSuccess:hSolved/trials,saSuccess:saSolved/trials,policyMoves:median(pMoves),heuristicMoves:median(hMoves)});
    logLine(`BENCH n=${n} gnn=${Math.round(100*pSolved/trials)}% oracleVariable=${Math.round(100*oracleVarSolved/trials)}% oracleValue=${Math.round(100*oracleValueSolved/trials)}% teacher=${Math.round(100*hSolved/trials)}% sa=${Math.round(100*saSolved/trials)}% gnnAvgViol=${(pViol/trials).toFixed(2)} teacherAvgViol=${(hViol/trials).toFixed(2)}`);
  }
  drawScaling(rows);
  const longest=rows[rows.length-1], n=longest.n, p=runLearned(n,mulberry32(9101),true), h=runHandcrafted(n,mulberry32(9102)), sa=runSA(n,120*n,mulberry32(9103)); drawAssignment(p.x,h.x,sa.x);
  const ed=endpointDiagnostics(trainedRange.max,mulberry32(9200),48), vd=valueDiagnostics(trainedRange.max,mulberry32(9201),12);
  $('policySuccess').textContent=`${Math.round(100*longest.policySuccess)}%`; $('heuristicSuccess').textContent=`${Math.round(100*longest.heuristicSuccess)}%`; $('saSuccess').textContent=`${Math.round(100*longest.saSuccess)}%`;
  $('candidateMass').textContent=`${Math.round(100*ed.f1)}%`; $('meanMae').textContent=vd.mae.toFixed(1);
  $('policyMoves').textContent=Number.isFinite(longest.policyMoves)?Math.round(longest.policyMoves):'—'; $('heuristicMoves').textContent=Number.isFinite(longest.heuristicMoves)?Math.round(longest.heuristicMoves):'—'; $('saEffort').textContent=(120*n).toLocaleString();
  $('metricLength').textContent=`n=${n} · generic variable-score + candidate-value-score architecture · train ${trainedRange.min}…${trainedRange.max} · depth ${MESSAGE_ROUNDS}`;
  appendProbe(trainedRange.max); appendTrace(trainedRange.max); logLine('END');
  statusEl.textContent='Done. The architecture contains no hard-coded value formula; copy diagnostics if needed.'; trainBtn.disabled=false; benchmarkBtn.disabled=false;
}
function appendProbe(n) {
  const sample=makeRepairTrainingState(n,mulberry32(9300+n)), positives=new Set(sample.candidateIds), logits=variableSnapshot(sample.state);
  const probs=softmaxDistribution(logits,Array.from({length:n},(_,i)=>i),VARIABLE_TEMPERATURE).sort((a,b)=>b[1]-a[1]);
  logLine(`PROBE_VARIABLE n=${n} violations=${violatedConstraintCount(sample.state.values)} candidates=[${[...positives].join(',')}] top=[${probs.slice(0,12).map(([i,p])=>`${i}:${(100*p).toFixed(1)}%${positives.has(i)?'*':''}`).join(',')}]`);
  const i=Array.from(sample.candidateIds)[0];
  const vLogits=valueSnapshot(sample.state,i,FULL_DOMAIN_VALUES), ids=Array.from({length:vLogits.length},(_,k)=>k), vProbs=softmaxDistribution(vLogits,ids,VALUE_TEMPERATURE).sort((a,b)=>b[1]-a[1]);
  logLine(`PROBE_VALUE n=${n} variable=${i} teacherMean=${teacherMeanValue(i)} top=[${vProbs.slice(0,10).map(([k,p])=>`${FULL_DOMAIN_VALUES[k]}:${(100*p).toFixed(2)}%`).join(',')}]`);
}
function appendTrace(n) {
  const rng=mulberry32(9400+n), state=emptyState(n), maxMoves=MOVE_MULTIPLIER*n; let moves=0;
  while(state.assignedCount<n&&moves<maxMoves){const rem=[];for(let i=0;i<n;i++)if(!state.assigned[i])rem.push(i);const i=rem[Math.floor(rng()*rem.length)],v=chooseValue(state,i,rng,true,false);applyAction(state,i,v);moves++;}
  logLine(`TRACE_START n=${n} violations=${violatedConstraintCount(state.values)} energy=${violationEnergy(state.values)}`);
  for(let step=0;step<16&&moves<maxMoves&&!isStrictChain(state.values);step++){
    const valid=new Set(violatedEndpointIds(state.values)),before=violatedConstraintCount(state.values),i=chooseRepairVariable(state,rng,true),v=chooseValue(state,i,rng,true,false);applyAction(state,i,v);moves++;
    logLine(`TRACE_REPAIR step=${step} choose=${i} valid=${valid.has(i)?1:0} v=${v} viol=${before}->${violatedConstraintCount(state.values)} energy=${violationEnergy(state.values)}`);
  }
  logLine(`TRACE_END solved=${isStrictChain(state.values)?1:0} moves=${moves} violations=${violatedConstraintCount(state.values)} energy=${violationEnergy(state.values)}`);
}
function setupCanvas(canvas){const dpr=window.devicePixelRatio||1,rect=canvas.getBoundingClientRect();canvas.width=Math.max(300,Math.floor(rect.width*dpr));canvas.height=Math.floor(Math.max(260,rect.width*.46)*dpr);const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return{ctx,w:canvas.width/dpr,h:canvas.height/dpr};}
function axes(ctx,w,h,pad,xl,yl){ctx.clearRect(0,0,w,h);ctx.strokeStyle='#dbe1eb';ctx.beginPath();ctx.moveTo(pad,14);ctx.lineTo(pad,h-pad);ctx.lineTo(w-12,h-pad);ctx.stroke();ctx.fillStyle='#697386';ctx.font='12px system-ui';ctx.fillText(yl,pad+6,24);ctx.textAlign='right';ctx.fillText(xl,w-12,h-10);ctx.textAlign='left';}
function line(ctx,pts,color,width=2.4){if(!pts.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();ctx.fillStyle=color;for(const p of pts){ctx.beginPath();ctx.arc(p[0],p[1],2.6,0,Math.PI*2);ctx.fill();}}
function drawScaling(rows){const{ctx,w,h}=setupCanvas($('scalingChart')),pad=44;axes(ctx,w,h,pad,'chain length n','solve rate');const lo=Math.min(...rows.map(r=>r.n)),hi=Math.max(...rows.map(r=>r.n)),X=n=>pad+(n-lo)/Math.max(1,hi-lo)*(w-pad-20),Y=p=>h-pad-p*(h-pad-28);ctx.fillStyle='#7b8495';ctx.font='11px system-ui';for(const r of rows)ctx.fillText(String(r.n),X(r.n)-6,h-pad+17);for(let k=0;k<=4;k++)ctx.fillText(`${25*k}%`,5,Y(k/4)+4);const bx=X(trainedRange.max);ctx.save();ctx.setLineDash([5,5]);ctx.strokeStyle='#a9b1c2';ctx.beginPath();ctx.moveTo(bx,18);ctx.lineTo(bx,h-pad);ctx.stroke();ctx.restore();ctx.fillText('train max',Math.min(w-80,bx+5),28);line(ctx,rows.map(r=>[X(r.n),Y(r.policySuccess)]),'#5b67d6');line(ctx,rows.map(r=>[X(r.n),Y(r.heuristicSuccess)]),'#2e8b72');line(ctx,rows.map(r=>[X(r.n),Y(r.saSuccess)]),'#dd6b55');}
function drawAssignment(policy,teacher,sa){const{ctx,w,h}=setupCanvas($('instanceChart')),pad=44;axes(ctx,w,h,pad,'variable node id i','assigned value');const n=policy.length,X=i=>pad+i/Math.max(1,n-1)*(w-pad-20),Y=v=>h-pad-Math.max(0,v)/DOMAIN_MAX*(h-pad-28);line(ctx,Array.from(policy,(v,i)=>[X(i),Y(v)]),'#5b67d6',2.2);line(ctx,Array.from(teacher,(v,i)=>[X(i),Y(v)]),'#2e8b72',2);line(ctx,Array.from(sa,(v,i)=>[X(i),Y(v)]),'#dd6b55',1.8);}
trainBtn.addEventListener('click',()=>trainGeneric().catch(err=>{console.error(err);statusEl.textContent=`Error: ${err.message}`;logLine(`ERROR ${err.stack||err.message}`);trainBtn.disabled=false;benchmarkBtn.disabled=!trained;}));
benchmarkBtn.addEventListener('click',()=>runBenchmark().catch(err=>{console.error(err);statusEl.textContent=`Error: ${err.message}`;logLine(`ERROR ${err.stack||err.message}`);trainBtn.disabled=false;benchmarkBtn.disabled=false;}));
statusEl.textContent=`TensorFlow.js ready · backend: ${tf.getBackend()}. Generic GNN action model: variable scorer + value-candidate scorer.`;
