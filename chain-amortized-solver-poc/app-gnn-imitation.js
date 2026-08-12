const DOMAIN_MAX = 1000;
const MAX_TEST_N = 100;
const NODE_FEATURE_DIM = 8;
const EMBED_DIM = 16;
const EDGE_TYPES = 4;
const MOVE_MULTIPLIER = 3;
const TEACHER_SIGMA = 1.0;
const BATCH_EPISODES = 16;
const SAMPLES_PER_EPISODE = 6;
const UPDATE_EPOCHS = 2;

let MESSAGE_ROUNDS = 1;
let params = null;
let optimizer = null;
let trained = false;
let trainedRange = { min: 2, max: 20 };
let graphCache = new Map();
let paramGeneration = 0;

const $ = id => document.getElementById(id);
const trainMinN = $('trainMinN');
const trainMaxN = $('trainMaxN');
const episodes = $('episodes');
const messageRounds = $('messageRounds');
const trainBtn = $('trainBtn');
const benchmarkBtn = $('benchmarkBtn');
const statusEl = $('status');

function bindRange(input, labelId) {
  const update = () => $(labelId).textContent = input.value;
  input.addEventListener('input', update);
  update();
}
bindRange(trainMinN, 'trainMinNLabel');
bindRange(trainMaxN, 'trainMaxNLabel');
bindRange(episodes, 'episodesLabel');

trainMinN.addEventListener('input', () => {
  if (+trainMinN.value > +trainMaxN.value) {
    trainMaxN.value = trainMinN.value;
    $('trainMaxNLabel').textContent = trainMaxN.value;
  }
});
trainMaxN.addEventListener('input', () => {
  if (+trainMaxN.value < +trainMinN.value) {
    trainMinN.value = trainMaxN.value;
    $('trainMinNLabel').textContent = trainMinN.value;
  }
});

messageRounds.addEventListener('change', () => {
  if (trained) {
    trained = false;
    benchmarkBtn.disabled = true;
    statusEl.textContent = 'Message-passing depth changed. Retrain before benchmarking.';
  }
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
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + sigma * z;
}

function emptyState(n) {
  return {
    n,
    assigned: new Uint8Array(n),
    values: Int16Array.from({ length: n }, () => -1),
    assignedCount: 0
  };
}

function cloneState(state) {
  return {
    n: state.n,
    assigned: Uint8Array.from(state.assigned),
    values: Int16Array.from(state.values),
    assignedCount: state.assignedCount
  };
}

function applyAction(state, i, v) {
  if (!state.assigned[i]) {
    state.assigned[i] = 1;
    state.assignedCount++;
  }
  state.values[i] = v;
}

function isStrictChain(x) {
  for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) return false;
  return true;
}

function violationEnergy(x) {
  let e = 0;
  for (let i = 0; i + 1 < x.length; i++) e += Math.max(0, x[i] - x[i + 1] + 1);
  return e;
}

function violatedEndpointIds(x) {
  const marked = new Uint8Array(x.length);
  for (let i = 0; i + 1 < x.length; i++) {
    if (x[i] >= x[i + 1]) {
      marked[i] = 1;
      marked[i + 1] = 1;
    }
  }
  const ids = [];
  for (let i = 0; i < x.length; i++) if (marked[i]) ids.push(i);
  return ids;
}

function allowedVariableIds(state) {
  if (state.assignedCount === state.n) return Array.from({ length: state.n }, (_, i) => i);
  const ids = [];
  for (let i = 0; i < state.n; i++) if (!state.assigned[i]) ids.push(i);
  return ids;
}

function teacherCandidateIds(state) {
  if (state.assignedCount < state.n) return allowedVariableIds(state);
  return violatedEndpointIds(state.values);
}

function teacherMeanValue(i) {
  return Math.max(0, Math.min(DOMAIN_MAX, 2 * i));
}

function teacherSampleValue(i, rng) {
  return Math.max(0, Math.min(DOMAIN_MAX, Math.round(gaussian(teacherMeanValue(i), TEACHER_SIGMA, rng))));
}

function makeVariable(shape, scale, name) {
  const init = tf.randomNormal(shape, 0, scale);
  const v = tf.variable(init, true, `${name}_${paramGeneration}`);
  init.dispose();
  return v;
}

function makeBias(size, name) {
  const init = tf.zeros([size]);
  const v = tf.variable(init, true, `${name}_${paramGeneration}`);
  init.dispose();
  return v;
}

function trainableParams() {
  if (!params) return [];
  return [
    params.Winit, params.binit,
    params.Wself, params.brel, ...params.Wrel,
    params.Wvar, params.bvar,
    params.Wmean, params.bmean,
    params.WlogStd, params.blogStd
  ];
}

function disposeModel() {
  if (params) for (const p of trainableParams()) p.dispose();
  if (optimizer?.dispose) optimizer.dispose();
  params = null;
  optimizer = null;
}

function initModel() {
  disposeModel();
  paramGeneration++;
  const s = 0.08;
  params = {
    Winit: makeVariable([NODE_FEATURE_DIM, EMBED_DIM], s, 'Winit'),
    binit: makeBias(EMBED_DIM, 'binit'),
    Wself: makeVariable([EMBED_DIM, EMBED_DIM], s, 'Wself'),
    brel: makeBias(EMBED_DIM, 'brel'),
    Wrel: Array.from({ length: EDGE_TYPES }, (_, r) => makeVariable([EMBED_DIM, EMBED_DIM], s, `Wrel${r}`)),
    Wvar: makeVariable([EMBED_DIM, 1], s, 'Wvar'),
    bvar: makeBias(1, 'bvar'),
    Wmean: makeVariable([EMBED_DIM, 1], s * 0.3, 'Wmean'),
    bmean: makeBias(1, 'bmean'),
    WlogStd: makeVariable([EMBED_DIM, 1], s * 0.1, 'WlogStd'),
    blogStd: makeBias(1, 'blogStd')
  };
  optimizer = tf.train.adam(0.001);
}

function graphSpec(n) {
  if (graphCache.has(n)) return graphCache.get(n);
  const constraintCount = n - 1;
  const nodeCount = n + constraintCount;
  const baseFeatures = new Float32Array(nodeCount * NODE_FEATURE_DIM);
  const absScale = Math.max(1, MAX_TEST_N - 1);
  const sizeFeature = n / MAX_TEST_N;

  for (let i = 0; i < n; i++) {
    const p = i * NODE_FEATURE_DIM;
    baseFeatures[p] = 1;
    baseFeatures[p + 1] = 0;
    baseFeatures[p + 2] = i / absScale;
    baseFeatures[p + 3] = n <= 1 ? 0 : i / (n - 1);
    baseFeatures[p + 4] = sizeFeature;
  }
  for (let k = 0; k < constraintCount; k++) {
    const node = n + k;
    const p = node * NODE_FEATURE_DIM;
    baseFeatures[p] = 0;
    baseFeatures[p + 1] = 1;
    baseFeatures[p + 2] = k / absScale;
    baseFeatures[p + 3] = constraintCount <= 1 ? 0 : k / (constraintCount - 1);
    baseFeatures[p + 4] = sizeFeature;
  }

  const adjs = Array.from({ length: EDGE_TYPES }, () => new Float32Array(nodeCount * nodeCount));
  const setEdge = (type, target, source) => { adjs[type][target * nodeCount + source] = 1; };
  for (let k = 0; k < constraintCount; k++) {
    const c = n + k;
    setEdge(0, c, k);
    setEdge(1, c, k + 1);
    setEdge(2, k, c);
    setEdge(3, k + 1, c);
  }

  const spec = { n, constraintCount, nodeCount, baseFeatures, adjs };
  graphCache.set(n, spec);
  return spec;
}

function nodeFeatureData(state) {
  const spec = graphSpec(state.n);
  const data = Float32Array.from(spec.baseFeatures);
  for (let i = 0; i < state.n; i++) {
    const p = i * NODE_FEATURE_DIM;
    data[p + 5] = state.assigned[i] ? 1 : 0;
    data[p + 6] = state.assigned[i] ? state.values[i] / DOMAIN_MAX : 0;
  }
  for (let k = 0; k + 1 < state.n; k++) {
    const node = state.n + k;
    const p = node * NODE_FEATURE_DIM;
    if (state.assigned[k] && state.assigned[k + 1]) {
      data[p + 7] = state.values[k] < state.values[k + 1] ? -1 : 1;
    } else {
      data[p + 7] = 0;
    }
  }
  return data;
}

function rowLayerNorm(x) {
  const mean = x.mean(1, true);
  const centered = x.sub(mean);
  const variance = centered.square().mean(1, true);
  return centered.div(variance.add(1e-5).sqrt());
}

function graphEmbeddings(state) {
  const spec = graphSpec(state.n);
  const nodeX = tf.tensor2d(nodeFeatureData(state), [spec.nodeCount, NODE_FEATURE_DIM]);
  const adj = spec.adjs.map(a => tf.tensor2d(a, [spec.nodeCount, spec.nodeCount]));
  let h = tf.relu(tf.matMul(nodeX, params.Winit).add(params.binit));
  for (let round = 0; round < MESSAGE_ROUNDS; round++) {
    let z = tf.matMul(h, params.Wself);
    for (let r = 0; r < EDGE_TYPES; r++) z = z.add(tf.matMul(adj[r], tf.matMul(h, params.Wrel[r])));
    h = tf.relu(rowLayerNorm(h.add(z.add(params.brel))));
  }
  return h;
}

function policyTensors(state) {
  const h = graphEmbeddings(state);
  const variableH = h.slice([0, 0], [state.n, EMBED_DIM]);
  const varLogits = tf.matMul(variableH, params.Wvar).add(params.bvar).reshape([state.n]);
  const meanNorm = tf.matMul(variableH, params.Wmean).add(params.bmean).reshape([state.n]);
  const logStd = tf.matMul(variableH, params.WlogStd).add(params.blogStd).reshape([state.n]).clipByValue(-3, 2);
  return { h, variableH, varLogits, meanNorm, logStd };
}

function policySnapshot(state) {
  return tf.tidy(() => {
    const p = policyTensors(state);
    return {
      varLogits: Float32Array.from(p.varLogits.dataSync()),
      meanNorm: Float32Array.from(p.meanNorm.dataSync()),
      logStd: Float32Array.from(p.logStd.dataSync())
    };
  });
}

function sampleCategorical(logits, allowedIds, rng) {
  let max = -Infinity;
  for (const i of allowedIds) max = Math.max(max, logits[i]);
  const weights = new Float64Array(allowedIds.length);
  let sum = 0;
  for (let k = 0; k < allowedIds.length; k++) {
    const w = Math.exp(logits[allowedIds[k]] - max);
    weights[k] = w;
    sum += w;
  }
  let r = rng() * sum;
  for (let k = 0; k < weights.length; k++) {
    r -= weights[k];
    if (r <= 0) return allowedIds[k];
  }
  return allowedIds[allowedIds.length - 1];
}

function teacherTrajectory(n, rng) {
  const state = emptyState(n);
  const construction = [];
  const repair = [];
  const maxMoves = MOVE_MULTIPLIER * n;
  let moves = 0;
  while (moves < maxMoves) {
    if (state.assignedCount === n && isStrictChain(state.values)) break;
    const candidates = teacherCandidateIds(state);
    if (!candidates.length) break;
    const bucket = state.assignedCount < n ? construction : repair;
    bucket.push({ state: cloneState(state), candidateIds: Int32Array.from(candidates) });
    const i = candidates[Math.floor(rng() * candidates.length)];
    applyAction(state, i, teacherSampleValue(i, rng));
    moves++;
  }
  return { construction, repair, state, moves };
}

function imitationAction(state, rng, stochastic = true) {
  const snap = policySnapshot(state);
  const allowed = allowedVariableIds(state);
  let i;
  if (stochastic) i = sampleCategorical(snap.varLogits, allowed, rng);
  else {
    i = allowed[0];
    for (const j of allowed) if (snap.varLogits[j] > snap.varLogits[i]) i = j;
  }
  const meanValue = Math.max(0, Math.min(DOMAIN_MAX, DOMAIN_MAX * snap.meanNorm[i]));
  const sigmaValue = Math.exp(snap.logStd[i]);
  const raw = stochastic ? gaussian(meanValue, sigmaValue, rng) : meanValue;
  const v = Math.max(0, Math.min(DOMAIN_MAX, Math.round(raw)));
  return { i, v, meanValue, sigmaValue };
}

function learnerTrajectoryLabeled(n, rng) {
  const state = emptyState(n);
  const construction = [];
  const repair = [];
  const maxMoves = MOVE_MULTIPLIER * n;
  let moves = 0;
  while (moves < maxMoves) {
    if (state.assignedCount === n && isStrictChain(state.values)) break;
    const candidates = teacherCandidateIds(state);
    if (!candidates.length) break;
    const bucket = state.assignedCount < n ? construction : repair;
    bucket.push({ state: cloneState(state), candidateIds: Int32Array.from(candidates) });
    const a = imitationAction(state, rng, true);
    applyAction(state, a.i, a.v);
    moves++;
  }
  return { construction, repair, state, moves };
}

function pickSamples(trajectory, rng) {
  const out = [];
  function take(arr, count) {
    if (!arr.length || count <= 0) return;
    const chosen = new Set();
    while (chosen.size < Math.min(count, arr.length)) chosen.add(Math.floor(rng() * arr.length));
    for (const idx of chosen) out.push(arr[idx]);
  }
  const repairTarget = Math.ceil(SAMPLES_PER_EPISODE / 2);
  take(trajectory.repair, repairTarget);
  take(trajectory.construction, SAMPLES_PER_EPISODE - out.length);
  if (out.length < SAMPLES_PER_EPISODE) take(trajectory.repair, SAMPLES_PER_EPISODE - out.length);
  return out;
}

function scaledHuber(error, delta) {
  const e = error.div(delta);
  const a = e.abs();
  const quadratic = tf.minimum(a, tf.scalar(1));
  const linear = a.sub(quadratic);
  return quadratic.square().mul(0.5).add(linear);
}

function imitationSampleLoss(sample) {
  const p = policyTensors(sample.state);
  const legal = allowedVariableIds(sample.state);
  const legalIds = tf.tensor1d(Int32Array.from(legal), 'int32');
  const legalLogits = tf.gather(p.varLogits, legalIds);
  const logPolicy = tf.logSoftmax(legalLogits);
  const candidateLocal = sample.candidateIds.map(id => legal.indexOf(id));
  const candidateLocalIds = tf.tensor1d(candidateLocal, 'int32');
  const variableLoss = tf.gather(logPolicy, candidateLocalIds).mean().neg();

  const predictedValue = p.meanNorm.mul(DOMAIN_MAX);
  const targetValue = tf.range(0, sample.state.n, 1, 'float32').mul(2);
  const meanLoss = scaledHuber(predictedValue.sub(targetValue), 5).mean().mul(2.0);

  const predictedSigma = tf.exp(p.logStd);
  const sigmaLoss = predictedSigma.sub(TEACHER_SIGMA).square().mean().mul(0.2);
  return variableLoss.add(meanLoss).add(sigmaLoss);
}

async function imitationUpdate(samples) {
  if (!samples.length) return NaN;
  let last = NaN;
  for (let epoch = 0; epoch < UPDATE_EPOCHS; epoch++) {
    const cost = optimizer.minimize(() => tf.tidy(() => {
      let total = tf.scalar(0);
      for (const sample of samples) total = total.add(imitationSampleLoss(sample));
      return total.div(samples.length);
    }), true, trainableParams());
    last = cost.dataSync()[0];
    cost.dispose();
    await tf.nextFrame();
  }
  return last;
}

function runImitation(n, rng, stochastic = true) {
  const state = emptyState(n);
  const maxMoves = MOVE_MULTIPLIER * n;
  let moves = 0;
  while (moves < maxMoves) {
    if (state.assignedCount === n && isStrictChain(state.values)) break;
    const a = imitationAction(state, rng, stochastic);
    applyAction(state, a.i, a.v);
    moves++;
  }
  return { x: Int16Array.from(state.values), solved: state.assignedCount === n && isStrictChain(state.values), moves };
}

function runHandcrafted(n, rng) {
  const t = teacherTrajectory(n, rng);
  return { x: Int16Array.from(t.state.values), solved: t.state.assignedCount === n && isStrictChain(t.state.values), moves: t.moves };
}

function randomAssignment(n, rng) {
  return Int16Array.from({ length: n }, () => Math.floor(rng() * (DOMAIN_MAX + 1)));
}

function runSA(n, evaluations, rng) {
  const x = randomAssignment(n, rng);
  let e = violationEnergy(x);
  let best = Int16Array.from(x);
  let bestE = e;
  const T0 = 80;
  const Tend = 0.02;
  for (let step = 0; step < evaluations && bestE > 0; step++) {
    const i = Math.floor(rng() * n);
    const old = x[i];
    x[i] = Math.floor(rng() * (DOMAIN_MAX + 1));
    const next = violationEnergy(x);
    const delta = next - e;
    const t = step / Math.max(1, evaluations - 1);
    const temp = T0 * Math.pow(Tend / T0, t);
    if (delta <= 0 || rng() < Math.exp(-delta / Math.max(1e-6, temp))) {
      e = next;
      if (e < bestE) { bestE = e; best = Int16Array.from(x); }
    } else x[i] = old;
  }
  return { x: best, solved: bestE === 0 };
}

function representationDiagnostics(n, rng, samplesTarget = 24) {
  const traj = teacherTrajectory(n, rng);
  let pool = [...traj.repair, ...traj.construction];
  if (!pool.length) return { candidateMass: NaN, meanMae: NaN, sigmaMae: NaN };
  if (pool.length > samplesTarget) {
    const picked = [];
    for (let k = 0; k < samplesTarget; k++) picked.push(pool[Math.floor(rng() * pool.length)]);
    pool = picked;
  }
  let candidateMass = 0;
  let meanMae = 0;
  let sigmaMae = 0;
  for (const sample of pool) {
    const snap = policySnapshot(sample.state);
    const legal = allowedVariableIds(sample.state);
    let max = -Infinity;
    for (const i of legal) max = Math.max(max, snap.varLogits[i]);
    let denom = 0;
    const weights = new Map();
    for (const i of legal) {
      const w = Math.exp(snap.varLogits[i] - max);
      weights.set(i, w);
      denom += w;
    }
    let mass = 0;
    for (const i of sample.candidateIds) mass += (weights.get(i) || 0) / denom;
    candidateMass += mass;
    let mae = 0;
    let smae = 0;
    for (let i = 0; i < sample.state.n; i++) {
      mae += Math.abs(DOMAIN_MAX * snap.meanNorm[i] - teacherMeanValue(i));
      smae += Math.abs(Math.exp(snap.logStd[i]) - TEACHER_SIGMA);
    }
    meanMae += mae / sample.state.n;
    sigmaMae += smae / sample.state.n;
  }
  return {
    candidateMass: candidateMass / pool.length,
    meanMae: meanMae / pool.length,
    sigmaMae: sigmaMae / pool.length
  };
}

async function trainImitation() {
  if (!window.tf) throw new Error('TensorFlow.js did not load.');
  MESSAGE_ROUNDS = +messageRounds.value;
  const startN = +trainMinN.value;
  const maxN = +trainMaxN.value;
  const totalEpisodes = +episodes.value;
  trainedRange = { min: startN, max: maxN };
  trained = false;
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;
  initModel();

  const rng = mulberry32(20260812);
  let completed = 0;
  let updates = 0;
  let lastLoss = NaN;
  while (completed < totalEpisodes) {
    const samples = [];
    const batchEpisodes = Math.min(BATCH_EPISODES, totalEpisodes - completed);
    for (let b = 0; b < batchEpisodes; b++) {
      const n = startN + Math.floor(rng() * (maxN - startN + 1));
      const useLearnerRollout = updates >= 8 && rng() < 0.6;
      const traj = useLearnerRollout ? learnerTrajectoryLabeled(n, rng) : teacherTrajectory(n, rng);
      samples.push(...pickSamples(traj, rng));
      completed++;
    }
    lastLoss = await imitationUpdate(samples);
    updates++;

    if (updates % 4 === 0 || completed >= totalEpisodes) {
      const diag = representationDiagnostics(maxN, mulberry32(900000 + updates), 24);
      const solve = solveRate(runImitation, maxN, 710000 + updates, 8);
      statusEl.textContent = `Episodes ${completed}/${totalEpisodes} · update ${updates} · train n=${startN}…${maxN} · depth ${MESSAGE_ROUNDS} · loss ${lastLoss.toFixed(3)} · candidate mass ${(100 * diag.candidateMass).toFixed(0)}% · mean MAE ${diag.meanMae.toFixed(1)} · solve@${maxN} ${(100 * solve).toFixed(0)}%`;
      await tf.nextFrame();
    }
  }

  trained = true;
  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
  statusEl.textContent = 'Imitation training complete. Running GNN / teacher / SA benchmark…';
  await runBenchmark();
}

function solveRate(fn, n, seed, trials = 12) {
  const rng = mulberry32(seed);
  let solved = 0;
  for (let t = 0; t < trials; t++) if (fn(n, rng, true).solved) solved++;
  return solved / trials;
}

function benchmarkLengths() {
  const m = trainedRange.max;
  return [...new Set([
    trainedRange.min,
    m,
    Math.min(MAX_TEST_N, m + 1),
    Math.min(MAX_TEST_N, Math.round(m * 1.5)),
    Math.min(MAX_TEST_N, m * 2),
    MAX_TEST_N
  ])].sort((a, b) => a - b);
}

function median(values) {
  if (!values.length) return NaN;
  const a = [...values].sort((x, y) => x - y);
  return a[Math.floor(a.length / 2)];
}

async function runBenchmark() {
  if (!trained || !params) return;
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;
  const rows = [];
  const rng = mulberry32(20260813);
  const trials = 12;

  for (const n of benchmarkLengths()) {
    statusEl.textContent = `Benchmarking imitation GNN, teacher, and SA at n=${n}…`;
    let policySolved = 0, teacherSolved = 0, saSolved = 0;
    const policyMoves = [], teacherMoves = [];
    for (let t = 0; t < trials; t++) {
      const p = runImitation(n, rng, true);
      if (p.solved) { policySolved++; policyMoves.push(p.moves); }
      const h = runHandcrafted(n, rng);
      if (h.solved) { teacherSolved++; teacherMoves.push(h.moves); }
      if (runSA(n, 120 * n, rng).solved) saSolved++;
      await tf.nextFrame();
    }
    rows.push({
      n,
      policySuccess: policySolved / trials,
      heuristicSuccess: teacherSolved / trials,
      saSuccess: saSolved / trials,
      policyMoves: median(policyMoves),
      heuristicMoves: median(teacherMoves)
    });
  }

  drawScaling(rows);
  const longest = rows[rows.length - 1];
  const n = longest.n;
  const policy = runImitation(n, mulberry32(9101), true);
  const teacher = runHandcrafted(n, mulberry32(9102));
  const sa = runSA(n, 120 * n, mulberry32(9103));
  drawAssignment(policy.x, teacher.x, sa.x);

  const diag = representationDiagnostics(Math.min(n, trainedRange.max), mulberry32(9200), 32);
  $('policySuccess').textContent = `${Math.round(100 * longest.policySuccess)}%`;
  $('heuristicSuccess').textContent = `${Math.round(100 * longest.heuristicSuccess)}%`;
  $('saSuccess').textContent = `${Math.round(100 * longest.saSuccess)}%`;
  $('policyMoves').textContent = Number.isFinite(longest.policyMoves) ? Math.round(longest.policyMoves).toString() : '—';
  $('heuristicMoves').textContent = Number.isFinite(longest.heuristicMoves) ? Math.round(longest.heuristicMoves).toString() : '—';
  $('saEffort').textContent = (120 * n).toLocaleString();
  if ($('candidateMass')) $('candidateMass').textContent = `${Math.round(100 * diag.candidateMass)}%`;
  if ($('meanMae')) $('meanMae').textContent = diag.meanMae.toFixed(1);
  $('metricLength').textContent = `n=${n} · imitation/teacher ${MOVE_MULTIPLIER}n · SA 120n · train ${trainedRange.min}…${trainedRange.max} · depth ${MESSAGE_ROUNDS}`;
  statusEl.textContent = 'Done. Candidate mass and mean MAE diagnose representation separately from closed-loop solve rate.';
  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(300, Math.floor(rect.width * dpr));
  canvas.height = Math.floor(Math.max(260, rect.width * 0.46) * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: canvas.width / dpr, h: canvas.height / dpr };
}

function axes(ctx, w, h, pad, xl, yl) {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#dbe1eb';
  ctx.beginPath();
  ctx.moveTo(pad, 14); ctx.lineTo(pad, h - pad); ctx.lineTo(w - 12, h - pad); ctx.stroke();
  ctx.fillStyle = '#697386'; ctx.font = '12px system-ui';
  ctx.fillText(yl, pad + 6, 24); ctx.textAlign = 'right'; ctx.fillText(xl, w - 12, h - 10); ctx.textAlign = 'left';
}

function line(ctx, pts, color, width = 2.4) {
  if (!pts.length) return;
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
  pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke();
  ctx.fillStyle = color;
  for (const p of pts) { ctx.beginPath(); ctx.arc(p[0], p[1], 2.6, 0, Math.PI * 2); ctx.fill(); }
}

function drawScaling(rows) {
  const { ctx, w, h } = setupCanvas($('scalingChart'));
  const pad = 44;
  axes(ctx, w, h, pad, 'chain length n', 'solve rate');
  const lo = Math.min(...rows.map(r => r.n));
  const hi = Math.max(...rows.map(r => r.n));
  const X = n => pad + (n - lo) / Math.max(1, hi - lo) * (w - pad - 20);
  const Y = p => h - pad - p * (h - pad - 28);
  ctx.fillStyle = '#7b8495'; ctx.font = '11px system-ui';
  for (const r of rows) ctx.fillText(String(r.n), X(r.n) - 6, h - pad + 17);
  for (let k = 0; k <= 4; k++) ctx.fillText(`${25 * k}%`, 5, Y(k / 4) + 4);
  const bx = X(trainedRange.max);
  ctx.save(); ctx.setLineDash([5, 5]); ctx.strokeStyle = '#a9b1c2'; ctx.beginPath(); ctx.moveTo(bx, 18); ctx.lineTo(bx, h - pad); ctx.stroke(); ctx.restore();
  ctx.fillText('train max', Math.min(w - 80, bx + 5), 28);
  line(ctx, rows.map(r => [X(r.n), Y(r.policySuccess)]), '#5b67d6');
  line(ctx, rows.map(r => [X(r.n), Y(r.heuristicSuccess)]), '#2e8b72');
  line(ctx, rows.map(r => [X(r.n), Y(r.saSuccess)]), '#dd6b55');
}

function drawAssignment(policy, teacher, sa) {
  const { ctx, w, h } = setupCanvas($('instanceChart'));
  const pad = 44;
  axes(ctx, w, h, pad, 'variable node id i', 'assigned value');
  const n = policy.length;
  const X = i => pad + i / Math.max(1, n - 1) * (w - pad - 20);
  const Y = v => h - pad - Math.max(0, v) / DOMAIN_MAX * (h - pad - 28);
  line(ctx, Array.from(policy, (v, i) => [X(i), Y(v)]), '#5b67d6', 2.2);
  line(ctx, Array.from(teacher, (v, i) => [X(i), Y(v)]), '#2e8b72', 2.0);
  line(ctx, Array.from(sa, (v, i) => [X(i), Y(v)]), '#dd6b55', 1.8);
}

trainBtn.addEventListener('click', () => trainImitation().catch(err => {
  console.error(err);
  statusEl.textContent = `Error: ${err.message}`;
  trainBtn.disabled = false;
  benchmarkBtn.disabled = !trained;
}));
benchmarkBtn.addEventListener('click', () => runBenchmark().catch(err => {
  console.error(err);
  statusEl.textContent = `Error: ${err.message}`;
  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
}));

statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Standalone imitation GNN with legal masks, value-space loss, and DAgger relabeling.`;
