const MAX_TEST_N = 100;
const FEATURE_DIM = 10;
const GAMMA = 0.97;
const REPLAY_LIMIT = 8000;

let qModel = null;
let trained = false;
let trainedRange = { min: 6, max: 20 };
let replay = [];

const $ = id => document.getElementById(id);
const trainMinN = $('trainMinN');
const trainMaxN = $('trainMaxN');
const episodes = $('episodes');
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

function mulberry32(seed) {
  return function () {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffledPermutation(n, rng) {
  const x = Int32Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [x[i], x[j]] = [x[j], x[i]];
  }
  return x;
}

function pairCount(n) {
  return Math.max(1, n * (n - 1) / 2);
}

// Dense monotonic-constraint energy. We count every violated implied ordering
// i < j => x[i] < x[j], not only adjacent pairs. This is just the transitive
// closure of the chain and gives RL a much denser signal.
// Values are restricted to 0..n-1, so energy=0 has exactly one solution: x[i]=i.
function energy(x) {
  let e = 0;
  for (let i = 0; i < x.length - 1; i++) {
    for (let j = i + 1; j < x.length; j++) {
      if (x[i] >= x[j]) e++;
    }
  }
  return e;
}

function isSolved(x) {
  return energy(x) === 0;
}

function nodeViolations(x, i, value = x[i]) {
  let predecessors = 0;
  let successors = 0;
  for (let j = 0; j < i; j++) if (x[j] >= value) predecessors++;
  for (let j = i + 1; j < x.length; j++) if (value >= x[j]) successors++;
  return { predecessors, successors };
}

function nodeContribution(x, i, value = x[i]) {
  const v = nodeViolations(x, i, value);
  return v.predecessors + v.successors;
}

// The same action-scoring network is reused for every candidate (i,v), at every n.
// It never receives a target label. Normalized i and v preserve the latent rule v=i
// while allowing the scorer to extrapolate to larger chains.
function actionFeatures(x, i, v) {
  const n = x.length;
  const d = Math.max(1, n - 1);
  const cur = x[i];
  const left = i > 0 ? x[i - 1] : 0;
  const right = i + 1 < n ? x[i + 1] : d;
  const viol = nodeViolations(x, i);
  return Float32Array.from([
    i / d,
    v / d,
    cur / d,
    left / d,
    right / d,
    i === 0 ? 1 : 0,
    i === n - 1 ? 1 : 0,
    viol.predecessors / Math.max(1, i),
    viol.successors / Math.max(1, n - 1 - i),
    (v - cur) / d
  ]);
}

function allActionFeatureData(x) {
  const n = x.length;
  const data = new Float32Array(n * n * FEATURE_DIM);
  let offset = 0;
  for (let i = 0; i < n; i++) {
    for (let v = 0; v < n; v++) {
      data.set(actionFeatures(x, i, v), offset);
      offset += FEATURE_DIM;
    }
  }
  return data;
}

function buildQModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [FEATURE_DIM], units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
  model.compile({ optimizer: tf.train.adam(0.0015), loss: 'meanSquaredError' });
  return model;
}

function randomAction(x, rng) {
  const n = x.length;
  const i = Math.floor(rng() * n);
  let v = Math.floor(rng() * (n - 1));
  if (v >= x[i]) v++;
  return { i, v, features: actionFeatures(x, i, v), q: NaN };
}

function greedyAction(x) {
  const n = x.length;
  const data = allActionFeatureData(x);
  const xs = tf.tensor2d(data, [n * n, FEATURE_DIM]);
  const pred = qModel.predict(xs);
  const q = pred.dataSync();

  let bestIndex = -1;
  let bestQ = -Infinity;
  for (let i = 0; i < n; i++) {
    for (let v = 0; v < n; v++) {
      if (v === x[i]) continue;
      const idx = i * n + v;
      if (q[idx] > bestQ) {
        bestQ = q[idx];
        bestIndex = idx;
      }
    }
  }

  xs.dispose();
  pred.dispose();
  const i = Math.floor(bestIndex / n);
  const v = bestIndex % n;
  return { i, v, features: actionFeatures(x, i, v), q: bestQ };
}

function epsilonGreedyAction(x, epsilon, rng) {
  return rng() < epsilon ? randomAction(x, rng) : greedyAction(x);
}

function applyAction(x, action) {
  x[action.i] = action.v;
}

function epsilonAt(episode, total) {
  const start = 0.90;
  const end = 0.04;
  const t = episode / Math.max(1, total - 1);
  return end + (start - end) * Math.pow(1 - t, 2);
}

function curriculumMaxN(episode, total, minN, maxN) {
  if (maxN <= minN) return maxN;
  const t = Math.min(1, episode / Math.max(1, total * 0.65));
  return Math.min(maxN, minN + Math.floor((maxN - minN) * t));
}

function addEpisodeToReplay(trajectory) {
  let G = 0;
  for (let t = trajectory.length - 1; t >= 0; t--) {
    G = trajectory[t].reward + GAMMA * G;
    replay.push({ features: trajectory[t].features, target: G });
  }
  if (replay.length > REPLAY_LIMIT) replay.splice(0, replay.length - REPLAY_LIMIT);
}

async function trainFromReplay(rng, batchSize = 96) {
  if (replay.length < 32) return NaN;
  const m = Math.min(batchSize, replay.length);
  const xData = new Float32Array(m * FEATURE_DIM);
  const yData = new Float32Array(m);
  for (let b = 0; b < m; b++) {
    const sample = replay[Math.floor(rng() * replay.length)];
    xData.set(sample.features, b * FEATURE_DIM);
    yData[b] = sample.target;
  }
  const xs = tf.tensor2d(xData, [m, FEATURE_DIM]);
  const ys = tf.tensor2d(yData, [m, 1]);
  try {
    const h = await qModel.fit(xs, ys, { epochs: 1, batchSize: Math.min(32, m), shuffle: true, verbose: 0 });
    return h.history.loss[0];
  } finally {
    xs.dispose();
    ys.dispose();
  }
}

async function trainRL() {
  if (!window.tf) throw new Error('TensorFlow.js did not load.');
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;

  const minN = +trainMinN.value;
  const maxN = +trainMaxN.value;
  const totalEpisodes = +episodes.value;
  trainedRange = { min: minN, max: maxN };
  replay = [];
  trained = false;

  if (qModel) qModel.dispose();
  qModel = buildQModel();

  const rng = mulberry32(1337);
  const recent = [];
  let lastLoss = NaN;

  for (let ep = 0; ep < totalEpisodes; ep++) {
    const activeMax = curriculumMaxN(ep, totalEpisodes, minN, maxN);
    const n = minN + Math.floor(rng() * (activeMax - minN + 1));
    const x = shuffledPermutation(n, rng);
    const epsilon = epsilonAt(ep, totalEpisodes);
    const maxSteps = Math.ceil(3 * n);
    const trajectory = [];
    const norm = pairCount(n);

    for (let step = 0; step < maxSteps; step++) {
      const before = energy(x);
      const action = epsilonGreedyAction(x, epsilon, rng);
      applyAction(x, action);
      const after = energy(x);
      const solved = after === 0;

      // Reward is derived only from monotonic-constraint improvement. No x[i]=i label is used.
      let reward = (before - after) / norm;
      reward -= 0.002;
      if (solved) reward += 1.0;

      trajectory.push({ features: action.features, reward });
      if (solved) break;
    }

    if (!isSolved(x) && trajectory.length) {
      trajectory[trajectory.length - 1].reward -= 0.20 * energy(x) / norm;
    }

    addEpisodeToReplay(trajectory);
    if (ep % 2 === 0) lastLoss = await trainFromReplay(rng);

    const success = isSolved(x) ? 1 : 0;
    recent.push(success);
    if (recent.length > 50) recent.shift();

    if (ep % 5 === 0 || ep === totalEpisodes - 1) {
      const successRate = recent.reduce((a, b) => a + b, 0) / Math.max(1, recent.length);
      statusEl.textContent = `Episode ${ep + 1}/${totalEpisodes} · curriculum n≤${activeMax} · sampled n=${n} · ε=${epsilon.toFixed(3)} · recent success ${(100 * successRate).toFixed(0)}% · replay ${replay.length}${Number.isFinite(lastLoss) ? ` · loss ${lastLoss.toFixed(4)}` : ''}`;
      await tf.nextFrame();
    }
  }

  trained = true;
  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
  statusEl.textContent = 'RL training complete. Running size-generalization benchmark…';
  await runBenchmark();
}

function runPolicy(start, maxMoves) {
  const x = Int32Array.from(start);
  const history = [energy(x)];
  for (let move = 0; move < maxMoves && energy(x) > 0; move++) {
    const action = greedyAction(x);
    applyAction(x, action);
    history.push(energy(x));
  }
  return { x, solved: isSolved(x), moves: history.length - 1, finalEnergy: energy(x), history };
}

function runSA(start, evaluations, rng) {
  const x = Int32Array.from(start);
  let e = energy(x);
  let best = Int32Array.from(x);
  let bestE = e;
  const T0 = 1.8;
  const Tend = 0.015;

  for (let step = 0; step < evaluations && bestE > 0; step++) {
    const i = Math.floor(rng() * x.length);
    let v = Math.floor(rng() * (x.length - 1));
    if (v >= x[i]) v++;
    const old = x[i];
    const oldContribution = nodeContribution(x, i, old);
    const newContribution = nodeContribution(x, i, v);
    const nextE = e - oldContribution + newContribution;
    const delta = nextE - e;
    const frac = step / Math.max(1, evaluations - 1);
    const temp = T0 * Math.pow(Tend / T0, frac);

    if (delta <= 0 || rng() < Math.exp(-delta / Math.max(1e-6, temp))) {
      x[i] = v;
      e = nextE;
      if (e < bestE) {
        bestE = e;
        best = Int32Array.from(x);
      }
    }
  }
  return { x: best, solved: bestE === 0, finalEnergy: bestE, evaluations };
}

function median(values) {
  if (!values.length) return NaN;
  const a = [...values].sort((a, b) => a - b);
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

function benchmarkLengths() {
  const maxTrain = trainedRange.max;
  return [...new Set([
    trainedRange.min,
    maxTrain,
    Math.min(MAX_TEST_N, Math.max(maxTrain + 1, Math.round(maxTrain * 1.5))),
    Math.min(MAX_TEST_N, Math.round(maxTrain * 2.5)),
    Math.min(MAX_TEST_N, Math.round(maxTrain * 5)),
    MAX_TEST_N
  ])].sort((a, b) => a - b);
}

async function runBenchmark() {
  if (!trained || !qModel) return;
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;

  const lengths = benchmarkLengths();
  const rows = [];
  const rng = mulberry32(20260809);
  const trials = 8;

  for (let li = 0; li < lengths.length; li++) {
    const n = lengths[li];
    statusEl.textContent = `Benchmarking n=${n} (${li + 1}/${lengths.length})…`;
    const policyMoves = [];
    let policySolved = 0;
    let saSolved = 0;
    const policyEnergy = [];
    const saEnergy = [];

    for (let t = 0; t < trials; t++) {
      const start = shuffledPermutation(n, rng);
      const policy = runPolicy(start, 2 * n);
      const sa = runSA(start, 100 * n, rng);
      if (policy.solved) {
        policySolved++;
        policyMoves.push(policy.moves);
      }
      if (sa.solved) saSolved++;
      policyEnergy.push(policy.finalEnergy);
      saEnergy.push(sa.finalEnergy);
    }

    rows.push({
      n,
      policySuccess: policySolved / trials,
      saSuccess: saSolved / trials,
      policyMoves: median(policyMoves),
      policyEnergy: median(policyEnergy),
      saEnergy: median(saEnergy)
    });
    await tf.nextFrame();
  }

  drawScaling(rows);

  const longest = rows[rows.length - 1];
  const n = longest.n;
  const start = shuffledPermutation(n, mulberry32(9001));
  const policy = runPolicy(start, 2 * n);
  const sa = runSA(start, 100 * n, mulberry32(9002));
  drawAssignment(start, policy.x, sa.x);

  $('policySuccess').textContent = `${Math.round(100 * longest.policySuccess)}%`;
  $('saSuccess').textContent = `${Math.round(100 * longest.saSuccess)}%`;
  $('policyMoves').textContent = Number.isFinite(longest.policyMoves) ? Math.round(longest.policyMoves).toString() : '—';
  $('saEffort').textContent = (100 * n).toLocaleString();
  $('metricLength').textContent = `n=${n} (trained only on n=${trainedRange.min}–${trainedRange.max})`;
  statusEl.textContent = `Done. Policy gets at most ${2 * n} joint (i,v) moves; SA gets ${(100 * n).toLocaleString()} random proposals per trial.`;

  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
}

function setupCanvas(canvas, ratio = 0.46) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(300, Math.floor(rect.width * dpr));
  canvas.height = Math.floor(Math.max(260, rect.width * ratio) * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: canvas.width / dpr, h: canvas.height / dpr };
}

function drawAxes(ctx, w, h, pad, xLabel, yLabel) {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#dbe1eb';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, 14);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - 12, h - pad);
  ctx.stroke();
  ctx.fillStyle = '#697386';
  ctx.font = '12px system-ui';
  ctx.fillText(yLabel, pad + 6, 24);
  ctx.textAlign = 'right';
  ctx.fillText(xLabel, w - 12, h - 10);
  ctx.textAlign = 'left';
}

function drawLine(ctx, points, color, width = 2.4) {
  if (!points.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  points.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.stroke();
  ctx.fillStyle = color;
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p[0], p[1], 2.8, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawScaling(rows) {
  const { ctx, w, h } = setupCanvas($('scalingChart'));
  const pad = 44;
  drawAxes(ctx, w, h, pad, 'chain length n', 'solve rate');
  const minN = Math.min(...rows.map(r => r.n));
  const maxN = Math.max(...rows.map(r => r.n));
  const X = n => pad + (n - minN) / Math.max(1, maxN - minN) * (w - pad - 20);
  const Y = p => h - pad - p * (h - pad - 28);

  ctx.fillStyle = '#7b8495';
  ctx.font = '11px system-ui';
  for (const row of rows) ctx.fillText(String(row.n), X(row.n) - 6, h - pad + 17);
  for (let k = 0; k <= 4; k++) ctx.fillText(`${k * 25}%`, 5, Y(k / 4) + 4);

  const boundaryX = X(trainedRange.max);
  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = '#a9b1c2';
  ctx.beginPath();
  ctx.moveTo(boundaryX, 18);
  ctx.lineTo(boundaryX, h - pad);
  ctx.stroke();
  ctx.restore();
  ctx.fillStyle = '#7b8495';
  ctx.fillText('training max', Math.min(w - 85, boundaryX + 5), 28);

  drawLine(ctx, rows.map(r => [X(r.n), Y(r.policySuccess)]), '#5b67d6');
  drawLine(ctx, rows.map(r => [X(r.n), Y(r.saSuccess)]), '#dd6b55');
}

function drawAssignment(start, policy, sa) {
  const { ctx, w, h } = setupCanvas($('instanceChart'));
  const pad = 44;
  drawAxes(ctx, w, h, pad, 'node id i', 'assigned value');
  const n = start.length;
  const X = i => pad + i / Math.max(1, n - 1) * (w - pad - 20);
  const Y = v => h - pad - v / Math.max(1, n - 1) * (h - pad - 28);
  const exact = Int32Array.from({ length: n }, (_, i) => i);

  drawLine(ctx, Array.from(start, (v, i) => [X(i), Y(v)]), '#9aa4b8', 1.2);
  drawLine(ctx, Array.from(exact, (v, i) => [X(i), Y(v)]), '#2e8b72', 2.6);
  drawLine(ctx, Array.from(policy, (v, i) => [X(i), Y(v)]), '#5b67d6', 2.2);
  drawLine(ctx, Array.from(sa, (v, i) => [X(i), Y(v)]), '#dd6b55', 1.8);
}

trainBtn.addEventListener('click', () => trainRL().catch(err => {
  console.error(err);
  statusEl.textContent = `Error: ${err.message}`;
  trainBtn.disabled = false;
}));

benchmarkBtn.addEventListener('click', () => runBenchmark().catch(err => {
  console.error(err);
  statusEl.textContent = `Error: ${err.message}`;
  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
}));

if (window.tf) {
  tf.ready().then(() => {
    statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Train the RL action scorer to begin.`;
  });
} else {
  statusEl.textContent = 'TensorFlow.js failed to load.';
}
