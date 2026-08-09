const MAX_VALUE = 127;
const VALUE_COUNT = MAX_VALUE + 1;
const MAX_TEST_LENGTH = 120;
let model = null;
let trained = false;
let trainedRange = { min: 12, max: 32 };

const $ = id => document.getElementById(id);
const trainMinN = $('trainMinN');
const trainMaxN = $('trainMaxN');
const trainCount = $('trainCount');
const epochs = $('epochs');
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
bindRange(trainCount, 'trainCountLabel');
bindRange(epochs, 'epochsLabel');

trainMinN.addEventListener('input', () => {
  if (Number(trainMinN.value) > Number(trainMaxN.value)) {
    trainMaxN.value = trainMinN.value;
    $('trainMaxNLabel').textContent = trainMaxN.value;
  }
});
trainMaxN.addEventListener('input', () => {
  if (Number(trainMaxN.value) < Number(trainMinN.value)) {
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

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// Structured recurring instance family: a descending global trend with smooth
// motifs, a local regime shift, and noise. The raw target often violates the
// monotonic chain, so a solver must coordinate many positions.
function generateInstance(n, rng = Math.random) {
  const phase1 = rng() * Math.PI * 2;
  const phase2 = rng() * Math.PI * 2;
  const amp1 = 10 + 14 * rng();
  const amp2 = 4 + 10 * rng();
  const freq1 = 1 + Math.floor(rng() * 3);
  const freq2 = 3 + Math.floor(rng() * 5);
  const top = 112 + (rng() - 0.5) * 8;
  const bottom = 15 + (rng() - 0.5) * 8;
  const shockAt = Math.floor(n * (0.25 + 0.5 * rng()));
  const shock = (rng() - 0.5) * 24;
  const target = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    const u = n === 1 ? 0 : i / (n - 1);
    const trend = top * (1 - u) + bottom * u;
    const motif = amp1 * Math.sin(2 * Math.PI * freq1 * u + phase1)
      + amp2 * Math.sin(2 * Math.PI * freq2 * u + phase2);
    const localShock = i >= shockAt
      ? shock * Math.exp(-(i - shockAt) / Math.max(3, n * 0.18))
      : 0;
    const noise = (rng() - 0.5) * 7;
    target[i] = Math.round(clamp(trend + motif + localShock + noise, 0, MAX_VALUE));
  }
  return target;
}

function objective(x, target) {
  let sum = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - target[i];
    sum += d * d;
  }
  return sum;
}

// Exact O(n * VALUE_COUNT) dynamic program for x[i-1] > x[i].
function solveExact(target) {
  const n = target.length;
  const INF = 1e30;
  let prev = new Float64Array(VALUE_COUNT);
  let curr = new Float64Array(VALUE_COUNT);
  const parent = new Int16Array(n * VALUE_COUNT);

  for (let v = 0; v < VALUE_COUNT; v++) {
    const d = v - target[0];
    prev[v] = d * d;
    parent[v] = -1;
  }

  for (let i = 1; i < n; i++) {
    const suffixCost = new Float64Array(VALUE_COUNT + 1);
    const suffixArg = new Int16Array(VALUE_COUNT + 1);
    suffixCost[VALUE_COUNT] = INF;
    suffixArg[VALUE_COUNT] = -1;
    let best = INF;
    let arg = -1;

    for (let v = VALUE_COUNT - 1; v >= 0; v--) {
      if (prev[v] < best) {
        best = prev[v];
        arg = v;
      }
      suffixCost[v] = best;
      suffixArg[v] = arg;
    }

    curr.fill(INF);
    for (let v = 0; v < VALUE_COUNT; v++) {
      const minPrevious = v + 1;
      if (minPrevious >= VALUE_COUNT || suffixCost[minPrevious] >= INF / 2) continue;
      const d = v - target[i];
      curr[v] = d * d + suffixCost[minPrevious];
      parent[i * VALUE_COUNT + v] = suffixArg[minPrevious];
    }
    [prev, curr] = [curr, prev];
  }

  let bestCost = INF;
  let bestV = -1;
  for (let v = 0; v < VALUE_COUNT; v++) {
    if (prev[v] < bestCost) {
      bestCost = prev[v];
      bestV = v;
    }
  }

  const x = new Int32Array(n);
  x[n - 1] = bestV;
  for (let i = n - 1; i > 0; i--) x[i - 1] = parent[i * VALUE_COUNT + x[i]];
  return { x, cost: bestCost };
}

function greedyInitial(target) {
  const n = target.length;
  const x = new Int32Array(n);
  let upper = MAX_VALUE;
  for (let i = 0; i < n; i++) {
    const lower = n - 1 - i;
    x[i] = Math.round(clamp(target[i], lower, upper));
    upper = x[i] - 1;
  }
  return x;
}

// Generic SA baseline: feasible single-variable moves only.
function solveSA(target, evaluations = target.length * 180, rng = Math.random) {
  const n = target.length;
  const x = greedyInitial(target);
  let cost = objective(x, target);
  let bestCost = cost;
  let best = Int32Array.from(x);
  const T0 = 160;
  const Tend = 0.08;

  for (let step = 0; step < evaluations; step++) {
    const i = Math.floor(rng() * n);
    const lo = i === n - 1 ? 0 : x[i + 1] + 1;
    const hi = i === 0 ? MAX_VALUE : x[i - 1] - 1;
    if (lo > hi || (lo === hi && x[i] === lo)) continue;

    const old = x[i];
    let proposal;
    if (rng() < 0.18) {
      proposal = lo + Math.floor(rng() * (hi - lo + 1));
    } else {
      const span = Math.max(1, Math.min(10, hi - lo));
      const sign = rng() < 0.5 ? -1 : 1;
      proposal = clamp(old + sign * (1 + Math.floor(rng() * span)), lo, hi);
    }
    if (proposal === old) continue;

    const oldD = old - target[i];
    const newD = proposal - target[i];
    const delta = newD * newD - oldD * oldD;
    const frac = step / Math.max(1, evaluations - 1);
    const temp = T0 * Math.pow(Tend / T0, frac);

    if (delta <= 0 || rng() < Math.exp(-delta / temp)) {
      x[i] = proposal;
      cost += delta;
      if (cost < bestCost) {
        bestCost = cost;
        best = Int32Array.from(x);
      }
    }
  }
  return { x: best, cost: bestCost, evaluations };
}

// Deliberately no explicit chain-length feature. The recurrent model sees only
// the target and relative position, plus the sequence boundaries themselves.
function featureTensor(instances) {
  const batch = instances.length;
  const n = instances[0].length;
  const data = new Float32Array(batch * n * 2);
  let k = 0;
  for (const target of instances) {
    for (let i = 0; i < n; i++) {
      data[k++] = target[i] / MAX_VALUE;
      data[k++] = n === 1 ? 0 : i / (n - 1);
    }
  }
  return tf.tensor3d(data, [batch, n, 2]);
}

function buildModel() {
  const input = tf.input({ shape: [null, 2] });

  // A bidirectional GRU has global prefix/suffix context and reuses the same
  // transition at every position. Unlike the old finite-receptive-field CNN,
  // it can be evaluated on sequence lengths never seen during training.
  let z = tf.layers.bidirectional({
    layer: tf.layers.gru({
      units: 48,
      returnSequences: true,
      activation: 'tanh',
      recurrentActivation: 'sigmoid'
    }),
    mergeMode: 'concat'
  }).apply(input);

  z = tf.layers.timeDistributed({
    layer: tf.layers.dense({ units: 64, activation: 'relu' })
  }).apply(z);

  const output = tf.layers.timeDistributed({
    layer: tf.layers.dense({ units: VALUE_COUNT, activation: 'softmax' })
  }).apply(z);

  const m = tf.model({ inputs: input, outputs: output });
  m.compile({ optimizer: tf.train.adam(0.0015), loss: 'categoricalCrossentropy' });
  return m;
}

function trainingLengths(minN, maxN) {
  const values = [];
  for (let n = minN; n <= maxN; n += 4) values.push(n);
  if (values[values.length - 1] !== maxN) values.push(maxN);
  return [...new Set(values)].sort((a, b) => a - b);
}

function makeTrainingBatch(n, count, rng) {
  const instances = [];
  const labels = new Int32Array(count * n);
  for (let b = 0; b < count; b++) {
    const target = generateInstance(n, rng);
    const exact = solveExact(target);
    instances.push(target);
    labels.set(exact.x, b * n);
  }

  const xs = featureTensor(instances);
  const labelIds = tf.tensor2d(labels, [count, n], 'int32');
  const ys = tf.oneHot(labelIds, VALUE_COUNT).toFloat();
  labelIds.dispose();
  return { xs, ys };
}

async function trainPolicy() {
  if (!window.tf) throw new Error('TensorFlow.js did not load.');
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;

  const minN = Number(trainMinN.value);
  const maxN = Number(trainMaxN.value);
  const countPerEpoch = Number(trainCount.value);
  const ep = Number(epochs.value);
  const lengths = trainingLengths(minN, maxN);
  const perLength = Math.max(8, Math.ceil(countPerEpoch / lengths.length));

  trainedRange = { min: minN, max: maxN };
  if (model) model.dispose();
  model = buildModel();

  console.log('model output', model.outputs[0].shape, 'training lengths', lengths, 'backend', tf.getBackend());
  const rng = mulberry32(1337);

  for (let epoch = 0; epoch < ep; epoch++) {
    let lossSum = 0;
    let buckets = 0;

    // Rotate the length order each epoch to avoid always ending on the longest.
    const order = lengths.map((_, i) => lengths[(i + epoch) % lengths.length]);
    for (let li = 0; li < order.length; li++) {
      const n = order[li];
      statusEl.textContent = `Epoch ${epoch + 1}/${ep} · training n=${n} · range ${minN}–${maxN}`;
      const { xs, ys } = makeTrainingBatch(n, perLength, rng);
      try {
        const history = await model.fit(xs, ys, {
          epochs: 1,
          batchSize: Math.min(32, perLength),
          shuffle: true,
          verbose: 0
        });
        lossSum += history.history.loss[0];
        buckets++;
      } finally {
        xs.dispose();
        ys.dispose();
      }
      await tf.nextFrame();
    }

    statusEl.textContent = `Epoch ${epoch + 1}/${ep} · mean loss ${(lossSum / Math.max(1, buckets)).toFixed(4)} · lengths ${minN}–${maxN}`;
    await tf.nextFrame();
  }

  trained = true;
  benchmarkBtn.disabled = false;
  trainBtn.disabled = false;
  statusEl.textContent = 'Training complete. Running in-range + extrapolation benchmark…';
  await runBenchmark();
}

function solvePolicy(target) {
  if (!model) throw new Error('Policy is not trained.');
  const xs = featureTensor([target]);
  const pred = model.predict(xs);
  const probs = pred.dataSync();
  const n = target.length;
  const x = new Int32Array(n);
  let previous = VALUE_COUNT;

  // Tiny feasibility-aware decoder. It only prevents impossible completions;
  // the neural model still supplies every value preference.
  for (let i = 0; i < n; i++) {
    const lo = n - 1 - i;
    const hi = Math.min(MAX_VALUE, previous - 1);
    let bestV = lo;
    let bestP = -1;
    const offset = i * VALUE_COUNT;
    for (let v = lo; v <= hi; v++) {
      const p = probs[offset + v];
      if (p > bestP) {
        bestP = p;
        bestV = v;
      }
    }
    x[i] = bestV;
    previous = bestV;
  }

  xs.dispose();
  pred.dispose();
  return { x, cost: objective(x, target), evaluations: 1 };
}

function median(values) {
  const a = [...values].sort((x, y) => x - y);
  const mid = Math.floor(a.length / 2);
  return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
}

function fmt(x) {
  if (!Number.isFinite(x)) return '—';
  if (x >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return x.toFixed(x < 10 ? 2 : 1);
}

function benchmarkLengths(minN, maxN) {
  const mid = Math.round((minN + maxN) / 2);
  const candidates = [
    minN,
    mid,
    maxN,
    Math.round(maxN * 1.5),
    maxN * 2,
    maxN * 3
  ];
  return [...new Set(candidates)]
    .map(n => Math.min(MAX_TEST_LENGTH, Math.max(4, Math.round(n))))
    .filter(n => n <= MAX_VALUE + 1)
    .sort((a, b) => a - b);
}

async function runBenchmark() {
  if (!trained) return;
  benchmarkBtn.disabled = true;
  trainBtn.disabled = true;

  const { min: minN, max: maxN } = trainedRange;
  const lengths = benchmarkLengths(minN, maxN);
  const rng = mulberry32(20260809);
  const rows = [];

  for (let li = 0; li < lengths.length; li++) {
    const n = lengths[li];
    const regime = n <= maxN && n >= minN ? 'in-range' : 'OOD';
    statusEl.textContent = `Benchmarking n=${n} · ${regime} (${li + 1}/${lengths.length})…`;
    const policyExcess = [];
    const saExcess = [];

    for (let r = 0; r < 8; r++) {
      const target = generateInstance(n, rng);
      const exact = solveExact(target);
      const policy = solvePolicy(target);
      const sa = solveSA(target, n * 180, rng);
      policyExcess.push((policy.cost - exact.cost) / n);
      saExcess.push((sa.cost - exact.cost) / n);
    }

    rows.push({
      n,
      policy: median(policyExcess),
      sa: median(saExcess),
      inRange: n >= minN && n <= maxN
    });
    await tf.nextFrame();
  }

  drawScaling(rows, maxN);

  const oodRows = rows.filter(r => !r.inRange);
  const focusRow = oodRows.length ? oodRows[oodRows.length - 1] : rows[rows.length - 1];
  const focusN = focusRow.n;
  const target = generateInstance(focusN, mulberry32(9001));
  const exact = solveExact(target);
  const policy = solvePolicy(target);
  const sa = solveSA(target, focusN * 180, mulberry32(9002));
  drawInstance(target, exact.x, policy.x, sa.x);

  $('policyGap').textContent = fmt(focusRow.policy);
  $('saGap').textContent = fmt(focusRow.sa);
  $('policyEffort').textContent = '1';
  $('saEffort').textContent = (focusN * 180).toLocaleString();
  $('metricLength').textContent = `n=${focusN} (${focusRow.inRange ? 'in-range' : 'OOD'})`;
  statusEl.textContent = `Done. Training lengths ${minN}–${maxN}; headline metrics are at n=${focusN}. Lower gap is better.`;

  benchmarkBtn.disabled = false;
  trainBtn.disabled = false;
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

function drawScaling(rows, trainMax) {
  const { ctx, w, h } = setupCanvas($('scalingChart'));
  const pad = 42;
  drawAxes(ctx, w, h, pad, 'chain length n', 'excess cost / variable');

  const maxY = Math.max(1, ...rows.flatMap(r => [r.policy, r.sa])) * 1.12;
  const minN = Math.min(...rows.map(r => r.n));
  const maxN = Math.max(...rows.map(r => r.n));
  const X = n => pad + (n - minN) / Math.max(1, maxN - minN) * (w - pad - 20);
  const Y = y => h - pad - y / maxY * (h - pad - 28);

  if (trainMax >= minN && trainMax <= maxN) {
    const x = X(trainMax);
    ctx.save();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = '#7b8495';
    ctx.beginPath();
    ctx.moveTo(x, 16);
    ctx.lineTo(x, h - pad);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#697386';
    ctx.font = '11px system-ui';
    ctx.fillText('training max', Math.min(x + 5, w - 80), 28);
    ctx.restore();
  }

  ctx.fillStyle = '#7b8495';
  ctx.font = '11px system-ui';
  for (const row of rows) ctx.fillText(String(row.n), X(row.n) - 6, h - pad + 17);
  for (let k = 0; k <= 4; k++) {
    const yv = maxY * k / 4;
    ctx.fillText(fmt(yv), 4, Y(yv) + 4);
  }

  drawLine(ctx, rows.map(r => [X(r.n), Y(r.policy)]), '#5b67d6');
  drawLine(ctx, rows.map(r => [X(r.n), Y(r.sa)]), '#dd6b55');
}

function drawInstance(target, exact, policy, sa) {
  const { ctx, w, h } = setupCanvas($('instanceChart'));
  const pad = 42;
  drawAxes(ctx, w, h, pad, 'position', 'value');

  const n = target.length;
  const X = i => pad + i / Math.max(1, n - 1) * (w - pad - 20);
  const Y = v => h - pad - v / MAX_VALUE * (h - pad - 28);

  drawLine(ctx, Array.from(target, (v, i) => [X(i), Y(v)]), '#9aa4b8', 1.5);
  drawLine(ctx, Array.from(exact, (v, i) => [X(i), Y(v)]), '#2e8b72', 2.5);
  drawLine(ctx, Array.from(policy, (v, i) => [X(i), Y(v)]), '#5b67d6', 2.2);
  drawLine(ctx, Array.from(sa, (v, i) => [X(i), Y(v)]), '#dd6b55', 2.0);
}

trainBtn.addEventListener('click', () => trainPolicy().catch(err => {
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
    statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Train the multi-length policy to begin.`;
  });
} else {
  statusEl.textContent = 'TensorFlow.js failed to load.';
}
