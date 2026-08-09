const MAX_TEST_N = 100;
const NODE_FEATURE_DIM = 3;
const EMBED_DIM = 16;
const MESSAGE_ROUNDS = 4;
const EDGE_TYPES = 4;
const STEPS_PER_EPISODE = 8;
const REPLAY_LIMIT_PER_N = 1000;
const BATCH_HALF = 16;

let params = null;
let optimizer = null;
let trained = false;
let trainedRange = { min: 4, max: 20 };
let positiveReplay = new Map();
let negativeReplay = new Map();
let graphCache = new Map();
let paramGeneration = 0;

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

// Generic bound propagation for the factor graph constraints
// G_k: x[k+1] > x[k], with x[k] in 0..n-1, after fixing x[i]=v.
// There is intentionally no x[i]=i rule in the environment.
function fixIsFeasible(n, i, v) {
  const lb = new Int32Array(n);
  const ub = new Int32Array(n);
  ub.fill(n - 1);
  lb[i] = v;
  ub[i] = v;

  for (let pass = 0; pass < n; pass++) {
    let changed = false;
    for (let k = 1; k < n; k++) {
      const next = Math.max(lb[k], lb[k - 1] + 1);
      if (next !== lb[k]) { lb[k] = next; changed = true; }
      if (lb[k] > ub[k]) return false;
    }
    for (let k = n - 2; k >= 0; k--) {
      const next = Math.min(ub[k], ub[k + 1] - 1);
      if (next !== ub[k]) { ub[k] = next; changed = true; }
      if (lb[k] > ub[k]) return false;
    }
    if (!changed) break;
  }
  for (let k = 0; k < n; k++) if (lb[k] > ub[k]) return false;
  return true;
}

function makeVariable(shape, scale, name) {
  const init = tf.randomNormal(shape, 0, scale);
  const variable = tf.variable(init, true, `${name}_${paramGeneration}`);
  init.dispose();
  return variable;
}

function makeBias(size, name) {
  const init = tf.zeros([size]);
  const variable = tf.variable(init, true, `${name}_${paramGeneration}`);
  init.dispose();
  return variable;
}

function disposeLearningState() {
  if (params) {
    for (const variable of trainableParams()) variable.dispose();
  }
  if (optimizer?.dispose) optimizer.dispose();
  params = null;
  optimizer = null;
}

function initGNN() {
  disposeLearningState();
  paramGeneration++;
  const s = 0.16;
  params = {
    Winit: makeVariable([NODE_FEATURE_DIM, EMBED_DIM], s, 'Winit'),
    binit: makeBias(EMBED_DIM, 'binit'),
    Wself: makeVariable([EMBED_DIM, EMBED_DIM], s, 'Wself'),
    brel: makeBias(EMBED_DIM, 'brel'),
    Wrel: Array.from({ length: EDGE_TYPES }, (_, r) => makeVariable([EMBED_DIM, EMBED_DIM], s, `Wrel${r}`)),
    Wscore1: makeVariable([EMBED_DIM + 1, 32], s, 'Wscore1'),
    bscore1: makeBias(32, 'bscore1'),
    Wscore2: makeVariable([32, 16], s, 'Wscore2'),
    bscore2: makeBias(16, 'bscore2'),
    Wout: makeVariable([16, 1], s, 'Wout'),
    bout: makeBias(1, 'bout')
  };
  optimizer = tf.train.adam(0.003);
}

function trainableParams() {
  if (!params) return [];
  return [
    params.Winit, params.binit, params.Wself, params.brel,
    ...params.Wrel,
    params.Wscore1, params.bscore1,
    params.Wscore2, params.bscore2,
    params.Wout, params.bout
  ];
}

// Bipartite factor graph:
// variable nodes V_i, i=0..n-1
// greater-than constraint nodes G_k, k=0..n-2, representing x[k+1] > x[k].
// Four directed edge types preserve operand role:
// 0: V_k -> G_k        (smaller operand -> constraint)
// 1: V_{k+1} -> G_k    (greater operand -> constraint)
// 2: G_k -> V_k        (constraint -> smaller operand)
// 3: G_k -> V_{k+1}    (constraint -> greater operand)
function graphSpec(n) {
  if (graphCache.has(n)) return graphCache.get(n);
  const constraintCount = Math.max(0, n - 1);
  const nodeCount = n + constraintCount;
  const nodeFeatures = new Float32Array(nodeCount * NODE_FEATURE_DIM);

  for (let i = 0; i < n; i++) {
    const p = i * NODE_FEATURE_DIM;
    nodeFeatures[p] = 1;
    nodeFeatures[p + 1] = 0;
    nodeFeatures[p + 2] = n <= 1 ? 0 : i / (n - 1);
  }
  for (let k = 0; k < constraintCount; k++) {
    const node = n + k;
    const p = node * NODE_FEATURE_DIM;
    nodeFeatures[p] = 0;
    nodeFeatures[p + 1] = 1;
    nodeFeatures[p + 2] = constraintCount <= 1 ? 0 : k / (constraintCount - 1);
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

  const spec = { n, nodeCount, nodeFeatures, adjs };
  graphCache.set(n, spec);
  return spec;
}

function graphEmbeddings(n) {
  const spec = graphSpec(n);
  const nodeX = tf.tensor2d(spec.nodeFeatures, [spec.nodeCount, NODE_FEATURE_DIM]);
  const adj = spec.adjs.map(a => tf.tensor2d(a, [spec.nodeCount, spec.nodeCount]));

  let h = tf.relu(tf.matMul(nodeX, params.Winit).add(params.binit));
  for (let round = 0; round < MESSAGE_ROUNDS; round++) {
    let z = tf.matMul(h, params.Wself);
    for (let r = 0; r < EDGE_TYPES; r++) {
      const transformed = tf.matMul(h, params.Wrel[r]);
      const message = tf.matMul(adj[r], transformed);
      z = z.add(message);
    }
    h = tf.relu(h.add(z.add(params.brel)));
  }
  return h;
}

function scoreActionTensor(actionX) {
  let z = tf.relu(tf.matMul(actionX, params.Wscore1).add(params.bscore1));
  z = tf.relu(tf.matMul(z, params.Wscore2).add(params.bscore2));
  return tf.matMul(z, params.Wout).add(params.bout);
}

function scoreAllTensor(n) {
  const h = graphEmbeddings(n);
  const variableH = h.slice([0, 0], [n, EMBED_DIM]);
  const repeated = tf.tile(variableH.expandDims(1), [1, n, 1]).reshape([n * n, EMBED_DIM]);
  const valueData = new Float32Array(n * n);
  const d = Math.max(1, n - 1);
  let p = 0;
  for (let i = 0; i < n; i++) for (let v = 0; v < n; v++) valueData[p++] = v / d;
  const values = tf.tensor2d(valueData, [n * n, 1]);
  return scoreActionTensor(tf.concat([repeated, values], 1));
}

function scoreAll(n) {
  const pred = tf.tidy(() => scoreAllTensor(n));
  const values = Float32Array.from(pred.dataSync());
  pred.dispose();
  return values;
}

function greedyBanditAction(n) {
  const q = scoreAll(n);
  let best = 0;
  for (let k = 1; k < q.length; k++) if (q[k] > q[best]) best = k;
  return { i: Math.floor(best / n), v: best % n, q: q[best] };
}

function epsilonAt(ep, total) {
  const t = ep / Math.max(1, total - 1);
  return 0.05 + 0.95 * Math.pow(1 - t, 2);
}

function bucket(map, n) {
  if (!map.has(n)) map.set(n, []);
  return map.get(n);
}

function remember(n, i, v, reward) {
  const arr = bucket(reward > 0 ? positiveReplay : negativeReplay, n);
  arr.push({ n, i, v, reward });
  if (arr.length > REPLAY_LIMIT_PER_N) arr.splice(0, arr.length - REPLAY_LIMIT_PER_N);
}

function eligibleTrainingLengths() {
  const out = [];
  for (const [n, pos] of positiveReplay.entries()) {
    const neg = negativeReplay.get(n);
    if (pos.length >= 4 && neg && neg.length >= 4) out.push(n);
  }
  return out;
}

async function trainBalancedBatch(rng) {
  const eligible = eligibleTrainingLengths();
  if (!eligible.length) return NaN;
  const n = eligible[Math.floor(rng() * eligible.length)];
  const pos = positiveReplay.get(n);
  const neg = negativeReplay.get(n);
  const half = Math.min(BATCH_HALF, pos.length, neg.length);
  const samples = [];
  for (let b = 0; b < half; b++) {
    samples.push(pos[Math.floor(rng() * pos.length)]);
    samples.push(neg[Math.floor(rng() * neg.length)]);
  }

  const ids = Int32Array.from(samples, s => s.i);
  const vals = Float32Array.from(samples, s => s.v / Math.max(1, n - 1));
  const labels = Float32Array.from(samples, s => s.reward);

  const cost = optimizer.minimize(() => tf.tidy(() => {
    const h = graphEmbeddings(n);
    const idTensor = tf.tensor1d(ids, 'int32');
    const chosenH = tf.gather(h, idTensor);
    const valueTensor = tf.tensor2d(vals, [samples.length, 1]);
    const actionX = tf.concat([chosenH, valueTensor], 1);
    const pred = scoreActionTensor(actionX);
    const y = tf.tensor2d(labels, [samples.length, 1]);
    return pred.sub(y).square().mean();
  }), true, trainableParams());

  const loss = cost.dataSync()[0];
  cost.dispose();
  return loss;
}

function ruleAccuracy(n) {
  const q = scoreAll(n);
  let correct = 0;
  for (let i = 0; i < n; i++) {
    let bestV = 0;
    for (let v = 1; v < n; v++) if (q[i * n + v] > q[i * n + bestV]) bestV = v;
    if (bestV === i) correct++;
  }
  return correct / n;
}

function replayCounts() {
  let pos = 0, neg = 0;
  for (const arr of positiveReplay.values()) pos += arr.length;
  for (const arr of negativeReplay.values()) neg += arr.length;
  return { pos, neg };
}

async function trainRL() {
  if (!window.tf) throw new Error('TensorFlow.js did not load.');
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;
  const minN = +trainMinN.value;
  const maxN = +trainMaxN.value;
  const total = +episodes.value;
  trainedRange = { min: minN, max: maxN };
  positiveReplay = new Map();
  negativeReplay = new Map();
  trained = false;
  initGNN();

  const rng = mulberry32(1337);
  let lastLoss = NaN;
  let accuracy = 0;

  for (let ep = 0; ep < total; ep++) {
    const epsilon = epsilonAt(ep, total);
    for (let s = 0; s < STEPS_PER_EPISODE; s++) {
      const n = minN + Math.floor(rng() * (maxN - minN + 1));
      let action;
      const counts = replayCounts();
      if (rng() < epsilon || counts.pos < 8) {
        action = { i: Math.floor(rng() * n), v: Math.floor(rng() * n) };
      } else {
        action = greedyBanditAction(n);
      }
      const reward = fixIsFeasible(n, action.i, action.v) ? 1 : -1;
      remember(n, action.i, action.v, reward);
    }

    if (ep % 2 === 0) lastLoss = await trainBalancedBatch(rng);

    if (ep % 20 === 0 || ep === total - 1) {
      accuracy = ruleAccuracy(maxN);
      const counts = replayCounts();
      statusEl.textContent = `Episode ${ep + 1}/${total} · ε=${epsilon.toFixed(3)} · GNN rule accuracy ${(100 * accuracy).toFixed(0)}% at n=${maxN} · +${counts.pos}/−${counts.neg}${Number.isFinite(lastLoss) ? ` · loss ${lastLoss.toFixed(4)}` : ''}`;
      await tf.nextFrame();
    }
  }

  trained = true;
  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
  statusEl.textContent = `GNN training complete · rule accuracy ${(100 * accuracy).toFixed(0)}% at n=${maxN}. Running benchmark…`;
  await runBenchmark();
}

function constructPolicy(n) {
  const q = scoreAll(n);
  const x = new Int32Array(n);
  x.fill(-1);
  const unusedI = new Set(Array.from({ length: n }, (_, i) => i));
  const unusedV = new Set(Array.from({ length: n }, (_, i) => i));
  for (let move = 0; move < n; move++) {
    let bestI = -1, bestV = -1, bestQ = -Infinity;
    for (const i of unusedI) for (const v of unusedV) {
      const value = q[i * n + v];
      if (value > bestQ) { bestQ = value; bestI = i; bestV = v; }
    }
    x[bestI] = bestV;
    unusedI.delete(bestI);
    unusedV.delete(bestV);
  }
  return { x, solved: isStrictChain(x), moves: n };
}

function isStrictChain(x) {
  for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) return false;
  return true;
}

function gapEnergy(x) {
  let e = 0;
  for (let i = 0; i < x.length - 1; i++) for (let j = i + 1; j < x.length; j++) {
    e += Math.max(0, (j - i) - (x[j] - x[i]));
  }
  return e;
}

function shuffledPermutation(n, rng) {
  const x = Int32Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [x[i], x[j]] = [x[j], x[i]];
  }
  return x;
}

function runSA(n, evaluations, rng) {
  const x = shuffledPermutation(n, rng);
  let e = gapEnergy(x), bestE = e, best = Int32Array.from(x);
  const T0 = Math.max(1, n / 2), Tend = 0.01;
  for (let step = 0; step < evaluations && bestE > 0; step++) {
    const a = Math.floor(rng() * n), b = Math.floor(rng() * n);
    if (a === b) continue;
    [x[a], x[b]] = [x[b], x[a]];
    const next = gapEnergy(x), delta = next - e;
    const t = step / Math.max(1, evaluations - 1);
    const temp = T0 * Math.pow(Tend / T0, t);
    if (delta <= 0 || rng() < Math.exp(-delta / temp)) {
      e = next;
      if (e < bestE) { bestE = e; best = Int32Array.from(x); }
    } else {
      [x[a], x[b]] = [x[b], x[a]];
    }
  }
  return { x: best, solved: bestE === 0, finalEnergy: bestE, evaluations };
}

function benchmarkLengths() {
  const m = trainedRange.max;
  return [...new Set([
    trainedRange.min,
    m,
    Math.min(MAX_TEST_N, Math.round(m * 1.5)),
    Math.min(MAX_TEST_N, m * 2),
    Math.min(MAX_TEST_N, m * 3),
    MAX_TEST_N
  ])].sort((a, b) => a - b);
}

async function runBenchmark() {
  if (!trained || !params) return;
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;
  const rows = [];
  const rng = mulberry32(20260809);
  const trials = 8;

  for (const n of benchmarkLengths()) {
    statusEl.textContent = `Benchmarking GNN at n=${n}…`;
    const policy = constructPolicy(n);
    let saSolved = 0;
    for (let t = 0; t < trials; t++) if (runSA(n, 100 * n, rng).solved) saSolved++;
    rows.push({ n, policySuccess: policy.solved ? 1 : 0, saSuccess: saSolved / trials });
    await tf.nextFrame();
  }

  drawScaling(rows);
  const longest = rows[rows.length - 1];
  const n = longest.n;
  const policy = constructPolicy(n);
  const sa = runSA(n, 100 * n, mulberry32(9002));
  drawAssignment(policy.x, sa.x);

  $('policySuccess').textContent = `${Math.round(100 * longest.policySuccess)}%`;
  $('saSuccess').textContent = `${Math.round(100 * longest.saSuccess)}%`;
  $('policyMoves').textContent = n.toString();
  $('saEffort').textContent = (100 * n).toLocaleString();
  $('metricLength').textContent = `n=${n} (GNN trained on n=${trainedRange.min}–${trainedRange.max})`;
  statusEl.textContent = `Done. The GNN scorer constructs a complete assignment in exactly ${n} joint (i,v) choices.`;

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
  ctx.moveTo(pad, 14);
  ctx.lineTo(pad, h - pad);
  ctx.lineTo(w - 12, h - pad);
  ctx.stroke();
  ctx.fillStyle = '#697386';
  ctx.font = '12px system-ui';
  ctx.fillText(yl, pad + 6, 24);
  ctx.textAlign = 'right';
  ctx.fillText(xl, w - 12, h - 10);
  ctx.textAlign = 'left';
}

function line(ctx, pts, color, width = 2.4) {
  if (!pts.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.stroke();
  ctx.fillStyle = color;
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p[0], p[1], 2.8, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawScaling(rows) {
  const { ctx, w, h } = setupCanvas($('scalingChart'));
  const pad = 44;
  axes(ctx, w, h, pad, 'chain length n', 'solve rate');
  const lo = Math.min(...rows.map(r => r.n));
  const hi = Math.max(...rows.map(r => r.n));
  const X = n => pad + (n - lo) / Math.max(1, hi - lo) * (w - pad - 20);
  const Y = p => h - pad - p * (h - pad - 28);
  ctx.fillStyle = '#7b8495';
  ctx.font = '11px system-ui';
  for (const r of rows) ctx.fillText(String(r.n), X(r.n) - 6, h - pad + 17);
  for (let k = 0; k <= 4; k++) ctx.fillText(`${25 * k}%`, 5, Y(k / 4) + 4);
  const bx = X(trainedRange.max);
  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = '#a9b1c2';
  ctx.beginPath();
  ctx.moveTo(bx, 18);
  ctx.lineTo(bx, h - pad);
  ctx.stroke();
  ctx.restore();
  ctx.fillText('training max', Math.min(w - 85, bx + 5), 28);
  line(ctx, rows.map(r => [X(r.n), Y(r.policySuccess)]), '#5b67d6');
  line(ctx, rows.map(r => [X(r.n), Y(r.saSuccess)]), '#dd6b55');
}

function drawAssignment(policy, sa) {
  const { ctx, w, h } = setupCanvas($('instanceChart'));
  const pad = 44;
  axes(ctx, w, h, pad, 'variable node id i', 'assigned value');
  const n = policy.length;
  const X = i => pad + i / Math.max(1, n - 1) * (w - pad - 20);
  const Y = v => h - pad - v / Math.max(1, n - 1) * (h - pad - 28);
  const exact = Int32Array.from({ length: n }, (_, i) => i);
  line(ctx, Array.from(exact, (v, i) => [X(i), Y(v)]), '#2e8b72', 2.6);
  line(ctx, Array.from(policy, (v, i) => [X(i), Y(v)]), '#5b67d6', 2.2);
  line(ctx, Array.from(sa, (v, i) => [X(i), Y(v)]), '#dd6b55', 1.8);
}

trainBtn.addEventListener('click', () => trainRL().catch(err => {
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

if (window.tf) {
  tf.ready().then(() => {
    statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Train the typed factor-graph GNN to begin.`;
  });
} else {
  statusEl.textContent = 'TensorFlow.js failed to load.';
}
