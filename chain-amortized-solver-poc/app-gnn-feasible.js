const DOMAIN_MAX = 1000;
const VALUE_COUNT = DOMAIN_MAX + 1;
const MAX_TEST_N = 100;
const NODE_FEATURE_DIM = 5;
const EMBED_DIM = 24;
const MESSAGE_ROUNDS = 6;
const EDGE_TYPES = 4;
const STEPS_PER_EPISODE = 1;
const REPLAY_LIMIT_PER_N = 700;
const BATCH_HALF = 8;
const TRAIN_CANDIDATES = 256;

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

// Generic bound propagation for x[k+1] > x[k], x[k] in 0..1000.
// The environment knows only domains, fixed assignments, and the > constraints.
function propagate(state, extraI = -1, extraV = -1) {
  const n = state.n;
  const lb = new Int32Array(n);
  const ub = new Int32Array(n);
  ub.fill(DOMAIN_MAX);

  for (let i = 0; i < n; i++) {
    if (state.assigned[i]) {
      lb[i] = state.values[i];
      ub[i] = state.values[i];
    }
  }
  if (extraI >= 0) {
    if (state.assigned[extraI] && state.values[extraI] !== extraV) return { feasible: false, lb, ub };
    lb[extraI] = extraV;
    ub[extraI] = extraV;
  }

  for (let pass = 0; pass < n + 1; pass++) {
    let changed = false;
    for (let k = 1; k < n; k++) {
      const next = Math.max(lb[k], lb[k - 1] + 1);
      if (next !== lb[k]) { lb[k] = next; changed = true; }
      if (lb[k] > ub[k]) return { feasible: false, lb, ub };
    }
    for (let k = n - 2; k >= 0; k--) {
      const next = Math.min(ub[k], ub[k + 1] - 1);
      if (next !== ub[k]) { ub[k] = next; changed = true; }
      if (lb[k] > ub[k]) return { feasible: false, lb, ub };
    }
    if (!changed) break;
  }
  return { feasible: true, lb, ub };
}

function tryAction(state, i, v, commit = false) {
  if (state.assigned[i] || v < 0 || v > DOMAIN_MAX) return false;
  const result = propagate(state, i, v);
  if (!result.feasible) return false;
  if (commit) {
    state.assigned[i] = 1;
    state.values[i] = v;
    state.assignedCount++;
  }
  return true;
}

function isStrictChain(x) {
  for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) return false;
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

function disposeLearningState() {
  if (params) for (const variable of trainableParams()) variable.dispose();
  if (optimizer?.dispose) optimizer.dispose();
  params = null;
  optimizer = null;
}

function initGNN() {
  disposeLearningState();
  paramGeneration++;
  const s = 0.12;
  params = {
    Winit: makeVariable([NODE_FEATURE_DIM, EMBED_DIM], s, 'Winit'),
    binit: makeBias(EMBED_DIM, 'binit'),
    Wself: makeVariable([EMBED_DIM, EMBED_DIM], s, 'Wself'),
    brel: makeBias(EMBED_DIM, 'brel'),
    Wrel: Array.from({ length: EDGE_TYPES }, (_, r) => makeVariable([EMBED_DIM, EMBED_DIM], s, `Wrel${r}`)),
    Wscore1: makeVariable([EMBED_DIM + 1, 48], s, 'Wscore1'),
    bscore1: makeBias(48, 'bscore1'),
    Wscore2: makeVariable([48, 24], s, 'Wscore2'),
    bscore2: makeBias(24, 'bscore2'),
    Wout: makeVariable([24, 1], s, 'Wout'),
    bout: makeBias(1, 'bout')
  };
  optimizer = tf.train.adam(0.002);
}

// Bipartite factor graph with two node types and four typed directed edges.
// G_k represents x[k+1] > x[k].
// 0 V_k -> G_k          smaller operand -> constraint
// 1 V_{k+1} -> G_k      greater operand -> constraint
// 2 G_k -> V_k          constraint -> smaller operand
// 3 G_k -> V_{k+1}      constraint -> greater operand
function graphSpec(n) {
  if (graphCache.has(n)) return graphCache.get(n);
  const constraintCount = n - 1;
  const nodeCount = n + constraintCount;
  const baseFeatures = new Float32Array(nodeCount * NODE_FEATURE_DIM);

  for (let i = 0; i < n; i++) {
    const p = i * NODE_FEATURE_DIM;
    baseFeatures[p] = 1;
    baseFeatures[p + 1] = 0;
    baseFeatures[p + 2] = n <= 1 ? 0 : i / (n - 1);
  }
  for (let k = 0; k < constraintCount; k++) {
    const node = n + k;
    const p = node * NODE_FEATURE_DIM;
    baseFeatures[p] = 0;
    baseFeatures[p + 1] = 1;
    baseFeatures[p + 2] = constraintCount <= 1 ? 0 : k / (constraintCount - 1);
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
    data[p + 3] = state.assigned[i] ? 1 : 0;
    data[p + 4] = state.assigned[i] ? state.values[i] / DOMAIN_MAX : 0;
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
    for (let r = 0; r < EDGE_TYPES; r++) {
      const transformed = tf.matMul(h, params.Wrel[r]);
      z = z.add(tf.matMul(adj[r], transformed));
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

function scoreActions(state, actions) {
  if (!actions.length) return new Float32Array(0);
  const pred = tf.tidy(() => {
    const h = graphEmbeddings(state);
    const ids = tf.tensor1d(Int32Array.from(actions, a => a.i), 'int32');
    const chosenH = tf.gather(h, ids);
    const vals = tf.tensor2d(Float32Array.from(actions, a => a.v / DOMAIN_MAX), [actions.length, 1]);
    return scoreActionTensor(tf.concat([chosenH, vals], 1));
  });
  const out = Float32Array.from(pred.dataSync());
  pred.dispose();
  return out;
}

function randomAction(state, rng) {
  const open = [];
  for (let i = 0; i < state.n; i++) if (!state.assigned[i]) open.push(i);
  const i = open[Math.floor(rng() * open.length)];
  const v = Math.floor(rng() * VALUE_COUNT);
  return { i, v };
}

function sampledGreedyAction(state, rng, sampleCount = TRAIN_CANDIDATES) {
  const actions = [];
  const seen = new Set();
  const open = [];
  for (let i = 0; i < state.n; i++) if (!state.assigned[i]) open.push(i);
  if (!open.length) return null;
  const limit = Math.min(sampleCount, open.length * VALUE_COUNT);
  while (actions.length < limit) {
    const i = open[Math.floor(rng() * open.length)];
    const v = Math.floor(rng() * VALUE_COUNT);
    const key = `${i}:${v}`;
    if (seen.has(key)) continue;
    seen.add(key);
    actions.push({ i, v });
  }
  const q = scoreActions(state, actions);
  let best = 0;
  for (let k = 1; k < q.length; k++) if (q[k] > q[best]) best = k;
  return actions[best];
}

function allActions(state) {
  const actions = [];
  for (let i = 0; i < state.n; i++) {
    if (state.assigned[i]) continue;
    for (let v = 0; v <= DOMAIN_MAX; v++) actions.push({ i, v });
  }
  return actions;
}

function epsilonAt(ep, total) {
  const t = ep / Math.max(1, total - 1);
  return 0.08 + 0.92 * Math.pow(1 - t, 2);
}

function bucket(map, n) {
  if (!map.has(n)) map.set(n, []);
  return map.get(n);
}

function remember(sample) {
  const arr = bucket(sample.reward > 0 ? positiveReplay : negativeReplay, sample.n);
  arr.push(sample);
  if (arr.length > REPLAY_LIMIT_PER_N) arr.splice(0, arr.length - REPLAY_LIMIT_PER_N);
}

function snapshotSample(state, action, reward) {
  return {
    n: state.n,
    assigned: Uint8Array.from(state.assigned),
    values: Int16Array.from(state.values),
    i: action.i,
    v: action.v,
    reward
  };
}

function stateFromSample(s) {
  let assignedCount = 0;
  for (const a of s.assigned) assignedCount += a;
  return {
    n: s.n,
    assigned: Uint8Array.from(s.assigned),
    values: Int16Array.from(s.values),
    assignedCount
  };
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

  const cost = optimizer.minimize(() => tf.tidy(() => {
    let total = tf.scalar(0);
    for (const s of samples) {
      const state = stateFromSample(s);
      const h = graphEmbeddings(state);
      const chosenH = h.slice([s.i, 0], [1, EMBED_DIM]);
      const valueTensor = tf.tensor2d([s.v / DOMAIN_MAX], [1, 1]);
      const pred = scoreActionTensor(tf.concat([chosenH, valueTensor], 1));
      const target = tf.scalar(s.reward);
      total = total.add(pred.squeeze().sub(target).square());
    }
    return total.div(samples.length);
  }), true, trainableParams());
  const loss = cost.dataSync()[0];
  cost.dispose();
  return loss;
}

function replayCounts() {
  let pos = 0, neg = 0;
  for (const arr of positiveReplay.values()) pos += arr.length;
  for (const arr of negativeReplay.values()) neg += arr.length;
  return { pos, neg };
}

function runEpisode(n, epsilon, rng, collect = true) {
  const state = emptyState(n);
  let proposals = 0;
  let accepted = 0;
  const maxProposals = Math.max(8 * n, 32);

  while (state.assignedCount < n && proposals < maxProposals) {
    let action;
    const counts = replayCounts();
    if (rng() < epsilon || counts.pos < 8 || counts.neg < 8) action = randomAction(state, rng);
    else action = sampledGreedyAction(state, rng);
    if (!action) break;

    const feasible = tryAction(state, action.i, action.v, false);
    const reward = feasible ? 1 : -1;
    if (collect) remember(snapshotSample(state, action, reward));
    proposals++;
    if (feasible) {
      tryAction(state, action.i, action.v, true);
      accepted++;
    }
  }

  return {
    solved: state.assignedCount === n && isStrictChain(state.values),
    proposals,
    accepted,
    x: Int16Array.from(state.values)
  };
}

function greedyRollout(n, maxProposals = 8 * n) {
  const state = emptyState(n);
  let proposals = 0;
  while (state.assignedCount < n && proposals < maxProposals) {
    const actions = allActions(state);
    if (!actions.length) break;
    const q = scoreActions(state, actions);
    let committed = false;
    while (!committed && proposals < maxProposals) {
      let best = -1;
      let bestQ = -Infinity;
      for (let k = 0; k < q.length; k++) {
        if (q[k] > bestQ) { bestQ = q[k]; best = k; }
      }
      if (best < 0) break;
      q[best] = -Infinity;
      proposals++;
      const a = actions[best];
      if (tryAction(state, a.i, a.v, true)) committed = true;
    }
    if (!committed) break;
  }
  return {
    solved: state.assignedCount === n && isStrictChain(state.values),
    proposals,
    x: Int16Array.from(state.values)
  };
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
  const recent = [];

  for (let ep = 0; ep < total; ep++) {
    const epsilon = epsilonAt(ep, total);
    const n = minN + Math.floor(rng() * (maxN - minN + 1));
    const episode = runEpisode(n, epsilon, rng, true);
    recent.push(episode.solved ? 1 : 0);
    if (recent.length > 50) recent.shift();

    if (ep % 2 === 0) lastLoss = await trainBalancedBatch(rng);

    if (ep % 20 === 0 || ep === total - 1) {
      const counts = replayCounts();
      const trainSuccess = recent.reduce((a, b) => a + b, 0) / Math.max(1, recent.length);
      const probe = ep > 40 ? greedyRollout(maxN, 6 * maxN) : { solved: false, proposals: 0 };
      statusEl.textContent = `Episode ${ep + 1}/${total} · ε=${epsilon.toFixed(3)} · recent exploratory solve ${(100 * trainSuccess).toFixed(0)}% · greedy n=${maxN}: ${probe.solved ? 'SOLVED' : 'not yet'}${probe.proposals ? ` in ${probe.proposals} proposals` : ''} · +${counts.pos}/−${counts.neg}${Number.isFinite(lastLoss) ? ` · loss ${lastLoss.toFixed(4)}` : ''}`;
      await tf.nextFrame();
    }
  }

  trained = true;
  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
  statusEl.textContent = 'GNN feasibility training complete. Running size-generalization benchmark…';
  await runBenchmark();
}

function violationEnergy(x) {
  let e = 0;
  for (let i = 0; i + 1 < x.length; i++) e += Math.max(0, x[i] - x[i + 1] + 1);
  return e;
}

function randomAssignment(n, rng) {
  return Int16Array.from({ length: n }, () => Math.floor(rng() * VALUE_COUNT));
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
    const v = Math.floor(rng() * VALUE_COUNT);
    x[i] = v;
    const next = violationEnergy(x);
    const delta = next - e;
    const t = step / Math.max(1, evaluations - 1);
    const temp = T0 * Math.pow(Tend / T0, t);
    if (delta <= 0 || rng() < Math.exp(-delta / Math.max(1e-6, temp))) {
      e = next;
      if (e < bestE) { bestE = e; best = Int16Array.from(x); }
    } else x[i] = old;
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
  const trials = 5;

  for (const n of benchmarkLengths()) {
    statusEl.textContent = `Benchmarking GNN at n=${n}…`;
    let policySolved = 0;
    const policyProps = [];
    let saSolved = 0;
    for (let t = 0; t < trials; t++) {
      const policy = greedyRollout(n, 8 * n);
      if (policy.solved) { policySolved++; policyProps.push(policy.proposals); }
      const sa = runSA(n, 120 * n, rng);
      if (sa.solved) saSolved++;
      await tf.nextFrame();
    }
    rows.push({
      n,
      policySuccess: policySolved / trials,
      saSuccess: saSolved / trials,
      policyProposals: policyProps.length ? policyProps.sort((a, b) => a - b)[Math.floor(policyProps.length / 2)] : NaN
    });
  }

  drawScaling(rows);
  const longest = rows[rows.length - 1];
  const n = longest.n;
  const policy = greedyRollout(n, 8 * n);
  const sa = runSA(n, 120 * n, mulberry32(9002));
  drawAssignment(policy.x, sa.x);

  $('policySuccess').textContent = `${Math.round(100 * longest.policySuccess)}%`;
  $('saSuccess').textContent = `${Math.round(100 * longest.saSuccess)}%`;
  $('policyMoves').textContent = Number.isFinite(longest.policyProposals) ? Math.round(longest.policyProposals).toString() : '—';
  $('saEffort').textContent = (120 * n).toLocaleString();
  $('metricLength').textContent = `n=${n} · domain 0…${DOMAIN_MAX} · GNN trained on n=${trainedRange.min}–${trainedRange.max}`;
  statusEl.textContent = `Done. Policy interacts only through joint (i,v) proposals and generic feasibility propagation.`;

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
    ctx.arc(p[0], p[1], 2.6, 0, Math.PI * 2);
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
  const Y = v => h - pad - Math.max(0, v) / DOMAIN_MAX * (h - pad - 28);
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
    statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Train the typed feasibility GNN to begin.`;
  });
} else {
  statusEl.textContent = 'TensorFlow.js failed to load.';
}
