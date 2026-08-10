const DOMAIN_MAX = 1000;
const VALUE_COUNT = DOMAIN_MAX + 1;
const MAX_TEST_N = 100;
const NODE_FEATURE_DIM = 5;
const EMBED_DIM = 24;
const MESSAGE_ROUNDS = 6;
const EDGE_TYPES = 4;
const MOVE_MULTIPLIER = 3;
const ENTROPY_BETA = 0.0015;
const PROMOTE_WINDOW = 30;
const PROMOTE_RATE = 0.70;

let params = null;
let optimizer = null;
let trained = false;
let trainedRange = { min: 2, max: 20 };
let graphCache = new Map();
let paramGeneration = 0;
let baselines = new Map();

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
  const s = 0.10;
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
  optimizer = tf.train.adam(0.0015);
}

// Bipartite factor graph:
// V_i = variable x_i
// G_k = constraint x[k+1] > x[k]
// Four directed edge types preserve smaller/greater operand roles both ways.
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

// Until every variable has been assigned once, actions are restricted to unassigned
// variables so the first n moves always create a complete assignment. After that,
// every variable is selectable again, allowing up to 2n further moves to repair it.
function allActions(state) {
  const actions = [];
  const correctionPhase = state.assignedCount === state.n;
  for (let i = 0; i < state.n; i++) {
    if (!correctionPhase && state.assigned[i]) continue;
    for (let v = 0; v <= DOMAIN_MAX; v++) actions.push({ i, v });
  }
  return actions;
}

function actionLogitsTensor(state, actions) {
  const h = graphEmbeddings(state);
  const ids = tf.tensor1d(Int32Array.from(actions, a => a.i), 'int32');
  const chosenH = tf.gather(h, ids);
  const vals = tf.tensor2d(Float32Array.from(actions, a => a.v / DOMAIN_MAX), [actions.length, 1]);
  return scoreActionTensor(tf.concat([chosenH, vals], 1)).reshape([actions.length]);
}

function scoreActions(state, actions) {
  const logits = tf.tidy(() => actionLogitsTensor(state, actions));
  const out = Float32Array.from(logits.dataSync());
  logits.dispose();
  return out;
}

function sampleIndex(logits, temperature, rng) {
  let max = -Infinity;
  for (const x of logits) if (x > max) max = x;
  const weights = new Float64Array(logits.length);
  let total = 0;
  const invT = 1 / Math.max(0.15, temperature);
  for (let i = 0; i < logits.length; i++) {
    const w = Math.exp((logits[i] - max) * invT);
    weights[i] = w;
    total += w;
  }
  let r = rng() * total;
  for (let i = 0; i < weights.length; i++) {
    r -= weights[i];
    if (r <= 0) return i;
  }
  return weights.length - 1;
}

function temperatureAt(stageEpisodes) {
  const t = Math.min(1, stageEpisodes / 160);
  return 1.35 - 0.55 * t;
}

function rollout(n, rng, stochastic = true, temperature = 1.0) {
  const state = emptyState(n);
  const trajectory = [];
  const maxMoves = MOVE_MULTIPLIER * n;
  let moves = 0;
  let solved = false;

  for (let move = 0; move < maxMoves; move++) {
    const actions = allActions(state);
    const logits = scoreActions(state, actions);
    let chosen = 0;
    if (stochastic) chosen = sampleIndex(logits, temperature, rng);
    else for (let k = 1; k < logits.length; k++) if (logits[k] > logits[chosen]) chosen = k;

    const action = actions[chosen];
    if (stochastic) trajectory.push({ state: cloneState(state), i: action.i, v: action.v });

    if (!state.assigned[action.i]) {
      state.assigned[action.i] = 1;
      state.assignedCount++;
    }
    state.values[action.i] = action.v;
    moves = move + 1;

    // A feasible full assignment is terminal. There is no intermediate reward:
    // this check only determines whether the episode can end with its +1 terminal return.
    if (state.assignedCount === n && isStrictChain(state.values)) {
      solved = true;
      break;
    }
  }

  return {
    solved,
    x: Int16Array.from(state.values),
    trajectory,
    moves
  };
}

function baselineFor(n) {
  return baselines.has(n) ? baselines.get(n) : 0;
}

function updateBaseline(n, reward) {
  const old = baselineFor(n);
  baselines.set(n, 0.9 * old + 0.1 * reward);
  return old;
}

async function reinforceEpisode(n, trajectory, reward, temperature) {
  if (!trajectory.length) return NaN;
  const baseline = updateBaseline(n, reward);
  const advantage = reward - baseline;

  const cost = optimizer.minimize(() => tf.tidy(() => {
    let logProbSum = tf.scalar(0);
    let entropySum = tf.scalar(0);

    for (const step of trajectory) {
      const actions = allActions(step.state);
      const logits = actionLogitsTensor(step.state, actions).div(Math.max(0.15, temperature));
      const logProbs = tf.logSoftmax(logits);
      let chosenIndex = -1;
      for (let k = 0; k < actions.length; k++) {
        const a = actions[k];
        if (a.i === step.i && a.v === step.v) { chosenIndex = k; break; }
      }
      const chosenLogProb = logProbs.gather(chosenIndex);
      const probs = tf.softmax(logits);
      const entropy = probs.mul(logProbs).sum().neg();
      logProbSum = logProbSum.add(chosenLogProb);
      entropySum = entropySum.add(entropy);
    }

    const scale = 1 / trajectory.length;
    const policyLoss = logProbSum.mul(-advantage * scale);
    const entropyBonus = entropySum.mul(-ENTROPY_BETA * scale);
    return policyLoss.add(entropyBonus);
  }), true, trainableParams());

  const loss = cost.dataSync()[0];
  cost.dispose();
  return loss;
}

async function trainRL() {
  if (!window.tf) throw new Error('TensorFlow.js did not load.');
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;

  const startN = +trainMinN.value;
  const maxN = +trainMaxN.value;
  const total = +episodes.value;
  trainedRange = { min: startN, max: maxN };
  trained = false;
  baselines = new Map();
  initGNN();

  const rng = mulberry32(1337);
  let currentN = startN;
  let stageEpisodes = 0;
  let lastLoss = NaN;
  const recent = [];
  let promotions = 0;

  for (let ep = 0; ep < total; ep++) {
    const temp = temperatureAt(stageEpisodes);
    const episode = rollout(currentN, rng, true, temp);
    const reward = episode.solved ? 1 : -1;
    lastLoss = await reinforceEpisode(currentN, episode.trajectory, reward, temp);

    recent.push(episode.solved ? 1 : 0);
    if (recent.length > PROMOTE_WINDOW) recent.shift();
    stageEpisodes++;

    let greedySolved = false;
    if (ep % 10 === 0 || ep === total - 1) greedySolved = rollout(currentN, rng, false).solved;

    const rate = recent.reduce((a, b) => a + b, 0) / Math.max(1, recent.length);
    const readyToPromote = currentN < maxN
      && recent.length >= PROMOTE_WINDOW
      && rate >= PROMOTE_RATE
      && greedySolved;

    if (readyToPromote) {
      currentN++;
      promotions++;
      stageEpisodes = 0;
      recent.length = 0;
    }

    if (ep % 10 === 0 || ep === total - 1 || readyToPromote) {
      statusEl.textContent = `Episode ${ep + 1}/${total} · curriculum n=${currentN}${currentN < maxN ? `/${maxN}` : ' (max)'} · terminal success ${(100 * rate).toFixed(0)}% · greedy ${greedySolved ? 'SOLVED' : 'not yet'} · budget ≤${MOVE_MULTIPLIER}n · T=${temp.toFixed(2)} · promotions ${promotions}${Number.isFinite(lastLoss) ? ` · loss ${lastLoss.toFixed(4)}` : ''}`;
      await tf.nextFrame();
    }
  }

  trained = true;
  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
  statusEl.textContent = `Training complete · curriculum reached n=${currentN}. Running benchmark…`;
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
    x[i] = Math.floor(rng() * VALUE_COUNT);
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
    statusEl.textContent = `Benchmarking ≤${MOVE_MULTIPLIER}n-move terminal GNN at n=${n}…`;
    let policySolved = 0;
    let saSolved = 0;
    const policyMoves = [];
    for (let t = 0; t < trials; t++) {
      const policy = rollout(n, rng, false);
      if (policy.solved) {
        policySolved++;
        policyMoves.push(policy.moves);
      }
      const sa = runSA(n, 120 * n, rng);
      if (sa.solved) saSolved++;
      await tf.nextFrame();
    }
    policyMoves.sort((a, b) => a - b);
    rows.push({
      n,
      policySuccess: policySolved / trials,
      saSuccess: saSolved / trials,
      policyMoves: policyMoves.length ? policyMoves[Math.floor(policyMoves.length / 2)] : NaN
    });
  }

  drawScaling(rows);
  const longest = rows[rows.length - 1];
  const n = longest.n;
  const policy = rollout(n, mulberry32(9001), false);
  const sa = runSA(n, 120 * n, mulberry32(9002));
  drawAssignment(policy.x, sa.x);

  $('policySuccess').textContent = `${Math.round(100 * longest.policySuccess)}%`;
  $('saSuccess').textContent = `${Math.round(100 * longest.saSuccess)}%`;
  $('policyMoves').textContent = Number.isFinite(longest.policyMoves) ? Math.round(longest.policyMoves).toString() : '—';
  $('saEffort').textContent = (120 * n).toLocaleString();
  $('metricLength').textContent = `n=${n} · domain 0…${DOMAIN_MAX} · terminal reward · ≤${MOVE_MULTIPLIER}n moves`;
  statusEl.textContent = `Done. The GNN gets up to ${MOVE_MULTIPLIER * n} joint (i,v) choices and stops immediately when it first reaches a feasible full assignment.`;

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
  ctx.fillText('configured max', Math.min(w - 95, bx + 5), 28);
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
    statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Train the ≤${MOVE_MULTIPLIER}n-move terminal-reward GNN curriculum to begin.`;
  });
} else {
  statusEl.textContent = 'TensorFlow.js failed to load.';
}
