const DOMAIN_MAX = 1000;
const MAX_TEST_N = 100;
const NODE_FEATURE_DIM = 5;
const EMBED_DIM = 24;
const MESSAGE_ROUNDS = 6;
const EDGE_TYPES = 4;
const MOVE_MULTIPLIER = 3;
const INFERENCE_MOVE_MULTIPLIER = 30;

const PPO_BATCH_EPISODES = 4;
const PPO_EPOCHS = 2;
const PPO_CLIP = 0.20;
const GAMMA = 0.995;
const GAE_LAMBDA = 0.95;
const VALUE_COEF = 0.50;
const ENTROPY_COEF = 0.002;
const SELF_IMITATION_COEF = 0.03;
const SUCCESS_REPLAY_LIMIT = 1200;
const PROMOTE_EVAL_EPISODES = 10;
const PROMOTE_GREEDY_RATE = 0.80;
const MIN_STAGE_EPISODES = 40;

const SEARCH_EPS_START = 0.20;
const SEARCH_EPS_END = 0.01;
const SEARCH_T0 = 80;
const SEARCH_TEND = 0.02;

let params = null;
let optimizer = null;
let trained = false;
let trainedRange = { min: 2, max: 20 };
let graphCache = new Map();
let paramGeneration = 0;
let successReplay = [];
let reachedN = 2;

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

function satisfiedConstraintFraction(x) {
  if (x.length <= 1) return 1;
  let satisfied = 0;
  for (let i = 0; i + 1 < x.length; i++) if (x[i] < x[i + 1]) satisfied++;
  return satisfied / (x.length - 1);
}

function terminalReward(x, solved, moves, n) {
  const satisfied = satisfiedConstraintFraction(x);
  if (!solved) return { reward: satisfied, satisfied, efficiency: 0 };

  const minMoves = n;
  const maxMoves = MOVE_MULTIPLIER * n;
  const efficiency = Math.max(0, Math.min(1,
    (maxMoves - moves) / Math.max(1, maxMoves - minMoves)
  ));
  return {
    reward: satisfied + 2 + efficiency,
    satisfied,
    efficiency
  };
}

function sigmoidScalar(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

function latentToValue(y) {
  return Math.max(0, Math.min(DOMAIN_MAX, Math.round(DOMAIN_MAX * sigmoidScalar(y))));
}

function gaussianLogProb(y, mu, logStd) {
  const invVar = Math.exp(-2 * logStd);
  const d = y - mu;
  return -0.5 * d * d * invVar - logStd - 0.5 * Math.log(2 * Math.PI);
}

function gaussianSample(mu, logStd, rng) {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  const eps = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mu + Math.exp(logStd) * eps;
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

function categoricalLogProb(logits, allowedIds, chosenId) {
  let max = -Infinity;
  for (const i of allowedIds) max = Math.max(max, logits[i]);
  let sum = 0;
  for (const i of allowedIds) sum += Math.exp(logits[i] - max);
  return logits[chosenId] - max - Math.log(sum);
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
    params.Winit, params.binit, params.Wself, params.brel, ...params.Wrel,
    params.Wvar, params.bvar,
    params.Wmu, params.bmu, params.WlogStd, params.blogStd,
    params.Wcritic, params.bcritic, params.Wvalue, params.bvalue
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
    Wvar: makeVariable([EMBED_DIM, 1], s, 'Wvar'),
    bvar: makeBias(1, 'bvar'),
    Wmu: makeVariable([EMBED_DIM, 1], s, 'Wmu'),
    bmu: makeBias(1, 'bmu'),
    WlogStd: makeVariable([EMBED_DIM, 1], s * 0.5, 'WlogStd'),
    blogStd: makeBias(1, 'blogStd'),
    Wcritic: makeVariable([EMBED_DIM, 32], s, 'Wcritic'),
    bcritic: makeBias(32, 'bcritic'),
    Wvalue: makeVariable([32, 1], s, 'Wvalue'),
    bvalue: makeBias(1, 'bvalue')
  };
  optimizer = tf.train.adam(0.0008);
}

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
      z = z.add(tf.matMul(adj[r], tf.matMul(h, params.Wrel[r])));
    }
    h = tf.relu(h.add(z.add(params.brel)));
  }
  return h;
}

function policyTensors(state) {
  const h = graphEmbeddings(state);
  const variableH = h.slice([0, 0], [state.n, EMBED_DIM]);
  const varLogits = tf.matMul(variableH, params.Wvar).add(params.bvar).reshape([state.n]);
  const mu = tf.matMul(variableH, params.Wmu).add(params.bmu).reshape([state.n]).clipByValue(-6, 6);
  const logStd = tf.matMul(variableH, params.WlogStd).add(params.blogStd).reshape([state.n]).clipByValue(-2.5, 0.6);
  const pooled = h.mean(0).reshape([1, EMBED_DIM]);
  const criticH = tf.relu(tf.matMul(pooled, params.Wcritic).add(params.bcritic));
  const value = tf.matMul(criticH, params.Wvalue).add(params.bvalue).reshape([]);
  return { h, variableH, varLogits, mu, logStd, value };
}

function policySnapshot(state) {
  return tf.tidy(() => {
    const p = policyTensors(state);
    return {
      varLogits: Float32Array.from(p.varLogits.dataSync()),
      mu: Float32Array.from(p.mu.dataSync()),
      logStd: Float32Array.from(p.logStd.dataSync()),
      value: p.value.dataSync()[0]
    };
  });
}

function allowedVariableIds(state) {
  const ids = [];
  const correctionPhase = state.assignedCount === state.n;
  for (let i = 0; i < state.n; i++) {
    if (correctionPhase || !state.assigned[i]) ids.push(i);
  }
  return ids;
}

function applyAction(state, i, v) {
  if (!state.assigned[i]) {
    state.assigned[i] = 1;
    state.assignedCount++;
  }
  state.values[i] = v;
}

function rollout(n, rng, stochastic = true) {
  const state = emptyState(n);
  const trajectory = [];
  const maxMoves = MOVE_MULTIPLIER * n;
  let solved = false;
  let moves = 0;

  for (let move = 0; move < maxMoves; move++) {
    const snap = policySnapshot(state);
    const allowed = allowedVariableIds(state);
    let i;
    if (stochastic) i = sampleCategorical(snap.varLogits, allowed, rng);
    else {
      i = allowed[0];
      for (const candidate of allowed) if (snap.varLogits[candidate] > snap.varLogits[i]) i = candidate;
    }

    const mu = snap.mu[i];
    const logStd = snap.logStd[i];
    const y = stochastic ? gaussianSample(mu, logStd, rng) : mu;
    const v = latentToValue(y);
    const oldLogProb = categoricalLogProb(snap.varLogits, allowed, i) + gaussianLogProb(y, mu, logStd);

    if (stochastic) {
      trajectory.push({
        state: cloneState(state),
        i, y, v,
        oldLogProb,
        value: snap.value,
        reward: 0,
        done: false,
        advantage: 0,
        ret: 0
      });
    }

    applyAction(state, i, v);
    moves = move + 1;

    if (state.assignedCount === n && isStrictChain(state.values)) {
      solved = true;
      break;
    }
  }

  const terminal = terminalReward(state.values, solved, moves, n);
  if (stochastic && trajectory.length) {
    const last = trajectory[trajectory.length - 1];
    last.reward = terminal.reward;
    last.done = true;

    let gae = 0;
    for (let t = trajectory.length - 1; t >= 0; t--) {
      const nextValue = t + 1 < trajectory.length ? trajectory[t + 1].value : 0;
      const nonterminal = trajectory[t].done ? 0 : 1;
      const delta = trajectory[t].reward + GAMMA * nextValue * nonterminal - trajectory[t].value;
      gae = delta + GAMMA * GAE_LAMBDA * nonterminal * gae;
      trajectory[t].advantage = gae;
      trajectory[t].ret = gae + trajectory[t].value;
    }
  }

  return {
    solved,
    x: Int16Array.from(state.values),
    trajectory,
    moves,
    reward: terminal.reward,
    satisfiedFraction: terminal.satisfied,
    efficiency: terminal.efficiency
  };
}

function normalizeAdvantages(samples) {
  if (!samples.length) return;
  let mean = 0;
  for (const s of samples) mean += s.advantage;
  mean /= samples.length;
  let variance = 0;
  for (const s of samples) variance += (s.advantage - mean) ** 2;
  variance /= samples.length;
  const sd = Math.sqrt(variance + 1e-8);
  for (const s of samples) s.advantage = (s.advantage - mean) / sd;
}

function logProbAndEntropyTensors(step) {
  const p = policyTensors(step.state);
  const allowed = allowedVariableIds(step.state);
  const ids = tf.tensor1d(Int32Array.from(allowed), 'int32');
  const allowedLogits = tf.gather(p.varLogits, ids);
  const logProbs = tf.logSoftmax(allowedLogits);
  const probs = tf.softmax(allowedLogits);
  const localIndex = allowed.indexOf(step.i);
  const catLogProb = logProbs.gather(localIndex);
  const catEntropy = probs.mul(logProbs).sum().neg();

  const mu = p.mu.gather(step.i);
  const logStd = p.logStd.gather(step.i);
  const y = tf.scalar(step.y);
  const diff = y.sub(mu);
  const gaussianLogP = diff.square().mul(tf.exp(logStd.mul(-2))).mul(-0.5)
    .sub(logStd)
    .sub(0.5 * Math.log(2 * Math.PI));
  const gaussianEntropy = logStd.add(0.5 * Math.log(2 * Math.PI * Math.E));
  return {
    logProb: catLogProb.add(gaussianLogP),
    entropy: catEntropy.add(gaussianEntropy),
    value: p.value
  };
}

function rememberSuccess(trajectory) {
  for (const step of trajectory) successReplay.push({ state: step.state, i: step.i, y: step.y });
  if (successReplay.length > SUCCESS_REPLAY_LIMIT) {
    successReplay.splice(0, successReplay.length - SUCCESS_REPLAY_LIMIT);
  }
}

async function ppoUpdate(samples, rng) {
  if (!samples.length) return NaN;
  normalizeAdvantages(samples);
  let lastLoss = NaN;

  for (let epoch = 0; epoch < PPO_EPOCHS; epoch++) {
    const imitationCount = Math.min(12, successReplay.length);
    const imitationSamples = [];
    for (let k = 0; k < imitationCount; k++) {
      imitationSamples.push(successReplay[Math.floor(rng() * successReplay.length)]);
    }

    const cost = optimizer.minimize(() => tf.tidy(() => {
      let total = tf.scalar(0);
      for (const step of samples) {
        const cur = logProbAndEntropyTensors(step);
        const ratio = tf.exp(cur.logProb.sub(step.oldLogProb));
        const adv = tf.scalar(step.advantage);
        const unclipped = ratio.mul(adv);
        const clipped = tf.clipByValue(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP).mul(adv);
        const policyLoss = tf.minimum(unclipped, clipped).neg();
        const valueLoss = cur.value.sub(step.ret).square().mul(VALUE_COEF);
        const entropyLoss = cur.entropy.mul(-ENTROPY_COEF);
        total = total.add(policyLoss.add(valueLoss).add(entropyLoss));
      }
      total = total.div(samples.length);

      if (imitationSamples.length) {
        let imitation = tf.scalar(0);
        for (const s of imitationSamples) {
          const cur = logProbAndEntropyTensors({ state: s.state, i: s.i, y: s.y });
          imitation = imitation.add(cur.logProb.neg());
        }
        total = total.add(imitation.div(imitationSamples.length).mul(SELF_IMITATION_COEF));
      }
      return total;
    }), true, trainableParams());

    lastLoss = cost.dataSync()[0];
    cost.dispose();
    await tf.nextFrame();
  }
  return lastLoss;
}

function chooseCurriculumN(currentN, startN, rng) {
  if (currentN <= startN) return currentN;
  const r = rng();
  if (r < 0.65) return currentN;
  if (r < 0.90) return currentN - 1;
  if (currentN - 2 < startN) return startN;
  return startN + Math.floor(rng() * (currentN - 1 - startN));
}

function greedySolveRate(n, rng, trials = PROMOTE_EVAL_EPISODES) {
  let solved = 0;
  let moves = 0;
  for (let t = 0; t < trials; t++) {
    const out = rollout(n, rng, false);
    if (out.solved) solved++;
    moves += out.moves;
  }
  return { rate: solved / trials, avgMoves: moves / trials };
}

async function trainRL() {
  if (!window.tf) throw new Error('TensorFlow.js did not load.');
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;

  const startN = +trainMinN.value;
  const maxN = +trainMaxN.value;
  const totalEpisodes = +episodes.value;
  trainedRange = { min: startN, max: maxN };
  trained = false;
  reachedN = startN;
  successReplay = [];
  initGNN();

  const rng = mulberry32(1337);
  let currentN = startN;
  let stageEpisodes = 0;
  let completed = 0;
  let updates = 0;
  let lastLoss = NaN;
  let lastGreedy = { rate: 0, avgMoves: 0 };
  let stageSuccess = [];

  while (completed < totalEpisodes) {
    const batch = [];
    const batchEpisodes = Math.min(PPO_BATCH_EPISODES, totalEpisodes - completed);

    for (let b = 0; b < batchEpisodes; b++) {
      const n = chooseCurriculumN(currentN, startN, rng);
      const out = rollout(n, rng, true);
      if (out.solved) rememberSuccess(out.trajectory);
      for (const step of out.trajectory) batch.push(step);

      if (n === currentN) {
        stageSuccess.push(out.solved ? 1 : 0);
        if (stageSuccess.length > 40) stageSuccess.shift();
        stageEpisodes++;
      }
      completed++;
    }

    lastLoss = await ppoUpdate(batch, rng);
    updates++;

    const shouldEvaluate = updates % 5 === 0 || completed >= totalEpisodes;
    if (shouldEvaluate) {
      lastGreedy = greedySolveRate(currentN, rng);
      const stochasticRate = stageSuccess.length
        ? stageSuccess.reduce((a, b) => a + b, 0) / stageSuccess.length
        : 0;

      const promote = currentN < maxN
        && stageEpisodes >= MIN_STAGE_EPISODES
        && lastGreedy.rate >= PROMOTE_GREEDY_RATE;

      if (promote) {
        currentN++;
        reachedN = Math.max(reachedN, currentN);
        stageEpisodes = 0;
        stageSuccess = [];
      }

      statusEl.textContent = `Episode ${completed}/${totalEpisodes} · PPO update ${updates} · curriculum n=${currentN}${currentN < maxN ? `/${maxN}` : ' (max)'} · greedy ${(100 * lastGreedy.rate).toFixed(0)}% · exploratory ${(100 * stochasticRate).toFixed(0)}% · success replay ${successReplay.length}${Number.isFinite(lastLoss) ? ` · loss ${lastLoss.toFixed(4)}` : ''}${promote ? ' · PROMOTED' : ''}`;
      await tf.nextFrame();
    }
  }

  reachedN = Math.max(reachedN, currentN);
  trained = true;
  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
  statusEl.textContent = `PPO training complete · curriculum reached n=${reachedN}. Running hybrid-search benchmark…`;
  await runBenchmark();
}

function violationEnergy(x) {
  let e = 0;
  for (let i = 0; i + 1 < x.length; i++) e += Math.max(0, x[i] - x[i + 1] + 1);
  return e;
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
  return { x: best, solved: bestE === 0, finalEnergy: bestE, evaluations };
}

function greedyPolicyProposal(state) {
  const snap = policySnapshot(state);
  const allowed = allowedVariableIds(state);
  let i = allowed[0];
  for (const candidate of allowed) {
    if (snap.varLogits[candidate] > snap.varLogits[i]) i = candidate;
  }
  return { i, v: latentToValue(snap.mu[i]) };
}

function learnedSearchProposal(state, rng) {
  const snap = policySnapshot(state);
  const allowed = allowedVariableIds(state);
  const i = sampleCategorical(snap.varLogits, allowed, rng);
  const searchLogStd = Math.max(-3.0, snap.logStd[i] - 0.6);
  const y = gaussianSample(snap.mu[i], searchLogStd, rng);
  return { i, v: latentToValue(y) };
}

function randomSearchProposal(state, progress, rng) {
  const i = Math.floor(rng() * state.n);
  let v;
  if (rng() < 0.70) {
    const span = Math.max(2, Math.round(DOMAIN_MAX * (0.25 * (1 - progress) + 0.015)));
    const delta = Math.round((2 * rng() - 1) * span);
    v = Math.max(0, Math.min(DOMAIN_MAX, state.values[i] + delta));
  } else {
    v = Math.floor(rng() * (DOMAIN_MAX + 1));
  }
  return { i, v };
}

function hybridSolve(n, rng) {
  const state = emptyState(n);
  let moves = 0;

  while (state.assignedCount < n) {
    const proposal = greedyPolicyProposal(state);
    applyAction(state, proposal.i, proposal.v);
    moves++;
  }

  let currentEnergy = violationEnergy(state.values);
  let bestEnergy = currentEnergy;
  let best = Int16Array.from(state.values);
  const maxMoves = INFERENCE_MOVE_MULTIPLIER * n;
  if (bestEnergy === 0) return { solved: true, x: best, moves, finalEnergy: 0, maxMoves };

  const searchMoves = Math.max(1, maxMoves - n);
  while (moves < maxMoves && bestEnergy > 0) {
    const progress = Math.max(0, Math.min(1, (moves - n) / Math.max(1, searchMoves - 1)));
    const epsilon = SEARCH_EPS_START + (SEARCH_EPS_END - SEARCH_EPS_START) * progress;
    const temperature = SEARCH_T0 * Math.pow(SEARCH_TEND / SEARCH_T0, progress);
    const proposal = rng() < epsilon
      ? randomSearchProposal(state, progress, rng)
      : learnedSearchProposal(state, rng);

    const old = state.values[proposal.i];
    state.values[proposal.i] = proposal.v;
    const nextEnergy = violationEnergy(state.values);
    const delta = nextEnergy - currentEnergy;
    const accept = delta <= 0 || rng() < Math.exp(-delta / Math.max(1e-6, temperature));
    moves++;

    if (accept) {
      currentEnergy = nextEnergy;
      if (currentEnergy < bestEnergy) {
        bestEnergy = currentEnergy;
        best = Int16Array.from(state.values);
      }
    } else {
      state.values[proposal.i] = old;
    }
  }

  return { solved: bestEnergy === 0, x: best, moves, finalEnergy: bestEnergy, maxMoves };
}

function benchmarkLengths() {
  const m = Math.max(trainedRange.min, reachedN);
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
  const rng = mulberry32(20260811);
  const trials = 5;

  for (const n of benchmarkLengths()) {
    statusEl.textContent = `Benchmarking PPO-guided annealed search at n=${n}…`;
    let policySolved = 0;
    let saSolved = 0;
    const policyMoves = [];
    for (let t = 0; t < trials; t++) {
      const policy = hybridSolve(n, rng);
      if (policy.solved) {
        policySolved++;
        policyMoves.push(policy.moves);
      }
      const sa = runSA(n, 120 * n, rng);
      if (sa.solved) saSolved++;
      await tf.nextFrame();
    }
    rows.push({
      n,
      policySuccess: policySolved / trials,
      saSuccess: saSolved / trials,
      policyMoves: median(policyMoves)
    });
  }

  drawScaling(rows);
  const longest = rows[rows.length - 1];
  const n = longest.n;
  const policy = hybridSolve(n, mulberry32(9001));
  const sa = runSA(n, 120 * n, mulberry32(9002));
  drawAssignment(policy.x, sa.x);

  $('policySuccess').textContent = `${Math.round(100 * longest.policySuccess)}%`;
  $('saSuccess').textContent = `${Math.round(100 * longest.saSuccess)}%`;
  $('policyMoves').textContent = Number.isFinite(longest.policyMoves) ? Math.round(longest.policyMoves).toString() : '—';
  $('saEffort').textContent = (120 * n).toLocaleString();
  $('metricLength').textContent = `n=${n} · domain 0…${DOMAIN_MAX} · training ${MOVE_MULTIPLIER}n · hybrid inference ${INFERENCE_MOVE_MULTIPLIER}n · curriculum reached n=${reachedN}`;
  statusEl.textContent = `Done. Training stays at ${MOVE_MULTIPLIER}n moves; inference gets up to ${INFERENCE_MOVE_MULTIPLIER}n moves with PPO proposals, ε exploration, annealed acceptance, and a protected incumbent.`;

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
  const bx = X(reachedN);
  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = '#a9b1c2';
  ctx.beginPath();
  ctx.moveTo(bx, 18);
  ctx.lineTo(bx, h - pad);
  ctx.stroke();
  ctx.restore();
  ctx.fillText('curriculum reached', Math.min(w - 115, bx + 5), 28);
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
    statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Train the PPO actor-critic GNN curriculum; benchmark uses ${INFERENCE_MOVE_MULTIPLIER}n hybrid search.`;
  });
} else {
  statusEl.textContent = 'TensorFlow.js failed to load.';
}
