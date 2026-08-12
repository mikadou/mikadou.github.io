const DOMAIN_MIN = 0;
const DOMAIN_MAX = 1000;
const DOMAIN_SPAN = DOMAIN_MAX - DOMAIN_MIN;
const MAX_TEST_N = 100;
const NODE_FEATURE_DIM = 15;
const EMBED_DIM = 24;
const EDGE_TYPES = 6;
const MOVE_MULTIPLIER = 3;
const TEACHER_SIGMA = 1.0;

const BATCH_STATES = 12;
const UPDATE_EPOCHS = 3;
const PROPOSAL_SAMPLES = 4;
const RANDOM_CANDIDATES = 2;
const VARIABLE_TEMPERATURE = 0.30;
const ACTION_TEMPERATURE = 0.12;
const VARIABLE_THRESHOLD = 0.5;
const VARIABLE_MARGIN = 2.0;
const PROPOSAL_LOSS_WEIGHT = 0.35;
const CRITIC_LOSS_WEIGHT = 0.75;
const GAMMA = 0.97;
const DAGGER_WARMUP_FRACTION = 0.22;
const DAGGER_MAX_RATE = 0.75;

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
  section.innerHTML = `<h2>Copyable diagnostic log</h2><p>After training and benchmark finish, tap <strong>Copy diagnostics</strong> and paste the block into ChatGPT.</p><div class="buttons" style="margin:12px 0"><button id="copyLogBtn" type="button">Copy diagnostics</button><button id="clearLogBtn" type="button" class="secondary">Clear log</button></div><textarea id="diagnosticLog" readonly spellcheck="false" style="width:100%;min-height:390px;resize:vertical;border:1px solid #dfe4ee;border-radius:10px;padding:12px;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#fbfcff;color:#20283a"></textarea>`;
  const footer = document.querySelector('footer');
  footer.parentNode.insertBefore(section, footer);
  $('copyLogBtn').addEventListener('click', async () => {
    const text = diagnosticLines.join('\n');
    try { await navigator.clipboard.writeText(text); $('copyLogBtn').textContent = 'Copied'; setTimeout(() => { $('copyLogBtn').textContent = 'Copy diagnostics'; }, 1000); }
    catch { const el = $('diagnosticLog'); el.focus(); el.select(); document.execCommand('copy'); }
  });
  $('clearLogBtn').addEventListener('click', () => { diagnosticLines = []; syncLog(); });
}
function syncLog() { const el = $('diagnosticLog'); if (el) { el.value = diagnosticLines.join('\n'); el.scrollTop = el.scrollHeight; } }
function logLine(line) { diagnosticLines.push(String(line)); syncLog(); }
function resetLog() { diagnosticLines = []; syncLog(); }
installDiagnosticsUI();

function bindRange(input, labelId) { const update = () => $(labelId).textContent = input.value; input.addEventListener('input', update); update(); }
bindRange(trainMinN, 'trainMinNLabel'); bindRange(trainMaxN, 'trainMaxNLabel'); bindRange(episodes, 'episodesLabel');
trainMinN.addEventListener('input', () => { if (+trainMinN.value > +trainMaxN.value) { trainMaxN.value = trainMinN.value; $('trainMaxNLabel').textContent = trainMaxN.value; } });
trainMaxN.addEventListener('input', () => { if (+trainMaxN.value < +trainMinN.value) { trainMinN.value = trainMaxN.value; $('trainMinNLabel').textContent = trainMinN.value; } });
messageRounds.addEventListener('change', () => { if (trained) { trained = false; benchmarkBtn.disabled = true; statusEl.textContent = 'Message-passing depth changed. Retrain before benchmarking.'; } });

function mulberry32(seed) { return function () { let t = seed += 0x6D2B79F5; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function gaussian(mean, sigma, rng) { const u1 = Math.max(1e-12, rng()), u2 = rng(); return mean + sigma * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2); }
function sigmoidScalar(x) { return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); }
function clampValue(v) { return Math.max(DOMAIN_MIN, Math.min(DOMAIN_MAX, Math.round(v))); }
function emptyState(n) { return { n, assigned: new Uint8Array(n), values: Int16Array.from({ length: n }, () => -1), assignedCount: 0 }; }
function cloneState(state) { return { n: state.n, assigned: Uint8Array.from(state.assigned), values: Int16Array.from(state.values), assignedCount: state.assignedCount }; }
function fullStateFromValues(values) { const n = values.length; return { n, assigned: Uint8Array.from({ length: n }, () => 1), values: Int16Array.from(values), assignedCount: n }; }
function applyAction(state, i, v) { if (!state.assigned[i]) { state.assigned[i] = 1; state.assignedCount++; } state.values[i] = v; }
function isStrictChain(x) { for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) return false; return true; }
function violationEnergy(x) { let e = 0; for (let i = 0; i + 1 < x.length; i++) e += Math.max(0, x[i] - x[i + 1] + 1); return e; }
function violatedConstraintCount(x) { let c = 0; for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) c++; return c; }
function violatedEndpointIds(x) { const marked = new Uint8Array(x.length); for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) { marked[i] = 1; marked[i + 1] = 1; } const ids = []; for (let i = 0; i < x.length; i++) if (marked[i]) ids.push(i); return ids; }

// Problem adapter: exact solver state information exposed to the generic graph/value machinery.
function observedStats(state) {
  let observed = 0, satisfied = 0, violated = 0, energy = 0;
  for (let k = 0; k + 1 < state.n; k++) {
    if (!state.assigned[k] || !state.assigned[k + 1]) continue;
    observed++;
    if (state.values[k] < state.values[k + 1]) satisfied++;
    else { violated++; energy += state.values[k] - state.values[k + 1] + 1; }
  }
  return { observed, satisfied, violated, energy };
}
function solverObjective(state) {
  const d = Math.max(1, state.n - 1), s = observedStats(state);
  // Adapter-provided objective/quality. Satisfied known constraints help; violations hurt more.
  return (s.satisfied - 2 * s.violated - 0.25 * Math.tanh(s.energy / 8)) / d;
}
function terminalUtility(state) { return state.assignedCount === state.n && isStrictChain(state.values) ? 1 : 0; }
function immediateReward(before, after) {
  const base = solverObjective(after) - solverObjective(before);
  const bonus = terminalUtility(after) - terminalUtility(before);
  return base + bonus;
}
function propagateSuccessor(state, variableId, value) {
  // Generic extension point: a CP/MIP adapter would apply the tentative assignment and propagate domains/constraints here.
  const next = cloneState(state);
  applyAction(next, variableId, value);
  return next;
}

// Teacher exists only to generate proposal demonstrations and rollout value targets.
function teacherMeanValue(i) { return Math.max(DOMAIN_MIN, Math.min(DOMAIN_MAX, 2 * i)); }
function teacherSampleValue(i, rng) { return clampValue(gaussian(teacherMeanValue(i), TEACHER_SIGMA, rng)); }

function makeRepairTrainingState(n, rng) {
  const values = new Int16Array(n); let current = Math.floor(rng() * 12);
  for (let i = 0; i < n; i++) { if (i > 0) current += 1 + Math.floor(rng() * 4); values[i] = Math.min(900, current); }
  const maxInject = Math.max(1, Math.min(4, Math.ceil(n / 7))), injectCount = 1 + Math.floor(rng() * maxInject), chosen = new Set();
  while (chosen.size < Math.min(injectCount, Math.max(1, n - 1))) chosen.add(Math.floor(rng() * Math.max(1, n - 1)));
  for (const k of chosen) { if (k + 1 >= n) continue; if (rng() < 0.5) values[k + 1] = Math.max(DOMAIN_MIN, values[k] - Math.floor(3 * rng())); else values[k] = Math.min(DOMAIN_MAX, values[k + 1] + Math.floor(3 * rng())); }
  if (isStrictChain(values) && n > 1) { const k = Math.floor(rng() * (n - 1)); values[k + 1] = values[k]; }
  const state = fullStateFromValues(values); return { state, candidateIds: Int32Array.from(violatedEndpointIds(values)) };
}
function makeSyntheticState(n, targetVar, rng) {
  if (rng() < 0.50) return makeRepairTrainingState(n, rng).state;
  const state = emptyState(n), ids = Array.from({ length: n }, (_, i) => i).filter(i => i !== targetVar);
  for (let k = ids.length - 1; k > 0; k--) { const j = Math.floor(rng() * (k + 1)); [ids[k], ids[j]] = [ids[j], ids[k]]; }
  const assignCount = Math.floor(rng() * Math.max(1, n - 1));
  for (let k = 0; k < assignCount; k++) applyAction(state, ids[k], teacherSampleValue(ids[k], rng));
  return state;
}

function makeVariable(shape, scale, name, mean = 0) { const init = tf.randomNormal(shape, mean, scale), v = tf.variable(init, true, `${name}_${paramGeneration}`); init.dispose(); return v; }
function makeBias(size, name, value = 0) { const init = tf.fill([size], value), v = tf.variable(init, true, `${name}_${paramGeneration}`); init.dispose(); return v; }
function trainableParams() { if (!params) return []; return [params.Winit, params.binit, params.Wself, params.brel, ...params.Wrel, params.Wvar, params.bvar, params.Wproposal1, params.bproposal1, params.WproposalMean, params.bproposalMean, params.WproposalLogStd, params.bproposalLogStd, params.Wcritic1, params.bcritic1, params.Wcritic2, params.bcritic2]; }
function disposeModel() { if (params) for (const p of trainableParams()) p.dispose(); if (optimizer?.dispose) optimizer.dispose(); params = null; optimizer = null; }
function initModel() {
  disposeModel(); paramGeneration++; const s = 0.08;
  params = {
    Winit: makeVariable([NODE_FEATURE_DIM, EMBED_DIM], s, 'Winit'), binit: makeBias(EMBED_DIM, 'binit'),
    Wself: makeVariable([EMBED_DIM, EMBED_DIM], s, 'Wself'), brel: makeBias(EMBED_DIM, 'brel'),
    Wrel: Array.from({ length: EDGE_TYPES }, (_, r) => makeVariable([EMBED_DIM, EMBED_DIM], s, `Wrel${r}`)),
    Wvar: makeVariable([EMBED_DIM, 1], s, 'Wvar'), bvar: makeBias(1, 'bvar'),
    Wproposal1: makeVariable([3 * EMBED_DIM, 48], s, 'Wproposal1'), bproposal1: makeBias(48, 'bproposal1'),
    WproposalMean: makeVariable([48, 1], s * 0.5, 'WproposalMean'), bproposalMean: makeBias(1, 'bproposalMean'),
    WproposalLogStd: makeVariable([48, 1], s * 0.3, 'WproposalLogStd'), bproposalLogStd: makeBias(1, 'bproposalLogStd', -2.3),
    Wcritic1: makeVariable([2 * EMBED_DIM, 48], s, 'Wcritic1'), bcritic1: makeBias(48, 'bcritic1'),
    Wcritic2: makeVariable([48, 1], s * 0.6, 'Wcritic2'), bcritic2: makeBias(1, 'bcritic2')
  };
  optimizer = tf.train.adam(0.002);
}

function graphSpec(n) {
  if (graphCache.has(n)) return graphCache.get(n);
  const constraintCount = n - 1, globalId = n + constraintCount, nodeCount = globalId + 1;
  const baseFeatures = new Float32Array(nodeCount * NODE_FEATURE_DIM), absScale = Math.max(1, MAX_TEST_N - 1), sizeFeature = n / MAX_TEST_N;
  for (let i = 0; i < n; i++) { const p = i * NODE_FEATURE_DIM; baseFeatures[p] = 1; baseFeatures[p + 3] = i / absScale; baseFeatures[p + 4] = n <= 1 ? 0 : i / (n - 1); baseFeatures[p + 5] = sizeFeature; }
  for (let k = 0; k < constraintCount; k++) { const p = (n + k) * NODE_FEATURE_DIM; baseFeatures[p + 1] = 1; baseFeatures[p + 3] = k / absScale; baseFeatures[p + 4] = constraintCount <= 1 ? 0 : k / Math.max(1, constraintCount - 1); baseFeatures[p + 5] = sizeFeature; }
  const gp = globalId * NODE_FEATURE_DIM; baseFeatures[gp + 2] = 1; baseFeatures[gp + 5] = sizeFeature;
  const adjs = Array.from({ length: EDGE_TYPES }, () => new Float32Array(nodeCount * nodeCount));
  const setEdge = (type, target, source) => { adjs[type][target * nodeCount + source] = 1; };
  for (let k = 0; k < constraintCount; k++) { const c = n + k; setEdge(0, c, k); setEdge(1, c, k + 1); setEdge(2, k, c); setEdge(3, k + 1, c); }
  for (let node = 0; node < globalId; node++) { setEdge(4, globalId, node); setEdge(5, node, globalId); }
  const spec = { n, constraintCount, globalId, nodeCount, baseFeatures, adjs }; graphCache.set(n, spec); return spec;
}
function nodeFeatureData(state) {
  const spec = graphSpec(state.n), data = Float32Array.from(spec.baseFeatures), stats = observedStats(state), d = Math.max(1, state.n - 1);
  for (let i = 0; i < state.n; i++) { const p = i * NODE_FEATURE_DIM; data[p + 6] = state.assigned[i] ? 1 : 0; data[p + 7] = state.assigned[i] ? (state.values[i] - DOMAIN_MIN) / DOMAIN_SPAN : 0; }
  for (let k = 0; k + 1 < state.n; k++) {
    const p = (state.n + k) * NODE_FEATURE_DIM;
    if (state.assigned[k] && state.assigned[k + 1]) { const violated = state.values[k] >= state.values[k + 1]; data[p + 8] = violated ? 1 : 0; data[p + 9] = violated ? 0 : 1; }
    else data[p + 10] = 1;
  }
  const gp = spec.globalId * NODE_FEATURE_DIM;
  data[gp + 11] = state.assignedCount / Math.max(1, state.n);
  data[gp + 12] = stats.violated / d;
  data[gp + 13] = Math.tanh(stats.energy / Math.max(1, 4 * d));
  data[gp + 14] = solverObjective(state);
  return data;
}
function graphEmbeddings(state) {
  const spec = graphSpec(state.n), nodeX = tf.tensor2d(nodeFeatureData(state), [spec.nodeCount, NODE_FEATURE_DIM]), adj = spec.adjs.map(a => tf.tensor2d(a, [spec.nodeCount, spec.nodeCount]));
  const h0 = tf.relu(tf.matMul(nodeX, params.Winit).add(params.binit)); let h = h0;
  for (let round = 0; round < MESSAGE_ROUNDS; round++) { let z = tf.matMul(h, params.Wself); for (let r = 0; r < EDGE_TYPES; r++) z = z.add(tf.matMul(adj[r], tf.matMul(h, params.Wrel[r]))); h = tf.relu(h.add(z.add(params.brel))); }
  return { h0, h };
}
function variableLogitsFromRep(rep, state) { const variableH = rep.h.slice([0, 0], [state.n, EMBED_DIM]); return tf.matMul(variableH, params.Wvar).add(params.bvar).reshape([state.n]); }
function proposalTensorsFromRep(rep, state, variableId) {
  const spec = graphSpec(state.n), local0 = rep.h0.slice([variableId, 0], [1, EMBED_DIM]), local = rep.h.slice([variableId, 0], [1, EMBED_DIM]), global = rep.h.slice([spec.globalId, 0], [1, EMBED_DIM]), x = tf.concat([local0, local, global], 1), hidden = tf.relu(tf.matMul(x, params.Wproposal1).add(params.bproposal1)), meanNorm = tf.sigmoid(tf.matMul(hidden, params.WproposalMean).add(params.bproposalMean)).reshape([]), logStdNorm = tf.matMul(hidden, params.WproposalLogStd).add(params.bproposalLogStd).reshape([]).clipByValue(-7.5, -0.15); return { meanNorm, logStdNorm };
}
function criticTensorFromRep(rep, state) {
  const spec = graphSpec(state.n), global = rep.h.slice([spec.globalId, 0], [1, EMBED_DIM]), pooled = rep.h.mean(0).reshape([1, EMBED_DIM]), x = tf.concat([global, pooled], 1), hidden = tf.relu(tf.matMul(x, params.Wcritic1).add(params.bcritic1)); return tf.matMul(hidden, params.Wcritic2).add(params.bcritic2).reshape([]);
}
function variableSnapshot(state) { return tf.tidy(() => { const rep = graphEmbeddings(state); return Float32Array.from(variableLogitsFromRep(rep, state).dataSync()); }); }
function proposalSnapshot(state, variableId) { return tf.tidy(() => { const rep = graphEmbeddings(state), p = proposalTensorsFromRep(rep, state, variableId), meanNorm = p.meanNorm.dataSync()[0], logStdNorm = p.logStdNorm.dataSync()[0]; return { mean: DOMAIN_MIN + DOMAIN_SPAN * meanNorm, sigma: Math.max(0.25, DOMAIN_SPAN * Math.exp(logStdNorm)) }; }); }
function criticSnapshot(state) { return tf.tidy(() => { const rep = graphEmbeddings(state); return criticTensorFromRep(rep, state).dataSync()[0]; }); }

function variableLossFromLogits(logits, candidateIds, n) {
  const labelsArray = new Float32Array(n); for (const i of candidateIds) labelsArray[i] = 1;
  const labels = tf.tensor1d(labelsArray), negatives = tf.onesLike(labels).sub(labels), posCount = candidateIds.length, negCount = n - posCount;
  const posBce = tf.softplus(logits.neg()).mul(labels).sum().div(Math.max(1, posCount)), negBce = tf.softplus(logits).mul(negatives).sum().div(Math.max(1, negCount));
  const bce = posCount && negCount ? posBce.add(negBce).mul(0.5) : (posCount ? posBce : negBce);
  const posMargin = tf.relu(tf.scalar(VARIABLE_MARGIN).sub(logits)).mul(labels).sum().div(Math.max(1, posCount)), negMargin = tf.relu(tf.scalar(VARIABLE_MARGIN).add(logits)).mul(negatives).sum().div(Math.max(1, negCount));
  const margin = posCount && negCount ? posMargin.add(negMargin).mul(0.5) : (posCount ? posMargin : negMargin); return bce.add(margin.mul(0.25));
}
function proposalNllFromRep(rep, state, variableId, teacherValue) { const p = proposalTensorsFromRep(rep, state, variableId), targetNorm = tf.scalar((teacherValue - DOMAIN_MIN) / DOMAIN_SPAN), sigmaNorm = tf.exp(p.logStdNorm), z = targetNorm.sub(p.meanNorm).div(sigmaNorm); return z.square().mul(0.5).add(p.logStdNorm); }
function criticHuber(pred, target, delta = 0.5) { const e = pred.sub(tf.scalar(target)), a = e.abs(), q = tf.minimum(a, tf.scalar(delta)), l = a.sub(q); return q.square().mul(0.5).add(l.mul(delta)); }
function trainingSampleLoss(sample) {
  const repairRep = graphEmbeddings(sample.repairState), variableLoss = variableLossFromLogits(variableLogitsFromRep(repairRep, sample.repairState), sample.candidateIds, sample.repairState.n);
  const proposalRep = graphEmbeddings(sample.proposalState), proposalLoss = proposalNllFromRep(proposalRep, sample.proposalState, sample.proposalVar, sample.teacherValue);
  const criticRep = graphEmbeddings(sample.criticState), criticLoss = criticHuber(criticTensorFromRep(criticRep, sample.criticState), sample.criticTarget);
  return variableLoss.add(proposalLoss.mul(PROPOSAL_LOSS_WEIGHT)).add(criticLoss.mul(CRITIC_LOSS_WEIGHT));
}
async function trainingUpdate(samples) { let last = NaN; for (let epoch = 0; epoch < UPDATE_EPOCHS; epoch++) { const cost = optimizer.minimize(() => tf.tidy(() => { let total = tf.scalar(0); for (const s of samples) total = total.add(trainingSampleLoss(s)); return total.div(samples.length); }), true, trainableParams()); last = cost.dataSync()[0]; cost.dispose(); } await tf.nextFrame(); return last; }

function softmaxDistribution(logits, ids, temperature = 1) { const temp = Math.max(1e-3, temperature); let max = -Infinity; for (const i of ids) max = Math.max(max, logits[i] / temp); let sum = 0; const out = []; for (const i of ids) { const w = Math.exp(logits[i] / temp - max); out.push([i, w]); sum += w; } return out.map(([i, w]) => [i, w / sum]); }
function sampleDistribution(entries, rng) { let r = rng(); for (const [id, p] of entries) { r -= p; if (r <= 0) return id; } return entries[entries.length - 1][0]; }
function chooseRepairVariable(state, rng, stochastic = true) { const logits = variableSnapshot(state), predicted = []; for (let i = 0; i < state.n; i++) if (sigmoidScalar(logits[i]) >= VARIABLE_THRESHOLD) predicted.push(i); if (predicted.length) { if (stochastic) return predicted[Math.floor(rng() * predicted.length)]; return predicted.reduce((best, i) => logits[i] > logits[best] ? i : best, predicted[0]); } const ids = Array.from({ length: state.n }, (_, i) => i); if (!stochastic) return ids.reduce((best, i) => logits[i] > logits[best] ? i : best, 0); return sampleDistribution(softmaxDistribution(logits, ids, VARIABLE_TEMPERATURE), rng); }

function addCandidate(set, v) { set.add(clampValue(v)); }
function generateValueCandidates(state, variableId, rng, stochastic = true) {
  const proposal = proposalSnapshot(state, variableId), set = new Set();
  addCandidate(set, proposal.mean);
  for (const z of [-1,-0.5,0.5,1]) addCandidate(set, proposal.mean + z * proposal.sigma);
  if (stochastic) for (let k = 0; k < PROPOSAL_SAMPLES; k++) addCandidate(set, gaussian(proposal.mean, proposal.sigma, rng));
  if (state.assigned[variableId]) { const current = state.values[variableId]; addCandidate(set, current); for (const d of [4]) { addCandidate(set, current - d); addCandidate(set, current + d); } }
  addCandidate(set, DOMAIN_MIN); addCandidate(set, DOMAIN_MAX);
  for (let k = 0; k < RANDOM_CANDIDATES; k++) addCandidate(set, DOMAIN_MIN + Math.floor(rng() * (DOMAIN_SPAN + 1)));
  return { values: Int32Array.from(set), proposal };
}
function chooseValue(state, variableId, rng, stochastic = true, oracleValue = false, proposalOnly = false, myopicOnly = false) {
  if (oracleValue) return teacherSampleValue(variableId, rng);
  const proposal = proposalSnapshot(state, variableId);
  if (proposalOnly) return clampValue(stochastic ? gaussian(proposal.mean, proposal.sigma, rng) : proposal.mean);
  const candidates = generateValueCandidates(state, variableId, rng, stochastic).values, evals = [];
  for (const v of candidates) { const successor = propagateSuccessor(state, variableId, v), reward = immediateReward(state, successor), future = myopicOnly || terminalUtility(successor) ? 0 : criticSnapshot(successor); evals.push({ value: v, reward, future, score: reward + GAMMA * future }); }
  if (!stochastic) return evals.reduce((a,b)=>b.score>a.score?b:a).value;
  const scores = Float32Array.from(evals, e => e.score), ids = Array.from({ length: evals.length }, (_, i) => i), picked = sampleDistribution(softmaxDistribution(scores, ids, ACTION_TEMPERATURE), rng); return evals[picked].value;
}

function teacherContinuationReturn(startState, rng, maxSteps = null) {
  let state = cloneState(startState), discount = 1, ret = 0, steps = 0, limit = maxSteps == null ? 2 * state.n : maxSteps;
  while (state.assignedCount < state.n && steps < limit) {
    const remaining = []; for (let i = 0; i < state.n; i++) if (!state.assigned[i]) remaining.push(i);
    const i = remaining[Math.floor(rng() * remaining.length)], next = propagateSuccessor(state, i, teacherSampleValue(i, rng)); ret += discount * immediateReward(state, next); discount *= GAMMA; state = next; steps++;
  }
  while (steps < limit && state.assignedCount === state.n && !isStrictChain(state.values)) {
    const ids = violatedEndpointIds(state.values); if (!ids.length) break;
    const i = ids[Math.floor(rng() * ids.length)], next = propagateSuccessor(state, i, teacherSampleValue(i, rng)); ret += discount * immediateReward(state, next); discount *= GAMMA; state = next; steps++;
  }
  return ret;
}
function makeLearnerRolloutState(n, rng, steps = null) {
  let state = emptyState(n), count = steps == null ? Math.floor(rng() * Math.max(1, Math.min(2 * n, 24))) : steps;
  for (let t = 0; t < count; t++) {
    let i;
    if (state.assignedCount < n) { const rem=[]; for(let k=0;k<n;k++) if(!state.assigned[k]) rem.push(k); i=rem[Math.floor(rng()*rem.length)]; }
    else { if (isStrictChain(state.values)) break; const ids=violatedEndpointIds(state.values); i=ids[Math.floor(rng()*ids.length)]; }
    // Cheap learner-state collection uses proposal samples; full successor lookahead is used by the deployed policy.
    state = propagateSuccessor(state, i, chooseValue(state, i, rng, true, false, true, false));
  }
  return state;
}
function makeCriticState(n, rng, useOnPolicy) {
  let base = useOnPolicy ? makeLearnerRolloutState(n, rng) : makeSyntheticState(n, Math.floor(rng()*n), rng);
  let i;
  if (base.assignedCount < n) { const rem=[]; for(let k=0;k<n;k++) if(!base.assigned[k]) rem.push(k); i=rem[Math.floor(rng()*rem.length)]; }
  else { const ids=violatedEndpointIds(base.values); i=ids.length ? ids[Math.floor(rng()*ids.length)] : Math.floor(rng()*n); }
  const proposal = proposalSnapshot(base, i), v = clampValue(gaussian(proposal.mean, Math.max(1, proposal.sigma), rng));
  return propagateSuccessor(base, i, v);
}
function daggerRate(progress) { if (progress <= DAGGER_WARMUP_FRACTION) return 0; const t = (progress - DAGGER_WARMUP_FRACTION) / Math.max(1e-6, 1 - DAGGER_WARMUP_FRACTION); return DAGGER_MAX_RATE * Math.min(1, t); }
function makeTrainingSample(n, rng, useOnPolicy) {
  const repair = makeRepairTrainingState(n, rng), proposalVar = Math.floor(rng() * n), proposalState = useOnPolicy ? makeLearnerRolloutState(n, rng) : makeSyntheticState(n, proposalVar, rng), teacherValue = teacherSampleValue(proposalVar, rng), criticState = makeCriticState(n, rng, useOnPolicy), criticTarget = teacherContinuationReturn(criticState, rng);
  return { repairState: repair.state, candidateIds: repair.candidateIds, proposalState, proposalVar, teacherValue, criticState, criticTarget, onPolicy: useOnPolicy };
}

function runLearned(n, rng, stochastic = true, oracleVariable = false, oracleValue = false, proposalOnly = false, myopicOnly = false) {
  let state = emptyState(n), maxMoves = MOVE_MULTIPLIER * n, moves = 0;
  while (state.assignedCount < n && moves < maxMoves) {
    const remaining=[]; for(let i=0;i<n;i++) if(!state.assigned[i]) remaining.push(i); const i=stochastic?remaining[Math.floor(rng()*remaining.length)]:remaining[0], v=chooseValue(state,i,rng,stochastic,oracleValue,proposalOnly,myopicOnly); state=propagateSuccessor(state,i,v); moves++;
  }
  while (moves < maxMoves && !isStrictChain(state.values)) {
    let i; if (oracleVariable) { const ids=violatedEndpointIds(state.values); if(!ids.length) break; i=stochastic?ids[Math.floor(rng()*ids.length)]:ids[0]; } else i=chooseRepairVariable(state,rng,stochastic);
    state=propagateSuccessor(state,i,chooseValue(state,i,rng,stochastic,oracleValue,proposalOnly,myopicOnly)); moves++;
  }
  return { x:Int16Array.from(state.values), solved:isStrictChain(state.values), moves };
}
function runHandcrafted(n, rng) { let state=emptyState(n),maxMoves=MOVE_MULTIPLIER*n,moves=0;for(let i=0;i<n;i++){state=propagateSuccessor(state,i,teacherSampleValue(i,rng));moves++;}while(moves<maxMoves&&!isStrictChain(state.values)){const ids=violatedEndpointIds(state.values);if(!ids.length)break;const i=ids[Math.floor(rng()*ids.length)];state=propagateSuccessor(state,i,teacherSampleValue(i,rng));moves++;}return{x:Int16Array.from(state.values),solved:isStrictChain(state.values),moves};}
function randomAssignment(n,rng){return Int16Array.from({length:n},()=>DOMAIN_MIN+Math.floor(rng()*(DOMAIN_SPAN+1)));}
function runSA(n,evaluations,rng){const x=randomAssignment(n,rng);let e=violationEnergy(x),best=Int16Array.from(x),bestE=e;const T0=80,Tend=.02;for(let step=0;step<evaluations&&bestE>0;step++){const i=Math.floor(rng()*n),old=x[i];x[i]=DOMAIN_MIN+Math.floor(rng()*(DOMAIN_SPAN+1));const next=violationEnergy(x),delta=next-e,t=step/Math.max(1,evaluations-1),temp=T0*Math.pow(Tend/T0,t);if(delta<=0||rng()<Math.exp(-delta/Math.max(1e-6,temp))){e=next;if(e<bestE){bestE=e;best=Int16Array.from(x);}}else x[i]=old;}return{x:best,solved:bestE===0};}

function endpointDiagnostics(n,rng,states=20){let tp=0,fp=0,fn=0;for(let s=0;s<states;s++){const sample=makeRepairTrainingState(n,rng),positive=new Set(sample.candidateIds),logits=variableSnapshot(sample.state);for(let i=0;i<n;i++){const pred=sigmoidScalar(logits[i])>=VARIABLE_THRESHOLD;if(pred&&positive.has(i))tp++;else if(pred)fp++;else if(positive.has(i))fn++;}}const precision=tp/Math.max(1,tp+fp),recall=tp/Math.max(1,tp+fn),f1=2*precision*recall/Math.max(1e-9,precision+recall);return{precision,recall,f1};}
function valueDiagnostics(n,rng,states=6){let synthMae=0,onMae=0,criticMae=0,rankHit=0,myopicHit=0;for(let s=0;s<states;s++){const i=Math.floor(rng()*n),synth=makeSyntheticState(n,i,rng),on=makeLearnerRolloutState(n,rng),target=teacherMeanValue(i),p1=proposalSnapshot(synth,i),p2=proposalSnapshot(on,i);synthMae+=Math.abs(p1.mean-target);onMae+=Math.abs(p2.mean-target);const cs=makeCriticState(n,rng,true),actual=teacherContinuationReturn(cs,rng),pred=criticSnapshot(cs);criticMae+=Math.abs(pred-actual);let base=makeLearnerRolloutState(n,rng);let v;if(base.assignedCount<n){const rem=[];for(let k=0;k<n;k++)if(!base.assigned[k])rem.push(k);v=rem[Math.floor(rng()*rem.length)];}else{if(isStrictChain(base.values)){base=makeRepairTrainingState(n,rng).state;}const ids=violatedEndpointIds(base.values);v=ids[Math.floor(rng()*ids.length)];}const cand=generateValueCandidates(base,v,rng,false).values;let bestReturn=-Infinity,bestScore=-Infinity,bestImmediate=-Infinity,scorePick=-1,returnPick=-1,immediatePick=-1;for(let k=0;k<cand.length;k++){const next=propagateSuccessor(base,v,cand[k]),r=immediateReward(base,next),futureTarget=teacherContinuationReturn(next,rng),ret=r+GAMMA*futureTarget,score=r+GAMMA*(terminalUtility(next)?0:criticSnapshot(next));if(ret>bestReturn){bestReturn=ret;returnPick=k;}if(score>bestScore){bestScore=score;scorePick=k;}if(r>bestImmediate){bestImmediate=r;immediatePick=k;}}if(scorePick===returnPick)rankHit++;if(immediatePick===returnPick)myopicHit++;}return{synthMae:synthMae/states,onMae:onMae/states,criticMae:criticMae/states,rankHit:rankHit/states,myopicHit:myopicHit/states};}
function solveRate(fn,n,seed,trials=3){const rng=mulberry32(seed);let solved=0;for(let t=0;t<trials;t++)if(fn(n,rng,true).solved)solved++;return solved/trials;}

async function trainGeneric(){if(!window.tf)throw new Error('TensorFlow.js did not load.');MESSAGE_ROUNDS=+messageRounds.value;const startN=+trainMinN.value,maxN=+trainMaxN.value,totalStates=+episodes.value;trainedRange={min:startN,max:maxN};trained=false;trainBtn.disabled=true;benchmarkBtn.disabled=true;initModel();resetLog();logLine('GNN_SUCCESSOR_CRITIC_DIAGNOSTIC v1');logLine(`CONFIG backend=${tf.getBackend()} train=${startN}..${maxN} states=${totalStates} rounds=${MESSAGE_ROUNDS} embed=${EMBED_DIM} proposalSamples=${PROPOSAL_SAMPLES} randomCandidates=${RANDOM_CANDIDATES} gamma=${GAMMA} daggerWarmup=${DAGGER_WARMUP_FRACTION} daggerMax=${DAGGER_MAX_RATE} budget=${MOVE_MULTIPLIER}n`);const rng=mulberry32(20260823);let completed=0,updates=0,lastLoss=NaN;while(completed<totalStates){const progress=completed/Math.max(1,totalStates),rate=daggerRate(progress),batch=[],count=Math.min(BATCH_STATES,totalStates-completed);let onCount=0;for(let b=0;b<count;b++){const n=startN+Math.floor(rng()*(maxN-startN+1)),on=!!params&&rng()<rate;if(on)onCount++;batch.push(makeTrainingSample(n,rng,on));completed++;}lastLoss=await trainingUpdate(batch);updates++;if(updates%4===0||completed>=totalStates){const ed=endpointDiagnostics(maxN,mulberry32(810000+updates),16),vd=valueDiagnostics(maxN,mulberry32(820000+updates),4),solve=solveRate(runLearned,maxN,830000+updates,3);statusEl.textContent=`States ${completed}/${totalStates} · endpoint F1 ${(100*ed.f1).toFixed(0)}% · proposal MAE on ${vd.onMae.toFixed(1)} · critic MAE ${vd.criticMae.toFixed(2)} · successor rank ${(100*vd.rankHit).toFixed(0)}% · solve@${maxN} ${(100*solve).toFixed(0)}%`;logLine(`TRAIN states=${completed} update=${updates} loss=${lastLoss.toFixed(4)} daggerRate=${rate.toFixed(2)} onPolicyBatch=${onCount}/${count} P=${(100*ed.precision).toFixed(1)}% R=${(100*ed.recall).toFixed(1)}% F1=${(100*ed.f1).toFixed(1)}% proposalMAE_synth=${vd.synthMae.toFixed(2)} proposalMAE_on=${vd.onMae.toFixed(2)} criticMAE=${vd.criticMae.toFixed(3)} successorRankHit=${(100*vd.rankHit).toFixed(0)}% myopicRankHit=${(100*vd.myopicHit).toFixed(0)}% solve@n${maxN}=${(100*solve).toFixed(0)}%`);await tf.nextFrame();}}trained=true;trainBtn.disabled=false;benchmarkBtn.disabled=false;statusEl.textContent='Successor-state critic training complete. Running benchmark…';await runBenchmark();}
function benchmarkLengths(){const m=trainedRange.max;return[...new Set([trainedRange.min,m,Math.min(MAX_TEST_N,m+1),Math.min(MAX_TEST_N,Math.round(m*1.5)),Math.min(MAX_TEST_N,m*2),MAX_TEST_N])].sort((a,b)=>a-b);}
function benchmarkTrials(n){return n>=80?1:n>=40?3:6;}
function median(values){if(!values.length)return NaN;const a=[...values].sort((x,y)=>x-y);return a[Math.floor(a.length/2)];}
async function runBenchmark(){if(!trained||!params)return;trainBtn.disabled=true;benchmarkBtn.disabled=true;const rows=[],rng=mulberry32(20260824);let finalExamples=null;for(const n of benchmarkLengths()){const trials=benchmarkTrials(n);statusEl.textContent=`Benchmarking successor-state critic at n=${n} · ${trials} trial${trials===1?'':'s'}…`;let pSolved=0,hSolved=0,saSolved=0,proposalSolved=0,oracleVarSolved=0,oracleValueSolved=0,myopicSolved=0,pViol=0,hViol=0;const pMoves=[],hMoves=[];for(let t=0;t<trials;t++){const p=runLearned(n,rng,true,false,false,false,false);if(p.solved){pSolved++;pMoves.push(p.moves);}pViol+=violatedConstraintCount(p.x);const prop=runLearned(n,rng,true,false,false,true,false);if(prop.solved)proposalSolved++;if(n<80){const ov=runLearned(n,rng,true,true,false,false,false);if(ov.solved)oracleVarSolved++;}const oval=runLearned(n,rng,true,false,true,false,false);if(oval.solved)oracleValueSolved++;const my=runLearned(n,rng,true,false,false,false,true);if(my.solved)myopicSolved++;const h=runHandcrafted(n,rng);if(h.solved){hSolved++;hMoves.push(h.moves);}hViol+=violatedConstraintCount(h.x);const sa=runSA(n,120*n,rng);if(sa.solved)saSolved++;if(t===0&&n===MAX_TEST_N)finalExamples={p,h,sa};await tf.nextFrame();}rows.push({n,policySuccess:pSolved/trials,heuristicSuccess:hSolved/trials,saSuccess:saSolved/trials,policyMoves:median(pMoves),heuristicMoves:median(hMoves)});logLine(`BENCH n=${n} trials=${trials} gnn=${Math.round(100*pSolved/trials)}% proposalOnly=${Math.round(100*proposalSolved/trials)}% oracleVariable=${n<80?`${Math.round(100*oracleVarSolved/trials)}%`:'skip'} oracleValue=${Math.round(100*oracleValueSolved/trials)}% myopicOnly=${Math.round(100*myopicSolved/trials)}% teacher=${Math.round(100*hSolved/trials)}% sa=${Math.round(100*saSolved/trials)}% gnnAvgViol=${(pViol/trials).toFixed(2)} teacherAvgViol=${(hViol/trials).toFixed(2)}`);}drawScaling(rows);const longest=rows[rows.length-1],n=longest.n;if(!finalExamples)finalExamples={p:runLearned(n,mulberry32(9101),true),h:runHandcrafted(n,mulberry32(9102)),sa:runSA(n,120*n,mulberry32(9103))};drawAssignment(finalExamples.p.x,finalExamples.h.x,finalExamples.sa.x);const ed=endpointDiagnostics(trainedRange.max,mulberry32(9200),24),vd=valueDiagnostics(trainedRange.max,mulberry32(9201),6);$('policySuccess').textContent=`${Math.round(100*longest.policySuccess)}%`;$('heuristicSuccess').textContent=`${Math.round(100*longest.heuristicSuccess)}%`;$('saSuccess').textContent=`${Math.round(100*longest.saSuccess)}%`;$('candidateMass').textContent=`${Math.round(100*ed.f1)}%`;$('meanMae').textContent=vd.criticMae.toFixed(2);$('policyMoves').textContent=Number.isFinite(longest.policyMoves)?Math.round(longest.policyMoves):'—';$('heuristicMoves').textContent=Number.isFinite(longest.heuristicMoves)?Math.round(longest.heuristicMoves):'—';$('saEffort').textContent=(120*n).toLocaleString();$('metricLength').textContent=`n=${n} · propose → apply/propagate → immediate reward + γV(successor) · train ${trainedRange.min}…${trainedRange.max} · depth ${MESSAGE_ROUNDS}`;appendProbe(trainedRange.max);appendTrace(trainedRange.max);logLine('END');statusEl.textContent='Done. Candidate actions are evaluated on their actual successor graphs.';trainBtn.disabled=false;benchmarkBtn.disabled=false;}
function appendProbe(n){let state=makeLearnerRolloutState(n,mulberry32(9300+n));let i;if(state.assignedCount<n){const rem=[];for(let k=0;k<n;k++)if(!state.assigned[k])rem.push(k);i=rem[0];}else{if(isStrictChain(state.values))state=makeRepairTrainingState(n,mulberry32(9301+n)).state;const ids=violatedEndpointIds(state.values);i=ids[0];}const generated=generateValueCandidates(state,i,mulberry32(9302+n),false),rows=[];for(const v of generated.values){const next=propagateSuccessor(state,i,v),r=immediateReward(state,next),future=terminalUtility(next)?0:criticSnapshot(next);rows.push({v,r,future,score:r+GAMMA*future});}rows.sort((a,b)=>b.score-a.score);logLine(`PROBE_SUCCESSOR n=${n} variable=${i} phase=${state.assignedCount<n?'construct':'repair'} objective=${solverObjective(state).toFixed(3)} top=[${rows.slice(0,8).map(x=>`${x.v}:r${x.r.toFixed(2)}/V${x.future.toFixed(2)}/q${x.score.toFixed(2)}`).join(',')}]`);}
function appendTrace(n){const rng=mulberry32(9400+n);let state=emptyState(n),moves=0,maxMoves=MOVE_MULTIPLIER*n;for(let step=0;step<16&&moves<maxMoves;step++){let i,phase;if(state.assignedCount<n){const rem=[];for(let k=0;k<n;k++)if(!state.assigned[k])rem.push(k);i=rem[Math.floor(rng()*rem.length)];phase='construct';}else{if(isStrictChain(state.values))break;i=chooseRepairVariable(state,rng,true);phase='repair';}const candidates=generateValueCandidates(state,i,rng,false).values;let best=null;for(const v of candidates){const next=propagateSuccessor(state,i,v),r=immediateReward(state,next),future=terminalUtility(next)?0:criticSnapshot(next),q=r+GAMMA*future;if(!best||q>best.q)best={v,next,r,future,q};}const beforeObj=solverObjective(state);state=best.next;moves++;logLine(`TRACE step=${step} phase=${phase} i=${i} v=${best.v} immediate=${best.r.toFixed(3)} Vnext=${best.future.toFixed(3)} q=${best.q.toFixed(3)} objective=${beforeObj.toFixed(3)}->${solverObjective(state).toFixed(3)} assigned=${state.assignedCount}/${n}${state.assignedCount===n?` viol=${violatedConstraintCount(state.values)} energy=${violationEnergy(state.values)}`:''}`);}logLine(`TRACE_END solved=${state.assignedCount===n&&isStrictChain(state.values)?1:0} movesShown=${moves} assigned=${state.assignedCount}/${n}${state.assignedCount===n?` violations=${violatedConstraintCount(state.values)} energy=${violationEnergy(state.values)}`:''}`);}

function setupCanvas(canvas){const dpr=window.devicePixelRatio||1,rect=canvas.getBoundingClientRect();canvas.width=Math.max(300,Math.floor(rect.width*dpr));canvas.height=Math.floor(Math.max(260,rect.width*.46)*dpr);const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);return{ctx,w:canvas.width/dpr,h:canvas.height/dpr};}
function axes(ctx,w,h,pad,xl,yl){ctx.clearRect(0,0,w,h);ctx.strokeStyle='#dbe1eb';ctx.beginPath();ctx.moveTo(pad,14);ctx.lineTo(pad,h-pad);ctx.lineTo(w-12,h-pad);ctx.stroke();ctx.fillStyle='#697386';ctx.font='12px system-ui';ctx.fillText(yl,pad+6,24);ctx.textAlign='right';ctx.fillText(xl,w-12,h-10);ctx.textAlign='left';}
function line(ctx,pts,color,width=2.4){if(!pts.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();ctx.fillStyle=color;for(const p of pts){ctx.beginPath();ctx.arc(p[0],p[1],2.6,0,Math.PI*2);ctx.fill();}}
function drawScaling(rows){const{ctx,w,h}=setupCanvas($('scalingChart')),pad=44;axes(ctx,w,h,pad,'chain length n','solve rate');const lo=Math.min(...rows.map(r=>r.n)),hi=Math.max(...rows.map(r=>r.n)),X=n=>pad+(n-lo)/Math.max(1,hi-lo)*(w-pad-20),Y=p=>h-pad-p*(h-pad-28);ctx.fillStyle='#7b8495';ctx.font='11px system-ui';for(const r of rows)ctx.fillText(String(r.n),X(r.n)-6,h-pad+17);for(let k=0;k<=4;k++)ctx.fillText(`${25*k}%`,5,Y(k/4)+4);const bx=X(trainedRange.max);ctx.save();ctx.setLineDash([5,5]);ctx.strokeStyle='#a9b1c2';ctx.beginPath();ctx.moveTo(bx,18);ctx.lineTo(bx,h-pad);ctx.stroke();ctx.restore();ctx.fillText('train max',Math.min(w-80,bx+5),28);line(ctx,rows.map(r=>[X(r.n),Y(r.policySuccess)]),'#5b67d6');line(ctx,rows.map(r=>[X(r.n),Y(r.heuristicSuccess)]),'#2e8b72');line(ctx,rows.map(r=>[X(r.n),Y(r.saSuccess)]),'#dd6b55');}
function drawAssignment(policy,teacher,sa){const{ctx,w,h}=setupCanvas($('instanceChart')),pad=44;axes(ctx,w,h,pad,'variable node id i','assigned value');const n=policy.length,X=i=>pad+i/Math.max(1,n-1)*(w-pad-20),Y=v=>h-pad-Math.max(0,v)/DOMAIN_MAX*(h-pad-28);line(ctx,Array.from(policy,(v,i)=>[X(i),Y(v)]),'#5b67d6',2.2);line(ctx,Array.from(teacher,(v,i)=>[X(i),Y(v)]),'#2e8b72',2);line(ctx,Array.from(sa,(v,i)=>[X(i),Y(v)]),'#dd6b55',1.8);}
trainBtn.addEventListener('click',()=>trainGeneric().catch(err=>{console.error(err);statusEl.textContent=`Error: ${err.message}`;logLine(`ERROR ${err.stack||err.message}`);trainBtn.disabled=false;benchmarkBtn.disabled=!trained;}));
benchmarkBtn.addEventListener('click',()=>runBenchmark().catch(err=>{console.error(err);statusEl.textContent=`Error: ${err.message}`;logLine(`ERROR ${err.stack||err.message}`);trainBtn.disabled=false;benchmarkBtn.disabled=false;}));
statusEl.textContent=`TensorFlow.js ready · backend: ${tf.getBackend()}. Successor-state critic: apply candidate, update graph, score immediate reward + γV(next).`;
