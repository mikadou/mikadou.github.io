// Minimal REINFORCE fine-tuning layered on top of app-gnn-simple-imitation.js.
// The base script is loaded as a classic script so these bindings share one global scope.

const RL_STEP_COST = 0.05;
const RL_SUCCESS_BONUS = 1.0;
const RL_FAILURE_PENALTY = 1.0;
const RL_NODE_TEMPERATURE = 0.75;
const RL_VALUE_SIGMA = 1.0;
const RL_VALUE_LOGPROB_WEIGHT = 0.02;
const RL_ENTROPY_BETA = 0.001;
const RL_LEARNING_RATE = 0.0001;
const RL_BASELINE_RATE = 0.05;
const RL_LOG_EVERY = 50;

let rlOptimizer = null;
let rlBaseline = 0;
let rlEpisodesInput = null;
const baseTrainImitation = trainImitation;
const baseRunBenchmark = runBenchmark;

function rlCloneState(state) {
  return {
    n: state.n,
    assigned: Uint8Array.from(state.assigned),
    values: Int16Array.from(state.values),
    assignedCount: state.assignedCount
  };
}

function sampleCategorical(logits, temperature, rng) {
  let max = -Infinity;
  for (const x of logits) max = Math.max(max, x);
  const weights = new Float64Array(logits.length);
  let total = 0;
  const invT = 1 / Math.max(0.1, temperature);
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

function repairCandidateLogitsSnapshot(state, candidateIds) {
  const snap = endpointSnapshot(state);
  return Float32Array.from(candidateIds, i => snap.endpointLogits[i]);
}

function sampleRlRepairAction(state, rng) {
  const candidateIds = violatedEndpointIds(state.values);
  if (!candidateIds.length) return null;
  const logits = repairCandidateLogitsSnapshot(state, candidateIds);
  const chosenK = sampleCategorical(logits, RL_NODE_TEMPERATURE, rng);
  const variableId = candidateIds[chosenK];
  const meanValue = valueMeanSnapshot(state, variableId);
  const rawValue = gaussian(meanValue, RL_VALUE_SIGMA, rng);
  return {
    variableId,
    rawValue,
    appliedValue: clampValue(rawValue)
  };
}

function constructWithCurrentPolicy(n, rng, stochastic = true) {
  const state = emptyState(n);
  while (state.assignedCount < n) {
    const remaining = [];
    for (let i = 0; i < n; i++) if (!state.assigned[i]) remaining.push(i);
    const i = stochastic ? remaining[Math.floor(rng() * remaining.length)] : remaining[0];
    applyAction(state, i, learnedValue(state, i, rng, stochastic, false).v);
  }
  return state;
}

function rolloutRlRepair(n, rng) {
  const state = constructWithCurrentPolicy(n, rng, true);
  const trajectory = [];
  const maxRepairMoves = 2 * n;

  for (let step = 0; step < maxRepairMoves && !isStrictChain(state.values); step++) {
    const action = sampleRlRepairAction(state, rng);
    if (!action) break;
    const before = violationEnergy(state.values);
    const stateBefore = rlCloneState(state);
    applyAction(state, action.variableId, action.appliedValue);
    const after = violationEnergy(state.values);
    let reward = before - after - RL_STEP_COST;
    if (isStrictChain(state.values)) reward += RL_SUCCESS_BONUS;
    trajectory.push({
      state: stateBefore,
      variableId: action.variableId,
      rawValue: action.rawValue,
      reward
    });
  }

  const solved = isStrictChain(state.values);
  if (!solved && trajectory.length) trajectory[trajectory.length - 1].reward -= RL_FAILURE_PENALTY;

  const returns = new Float32Array(trajectory.length);
  let g = 0;
  for (let t = trajectory.length - 1; t >= 0; t--) {
    g = trajectory[t].reward + g;
    returns[t] = g;
  }
  return {
    x: Int16Array.from(state.values),
    solved,
    repairMoves: trajectory.length,
    moves: n + trajectory.length,
    trajectory,
    returns,
    totalReturn: trajectory.length ? returns[0] : (solved ? RL_SUCCESS_BONUS : 0)
  };
}

function nodeLogProbAndEntropyTensor(state, variableId) {
  const candidateIds = violatedEndpointIds(state.values);
  const p = endpointTensors(state);
  const ids = tf.tensor1d(Int32Array.from(candidateIds), 'int32');
  const logits = tf.gather(p.endpointLogits, ids).div(RL_NODE_TEMPERATURE);
  const logProbs = tf.logSoftmax(logits);
  const probs = tf.softmax(logits);
  let chosen = candidateIds.indexOf(variableId);
  if (chosen < 0) chosen = 0;
  return {
    logProb: logProbs.gather(chosen),
    entropy: probs.mul(logProbs).sum().neg()
  };
}

function valueLogProbTensor(state, variableId, rawValue) {
  const mean = valueTensor(state, variableId).mul(VALUE_SCALE);
  const z = tf.scalar(rawValue).sub(mean).div(RL_VALUE_SIGMA);
  return z.square().mul(-0.5);
}

async function reinforceRepairEpisode(episode) {
  if (!episode.trajectory.length) return NaN;
  const meanReturn = episode.returns.reduce((a, b) => a + b, 0) / episode.returns.length;
  const baselineBefore = rlBaseline;
  rlBaseline = (1 - RL_BASELINE_RATE) * rlBaseline + RL_BASELINE_RATE * meanReturn;

  const cost = rlOptimizer.minimize(() => tf.tidy(() => {
    let total = tf.scalar(0);
    for (let t = 0; t < episode.trajectory.length; t++) {
      const step = episode.trajectory[t];
      const advantage = episode.returns[t] - baselineBefore;
      const node = nodeLogProbAndEntropyTensor(step.state, step.variableId);
      const valueLogProb = valueLogProbTensor(step.state, step.variableId, step.rawValue);
      const joint = node.logProb.add(valueLogProb.mul(RL_VALUE_LOGPROB_WEIGHT));
      const policyLoss = joint.mul(-advantage);
      total = total.add(policyLoss).add(node.entropy.mul(-RL_ENTROPY_BETA));
    }
    return total.div(episode.trajectory.length);
  }), true, allParams());

  const loss = cost.dataSync()[0];
  cost.dispose();
  return loss;
}

function runRlPolicy(n, rng, stochastic = true) {
  const state = constructWithCurrentPolicy(n, rng, stochastic);
  const maxRepairMoves = 2 * n;
  let repairMoves = 0;
  while (repairMoves < maxRepairMoves && !isStrictChain(state.values)) {
    const candidateIds = violatedEndpointIds(state.values);
    if (!candidateIds.length) break;
    const logits = repairCandidateLogitsSnapshot(state, candidateIds);
    let chosenK = 0;
    if (stochastic) chosenK = sampleCategorical(logits, RL_NODE_TEMPERATURE, rng);
    else for (let k = 1; k < logits.length; k++) if (logits[k] > logits[chosenK]) chosenK = k;
    const i = candidateIds[chosenK];
    applyAction(state, i, learnedValue(state, i, rng, stochastic, false).v);
    repairMoves++;
  }
  return {
    x: Int16Array.from(state.values),
    solved: isStrictChain(state.values),
    repairMoves,
    moves: n + repairMoves
  };
}

function evaluateMoves(policyFn, n, seed, trials = 24) {
  const rng = mulberry32(seed);
  let solved = 0, totalMoves = 0, totalRepair = 0, violations = 0;
  const solvedMoves = [];
  for (let t = 0; t < trials; t++) {
    const r = policyFn(n, rng, true);
    if (r.solved) { solved++; solvedMoves.push(r.moves); }
    totalMoves += r.moves;
    totalRepair += Math.max(0, r.moves - n);
    violations += violatedConstraintCount(r.x);
  }
  return {
    solveRate: solved / trials,
    avgMoves: totalMoves / trials,
    avgRepair: totalRepair / trials,
    medianSolvedMoves: median(solvedMoves),
    avgViol: violations / trials
  };
}

async function runRlFineTune(startN, maxN, totalEpisodes) {
  if (rlOptimizer?.dispose) rlOptimizer.dispose();
  rlOptimizer = tf.train.adam(RL_LEARNING_RATE);
  rlBaseline = 0;
  const rng = mulberry32(20260830);
  const recent = [];
  let lastLoss = NaN;

  logLine(`RL_START episodes=${totalEpisodes} reward=energyDelta-${RL_STEP_COST}+terminal${RL_SUCCESS_BONUS} failurePenalty=${RL_FAILURE_PENALTY} nodeTemp=${RL_NODE_TEMPERATURE} valueSigma=${RL_VALUE_SIGMA} valueLogProbWeight=${RL_VALUE_LOGPROB_WEIGHT} lr=${RL_LEARNING_RATE} critic=0`);

  for (let ep = 1; ep <= totalEpisodes; ep++) {
    const n = startN + Math.floor(rng() * (maxN - startN + 1));
    const episode = rolloutRlRepair(n, rng);
    lastLoss = await reinforceRepairEpisode(episode);
    recent.push({ solved: episode.solved ? 1 : 0, repair: episode.repairMoves, ret: episode.totalReturn });
    if (recent.length > 50) recent.shift();

    if (ep % RL_LOG_EVERY === 0 || ep === totalEpisodes) {
      const avgSolved = recent.reduce((s, x) => s + x.solved, 0) / Math.max(1, recent.length);
      const avgRepair = recent.reduce((s, x) => s + x.repair, 0) / Math.max(1, recent.length);
      const avgReturn = recent.reduce((s, x) => s + x.ret, 0) / Math.max(1, recent.length);
      const probe = evaluateMoves(runRlPolicy, maxN, 850000 + ep, 12);
      statusEl.textContent = `RL ${ep}/${totalEpisodes} · recent solve ${(100*avgSolved).toFixed(0)}% · repair ${avgRepair.toFixed(1)} · probe@${maxN} ${(100*probe.solveRate).toFixed(0)}% / ${probe.avgRepair.toFixed(1)} repair moves`;
      logLine(`RL ep=${ep} loss=${Number.isFinite(lastLoss)?lastLoss.toFixed(4):'nan'} recentSolve=${(100*avgSolved).toFixed(0)}% recentRepair=${avgRepair.toFixed(2)} recentReturn=${avgReturn.toFixed(3)} baseline=${rlBaseline.toFixed(3)} probeSolve@n${maxN}=${(100*probe.solveRate).toFixed(0)}% probeRepair=${probe.avgRepair.toFixed(2)}`);
      await tf.nextFrame();
    }
  }
}

async function runRlBenchmark() {
  if (!trained || !params) return;
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;
  const rows = [], rng = mulberry32(20260831);
  let finalExamples = null;

  for (const n of benchmarkLengths()) {
    const trials = benchmarkTrials(n);
    let pSolved = 0, hSolved = 0, saSolved = 0, pViol = 0, hViol = 0;
    const pMoves = [], hMoves = [];
    for (let t = 0; t < trials; t++) {
      const p = runRlPolicy(n, rng, true);
      if (p.solved) { pSolved++; pMoves.push(p.moves); }
      pViol += violatedConstraintCount(p.x);
      const h = runHandcrafted(n, rng);
      if (h.solved) { hSolved++; hMoves.push(h.moves); }
      hViol += violatedConstraintCount(h.x);
      const sa = runSA(n, 120*n, rng); if (sa.solved) saSolved++;
      if (t === 0 && n === MAX_TEST_N) finalExamples = { p, h, sa };
      await tf.nextFrame();
    }
    rows.push({ n, policySuccess:pSolved/trials, heuristicSuccess:hSolved/trials, saSuccess:saSolved/trials, policyMoves:median(pMoves), heuristicMoves:median(hMoves) });
    logLine(`RL_BENCH n=${n} trials=${trials} rl=${Math.round(100*pSolved/trials)}% teacher=${Math.round(100*hSolved/trials)}% sa=${Math.round(100*saSolved/trials)}% rlMedianMoves=${Number.isFinite(median(pMoves))?median(pMoves):'na'} teacherMedianMoves=${Number.isFinite(median(hMoves))?median(hMoves):'na'} rlAvgViol=${(pViol/trials).toFixed(2)} teacherAvgViol=${(hViol/trials).toFixed(2)}`);
  }

  drawScaling(rows);
  const longest = rows[rows.length - 1], n = longest.n;
  if (!finalExamples) finalExamples = { p:runRlPolicy(n,mulberry32(9601),true), h:runHandcrafted(n,mulberry32(9602)), sa:runSA(n,120*n,mulberry32(9603)) };
  drawAssignment(finalExamples.p.x, finalExamples.h.x, finalExamples.sa.x);
  const ed = endpointDiagnostics(trainedRange.max, mulberry32(9700), 64), vd = valueDiagnostics(trainedRange.max, mulberry32(9701), 96);
  $('policySuccess').textContent = `${Math.round(100*longest.policySuccess)}%`;
  $('heuristicSuccess').textContent = `${Math.round(100*longest.heuristicSuccess)}%`;
  $('saSuccess').textContent = `${Math.round(100*longest.saSuccess)}%`;
  $('candidateMass').textContent = `${Math.round(100*ed.f1)}%`;
  $('meanMae').textContent = vd.repairMae.toFixed(2);
  $('policyMoves').textContent = Number.isFinite(longest.policyMoves) ? Math.round(longest.policyMoves) : '—';
  $('heuristicMoves').textContent = Number.isFinite(longest.heuristicMoves) ? Math.round(longest.heuristicMoves) : '—';
  $('saEffort').textContent = (120*n).toLocaleString();
  $('metricLength').textContent = `n=${n} · imitation initialization + REINFORCE repair fine-tuning · no critic · train ${trainedRange.min}…${trainedRange.max}`;
  logLine('RL_END');
  statusEl.textContent = 'Done. Compare the earlier BENCH lines (behavioral cloning) with RL_BENCH lines (after REINFORCE).';
  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
}

async function trainBcThenRl() {
  // Leave runBenchmark pointing at the original benchmark while BC trains so we get a clean baseline.
  await baseTrainImitation();
  const startN = +trainMinN.value, maxN = +trainMaxN.value;
  const rlEpisodes = +rlEpisodesInput.value;
  const bcEval = evaluateMoves(runImitation, maxN, 20260901, 32);
  logLine(`BC_BASELINE n=${maxN} trials=32 solve=${(100*bcEval.solveRate).toFixed(0)}% avgMoves=${bcEval.avgMoves.toFixed(2)} avgRepair=${bcEval.avgRepair.toFixed(2)} medianSolvedMoves=${Number.isFinite(bcEval.medianSolvedMoves)?bcEval.medianSolvedMoves:'na'}`);

  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;
  statusEl.textContent = 'Behavioral cloning complete. Starting REINFORCE repair fine-tuning…';
  await runRlFineTune(startN, maxN, rlEpisodes);

  trained = true;
  runBenchmark = runRlBenchmark;
  const rlEval = evaluateMoves(runRlPolicy, maxN, 20260902, 32);
  logLine(`RL_FINAL n=${maxN} trials=32 solve=${(100*rlEval.solveRate).toFixed(0)}% avgMoves=${rlEval.avgMoves.toFixed(2)} avgRepair=${rlEval.avgRepair.toFixed(2)} medianSolvedMoves=${Number.isFinite(rlEval.medianSolvedMoves)?rlEval.medianSolvedMoves:'na'}`);
  await runRlBenchmark();
}

function installRlUi() {
  const controls = document.querySelector('.controls');
  const messageBlock = messageRounds?.closest('div');
  const wrap = document.createElement('div');
  wrap.innerHTML = `<label>RL fine-tune episodes <span id="rlEpisodesLabel">1000</span></label><input id="rlEpisodes" type="range" min="100" max="5000" step="100" value="1000" />`;
  if (messageBlock && messageBlock.parentNode === controls) controls.insertBefore(wrap, messageBlock);
  else controls?.insertBefore(wrap, controls.querySelector('.buttons'));
  rlEpisodesInput = $('rlEpisodes');
  const label = $('rlEpisodesLabel');
  const update = () => { if (label) label.textContent = rlEpisodesInput.value; };
  rlEpisodesInput.addEventListener('input', update); update();

  trainBtn.textContent = 'Train imitation + RL';
  document.title = 'GNN Imitation + REINFORCE Solver';
  const h1 = document.querySelector('h1');
  if (h1) h1.textContent = 'Imitate the heuristic, then improve it with solver reward';
  const lede = document.querySelector('.lede');
  if (lede) lede.innerHTML = 'Phase 1 behaviorally clones the working repair heuristic. Phase 2 keeps the same GNN repair-node head and numeric value MLP, and fine-tunes them with REINFORCE using violation-energy improvement and a small per-move cost. There is no learned critic.';

  const footer = document.querySelector('footer');
  if (footer && !$('rlExplanation')) {
    const section = document.createElement('section');
    section.id = 'rlExplanation'; section.className = 'card explanation';
    section.innerHTML = `<h2>Minimal RL experiment</h2><p>Behavioral cloning is benchmarked first. REINFORCE then trains only on repair trajectories. At each repair step the solver supplies the currently violated endpoint candidates; the GNN chooses which one to repair and the value MLP samples the replacement value. Reward is <code>violationEnergy(before) − violationEnergy(after) − ${RL_STEP_COST}</code>, plus a terminal success bonus. A scalar running baseline reduces variance; there is no critic.</p><p>Because imitation already solves the trained sizes, the main question is whether RL lowers repair moves without sacrificing solve rate. Compare <code>BC_BASELINE</code>/<code>BENCH</code> with <code>RL_FINAL</code>/<code>RL_BENCH</code> in the diagnostic log.</p>`;
    footer.parentNode.insertBefore(section, footer);
  }
}

installRlUi();
trainImitation = trainBcThenRl;
statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Behavioral cloning followed by minimal REINFORCE fine-tuning; no critic.`;
