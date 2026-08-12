// GNN imitation-learning experiment for the handcrafted local-repair heuristic.
// The base typed factor-graph implementation is retained, but PPO training is
// bypassed. Constraint nodes receive dynamic satisfaction state, and supervised
// distribution matching asks whether the GNN can represent the teacher policy.

function replaceOnce(source, needle, replacement, label) {
  if (!source.includes(needle)) throw new Error(`GNN imitation patch failed: ${label}`);
  return source.replace(needle, replacement);
}

async function bootImitation() {
  const response = await fetch('./app-gnn-ppo-hybrid.js?v=20260811-22-base', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load GNN base engine (${response.status}).`);
  let source = await response.text();

  source = replaceOnce(source, 'const NODE_FEATURE_DIM = 5;', 'const NODE_FEATURE_DIM = 8;', 'feature dimension');
  source = replaceOnce(source, 'const EMBED_DIM = 24;', 'const EMBED_DIM = 16;', 'embedding size');
  source = replaceOnce(source, 'const MESSAGE_ROUNDS = 6;', 'let MESSAGE_ROUNDS = 1;', 'message depth');
  source = replaceOnce(source, 'optimizer = tf.train.adam(0.0008);', 'optimizer = tf.train.adam(0.002);', 'imitation learning rate');
  source = replaceOnce(
    source,
    "const mu = tf.matMul(variableH, params.Wmu).add(params.bmu).reshape([state.n]).clipByValue(-6, 6);",
    "const mu = tf.matMul(variableH, params.Wmu).add(params.bmu).reshape([state.n]).clipByValue(-12, 6);",
    'mean range'
  );

  source = replaceOnce(
    source,
    "const statusEl = $('status');",
    "const statusEl = $('status');\nconst messageRounds = $('messageRounds');",
    'message-depth control'
  );

  const oldGraphFeatures = `  for (let i = 0; i < n; i++) {
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
  }`;

  const imitationGraphFeatures = `  // Static features:
  // [variable-type, constraint-type, absolute-position, relative-position,
  //  problem-size, assigned, assigned-value, constraint-status].
  const absoluteScale = Math.max(1, MAX_TEST_N - 1);
  const sizeFeature = n / MAX_TEST_N;

  for (let i = 0; i < n; i++) {
    const p = i * NODE_FEATURE_DIM;
    baseFeatures[p] = 1;
    baseFeatures[p + 1] = 0;
    baseFeatures[p + 2] = i / absoluteScale;
    baseFeatures[p + 3] = n <= 1 ? 0 : i / (n - 1);
    baseFeatures[p + 4] = sizeFeature;
  }
  for (let k = 0; k < constraintCount; k++) {
    const node = n + k;
    const p = node * NODE_FEATURE_DIM;
    baseFeatures[p] = 0;
    baseFeatures[p + 1] = 1;
    baseFeatures[p + 2] = k / absoluteScale;
    baseFeatures[p + 3] = constraintCount <= 1 ? 0 : k / (constraintCount - 1);
    baseFeatures[p + 4] = sizeFeature;
  }`;

  source = replaceOnce(source, oldGraphFeatures, imitationGraphFeatures, 'static graph features');

  const oldNodeFeatureData = `function nodeFeatureData(state) {
  const spec = graphSpec(state.n);
  const data = Float32Array.from(spec.baseFeatures);
  for (let i = 0; i < state.n; i++) {
    const p = i * NODE_FEATURE_DIM;
    data[p + 3] = state.assigned[i] ? 1 : 0;
    data[p + 4] = state.assigned[i] ? state.values[i] / DOMAIN_MAX : 0;
  }
  return data;
}`;

  const imitationNodeFeatureData = `function nodeFeatureData(state) {
  const spec = graphSpec(state.n);
  const data = Float32Array.from(spec.baseFeatures);

  for (let i = 0; i < state.n; i++) {
    const p = i * NODE_FEATURE_DIM;
    data[p + 5] = state.assigned[i] ? 1 : 0;
    data[p + 6] = state.assigned[i] ? state.values[i] / DOMAIN_MAX : 0;
  }

  // Constraint status is observable solver state, not an action label:
  // +1 = violated, -1 = satisfied, 0 = unresolved.
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
}`;

  source = replaceOnce(source, oldNodeFeatureData, imitationNodeFeatureData, 'dynamic constraint status');

  source = replaceOnce(
    source,
    'function graphEmbeddings(state) {',
    `function rowLayerNorm(x) {
  const mean = x.mean(1, true);
  const centered = x.sub(mean);
  const variance = centered.square().mean(1, true);
  return centered.div(variance.add(1e-5).sqrt());
}

function graphEmbeddings(state) {`,
    'row normalization helper'
  );
  source = replaceOnce(
    source,
    '    h = tf.relu(h.add(z.add(params.brel)));',
    '    h = tf.relu(rowLayerNorm(h.add(z.add(params.brel))));',
    'normalized message update'
  );

  source = replaceOnce(source, 'async function trainRL() {', 'async function trainPPOUnused() {', 'disable PPO trainer');
  source = replaceOnce(source, 'async function runBenchmark() {', 'async function runBenchmarkUnused() {', 'disable old benchmark');
  source = replaceOnce(source, 'function drawScaling(rows) {', 'function drawScalingUnused(rows) {', 'disable old scaling plot');
  source = replaceOnce(source, 'function drawAssignment(policy, sa) {', 'function drawAssignmentUnused(policy, sa) {', 'disable old assignment plot');

  source += `

// ---------------------------------------------------------------------------
// Handcrafted teacher + imitation learner
// ---------------------------------------------------------------------------
const TEACHER_SIGMA = 1.0;
const IMITATION_BATCH_EPISODES = 16;
const IMITATION_SAMPLES_PER_EPISODE = 4;
const VALUE_MEAN_COEF = 12.0;
const VALUE_SIGMA_COEF = 0.15;

function teacherGaussian(mean, sigma, rng) {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + sigma * z;
}

function teacherMeanValue(i) {
  return Math.max(0, Math.min(DOMAIN_MAX, 2 * i));
}

function teacherSampleValue(i, rng) {
  return Math.max(0, Math.min(DOMAIN_MAX,
    Math.round(teacherGaussian(teacherMeanValue(i), TEACHER_SIGMA, rng))
  ));
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
  for (let i = 0; i < marked.length; i++) if (marked[i]) ids.push(i);
  return ids;
}

function teacherCandidateIds(state) {
  if (state.assignedCount < state.n) {
    const ids = [];
    for (let i = 0; i < state.n; i++) if (!state.assigned[i]) ids.push(i);
    return ids;
  }
  return violatedEndpointIds(state.values);
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

function pickTeacherSamples(trajectory, rng) {
  const out = [];
  const take = (arr, count) => {
    if (!arr.length || count <= 0) return;
    const chosen = new Set();
    while (chosen.size < Math.min(count, arr.length)) chosen.add(Math.floor(rng() * arr.length));
    for (const idx of chosen) out.push(arr[idx]);
  };

  take(trajectory.repair, Math.ceil(IMITATION_SAMPLES_PER_EPISODE / 2));
  take(trajectory.construction, IMITATION_SAMPLES_PER_EPISODE - out.length);
  if (out.length < IMITATION_SAMPLES_PER_EPISODE) {
    take(trajectory.repair, IMITATION_SAMPLES_PER_EPISODE - out.length);
  }
  return out;
}

function imitationTrainableParams() {
  return [
    params.Winit, params.binit, params.Wself, params.brel, ...params.Wrel,
    params.Wvar, params.bvar, params.Wmu, params.bmu,
    params.WlogStd, params.blogStd
  ];
}

function imitationSampleLoss(sample) {
  const p = policyTensors(sample.state);
  const ids = tf.tensor1d(sample.candidateIds, 'int32');

  const logPolicy = tf.logSoftmax(p.varLogits);
  const variableLoss = tf.gather(logPolicy, ids).mean().neg();

  const candidateMu = tf.gather(p.mu, ids);
  const predictedMean = tf.sigmoid(candidateMu);
  const targetMean = ids.toFloat().mul(2 / DOMAIN_MAX);
  const meanLoss = predictedMean.sub(targetMean).square().mean().mul(VALUE_MEAN_COEF);

  // Reinterpret logStd directly in value units for imitation: teacher sigma=1.
  const candidateLogStd = tf.gather(p.logStd, ids);
  const sigmaLoss = candidateLogStd.square().mean().mul(VALUE_SIGMA_COEF);

  return variableLoss.add(meanLoss).add(sigmaLoss);
}

async function imitationUpdate(samples) {
  if (!samples.length) return NaN;
  const cost = optimizer.minimize(() => tf.tidy(() => {
    let total = tf.scalar(0);
    for (const sample of samples) total = total.add(imitationSampleLoss(sample));
    return total.div(samples.length);
  }), true, imitationTrainableParams());
  const value = cost.dataSync()[0];
  cost.dispose();
  await tf.nextFrame();
  return value;
}

function imitationAction(state, rng, stochastic = true) {
  const snap = policySnapshot(state);
  const all = Array.from({ length: state.n }, (_, i) => i);
  let i;
  if (stochastic) {
    i = sampleCategorical(snap.varLogits, all, rng);
  } else {
    i = 0;
    for (let j = 1; j < state.n; j++) if (snap.varLogits[j] > snap.varLogits[i]) i = j;
  }

  const meanValue = DOMAIN_MAX * sigmoidScalar(snap.mu[i]);
  const sigmaValue = Math.exp(snap.logStd[i]);
  const sampled = stochastic ? teacherGaussian(meanValue, sigmaValue, rng) : meanValue;
  const v = Math.max(0, Math.min(DOMAIN_MAX, Math.round(sampled)));
  return { i, v, meanValue, sigmaValue };
}

function runImitation(n, rng, stochastic = true) {
  const state = emptyState(n);
  const maxMoves = MOVE_MULTIPLIER * n;
  let moves = 0;

  while (moves < maxMoves) {
    if (state.assignedCount === n && isStrictChain(state.values)) break;
    const action = imitationAction(state, rng, stochastic);
    applyAction(state, action.i, action.v);
    moves++;
  }

  return {
    x: Int16Array.from(state.values),
    solved: state.assignedCount === n && isStrictChain(state.values),
    moves,
    finalEnergy: state.assignedCount === n ? violationEnergy(state.values) : Infinity
  };
}

function runHandcrafted(n, rng) {
  const t = teacherTrajectory(n, rng);
  return {
    x: Int16Array.from(t.state.values),
    solved: t.state.assignedCount === n && isStrictChain(t.state.values),
    moves: t.moves,
    finalEnergy: t.state.assignedCount === n ? violationEnergy(t.state.values) : Infinity
  };
}

function imitationSolveRate(n, seed, trials = 12) {
  const rng = mulberry32(seed);
  let solved = 0;
  let moves = 0;
  for (let t = 0; t < trials; t++) {
    const out = runImitation(n, rng, true);
    if (out.solved) solved++;
    moves += out.moves;
  }
  return { rate: solved / trials, avgMoves: moves / trials };
}

async function trainRL() {
  if (!window.tf) throw new Error('TensorFlow.js did not load.');
  MESSAGE_ROUNDS = +messageRounds.value;
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;

  const startN = +trainMinN.value;
  const maxN = +trainMaxN.value;
  const totalEpisodes = +episodes.value;
  trainedRange = { min: startN, max: maxN };
  reachedN = maxN;
  trained = false;
  initGNN();

  const rng = mulberry32(20260812);
  let completed = 0;
  let updates = 0;
  let lastLoss = NaN;

  while (completed < totalEpisodes) {
    const samples = [];
    const batchEpisodes = Math.min(IMITATION_BATCH_EPISODES, totalEpisodes - completed);

    for (let b = 0; b < batchEpisodes; b++) {
      const n = startN + Math.floor(rng() * (maxN - startN + 1));
      const trajectory = teacherTrajectory(n, rng);
      samples.push(...pickTeacherSamples(trajectory, rng));
      completed++;
    }

    lastLoss = await imitationUpdate(samples);
    updates++;

    if (updates % 4 === 0 || completed >= totalEpisodes) {
      const evalNow = imitationSolveRate(maxN, 7000 + updates, 8);
      statusEl.textContent = 'Teacher episodes ' + completed + '/' + totalEpisodes +
        ' · imitation update ' + updates +
        ' · train n=' + startN + '…' + maxN +
        ' · depth ' + MESSAGE_ROUNDS +
        ' · samples ' + samples.length +
        ' · solve@' + maxN + ' ' + (100 * evalNow.rate).toFixed(0) + '%' +
        (Number.isFinite(lastLoss) ? ' · loss ' + lastLoss.toFixed(4) : '');
      await tf.nextFrame();
    }
  }

  trained = true;
  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
  statusEl.textContent = 'Imitation training complete · constraint-status GNN · depth ' + MESSAGE_ROUNDS + '. Running teacher / imitation / SA benchmark…';
  await runBenchmark();
}

async function runBenchmark() {
  if (!trained || !params) return;
  trainBtn.disabled = true;
  benchmarkBtn.disabled = true;

  const rows = [];
  const rng = mulberry32(20260813);
  const trials = 12;

  for (const n of benchmarkLengths()) {
    statusEl.textContent = 'Benchmarking imitation GNN, handcrafted teacher, and SA at n=' + n + '…';
    let imitationSolved = 0;
    let teacherSolved = 0;
    let saSolved = 0;
    const imitationMoves = [];
    const teacherMoves = [];

    for (let t = 0; t < trials; t++) {
      const imitation = runImitation(n, rng, true);
      if (imitation.solved) {
        imitationSolved++;
        imitationMoves.push(imitation.moves);
      }

      const teacher = runHandcrafted(n, rng);
      if (teacher.solved) {
        teacherSolved++;
        teacherMoves.push(teacher.moves);
      }

      const sa = runSA(n, 120 * n, rng);
      if (sa.solved) saSolved++;
    }

    rows.push({
      n,
      policySuccess: imitationSolved / trials,
      heuristicSuccess: teacherSolved / trials,
      saSuccess: saSolved / trials,
      policyMoves: median(imitationMoves),
      heuristicMoves: median(teacherMoves)
    });
    await tf.nextFrame();
  }

  drawScaling(rows);
  const longest = rows[rows.length - 1];
  const n = longest.n;
  const imitation = runImitation(n, mulberry32(9101), true);
  const teacher = runHandcrafted(n, mulberry32(9102));
  const sa = runSA(n, 120 * n, mulberry32(9103));
  drawAssignment(imitation.x, teacher.x, sa.x);

  $('policySuccess').textContent = Math.round(100 * longest.policySuccess) + '%';
  $('heuristicSuccess').textContent = Math.round(100 * longest.heuristicSuccess) + '%';
  $('saSuccess').textContent = Math.round(100 * longest.saSuccess) + '%';
  $('policyMoves').textContent = Number.isFinite(longest.policyMoves) ? Math.round(longest.policyMoves).toString() : '—';
  $('heuristicMoves').textContent = Number.isFinite(longest.heuristicMoves) ? Math.round(longest.heuristicMoves).toString() : '—';
  $('saEffort').textContent = (120 * n).toLocaleString();
  $('metricLength').textContent = 'n=' + n + ' · imitation/teacher budget ' + MOVE_MULTIPLIER + 'n · train range ' + trainedRange.min + '…' + trainedRange.max + ' · ' + MESSAGE_ROUNDS + ' message round(s)';
  statusEl.textContent = 'Done. Blue is the behavior-cloned GNN; green is its handcrafted teacher; orange is SA.';

  trainBtn.disabled = false;
  benchmarkBtn.disabled = false;
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
  for (let k = 0; k <= 4; k++) ctx.fillText((25 * k) + '%', 5, Y(k / 4) + 4);
  const bx = X(trainedRange.max);
  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = '#a9b1c2';
  ctx.beginPath();
  ctx.moveTo(bx, 18);
  ctx.lineTo(bx, h - pad);
  ctx.stroke();
  ctx.restore();
  ctx.fillText('train max', Math.min(w - 80, bx + 5), 28);
  line(ctx, rows.map(r => [X(r.n), Y(r.policySuccess)]), '#5b67d6');
  line(ctx, rows.map(r => [X(r.n), Y(r.heuristicSuccess)]), '#2e8b72');
  line(ctx, rows.map(r => [X(r.n), Y(r.saSuccess)]), '#dd6b55');
}

function drawAssignment(imitation, teacher, sa) {
  const { ctx, w, h } = setupCanvas($('instanceChart'));
  const pad = 44;
  axes(ctx, w, h, pad, 'variable node id i', 'assigned value');
  const n = imitation.length;
  const X = i => pad + i / Math.max(1, n - 1) * (w - pad - 20);
  const Y = v => h - pad - Math.max(0, v) / DOMAIN_MAX * (h - pad - 28);
  line(ctx, Array.from(imitation, (v, i) => [X(i), Y(v)]), '#5b67d6', 2.2);
  line(ctx, Array.from(teacher, (v, i) => [X(i), Y(v)]), '#2e8b72', 2.0);
  line(ctx, Array.from(sa, (v, i) => [X(i), Y(v)]), '#dd6b55', 1.8);
}

messageRounds.addEventListener('change', () => {
  if (trained) {
    trained = false;
    benchmarkBtn.disabled = true;
    statusEl.textContent = 'Message-passing depth changed. Retrain before benchmarking.';
  }
});

statusEl.textContent = 'TensorFlow.js ready · backend: ' + tf.getBackend() + '. Imitation GNN: constraint-status nodes + handcrafted teacher.';
//# sourceURL=app-gnn-imitation-runtime.js
`;

  const blobUrl = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  try {
    await import(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

bootImitation().catch(err => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `Error loading GNN imitation experiment: ${err.message}`;
});
