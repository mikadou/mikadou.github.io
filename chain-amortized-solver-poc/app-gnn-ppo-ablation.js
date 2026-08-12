// Scale-aware small-GNN PPO ablation + handcrafted heuristic comparison.
// Loads the reward-shaped PPO engine and changes representation/training only.
// Environment, joint (i,v) action semantics, 3n PPO budget, and terminal reward
// remain unchanged. The handcrafted solver is benchmark-only.

function replaceOnce(source, needle, replacement, label) {
  if (!source.includes(needle)) throw new Error(`PPO ablation patch failed: ${label}`);
  return source.replace(needle, replacement);
}

async function bootAblation() {
  const response = await fetch('./app-gnn-ppo-hybrid.js?v=20260811-22-base', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load PPO base engine (${response.status}).`);
  let source = await response.text();

  // Small representation, but now explicitly scale-aware.
  source = replaceOnce(source, 'const NODE_FEATURE_DIM = 5;', 'const NODE_FEATURE_DIM = 7;', 'scale-aware feature dimension');
  source = replaceOnce(source, 'const EMBED_DIM = 24;', 'const EMBED_DIM = 8;', 'embedding size');
  source = replaceOnce(source, 'const MESSAGE_ROUNDS = 6;', 'let MESSAGE_ROUNDS = 0;', 'message depth');
  source = replaceOnce(source, 'const PPO_BATCH_EPISODES = 4;', 'const PPO_BATCH_EPISODES = 32;', 'PPO batch');
  source = replaceOnce(source, 'const SELF_IMITATION_COEF = 0.03;', 'const SELF_IMITATION_COEF = 0;', 'self imitation');

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

  const scaleAwareGraphFeatures = `  // Scale-aware static features:
  // [variable-type, constraint-type, absolute-position, relative-position,
  //  problem-size, assigned, assigned-value].
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

  source = replaceOnce(source, oldGraphFeatures, scaleAwareGraphFeatures, 'scale-aware graph features');

  source = replaceOnce(
    source,
    `    data[p + 3] = state.assigned[i] ? 1 : 0;
    data[p + 4] = state.assigned[i] ? state.values[i] / DOMAIN_MAX : 0;`,
    `    data[p + 5] = state.assigned[i] ? 1 : 0;
    data[p + 6] = state.assigned[i] ? state.values[i] / DOMAIN_MAX : 0;`,
    'dynamic feature indices'
  );

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
    'row layer norm helper'
  );

  source = replaceOnce(
    source,
    '    h = tf.relu(h.add(z.add(params.brel)));',
    '    h = tf.relu(rowLayerNorm(h.add(z.add(params.brel))));',
    'normalized residual message update'
  );

  source = replaceOnce(
    source,
    "async function trainRL() {\n  if (!window.tf) throw new Error('TensorFlow.js did not load.');",
    "async function trainRL() {\n  if (!window.tf) throw new Error('TensorFlow.js did not load.');\n  MESSAGE_ROUNDS = +messageRounds.value;",
    'read selected message depth'
  );

  // Mixed-size training: every episode samples uniformly from the configured
  // training range, rather than spending most updates on one tiny size.
  source = replaceOnce(source, '  reachedN = startN;', '  reachedN = maxN;', 'training max marker');
  source = replaceOnce(source, '  let currentN = startN;', '  let currentN = maxN;', 'disable progressive curriculum');
  source = replaceOnce(
    source,
    '      const n = chooseCurriculumN(currentN, startN, rng);',
    '      const n = startN + Math.floor(rng() * (maxN - startN + 1));',
    'uniform mixed-size training'
  );

  source = replaceOnce(
    source,
    '      if (out.solved) rememberSuccess(out.trajectory);',
    '      // Self-imitation is deliberately disabled in this ablation.',
    'disable success replay'
  );

  source = replaceOnce(
    source,
    "      statusEl.textContent = `Episode ${completed}/${totalEpisodes} · PPO update ${updates} · curriculum n=${currentN}${currentN < maxN ? `/${maxN}` : ' (max)'} · greedy ${(100 * lastGreedy.rate).toFixed(0)}% · exploratory ${(100 * stochasticRate).toFixed(0)}% · success replay ${successReplay.length}${Number.isFinite(lastLoss) ? ` · loss ${lastLoss.toFixed(4)}` : ''}${promote ? ' · PROMOTED' : ''}`;",
    "      statusEl.textContent = `Episode ${completed}/${totalEpisodes} · PPO update ${updates} · train n=${startN}…${maxN} uniform · greedy@${maxN} ${(100 * lastGreedy.rate).toFixed(0)}% · exploratory@${maxN} ${(100 * stochasticRate).toFixed(0)}% · depth ${MESSAGE_ROUNDS} · batch ${PPO_BATCH_EPISODES}${Number.isFinite(lastLoss) ? ` · loss ${lastLoss.toFixed(4)}` : ''}`;",
    'mixed-size training status'
  );

  source = replaceOnce(
    source,
    '  statusEl.textContent = `PPO training complete · curriculum reached n=${reachedN}. Running hybrid-search benchmark…`;',
    '  statusEl.textContent = `PPO training complete · trained uniformly on n=${startN}…${maxN} · depth ${MESSAGE_ROUNDS}. Running PPO / handcrafted / SA benchmark…`;',
    'post-training status'
  );

  // Handcrafted robust local-repair heuristic. The first n moves sample a full
  // assignment around mean 2*i. Remaining moves resample a random endpoint of
  // a currently violated adjacent constraint. Total budget matches PPO: 3n.
  source = replaceOnce(
    source,
    'function greedyPolicyProposal(state) {',
    `const HANDCRAFTED_SIGMA = 1.0;

function handcraftedGaussian(mean, sigma, rng) {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + sigma * z;
}

function handcraftedValue(i, rng) {
  const raw = handcraftedGaussian(2 * i, HANDCRAFTED_SIGMA, rng);
  return Math.max(0, Math.min(DOMAIN_MAX, Math.round(raw)));
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

function runHandcrafted(n, rng) {
  const x = new Int16Array(n);
  const maxMoves = MOVE_MULTIPLIER * n;
  let moves = 0;

  // Imperfect construction: independently sample every variable around 2*i.
  for (let i = 0; i < n && moves < maxMoves; i++) {
    x[i] = handcraftedValue(i, rng);
    moves++;
  }

  while (moves < maxMoves && !isStrictChain(x)) {
    const candidates = violatedEndpointIds(x);
    if (!candidates.length) break;
    const i = candidates[Math.floor(rng() * candidates.length)];
    x[i] = handcraftedValue(i, rng);
    moves++;
  }

  return {
    x: Int16Array.from(x),
    solved: isStrictChain(x),
    moves,
    finalEnergy: violationEnergy(x),
    maxMoves
  };
}

function greedyPolicyProposal(state) {`,
    'handcrafted local-repair solver'
  );

  source = replaceOnce(
    source,
    '    statusEl.textContent = `Benchmarking PPO-guided annealed search at n=${n}…`;',
    '    statusEl.textContent = `Benchmarking PPO, handcrafted repair, and SA at n=${n} · depth ${MESSAGE_ROUNDS}…`;',
    'benchmark status'
  );

  source = replaceOnce(
    source,
    `    let policySolved = 0;
    let saSolved = 0;
    const policyMoves = [];`,
    `    let policySolved = 0;
    let heuristicSolved = 0;
    let saSolved = 0;
    const policyMoves = [];
    const heuristicMoves = [];`,
    'benchmark counters'
  );

  source = replaceOnce(
    source,
    `      const policy = hybridSolve(n, rng);
      if (policy.solved) {
        policySolved++;
        policyMoves.push(policy.moves);
      }
      const sa = runSA(n, 120 * n, rng);
      if (sa.solved) saSolved++;`,
    `      const policy = hybridSolve(n, rng);
      if (policy.solved) {
        policySolved++;
        policyMoves.push(policy.moves);
      }
      const heuristic = runHandcrafted(n, rng);
      if (heuristic.solved) {
        heuristicSolved++;
        heuristicMoves.push(heuristic.moves);
      }
      const sa = runSA(n, 120 * n, rng);
      if (sa.solved) saSolved++;`,
    'handcrafted trial benchmark'
  );

  source = replaceOnce(
    source,
    `      policySuccess: policySolved / trials,
      saSuccess: saSolved / trials,
      policyMoves: median(policyMoves)`,
    `      policySuccess: policySolved / trials,
      heuristicSuccess: heuristicSolved / trials,
      saSuccess: saSolved / trials,
      policyMoves: median(policyMoves),
      heuristicMoves: median(heuristicMoves)`,
    'handcrafted benchmark row'
  );

  source = replaceOnce(
    source,
    `  const policy = hybridSolve(n, mulberry32(9001));
  const sa = runSA(n, 120 * n, mulberry32(9002));
  drawAssignment(policy.x, sa.x);`,
    `  const policy = hybridSolve(n, mulberry32(9001));
  const heuristic = runHandcrafted(n, mulberry32(9003));
  const sa = runSA(n, 120 * n, mulberry32(9002));
  drawAssignment(policy.x, heuristic.x, sa.x);`,
    'longest handcrafted benchmark'
  );

  source = replaceOnce(
    source,
    `  $('policySuccess').textContent = \`${'${Math.round(100 * longest.policySuccess)}%'}\`;
  $('saSuccess').textContent = \`${'${Math.round(100 * longest.saSuccess)}%'}\`;
  $('policyMoves').textContent = Number.isFinite(longest.policyMoves) ? Math.round(longest.policyMoves).toString() : '—';
  $('saEffort').textContent = (120 * n).toLocaleString();`,
    `  $('policySuccess').textContent = \`${'${Math.round(100 * longest.policySuccess)}%'}\`;
  $('heuristicSuccess').textContent = \`${'${Math.round(100 * longest.heuristicSuccess)}%'}\`;
  $('saSuccess').textContent = \`${'${Math.round(100 * longest.saSuccess)}%'}\`;
  $('policyMoves').textContent = Number.isFinite(longest.policyMoves) ? Math.round(longest.policyMoves).toString() : '—';
  $('heuristicMoves').textContent = Number.isFinite(longest.heuristicMoves) ? Math.round(longest.heuristicMoves).toString() : '—';
  $('saEffort').textContent = (120 * n).toLocaleString();`,
    'handcrafted benchmark metrics'
  );

  source = replaceOnce(
    source,
    `  line(ctx, rows.map(r => [X(r.n), Y(r.policySuccess)]), '#5b67d6');
  line(ctx, rows.map(r => [X(r.n), Y(r.saSuccess)]), '#dd6b55');`,
    `  line(ctx, rows.map(r => [X(r.n), Y(r.policySuccess)]), '#5b67d6');
  line(ctx, rows.map(r => [X(r.n), Y(r.heuristicSuccess)]), '#2e8b72');
  line(ctx, rows.map(r => [X(r.n), Y(r.saSuccess)]), '#dd6b55');`,
    'handcrafted solve-rate line'
  );

  source = replaceOnce(source, 'function drawAssignment(policy, sa) {', 'function drawAssignment(policy, heuristic, sa) {', 'handcrafted assignment signature');
  source = replaceOnce(
    source,
    `  line(ctx, Array.from(policy, (v, i) => [X(i), Y(v)]), '#5b67d6', 2.2);
  line(ctx, Array.from(sa, (v, i) => [X(i), Y(v)]), '#dd6b55', 1.8);`,
    `  line(ctx, Array.from(policy, (v, i) => [X(i), Y(v)]), '#5b67d6', 2.2);
  line(ctx, Array.from(heuristic, (v, i) => [X(i), Y(v)]), '#2e8b72', 2.0);
  line(ctx, Array.from(sa, (v, i) => [X(i), Y(v)]), '#dd6b55', 1.8);`,
    'handcrafted assignment line'
  );

  // Keep the ablation benchmark on the direct 3n PPO policy, not hybrid search.
  source = replaceOnce(source, '      const policy = hybridSolve(n, rng);', '      const policy = rollout(n, rng, false);', 'direct benchmark rollout');
  source = replaceOnce(source, '  const policy = hybridSolve(n, mulberry32(9001));', '  const policy = rollout(n, mulberry32(9001), false);', 'direct longest rollout');

  source = replaceOnce(
    source,
    "  $('metricLength').textContent = `n=${n} · domain 0…${DOMAIN_MAX} · training ${MOVE_MULTIPLIER}n · hybrid inference ${INFERENCE_MOVE_MULTIPLIER}n · curriculum reached n=${reachedN}`;",
    "  $('metricLength').textContent = `n=${n} · PPO ${MOVE_MULTIPLIER}n · handcrafted ${MOVE_MULTIPLIER}n (mean 2i, σ=${HANDCRAFTED_SIGMA}) · SA 120n · train range ${trainedRange.min}…${trainedRange.max}`;",
    'benchmark headline'
  );

  source = replaceOnce(
    source,
    '  statusEl.textContent = `Done. Training stays at ${MOVE_MULTIPLIER}n moves; inference gets up to ${INFERENCE_MOVE_MULTIPLIER}n moves with PPO proposals, ε exploration, annealed acceptance, and a protected incumbent.`;',
    '  statusEl.textContent = `Done. Comparing direct PPO with handcrafted violated-constraint repair (mean 2i, σ=${HANDCRAFTED_SIGMA}) and SA.`;',
    'final benchmark status'
  );

  source = replaceOnce(
    source,
    "  ctx.fillText('curriculum reached', Math.min(w - 115, bx + 5), 28);",
    "  ctx.fillText('train max', Math.min(w - 80, bx + 5), 28);",
    'training-range chart marker'
  );

  source = replaceOnce(
    source,
    '    statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Train the PPO actor-critic GNN curriculum; benchmark uses ${INFERENCE_MOVE_MULTIPLIER}n hybrid search.`;',
    '    statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Scale-aware PPO + handcrafted local-repair comparison ready.`;',
    'ready status'
  );

  source += `

messageRounds.addEventListener('change', () => {
  if (trained) {
    trained = false;
    benchmarkBtn.disabled = true;
    statusEl.textContent = 'Message-passing depth changed. Retrain before benchmarking so train and evaluation depths match.';
  }
});
`;

  source += '\n//# sourceURL=app-gnn-ppo-handcrafted-comparison-runtime.js\n';
  const blobUrl = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  try {
    await import(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

bootAblation().catch(err => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `Error loading PPO / handcrafted comparison: ${err.message}`;
});