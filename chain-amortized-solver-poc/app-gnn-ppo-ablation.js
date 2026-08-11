// Small-GNN PPO ablation entrypoint.
// Loads the current reward-shaped PPO engine and changes only representation /
// training complexity plus benchmark mode. Environment, action semantics, and
// terminal reward stay unchanged.

function replaceOnce(source, needle, replacement, label) {
  if (!source.includes(needle)) throw new Error(`PPO ablation patch failed: ${label}`);
  return source.replace(needle, replacement);
}

async function bootAblation() {
  const response = await fetch('./app-gnn-ppo-hybrid.js?v=20260811-22-base', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load PPO base engine (${response.status}).`);
  let source = await response.text();

  source = replaceOnce(source, 'const EMBED_DIM = 24;', 'const EMBED_DIM = 8;', 'embedding size');
  source = replaceOnce(source, 'const MESSAGE_ROUNDS = 6;', 'let MESSAGE_ROUNDS = 1;', 'message depth');
  source = replaceOnce(source, 'const PPO_BATCH_EPISODES = 4;', 'const PPO_BATCH_EPISODES = 32;', 'PPO batch');
  source = replaceOnce(source, 'const SELF_IMITATION_COEF = 0.03;', 'const SELF_IMITATION_COEF = 0;', 'self imitation');

  source = replaceOnce(
    source,
    "const statusEl = $('status');",
    "const statusEl = $('status');\nconst messageRounds = $('messageRounds');",
    'message-depth control'
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

  source = replaceOnce(
    source,
    '      if (out.solved) rememberSuccess(out.trajectory);',
    '      // Self-imitation is deliberately disabled in this ablation.',
    'disable success replay'
  );

  source = source.replace(
    ' · success replay ${successReplay.length}',
    ' · depth ${MESSAGE_ROUNDS} · batch ${PPO_BATCH_EPISODES}'
  );

  source = replaceOnce(
    source,
    '  statusEl.textContent = `PPO training complete · curriculum reached n=${reachedN}. Running hybrid-search benchmark…`;',
    '  statusEl.textContent = `PPO training complete · depth ${MESSAGE_ROUNDS} · curriculum reached n=${reachedN}. Running direct-policy benchmark…`;',
    'post-training status'
  );

  source = replaceOnce(
    source,
    '    statusEl.textContent = `Benchmarking PPO-guided annealed search at n=${n}…`;',
    '    statusEl.textContent = `Benchmarking direct PPO policy at n=${n} · depth ${MESSAGE_ROUNDS}…`;',
    'benchmark status'
  );

  source = replaceOnce(
    source,
    '      const policy = hybridSolve(n, rng);',
    '      const policy = rollout(n, rng, false);',
    'direct benchmark rollout'
  );

  source = replaceOnce(
    source,
    '  const policy = hybridSolve(n, mulberry32(9001));',
    '  const policy = rollout(n, mulberry32(9001), false);',
    'direct longest rollout'
  );

  source = replaceOnce(
    source,
    "  $('metricLength').textContent = `n=${n} · domain 0…${DOMAIN_MAX} · training ${MOVE_MULTIPLIER}n · hybrid inference ${INFERENCE_MOVE_MULTIPLIER}n · curriculum reached n=${reachedN}`;",
    "  $('metricLength').textContent = `n=${n} · 8-d embedding · ${MESSAGE_ROUNDS} message round(s) · PPO batch ${PPO_BATCH_EPISODES} · curriculum reached n=${reachedN}`;",
    'benchmark headline'
  );

  source = replaceOnce(
    source,
    '  statusEl.textContent = `Done. Training stays at ${MOVE_MULTIPLIER}n moves; inference gets up to ${INFERENCE_MOVE_MULTIPLIER}n moves with PPO proposals, ε exploration, annealed acceptance, and a protected incumbent.`;',
    '  statusEl.textContent = `Done. Direct ${MOVE_MULTIPLIER}n PPO policy · 8-d embeddings · ${MESSAGE_ROUNDS} message round(s) · batch ${PPO_BATCH_EPISODES} · no self-imitation.`;',
    'final benchmark status'
  );

  source = replaceOnce(
    source,
    '    statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Train the PPO actor-critic GNN curriculum; benchmark uses ${INFERENCE_MOVE_MULTIPLIER}n hybrid search.`;',
    '    statusEl.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()}. Small-GNN PPO ablation: choose 0/1/2/6 message rounds, then train.`;',
    'ready status'
  );

  source += `\n\nmessageRounds.addEventListener('change', () => {
  if (trained) {
    trained = false;
    benchmarkBtn.disabled = true;
    statusEl.textContent = 'Message-passing depth changed. Retrain before benchmarking so train and evaluation depths match.';
  }
});\n`;

  source += '\n//# sourceURL=app-gnn-ppo-ablation-runtime.js\n';
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
  if (statusEl) statusEl.textContent = `Error loading PPO ablation: ${err.message}`;
});
