// Diagnostic wrapper: keep the current absolute-utility critic unchanged,
// then add an exact oracleStepsToGo control over the same proposal candidate pool.
async function bootOracleStepsVariant() {
  const baseUrl = new URL('./app-gnn-successor-absolute.js?v=20260813-39-base', import.meta.url);
  const response = await fetch(baseUrl, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Failed to load absolute-utility successor critic: HTTP ${response.status}`);
  let wrapper = await response.text();

  // The patched absolute wrapper is imported from a blob URL, so make its own base-engine URL absolute first.
  const criticUrl = new URL('./app-gnn-successor-critic.js?v=20260813-37-base', baseUrl).href;
  const relativeCriticLine = "  const baseUrl = new URL('./app-gnn-successor-critic.js?v=20260813-37-base', import.meta.url);";
  if (!wrapper.includes(relativeCriticLine)) throw new Error('Oracle steps patch failed: absolute wrapper base URL marker');
  wrapper = wrapper.replace(relativeCriticLine, '  const baseUrl = new URL(' + JSON.stringify(criticUrl) + ');');

  const marker = "  const blobUrl = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));";
  if (!wrapper.includes(marker)) throw new Error('Oracle steps patch failed: absolute wrapper insertion marker');

  const injection = String.raw`
  // --- oracleStepsToGo diagnostic control ---
  const oracleCode = [
    'const ORACLE_STEPS_ROLLOUTS = 5;',
    'function teacherStepsToGo(startState, rng, maxSteps = null) {',
    '  let state = cloneState(startState), steps = 0, limit = maxSteps == null ? 3 * state.n : maxSteps;',
    '  if (terminalUtility(state)) return 0;',
    '  while (state.assignedCount < state.n && steps < limit) {',
    '    const remaining = []; for (let i = 0; i < state.n; i++) if (!state.assigned[i]) remaining.push(i);',
    '    const i = remaining[Math.floor(rng() * remaining.length)];',
    '    state = propagateSuccessor(state, i, teacherSampleValue(i, rng)); steps++;',
    '  }',
    '  while (steps < limit && state.assignedCount === state.n && !isStrictChain(state.values)) {',
    '    const ids = violatedEndpointIds(state.values); if (!ids.length) break;',
    '    const i = ids[Math.floor(rng() * ids.length)];',
    '    state = propagateSuccessor(state, i, teacherSampleValue(i, rng)); steps++;',
    '  }',
    '  return terminalUtility(state) ? steps : limit + state.n;',
    '}',
    'function oracleCandidateSteps(state, variableId, value, seedBase) {',
    '  const successor = propagateSuccessor(state, variableId, value);',
    '  if (terminalUtility(successor)) return 0;',
    '  let total = 0;',
    '  for (let r = 0; r < ORACLE_STEPS_ROLLOUTS; r++) {',
    '    const seed = (seedBase ^ Math.imul(r + 1, 1013904223)) >>> 0;',
    '    total += teacherStepsToGo(successor, mulberry32(seed));',
    '  }',
    '  return total / ORACLE_STEPS_ROLLOUTS;',
    '}',
    'function chooseValueOracleSteps(state, variableId, rng, stochastic = true) {',
    '  const candidates = generateValueCandidates(state, variableId, rng, stochastic).values;',
    '  const seedBase = Math.floor(rng() * 4294967296) >>> 0;',
    '  let bestValue = candidates[0], bestSteps = Infinity;',
    '  for (let k = 0; k < candidates.length; k++) {',
    '    const v = candidates[k], steps = oracleCandidateSteps(state, variableId, v, seedBase);',
    '    if (steps < bestSteps - 1e-9) { bestSteps = steps; bestValue = v; }',
    '  }',
    '  return bestValue;',
    '}',
    'function runOracleSteps(n, rng, stochastic = true) {',
    '  let state = emptyState(n), maxMoves = MOVE_MULTIPLIER * n, moves = 0;',
    '  while (state.assignedCount < n && moves < maxMoves) {',
    '    const remaining = []; for (let i = 0; i < n; i++) if (!state.assigned[i]) remaining.push(i);',
    '    const i = stochastic ? remaining[Math.floor(rng() * remaining.length)] : remaining[0];',
    '    state = propagateSuccessor(state, i, chooseValueOracleSteps(state, i, rng, stochastic)); moves++;',
    '  }',
    '  while (moves < maxMoves && !isStrictChain(state.values)) {',
    '    const i = chooseRepairVariable(state, rng, stochastic);',
    '    state = propagateSuccessor(state, i, chooseValueOracleSteps(state, i, rng, stochastic)); moves++;',
    '  }',
    '  return { x: Int16Array.from(state.values), solved: isStrictChain(state.values), moves };',
    '}'
  ].join('\n');

  if (!src.includes('\nfunction runLearned(')) throw new Error('Oracle steps patch failed: runLearned marker');
  src = src.replace('\nfunction runLearned(', '\n' + oracleCode + '\nfunction runLearned(');

  if (!src.includes('async function runBenchmark(){')) throw new Error('Oracle steps patch failed: runBenchmark marker');
  src = src.replace('async function runBenchmark(){', 'async function runBenchmarkBase(){');
  src = src.replace("logLine('END');statusEl.textContent='Done. V(s) predicts final solution utility minus future solver cost.';",
                    "logLine('BASE_BENCH_END');statusEl.textContent='Base benchmark complete. Running oracle steps-to-go control…';");

  const oracleBenchmarkCode = [
    'function oracleStepsTrials(n) { return n <= 21 ? 3 : 1; }',
    'async function runOracleStepsBenchmark() {',
    '  const lengths = benchmarkLengths().filter(n => n >= trainedRange.max && n <= 40);',
    '  const rng = mulberry32(20260825);',
    '  trainBtn.disabled = true; benchmarkBtn.disabled = true;',
    '  for (const n of lengths) {',
    '    const trials = oracleStepsTrials(n);',
    '    let solved = 0, violations = 0, moveSum = 0;',
    "    statusEl.textContent = 'Oracle steps-to-go at n=' + n + ' · ' + trials + ' trial' + (trials === 1 ? '' : 's') + '…';",
    '    for (let t = 0; t < trials; t++) {',
    '      const result = runOracleSteps(n, rng, true);',
    '      if (result.solved) solved++; violations += violatedConstraintCount(result.x); moveSum += result.moves;',
    '      await tf.nextFrame();',
    '    }',
    "    logLine('ORACLE_STEPS n=' + n + ' trials=' + trials + ' rolloutsPerCandidate=' + ORACLE_STEPS_ROLLOUTS + ' solve=' + Math.round(100 * solved / trials) + '% avgMoves=' + (moveSum / trials).toFixed(1) + ' avgViol=' + (violations / trials).toFixed(2));",
    '  }',
    '  appendOracleStepsProbe(trainedRange.max);',
    "  logLine('END');",
    "  statusEl.textContent = 'Done. oracleStepsToGo reports whether the existing candidate pool can solve when successor ranking is exact.';",
    '  trainBtn.disabled = false; benchmarkBtn.disabled = false;',
    '}',
    'async function runBenchmark() {',
    '  await runBenchmarkBase();',
    '  await runOracleStepsBenchmark();',
    '}',
    'function appendOracleStepsProbe(n) {',
    '  let state = makeLearnerRolloutState(n, mulberry32(9500 + n)), i;',
    '  if (state.assignedCount < n) { const rem = []; for (let k = 0; k < n; k++) if (!state.assigned[k]) rem.push(k); i = rem[0]; }',
    '  else { if (isStrictChain(state.values)) state = makeRepairTrainingState(n, mulberry32(9501 + n)).state; const ids = violatedEndpointIds(state.values); i = ids[0]; }',
    '  const candidates = generateValueCandidates(state, i, mulberry32(9502 + n), false).values;',
    '  const seedBase = (950300 + n) >>> 0, rows = [];',
    '  for (const v of candidates) rows.push({ v, steps: oracleCandidateSteps(state, i, v, seedBase) });',
    '  rows.sort((a, b) => a.steps - b.steps);',
    "  logLine('ORACLE_STEPS_PROBE n=' + n + ' variable=' + i + ' phase=' + (state.assignedCount < n ? 'construct' : 'repair') + ' best=[' + rows.slice(0, 10).map(x => x.v + ':' + x.steps.toFixed(1)).join(',') + ']');",
    '}'
  ].join('\n');

  if (!src.includes('\nfunction appendProbe(')) throw new Error('Oracle steps patch failed: appendProbe marker');
  src = src.replace('\nfunction appendProbe(', '\n' + oracleBenchmarkCode + '\nfunction appendProbe(');

  src = src.replace('GNN_SUCCESSOR_ABSOLUTE_CRITIC_DIAGNOSTIC v3', 'GNN_SUCCESSOR_ABSOLUTE_CRITIC_DIAGNOSTIC v3_oracle1');
  src = src.replace('criticTarget=absoluteUtilityMinusFutureCost daggerWarmup=', 'criticTarget=absoluteUtilityMinusFutureCost oracleStepsRollouts=5 daggerWarmup=');
  // --- end oracleStepsToGo diagnostic control ---
`;

  wrapper = wrapper.replace(marker, injection + '\n' + marker);

  const blobUrl = URL.createObjectURL(new Blob([wrapper], { type: 'text/javascript' }));
  try { await import(blobUrl); }
  finally { URL.revokeObjectURL(blobUrl); }
}

bootOracleStepsVariant().catch(err => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `Error loading oracle steps-to-go diagnostic: ${err.message}`;
});
