// Diagnostic wrapper for the standalone imitation GNN.
// Keeps the learning algorithm unchanged and adds a compact, copyable text log.

function replaceOnce(source, needle, replacement, label) {
  if (!source.includes(needle)) throw new Error(`Imitation diagnostics patch failed: ${label}`);
  return source.replace(needle, replacement);
}

function installDiagnosticsUI() {
  if (document.getElementById('diagnosticLog')) return;
  const section = document.createElement('section');
  section.className = 'card explanation';
  section.innerHTML = `
    <h2>Copyable diagnostic log</h2>
    <p>After training and the automatic benchmark finish, tap <strong>Copy diagnostics</strong> and paste the text into ChatGPT. The log contains compact training snapshots, benchmark results, a repair-state policy probe, and one closed-loop repair trace.</p>
    <div class="buttons" style="margin:12px 0">
      <button id="copyLogBtn" type="button">Copy diagnostics</button>
      <button id="clearLogBtn" type="button" class="secondary">Clear log</button>
    </div>
    <textarea id="diagnosticLog" readonly spellcheck="false" style="width:100%;min-height:320px;resize:vertical;border:1px solid #dfe4ee;border-radius:10px;padding:12px;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#fbfcff;color:#20283a"></textarea>
  `;
  const footer = document.querySelector('footer');
  if (footer?.parentNode) footer.parentNode.insertBefore(section, footer);
  else document.querySelector('main')?.appendChild(section);
}

async function bootDiagnostics() {
  installDiagnosticsUI();

  const response = await fetch('./app-gnn-imitation.js?v=20260812-29-base', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load imitation engine (${response.status}).`);
  let source = await response.text();

  source = replaceOnce(
    source,
    "const statusEl = $('status');",
    `const statusEl = $('status');
const diagnosticLogEl = $('diagnosticLog');
const copyLogBtn = $('copyLogBtn');
const clearLogBtn = $('clearLogBtn');
let diagnosticLines = [];

function syncDiagnosticLog() {
  if (!diagnosticLogEl) return;
  diagnosticLogEl.value = diagnosticLines.join('\\n');
  diagnosticLogEl.scrollTop = diagnosticLogEl.scrollHeight;
}

function appendDiagnostic(line) {
  diagnosticLines.push(String(line));
  syncDiagnosticLog();
}

function resetDiagnostic() {
  diagnosticLines = [];
  syncDiagnosticLog();
}

async function copyDiagnosticLog() {
  const text = diagnosticLines.join('\\n');
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    if (copyLogBtn) copyLogBtn.textContent = 'Copied';
    setTimeout(() => { if (copyLogBtn) copyLogBtn.textContent = 'Copy diagnostics'; }, 1200);
  } catch (err) {
    if (diagnosticLogEl) {
      diagnosticLogEl.focus();
      diagnosticLogEl.select();
      document.execCommand('copy');
    }
  }
}

copyLogBtn?.addEventListener('click', copyDiagnosticLog);
clearLogBtn?.addEventListener('click', resetDiagnostic);`,
    'diagnostic helpers'
  );

  source = replaceOnce(
    source,
    `  initModel();

  const rng = mulberry32(20260812);`,
    `  initModel();
  resetDiagnostic();
  appendDiagnostic('GNN_IMITATION_DIAGNOSTIC v1');
  appendDiagnostic('CONFIG backend=' + tf.getBackend() +
    ' train=' + startN + '..' + maxN +
    ' episodes=' + totalEpisodes +
    ' rounds=' + MESSAGE_ROUNDS +
    ' embed=' + EMBED_DIM +
    ' nodeFeatures=' + NODE_FEATURE_DIM +
    ' batchEpisodes=' + BATCH_EPISODES +
    ' samplesPerEpisode=' + SAMPLES_PER_EPISODE +
    ' teacherSigma=' + TEACHER_SIGMA +
    ' budget=' + MOVE_MULTIPLIER + 'n');

  const rng = mulberry32(20260812);`,
    'training log header'
  );

  source = replaceOnce(
    source,
    `      statusEl.textContent = \`Episodes \${completed}/\${totalEpisodes} · update \${updates} · train n=\${startN}…\${maxN} · depth \${MESSAGE_ROUNDS} · loss \${lastLoss.toFixed(3)} · candidate mass \${(100 * diag.candidateMass).toFixed(0)}% · mean MAE \${diag.meanMae.toFixed(1)} · solve@\${maxN} \${(100 * solve).toFixed(0)}%\`;
      await tf.nextFrame();`,
    `      statusEl.textContent = \`Episodes \${completed}/\${totalEpisodes} · update \${updates} · train n=\${startN}…\${maxN} · depth \${MESSAGE_ROUNDS} · loss \${lastLoss.toFixed(3)} · candidate mass \${(100 * diag.candidateMass).toFixed(0)}% · mean MAE \${diag.meanMae.toFixed(1)} · solve@\${maxN} \${(100 * solve).toFixed(0)}%\`;
      if (updates % 8 === 0 || completed >= totalEpisodes) {
        appendDiagnostic('TRAIN ep=' + completed +
          ' update=' + updates +
          ' loss=' + lastLoss.toFixed(4) +
          ' candMass=' + (100 * diag.candidateMass).toFixed(1) + '%' +
          ' meanMAE=' + diag.meanMae.toFixed(2) +
          ' sigmaMAE=' + diag.sigmaMae.toFixed(3) +
          ' solve@n' + maxN + '=' + (100 * solve).toFixed(0) + '%');
      }
      await tf.nextFrame();`,
    'training snapshot logging'
  );

  source = replaceOnce(
    source,
    `    let policySolved = 0, teacherSolved = 0, saSolved = 0;
    const policyMoves = [], teacherMoves = [];`,
    `    let policySolved = 0, teacherSolved = 0, saSolved = 0;
    let policyEnergySum = 0, teacherEnergySum = 0;
    let policyViolationSum = 0, teacherViolationSum = 0;
    const policyMoves = [], teacherMoves = [];`,
    'benchmark diagnostic counters'
  );

  source = replaceOnce(
    source,
    `      const p = runImitation(n, rng, true);
      if (p.solved) { policySolved++; policyMoves.push(p.moves); }
      const h = runHandcrafted(n, rng);
      if (h.solved) { teacherSolved++; teacherMoves.push(h.moves); }`,
    `      const p = runImitation(n, rng, true);
      if (p.solved) { policySolved++; policyMoves.push(p.moves); }
      policyEnergySum += violationEnergy(p.x);
      policyViolationSum += violatedConstraintCount(p.x);
      const h = runHandcrafted(n, rng);
      if (h.solved) { teacherSolved++; teacherMoves.push(h.moves); }
      teacherEnergySum += violationEnergy(h.x);
      teacherViolationSum += violatedConstraintCount(h.x);`,
    'benchmark diagnostic accumulation'
  );

  source = replaceOnce(
    source,
    `    rows.push({
      n,
      policySuccess: policySolved / trials,
      heuristicSuccess: teacherSolved / trials,
      saSuccess: saSolved / trials,
      policyMoves: median(policyMoves),
      heuristicMoves: median(teacherMoves)
    });`,
    `    rows.push({
      n,
      policySuccess: policySolved / trials,
      heuristicSuccess: teacherSolved / trials,
      saSuccess: saSolved / trials,
      policyMoves: median(policyMoves),
      heuristicMoves: median(teacherMoves)
    });
    appendDiagnostic('BENCH n=' + n +
      ' gnn=' + Math.round(100 * policySolved / trials) + '%' +
      ' teacher=' + Math.round(100 * teacherSolved / trials) + '%' +
      ' sa=' + Math.round(100 * saSolved / trials) + '%' +
      ' gnnAvgViol=' + (policyViolationSum / trials).toFixed(2) +
      ' teacherAvgViol=' + (teacherViolationSum / trials).toFixed(2) +
      ' gnnAvgEnergy=' + (policyEnergySum / trials).toFixed(2) +
      ' teacherAvgEnergy=' + (teacherEnergySum / trials).toFixed(2) +
      ' gnnMedMoves=' + (Number.isFinite(median(policyMoves)) ? median(policyMoves) : 'NA') +
      ' teacherMedMoves=' + (Number.isFinite(median(teacherMoves)) ? median(teacherMoves) : 'NA'));`,
    'benchmark row logging'
  );

  source = replaceOnce(
    source,
    `  statusEl.textContent = 'Done. Candidate mass and mean MAE diagnose representation separately from closed-loop solve rate.';`,
    `  appendDiagnosticProbes();
  appendDiagnostic('END');
  statusEl.textContent = 'Done. Copy the diagnostic log below and paste it into ChatGPT for analysis.';`,
    'final diagnostic probes'
  );

  source += `

function violatedConstraintCount(x) {
  let count = 0;
  for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) count++;
  return count;
}

function probabilitySnapshot(logits, allowed) {
  let max = -Infinity;
  for (const i of allowed) max = Math.max(max, logits[i]);
  let denom = 0;
  const entries = [];
  for (const i of allowed) {
    const w = Math.exp(logits[i] - max);
    entries.push([i, w]);
    denom += w;
  }
  return entries.map(([i, w]) => [i, w / denom]);
}

function findRepairProbe(n) {
  for (let attempt = 0; attempt < 40; attempt++) {
    const t = teacherTrajectory(n, mulberry32(930000 + n * 101 + attempt));
    if (t.repair.length) return t.repair[Math.floor(t.repair.length / 2)];
  }
  return null;
}

function appendRepairProbe(n) {
  const sample = findRepairProbe(n);
  if (!sample) {
    appendDiagnostic('PROBE_REPAIR n=' + n + ' noTeacherRepairState');
    return;
  }
  const state = sample.state;
  const candidates = Array.from(sample.candidateIds);
  const candidateSet = new Set(candidates);
  const snap = policySnapshot(state);
  const allowed = allowedVariableIds(state);
  const probs = probabilitySnapshot(snap.varLogits, allowed).sort((a, b) => b[1] - a[1]);
  const mass = probs.reduce((s, [i, p]) => s + (candidateSet.has(i) ? p : 0), 0);
  const top = probs.slice(0, Math.min(10, probs.length)).map(([i, p]) =>
    i + ':' + (100 * p).toFixed(1) + '%' + (candidateSet.has(i) ? '*' : '')
  ).join(',');
  appendDiagnostic('PROBE_REPAIR n=' + n +
    ' violations=' + violatedConstraintCount(state.values) +
    ' energy=' + violationEnergy(state.values) +
    ' candidates=[' + candidates.join(',') + ']' +
    ' candidateMass=' + (100 * mass).toFixed(1) + '%' +
    ' top=[' + top + ']');

  const ids = [...new Set([0, Math.floor((n - 1) / 4), Math.floor((n - 1) / 2), Math.floor(3 * (n - 1) / 4), n - 1])];
  const values = ids.map(i => {
    const rawMean = DOMAIN_MAX * snap.meanNorm[i];
    const clippedMean = Math.max(0, Math.min(DOMAIN_MAX, rawMean));
    return i + ':pred=' + clippedMean.toFixed(2) + '/target=' + teacherMeanValue(i).toFixed(0) + '/sigma=' + Math.exp(snap.logStd[i]).toFixed(2);
  }).join(' | ');
  appendDiagnostic('PROBE_VALUES n=' + n + ' ' + values);
}

function appendRolloutTrace(n, seed = 940001, maxRepairLines = 16) {
  const rng = mulberry32(seed + n);
  const state = emptyState(n);
  const maxMoves = MOVE_MULTIPLIER * n;
  let moves = 0;
  while (moves < maxMoves && state.assignedCount < n) {
    const a = imitationAction(state, rng, true);
    applyAction(state, a.i, a.v);
    moves++;
  }
  appendDiagnostic('TRACE_START n=' + n +
    ' constructionMoves=' + moves +
    ' violations=' + violatedConstraintCount(state.values) +
    ' energy=' + violationEnergy(state.values));

  let logged = 0;
  while (moves < maxMoves && !isStrictChain(state.values) && logged < maxRepairLines) {
    const candidates = teacherCandidateIds(state);
    const candidateSet = new Set(candidates);
    const beforeViol = violatedConstraintCount(state.values);
    const beforeEnergy = violationEnergy(state.values);
    const a = imitationAction(state, rng, true);
    applyAction(state, a.i, a.v);
    moves++;
    const afterViol = violatedConstraintCount(state.values);
    const afterEnergy = violationEnergy(state.values);
    appendDiagnostic('TRACE_REPAIR step=' + logged +
      ' move=' + moves +
      ' choose=' + a.i +
      ' valid=' + (candidateSet.has(a.i) ? 1 : 0) +
      ' v=' + a.v +
      ' mean=' + a.meanValue.toFixed(2) +
      ' sigma=' + a.sigmaValue.toFixed(2) +
      ' viol=' + beforeViol + '->' + afterViol +
      ' energy=' + beforeEnergy + '->' + afterEnergy);
    logged++;
  }
  appendDiagnostic('TRACE_END n=' + n +
    ' solved=' + (isStrictChain(state.values) ? 1 : 0) +
    ' moves=' + moves +
    ' violations=' + violatedConstraintCount(state.values) +
    ' energy=' + violationEnergy(state.values));
}

function appendDiagnosticProbes() {
  const trainedN = trainedRange.max;
  appendDiagnostic('--- PROBES ---');
  appendRepairProbe(trainedN);
  if (trainedN < MAX_TEST_N) appendRepairProbe(Math.min(MAX_TEST_N, Math.max(trainedN + 1, trainedN * 2)));
  appendRolloutTrace(trainedN);
}
`;

  source += '\n//# sourceURL=app-gnn-imitation-diagnostics-runtime.js\n';
  const blobUrl = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  try {
    await import(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

bootDiagnostics().catch(err => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `Error loading imitation diagnostics: ${err.message}`;
  const log = document.getElementById('diagnosticLog');
  if (log) log.value = `DIAGNOSTIC_BOOT_ERROR\n${err.stack || err.message}`;
});
