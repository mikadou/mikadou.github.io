'use strict';

function resetModel() {
  if (state.policy) state.policy.dispose();
  if (state.optimizer && typeof state.optimizer.dispose === 'function') {
    state.optimizer.dispose();
  }
  state.policy = new RecurrentGraphPolicy();
  state.optimizer = tf.train.adam(DEFAULT_LEARNING_RATE);
  state.trainedEpisodes = 0;
  ui.policyStateBadge.textContent = 'Untrained proposer';
  ui.policyStateBadge.classList.add('muted');
  ui.compareBtn.disabled = true;
  ui.benchmarkBtn.disabled = true;
  ui.progressBar.style.width = '0%';
  ui.statusText.textContent = 'Model reset. Neural mutation training starts from scratch.';
  clearResults();
}

function createCurrentProblem() {
  const n = clamp(readInt(ui.testN, 12), 4, 64);
  ui.testN.value = String(n);
  state.currentProblem = makeProblem(n);
  renderProblem(state.currentProblem);
  clearResults();
  setStatus(`New N=${n} problem created. Both SA variants start from this assignment.`);
}

function renderProblem(problem) {
  ui.domainLabel.textContent = `0…${problem.domainMax}`;
  ui.storageLabel.textContent = problem.nodeIds.join(', ');
  renderChain(ui.initialGraph, problem, problem.initialValues);
  renderChain(ui.policyGraph, problem, problem.initialValues);
  renderChain(ui.saGraph, problem, problem.initialValues);
}

function renderChain(container, problem, values, hiddenNorms = null, recentNodeId = null) {
  container.replaceChildren();
  const maxNorm = hiddenNorms && hiddenNorms.length
    ? Math.max(1e-6, ...hiddenNorms)
    : 1;

  for (let depth = 0; depth < problem.n; depth++) {
    const id = problem.chain[depth];
    const storageIndex = problem.indexById.get(id);
    const node = document.createElement('div');
    node.className = `chain-node${recentNodeId === id ? ' recent' : ''}`;

    const idLabel = document.createElement('span');
    idLabel.className = 'node-id';
    idLabel.textContent = id;

    const valueLabel = document.createElement('span');
    valueLabel.className = 'node-value';
    valueLabel.textContent = String(values[storageIndex]);

    const depthLabel = document.createElement('span');
    depthLabel.className = 'node-depth';
    depthLabel.textContent = `depth ${depth}`;

    node.append(idLabel, valueLabel, depthLabel);

    if (hiddenNorms) {
      const track = document.createElement('div');
      track.className = 'hidden-track';
      const bar = document.createElement('div');
      bar.className = 'hidden-bar';
      bar.style.width = `${Math.min(100, (hiddenNorms[storageIndex] / maxNorm) * 100)}%`;
      track.append(bar);
      node.append(track);
    }

    container.append(node);

    if (depth < problem.n - 1) {
      const nextValue = values[problem.indexById.get(problem.chain[depth + 1])];
      const ok = values[storageIndex] > nextValue;
      const edge = document.createElement('div');
      edge.className = `chain-edge ${ok ? 'ok' : 'bad'}`;
      edge.innerHTML = `&gt;<small>${ok ? 'ok' : 'violation'}</small>`;
      container.append(edge);
    }
  }
}

function updateMetrics(container, data) {
  const items = container.querySelectorAll('div');
  const values = [
    data.energy,
    data.steps,
    `${data.runtimeMs.toFixed(1)} ms`,
    data.success ? 'Yes' : 'No'
  ];
  items.forEach((item, index) => {
    item.querySelector('strong').textContent = String(values[index]);
  });
}

function clearResults() {
  if (!state.currentProblem) return;
  renderChain(ui.policyGraph, state.currentProblem, state.currentProblem.initialValues);
  renderChain(ui.saGraph, state.currentProblem, state.currentProblem.initialValues);
  resetMetricPanel(ui.policyMetrics);
  resetMetricPanel(ui.saMetrics);
  drawEnergyChart([], []);
  ui.benchmarkSummary.className = 'benchmark-empty';
  ui.benchmarkSummary.textContent = 'Run a 30-instance benchmark after training the neural mutation proposer.';
}

function resetMetricPanel(container) {
  container.querySelectorAll('div').forEach(item => {
    item.querySelector('strong').textContent = '–';
  });
}

function hiddenNormsFromTensor(hiddenTensor, n) {
  const data = hiddenTensor.dataSync();
  const norms = new Array(n).fill(0);
  for (let node = 0; node < n; node++) {
    let sum = 0;
    for (let channel = 0; channel < HIDDEN_DIM; channel++) {
      const value = data[node * HIDDEN_DIM + channel];
      sum += value * value;
    }
    norms[node] = Math.sqrt(sum);
  }
  return norms;
}

async function runNeuralAnnealing(problem, requestedMaxSteps, seed = problem.seed) {
  const maxSteps = Math.max(1, Math.floor(requestedMaxSteps));
  const proposalRng = new SeededRandom(seed ^ 0x13579BDF);
  const acceptanceRng = new SeededRandom(seed ^ 0x2468ACE0);
  const values = problem.initialValues.slice();
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let bestValues = values.slice();
  const history = [currentEnergy];
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let lastNodeId = null;
  let lastAccepted = true;
  let lastDelta = 0;
  let acceptedCount = 0;
  let steps = 0;
  const started = performance.now();

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    if (state.stopRequested) break;

    const temperature = annealingTemperature(problem, step, maxSteps);
    const context = { temperature, lastAccepted, lastDelta };
    const output = tf.tidy(() =>
      state.policy.forward(problem, values, hidden, step, maxSteps, context)
    );
    const actionIndex = samplePolicyAction(
      output.logits,
      EVAL_PROPOSAL_TEMPERATURE,
      proposalRng.int(0x7fffffff)
    );

    hidden.dispose();
    hidden = output.hidden;
    output.logits.dispose();

    const action = decodeAction(problem, actionIndex);
    const oldValue = values[action.storageIndex];
    values[action.storageIndex] = action.value;
    const candidateEnergy = energy(problem, values);
    const delta = candidateEnergy - currentEnergy;
    const acceptanceProbability = annealingAcceptanceProbability(delta, temperature);
    const accepted = acceptanceRng.next() < acceptanceProbability;

    if (accepted) {
      currentEnergy = candidateEnergy;
      acceptedCount++;
      lastNodeId = action.nodeId;
      if (currentEnergy < bestEnergy) {
        bestEnergy = currentEnergy;
        bestValues = values.slice();
      }
    } else {
      values[action.storageIndex] = oldValue;
    }

    lastAccepted = accepted;
    lastDelta = delta;
    steps++;
    history.push(currentEnergy);
    if (steps % 20 === 0) await tf.nextFrame();
  }

  const hiddenNorms = hiddenNormsFromTensor(hidden, problem.n);
  hidden.dispose();

  return {
    values: bestValues,
    energy: bestEnergy,
    steps,
    accepted: acceptedCount,
    acceptanceRate: steps ? acceptedCount / steps : 0,
    runtimeMs: performance.now() - started,
    success: bestEnergy === 0,
    history,
    hiddenNorms,
    lastNodeId
  };
}

function runRandomAnnealing(problem, requestedMaxSteps, seed = problem.seed) {
  const maxSteps = Math.max(1, Math.floor(requestedMaxSteps));
  const proposalRng = new SeededRandom(seed ^ 0x13579BDF);
  const acceptanceRng = new SeededRandom(seed ^ 0x2468ACE0);
  const values = problem.initialValues.slice();
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let bestValues = values.slice();
  const history = [currentEnergy];
  let lastNodeId = null;
  let acceptedCount = 0;
  let steps = 0;
  const started = performance.now();

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    if (state.stopRequested) break;

    const storageIndex = proposalRng.int(problem.n);
    let candidateValue = proposalRng.int(problem.domainSize);
    if (candidateValue === values[storageIndex]) {
      candidateValue =
        (candidateValue + 1 + proposalRng.int(problem.domainSize - 1)) %
        problem.domainSize;
    }

    const oldValue = values[storageIndex];
    values[storageIndex] = candidateValue;
    const candidateEnergy = energy(problem, values);
    const delta = candidateEnergy - currentEnergy;
    const temperature = annealingTemperature(problem, step, maxSteps);
    const acceptanceProbability = annealingAcceptanceProbability(delta, temperature);
    const accepted = acceptanceRng.next() < acceptanceProbability;

    if (accepted) {
      currentEnergy = candidateEnergy;
      acceptedCount++;
      lastNodeId = problem.nodeIds[storageIndex];
      if (currentEnergy < bestEnergy) {
        bestEnergy = currentEnergy;
        bestValues = values.slice();
      }
    } else {
      values[storageIndex] = oldValue;
    }

    steps++;
    history.push(currentEnergy);
  }

  return {
    values: bestValues,
    energy: bestEnergy,
    steps,
    accepted: acceptedCount,
    acceptanceRate: steps ? acceptedCount / steps : 0,
    runtimeMs: performance.now() - started,
    success: bestEnergy === 0,
    history,
    lastNodeId
  };
}

async function trainPolicy() {
  if (state.busy) return;

  const episodes = clamp(readInt(ui.trainEpisodes, 2000), 50, 10000);
  let minN = clamp(readInt(ui.trainMinN, 4), 3, 32);
  let maxN = clamp(readInt(ui.trainMaxN, 8), 4, 40);
  readPositiveInt(ui.trainBudgetMultiplier, 8);

  if (minN > maxN) [minN, maxN] = [maxN, minN];
  ui.trainMinN.value = String(minN);
  ui.trainMaxN.value = String(maxN);
  setBusy(true);
  state.stopRequested = false;

  let movingLoss = null;
  let movingReturn = null;
  let movingFinalEnergy = null;
  let movingAcceptance = null;
  let recentSuccesses = [];

  try {
    for (let episode = 1; episode <= episodes; episode++) {
      if (state.stopRequested) break;
      const n = minN + Math.floor(Math.random() * (maxN - minN + 1));
      const loss = state.policy.trainEpisode(makeProblem(n), state.optimizer);
      const stats = state.policy.lastRewardStats;

      movingLoss = movingLoss === null ? loss : movingLoss * 0.95 + loss * 0.05;
      movingReturn = movingReturn === null
        ? stats.episodeReturn
        : movingReturn * 0.95 + stats.episodeReturn * 0.05;
      movingFinalEnergy = movingFinalEnergy === null
        ? stats.finalEnergy
        : movingFinalEnergy * 0.95 + stats.finalEnergy * 0.05;
      movingAcceptance = movingAcceptance === null
        ? stats.acceptanceRate
        : movingAcceptance * 0.95 + stats.acceptanceRate * 0.05;
      recentSuccesses.push(stats.success ? 1 : 0);
      if (recentSuccesses.length > 100) recentSuccesses = recentSuccesses.slice(-100);

      state.trainedEpisodes++;
      if (episode === 1 || episode % 10 === 0 || episode === episodes) {
        const successRate = recentSuccesses.reduce((sum, value) => sum + value, 0) /
          Math.max(1, recentSuccesses.length);
        ui.progressBar.style.width = `${(episode / episodes) * 100}%`;
        setStatus(
          `Neural-SA training ${episode}/${episodes} · loss ${movingLoss.toFixed(3)} · ` +
          `return ${movingReturn.toFixed(2)} · final energy ${movingFinalEnergy.toFixed(2)} · ` +
          `accepted ${(movingAcceptance * 100).toFixed(0)}% · ` +
          `recent success ${(successRate * 100).toFixed(0)}%`
        );
        await tf.nextFrame();
      }
    }

    if (state.trainedEpisodes > 0) {
      ui.policyStateBadge.textContent = `${state.trainedEpisodes} training episodes`;
      ui.policyStateBadge.classList.remove('muted');
      ui.compareBtn.disabled = false;
      ui.benchmarkBtn.disabled = false;
    }

    setStatus(
      state.stopRequested
        ? `Neural mutation training stopped after ${state.trainedEpisodes} total episodes.`
        : `Neural mutation training complete: ${state.trainedEpisodes} total episodes.`
    );
  } catch (error) {
    console.error(error);
    setStatus(`Neural mutation training failed: ${error.message}`, true);
  } finally {
    setBusy(false);
    state.stopRequested = false;
  }
}

async function runComparison() {
  if (state.busy || !state.currentProblem || state.trainedEpisodes === 0) return;
  setBusy(true);
  state.stopRequested = false;

  try {
    const n = state.currentProblem.n;
    const proposalBudget = readPositiveInt(ui.proposalBudgetMultiplier, 250) * n;
    setStatus(
      `Running neural-mutation SA and random-mutation SA with ${proposalBudget} proposals each…`
    );

    const neuralResult = await runNeuralAnnealing(
      state.currentProblem,
      proposalBudget,
      state.currentProblem.seed
    );
    await tf.nextFrame();
    const randomResult = runRandomAnnealing(
      state.currentProblem,
      proposalBudget,
      state.currentProblem.seed
    );

    renderChain(
      ui.policyGraph,
      state.currentProblem,
      neuralResult.values,
      neuralResult.hiddenNorms,
      neuralResult.lastNodeId
    );
    renderChain(
      ui.saGraph,
      state.currentProblem,
      randomResult.values,
      null,
      randomResult.lastNodeId
    );
    updateMetrics(ui.policyMetrics, neuralResult);
    updateMetrics(ui.saMetrics, randomResult);
    drawEnergyChart(neuralResult.history, randomResult.history);
    setStatus(
      `Comparison complete. Neural SA: ${neuralResult.success ? 'feasible' : 'not feasible'}, ` +
      `${neuralResult.accepted}/${neuralResult.steps} accepted. Random SA: ` +
      `${randomResult.success ? 'feasible' : 'not feasible'}, ` +
      `${randomResult.accepted}/${randomResult.steps} accepted.`
    );
  } catch (error) {
    console.error(error);
    setStatus(`Comparison failed: ${error.message}`, true);
  } finally {
    setBusy(false);
    state.stopRequested = false;
  }
}

function median(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

async function runBenchmark() {
  if (state.busy || state.trainedEpisodes === 0) return;
  setBusy(true);
  state.stopRequested = false;

  const count = 30;
  const n = clamp(readInt(ui.testN, 12), 4, 64);
  const proposalBudget = readPositiveInt(ui.proposalBudgetMultiplier, 250) * n;
  const neuralResults = [];
  const randomResults = [];

  try {
    for (let index = 0; index < count; index++) {
      if (state.stopRequested) break;
      const problem = makeProblem(n, 500000 + index * 7919 + state.trainedEpisodes);
      neuralResults.push(await runNeuralAnnealing(problem, proposalBudget, problem.seed));
      randomResults.push(runRandomAnnealing(problem, proposalBudget, problem.seed));
      ui.progressBar.style.width = `${((index + 1) / count) * 100}%`;
      setStatus(`Benchmarking ${index + 1}/${count} paired SA runs at N=${n}…`);
      if ((index + 1) % 3 === 0) await tf.nextFrame();
    }

    renderBenchmark(n, neuralResults, randomResults, proposalBudget);
    setStatus(`Benchmark complete on ${neuralResults.length} paired instances at N=${n}.`);
  } catch (error) {
    console.error(error);
    setStatus(`Benchmark failed: ${error.message}`, true);
  } finally {
    setBusy(false);
    state.stopRequested = false;
  }
}

function median(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function summarizeResults(results) {
  const successes = results.filter(result => result.success);
  return {
    successRate: results.length ? successes.length / results.length : 0,
    medianSteps: median(successes.map(result => result.steps)),
    medianRuntime: median(results.map(result => result.runtimeMs)),
    medianFinalEnergy: median(results.map(result => result.energy)),
    medianAcceptance: median(results.map(result => result.acceptanceRate))
  };
}

function renderBenchmark(n, neuralResults, randomResults, proposalBudget) {
  const neural = summarizeResults(neuralResults);
  const random = summarizeResults(randomResults);
  ui.benchmarkSummary.className = '';
  ui.benchmarkSummary.innerHTML = `
    <div class="benchmark-grid">
      <div class="benchmark-card"><span>Neural SA success</span><strong>${(neural.successRate * 100).toFixed(0)}%</strong></div>
      <div class="benchmark-card"><span>Random SA success</span><strong>${(random.successRate * 100).toFixed(0)}%</strong></div>
      <div class="benchmark-card"><span>Neural median proposals</span><strong>${neural.medianSteps ?? '–'}</strong></div>
      <div class="benchmark-card"><span>Random median proposals</span><strong>${random.medianSteps ?? '–'}</strong></div>
    </div>
    <table class="benchmark-table">
      <thead><tr><th>Method</th><th>Budget</th><th>Success</th><th>Median proposals</th><th>Acceptance</th><th>Median runtime</th><th>Median final energy</th></tr></thead>
      <tbody>
        <tr><td>SA + neural mutation</td><td>${proposalBudget}</td><td>${(neural.successRate * 100).toFixed(1)}%</td><td>${neural.medianSteps ?? '–'}</td><td>${neural.medianAcceptance === null ? '–' : `${(neural.medianAcceptance * 100).toFixed(1)}%`}</td><td>${neural.medianRuntime?.toFixed(2) ?? '–'} ms</td><td>${neural.medianFinalEnergy ?? '–'}</td></tr>
        <tr><td>SA + random mutation</td><td>${proposalBudget}</td><td>${(random.successRate * 100).toFixed(1)}%</td><td>${random.medianSteps ?? '–'}</td><td>${random.medianAcceptance === null ? '–' : `${(random.medianAcceptance * 100).toFixed(1)}%`}</td><td>${random.medianRuntime?.toFixed(2) ?? '–'} ms</td><td>${random.medianFinalEnergy ?? '–'}</td></tr>
      </tbody>
    </table>
    <p class="caption">${neuralResults.length} paired instances, N=${n}. Both methods use the same initial assignments, proposal budget, temperature schedule, and SA acceptance rule.</p>`;
}

function drawEnergyChart(neuralHistory, randomHistory) {
  const canvas = ui.energyChart;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const styles = getComputedStyle(document.documentElement);
  const textColor = styles.getPropertyValue('--muted').trim() || '#677085';
  const gridColor = styles.getPropertyValue('--border').trim() || '#dfe4ef';
  const neuralColor = styles.getPropertyValue('--accent').trim() || '#4f46e5';
  const randomColor = styles.getPropertyValue('--warning').trim() || '#9a6700';
  const pad = { left: 56, right: 18, top: 22, bottom: 42 };

  ctx.clearRect(0, 0, width, height);
  ctx.font = '14px system-ui, sans-serif';
  ctx.strokeStyle = gridColor;
  ctx.fillStyle = textColor;
  ctx.lineWidth = 1;

  const maxX = Math.max(1, neuralHistory.length - 1, randomHistory.length - 1);
  const maxY = Math.max(1, ...neuralHistory, ...randomHistory);
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;

  for (let tick = 0; tick <= 4; tick++) {
    const y = pad.top + (plotHeight * tick) / 4;
    const value = Math.round(maxY * (1 - tick / 4));
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    ctx.fillText(String(value), 10, y + 5);
  }

  ctx.fillText('0', pad.left - 4, height - 12);
  ctx.fillText(String(maxX), width - pad.right - 28, height - 12);
  ctx.fillText('Proposal', width / 2 - 24, height - 10);

  function plot(history, color) {
    if (history.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    history.forEach((value, index) => {
      const x = pad.left + (index / maxX) * plotWidth;
      const y = pad.top + (1 - value / maxY) * plotHeight;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  plot(randomHistory, randomColor);
  plot(neuralHistory, neuralColor);

  if (!neuralHistory.length && !randomHistory.length) {
    ctx.fillStyle = textColor;
    ctx.textAlign = 'center';
    ctx.fillText('Run a comparison to plot SA search progress.', width / 2, height / 2);
    ctx.textAlign = 'start';
  }
}

function setStatus(message, isError = false) {
  ui.statusText.textContent = message;
  ui.statusText.style.color = isError ? 'var(--danger)' : '';
}

function setBusy(busy) {
  state.busy = busy;
  ui.newProblemBtn.disabled = busy;
  ui.trainBtn.disabled = busy;
  ui.compareBtn.disabled = busy || state.trainedEpisodes === 0;
  ui.benchmarkBtn.disabled = busy || state.trainedEpisodes === 0;
  ui.resetModelBtn.disabled = busy;
  ui.stopBtn.disabled = !busy;
}

async function initialize() {
  try {
    await tf.ready();
    ui.backendNotice.textContent =
      `TensorFlow.js ready · backend: ${tf.getBackend()} · shared simulated-annealing loop`;
    resetModel();
    createCurrentProblem();
  } catch (error) {
    console.error(error);
    ui.backendNotice.textContent = `TensorFlow.js failed to initialize: ${error.message}`;
    ui.backendNotice.classList.add('error');
    ui.trainBtn.disabled = true;
  }
}

ui.newProblemBtn.addEventListener('click', createCurrentProblem);
ui.trainBtn.addEventListener('click', trainPolicy);
ui.compareBtn.addEventListener('click', runComparison);
ui.benchmarkBtn.addEventListener('click', runBenchmark);
ui.stopBtn.addEventListener('click', () => {
  state.stopRequested = true;
  setStatus('Stopping after the current operation…');
});
ui.resetModelBtn.addEventListener('click', resetModel);
ui.testN.addEventListener('change', createCurrentProblem);
window.matchMedia('(prefers-color-scheme: dark)')
  .addEventListener('change', () => drawEnergyChart([], []));

initialize();
