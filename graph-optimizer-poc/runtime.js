'use strict';

function resetModel() {
  if (state.policy) state.policy.dispose();
  if (state.optimizer && typeof state.optimizer.dispose === 'function') {
    state.optimizer.dispose();
  }
  state.policy = new RecurrentGraphPolicy();
  state.optimizer = tf.train.adam(DEFAULT_LEARNING_RATE);
  state.trainedEpisodes = 0;
  ui.policyStateBadge.textContent = 'Untrained';
  ui.policyStateBadge.classList.add('muted');
  ui.compareBtn.disabled = true;
  ui.benchmarkBtn.disabled = true;
  ui.progressBar.style.width = '0%';
  ui.statusText.textContent = 'Model reset. Reward-only training starts from scratch.';
  clearResults();
}

function createCurrentProblem() {
  const n = clamp(readInt(ui.testN, 12), 4, 64);
  ui.testN.value = String(n);
  state.currentProblem = makeProblem(n);
  renderProblem(state.currentProblem);
  clearResults();
  setStatus(`New N=${n} problem created. Hidden state resets to zero for each optimizer run.`);
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

function updateMetrics(container, data, actionLabel) {
  const items = container.querySelectorAll('div');
  const values = [
    data.energy,
    data.steps,
    `${data.runtimeMs.toFixed(1)} ms`,
    data.success ? 'Yes' : 'No'
  ];
  items[1].querySelector('span').textContent = actionLabel;
  items.forEach((item, index) => {
    item.querySelector('strong').textContent = String(values[index]);
  });
}

function clearResults() {
  if (!state.currentProblem) return;
  renderChain(ui.policyGraph, state.currentProblem, state.currentProblem.initialValues);
  renderChain(ui.saGraph, state.currentProblem, state.currentProblem.initialValues);
  resetMetricPanel(ui.policyMetrics, 'Actions');
  resetMetricPanel(ui.saMetrics, 'Proposals');
  drawEnergyChart([], []);
  ui.benchmarkSummary.className = 'benchmark-empty';
  ui.benchmarkSummary.textContent = 'Run a 30-instance benchmark after reward training.';
}

function resetMetricPanel(container, actionLabel) {
  const labels = ['Energy', actionLabel, 'Runtime', 'Success'];
  container.querySelectorAll('div').forEach((item, index) => {
    item.querySelector('span').textContent = labels[index];
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

async function runLearnedPolicy(problem, requestedMaxSteps) {
  const maxSteps = Math.max(1, Math.floor(requestedMaxSteps));
  const values = problem.initialValues.slice();
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let bestValues = values.slice();
  const history = [currentEnergy];
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let lastNodeId = null;
  let steps = 0;
  const started = performance.now();

  for (let step = 0; step < maxSteps && currentEnergy > 0; step++) {
    if (state.stopRequested) break;

    const output = tf.tidy(() =>
      state.policy.forward(problem, values, hidden, step, maxSteps)
    );
    const logitsData = output.logits.dataSync();
    let actionIndex = 0;
    let bestLogit = Number.NEGATIVE_INFINITY;

    for (let index = 0; index < logitsData.length; index++) {
      if (logitsData[index] > bestLogit) {
        bestLogit = logitsData[index];
        actionIndex = index;
      }
    }

    hidden.dispose();
    hidden = output.hidden;
    output.logits.dispose();

    const action = applyAction(problem, values, actionIndex);
    lastNodeId = action.nodeId;
    steps++;
    currentEnergy = energy(problem, values);

    if (currentEnergy < bestEnergy) {
      bestEnergy = currentEnergy;
      bestValues = values.slice();
    }

    history.push(currentEnergy);
    if (steps % 20 === 0) await tf.nextFrame();
  }

  const hiddenNorms = hiddenNormsFromTensor(hidden, problem.n);
  hidden.dispose();

  return {
    values: bestValues,
    energy: bestEnergy,
    steps,
    runtimeMs: performance.now() - started,
    success: bestEnergy === 0,
    history,
    hiddenNorms,
    lastNodeId
  };
}

function runSimulatedAnnealing(problem, maxSteps, seed = problem.seed ^ 0xA5A5A5A5) {
  const rng = new SeededRandom(seed);
  const values = problem.initialValues.slice();
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let bestValues = values.slice();
  const history = [currentEnergy];
  const initialTemperature = Math.max(1, problem.n / 2);
  let steps = 0;
  let lastNodeId = null;
  const started = performance.now();

  for (let step = 0; step < maxSteps && currentEnergy > 0; step++) {
    if (state.stopRequested) break;
    const storageIndex = rng.int(problem.n);
    let candidateValue = rng.int(problem.domainSize);
    if (candidateValue === values[storageIndex]) {
      candidateValue = (candidateValue + 1 + rng.int(problem.domainSize - 1)) % problem.domainSize;
    }

    const oldValue = values[storageIndex];
    values[storageIndex] = candidateValue;
    const candidateEnergy = energy(problem, values);
    const delta = candidateEnergy - currentEnergy;
    const progress = maxSteps > 1 ? step / (maxSteps - 1) : 1;
    const temperature = initialTemperature * Math.pow(0.001, progress);
    const accept = delta <= 0 ||
      rng.next() < Math.exp(-delta / Math.max(1e-6, temperature));

    if (accept) {
      currentEnergy = candidateEnergy;
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
  readPositiveInt(ui.trainBudgetMultiplier, 4);

  if (minN > maxN) [minN, maxN] = [maxN, minN];
  ui.trainMinN.value = String(minN);
  ui.trainMaxN.value = String(maxN);
  setBusy(true);
  state.stopRequested = false;

  let movingLoss = null;
  let movingReturn = null;
  let movingFinalEnergy = null;
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
      recentSuccesses.push(stats.success ? 1 : 0);
      if (recentSuccesses.length > 100) recentSuccesses = recentSuccesses.slice(-100);

      state.trainedEpisodes++;
      if (episode === 1 || episode % 10 === 0 || episode === episodes) {
        const successRate = recentSuccesses.reduce((sum, value) => sum + value, 0) /
          Math.max(1, recentSuccesses.length);
        ui.progressBar.style.width = `${(episode / episodes) * 100}%`;
        setStatus(
          `Reward training ${episode}/${episodes} · loss ${movingLoss.toFixed(3)} · ` +
          `return ${movingReturn.toFixed(2)} · final energy ${movingFinalEnergy.toFixed(2)} · ` +
          `recent success ${(successRate * 100).toFixed(0)}%`
        );
        await tf.nextFrame();
      }
    }

    if (state.trainedEpisodes > 0) {
      ui.policyStateBadge.textContent = `${state.trainedEpisodes} RL episodes`;
      ui.policyStateBadge.classList.remove('muted');
      ui.compareBtn.disabled = false;
      ui.benchmarkBtn.disabled = false;
    }

    setStatus(
      state.stopRequested
        ? `Reward training stopped after ${state.trainedEpisodes} total episodes.`
        : `Reward training complete: ${state.trainedEpisodes} total episodes. Run a paired comparison.`
    );
  } catch (error) {
    console.error(error);
    setStatus(`Reward training failed: ${error.message}`, true);
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
    const policyBudget = readPositiveInt(ui.policyBudgetMultiplier, 6) * n;
    const saBudget = readPositiveInt(ui.saBudgetMultiplier, 250) * n;
    setStatus(
      `Running reward-trained policy (${policyBudget} actions) and ` +
      `SA (${saBudget} proposals) from the same assignment…`
    );

    const policyResult = await runLearnedPolicy(state.currentProblem, policyBudget);
    await tf.nextFrame();
    const saResult = runSimulatedAnnealing(state.currentProblem, saBudget);

    renderChain(ui.policyGraph, state.currentProblem, policyResult.values,
      policyResult.hiddenNorms, policyResult.lastNodeId);
    renderChain(ui.saGraph, state.currentProblem, saResult.values,
      null, saResult.lastNodeId);
    updateMetrics(ui.policyMetrics, policyResult, 'Actions');
    updateMetrics(ui.saMetrics, saResult, 'Proposals');
    drawEnergyChart(policyResult.history, saResult.history);
    setStatus(
      `Comparison complete. Policy: ${policyResult.success ? 'feasible' : 'not feasible'} ` +
      `in ${policyResult.steps} actions; SA: ${saResult.success ? 'feasible' : 'not feasible'} ` +
      `in ${saResult.steps} proposals.`
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
  const policyBudget = readPositiveInt(ui.policyBudgetMultiplier, 6) * n;
  const saBudget = readPositiveInt(ui.saBudgetMultiplier, 250) * n;
  const policyResults = [];
  const saResults = [];

  try {
    for (let index = 0; index < count; index++) {
      if (state.stopRequested) break;
      const problem = makeProblem(n, 500000 + index * 7919 + state.trainedEpisodes);
      policyResults.push(await runLearnedPolicy(problem, policyBudget));
      saResults.push(runSimulatedAnnealing(problem, saBudget, problem.seed ^ 0x55AA55AA));
      ui.progressBar.style.width = `${((index + 1) / count) * 100}%`;
      setStatus(`Benchmarking ${index + 1}/${count} instances at N=${n}…`);
      if ((index + 1) % 3 === 0) await tf.nextFrame();
    }

    renderBenchmark(n, policyResults, saResults, policyBudget, saBudget);
    setStatus(`Benchmark complete on ${policyResults.length} instances at N=${n}.`);
  } catch (error) {
    console.error(error);
    setStatus(`Benchmark failed: ${error.message}`, true);
  } finally {
    setBusy(false);
    state.stopRequested = false;
  }
}

function summarizeResults(results) {
  const successes = results.filter(result => result.success);
  return {
    successRate: results.length ? successes.length / results.length : 0,
    medianSteps: median(successes.map(result => result.steps)),
    medianRuntime: median(results.map(result => result.runtimeMs)),
    medianFinalEnergy: median(results.map(result => result.energy))
  };
}

function renderBenchmark(n, policyResults, saResults, policyBudget, saBudget) {
  const policy = summarizeResults(policyResults);
  const sa = summarizeResults(saResults);
  ui.benchmarkSummary.className = '';
  ui.benchmarkSummary.innerHTML = `
    <div class="benchmark-grid">
      <div class="benchmark-card"><span>Policy success</span><strong>${(policy.successRate * 100).toFixed(0)}%</strong></div>
      <div class="benchmark-card"><span>SA success</span><strong>${(sa.successRate * 100).toFixed(0)}%</strong></div>
      <div class="benchmark-card"><span>Policy median actions</span><strong>${policy.medianSteps ?? '–'}</strong></div>
      <div class="benchmark-card"><span>SA median proposals</span><strong>${sa.medianSteps ?? '–'}</strong></div>
    </div>
    <table class="benchmark-table">
      <thead><tr><th>Method</th><th>Budget</th><th>Success</th><th>Median steps</th><th>Median runtime</th><th>Median final energy</th></tr></thead>
      <tbody>
        <tr><td>Reward-trained recurrent policy</td><td>${policyBudget}</td><td>${(policy.successRate * 100).toFixed(1)}%</td><td>${policy.medianSteps ?? '–'}</td><td>${policy.medianRuntime?.toFixed(2) ?? '–'} ms</td><td>${policy.medianFinalEnergy ?? '–'}</td></tr>
        <tr><td>Simulated annealing</td><td>${saBudget}</td><td>${(sa.successRate * 100).toFixed(1)}%</td><td>${sa.medianSteps ?? '–'}</td><td>${sa.medianRuntime?.toFixed(2) ?? '–'} ms</td><td>${sa.medianFinalEnergy ?? '–'}</td></tr>
      </tbody>
    </table>
    <p class="caption">${policyResults.length} paired instances, N=${n}. Median step counts use successful runs only.</p>`;
}

function drawEnergyChart(policyHistory, saHistory) {
  const canvas = ui.energyChart;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const styles = getComputedStyle(document.documentElement);
  const textColor = styles.getPropertyValue('--muted').trim() || '#677085';
  const gridColor = styles.getPropertyValue('--border').trim() || '#dfe4ef';
  const policyColor = styles.getPropertyValue('--accent').trim() || '#4f46e5';
  const saColor = styles.getPropertyValue('--warning').trim() || '#9a6700';
  const pad = { left: 56, right: 18, top: 22, bottom: 42 };

  ctx.clearRect(0, 0, width, height);
  ctx.font = '14px system-ui, sans-serif';
  ctx.strokeStyle = gridColor;
  ctx.fillStyle = textColor;
  ctx.lineWidth = 1;

  const maxX = Math.max(1, policyHistory.length - 1, saHistory.length - 1);
  const maxY = Math.max(1, ...policyHistory, ...saHistory);
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
  ctx.fillText('Iteration', width / 2 - 24, height - 10);

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

  plot(saHistory, saColor);
  plot(policyHistory, policyColor);

  if (!policyHistory.length && !saHistory.length) {
    ctx.fillStyle = textColor;
    ctx.textAlign = 'center';
    ctx.fillText('Run a comparison to plot search progress.', width / 2, height / 2);
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
    ui.backendNotice.textContent = `TensorFlow.js ready · backend: ${tf.getBackend()} · reward-only training`;
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
