async function runActorCriticSearch(problem, requestedMaxSteps, seed = problem.seed) {
  ensureActorCriticParameters(state.policy);
  const maxSteps = Math.max(1, Math.floor(requestedMaxSteps));
  const actorRng = new SeededRandom(seed ^ 0xA17C0A11);
  const randomRng = new SeededRandom(seed ^ 0xBADC0FFE);
  const selectionRng = new SeededRandom(seed ^ 0xFACEFEED);
  const values = problem.initialValues.slice();
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let bestValues = values.slice();
  const history = [currentEnergy];
  const actionHistory = [];
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let lastNodeId = null;
  let lastDelta = 0;
  let lastReward = 0;
  let recentImprovement = 0;
  let stagnation = 0;
  let steps = 0;
  let candidateCount = 0;
  const started = performance.now();

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    if (state.stopRequested) break;
    if (step > 0 && step % POLICY_MEMORY_WINDOW === 0) {
      hidden.dispose();
      hidden = tf.zeros([problem.n, HIDDEN_DIM]);
    }

    const context = {
      currentEnergy,
      bestEnergy,
      violationCount: violationCount(problem, values),
      lastDelta,
      lastReward,
      recentImprovement,
      stagnation
    };
    const encoded = tf.tidy(() => state.policy.encodeActorCriticState(
      problem,
      values,
      hidden,
      step,
      maxSteps,
      context
    ));
    const generated = generateActorCriticCandidates(
      state.policy,
      problem,
      values,
      encoded,
      actorRng,
      randomRng
    );
    candidateCount = generated.candidates.length;
    const prepared = candidateDiagnostics(
      problem,
      values,
      encoded.context,
      generated.candidates
    );
    const output = tf.tidy(() => state.policy.scoreActorCriticCandidates(
      problem,
      values,
      encoded,
      generated.candidates,
      prepared
    ));
    const continuationValues = Array.from(output.continuationValues.dataSync());
    const totalScores = continuationValues.map((continuation, index) =>
      prepared.immediateRewards[index] + ACTOR_CRITIC_DISCOUNT * continuation
    );
    const selectedPosition = state.trainedEpisodes === 0
      ? argMax(prepared.immediateRewards)
      : sampleFromScores(totalScores, EVAL_SELECTION_TEMPERATURE, selectionRng);
    const actionIndex = generated.candidates[selectedPosition];

    hidden.dispose();
    hidden = encoded.hidden;
    encoded.nodeContext.dispose();
    output.actorLogits.dispose();
    output.continuationValues.dispose();

    actionHistory.push(actionIndex);
    const action = decodeAction(problem, actionIndex);
    const previousBest = bestEnergy;
    values[action.storageIndex] = action.value;
    const candidateEnergy = energy(problem, values);
    const delta = candidateEnergy - currentEnergy;
    const reward = prepared.immediateRewards[selectedPosition];
    currentEnergy = candidateEnergy;
    lastNodeId = action.nodeId;
    if (currentEnergy < bestEnergy) {
      bestEnergy = currentEnergy;
      bestValues = values.slice();
    }

    lastDelta = delta;
    lastReward = reward;
    recentImprovement = 0.90 * recentImprovement + 0.10 * reward;
    stagnation = bestEnergy < previousBest ? 0 : stagnation + 1;
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
    accepted: steps,
    acceptanceRate: steps ? 1 : 0,
    runtimeMs: performance.now() - started,
    success: bestEnergy === 0,
    history,
    actionHistory,
    hiddenNorms,
    lastNodeId,
    candidateCount
  };
}

runNeuralAnnealing = runActorCriticSearch;


async function runActorCriticComparison() {
  if (state.busy || !state.currentProblem) return;
  setBusy(true);
  state.stopRequested = false;

  try {
    const n = state.currentProblem.n;
    const proposalBudget = readPositiveInt(ui.proposalBudgetMultiplier, 8) * n;
    setStatus(
      `Running actor–critic graph search and random SA with ${proposalBudget} moves each…`
    );
    const actorCriticResult = await runActorCriticSearch(
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
      actorCriticResult.values,
      actorCriticResult.hiddenNorms,
      actorCriticResult.lastNodeId
    );
    renderChain(
      ui.saGraph,
      state.currentProblem,
      randomResult.values,
      null,
      randomResult.lastNodeId
    );
    updateMetrics(ui.policyMetrics, actorCriticResult);
    updateMetrics(ui.saMetrics, randomResult);
    drawEnergyChart(actorCriticResult.history, randomResult.history);
    setStatus(
      `Comparison complete. Actor–critic: ${actorCriticResult.success ? 'feasible' : 'not feasible'}, ` +
      `best energy ${actorCriticResult.energy}, ${actorCriticResult.steps} moves. Random SA: ` +
      `${randomResult.success ? 'feasible' : 'not feasible'}, best energy ${randomResult.energy}, ` +
      `${randomResult.accepted}/${randomResult.steps} accepted.` + trainingEnvelopeWarning()
    );
  } catch (error) {
    console.error(error);
    setStatus(`Comparison failed: ${error.message}`, true);
  } finally {
    setBusy(false);
    state.stopRequested = false;
  }
}

function renderActorCriticBenchmark(n, actorCriticResults, randomResults, proposalBudget) {
  const learned = summarizeResults(actorCriticResults);
  const random = summarizeResults(randomResults);
  ui.benchmarkSummary.className = '';
  ui.benchmarkSummary.innerHTML = `
    <div class="benchmark-grid">
      <div class="benchmark-card"><span>Actor–critic success</span><strong>${(learned.successRate * 100).toFixed(0)}%</strong></div>
      <div class="benchmark-card"><span>Random SA success</span><strong>${(random.successRate * 100).toFixed(0)}%</strong></div>
      <div class="benchmark-card"><span>Actor–critic median moves</span><strong>${learned.medianSteps ?? '–'}</strong></div>
      <div class="benchmark-card"><span>Random SA median proposals</span><strong>${random.medianSteps ?? '–'}</strong></div>
    </div>
    <table class="benchmark-table">
      <thead><tr><th>Method</th><th>Budget</th><th>Success</th><th>Median moves</th><th>Applied / accepted</th><th>Median runtime</th><th>Median final energy</th></tr></thead>
      <tbody>
        <tr><td>Actor–critic graph search</td><td>${proposalBudget}</td><td>${(learned.successRate * 100).toFixed(1)}%</td><td>${learned.medianSteps ?? '–'}</td><td>100%</td><td>${learned.medianRuntime?.toFixed(2) ?? '–'} ms</td><td>${learned.medianFinalEnergy ?? '–'}</td></tr>
        <tr><td>Random simulated annealing</td><td>${proposalBudget}</td><td>${(random.successRate * 100).toFixed(1)}%</td><td>${random.medianSteps ?? '–'}</td><td>${random.medianAcceptance === null ? '–' : `${(random.medianAcceptance * 100).toFixed(1)}%`}</td><td>${random.medianRuntime?.toFixed(2) ?? '–'} ms</td><td>${random.medianFinalEnergy ?? '–'}</td></tr>
      </tbody>
    </table>
    <p class="caption">${actorCriticResults.length} paired instances, N=${n}. Both methods share the initial assignment, move budget, and objective; their search policies differ.</p>`;
}

async function runActorCriticBenchmark() {
  if (state.busy) return;
  setBusy(true);
  state.stopRequested = false;

  const count = 30;
  const n = clamp(readInt(ui.testN, 8), 4, 64);
  const proposalBudget = readPositiveInt(ui.proposalBudgetMultiplier, 8) * n;
  const actorCriticResults = [];
  const randomResults = [];

  try {
    for (let index = 0; index < count; index++) {
      if (state.stopRequested) break;
      const problem = makeProblem(n, 500000 + index * 7919 + state.trainedEpisodes);
      actorCriticResults.push(await runActorCriticSearch(problem, proposalBudget, problem.seed));
      randomResults.push(runRandomAnnealing(problem, proposalBudget, problem.seed));
      ui.progressBar.style.width = `${((index + 1) / count) * 100}%`;
      setStatus(`Benchmarking ${index + 1}/${count} actor–critic vs. random-SA runs at N=${n}…`);
      if ((index + 1) % 3 === 0) await tf.nextFrame();
    }

    renderActorCriticBenchmark(n, actorCriticResults, randomResults, proposalBudget);
    setStatus(
      `Benchmark complete on ${actorCriticResults.length} paired instances at N=${n}.` +
      trainingEnvelopeWarning()
    );
  } catch (error) {
    console.error(error);
    setStatus(`Benchmark failed: ${error.message}`, true);
  } finally {
    setBusy(false);
    state.stopRequested = false;
  }
}

