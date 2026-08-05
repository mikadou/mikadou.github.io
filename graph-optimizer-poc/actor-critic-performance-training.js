'use strict';

function localTeacherProbability() {
  return Math.max(
    LOCAL_TEACHER_MIN_PROBABILITY,
    1 - state.trainedEpisodes / LOCAL_TEACHER_DECAY_EPISODES
  );
}

function localChooseTeacherAction(problem, values, teacher) {
  const order = teacher.nodeScores
    .map((score, node) => ({ score, node }))
    .sort((left, right) => right.score - left.score);
  for (const item of order) {
    const value = Math.round(
      teacher.targetValues[item.node] * Math.max(1, problem.domainMax)
    );
    if (value !== values[item.node]) {
      return item.node * problem.domainSize + value;
    }
  }
  return sampleUniformMutationAction(
    problem,
    values,
    new SeededRandom(problem.seed ^ 0x10CA1)
  );
}

function localSampleCriticReplay(policy, rng) {
  const count = Math.min(LOCAL_CRITIC_BATCH_SIZE, policy.stateCriticReplay.length);
  const selected = [];
  const seen = new Set();
  while (selected.length < count) {
    const index = rng.int(policy.stateCriticReplay.length);
    if (seen.has(index)) continue;
    seen.add(index);
    selected.push(policy.stateCriticReplay[index]);
  }
  return selected;
}

RecurrentGraphPolicy.prototype.trainEpisode = function trainLocalActorEpisode(
  problem,
  optimizer
) {
  ensureLocalActor(this);
  const maxSteps = readPositiveInt(ui.trainBudgetMultiplier, 2) * problem.n;
  const rolloutRng = new SeededRandom(problem.seed ^ 0x10CA1AC7);
  const actorRng = new SeededRandom(problem.seed ^ 0xA17C0A11);
  const randomRng = new SeededRandom(problem.seed ^ 0xBADC0FFE);
  const replayRng = new SeededRandom(problem.seed ^ 0xC71C1C00);
  const values = problem.initialValues.slice();
  const trainingStates = [];
  const rewards = [];
  const replayItems = [];
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let lastDelta = 0;
  let lastReward = 0;
  let recentImprovement = 0;
  let stagnation = 0;
  let teacherMoves = 0;

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
    const context = actorCriticContext(problem, values, step, maxSteps, {
      currentEnergy,
      bestEnergy,
      violationCount: violationCount(problem, values),
      lastDelta,
      lastReward,
      recentImprovement,
      stagnation
    });
    const teacher = localActorTeacher(problem, values, context);
    trainingStates.push({
      features: teacher.features,
      nodeTargets: teacher.nodeTargets,
      targetValues: teacher.targetValues,
      valueWeights: teacher.valueWeights
    });

    let actionIndex;
    if (rolloutRng.next() < localTeacherProbability()) {
      actionIndex = localChooseTeacherAction(problem, values, teacher);
      teacherMoves++;
    } else {
      const encoded = {
        hidden: tf.zeros([problem.n, HIDDEN_DIM]),
        nodeContext: tf.tensor2d(
          teacher.features,
          [problem.n, LOCAL_ACTOR_FEATURE_DIM]
        ),
        context
      };
      const generated = generateActorCriticCandidates(
        this,
        problem,
        values,
        encoded,
        actorRng,
        randomRng
      );
      const prepared = candidateDiagnostics(
        problem,
        values,
        context,
        generated.candidates
      );
      const output = tf.tidy(() => this.scoreActorCriticCandidates(
        problem,
        values,
        encoded,
        generated.candidates,
        prepared
      ));
      const continuations = Array.from(output.continuationValues.dataSync());
      const scores = continuations.map((continuation, index) =>
        prepared.immediateRewards[index] + ACTOR_CRITIC_DISCOUNT * continuation
      );
      actionIndex = generated.candidates[argMax(scores)];
      encoded.hidden.dispose();
      encoded.nodeContext.dispose();
      output.actorLogits.dispose();
      output.continuationValues.dispose();
    }

    const diagnostics = localMutationDiagnostics(
      problem,
      values,
      Math.floor(actionIndex / problem.domainSize),
      actionIndex % problem.domainSize,
      { currentEnergy, bestEnergy, temperature: 1 }
    );
    let reward = -diagnostics.deltaEnergy / Math.max(1, problem.n);
    if (currentEnergy > 0 && diagnostics.candidateEnergy === 0) {
      reward += TERMINAL_SUCCESS_REWARD;
    }

    const action = decodeAction(problem, actionIndex);
    const previousBest = bestEnergy;
    values[action.storageIndex] = action.value;
    currentEnergy = energy(problem, values);
    bestEnergy = Math.min(bestEnergy, currentEnergy);
    lastDelta = diagnostics.deltaEnergy;
    lastReward = reward;
    recentImprovement = 0.90 * recentImprovement + 0.10 * reward;
    stagnation = bestEnergy < previousBest ? 0 : stagnation + 1;
    rewards.push(reward);

    const nextContext = {
      currentEnergy,
      bestEnergy,
      violationCount: violationCount(problem, values),
      lastDelta,
      lastReward,
      recentImprovement,
      stagnation
    };
    replayItems.push({
      features: stableCriticStateFeatures(
        problem,
        values,
        step + 1,
        maxSteps,
        nextContext
      ),
      target: 0
    });
  }

  if (!trainingStates.length) {
    this.lastRewardStats = {
      episodeReturn: 0,
      finalEnergy: 0,
      success: true,
      steps: 0,
      acceptanceRate: 1,
      selectedUtility: 0,
      candidateCount: LOCAL_ACTOR_CANDIDATE_COUNT + LOCAL_RANDOM_CANDIDATE_COUNT,
      actorLoss: 0,
      criticLoss: 0,
      replaySize: this.stateCriticReplay.length,
      criticInfluence: performanceCriticInfluence(this)
    };
    return 0;
  }

  const continuationTargets = computeContinuationTargets(rewards);
  replayItems.forEach((item, index) => {
    item.target = continuationTargets[index];
  });
  stableAddReplay(this, replayItems);

  const actorResult = tf.tidy(() => tf.variableGrads(() => {
    let totalLoss = tf.scalar(0);
    trainingStates.forEach(item => {
      const features = tf.tensor2d(
        item.features,
        [problem.n, LOCAL_ACTOR_FEATURE_DIM]
      );
      const distribution = this.actorDistribution({ nodeContext: features });
      const nodeTargets = tf.tensor1d(item.nodeTargets);
      const nodeLoss = nodeTargets.mul(
        tf.logSoftmax(distribution.nodeLogits)
      ).sum().neg();
      const targetValues = tf.tensor1d(item.targetValues);
      const valueWeights = tf.tensor1d(item.valueWeights);
      const valueError = distribution.valueMeans.sub(targetValues).square();
      const valueLoss = valueError.mul(valueWeights).sum()
        .div(valueWeights.sum().add(1e-6));
      totalLoss = totalLoss.add(
        nodeLoss.add(valueLoss.mul(LOCAL_VALUE_LOSS_WEIGHT))
      );
    });
    return totalLoss.div(trainingStates.length);
  }, this.actorVars));
  const actorLoss = stableApplyGradients(optimizer, actorResult);

  let criticLoss = 0;
  const criticBatch = localSampleCriticReplay(this, replayRng);
  if (criticBatch.length) {
    const featureData = new Float32Array(
      criticBatch.length * STABLE_CRITIC_STATE_DIM
    );
    const targetData = new Float32Array(criticBatch.length);
    criticBatch.forEach((item, index) => {
      featureData.set(item.features, index * STABLE_CRITIC_STATE_DIM);
      targetData[index] = item.target;
    });
    const criticResult = tf.tidy(() => tf.variableGrads(() => {
      const features = tf.tensor2d(
        featureData,
        [criticBatch.length, STABLE_CRITIC_STATE_DIM]
      );
      const targets = tf.tensor1d(targetData);
      return stableHuber(this.stateCriticValues(features).sub(targets)).mean();
    }, this.stateCriticVars));
    criticLoss = stableApplyGradients(optimizer, criticResult);
  }

  const episodeReturn = rewards.reduce((sum, reward) => sum + reward, 0);
  this.lastRewardStats = {
    episodeReturn,
    finalEnergy: bestEnergy,
    success: bestEnergy === 0,
    steps: trainingStates.length,
    acceptanceRate: 1,
    selectedUtility: episodeReturn / trainingStates.length,
    candidateCount: LOCAL_ACTOR_CANDIDATE_COUNT + LOCAL_RANDOM_CANDIDATE_COUNT,
    actorLoss,
    criticLoss,
    meanReward: episodeReturn / trainingStates.length,
    replaySize: this.stateCriticReplay.length,
    criticInfluence: performanceCriticInfluence(this),
    teacherRate: teacherMoves / trainingStates.length
  };
  return actorLoss + criticLoss;
};

async function runLocalActorSearch(problem, requestedMaxSteps, seed = problem.seed) {
  ensureLocalActor(state.policy);
  const maxSteps = Math.max(1, Math.floor(requestedMaxSteps));
  const actorRng = new SeededRandom(seed ^ 0xA17C0A11);
  const randomRng = new SeededRandom(seed ^ 0xBADC0FFE);
  const values = problem.initialValues.slice();
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let bestValues = values.slice();
  const history = [currentEnergy];
  const actionHistory = [];
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
    const context = actorCriticContext(problem, values, step, maxSteps, {
      currentEnergy,
      bestEnergy,
      violationCount: violationCount(problem, values),
      lastDelta,
      lastReward,
      recentImprovement,
      stagnation
    });
    const local = localActorFeatures(problem, values);
    const encoded = {
      hidden: tf.zeros([problem.n, HIDDEN_DIM]),
      nodeContext: tf.tensor2d(
        local.features,
        [problem.n, LOCAL_ACTOR_FEATURE_DIM]
      ),
      context
    };
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
      context,
      generated.candidates
    );
    const output = tf.tidy(() => state.policy.scoreActorCriticCandidates(
      problem,
      values,
      encoded,
      generated.candidates,
      prepared
    ));
    const continuations = Array.from(output.continuationValues.dataSync());
    const scores = continuations.map((continuation, index) =>
      prepared.immediateRewards[index] + ACTOR_CRITIC_DISCOUNT * continuation
    );
    const selectedPosition = argMax(scores);
    const actionIndex = generated.candidates[selectedPosition];
    encoded.hidden.dispose();
    encoded.nodeContext.dispose();
    output.actorLogits.dispose();
    output.continuationValues.dispose();

    const action = decodeAction(problem, actionIndex);
    const previousBest = bestEnergy;
    values[action.storageIndex] = action.value;
    currentEnergy = energy(problem, values);
    const reward = prepared.immediateRewards[selectedPosition];
    const delta = currentEnergy - history[history.length - 1];
    if (currentEnergy < bestEnergy) {
      bestEnergy = currentEnergy;
      bestValues = values.slice();
    }
    lastNodeId = action.nodeId;
    lastDelta = delta;
    lastReward = reward;
    recentImprovement = 0.90 * recentImprovement + 0.10 * reward;
    stagnation = bestEnergy < previousBest ? 0 : stagnation + 1;
    steps++;
    actionHistory.push(actionIndex);
    history.push(currentEnergy);
    if (steps % 40 === 0) await tf.nextFrame();
  }

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
    hiddenNorms: new Array(problem.n).fill(0),
    lastNodeId,
    candidateCount
  };
}

runActorCriticSearch = runLocalActorSearch;
runNeuralAnnealing = runLocalActorSearch;

function updateLocalActorUi() {
  ui.trainBudgetMultiplier.value = '2';
  const controlCaption = document.querySelector('.controls-panel .caption');
  if (controlCaption) {
    controlCaption.textContent =
      'The local actor uses 13 per-node features, 6 learned proposals, and 2 uniform proposals. Training defaults to 2N moves; comparison remains 4N.';
  }
  const architectureTitle = document.getElementById('architecture-title');
  if (architectureTitle) {
    architectureTitle.textContent = 'Local repair actor plus delayed future value';
  }
  const explanationCards = document.querySelectorAll('.explain-grid > div');
  if (explanationCards.length >= 4) {
    explanationCards[0].innerHTML =
      '<h3>Local node policy</h3><p>A 13 → 16 shared MLP sees chain position, neighboring values, feasible bounds, and adjacent violations. There is no recurrent state or message passing.</p>';
    explanationCards[1].innerHTML =
      '<h3>Dense actor targets</h3><p>Every visited state provides a target node distribution and a target value for every node. Early rollouts follow the exact local teacher, then gradually hand control to the actor.</p>';
    explanationCards[2].innerHTML =
      '<h3>Delayed critic</h3><p>The critic remains a residual continuation estimate. It has zero influence for the first 300 episodes and ramps in only after replay coverage is substantial.</p>';
    explanationCards[3].innerHTML =
      '<h3>Small search set</h3><p>Six actor proposals and two uniform proposals are evaluated exactly. Evaluation is greedy for stability, and actor training has no backpropagation through time.</p>';
  }
  ui.backendNotice.textContent =
    `TensorFlow.js ready · backend: ${tf.getBackend()} · local 13-feature actor`;
  if (state.trainedEpisodes === 0) {
    ui.statusText.textContent =
      'The local actor starts untrained. Dense local teacher targets should produce visible learning within the first tens of episodes.';
  }
}

const resetBeforeLocalActor = resetModel;
function resetLocalActorModel() {
  resetBeforeLocalActor();
  updateLocalActorUi();
}
resetModel = resetLocalActorModel;
replaceButtonListener('resetModelBtn', resetLocalActorModel);
updateLocalActorUi();
