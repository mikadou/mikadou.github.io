RecurrentGraphPolicy.prototype.trainEpisode = function trainPerformanceActorCriticEpisode(
  problem,
  optimizer
) {
  ensureStableActorCritic(this);
  const maxSteps = readPositiveInt(ui.trainBudgetMultiplier, 4) * problem.n;
  const actorRng = new SeededRandom(problem.seed ^ 0xA17C0A11);
  const randomRng = new SeededRandom(problem.seed ^ 0xBADC0FFE);
  const selectionRng = new SeededRandom(problem.seed ^ 0xC011EC70);
  const replayRng = new SeededRandom(problem.seed ^ 0xC71C1C00);
  const values = problem.initialValues.slice();
  const trajectory = [];
  const rewards = [];
  let hidden = tf.zeros([problem.n, HIDDEN_DIM]);
  let currentEnergy = energy(problem, values);
  let bestEnergy = currentEnergy;
  let lastDelta = 0;
  let lastReward = 0;
  let recentImprovement = 0;
  let stagnation = 0;
  let selectedScoreSum = 0;

  for (let step = 0; step < maxSteps && bestEnergy > 0; step++) {
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
    const actorStateValues = values.slice();
    const actorHidden = hidden.clone();
    const encoded = tf.tidy(() => this.encodeActorCriticState(
      problem,
      values,
      hidden,
      step,
      maxSteps,
      context
    ));
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
      encoded.context,
      generated.candidates
    );

    // Actor logits and continuation values are read together, avoiding a second
    // WebGL synchronization for the same candidate set.
    const packedOutput = tf.tidy(() => {
      const output = this.scoreActorCriticCandidates(
        problem,
        values,
        encoded,
        generated.candidates,
        prepared
      );
      return tf.concat([output.actorLogits, output.continuationValues]);
    });
    const packedData = packedOutput.dataSync();
    packedOutput.dispose();
    const candidateCount = generated.candidates.length;
    const actorLogits = Array.from(packedData.slice(0, candidateCount));
    const continuationValues = Array.from(
      packedData.slice(candidateCount, 2 * candidateCount)
    );
    const totalScores = continuationValues.map((continuation, index) =>
      prepared.immediateRewards[index] + ACTOR_CRITIC_DISCOUNT * continuation
    );
    const selectedPosition = selectionRng.next() < TRAIN_ACTION_EXPLORATION
      ? selectionRng.int(candidateCount)
      : sampleFromScores(totalScores, TRAIN_SELECTION_TEMPERATURE, selectionRng);
    const actionIndex = generated.candidates[selectedPosition];
    const reward = prepared.immediateRewards[selectedPosition];
    selectedScoreSum += totalScores[selectedPosition];

    hidden.dispose();
    hidden = encoded.hidden;
    encoded.nodeContext.dispose();

    const action = decodeAction(problem, actionIndex);
    const previousBest = bestEnergy;
    values[action.storageIndex] = action.value;
    const candidateEnergy = energy(problem, values);
    const delta = candidateEnergy - currentEnergy;
    currentEnergy = candidateEnergy;
    if (currentEnergy < bestEnergy) bestEnergy = currentEnergy;
    lastDelta = delta;
    lastReward = reward;
    recentImprovement = 0.90 * recentImprovement + 0.10 * reward;
    stagnation = bestEnergy < previousBest ? 0 : stagnation + 1;

    const nextContext = {
      currentEnergy,
      bestEnergy,
      violationCount: violationCount(problem, values),
      lastDelta,
      lastReward,
      recentImprovement,
      stagnation
    };
    trajectory.push({
      step,
      actorValues: actorStateValues,
      actorHidden,
      context: { ...context },
      candidates: generated.candidates.slice(),
      actorLogits,
      immediateRewards: Float32Array.from(prepared.immediateRewards),
      selectedPosition,
      nextCriticFeatures: stableCriticStateFeatures(
        problem,
        values,
        step + 1,
        maxSteps,
        nextContext
      )
    });
    rewards.push(reward);
  }

  hidden.dispose();
  if (!trajectory.length) {
    this.lastRewardStats = {
      episodeReturn: 0,
      finalEnergy: 0,
      success: true,
      steps: 0,
      acceptanceRate: 1,
      selectedUtility: 0,
      candidateCount: 0,
      actorLoss: 0,
      criticLoss: 0,
      replaySize: this.stateCriticReplay.length,
      criticInfluence: performanceCriticInfluence(this)
    };
    return 0;
  }

  const continuationTargets = computeContinuationTargets(rewards);
  const replayItems = [];
  trajectory.forEach((item, step) => {
    const targetScores = Array.from(item.immediateRewards);
    targetScores[item.selectedPosition] = clamp(
      rewards[step] + ACTOR_CRITIC_DISCOUNT * continuationTargets[step],
      -RETURN_CLIP,
      RETURN_CLIP
    );
    item.targetPolicy = Float32Array.from(softmaxArray(
      targetScores,
      PERFORMANCE_ACTOR_TARGET_TEMPERATURE
    ));
    replayItems.push({
      features: item.nextCriticFeatures,
      target: continuationTargets[step]
    });
  });
  stableAddReplay(this, replayItems);

  const actorBatch = performanceActorBatch(trajectory);
  let actorLossEstimate = 0;
  actorBatch.forEach(item => {
    actorLossEstimate += crossEntropy(
      item.targetPolicy,
      item.actorLogits,
      ACTOR_CANDIDATE_TEMPERATURE
    );
  });
  actorLossEstimate /= actorBatch.length;

  // Each sampled hidden state is treated as a constant. This keeps the actor's
  // recurrent context but avoids backpropagating through the complete episode.
  const actorResult = tf.tidy(() => tf.variableGrads(() => {
    let totalLoss = tf.scalar(0);
    actorBatch.forEach(item => {
      const encoded = this.encodeActorCriticState(
        problem,
        item.actorValues,
        item.actorHidden,
        item.step,
        maxSteps,
        item.context
      );
      const actor = this.actorCandidateLogits(
        problem,
        encoded,
        item.candidates
      );
      const logProbabilities = tf.logSoftmax(
        actor.logits.div(ACTOR_CANDIDATE_TEMPERATURE)
      );
      const probabilities = tf.softmax(
        actor.logits.div(ACTOR_CANDIDATE_TEMPERATURE)
      );
      const target = tf.tensor1d(item.targetPolicy);
      const crossEntropyLoss = target.mul(logProbabilities).sum().neg();
      const entropy = probabilities.mul(logProbabilities).sum().neg();
      totalLoss = totalLoss.add(
        crossEntropyLoss.sub(entropy.mul(ACTOR_ENTROPY_WEIGHT))
      );
    });
    return totalLoss.div(actorBatch.length);
  }, this.actorVars));
  const actorTrainingLoss = stableApplyGradients(optimizer, actorResult);
  trajectory.forEach(item => item.actorHidden.dispose());

  let criticTrainingLoss = 0;
  let criticUpdates = 0;
  for (let update = 0; update < PERFORMANCE_CRITIC_UPDATES; update++) {
    const batch = performanceSampleReplay(this, replayRng);
    if (!batch.length) break;
    const featureData = new Float32Array(
      batch.length * STABLE_CRITIC_STATE_DIM
    );
    const targetData = new Float32Array(batch.length);
    batch.forEach((item, index) => {
      featureData.set(item.features, index * STABLE_CRITIC_STATE_DIM);
      targetData[index] = item.target;
    });
    const criticResult = tf.tidy(() => tf.variableGrads(() => {
      const features = tf.tensor2d(
        featureData,
        [batch.length, STABLE_CRITIC_STATE_DIM]
      );
      const targets = tf.tensor1d(targetData);
      return stableHuber(this.stateCriticValues(features).sub(targets)).mean();
    }, this.stateCriticVars));
    criticTrainingLoss += stableApplyGradients(optimizer, criticResult);
    criticUpdates++;
  }
  criticTrainingLoss = criticUpdates
    ? criticTrainingLoss / criticUpdates
    : 0;

  const episodeReturn = rewards.reduce((sum, reward) => sum + reward, 0);
  this.lastRewardStats = {
    episodeReturn,
    finalEnergy: bestEnergy,
    success: bestEnergy === 0,
    steps: trajectory.length,
    acceptanceRate: 1,
    selectedUtility: selectedScoreSum / trajectory.length,
    candidateCount: trajectory[0].candidates.length,
    actorLoss: actorLossEstimate,
    criticLoss: criticTrainingLoss,
    meanReward: episodeReturn / trajectory.length,
    replaySize: this.stateCriticReplay.length,
    criticInfluence: performanceCriticInfluence(this)
  };
  return actorTrainingLoss + criticTrainingLoss;
};

ui.backendNotice.textContent =
  `${ui.backendNotice.textContent} · performance profile: 12+4 candidates`;
