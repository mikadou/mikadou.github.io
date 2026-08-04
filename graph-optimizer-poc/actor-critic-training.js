RecurrentGraphPolicy.prototype.trainEpisode = function trainActorCriticEpisode(
  problem,
  optimizer
) {
  ensureActorCriticParameters(this);
  const maxSteps = readPositiveInt(ui.trainBudgetMultiplier, 8) * problem.n;
  const actorRng = new SeededRandom(problem.seed ^ 0xA17C0A11);
  const randomRng = new SeededRandom(problem.seed ^ 0xBADC0FFE);
  const selectionRng = new SeededRandom(problem.seed ^ 0xC011EC70);
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
  let actorCrossEntropySum = 0;
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
    const output = tf.tidy(() => this.scoreActorCriticCandidates(
      problem,
      values,
      encoded,
      generated.candidates,
      prepared
    ));
    const actorLogits = Array.from(output.actorLogits.dataSync());
    const continuationValues = Array.from(output.continuationValues.dataSync());
    const totalScores = continuationValues.map((continuation, index) =>
      prepared.immediateRewards[index] + ACTOR_CRITIC_DISCOUNT * continuation
    );
    const targetPolicy = softmaxArray(totalScores, ACTOR_TARGET_TEMPERATURE);
    const selectedPosition = selectionRng.next() < TRAIN_ACTION_EXPLORATION
      ? selectionRng.int(generated.candidates.length)
      : sampleFromScores(totalScores, TRAIN_SELECTION_TEMPERATURE, selectionRng);
    const actionIndex = generated.candidates[selectedPosition];

    actorCrossEntropySum += crossEntropy(
      targetPolicy,
      actorLogits,
      ACTOR_POOL_TEMPERATURE
    );
    selectedScoreSum += totalScores[selectedPosition];
    trajectory.push({
      values: values.slice(),
      context: { ...context },
      candidates: generated.candidates.slice(),
      targetPolicy: Float32Array.from(targetPolicy),
      selectedPosition,
      selectedCriticPrediction: continuationValues[selectedPosition]
    });

    hidden.dispose();
    hidden = encoded.hidden;
    encoded.nodeContext.dispose();
    output.actorLogits.dispose();
    output.continuationValues.dispose();

    const action = decodeAction(problem, actionIndex);
    const previousBest = bestEnergy;
    values[action.storageIndex] = action.value;
    const candidateEnergy = energy(problem, values);
    const delta = candidateEnergy - currentEnergy;
    const reward = prepared.immediateRewards[selectedPosition];
    currentEnergy = candidateEnergy;
    if (currentEnergy < bestEnergy) bestEnergy = currentEnergy;

    lastDelta = delta;
    lastReward = reward;
    recentImprovement = 0.90 * recentImprovement + 0.10 * reward;
    stagnation = bestEnergy < previousBest ? 0 : stagnation + 1;
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
      criticLoss: 0
    };
    return 0;
  }

  const continuationTargets = computeContinuationTargets(rewards);
  let criticLossEstimate = 0;
  for (let step = 0; step < trajectory.length; step++) {
    const error = trajectory[step].selectedCriticPrediction - continuationTargets[step];
    criticLossEstimate += 0.5 * error * error;
  }
  criticLossEstimate /= trajectory.length;

  const result = tf.tidy(() => tf.variableGrads(() => {
    let replayHidden = tf.zeros([problem.n, HIDDEN_DIM]);
    let actorLossTotal = tf.scalar(0);
    let criticLossTotal = tf.scalar(0);

    for (let step = 0; step < trajectory.length; step++) {
      if (step > 0 && step % POLICY_MEMORY_WINDOW === 0) {
        replayHidden = tf.zeros([problem.n, HIDDEN_DIM]);
      }
      const item = trajectory[step];
      const encoded = this.encodeActorCriticState(
        problem,
        item.values,
        replayHidden,
        step,
        maxSteps,
        item.context
      );
      replayHidden = encoded.hidden;
      const output = this.scoreActorCriticCandidates(
        problem,
        item.values,
        encoded,
        item.candidates
      );

      const actorLogProbabilities = tf.logSoftmax(
        output.actorLogits.div(ACTOR_POOL_TEMPERATURE)
      );
      const actorProbabilities = tf.softmax(
        output.actorLogits.div(ACTOR_POOL_TEMPERATURE)
      );
      const targetPolicy = tf.tensor1d(item.targetPolicy);
      const actorCrossEntropy = targetPolicy.mul(actorLogProbabilities).sum().neg();
      const entropy = actorProbabilities.mul(actorLogProbabilities).sum().neg();
      actorLossTotal = actorLossTotal.add(
        actorCrossEntropy.sub(entropy.mul(ACTOR_ENTROPY_WEIGHT))
      );

      const selectedIndex = tf.tensor1d([item.selectedPosition], 'int32');
      const criticPrediction = output.continuationValues.gather(selectedIndex).reshape([]);
      const criticTarget = tf.scalar(continuationTargets[step]);
      const criticError = criticPrediction.sub(criticTarget);
      const absoluteError = criticError.abs();
      const quadratic = tf.minimum(absoluteError, tf.scalar(1));
      const linear = absoluteError.sub(quadratic);
      const huberLoss = quadratic.square().mul(0.5).add(linear);
      criticLossTotal = criticLossTotal.add(huberLoss);
    }

    const actorLoss = actorLossTotal.div(trajectory.length);
    const criticLoss = criticLossTotal.div(trajectory.length);
    return actorLoss.add(criticLoss.mul(CRITIC_LOSS_WEIGHT));
  }, this.vars));

  const lossValue = result.value.dataSync()[0];
  const clipped = {};
  for (const [name, gradient] of Object.entries(result.grads)) {
    clipped[name] = tf.clipByValue(gradient, -1, 1);
  }
  optimizer.applyGradients(clipped);
  disposeGradientResult(result, clipped);

  const episodeReturn = rewards.reduce((sum, reward) => sum + reward, 0);
  this.lastRewardStats = {
    episodeReturn,
    finalEnergy: bestEnergy,
    success: bestEnergy === 0,
    steps: trajectory.length,
    acceptanceRate: 1,
    selectedUtility: selectedScoreSum / trajectory.length,
    candidateCount: trajectory[0].candidates.length,
    actorLoss: actorCrossEntropySum / trajectory.length,
    criticLoss: criticLossEstimate,
    meanReward: episodeReturn / trajectory.length
  };

  return lossValue;
};


async function trainActorCriticPolicy() {
  if (state.busy) return;

  const episodes = clamp(readInt(ui.trainEpisodes, 2000), 50, 10000);
  let minN = clamp(readInt(ui.trainMinN, 8), 3, 32);
  let maxN = clamp(readInt(ui.trainMaxN, 8), 4, 40);
  readPositiveInt(ui.trainBudgetMultiplier, 8);
  if (minN > maxN) [minN, maxN] = [maxN, minN];
  ui.trainMinN.value = String(minN);
  ui.trainMaxN.value = String(maxN);
  setBusy(true);
  state.stopRequested = false;

  let movingLoss = null;
  let movingActorLoss = null;
  let movingCriticLoss = null;
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
      movingActorLoss = movingActorLoss === null
        ? stats.actorLoss
        : movingActorLoss * 0.95 + stats.actorLoss * 0.05;
      movingCriticLoss = movingCriticLoss === null
        ? stats.criticLoss
        : movingCriticLoss * 0.95 + stats.criticLoss * 0.05;
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
          `Actor–critic training ${episode}/${episodes} · total loss ${movingLoss.toFixed(3)} · ` +
          `actor ${movingActorLoss.toFixed(3)} · critic ${movingCriticLoss.toFixed(3)} · ` +
          `return ${movingReturn.toFixed(2)} · final energy ${movingFinalEnergy.toFixed(2)} · ` +
          `recent success ${(successRate * 100).toFixed(0)}%`
        );
        await tf.nextFrame();
      }
    }

    if (state.trainedEpisodes > 0) {
      ui.policyStateBadge.textContent = `${state.trainedEpisodes} actor–critic episodes`;
      ui.policyStateBadge.classList.remove('muted');
    }
    setStatus(
      state.stopRequested
        ? `Training stopped after ${state.trainedEpisodes} total episodes.`
        : `Actor–critic training complete: ${state.trainedEpisodes} total episodes.` +
          trainingEnvelopeWarning()
    );
  } catch (error) {
    console.error(error);
    setStatus(`Actor–critic training failed: ${error.message}`, true);
  } finally {
    setBusy(false);
    state.stopRequested = false;
  }
}

