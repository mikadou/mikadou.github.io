'use strict';

// Conservative browser-training performance patch.
//
// The model architecture and feature set remain unchanged. This patch removes
// avoidable work around them:
// - 12 actor proposals + 4 random proposals instead of 24 + 8;
// - one synchronized read for the actor distribution and one for scored actions;
// - actor gradients from 16 detached trajectory states instead of full-episode BPTT;
// - two critic updates of 64 replay samples instead of four updates of 32;
// - a sharper grounded actor target;
// - gradual critic influence while replay supervision is still immature.

const PERFORMANCE_ACTOR_CANDIDATES = 12;
const PERFORMANCE_RANDOM_CANDIDATES = 4;
const PERFORMANCE_TOTAL_CANDIDATES =
  PERFORMANCE_ACTOR_CANDIDATES + PERFORMANCE_RANDOM_CANDIDATES;
const PERFORMANCE_ACTOR_BATCH_SIZE = 16;
const PERFORMANCE_CRITIC_BATCH_SIZE = 64;
const PERFORMANCE_CRITIC_UPDATES = 2;
const PERFORMANCE_ACTOR_TARGET_TEMPERATURE = 0.10;
const PERFORMANCE_CRITIC_WARMUP_EPISODES = 100;
const PERFORMANCE_CRITIC_RAMP_EPISODES = 400;
const PERFORMANCE_CRITIC_MIN_REPLAY = 128;
const PERFORMANCE_CRITIC_FULL_REPLAY = 512;

function performanceCriticInfluence(policy) {
  const episodeProgress = clamp(
    (state.trainedEpisodes - PERFORMANCE_CRITIC_WARMUP_EPISODES) /
      PERFORMANCE_CRITIC_RAMP_EPISODES,
    0,
    1
  );
  const replaySize = policy && policy.stateCriticReplay
    ? policy.stateCriticReplay.length
    : 0;
  const replayProgress = clamp(
    (replaySize - PERFORMANCE_CRITIC_MIN_REPLAY) /
      (PERFORMANCE_CRITIC_FULL_REPLAY - PERFORMANCE_CRITIC_MIN_REPLAY),
    0,
    1
  );
  return Math.min(episodeProgress, replayProgress);
}

// Generate direct actor proposals with a single GPU -> CPU synchronization.
generateActorCriticCandidates = function generatePerformanceCandidates(
  policy,
  problem,
  values,
  encoded,
  actorRng,
  randomRng
) {
  const legalCount = legalActionCount(problem);
  if (legalCount <= PERFORMANCE_TOTAL_CANDIDATES) {
    return {
      candidates: allLegalActions(problem, values),
      actorCount: legalCount,
      randomCount: 0,
      directActor: true
    };
  }

  const packedDistribution = tf.tidy(() => {
    const distribution = policy.actorDistribution(encoded);
    return tf.concat([
      distribution.nodeLogits,
      distribution.valueMeans,
      distribution.valueScales
    ]);
  });
  const packedData = packedDistribution.dataSync();
  packedDistribution.dispose();
  const n = problem.n;
  const actorData = {
    nodeLogits: Float32Array.from(packedData.slice(0, n)),
    valueMeans: Float32Array.from(packedData.slice(n, 2 * n)),
    valueScales: Float32Array.from(packedData.slice(2 * n, 3 * n))
  };

  const actorCandidates = sampleActorActionsDirectly(
    problem,
    values,
    actorData,
    actorRng,
    PERFORMANCE_ACTOR_CANDIDATES
  );
  const seen = new Set(actorCandidates);
  const randomCandidates = sampleUniqueUniformActions(
    problem,
    values,
    randomRng,
    PERFORMANCE_RANDOM_CANDIDATES,
    seen
  );

  return {
    candidates: actorCandidates.concat(randomCandidates),
    actorCount: actorCandidates.length,
    randomCount: randomCandidates.length,
    directActor: true
  };
};

// Keep the exact immediate objective dominant until the critic has enough
// episodes and replay coverage to make its continuation estimate useful.
const scorePostActionCandidatesWithoutWarmup =
  RecurrentGraphPolicy.prototype.scoreActorCriticCandidates;
RecurrentGraphPolicy.prototype.scoreActorCriticCandidates =
  function scorePostActionCandidatesWithWarmup(
    problem,
    values,
    encoded,
    candidateActions,
    prepared = null
  ) {
    const output = scorePostActionCandidatesWithoutWarmup.call(
      this,
      problem,
      values,
      encoded,
      candidateActions,
      prepared
    );
    const influence = performanceCriticInfluence(this);
    if (influence < 1) {
      const scaled = output.continuationValues.mul(influence);
      output.continuationValues.dispose();
      output.continuationValues = scaled;
    }
    return output;
  };

function performanceSampleReplay(policy, rng) {
  const count = Math.min(
    PERFORMANCE_CRITIC_BATCH_SIZE,
    policy.stateCriticReplay.length
  );
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

function performanceActorBatch(trajectory) {
  if (trajectory.length <= PERFORMANCE_ACTOR_BATCH_SIZE) {
    return trajectory;
  }
  const selected = [];
  const stride = trajectory.length / PERFORMANCE_ACTOR_BATCH_SIZE;
  for (let index = 0; index < PERFORMANCE_ACTOR_BATCH_SIZE; index++) {
    selected.push(trajectory[Math.min(
      trajectory.length - 1,
      Math.floor((index + 0.5) * stride)
    )]);
  }
  return selected;
}

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

  // Each sampled hidden state is treated as a constant. ThhÈÙY\ÈHXİÜ‰ÜÂˆËÈ™Xİ\œ™[ÛÛ^]]›ÚYÈ˜XÚÜ›ÜYØ][™È›İYÚHÛÛ\]H\\ÛÙK‚ˆÛÛœİXİÜ”™\İ[H‹YJ

HOˆ‹˜\šXX›QÜ˜YÊ

HOˆÂˆ]İ[ÜÜÈH‹œØØ[\Š
NÂˆXİÜ˜]Ú™›Ü‘XXÚ
][HOˆÂˆÛÛœİ[˜ÛÙYH\Ë™[˜ÛÙPXİÜÜš]XÔİ]Jˆ›Ø›[Kˆ][K˜XİÜ•˜[Y\Ëˆ][K˜XİÜ’Y[‹ˆ][Kœİ\ˆX^İ\Ëˆ][K˜ÛÛ^ˆ
NÂˆÛÛœİXİÜˆH\Ë˜XİÜØ[™Y]SÙÚ]Êˆ›Ø›[Kˆ[˜ÛÙYˆ][K˜Ø[™Y]\Âˆ
NÂˆÛÛœİÙÔ›Ø˜Xš[]Y\ÈH‹›ÙÔÛÙX^
ˆXİÜ‹›ÙÚ]Ë™]ŠPÕÔ—ĞĞS‘QUWÕSTTUT‘Bˆ
NÂˆÛÛœİ›Ø˜Xš[]Y\ÈH‹œÛÙX^
ˆXİÜ‹›ÙÚ]Ë™]ŠPÕÔ—ĞĞS‘QUWÕSTTUT‘Bˆ
NÂˆÛÛœİ\™Ù]H‹[œÛÜŒY
][K\™Ù]ÛXŞJNÂˆÛÛœİÜ›ÜÜÑ[›ÜSÜÜÈH\™Ù]›][
ÙÔ›Ø˜Xš[]Y\ÊKœİ[J
K›™YÊ
NÂˆÛÛœİ[›ÜHH›Ø˜Xš[]Y\Ë›][
ÙÔ›Ø˜Xš[]Y\ÊKœİ[J
K›™YÊ
NÂˆİ[ÜÜÈHİ[ÜÜË˜Y
ˆÜ›ÜÜÑ[›ÜSÜÜËœİXŠ[›ÜK›][
PÕÔ—ÑS•“ÔWÕÑRQÒ
JBˆ
NÂˆJNÂˆ™]\›ˆİ[ÜÜË™]ŠXİÜ˜]Ú›[™İ
NÂˆK\Ë˜XİÜ•˜\œÊJNÂˆÛÛœİXİÜ•˜Z[š[™ÓÜÜÈHİX›P\QÜ˜YY[ÊÜ[Z^™\‹XİÜ”™\İ[
NÂˆ˜Z™XİÜK™›Ü‘XXÚ
][HOˆ][K˜XİÜ’Y[‹™\ÜÜÙJ
JNÂ‚ˆ]Üš]XÕ˜Z[š[™ÓÜÜÈHÂˆ]Üš]XÕ\]\ÈHÂˆ›Üˆ
]\]HHÈ\]HT‘“Ô“PSÑWĞÔ’UP×ÕTUTÎÈ\]JÊÊHÂˆÛÛœİ˜]ÚH\™›Ü›X[˜ÙTØ[\T™\^J\Ë™\^T›™ÊNÂˆYˆ
X˜]Ú›[™İ
Hœ™XZÎÂˆÛÛœİ™X]\™Q]HH™]È›Ø]Ì\œ˜^Jˆ˜]Ú›[™İ
ˆÕP“WĞÔ’UP×ÔÕUWÑSBˆ
NÂˆÛÛœİ\™Ù]]HH™]È›Ø]Ì\œ˜^J˜]Ú›[™İ
NÂˆ˜]Ú™›Ü‘XXÚ

][K[™^
HOˆÂˆ™X]\™Q]KœÙ]
][K™™X]\™\Ë[™^
ˆÕP“WĞÔ’UP×ÔÕUWÑSJNÂˆ\™Ù]]VÚ[™^HH][K\™Ù]ÂˆJNÂˆÛÛœİÜš]XÔ™\İ[H‹YJ

HOˆ‹˜\šXX›QÜ˜YÊ

HOˆÂˆÛÛœİ™X]\™\ÈH‹[œÛÜŒ™
ˆ™X]\™Q]KˆØ˜]Ú›[™İÕP“WĞÔ’UP×ÔÕUWÑSWBˆ
NÂˆÛÛœİ\™Ù]ÈH‹[œÛÜŒY
\™Ù]]JNÂˆ™]\›ˆİX›RX™\Š\Ëœİ]PÜš]XÕ˜[Y\Ê™X]\™\ÊKœİXŠ\™Ù]ÊJK›YX[Š
NÂˆK\Ëœİ]PÜš]XÕ˜\œÊJNÂˆÜš]XÕ˜Z[š[™ÓÜÜÈ
ÏHİX›P\QÜ˜YY[ÊÜ[Z^™\‹Üš]XÔ™\İ[
NÂˆÜš]XÕ\]\ÊÊÎÂˆBˆÜš]XÕ˜Z[š[™ÓÜÜÈHÜš]XÕ\]\ÂˆÈÜš]XÕ˜Z[š[™ÓÜÜÈÈÜš]XÕ\]\ÂˆˆÂ‚ˆÛÛœİ\\ÛÙT™]\›ˆH™]Ø\™Ëœ™YXÙJ
İ[K™]Ø\™
HOˆİ[H
È™]Ø\™
NÂˆ\Ë›\İ™]Ø\™İ]ÈHÂˆ\\ÛÙT™]\›‹ˆš[˜[[™\™ŞNˆ™\İ[™\™ŞKˆİXØÙ\ÜÎˆ™\İ[™\™ŞHOOHˆİ\Îˆ˜Z™XİÜK›[™İˆXØÙ\[˜ÙT˜]NˆKˆÙ[XİY][]NˆÙ[XİYØÛÜ™Tİ[HÈ˜Z™XİÜK›[™İˆØ[™Y]PÛİ[ˆ˜Z™XİÜVÌK˜Ø[™Y]\Ë›[™İˆXİÜ“ÜÜÎˆXİÜ“ÜÜÑ\İ[X]KˆÜš]XÓÜÜÎˆÜš]XÕ˜Z[š[™ÓÜÜËˆYX[”™]Ø\™ˆ\\ÛÙT™]\›ˆÈ˜Z™XİÜK›[™İˆ™\^TÚ^™Nˆ\Ëœİ]PÜš]XÔ™\^K›[™İˆÜš]XÒ[™›Y[˜ÙNˆ\™›Ü›X[˜ÙPÜš]XÒ[™›Y[˜ÙJ\ÊBˆNÂˆ™]\›ˆXİÜ•˜Z[š[™ÓÜÜÈ
ÈÜš]XÕ˜Z[š[™ÓÜÜÎÂŸNÂ‚ZK˜˜XÚÙ[™›İXÙK^ÛÛ[Bˆ	İZK˜˜XÚÙ[™›İXÙK^ÛÛ[H0­È\™›Ü›X[˜ÙH›Ùš[NˆLŠÍØ[™Y]\ØÂ