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
