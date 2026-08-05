'use strict';

// Small, dense-supervision actor for the chain toy problem.
// The actor sees only stationary local structure: chain position, neighboring
// values, feasible bounds, and adjacent violations. It does not use recurrent
// state or message passing.

const LOCAL_ACTOR_FEATURE_DIM = 13;
const LOCAL_ACTOR_HIDDEN_DIM = 16;
const LOCAL_ACTOR_CANDIDATE_COUNT = 6;
const LOCAL_RANDOM_CANDIDATE_COUNT = 2;
const LOCAL_ACTOR_VALUE_SCALE = 0.08;
const LOCAL_NODE_TARGET_TEMPERATURE = 0.06;
const LOCAL_VALUE_LOSS_WEIGHT = 2.0;
const LOCAL_TEACHER_MIN_PROBABILITY = 0.10;
const LOCAL_TEACHER_DECAY_EPISODES = 300;
const LOCAL_CRITIC_WARMUP_EPISODES = 300;
const LOCAL_CRITIC_RAMP_EPISODES = 600;
const LOCAL_CRITIC_MIN_REPLAY = 256;
const LOCAL_CRITIC_FULL_REPLAY = 1024;
const LOCAL_CRITIC_BATCH_SIZE = 64;

function localActorNodeInfo(problem, values, storageIndex) {
  const id = problem.nodeIds[storageIndex];
  const depth = problem.chainIndexById.get(id);
  const hasPred = depth > 0;
  const hasSucc = depth < problem.n - 1;
  const predIndex = hasPred
    ? problem.indexById.get(problem.chain[depth - 1])
    : -1;
  const succIndex = hasSucc
    ? problem.indexById.get(problem.chain[depth + 1])
    : -1;
  const currentValue = values[storageIndex];
  const predValue = hasPred ? values[predIndex] : problem.domainMax;
  const succValue = hasSucc ? values[succIndex] : 0;
  const lowerBound = hasSucc ? succValue + 1 : 0;
  const upperBound = hasPred ? predValue - 1 : problem.domainMax;
  const leftViolation = hasPred ? edgeViolation(predValue, currentValue) : 0;
  const rightViolation = hasSucc ? edgeViolation(currentValue, succValue) : 0;
  const incidentViolation = leftViolation + rightViolation;
  const position = problem.n > 1 ? depth / (problem.n - 1) : 0;
  const positionTarget = Math.round((1 - position) * problem.domainMax);
  const feasible = lowerBound <= upperBound;
  let distanceToInterval = 0;
  if (currentValue < lowerBound) distanceToInterval = lowerBound - currentValue;
  else if (currentValue > upperBound) distanceToInterval = currentValue - upperBound;

  return {
    storageIndex,
    depth,
    hasPred,
    hasSucc,
    currentValue,
    predValue,
    succValue,
    lowerBound,
    upperBound,
    leftViolation,
    rightViolation,
    incidentViolation,
    position,
    positionTarget,
    feasible,
    distanceToInterval
  };
}

function localActorTargetValue(problem, info) {
  if (info.feasible) {
    return clamp(info.positionTarget, info.lowerBound, info.upperBound);
  }

  let bestValue = info.currentValue;
  let bestCost = Number.POSITIVE_INFINITY;
  const denom = Math.max(1, problem.domainMax);
  for (let value = 0; value <= problem.domainMax; value++) {
    const localViolation =
      (info.hasPred ? edgeViolation(info.predValue, value) : 0) +
      (info.hasSucc ? edgeViolation(value, info.succValue) : 0);
    const positionCost = Math.abs(value - info.positionTarget) / denom;
    const moveCost = Math.abs(value - info.currentValue) / denom;
    const cost = localViolation + 0.05 * positionCost + 0.002 * moveCost;
    if (cost < bestCost - 1e-9) {
      bestCost = cost;
      bestValue = value;
    }
  }

  if (bestValue === info.currentValue && info.incidentViolation > 0) {
    let alternativeCost = Number.POSITIVE_INFINITY;
    for (let value = 0; value <= problem.domainMax; value++) {
      if (value === info.currentValue) continue;
      const localViolation =
        (info.hasPred ? edgeViolation(info.predValue, value) : 0) +
        (info.hasSucc ? edgeViolation(value, info.succValue) : 0);
      const positionCost = Math.abs(value - info.positionTarget) / denom;
      const cost = localViolation + 0.05 * positionCost;
      if (cost < alternativeCost - 1e-9) {
        alternativeCost = cost;
        bestValue = value;
      }
    }
  }
  return bestValue;
}

function localActorFeatures(problem, values) {
  const features = new Float32Array(problem.n * LOCAL_ACTOR_FEATURE_DIM);
  const infos = new Array(problem.n);
  const denom = Math.max(1, problem.domainMax);

  for (let node = 0; node < problem.n; node++) {
    const info = localActorNodeInfo(problem, values, node);
    infos[node] = info;
    const offset = node * LOCAL_ACTOR_FEATURE_DIM;
    features[offset] = info.currentValue / denom;
    features[offset + 1] = info.position;
    features[offset + 2] = 1 - info.position;
    features[offset + 3] = info.predValue / denom;
    features[offset + 4] = info.succValue / denom;
    features[offset + 5] = clamp(info.lowerBound / denom, 0, 1);
    features[offset + 6] = clamp(info.upperBound / denom, 0, 1);
    features[offset + 7] = clamp(info.leftViolation / denom, 0, 2);
    features[offset + 8] = clamp(info.rightViolation / denom, 0, 2);
    features[offset + 9] = clamp(info.incidentViolation / denom, 0, 2);
    features[offset + 10] = clamp(info.distanceToInterval / denom, 0, 2);
    features[offset + 11] = clamp(
      (info.upperBound - info.lowerBound) / denom,
      -1,
      1
    );
    features[offset + 12] = info.feasible ? 1 : 0;
  }
  return { features, infos };
}

function localActorTeacher(problem, values, context) {
  const local = localActorFeatures(problem, values);
  const targetValues = new Float32Array(problem.n);
  const nodeScores = new Array(problem.n);
  const valueWeights = new Float32Array(problem.n);
  const energyDenom = Math.max(1, problem.n);
  const valueDenom = Math.max(1, problem.domainMax);
  let maxIncident = 0;

  local.infos.forEach(info => {
    maxIncident = Math.max(maxIncident, info.incidentViolation);
  });

  local.infos.forEach((info, node) => {
    const targetValue = localActorTargetValue(problem, info);
    targetValues[node] = targetValue / valueDenom;
    let reward = 0;
    if (targetValue !== info.currentValue) {
      const diagnostics = localMutationDiagnostics(
        problem,
        values,
        node,
        targetValue,
        {
          currentEnergy: context.currentEnergy,
          bestEnergy: context.bestEnergy,
          temperature: 1
        }
      );
      reward = -diagnostics.deltaEnergy / energyDenom;
      if (context.currentEnergy > 0 && diagnostics.candidateEnergy === 0) {
        reward += TERMINAL_SUCCESS_REWARD;
      }
    }
    const violationFocus = 0.10 * info.incidentViolation / valueDenom;
    nodeScores[node] = reward + violationFocus;
    valueWeights[node] = maxIncident > 0
      ? 0.20 + 0.80 * info.incidentViolation / maxIncident
      : 0.20;
  });

  return {
    ...local,
    targetValues,
    valueWeights,
    nodeTargets: Float32Array.from(
      softmaxArray(nodeScores, LOCAL_NODE_TARGET_TEMPERATURE)
    ),
    nodeScores
  };
}

function ensureLocalActor(policy) {
  ensureStableActorCritic(policy);
  if (policy.localActorReady) return;
  policy.wLocalActor1 = policy.add(initVariable(
    [LOCAL_ACTOR_FEATURE_DIM, LOCAL_ACTOR_HIDDEN_DIM],
    `${policy.prefix}_wLocalActor1`,
    LOCAL_ACTOR_FEATURE_DIM
  ));
  policy.bLocalActor1 = policy.add(tf.variable(
    tf.zeros([LOCAL_ACTOR_HIDDEN_DIM]),
    true,
    `${policy.prefix}_bLocalActor1`
  ));
  policy.wLocalActorNode = policy.add(initVariable(
    [LOCAL_ACTOR_HIDDEN_DIM, 1],
    `${policy.prefix}_wLocalActorNode`,
    LOCAL_ACTOR_HIDDEN_DIM
  ));
  policy.bLocalActorNode = policy.add(tf.variable(
    tf.zeros([1]),
    true,
    `${policy.prefix}_bLocalActorNode`
  ));
  policy.wLocalActorValue = policy.add(initVariable(
    [LOCAL_ACTOR_HIDDEN_DIM, 1],
    `${policy.prefix}_wLocalActorValue`,
    LOCAL_ACTOR_HIDDEN_DIM
  ));
  policy.bLocalActorValue = policy.add(tf.variable(
    tf.zeros([1]),
    true,
    `${policy.prefix}_bLocalActorValue`
  ));
  policy.localActorVars = [
    policy.wLocalActor1,
    policy.bLocalActor1,
    policy.wLocalActorNode,
    policy.bLocalActorNode,
    policy.wLocalActorValue,
    policy.bLocalActorValue
  ];
  policy.localActorReady = true;
}

Object.defineProperty(RecurrentGraphPolicy.prototype, 'actorVars', {
  configurable: true,
  get() {
    ensureLocalActor(this);
    return this.localActorVars;
  }
});

RecurrentGraphPolicy.prototype.encodeActorCriticState = function encodeLocalActorState(
  problem,
  values,
  hidden,
  step,
  maxSteps,
  rawContext = {}
) {
  const context = actorCriticContext(problem, values, step, maxSteps, rawContext);
  const local = localActorFeatures(problem, values);
  return {
    hidden: tf.zeros([problem.n, HIDDEN_DIM]),
    nodeContext: tf.tensor2d(
      local.features,
      [problem.n, LOCAL_ACTOR_FEATURE_DIM]
    ),
    context,
    localInfos: local.infos
  };
};

RecurrentGraphPolicy.prototype.actorDistribution = function localActorDistribution(encoded) {
  ensureLocalActor(this);
  const hidden = tf.relu(
    encoded.nodeContext.matMul(this.wLocalActor1).add(this.bLocalActor1)
  );
  const nodeLogits = hidden.matMul(this.wLocalActorNode)
    .add(this.bLocalActorNode)
    .reshape([encoded.nodeContext.shape[0]]);
  const valueMeans = tf.sigmoid(
    hidden.matMul(this.wLocalActorValue).add(this.bLocalActorValue)
  ).reshape([encoded.nodeContext.shape[0]]);
  const valueScales = tf.fill(
    [encoded.nodeContext.shape[0]],
    LOCAL_ACTOR_VALUE_SCALE
  );
  return { nodeLogits, valueMeans, valueScales };
};

RecurrentGraphPolicy.prototype.actorCandidateLogits = function localActorCandidateLogits(
  problem,
  encoded,
  candidateActions
) {
  const distribution = this.actorDistribution(encoded);
  const candidateNodes = tf.tensor1d(
    Int32Array.from(
      candidateActions,
      actionIndex => Math.floor(actionIndex / problem.domainSize)
    ),
    'int32'
  );
  const normalizedValues = tf.tensor1d(Float32Array.from(
    candidateActions,
    actionIndex => (actionIndex % problem.domainSize) / Math.max(1, problem.domainMax)
  ));
  const nodeLogProbabilities = tf.logSoftmax(distribution.nodeLogits);
  const selectedNodeLogProbabilities = nodeLogProbabilities.gather(candidateNodes);
  const means = distribution.valueMeans.gather(candidateNodes);
  const scales = distribution.valueScales.gather(candidateNodes);
  const standardized = normalizedValues.sub(means).div(scales);
  const valueLogDensity = standardized.square().mul(-0.5).sub(scales.log());
  return {
    logits: selectedNodeLogProbabilities.add(valueLogDensity),
    distribution
  };
};

generateActorCriticCandidates = function generateLocalActorCandidates(
  policy,
  problem,
  values,
  encoded,
  actorRng,
  randomRng
) {
  const requested = LOCAL_ACTOR_CANDIDATE_COUNT + LOCAL_RANDOM_CANDIDATE_COUNT;
  const legalCount = legalActionCount(problem);
  if (legalCount <= requested) {
    return {
      candidates: allLegalActions(problem, values),
      actorCount: legalCount,
      randomCount: 0,
      directActor: true
    };
  }

  const packed = tf.tidy(() => {
    const distribution = policy.actorDistribution(encoded);
    return tf.concat([
      distribution.nodeLogits,
      distribution.valueMeans,
      distribution.valueScales
    ]);
  });
  const data = packed.dataSync();
  packed.dispose();
  const n = problem.n;
  const actorData = {
    nodeLogits: Float32Array.from(data.slice(0, n)),
    valueMeans: Float32Array.from(data.slice(n, 2 * n)),
    valueScales: Float32Array.from(data.slice(2 * n, 3 * n))
  };
  const actorCandidates = sampleActorActionsDirectly(
    problem,
    values,
    actorData,
    actorRng,
    LOCAL_ACTOR_CANDIDATE_COUNT
  );
  const seen = new Set(actorCandidates);
  const randomCandidates = sampleUniqueUniformActions(
    problem,
    values,
    randomRng,
    LOCAL_RANDOM_CANDIDATE_COUNT,
    seen
  );
  return {
    candidates: actorCandidates.concat(randomCandidates),
    actorCount: actorCandidates.length,
    randomCount: randomCandidates.length,
    directActor: true
  };
};

performanceCriticInfluence = function localCriticInfluence(policy) {
  const episodeProgress = clamp(
    (state.trainedEpisodes - LOCAL_CRITIC_WARMUP_EPISODES) /
      LOCAL_CRITIC_RAMP_EPISODES,
    0,
    1
  );
  const replaySize = policy && policy.stateCriticReplay
    ? policy.stateCriticReplay.length
    : 0;
  const replayProgress = clamp(
    (replaySize - LOCAL_CRITIC_MIN_REPLAY) /
      (LOCAL_CRITIC_FULL_REPLAY - LOCAL_CRITIC_MIN_REPLAY),
    0,
    1
  );
  return Math.min(episodeProgress, replayProgress);
};

// Critic-only candidate scorer: the local actor was already evaluated while
// generating candidates, so do not run it a second time merely to return logits.
RecurrentGraphPolicy.prototype.scoreActorCriticCandidates = function scoreLocalCandidates(
  problem,
  values,
  encoded,
  candidateActions,
  prepared = null
) {
  ensureStableActorCritic(this);
  const actionData = prepared || candidateDiagnostics(
    problem,
    values,
    encoded.context,
    candidateActions
  );
  const featureData = new Float32Array(
    candidateActions.length * STABLE_CRITIC_STATE_DIM
  );
  const activeMask = new Float32Array(candidateActions.length);
  const nextStep = encoded.context.step + 1;

  candidateActions.forEach((actionIndex, position) => {
    const postAction = stablePostActionState(
      problem,
      values,
      encoded.context,
      actionIndex,
      actionData.diagnostics[position],
      actionData.immediateRewards[position],
      nextStep,
      encoded.context.maxSteps
    );
    featureData.set(stableCriticStateFeatures(
      problem,
      postAction.nextValues,
      nextStep,
      encoded.context.maxSteps,
      postAction.nextContext
    ), position * STABLE_CRITIC_STATE_DIM);
    const terminal = actionData.diagnostics[position].candidateEnergy === 0 ||
      nextStep >= encoded.context.maxSteps;
    activeMask[position] = terminal ? 0 : 1;
  });

  const features = tf.tensor2d(
    featureData,
    [candidateActions.length, STABLE_CRITIC_STATE_DIM]
  );
  const influence = performanceCriticInfluence(this);
  const continuationValues = this.stateCriticValues(features)
    .mul(tf.tensor1d(activeMask))
    .mul(influence);
  return {
    actorLogits: tf.zeros([candidateActions.length]),
    continuationValues,
    immediateRewards: actionData.immediateRewards,
    diagnostics: actionData.diagnostics
  };
};
