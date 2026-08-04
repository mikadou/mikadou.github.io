function ensureActorCriticParameters(policy) {
  if (policy.actorCriticReady) return;
  const nodeContextDim = HIDDEN_DIM * 2 + FEATURE_DIM;
  const criticContextDim = nodeContextDim + ACTION_FEATURE_DIM;

  policy.wActorNode = policy.add(tf.variable(
    tf.zeros([nodeContextDim, 1]),
    true,
    `${policy.prefix}_wActorNode`
  ));
  policy.bActorNode = policy.add(tf.variable(
    tf.zeros([1]),
    true,
    `${policy.prefix}_bActorNode`
  ));
  policy.wActorValueMean = policy.add(initVariable(
    [nodeContextDim, 1],
    `${policy.prefix}_wActorValueMean`,
    nodeContextDim
  ));
  policy.bActorValueMean = policy.add(tf.variable(
    tf.zeros([1]),
    true,
    `${policy.prefix}_bActorValueMean`
  ));
  policy.wActorValueScale = policy.add(tf.variable(
    tf.zeros([nodeContextDim, 1]),
    true,
    `${policy.prefix}_wActorValueScale`
  ));
  policy.bActorValueScale = policy.add(tf.variable(
    tf.zeros([1]),
    true,
    `${policy.prefix}_bActorValueScale`
  ));

  policy.wCritic1 = policy.add(initVariable(
    [criticContextDim, ACTION_DIM],
    `${policy.prefix}_wCritic1`,
    criticContextDim
  ));
  policy.bCritic1 = policy.add(tf.variable(
    tf.zeros([ACTION_DIM]),
    true,
    `${policy.prefix}_bCritic1`
  ));
  policy.wCritic2 = policy.add(tf.variable(
    tf.zeros([ACTION_DIM, 1]),
    true,
    `${policy.prefix}_wCritic2`
  ));
  policy.bCritic2 = policy.add(tf.variable(
    tf.zeros([1]),
    true,
    `${policy.prefix}_bCritic2`
  ));
  policy.actorCriticReady = true;
}

RecurrentGraphPolicy.prototype.encodeActorCriticState = function encodeActorCriticState(
  problem,
  values,
  hidden,
  step,
  maxSteps,
  rawContext = {}
) {
  const context = actorCriticContext(problem, values, step, maxSteps, rawContext);
  const nodeFeatureData = buildActorCriticNodeFeatures(
    problem,
    values,
    step,
    maxSteps,
    context
  );
  const features = tf.tensor2d(nodeFeatureData, [problem.n, FEATURE_DIM]);
  const predAdj = tf.tensor2d(problem.predMatrix, [problem.n, problem.n]);
  const succAdj = tf.tensor2d(problem.succMatrix, [problem.n, problem.n]);
  const projected = tf.tanh(features.matMul(this.wInput).add(this.bInput));
  let nextHidden = hidden;

  for (let pass = 0; pass < MESSAGE_PASSES; pass++) {
    const predMessage = predAdj.matMul(nextHidden);
    const succMessage = succAdj.matMul(nextHidden);
    const gate = tf.sigmoid(projected.matMul(this.wGateInput)
      .add(predMessage.matMul(this.wGatePred))
      .add(succMessage.matMul(this.wGateSucc))
      .add(nextHidden.matMul(this.wGateSelf))
      .add(this.bGate));
    const candidate = tf.tanh(projected.matMul(this.wCandInput)
      .add(predMessage.matMul(this.wCandPred))
      .add(succMessage.matMul(this.wCandSucc))
      .add(nextHidden.matMul(this.wCandSelf))
      .add(this.bCand));
    nextHidden = gate.mul(candidate)
      .add(tf.scalar(1).sub(gate).mul(nextHidden));
  }

  const nodeContext = tf.concat([nextHidden, projected, features], 1);
  return { hidden: nextHidden, nodeContext, context };
};

RecurrentGraphPolicy.prototype.actorDistribution = function actorDistribution(encoded) {
  ensureActorCriticParameters(this);
  const nodeLogits = encoded.nodeContext.matMul(this.wActorNode)
    .add(this.bActorNode)
    .reshape([encoded.nodeContext.shape[0]]);
  const valueMeans = tf.sigmoid(
    encoded.nodeContext.matMul(this.wActorValueMean).add(this.bActorValueMean)
  ).reshape([encoded.nodeContext.shape[0]]);
  const valueScales = tf.sigmoid(
    encoded.nodeContext.matMul(this.wActorValueScale).add(this.bActorValueScale)
  ).mul(MAX_VALUE_SCALE - MIN_VALUE_SCALE)
    .add(MIN_VALUE_SCALE)
    .reshape([encoded.nodeContext.shape[0]]);
  return { nodeLogits, valueMeans, valueScales };
};

RecurrentGraphPolicy.prototype.actorCandidateLogits = function actorCandidateLogits(
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
  const valueLogDensity = standardized.square().mul(-0.5)
    .sub(scales.log());
  return {
    logits: selectedNodeLogProbabilities.add(valueLogDensity),
    distribution
  };
};

RecurrentGraphPolicy.prototype.scoreActorCriticCandidates = function scoreActorCriticCandidates(
  problem,
  values,
  encoded,
  candidateActions,
  prepared = null
) {
  ensureActorCriticParameters(this);
  if (!candidateActions.length) throw new Error('At least one candidate action is required.');

  const actionData = buildActorCriticActionFeatures(
    problem,
    values,
    encoded.context,
    candidateActions,
    prepared
  );
  const actionFeatures = tf.tensor2d(
    actionData.features,
    [candidateActions.length, ACTION_FEATURE_DIM]
  );
  const candidateNodes = tf.tensor1d(
    Int32Array.from(
      candidateActions,
      actionIndex => Math.floor(actionIndex / problem.domainSize)
    ),
    'int32'
  );
  const selectedNodeContext = encoded.nodeContext.gather(candidateNodes);
  const actionInput = tf.concat([selectedNodeContext, actionFeatures], 1);
  const actorOutput = this.actorCandidateLogits(problem, encoded, candidateActions);

  const criticHidden = tf.relu(actionInput.matMul(this.wCritic1).add(this.bCritic1));
  const continuationValues = criticHidden.matMul(this.wCritic2).add(this.bCritic2)
    .reshape([candidateActions.length]);

  return {
    actorLogits: actorOutput.logits,
    continuationValues,
    immediateRewards: actionData.immediateRewards,
    diagnostics: actionData.diagnostics
  };
};

function generateActorCriticCandidates(
  policy,
  problem,
  values,
  encoded,
  actorRng,
  randomRng
) {
  const legalCount = legalActionCount(problem);
  if (legalCount <= TOTAL_CANDIDATE_COUNT) {
    return {
      candidates: allLegalActions(problem, values),
      actorCount: legalCount,
      randomCount: 0,
      directActor: true
    };
  }

  const actorOutput = tf.tidy(() => policy.actorDistribution(encoded));
  const actorData = {
    nodeLogits: Float32Array.from(actorOutput.nodeLogits.dataSync()),
    valueMeans: Float32Array.from(actorOutput.valueMeans.dataSync()),
    valueScales: Float32Array.from(actorOutput.valueScales.dataSync())
  };
  actorOutput.nodeLogits.dispose();
  actorOutput.valueMeans.dispose();
  actorOutput.valueScales.dispose();

  const actorCandidates = sampleActorActionsDirectly(
    problem,
    values,
    actorData,
    actorRng,
    ACTOR_CANDIDATE_COUNT
  );
  const seen = new Set(actorCandidates);
  const randomCandidates = sampleUniqueUniformActions(
    problem,
    values,
    randomRng,
    RANDOM_CANDIDATE_COUNT,
    seen
  );

  return {
    candidates: actorCandidates.concat(randomCandidates),
    actorCount: actorCandidates.length,
    randomCount: randomCandidates.length,
    directActor: true
  };
}

function computeContinuationTargets(rewards) {
  const targets = new Float32Array(rewards.length);
  let futureReturn = 0;
  for (let index = rewards.length - 1; index >= 0; index--) {
    targets[index] = clamp(futureReturn, -RETURN_CLIP, RETURN_CLIP);
    futureReturn = clamp(
      rewards[index] + ACTOR_CRITIC_DISCOUNT * futureReturn,
      -RETURN_CLIP,
      RETURN_CLIP
    );
  }
  return targets;
}

function crossEntropy(targetProbabilities, logits, temperature) {
  const logProbabilities = logSoftmaxArray(logits, temperature);
  let total = 0;
  for (let index = 0; index < targetProbabilities.length; index++) {
    total -= targetProbabilities[index] * logProbabilities[index];
  }
  return total;
}

function disposeGradientResult(result, clipped) {
  result.value.dispose();
  Object.values(result.grads).forEach(tensor => tensor.dispose());
  Object.values(clipped).forEach(tensor => tensor.dispose());
}
