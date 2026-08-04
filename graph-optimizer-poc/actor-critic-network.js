function ensureActorCriticParameters(policy) {
  if (policy.actorCriticReady) return;
  const contextDim = HIDDEN_DIM * 2 + FEATURE_DIM + ACTION_FEATURE_DIM;
  policy.wCritic1 = policy.add(initVariable(
    [contextDim, ACTION_DIM],
    `${policy.prefix}_wCritic1`,
    contextDim
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

  const actorHidden = tf.relu(actionInput.matMul(this.wAction1).add(this.bAction1));
  const actorLogits = actorHidden.matMul(this.wAction2).add(this.bAction2)
    .reshape([candidateActions.length]);

  const criticHidden = tf.relu(actionInput.matMul(this.wCritic1).add(this.bCritic1));
  const continuationValues = criticHidden.matMul(this.wCritic2).add(this.bCritic2)
    .reshape([candidateActions.length]);

  return {
    actorLogits,
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
      poolSize: legalCount,
      actorCount: Math.max(0, legalCount - Math.min(RANDOM_CANDIDATE_COUNT, legalCount)),
      randomCount: Math.min(RANDOM_CANDIDATE_COUNT, legalCount)
    };
  }

  const actorPool = sampleUniqueUniformActions(
    problem,
    values,
    actorRng,
    Math.min(ACTOR_POOL_COUNT, legalCount)
  );
  const poolOutput = tf.tidy(() => policy.scoreActorCriticCandidates(
    problem,
    values,
    encoded,
    actorPool
  ));
  const poolLogits = Array.from(poolOutput.actorLogits.dataSync());
  poolOutput.actorLogits.dispose();
  poolOutput.continuationValues.dispose();

  const actorPositions = samplePositionsWithoutReplacement(
    poolLogits,
    Math.min(ACTOR_CANDIDATE_COUNT, actorPool.length),
    actorTrainingTemperature(),
    actorRng
  );
  const actorCandidates = actorPositions.map(position => actorPool[position]);
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
    poolSize: actorPool.length,
    actorCount: actorCandidates.length,
    randomCount: randomCandidates.length
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

