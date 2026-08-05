'use strict';

// The stability patch trains actor and critic with separate variable lists.
// Main did not previously expose the actor-only list, so provide it lazily
// after the actor-critic parameters have been initialized.
Object.defineProperty(RecurrentGraphPolicy.prototype, 'actorVars', {
  configurable: true,
  get() {
    return [
      this.wInput,
      this.bInput,
      this.wGateInput,
      this.wGatePred,
      this.wGateSucc,
      this.wGateSelf,
      this.bGate,
      this.wCandInput,
      this.wCandPred,
      this.wCandSucc,
      this.wCandSelf,
      this.bCand,
      this.wActorNode,
      this.bActorNode,
      this.wActorValueMean,
      this.bActorValueMean,
      this.wActorValueScale,
      this.bActorValueScale
    ];
  }
});

// TensorFlow.js Adam optimizers keep moment tensors for the variables they
// update. Reusing one optimizer first for actor variables such as [19, 32] and
// then for critic variables such as [76, 48] makes those moment tensors collide,
// producing an incompatible broadcasting error. Give the critic its own Adam
// instance while retaining the existing optimizer for the actor.
const ensureStableActorCriticWithSharedOptimizer = ensureStableActorCritic;
ensureStableActorCritic = function ensureStableActorCriticWithSeparateOptimizer(policy) {
  ensureStableActorCriticWithSharedOptimizer(policy);
  if (!policy.stateCriticOptimizer) {
    policy.stateCriticOptimizer = tf.train.adam(DEFAULT_LEARNING_RATE);
  }
};

function stableGradientSetIsStateCritic(result) {
  const names = Object.keys(result.grads);
  return names.length > 0 && names.every(name => name.includes('StateCritic'));
}

const stableApplyGradientsWithSharedOptimizer = stableApplyGradients;
stableApplyGradients = function stableApplyGradientsWithSeparateOptimizers(
  actorOptimizer,
  result
) {
  const optimizer = stableGradientSetIsStateCritic(result)
    ? state.policy.stateCriticOptimizer
    : actorOptimizer;
  if (!optimizer) throw new Error('Actor-critic optimizer was not initialized.');
  return stableApplyGradientsWithSharedOptimizer(optimizer, result);
};

const disposePolicyWithoutCriticOptimizer = RecurrentGraphPolicy.prototype.dispose;
RecurrentGraphPolicy.prototype.dispose = function disposePolicyWithCriticOptimizer() {
  if (this.stateCriticOptimizer && typeof this.stateCriticOptimizer.dispose === 'function') {
    this.stateCriticOptimizer.dispose();
    this.stateCriticOptimizer = null;
  }
  disposePolicyWithoutCriticOptimizer.call(this);
};
