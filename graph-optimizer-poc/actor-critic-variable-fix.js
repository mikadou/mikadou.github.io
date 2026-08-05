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
