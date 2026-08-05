# Graph actor-critic optimizer POC

This browser-only experiment compares two search strategies on the same finite-domain graph problem:

1. **Actor-critic graph search:** a recurrent graph actor proposes mutations and a value critic estimates the continuation value of the actual resulting state. The selected move is always applied.
2. **Random simulated annealing:** legal mutations are sampled uniformly and accepted or rejected by the fixed Metropolis rule.

The methods share the initial assignment, objective function, and move budget. Simulated annealing is only the comparison baseline.

## Problem

For a shuffled chain of `N` variable nodes, find integer values in `[0, 2N - 1]` satisfying:

```text
x(chain[0]) > x(chain[1]) > ... > x(chain[N - 1])
```

Violation energy is:

```text
sum(max(0, rightValue - leftValue + 1))
```

A feasible assignment has energy zero.

## Direct actor

The actor never enumerates the Cartesian `(node, value)` action space. It predicts:

- a probability distribution over variable nodes;
- a normalized replacement-value mean for each node;
- a replacement-value scale for each node.

Each proposal samples a node and then a value for that node. A learned-search step uses 12 unique actor proposals plus 4 independent uniformly random legal actions. If the complete legal action space has 16 or fewer actions, all legal actions are used.

## Exact immediate reward

For every final candidate, the optimizer calculates the exact objective consequence analytically. A mutation changes one node, so only its predecessor and successor constraint terms can change.

```text
immediateReward = (currentEnergy - candidateEnergy) / N
```

Reaching energy zero adds a terminal reward. The selected action is then actually applied and the full energy function is recomputed.

## Post-action value critic

For every candidate, the critic receives a representation of the actual resulting state:

1. apply the action to a temporary value array;
2. update the objective, best energy, violations, reward history, and stagnation;
3. build node features from that resulting state;
4. pool every feature by mean, minimum, maximum, and standard deviation;
5. predict continuation value from the fixed-size pooled state vector.

Candidate selection uses:

```text
score = exactImmediateReward + criticInfluence * 0.97 * V(resultingState)
```

The critic starts with zero influence for 100 completed episodes. Its influence then ramps toward one over 400 episodes, and it also requires useful replay coverage. This prevents an immature critic from overriding the exact objective signal.

## Grounded actor training

Every candidate receives its exact immediate reward. The executed candidate additionally receives its realized discounted continuation return. The target distribution now uses temperature `0.10`, making objective improvements more distinct than the previous `0.35` target.

Actor training no longer backpropagates through every step of the full episode. The rollout stores recurrent hidden states as detached tensors, then trains on at most 16 evenly distributed trajectory states. The actor still receives the recurrent context that produced each decision, but gradients do not traverse the entire history.

## Critic replay

The replay buffer retains 1,024 post-action transitions. After every episode, the critic performs two updates of up to 64 transitions each. This processes the same maximum number of replay examples as the previous four-by-32 schedule with half as many optimizer invocations.

## Browser performance profile

The performance profile targets avoidable browser overhead while leaving the feature set and network widths unchanged:

- 12 actor candidates plus 4 random candidates instead of 24 plus 8;
- one synchronized read for the actor distribution;
- one synchronized read for actor logits and critic values together;
- actor minibatches of at most 16 detached states;
- two critic batches of up to 64;
- TensorFlow.js production mode enabled before model initialization;
- default training and comparison horizons reduced from `8N` to `4N` moves.

TensorFlow.js normally selects WebGL in a supported browser. The UI displays the active backend. If it reports another backend, browser or GPU configuration may be preventing WebGL use.

## Search loop

For each learned-search step:

1. the actor directly generates 12 proposals;
2. 4 random legal proposals are added;
3. exact immediate reward is calculated for each candidate;
4. the critic evaluates each actual resulting state;
5. one candidate is sampled from the combined scores;
6. the move is always applied;
7. the best state observed is retained.

Ten percent of training selections remain uniform within the final candidate set.

## Baseline simulated annealing

The random baseline uses:

```text
T(t) = max(1, N / 2) * 0.001^(t / (budget - 1))
```

A mutation with energy change `delta` is accepted with probability:

```text
1                         when delta <= 0
exp(-delta / T(t))        otherwise
```

## Suggested experiment

1. Reload the page and reset the model so the new optimizer and production settings start cleanly.
2. Train at `N = 8`, `4N` moves for 500 episodes.
3. Benchmark the trained model against its untrained exact-reward bootstrap and random SA.
4. Increase to `8N` only if the shorter horizon is demonstrably limiting success.
5. Judge learning with fixed evaluation runs rather than exploratory training success alone.

## Current limitations

The critic learns long-term outcomes only for executed actions. The pooled critic representation also discards graph position beyond what is encoded in each feature. Further improvements could include learned graph pooling, prioritized replay, critic ensembles, uncertainty-aware exploration, or parallel environment batches.
