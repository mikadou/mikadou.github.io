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

Each actor proposal samples a node and then a value for that node. A learned-search step uses 24 unique actor proposals plus 8 independent uniformly random legal actions. If the complete legal action space has 32 or fewer actions, all legal actions are used.

## Exact immediate reward

For every final candidate, the optimizer calculates the exact objective consequence analytically. A mutation changes one node, so only its predecessor and successor constraint terms can change.

```text
immediateReward = (currentEnergy - candidateEnergy) / N
```

Reaching energy zero adds a terminal reward. The selected action is then actually applied and the full energy function is recomputed.

## Post-action value critic

The critic is a state-value network. For every candidate:

1. apply the action to a temporary value array;
2. update the objective, best energy, violations, reward history, and stagnation;
3. build node features from that resulting state;
4. pool each node feature by mean, minimum, maximum, and standard deviation;
5. predict continuation value from the fixed-size pooled state vector.

Candidate selection uses:

```text
score = exactImmediateReward + 0.97 * V(resultingState)
```

Terminal states and states beyond the search horizon have zero continuation value.

The critic no longer receives the current state plus action-specific objective hints. It receives a representation of the actual post-action state. Its network is separate from the recurrent actor encoder:

```text
actor:  recurrent graph encoder -> node probability + value distribution
critic: pooled post-action state -> dense 48 ReLU -> continuation value
```

## Grounded actor training

The previous actor target depended on the critic's current predictions, creating a self-teaching loop. The new target is grounded in observed outcomes:

- every sampled candidate receives its exact immediate reward;
- the executed candidate additionally receives its realized discounted continuation return;
- the actor learns a soft target distribution formed from those grounded scores.

This lets an executed temporary uphill move become a positive actor example when it produces enough later improvement.

## Critic replay

The critic learns from Monte Carlo continuation returns of executed actions. Post-action state vectors and targets are stored in a replay buffer of 1,024 transitions. Four replay updates of up to 32 transitions are performed after every episode.

Replay provides repeated, mixed-episode supervision and avoids the previous single update over only the latest trajectory. Since the target is a realized return rather than a bootstrapped critic prediction, this version does not require a target network.

## Search loop

For each learned-search step:

1. the actor directly generates 24 proposals;
2. 8 random legal proposals are added;
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

1. Reset and compare the untrained exact-reward bootstrap against random SA.
2. Train at `N = 8`, `8N` moves for 500–2,000 episodes.
3. Watch actor loss, critic loss, replay size, episode return, final energy, and recent success.
4. Benchmark 30 paired instances with the same initial assignments and move budget.
5. Verify that trained search improves over the untrained immediate-reward-only behavior.

## Current limitations

The critic still learns only from executed actions; the objective provides exact immediate information for all candidates but not their unknown long-term outcomes. Further improvements could include prioritized replay, multi-step TD targets, critic ensembles, uncertainty-aware exploration, or short candidate rollouts.
