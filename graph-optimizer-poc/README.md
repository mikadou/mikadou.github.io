# Local actor-critic graph optimizer POC

This browser experiment compares a small learned local-repair policy against random simulated annealing on a finite-domain chain problem.

## Problem

For a shuffled chain of `N` variables, find integer values in `[0, 2N - 1]` satisfying:

```text
x(chain[0]) > x(chain[1]) > ... > x(chain[N - 1])
```

Violation energy is:

```text
sum(max(0, rightValue - leftValue + 1))
```

Energy zero is feasible.

## Why the actor was simplified

The previous recurrent graph actor mixed local constraints, global search history, two message-passing rounds, recurrent state, candidate scoring, and long-horizon return targets. That was unnecessary for this toy problem and produced slow, unstable learning.

A correct action can be estimated from local information:

- whether the node participates in an unsatisfied predecessor or successor constraint;
- its normalized position in the chain;
- predecessor and successor values;
- the immediate locally feasible lower and upper bounds;
- whether those bounds overlap;
- distance from the current value to the feasible interval.

The active actor is therefore a shared per-node MLP:

```text
13 local features -> 16 ReLU units
                  -> node-selection logit
                  -> replacement-value mean
```

There is no actor recurrence and no message passing.

## Actor features

Every node receives 13 normalized features:

1. current value;
2. position from the top of the chain;
3. inverse position, providing a high-to-low positional prior;
4. predecessor value;
5. successor value;
6. local lower bound (`successor + 1`);
7. local upper bound (`predecessor - 1`);
8. predecessor-edge violation;
9. successor-edge violation;
10. total incident violation;
11. distance to the local feasible interval;
12. signed feasible-interval width;
13. whether the local interval is feasible.

These features directly expose the behavior the actor should learn: prioritize violated nodes, propose high values near the top, low values near the bottom, medium values in the middle, and respect adjacent bounds when possible.

## Dense actor targets

Each visited training state generates supervision for every node.

For each node, a local teacher chooses a target value:

```text
positionTarget = round((1 - chainPosition) * domainMax)
```

When predecessor and successor bounds overlap, the target is the positional target clamped into that feasible interval. When they conflict, the teacher chooses the value minimizing adjacent violations, with chain position used as a small tie-breaker.

The teacher then calculates the exact immediate objective change for that target value. Node-selection targets are a sharp softmax over exact improvement plus a small incident-violation tie-breaker. The actor is trained with:

- cross-entropy for node selection;
- weighted mean-squared error for replacement value.

This is dense supervised policy learning from the known toy objective, rather than sparse policy gradients.

## Rollout curriculum

Early training rollouts follow the local teacher so the actor sees useful repair trajectories immediately. Teacher use decays over 300 completed episodes, with a 10% minimum retained for coverage.

The default training horizon is `2N` moves. Actor training uses all visited states in one small update and does not backpropagate through time.

## Search

A learned-search step uses:

- 6 direct actor proposals;
- 2 uniformly random legal proposals;
- exact immediate objective evaluation for all 8 candidates.

Selection is greedy for stability:

```text
score = exactImmediateReward + criticInfluence * 0.97 * V(resultingState)
```

The selected move is always applied and the best state observed is retained.

## Delayed critic

The critic remains a residual continuation-value model over the actual post-action state. It has zero influence for the first 300 episodes and ramps in over the next 600 episodes, subject to replay coverage.

The critic receives one replay update of up to 64 transitions per episode. Actor and critic retain separate Adam optimizer state.

This ordering is intentional:

1. first learn the obvious local repair policy;
2. verify that actor-only evaluation improves;
3. only then allow the critic to influence moves that may trade immediate improvement for future value.

## Suggested experiment

1. Reload the page and reset the model.
2. Train at `N = 8` with the default `2N` training horizon.
3. Compare after 25, 50, 100, and 200 episodes.
4. Watch evaluation success rather than only teacher-assisted training success.
5. Treat critic improvements after 300 episodes as a separate second-stage experiment.

## Baseline

Random simulated annealing samples uniformly from legal mutations and applies the fixed Metropolis acceptance rule. It shares the problem, initialization, objective, and move budget with learned search, but not the policy or acceptance mechanism.
