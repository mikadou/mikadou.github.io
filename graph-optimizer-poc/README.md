# Neural mutation simulated annealing POC

This browser-only experiment compares two simulated-annealing optimizers:

1. **SA + neural mutation:** a recurrent graph network proposes the next `(node, value)` mutation.
2. **SA + random mutation:** the mutation is a uniformly random node and replacement value.

Everything after proposal generation is shared: the initial assignment, proposal budget, temperature schedule, energy function, and Metropolis acceptance rule.

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

## Shared simulated annealing

At proposal step `t`, both methods use:

```text
T(t) = max(1, N / 2) * 0.001^(t / (budget - 1))
```

A candidate with energy change `delta` is accepted with probability:

```text
1                         when delta <= 0
exp(-delta / T(t))        otherwise
```

The best accepted assignment is retained for reporting.

## Neural mutation proposer

- Variable nodes only; directed edges encode `greater than`.
- Node IDs and tensor storage are randomized between problem instances.
- Each node has a persistent 32-dimensional hidden state.
- Two directional message-passing rounds run before every proposal.
- The action head scores every `(node, candidate value)` pair.
- Exact no-op replacements are masked.
- Input features include the current values and constraint margins, annealing progress, current temperature, whether the previous proposal was accepted, and the previous energy delta.

The hidden state persists across proposals in one annealing run and resets between problem instances.

## Training

Training uses REINFORCE while the neural proposals pass through the same simulated-annealing acceptance rule used at evaluation.

Reward contains:

- change in the current accepted energy;
- extra credit for improving the best energy found;
- a small proposal cost;
- a small rejection cost;
- a feasibility bonus;
- a terminal penalty for remaining violation energy.

No target assignment, graph depth label, constructive trajectory, or supervised action is supplied. Training changes only the proposal distribution; the annealing schedule and acceptance rule remain fixed.

## Fair comparison

For every paired comparison:

- both methods receive the same initial assignment;
- both receive the same proposal budget;
- both use the same temperature schedule;
- both use the same acceptance rule;
- separate proposal randomness is used, while acceptance thresholds are seeded identically.

The comparison therefore asks whether the learned mutation distribution is more useful than a uniform random mutation distribution inside simulated annealing.

## Suggested experiment

1. Train on `N = 4..8` for roughly 2,000 episodes.
2. Use a training rollout budget of `8N` proposals.
3. Compare both methods with the same `250N` proposal budget at `N = 8` or `12`.
4. Inspect success rate, median proposals, final energy, acceptance rate, and wall-clock runtime.

The neural version is expected to be slower per proposal because it runs a graph network. Its potential advantage is fewer proposals or a higher success rate at a fixed proposal budget.
