# Neural mutation simulated annealing POC

This browser-only experiment compares two simulated-annealing optimizers:

1. **SA + neural mutation:** a recurrent graph network proposes the next `(node, value)` mutation.
2. **SA + random mutation:** the mutation is sampled uniformly from all legal `(node, replacement value)` pairs.

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

## Exact random initialization

The neural action head scores every legal `(node, value)` mutation. Its final projection matrix and bias are initialized to zero, so every legal mutation has exactly the same logit before training.

Both the neural and random variants use the same flat ordering of legal actions and the same single uniform random draw to select a mutation. Therefore, immediately after reset and before any training, the two variants have:

- identical mutation proposals;
- identical acceptance random numbers;
- identical accept/reject decisions;
- identical energy histories;
- identical final assignments.

Wall-clock runtime is not identical because the neural variant still executes the graph-network forward pass. The UI reports whether untrained trajectory parity was verified.

The uniform baseline is equivalent to choosing a node uniformly and then choosing uniformly from all replacement values except its current value, because every node has the same number of legal replacement values.

## Neural mutation proposer

- Variable nodes only; directed edges encode `greater than`.
- Node IDs and tensor storage are randomized between problem instances.
- Each node has a persistent 32-dimensional hidden state.
- Two directional message-passing rounds run before every proposal.
- Exact no-op replacements are masked.
- Input features include current values, constraint margins, annealing progress, temperature, previous acceptance, and previous energy delta.

The hidden state persists across proposals in one annealing run and resets between problem instances.

## Training

Training uses REINFORCE while neural proposals pass through the same simulated-annealing acceptance rule used at evaluation.

Reward contains:

- change in current accepted energy;
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
- both use the same acceptance-threshold stream;
- before training, both use the same proposal stream exactly;
- after training, only the neural proposal probabilities differ.

## Suggested experiment

1. Reset the model and run a comparison. The UI should report exact untrained parity.
2. Run the 30-instance benchmark before training; success, proposal counts, acceptance rates, and final energies should match.
3. Train on `N = 4..8` for roughly 2,000 episodes with an `8N` rollout budget.
4. Compare both methods with the same `250N` proposal budget.
5. Measure whether training improves over the known-equivalent random starting point.

The neural version is expected to remain slower per proposal because it runs a graph network. Its potential advantage must come from fewer proposals or a higher success rate at a fixed proposal budget.
