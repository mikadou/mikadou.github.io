# Recurrent graph optimizer POC

This subfolder contains a browser-only proof of concept comparing:

1. a learned recurrent graph policy with persistent hidden state per node; and
2. simulated annealing with random variable/value proposals.

## Problem

For a chain of `N` shuffled variable nodes, find integer values in `[0, 2N - 1]` satisfying:

```text
x(chain[0]) > x(chain[1]) > ... > x(chain[N - 1])
```

The violation energy is:

```text
sum(max(0, rightValue - leftValue + 1))
```

A feasible assignment has energy zero.

## Learned policy

- Variable nodes only; directed adjacency represents `greater than`.
- Node IDs and internal tensor storage are randomized between episodes.
- Storage order remains fixed during an episode.
- Each node has a persistent 32-dimensional hidden state.
- Hidden state is initialized to zero at the start of every episode.
- Two directional message-passing rounds run before every assignment.
- The action head scores every `(node, candidate value)` pair.
- The policy is reused in a loop until feasible or the step budget is exhausted.

For stability, the POC uses supervised teacher trajectories. At each step, the teacher finds the first node in chain order whose value differs from:

```text
2N - 1 - depth
```

and assigns that target value. This tests whether the recurrent graph network can learn and generalize the constructive procedure rather than whether policy-gradient training can discover it from sparse reward.

## Running

Open `index.html` through GitHub Pages or any static web server. TensorFlow.js is loaded from a pinned jsDelivr URL.

## Suggested experiment

1. Train on chain lengths 4–12 for 800–2,000 episodes.
2. Compare on `N = 12`.
3. Test extrapolation on `N = 16`, `24`, or `32`.
4. Inspect success rate and assignment count versus SA proposal count.

For very small chains, SA can remain faster in wall-clock time because a neural forward pass costs more than a simple random proposal. The primary expected advantage is proposal efficiency.
