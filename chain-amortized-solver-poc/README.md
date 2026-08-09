# RL chain solver PoC

Browser-only proof of concept for the hypothesis that a learned optimizer can discover and amortize a simple recurring optimization rule using reinforcement learning rather than supervised labels.

## Problem

For `n` integer variables with domains `0..n-1`, satisfy the monotonic constraints

```text
x[0] < x[1] < ... < x[n-1]
```

Because there are exactly `n` variables and exactly `n` available integer values, there is only one feasible assignment:

```text
x[i] = i
```

The RL agent is never shown that assignment as a label.

## RL action space

The network chooses both parts of every move:

```text
(i, v)
```

meaning “assign variable `i` the value `v`.” There is no fixed variable order.

A small shared MLP scores every candidate `(i,v)` pair. The same scorer is reused for every node, value, and chain length, avoiding a fixed-size output head and making size extrapolation possible.

## State/action features

The scorer receives normalized structural features for the candidate action, including:

- node id `i`,
- candidate value `v`,
- current value of node `i`,
- neighboring current values,
- whether the node is at a boundary,
- fractions of violated predecessor/successor ordering constraints, and
- proposed value change.

No exact solution or oracle value appears in training data.

## Reward

Episodes start from random permutations. For denser feedback, the environment counts violated implied monotonic orderings

```text
i < j  =>  x[i] < x[j]
```

which are the transitive closure of the chain constraints. Reward is based on the reduction in that violation count, with a small move cost and a terminal bonus when all constraints are satisfied.

## RL algorithm

The current implementation uses approximate Monte-Carlo action-value learning with replay and epsilon-greedy control:

- epsilon starts near `0.90` and decays toward `0.04`,
- complete episode returns are regressed onto the chosen action features,
- a replay buffer mixes recent experience,
- a curriculum gradually expands from short chains to the selected maximum training length.

This is RL/self-discovery rather than imitation learning.

## Benchmark

Default training range: `n=6..20`.

The benchmark evaluates both inside and well outside that range, up to `n=100`, comparing:

- learned greedy policy with at most `2n` joint `(i,v)` moves,
- simulated annealing with `100n` random proposals.

The main metric is solve rate versus chain length. A second chart shows the random starting permutation, the unique `x[i]=i` solution, the learned policy result, and the SA result on the longest held-out chain.

Open the GitHub Pages deployment at `/chain-amortized-solver-poc/`.
