# Reward-trained recurrent graph optimizer POC

This subfolder contains a browser-only proof of concept comparing:

1. a recurrent graph policy trained only from optimization rewards; and
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
- Nodes may be revisited and candidate values may be reused.
- Only exact no-op assignments are masked.
- The same policy is reused until the graph becomes feasible, the user-selected action budget is exhausted, or the user stops the run.

## Reward-only training

The runtime training path does **not** provide target assignments, node depths, a constructive solution, or teacher actions.

During training, actions are sampled from the policy. The per-step reward is based on:

```text
(previous energy - new energy) / N
- small action cost
+ success bonus when energy reaches zero
- terminal penalty for remaining energy when the rollout ends
```

Training uses REINFORCE with:

- discounted returns;
- a moving return baseline;
- return scaling;
- an entropy bonus for exploration;
- gradient clipping;
- a gradually reduced sampling temperature.

Persistent node state is replayed through the sampled trajectory when computing gradients, so the model can learn history-dependent behavior across optimization steps.

## Running

Open `index.html` through GitHub Pages or any static web server. TensorFlow.js is loaded from a pinned jsDelivr URL.

Reward-only learning is substantially noisier than supervised imitation. A practical starting point is:

1. train on chain lengths 4–8;
2. use a rollout budget of `4 × N`;
3. train for at least 2,000 episodes;
4. compare first on `N = 8`;
5. expand the training range only after loss and benchmark energy begin improving.

The training rollout, evaluation, and SA budgets are user-controlled in the UI. Large evaluation budgets yield periodically to the browser so the Stop button remains responsive.

## Interpretation

Success would show that a recurrent graph policy can improve its search behavior from objective feedback alone. It would not prove general-purpose optimization, because the environment is still a very simple monotonic chain and the dense energy signal exposes useful local structure.
