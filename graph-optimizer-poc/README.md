# Neural mutation simulated annealing POC

This browser-only experiment compares two simulated-annealing optimizers:

1. **SA + neural mutation:** a recurrent graph network proposes the next `(node, value)` mutation.
2. **SA + random mutation:** the mutation is uniformly sampled from every legal node/replacement-value pair.

Both methods share the initial assignment, proposal budget, temperature schedule, energy function, and Metropolis acceptance rule.

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

## Exact untrained baseline

The final action projection starts at zero. All legal mutations therefore have equal logits. Before training, neural SA explicitly uses the same one-draw uniform mutation mapping and the same acceptance random stream as random SA, so proposals, accept/reject decisions, energy history, and final assignment are identical. Runtime differs because neural SA still executes the graph network.

## Features

Each node receives:

- normalized current value;
- predecessor and successor margins;
- incident violation magnitude;
- edge-existence flags;
- proposal progress and remaining budget;
- annealing temperature;
- previous acceptance and energy delta;
- normalized current and best energy;
- current-to-best energy gap;
- violated-edge fraction;
- recent acceptance rate;
- stagnation since the last best-energy improvement.

Each candidate `(node, value)` action additionally receives objective-derived diagnostics:

- proposed value and displacement;
- new predecessor and successor violations;
- exact energy delta;
- resulting total energy;
- Metropolis acceptance probability;
- distance from the locally feasible interval;
- improvement over the best energy found.

These quantities do not reveal a target assignment. They expose the energy function and fixed SA rule that already define the optimization problem.

## Counterfactual training

Training visits states through the same simulated-annealing environment used at evaluation. At every visited state, all legal mutations are evaluated with the real energy function and acceptance probability.

The differentiable objective maximizes expected standardized mutation utility under the policy. Utility rewards expected current-energy reduction, expected best-energy improvement, and immediate feasibility, while penalizing rejection probability and distance from local feasibility. A KL penalty toward the uniform distribution prevents premature collapse, and 10% of rollout proposals remain uniformly random.

This replaces high-variance one-trajectory REINFORCE credit assignment. There are still no solution labels, constructive trajectories, target values, or teacher actions.

## Recurrent horizon

Per-node hidden state persists for 32 proposals and is then reset. Current graph-level state is explicitly observable, so the network does not need to preserve total energy or best-energy history indefinitely.

The UI defaults intentionally match training and evaluation:

- test `N = 8`;
- training range `N = 8..8`;
- training horizon `8N` proposals;
- evaluation horizon `8N` proposals.

The UI warns when graph size or proposal horizon extrapolates substantially beyond the training envelope. Increase size and horizon together only after learning is visible at the matched setting.

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

## Suggested experiment

1. Reset and compare once to verify exact untrained parity.
2. Train at `N = 8`, `8N` proposals for 500–2,000 episodes.
3. Benchmark 30 paired instances with the same `N` and horizon.
4. Inspect success rate, median final energy, acceptance rate, and proposal count.
5. Increase the graph size or horizon gradually, keeping the training and evaluation envelopes close.

The neural version will remain slower per proposal because it evaluates a graph network and action features. Its intended advantage is proposal quality, not wall-clock speed.
