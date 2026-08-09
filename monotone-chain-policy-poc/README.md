# Monotone Chain Learned Solver — PoC 1

A browser-only JavaScript experiment for the hypothesis that a learned policy can amortize recurring optimization structure and beat generic local search as instances scale.

## Problem

For integer variables `x[0..n-1]` in `[0,255]`:

```text
x[0] > x[1] > ... > x[n-1]
```

Minimize a weighted squared distance to a structured target vector:

```text
sum_i w[i] * (x[i] - target[i])^2
```

Targets come from a random but structured generator (smooth trends, periodic motifs, regime steps, and noise).

## Learned solver

- Pure JavaScript, no ML library.
- Shared autoregressive MLP: `14 → 32 → 16 → 1`.
- One network prediction per variable.
- The environment only clips the prediction to the legal interval implied by the monotone chain and the remaining number of variables.
- Exact dynamic programming supplies optimal teacher trajectories for imitation learning and is used as an evaluation oracle only.
- At inference the policy calls no exact solver, CP solver, MCTS, or LNS.

This is deliberately **PoC A: representation / amortization**, not RL yet. A later experiment can replace oracle imitation with reward-driven self-improvement.

## Baselines

- Greedy feasible construction.
- Simulated annealing with single-variable feasible mutations.
- Stronger block simulated annealing that shifts random contiguous intervals.
- Exact DP optimum.

The main metric is mean excess objective cost per variable versus chain length. Policy inference requires exactly `n` decisions; SA is given a configurable multiple of `n` candidate evaluations.
