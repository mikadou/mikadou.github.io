# Amortized chain solver PoC

Browser-only proof of concept for the hypothesis that a learned policy can amortize recurring optimization structure across instances.

## Problem

For integer variables `x[0..n-1]` in `0..127`, solve

```
x[0] > x[1] > ... > x[n-1]
```

while minimizing squared distance to an instance-specific target sequence:

```
sum_i (x[i] - target[i])^2
```

The target generator has a recurring distribution (global trend, smooth motifs, regime shocks, and noise), producing many local violations of the monotone constraint.

## Solvers

- **Exact DP oracle:** O(n * 128), using suffix minima. Used for labels and evaluation only.
- **Learned policy:** a small TensorFlow.js 1-D dilated-convolution network. It maps the entire target sequence to value logits for every position in one forward pass. A minimal feasibility decoder masks values that make a strict descending completion impossible.
- **Simulated annealing:** starts from a feasible greedy chain, proposes only feasible single-variable changes, and uses a geometric temperature schedule. It receives no information from previous instances.

## Experiment

Train the policy on exact solutions at one chain length, then benchmark on held-out instances at both the training length and longer lengths. The main metric is excess objective cost per variable over the exact optimum, plus the amount of search effort (one policy inference versus thousands of SA objective evaluations).

This is intentionally **PoC A**: imitation learning tests whether a reusable neural solver can represent/generalize the structural heuristic. A later experiment should remove oracle supervision during training and learn from objective feedback/self-improvement.

Open the GitHub Pages deployment at `/chain-amortized-solver-poc/`.
