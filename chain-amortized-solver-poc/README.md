# Amortized chain solver PoC

Browser-only proof of concept for the hypothesis that a learned policy can amortize recurring optimization structure across instances and generalize that structure to larger problem sizes.

## Problem

For integer variables `x[0..n-1]` in `0..127`, solve

```text
x[0] > x[1] > ... > x[n-1]
```

while minimizing squared distance to an instance-specific target sequence:

```text
sum_i (x[i] - target[i])^2
```

The target generator has a recurring distribution (global trend, smooth motifs, regime shocks, and noise), producing many local violations of the monotone constraint.

## Solvers

- **Exact DP oracle:** `O(n * 128)`, using suffix minima. Used for labels and evaluation only.
- **Learned policy:** TensorFlow.js bidirectional GRU with per-position value logits. It sees target value plus relative position, but no explicit chain-length feature. A minimal feasibility decoder masks values that make a strict descending completion impossible.
- **Simulated annealing:** starts from a feasible greedy chain, proposes only feasible single-variable changes, and uses a geometric temperature schedule. It receives no information from previous instances.

## Experiment

The first version trained at one fixed length and used a finite-receptive-field CNN. That policy mostly won only at the exact training length, which is evidence of length-specific specialization rather than algorithmic generalization.

The current version instead:

1. trains on a range of lengths (default `12..32`, every fourth length),
2. generates fresh oracle-labeled instances every epoch,
3. uses a bidirectional GRU so the same recurrent computation can run for unseen sequence lengths,
4. removes the raw chain-length input feature, and
5. benchmarks both inside the training range and at lengths up to roughly `3x` the training maximum (capped at 120 because values are `0..127`).

The main metric is excess objective cost per variable over the exact optimum, plus search effort: one policy inference versus thousands of SA objective evaluations.

This is still intentionally **PoC A**: imitation learning tests whether a reusable neural solver can represent and extrapolate a structural heuristic. A later experiment should remove oracle supervision during training and learn from objective feedback/self-improvement.

Open the GitHub Pages deployment at `/chain-amortized-solver-poc/`.
