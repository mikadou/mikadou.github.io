# Graph actor-critic optimizer POC

This browser-only experiment compares two search strategies on the same finite-domain graph problem:

1. **Actor-critic graph search:** a recurrent graph actor directly samples candidate mutations and a learned critic estimates continuation value. The selected move is always applied.
2. **Random simulated annealing:** legal mutations are sampled uniformly and accepted or rejected by the fixed Metropolis rule.

The methods share the initial assignment, objective function, and move budget. They deliberately do not share an acceptance rule: simulated annealing is now only the baseline.

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

## What was removed

The learned path no longer uses:

- the handcrafted counterfactual utility formula;
- expected Metropolis acceptance as a training target;
- rejection and local-feasibility penalty weights;
- simulated-annealing acceptance during learned training or inference;
- full or bounded joint-action enumeration.

`reward.js` and the sampled-SA override were removed. The shared graph representation, exact energy calculation, reversible assignment mutations, and random-SA baseline remain.

## Direct factorized actor

The legal action is a pair `(variable node, replacement value)`. The actor does not build or rank a list of joint actions.

From the encoded graph state it predicts:

```text
P(variable node | state)
value mean and scale for each node
```

A proposal is sampled in two steps:

```text
sample node from P(node | state)
sample replacement value from that node's learned value distribution
```

The actor generates 24 unique proposals this way. Eight independent uniformly random legal actions are then added for exploration. Only this final candidate set reaches the critic. When the complete legal action space contains 32 or fewer moves, all legal actions are used.

The actor's proposal work is linear in the number of graph nodes plus the number of requested samples. It does not scale with the Cartesian product `N × domainSize`.

## Exact immediate reward

For every final candidate, the runtime computes the exact objective change:

```text
immediateReward = (currentEnergy - candidateEnergy) / N
```

Reaching energy zero adds a terminal success reward. No network is asked to approximate this known immediate effect.

## Learned critic

The critic predicts discounted **continuation return** after taking a candidate action. Candidate selection uses:

```text
candidateScore = immediateReward + gamma * criticContinuation
```

with `gamma = 0.97`.

The critic target is the discounted sum of rewards occurring after the selected action. This lets it learn whether a temporary objective increase creates better future opportunities.

The actor and critic share the graph encoder and use separate output heads:

```text
shared recurrent graph encoder
    ├── factorized actor: node probability + value distribution
    └── critic: continuation value for final candidates
```

The critic head starts at zero. Initial selection is therefore exact one-step hill climbing over the sampled candidates; as the critic learns, long-term continuation value can override immediate preferences.

## Training loop

For each training state:

1. directly sample 24 actor proposals;
2. add 8 random legal proposals;
3. compute exact immediate reward for every candidate;
4. predict continuation value for every candidate;
5. select one candidate from the combined scores with exploration;
6. always apply the selected move;
7. record the transition and continue the episode;
8. calculate discounted continuation-return targets backward through the episode;
9. train the critic against those realized targets;
10. train the factorized actor toward a soft distribution induced by exact reward plus critic value.

Ten percent of training selections are uniform within the final candidate set. The eight random candidates provide an additional exploration path even when the actor becomes concentrated.

## Inference

Inference uses the same architecture:

```text
factorized actor -> 24 direct proposals
                 + 8 random proposals
                 -> exact immediate reward
                 + learned continuation value
                 -> choose and apply one move
```

Before any training, the critic is zero and selection is exact one-step hill climbing over the sampled candidate set. After training, the learned continuation estimate participates in selection.

The implementation always retains the best state observed during a run, so exploratory or critic-driven uphill moves do not erase the best solution found.

## Recurrent horizon

Per-node recurrent hidden state persists for 32 moves and is then reset. Current energy, best energy, violations, recent reward, progress, and stagnation are directly observable, so the recurrent state is not responsible for remembering the entire search history.

The UI defaults intentionally match training and evaluation:

- test `N = 8`;
- training range `N = 8..8`;
- training horizon `8N` moves;
- evaluation horizon `8N` moves.

## Baseline simulated annealing

The random baseline is unchanged. At step `t` it uses:

```text
T(t) = max(1, N / 2) * 0.001^(t / (budget - 1))
```

A proposed mutation with energy change `delta` is accepted with probability:

```text
1                         when delta <= 0
exp(-delta / T(t))        otherwise
```

## Suggested experiment

1. Reset and compare the untrained hill-climb bootstrap against random SA.
2. Train at `N = 8`, `8N` moves for 500–2,000 episodes.
3. Watch actor loss, critic loss, episode return, final energy, and recent success.
4. Benchmark 30 paired instances with the same initial assignments and move budget.
5. Increase graph size gradually and verify that the critic improves over immediate-reward-only selection.

## Current limitation

The critic is trained only from actions actually executed, while exact objective effects are calculated for all final candidates. This is sample-based value learning, not exhaustive counterfactual supervision. A future version could add replay, target networks, multi-step TD updates, critic ensembles, or short rollouts for candidate calibration.
