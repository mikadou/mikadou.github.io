# Job Scheduling RL POC

Static browser implementation of a reinforcement-learning proof of concept for fixed-order parallel-machine job scheduling.

## What it implements

- Fixed number of machines, using fixed machine indices.
- Jobs arrive in the sampled order; no LPT sorting for the learned policy.
- Optional job types and changeover matrix.
- Empty-machine changeover is always zero.
- Baselines: random assignment, load-only list scheduling, cost-aware greedy list scheduling, and an offline processing-time-descending greedy reference.
- Neural policy: dependency-free MLP implemented in vanilla JavaScript.
- Training: REINFORCE-style policy gradient.
- Self-play: current agent trains against frozen previous-generation agent.
- Snapshotting: improvement-based only.
- Reward modes: win/loss, makespan difference, normalized makespan difference.

## Run

Open index.html directly in a browser, or serve the folder locally with python3 -m http.server 8000 from the repository root.

## GitHub Pages

This folder is deployed under the user Pages repository. The app is fully static and does not require a backend.
