// Reward-shaped PPO entrypoint.
// It loads the stable PPO engine and changes only terminal reward calculation.
// No intermediate feasibility checks or per-step rewards are introduced.

function replaceOnce(source, needle, replacement, label) {
  if (!source.includes(needle)) throw new Error(`PPO reward patch failed: ${label}`);
  return source.replace(needle, replacement);
}

async function bootRewardShapedPPO() {
  const response = await fetch('./app-gnn-ppo.js?v=20260810-20-base', { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load PPO engine (${response.status}).`);
  let source = await response.text();

  const strictChainFn = `function isStrictChain(x) {
  for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) return false;
  return true;
}`;

  const rewardHelpers = `${strictChainFn}

// Fraction of the model's original adjacent greater-than constraints satisfied
// by the terminal assignment. This is evaluated only at episode termination.
function satisfiedConstraintFraction(x) {
  if (x.length <= 1) return 1;
  let satisfied = 0;
  for (let i = 0; i + 1 < x.length; i++) if (x[i] < x[i + 1]) satisfied++;
  return satisfied / (x.length - 1);
}

// Terminal reward hierarchy:
//   failure: [0, 1)
//   solved at 3n moves: 3
//   solved at n moves: 4 (maximum)
// Thus every feasible solution dominates every partial solution, while the
// constraint fraction gives PPO/critic a dense terminal learning signal.
function terminalReward(x, solved, moves, n) {
  const satisfied = satisfiedConstraintFraction(x);
  if (!solved) return { reward: satisfied, satisfied, efficiency: 0 };

  const minMoves = n;
  const maxMoves = MOVE_MULTIPLIER * n;
  const efficiency = Math.max(0, Math.min(1,
    (maxMoves - moves) / Math.max(1, maxMoves - minMoves)
  ));
  return {
    reward: satisfied + 2 + efficiency,
    satisfied,
    efficiency
  };
}`;

  source = replaceOnce(source, strictChainFn, rewardHelpers, 'reward helpers');

  source = replaceOnce(
    source,
    `  if (stochastic && trajectory.length) {
    const last = trajectory[trajectory.length - 1];
    last.reward = solved ? 1 : -1;`,
    `  const terminal = terminalReward(state.values, solved, moves, n);

  if (stochastic && trajectory.length) {
    const last = trajectory[trajectory.length - 1];
    last.reward = terminal.reward;`,
    'terminal reward assignment'
  );

  source = replaceOnce(
    source,
    `  return { solved, x: Int16Array.from(state.values), trajectory, moves };`,
    `  return {
    solved,
    x: Int16Array.from(state.values),
    trajectory,
    moves,
    reward: terminal.reward,
    satisfiedFraction: terminal.satisfied,
    efficiency: terminal.efficiency
  };`,
    'rollout diagnostics'
  );

  source += '\n//# sourceURL=app-gnn-ppo-shaped-runtime.js\n';
  const blobUrl = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
  try {
    await import(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

bootRewardShapedPPO().catch(err => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `Error loading reward-shaped PPO: ${err.message}`;
});
