// Runtime variant of the tested successor-state critic.
// Keeps the architecture fixed and changes only return semantics / critic coverage.
async function bootSuccessorCostVariant() {
  const baseUrl = new URL('./app-gnn-successor-critic.js?v=20260813-37-base', import.meta.url);
  const response = await fetch(baseUrl, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Failed to load base successor critic: HTTP ${response.status}`);
  let src = await response.text();

  function mustReplace(from, to, label) {
    if (!src.includes(from)) throw new Error(`Successor-cost patch failed: ${label}`);
    src = src.replace(from, to);
  }

  mustReplace(
    'const GAMMA = 0.97;',
    'const GAMMA = 0.995;\nconst STEP_COST = 0.02;',
    'gamma constant'
  );

  mustReplace(
`function immediateReward(before, after) {
  const base = solverObjective(after) - solverObjective(before);
  const bonus = terminalUtility(after) - terminalUtility(before);
  return base + bonus;
}`,
`function immediateReward(before, after) {
  const base = solverObjective(after) - solverObjective(before);
  const bonus = terminalUtility(after) - terminalUtility(before);
  return base - STEP_COST + bonus;
}`,
    'per-action cost reward'
  );

  // The rollout cap remains only a practical training guard, not part of the state.
  // With explicit step cost, every extra recovery action lowers return.
  mustReplace(
    'limit = maxSteps == null ? 2 * state.n : maxSteps;',
    'limit = maxSteps == null ? 3 * state.n : maxSteps;',
    'teacher continuation guard'
  );

  // Critic targets must include genuinely bad successors too, otherwise low MAE can hide
  // an inability to distinguish cheap-to-recover from expensive-to-recover states.
  mustReplace(
    'const proposal = proposalSnapshot(base, i), v = clampValue(gaussian(proposal.mean, Math.max(1, proposal.sigma), rng));\n  return propagateSuccessor(base, i, v);',
    `const proposal = proposalSnapshot(base, i);
  let v;
  const mode = rng();
  if (mode < 0.55) v = clampValue(gaussian(proposal.mean, Math.max(1, proposal.sigma), rng));
  else if (mode < 0.85) v = DOMAIN_MIN + Math.floor(rng() * (DOMAIN_SPAN + 1));
  else v = rng() < 0.5 ? DOMAIN_MIN : DOMAIN_MAX;
  return propagateSuccessor(base, i, v);`,
    'broad critic successor sampling'
  );

  mustReplace(
    "logLine('GNN_SUCCESSOR_CRITIC_DIAGNOSTIC v1');",
    "logLine('GNN_SUCCESSOR_COST_CRITIC_DIAGNOSTIC v2');",
    'diagnostic header'
  );
  mustReplace(
    'gamma=${GAMMA} daggerWarmup=',
    'gamma=${GAMMA} stepCost=${STEP_COST} daggerWarmup=',
    'diagnostic config'
  );
  src = src.replace(
    "statusEl.textContent='Successor-state critic training complete. Running benchmark…';",
    "statusEl.textContent='Step-cost successor critic training complete. Running benchmark…';"
  );
  src = src.replace(
    "statusEl.textContent='Done. Candidate actions are evaluated on their actual successor graphs.';",
    "statusEl.textContent='Done. Successor values include an explicit cost for every additional solver action.';"
  );
  src = src.replace(
    'propose → apply/propagate → immediate reward + γV(successor)',
    'propose → apply/propagate → Δobjective − step cost + γV(successor)'
  );

  document.title = 'Generic GNN Successor Critic + Step Cost';
  const h1 = document.querySelector('h1');
  if (h1) h1.textContent = 'Evaluate real successor graphs, while charging for every extra solver step';
  const lede = document.querySelector('.lede');
  if (lede) lede.innerHTML = 'Each proposed value is temporarily applied and propagated before evaluation. The solver supplies the exact immediate objective change, every action pays a small compute cost, and a shared critic estimates the remaining long-horizon value of the resulting graph. Bad moves are recoverable, but recovery is no longer free.';

  const footer = document.querySelector('footer');
  if (footer && !document.getElementById('stepCostExplanation')) {
    const section = document.createElement('section');
    section.id = 'stepCostExplanation';
    section.className = 'card explanation';
    section.innerHTML = `<h2>Cost-to-go instead of a fixed remaining budget</h2>
      <p>The critic is trained with the same return used at inference: objective improvement, minus <code>${0.02}</code> for every solver action, plus the terminal success bonus. The discount is <code>0.995</code>, so explicit action cost — rather than a hidden short horizon — is the main pressure to solve efficiently.</p>
      <p>A poor tentative assignment is therefore not fatal. If it leads to a valuable state it can still be chosen, but any extra repair work needed later lowers that successor state's value automatically.</p>`;
    footer.parentNode.insertBefore(section, footer);
  }

  const blobUrl = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
  try { await import(blobUrl); }
  finally { URL.revokeObjectURL(blobUrl); }
}

bootSuccessorCostVariant().catch(err => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `Error loading step-cost successor critic: ${err.message}`;
});
