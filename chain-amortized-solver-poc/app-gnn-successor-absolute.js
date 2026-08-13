// Runtime variant of the tested successor-state critic.
// Keeps the graph architecture fixed while changing V(s) to absolute outcome minus future action cost.
async function bootSuccessorAbsoluteVariant() {
  const baseUrl = new URL('./app-gnn-successor-critic.js?v=20260813-37-base', import.meta.url);
  const response = await fetch(baseUrl, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Failed to load base successor critic: HTTP ${response.status}`);
  let src = await response.text();

  function mustReplace(from, to, label) {
    if (!src.includes(from)) throw new Error(`Successor-absolute patch failed: ${label}`);
    src = src.replace(from, to);
  }
  function replaceBetween(startMarker, endMarker, replacement, label) {
    const start = src.indexOf(startMarker);
    if (start < 0) throw new Error(`Successor-absolute patch failed: ${label} start`);
    const end = src.indexOf(endMarker, start);
    if (end < 0) throw new Error(`Successor-absolute patch failed: ${label} end`);
    src = src.slice(0, start) + replacement + src.slice(end);
  }

  mustReplace(
    'const GAMMA = 0.97;',
    'const GAMMA = 1.0;\nconst STEP_COST = 0.02;',
    'absolute-value constants'
  );

  replaceBetween(
    'function chooseValue(',
    '\n\nfunction teacherContinuationReturn(',
`function chooseValue(state, variableId, rng, stochastic = true, oracleValue = false, proposalOnly = false, myopicOnly = false) {
  if (oracleValue) return teacherSampleValue(variableId, rng);
  const proposal = proposalSnapshot(state, variableId);
  if (proposalOnly) return clampValue(stochastic ? gaussian(proposal.mean, proposal.sigma, rng) : proposal.mean);
  const candidates = generateValueCandidates(state, variableId, rng, stochastic).values, evals = [];
  for (const v of candidates) {
    const successor = propagateSuccessor(state, variableId, v);
    const local = solverObjective(successor) - solverObjective(state);
    const future = terminalUtility(successor) ? 1 : criticSnapshot(successor);
    const score = myopicOnly ? local - STEP_COST : -STEP_COST + GAMMA * future;
    evals.push({ value: v, local, future, score });
  }
  if (!stochastic) return evals.reduce((a,b)=>b.score>a.score?b:a).value;
  const scores = Float32Array.from(evals, e => e.score), ids = Array.from({ length: evals.length }, (_, i) => i), picked = sampleDistribution(softmaxDistribution(scores, ids, ACTION_TEMPERATURE), rng);
  return evals[picked].value;
}`,
    'successor action scoring'
  );

  replaceBetween(
    'function teacherContinuationReturn(',
    '\nfunction makeLearnerRolloutState(',
`function teacherContinuationReturn(startState, rng, maxSteps = null) {
  let state = cloneState(startState), steps = 0, limit = maxSteps == null ? 3 * state.n : maxSteps;
  if (terminalUtility(state)) return 1;
  while (state.assignedCount < state.n && steps < limit) {
    const remaining = []; for (let i = 0; i < state.n; i++) if (!state.assigned[i]) remaining.push(i);
    const i = remaining[Math.floor(rng() * remaining.length)];
    state = propagateSuccessor(state, i, teacherSampleValue(i, rng));
    steps++;
  }
  while (steps < limit && state.assignedCount === state.n && !isStrictChain(state.values)) {
    const ids = violatedEndpointIds(state.values); if (!ids.length) break;
    const i = ids[Math.floor(rng() * ids.length)];
    state = propagateSuccessor(state, i, teacherSampleValue(i, rng));
    steps++;
  }
  // Absolute continuation target: final solution utility minus compute spent from startState onward.
  // Unsolved capped rollouts get no success utility; their negative cost remains visible.
  const finalUtility = terminalUtility(state) ? 1 : 0;
  return finalUtility - STEP_COST * steps;
}`,
    'absolute continuation target'
  );

  replaceBetween(
    'function makeCriticState(',
    '\nfunction daggerRate(',
`function makeCriticState(n, rng, useOnPolicy) {
  let base = useOnPolicy ? makeLearnerRolloutState(n, rng) : makeSyntheticState(n, Math.floor(rng()*n), rng);
  let i;
  if (base.assignedCount < n) { const rem=[]; for(let k=0;k<n;k++) if(!base.assigned[k]) rem.push(k); i=rem[Math.floor(rng()*rem.length)]; }
  else { const ids=violatedEndpointIds(base.values); i=ids.length ? ids[Math.floor(rng()*ids.length)] : Math.floor(rng()*n); }
  const proposal = proposalSnapshot(base, i);
  let v;
  const mode = rng();
  if (mode < 0.55) v = clampValue(gaussian(proposal.mean, Math.max(1, proposal.sigma), rng));
  else if (mode < 0.85) v = DOMAIN_MIN + Math.floor(rng() * (DOMAIN_SPAN + 1));
  else v = rng() < 0.5 ? DOMAIN_MIN : DOMAIN_MAX;
  return propagateSuccessor(base, i, v);
}`,
    'broad critic successor sampling'
  );

  replaceBetween(
    'function valueDiagnostics(',
    '\nfunction solveRate(',
`function valueDiagnostics(n,rng,states=6){
  let synthMae=0,onMae=0,criticMae=0,rankHit=0,myopicHit=0,targetSpread=0;
  for(let s=0;s<states;s++){
    const i=Math.floor(rng()*n),synth=makeSyntheticState(n,i,rng),on=makeLearnerRolloutState(n,rng),target=teacherMeanValue(i),p1=proposalSnapshot(synth,i),p2=proposalSnapshot(on,i);
    synthMae+=Math.abs(p1.mean-target);onMae+=Math.abs(p2.mean-target);
    const cs=makeCriticState(n,rng,true),actual=teacherContinuationReturn(cs,rng),pred=criticSnapshot(cs);criticMae+=Math.abs(pred-actual);
    let base=makeLearnerRolloutState(n,rng),v;
    if(base.assignedCount<n){const rem=[];for(let k=0;k<n;k++)if(!base.assigned[k])rem.push(k);v=rem[Math.floor(rng()*rem.length)];}
    else{if(isStrictChain(base.values))base=makeRepairTrainingState(n,rng).state;const ids=violatedEndpointIds(base.values);v=ids[Math.floor(rng()*ids.length)];}
    const cand=generateValueCandidates(base,v,rng,false).values;
    let bestReturn=-Infinity,bestScore=-Infinity,bestMyopic=-Infinity,scorePick=-1,returnPick=-1,myopicPick=-1,minTarget=Infinity,maxTarget=-Infinity;
    for(let k=0;k<cand.length;k++){
      const next=propagateSuccessor(base,v,cand[k]);
      const futureTarget=terminalUtility(next)?1:teacherContinuationReturn(next,rng);
      const trueQ=-STEP_COST+GAMMA*futureTarget;
      const predictedFuture=terminalUtility(next)?1:criticSnapshot(next);
      const score=-STEP_COST+GAMMA*predictedFuture;
      const myopic=(solverObjective(next)-solverObjective(base))-STEP_COST;
      minTarget=Math.min(minTarget,trueQ);maxTarget=Math.max(maxTarget,trueQ);
      if(trueQ>bestReturn){bestReturn=trueQ;returnPick=k;}
      if(score>bestScore){bestScore=score;scorePick=k;}
      if(myopic>bestMyopic){bestMyopic=myopic;myopicPick=k;}
    }
    if(scorePick===returnPick)rankHit++;
    if(myopicPick===returnPick)myopicHit++;
    targetSpread+=Math.max(0,maxTarget-minTarget);
  }
  return{synthMae:synthMae/states,onMae:onMae/states,criticMae:criticMae/states,rankHit:rankHit/states,myopicHit:myopicHit/states,targetSpread:targetSpread/states};
}`,
    'absolute-value diagnostics'
  );

  replaceBetween(
    'function appendProbe(',
    '\nfunction appendTrace(',
`function appendProbe(n){
  let state=makeLearnerRolloutState(n,mulberry32(9300+n)),i;
  if(state.assignedCount<n){const rem=[];for(let k=0;k<n;k++)if(!state.assigned[k])rem.push(k);i=rem[0];}
  else{if(isStrictChain(state.values))state=makeRepairTrainingState(n,mulberry32(9301+n)).state;const ids=violatedEndpointIds(state.values);i=ids[0];}
  const generated=generateValueCandidates(state,i,mulberry32(9302+n),false),rows=[];
  for(const v of generated.values){
    const next=propagateSuccessor(state,i,v),local=solverObjective(next)-solverObjective(state),future=terminalUtility(next)?1:criticSnapshot(next),score=-STEP_COST+GAMMA*future;
    rows.push({v,local,future,score});
  }
  rows.sort((a,b)=>b.score-a.score);
  logLine(\`PROBE_SUCCESSOR n=\${n} variable=\${i} phase=\${state.assignedCount<n?'construct':'repair'} objective=\${solverObjective(state).toFixed(3)} top=[\${rows.slice(0,8).map(x=>\`\${x.v}:local\${x.local.toFixed(2)}/V\${x.future.toFixed(2)}/q\${x.score.toFixed(2)}\`).join(',')}]\`);
}`,
    'probe semantics'
  );

  replaceBetween(
    'function appendTrace(',
    '\n\nfunction setupCanvas(',
`function appendTrace(n){
  const rng=mulberry32(9400+n);let state=emptyState(n),moves=0,maxMoves=MOVE_MULTIPLIER*n;
  for(let step=0;step<16&&moves<maxMoves;step++){
    let i,phase;
    if(state.assignedCount<n){const rem=[];for(let k=0;k<n;k++)if(!state.assigned[k])rem.push(k);i=rem[Math.floor(rng()*rem.length)];phase='construct';}
    else{if(isStrictChain(state.values))break;i=chooseRepairVariable(state,rng,true);phase='repair';}
    const candidates=generateValueCandidates(state,i,rng,false).values;let best=null;
    for(const v of candidates){
      const next=propagateSuccessor(state,i,v),local=solverObjective(next)-solverObjective(state),future=terminalUtility(next)?1:criticSnapshot(next),q=-STEP_COST+GAMMA*future;
      if(!best||q>best.q)best={v,next,local,future,q};
    }
    const beforeObj=solverObjective(state);state=best.next;moves++;
    logLine(\`TRACE step=\${step} phase=\${phase} i=\${i} v=\${best.v} local=\${best.local.toFixed(3)} Vnext=\${best.future.toFixed(3)} q=\${best.q.toFixed(3)} objective=\${beforeObj.toFixed(3)}->\${solverObjective(state).toFixed(3)} assigned=\${state.assignedCount}/\${n}\${state.assignedCount===n?\` viol=\${violatedConstraintCount(state.values)} energy=\${violationEnergy(state.values)}\`:''}\`);
  }
  logLine(\`TRACE_END solved=\${state.assignedCount===n&&isStrictChain(state.values)?1:0} movesShown=\${moves} assigned=\${state.assignedCount}/\${n}\${state.assignedCount===n?\` violations=\${violatedConstraintCount(state.values)} energy=\${violationEnergy(state.values)}\`:''}\`);
}`,
    'trace semantics'
  );

  mustReplace("logLine('GNN_SUCCESSOR_CRITIC_DIAGNOSTIC v1');", "logLine('GNN_SUCCESSOR_ABSOLUTE_CRITIC_DIAGNOSTIC v3');", 'diagnostic header');
  mustReplace('gamma=${GAMMA} daggerWarmup=', 'gamma=${GAMMA} stepCost=${STEP_COST} criticTarget=absoluteUtilityMinusFutureCost daggerWarmup=', 'diagnostic config');
  mustReplace('successorRankHit=${(100*vd.rankHit).toFixed(0)}% myopicRankHit=', 'successorRankHit=${(100*vd.rankHit).toFixed(0)}% criticTargetSpread=${vd.targetSpread.toFixed(3)} myopicRankHit=', 'diagnostic target spread');
  src = src.replace("statusEl.textContent='Successor-state critic training complete. Running benchmark…';", "statusEl.textContent='Absolute-utility cost-to-go critic training complete. Running benchmark…';");
  src = src.replace("statusEl.textContent='Done. Candidate actions are evaluated on their actual successor graphs.';", "statusEl.textContent='Done. V(s) predicts final solution utility minus future solver cost.';");
  src = src.replace('propose → apply/propagate → immediate reward + γV(successor)', 'propose → apply/propagate → −step cost + V(successor)');

  document.title = 'Generic GNN Successor Critic · Absolute Utility';
  const h1 = document.querySelector('h1');
  if (h1) h1.textContent = 'Value real successor graphs by final solution quality minus future solver cost';
  const lede = document.querySelector('.lede');
  if (lede) lede.innerHTML = 'Each proposed value is temporarily applied and propagated. The critic does not get rewarded for having more room to improve: <code>V(s′)</code> predicts absolute final solution utility minus the future solver actions needed from that successor. Candidate actions are ranked by current action cost plus that successor value.';

  const footer = document.querySelector('footer');
  if (footer && !document.getElementById('absoluteValueExplanation')) {
    const section = document.createElement('section');
    section.id = 'absoluteValueExplanation';
    section.className = 'card explanation';
    section.innerHTML = `<h2>Absolute utility removes the “more room to improve” trap</h2>
      <p>The continuation target is now <code>1 − 0.02 × future steps</code> when the teacher reaches feasibility, and only the accumulated negative step cost if the practical rollout guard is exhausted unsolved. We no longer sum future objective deltas.</p>
      <p>The solver objective still lives in the successor graph as state information. A bad action may remain recoverable, but it only receives high value if the resulting state can still reach a good final solution without excessive additional work.</p>`;
    footer.parentNode.insertBefore(section, footer);
  }

  const blobUrl = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
  try { await import(blobUrl); }
  finally { URL.revokeObjectURL(blobUrl); }
}

bootSuccessorAbsoluteVariant().catch(err => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `Error loading absolute-utility successor critic: ${err.message}`;
});
