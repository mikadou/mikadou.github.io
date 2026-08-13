// Successor-state steps-to-go ranker.
// Phase 1 trains variable selection + proposal. Phase 2 freezes them and trains a separate GNN ranker.
async function bootSuccessorStepsRanker() {
  const baseUrl = new URL('./app-gnn-successor-critic.js?v=20260813-37-base', import.meta.url);
  const response = await fetch(baseUrl, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Failed to load base successor engine: HTTP ${response.status}`);
  let src = await response.text();

  function mustReplace(from, to, label) {
    if (!src.includes(from)) throw new Error(`Steps-ranker patch failed: ${label}`);
    src = src.replace(from, to);
  }
  function replaceBetween(startMarker, endMarker, replacement, label) {
    const start = src.indexOf(startMarker);
    if (start < 0) throw new Error(`Steps-ranker patch failed: ${label} start`);
    const end = src.indexOf(endMarker, start);
    if (end < 0) throw new Error(`Steps-ranker patch failed: ${label} end`);
    src = src.slice(0, start) + replacement + src.slice(end);
  }

  mustReplace(
    'const DAGGER_MAX_RATE = 0.75;',
`const DAGGER_MAX_RATE = 0.75;
const PRETRAIN_FRACTION = 0.60;
const RANKER_ROLLOUTS = 3;
const RANKER_BATCH = 2;
const RANKER_EPOCHS = 2;
const RANKER_TEMPERATURE = 0.12;
const RANKER_REGRESSION_WEIGHT = 0.35;`,
    'ranker constants'
  );
  mustReplace('let diagnosticLines = [];', 'let diagnosticLines = [];\nlet rankerParams = null;\nlet rankerOptimizer = null;', 'ranker globals');

  const rankerCode = String.raw`
function pretrainParams() {
  return [params.Winit, params.binit, params.Wself, params.brel, ...params.Wrel,
    params.Wvar, params.bvar, params.Wproposal1, params.bproposal1,
    params.WproposalMean, params.bproposalMean, params.WproposalLogStd, params.bproposalLogStd];
}
function rankerTrainableParams() {
  if (!rankerParams) return [];
  return [rankerParams.Winit, rankerParams.binit, rankerParams.Wself, rankerParams.brel,
    ...rankerParams.Wrel, rankerParams.W1, rankerParams.b1, rankerParams.W2, rankerParams.b2];
}
function disposeRanker() {
  if (rankerParams) for (const p of rankerTrainableParams()) p.dispose();
  if (rankerOptimizer?.dispose) rankerOptimizer.dispose();
  rankerParams = null; rankerOptimizer = null;
}
function initRanker() {
  disposeRanker();
  const s = 0.08;
  rankerParams = {
    Winit: makeVariable([NODE_FEATURE_DIM, EMBED_DIM], s, 'rankInit'),
    binit: makeBias(EMBED_DIM, 'rankInitB'),
    Wself: makeVariable([EMBED_DIM, EMBED_DIM], s, 'rankSelf'),
    brel: makeBias(EMBED_DIM, 'rankRelB'),
    Wrel: Array.from({ length: EDGE_TYPES }, (_, r) => makeVariable([EMBED_DIM, EMBED_DIM], s, 'rankRel' + r)),
    W1: makeVariable([4 * EMBED_DIM, 64], s, 'rankW1'), b1: makeBias(64, 'rankB1'),
    W2: makeVariable([64, 1], s * 0.6, 'rankW2'), b2: makeBias(1, 'rankB2', 0.5)
  };
  rankerOptimizer = tf.train.adam(0.002);
}
function rankerEmbeddings(state) {
  const spec = graphSpec(state.n);
  const nodeX = tf.tensor2d(nodeFeatureData(state), [spec.nodeCount, NODE_FEATURE_DIM]);
  const adj = spec.adjs.map(a => tf.tensor2d(a, [spec.nodeCount, spec.nodeCount]));
  const h0 = tf.relu(tf.matMul(nodeX, rankerParams.Winit).add(rankerParams.binit));
  let h = h0;
  for (let round = 0; round < MESSAGE_ROUNDS; round++) {
    let z = tf.matMul(h, rankerParams.Wself);
    for (let r = 0; r < EDGE_TYPES; r++) z = z.add(tf.matMul(adj[r], tf.matMul(h, rankerParams.Wrel[r])));
    h = tf.relu(h.add(z.add(rankerParams.brel)));
  }
  return { h0, h };
}
function rankerTensor(state, variableId) {
  const rep = rankerEmbeddings(state), spec = graphSpec(state.n);
  const local0 = rep.h0.slice([variableId, 0], [1, EMBED_DIM]);
  const local = rep.h.slice([variableId, 0], [1, EMBED_DIM]);
  const global = rep.h.slice([spec.globalId, 0], [1, EMBED_DIM]);
  const pooled = rep.h.mean(0).reshape([1, EMBED_DIM]);
  const x = tf.concat([local0, local, global, pooled], 1);
  const hidden = tf.relu(tf.matMul(x, rankerParams.W1).add(rankerParams.b1));
  return tf.softplus(tf.matMul(hidden, rankerParams.W2).add(rankerParams.b2).reshape([]));
}
function rankerStepsSnapshot(state, variableId) {
  if (terminalUtility(state)) return 0;
  return tf.tidy(() => rankerTensor(state, variableId).dataSync()[0] * state.n);
}

function makePretrainSample(n, rng, useOnPolicy) {
  const repair = makeRepairTrainingState(n, rng), proposalVar = Math.floor(rng() * n);
  const proposalState = useOnPolicy ? makeLearnerRolloutState(n, rng) : makeSyntheticState(n, proposalVar, rng);
  return { repairState: repair.state, candidateIds: repair.candidateIds, proposalState, proposalVar, teacherValue: teacherSampleValue(proposalVar, rng) };
}
function pretrainSampleLoss(sample) {
  const repairRep = graphEmbeddings(sample.repairState);
  const variableLoss = variableLossFromLogits(variableLogitsFromRep(repairRep, sample.repairState), sample.candidateIds, sample.repairState.n);
  const proposalRep = graphEmbeddings(sample.proposalState);
  const proposalLoss = proposalNllFromRep(proposalRep, sample.proposalState, sample.proposalVar, sample.teacherValue);
  return variableLoss.add(proposalLoss.mul(PROPOSAL_LOSS_WEIGHT));
}
async function pretrainUpdate(samples) {
  let last = NaN;
  for (let epoch = 0; epoch < UPDATE_EPOCHS; epoch++) {
    const cost = optimizer.minimize(() => tf.tidy(() => {
      let total = tf.scalar(0); for (const s of samples) total = total.add(pretrainSampleLoss(s));
      return total.div(samples.length);
    }), true, pretrainParams());
    last = cost.dataSync()[0]; cost.dispose();
  }
  await tf.nextFrame(); return last;
}
function proposalDiagnostics(n, rng, states = 8) {
  let synthMae = 0, onMae = 0;
  for (let s = 0; s < states; s++) {
    const i = Math.floor(rng() * n), target = teacherMeanValue(i);
    const synth = makeSyntheticState(n, i, rng), on = makeLearnerRolloutState(n, rng);
    synthMae += Math.abs(proposalSnapshot(synth, i).mean - target);
    onMae += Math.abs(proposalSnapshot(on, i).mean - target);
  }
  return { synthMae: synthMae / states, onMae: onMae / states };
}

function teacherStepsToGo(startState, rng, maxSteps = null) {
  let state = cloneState(startState), steps = 0, limit = maxSteps == null ? 3 * state.n : maxSteps;
  if (terminalUtility(state)) return 0;
  while (state.assignedCount < state.n && steps < limit) {
    const remaining = []; for (let i = 0; i < state.n; i++) if (!state.assigned[i]) remaining.push(i);
    const i = remaining[Math.floor(rng() * remaining.length)];
    state = propagateSuccessor(state, i, teacherSampleValue(i, rng)); steps++;
  }
  while (steps < limit && state.assignedCount === state.n && !isStrictChain(state.values)) {
    const ids = violatedEndpointIds(state.values); if (!ids.length) break;
    const i = ids[Math.floor(rng() * ids.length)];
    state = propagateSuccessor(state, i, teacherSampleValue(i, rng)); steps++;
  }
  return terminalUtility(state) ? steps : limit + state.n;
}
function meanTeacherSteps(successor, seedBase, rollouts = RANKER_ROLLOUTS) {
  if (terminalUtility(successor)) return 0;
  let total = 0;
  for (let r = 0; r < rollouts; r++) {
    const seed = (seedBase ^ Math.imul(r + 1, 1013904223)) >>> 0;
    total += teacherStepsToGo(successor, mulberry32(seed));
  }
  return total / rollouts;
}
function randomCandidateRolloutState(n, rng) {
  let state = emptyState(n), count = Math.floor(rng() * Math.max(1, Math.min(2 * n, 24)));
  for (let t = 0; t < count; t++) {
    let i;
    if (state.assignedCount < n) {
      const rem = []; for (let k = 0; k < n; k++) if (!state.assigned[k]) rem.push(k);
      i = rem[Math.floor(rng() * rem.length)];
    } else {
      if (isStrictChain(state.values)) break;
      const ids = violatedEndpointIds(state.values); i = ids[Math.floor(rng() * ids.length)];
    }
    const candidates = generateValueCandidates(state, i, rng, true).values;
    state = propagateSuccessor(state, i, candidates[Math.floor(rng() * candidates.length)]);
  }
  return state;
}
function makeRankerBaseState(n, rng) {
  const mode = rng();
  if (mode < 0.35) return makeSyntheticState(n, Math.floor(rng() * n), rng);
  if (mode < 0.70) return makeLearnerRolloutState(n, rng);
  return randomCandidateRolloutState(n, rng);
}
function selectRankerVariable(base, rng) {
  if (base.assignedCount < base.n) {
    const rem = []; for (let i = 0; i < base.n; i++) if (!base.assigned[i]) rem.push(i);
    return rem[Math.floor(rng() * rem.length)];
  }
  if (isStrictChain(base.values)) return null;
  const ids = violatedEndpointIds(base.values); return ids[Math.floor(rng() * ids.length)];
}
function makeRankerExample(n, rng) {
  let base = makeRankerBaseState(n, rng), variableId = selectRankerVariable(base, rng);
  if (variableId == null) { base = makeRepairTrainingState(n, rng).state; variableId = selectRankerVariable(base, rng); }
  const candidates = generateValueCandidates(base, variableId, rng, true).values;
  const seedBase = Math.floor(rng() * 4294967296) >>> 0, successors = [], targets = [];
  for (const v of candidates) {
    const next = propagateSuccessor(base, variableId, v); successors.push(next); targets.push(meanTeacherSteps(next, seedBase));
  }
  return { base, variableId, candidates, successors, targets };
}
function rankerExampleLoss(example) {
  const preds = tf.stack(example.successors.map(s => rankerTensor(s, example.variableId)));
  const target = tf.tensor1d(example.targets.map(x => x / example.base.n));
  const diff = preds.sub(target), abs = diff.abs(), q = tf.minimum(abs, tf.scalar(1)), l = abs.sub(q);
  const huber = q.square().mul(0.5).add(l).mean();
  const targetProb = tf.softmax(target.neg().div(RANKER_TEMPERATURE));
  const predLogProb = tf.logSoftmax(preds.neg().div(RANKER_TEMPERATURE));
  const listwise = targetProb.mul(predLogProb).sum().neg();
  return listwise.add(huber.mul(RANKER_REGRESSION_WEIGHT));
}
async function rankerUpdate(examples) {
  let last = NaN;
  for (let epoch = 0; epoch < RANKER_EPOCHS; epoch++) {
    const cost = rankerOptimizer.minimize(() => tf.tidy(() => {
      let total = tf.scalar(0); for (const ex of examples) total = total.add(rankerExampleLoss(ex));
      return total.div(examples.length);
    }), true, rankerTrainableParams());
    last = cost.dataSync()[0]; cost.dispose();
  }
  await tf.nextFrame(); return last;
}
function rankerDiagnostics(n, rng, states = 3) {
  let mae = 0, count = 0, top1 = 0, pairGood = 0, pairTotal = 0, targetSpread = 0;
  for (let s = 0; s < states; s++) {
    const ex = makeRankerExample(n, rng), pred = ex.successors.map(st => rankerStepsSnapshot(st, ex.variableId));
    let bestT = 0, bestP = 0, minT = Infinity, maxT = -Infinity;
    for (let k = 0; k < pred.length; k++) {
      mae += Math.abs(pred[k] - ex.targets[k]); count++;
      if (ex.targets[k] < ex.targets[bestT]) bestT = k;
      if (pred[k] < pred[bestP]) bestP = k;
      minT = Math.min(minT, ex.targets[k]); maxT = Math.max(maxT, ex.targets[k]);
    }
    if (bestT === bestP) top1++; targetSpread += maxT - minT;
    for (let a = 0; a < pred.length; a++) for (let b = a + 1; b < pred.length; b++) {
      const td = ex.targets[a] - ex.targets[b]; if (Math.abs(td) < 1e-6) continue;
      pairTotal++; if (td * (pred[a] - pred[b]) > 0) pairGood++;
    }
  }
  return { stepsMae: mae / Math.max(1, count), top1: top1 / states, pairAcc: pairGood / Math.max(1, pairTotal), targetSpread: targetSpread / states };
}
function chooseValueOracleSteps(state, variableId, rng, stochastic = true) {
  const candidates = generateValueCandidates(state, variableId, rng, stochastic).values;
  const seedBase = Math.floor(rng() * 4294967296) >>> 0;
  let bestValue = candidates[0], bestSteps = Infinity;
  for (const v of candidates) {
    const steps = meanTeacherSteps(propagateSuccessor(state, variableId, v), seedBase, 5);
    if (steps < bestSteps - 1e-9) { bestSteps = steps; bestValue = v; }
  }
  return bestValue;
}
function runOracleSteps(n, rng, stochastic = true) {
  let state = emptyState(n), maxMoves = MOVE_MULTIPLIER * n, moves = 0;
  while (state.assignedCount < n && moves < maxMoves) {
    const remaining = []; for (let i = 0; i < n; i++) if (!state.assigned[i]) remaining.push(i);
    const i = stochastic ? remaining[Math.floor(rng() * remaining.length)] : remaining[0];
    state = propagateSuccessor(state, i, chooseValueOracleSteps(state, i, rng, stochastic)); moves++;
  }
  while (moves < maxMoves && !isStrictChain(state.values)) {
    const i = chooseRepairVariable(state, rng, stochastic);
    state = propagateSuccessor(state, i, chooseValueOracleSteps(state, i, rng, stochastic)); moves++;
  }
  return { x: Int16Array.from(state.values), solved: isStrictChain(state.values), moves };
}
`;

  const initEnd = '  optimizer = tf.train.adam(0.002);\n}\n';
  if (!src.includes(initEnd)) throw new Error('Steps-ranker patch failed: initModel end');
  src = src.replace(initEnd, initEnd + rankerCode + '\n');

  replaceBetween('function chooseValue(', '\n\nfunction teacherContinuationReturn(',
`function chooseValue(state, variableId, rng, stochastic = true, oracleValue = false, proposalOnly = false, myopicOnly = false) {
  if (oracleValue) return teacherSampleValue(variableId, rng);
  const proposal = proposalSnapshot(state, variableId);
  if (proposalOnly || !rankerParams) return clampValue(stochastic ? gaussian(proposal.mean, proposal.sigma, rng) : proposal.mean);
  const candidates = generateValueCandidates(state, variableId, rng, stochastic).values;
  let bestValue = candidates[0], bestScore = Infinity;
  for (const v of candidates) {
    const next = propagateSuccessor(state, variableId, v);
    const score = myopicOnly ? -(solverObjective(next) - solverObjective(state)) : rankerStepsSnapshot(next, variableId);
    if (score < bestScore) { bestScore = score; bestValue = v; }
  }
  return bestValue;
}`, 'ranker chooseValue');

  replaceBetween('function valueDiagnostics(', '\nfunction solveRate(',
`function valueDiagnostics(n,rng,states=6){
  const p=proposalDiagnostics(n,rng,states),r=rankerParams?rankerDiagnostics(n,rng,Math.min(3,states)):{stepsMae:NaN,top1:0,pairAcc:0,targetSpread:0};
  return{synthMae:p.synthMae,onMae:p.onMae,criticMae:r.stepsMae,rankHit:r.top1,myopicHit:0,pairAcc:r.pairAcc,targetSpread:r.targetSpread};
}`, 'ranker diagnostics alias');

  const trainCode = String.raw`async function trainGeneric(){
  if(!window.tf)throw new Error('TensorFlow.js did not load.');
  MESSAGE_ROUNDS=+messageRounds.value;const startN=+trainMinN.value,maxN=+trainMaxN.value,totalStates=+episodes.value;
  trainedRange={min:startN,max:maxN};trained=false;trainBtn.disabled=true;benchmarkBtn.disabled=true;initModel();disposeRanker();resetLog();
  const pretrainStates=Math.max(100,Math.min(totalStates,Math.round(totalStates*PRETRAIN_FRACTION))),rankStates=Math.max(0,totalStates-pretrainStates);
  logLine('GNN_SUCCESSOR_STEPS_RANKER_DIAGNOSTIC v4');
  logLine(\`CONFIG backend=\${tf.getBackend()} train=\${startN}..\${maxN} states=\${totalStates} pretrain=\${pretrainStates} rankStates=\${rankStates} rounds=\${MESSAGE_ROUNDS} embed=\${EMBED_DIM} proposalSamples=\${PROPOSAL_SAMPLES} randomCandidates=\${RANDOM_CANDIDATES} rankRollouts=\${RANKER_ROLLOUTS} rankTemp=\${RANKER_TEMPERATURE} separateRankerEncoder=1 budget=\${MOVE_MULTIPLIER}n\`);
  const rng=mulberry32(20260826);let completed=0,updates=0,lastLoss=NaN;
  while(completed<pretrainStates){
    const progress=completed/Math.max(1,pretrainStates),rate=daggerRate(progress),batch=[],count=Math.min(BATCH_STATES,pretrainStates-completed);let onCount=0;
    for(let b=0;b<count;b++){const n=startN+Math.floor(rng()*(maxN-startN+1)),on=rng()<rate;if(on)onCount++;batch.push(makePretrainSample(n,rng,on));completed++;}
    lastLoss=await pretrainUpdate(batch);updates++;
    if(updates%4===0||completed>=pretrainStates){const ed=endpointDiagnostics(maxN,mulberry32(811000+updates),16),pd=proposalDiagnostics(maxN,mulberry32(812000+updates),6);statusEl.textContent=\`Pretrain \${completed}/\${pretrainStates} · F1 \${(100*ed.f1).toFixed(0)}% · proposal MAE \${pd.onMae.toFixed(1)}\`;logLine(\`PRETRAIN states=\${completed} update=\${updates} loss=\${lastLoss.toFixed(4)} daggerRate=\${rate.toFixed(2)} onPolicyBatch=\${onCount}/\${count} P=\${(100*ed.precision).toFixed(1)}% R=\${(100*ed.recall).toFixed(1)}% F1=\${(100*ed.f1).toFixed(1)}% proposalMAE_synth=\${pd.synthMae.toFixed(2)} proposalMAE_on=\${pd.onMae.toFixed(2)}\`);await tf.nextFrame();}
  }
  initRanker();let ranked=0,rankUpdates=0;
  while(ranked<rankStates){
    const examples=[],count=Math.min(RANKER_BATCH,rankStates-ranked);
    for(let b=0;b<count;b++){const n=startN+Math.floor(rng()*(maxN-startN+1));statusEl.textContent=\`Labeling successor candidates \${ranked+b+1}/\${rankStates}…\`;examples.push(makeRankerExample(n,rng));}
    lastLoss=await rankerUpdate(examples);ranked+=count;rankUpdates++;
    if(rankUpdates%10===0||ranked>=rankStates){const rd=rankerDiagnostics(maxN,mulberry32(813000+rankUpdates),2),pd=proposalDiagnostics(maxN,mulberry32(814000+rankUpdates),4),solve=solveRate(runLearned,maxN,815000+rankUpdates,2);statusEl.textContent=\`Ranker \${ranked}/\${rankStates} · top1 \${(100*rd.top1).toFixed(0)}% · pair \${(100*rd.pairAcc).toFixed(0)}% · solve@\${maxN} \${(100*solve).toFixed(0)}%\`;logLine(\`RANK states=\${ranked} update=\${rankUpdates} loss=\${lastLoss.toFixed(4)} stepsMAE=\${rd.stepsMae.toFixed(2)} top1=\${(100*rd.top1).toFixed(0)}% pairAcc=\${(100*rd.pairAcc).toFixed(0)}% targetSpread=\${rd.targetSpread.toFixed(2)} proposalMAE_on=\${pd.onMae.toFixed(2)} solve@n\${maxN}=\${(100*solve).toFixed(0)}%\`);await tf.nextFrame();}
  }
  trained=true;trainBtn.disabled=false;benchmarkBtn.disabled=false;statusEl.textContent='Steps-to-go ranker training complete. Running benchmark…';await runBenchmark();
}`;
  replaceBetween('async function trainGeneric(){', '\nfunction benchmarkLengths()', trainCode, 'two-phase training');

  const benchmarkCode = String.raw`async function runBenchmark(){
  if(!trained||!params||!rankerParams)return;trainBtn.disabled=true;benchmarkBtn.disabled=true;const rows=[],rng=mulberry32(20260827);let finalExamples=null;
  for(const n of benchmarkLengths()){
    const trials=n>=80?1:n>=40?2:4;statusEl.textContent=\`Benchmarking steps ranker at n=\${n} · \${trials} trial\${trials===1?'':'s'}…\`;
    let pSolved=0,hSolved=0,saSolved=0,proposalSolved=0,oracleVarSolved=0,oracleValueSolved=0,myopicSolved=0,pViol=0,hViol=0;const pMoves=[],hMoves=[];
    for(let t=0;t<trials;t++){
      const p=runLearned(n,rng,true,false,false,false,false);if(p.solved){pSolved++;pMoves.push(p.moves);}pViol+=violatedConstraintCount(p.x);
      const prop=runLearned(n,rng,true,false,false,true,false);if(prop.solved)proposalSolved++;
      if(n<80){const ov=runLearned(n,rng,true,true,false,false,false);if(ov.solved)oracleVarSolved++;}
      const oval=runLearned(n,rng,true,false,true,false,false);if(oval.solved)oracleValueSolved++;
      const my=runLearned(n,rng,true,false,false,false,true);if(my.solved)myopicSolved++;
      const h=runHandcrafted(n,rng);if(h.solved){hSolved++;hMoves.push(h.moves);}hViol+=violatedConstraintCount(h.x);
      const sa=runSA(n,120*n,rng);if(sa.solved)saSolved++;if(t===0&&n===MAX_TEST_N)finalExamples={p,h,sa};await tf.nextFrame();
    }
    rows.push({n,policySuccess:pSolved/trials,heuristicSuccess:hSolved/trials,saSuccess:saSolved/trials,policyMoves:median(pMoves),heuristicMoves:median(hMoves)});
    logLine(\`BENCH n=\${n} trials=\${trials} gnn=\${Math.round(100*pSolved/trials)}% proposalOnly=\${Math.round(100*proposalSolved/trials)}% oracleVariable=\${n<80?`\${Math.round(100*oracleVarSolved/trials)}%`:'skip'} oracleValue=\${Math.round(100*oracleValueSolved/trials)}% myopicOnly=\${Math.round(100*myopicSolved/trials)}% teacher=\${Math.round(100*hSolved/trials)}% sa=\${Math.round(100*saSolved/trials)}% gnnAvgViol=\${(pViol/trials).toFixed(2)} teacherAvgViol=\${(hViol/trials).toFixed(2)}\`);
  }
  const oracleLengths=[...new Set([trainedRange.max,Math.min(40,trainedRange.max+1),Math.min(40,Math.round(trainedRange.max*1.5))])].filter(n=>n<=40);
  for(const n of oracleLengths){const trials=n<=22?2:1;let solved=0,moves=0,viol=0;for(let t=0;t<trials;t++){const o=runOracleSteps(n,rng,true);if(o.solved)solved++;moves+=o.moves;viol+=violatedConstraintCount(o.x);await tf.nextFrame();}logLine(\`ORACLE_STEPS n=\${n} trials=\${trials} solve=\${Math.round(100*solved/trials)}% avgMoves=\${(moves/trials).toFixed(1)} avgViol=\${(viol/trials).toFixed(2)}\`);}
  drawScaling(rows);const longest=rows[rows.length-1],n=longest.n;if(!finalExamples)finalExamples={p:runLearned(n,mulberry32(9101),true),h:runHandcrafted(n,mulberry32(9102)),sa:runSA(n,120*n,mulberry32(9103))};drawAssignment(finalExamples.p.x,finalExamples.h.x,finalExamples.sa.x);
  const ed=endpointDiagnostics(trainedRange.max,mulberry32(9200),24),rd=rankerDiagnostics(trainedRange.max,mulberry32(9201),3);
  $('policySuccess').textContent=\`\${Math.round(100*longest.policySuccess)}%\`;$('heuristicSuccess').textContent=\`\${Math.round(100*longest.heuristicSuccess)}%\`;$('saSuccess').textContent=\`\${Math.round(100*longest.saSuccess)}%\`;$('candidateMass').textContent=\`\${Math.round(100*ed.f1)}%\`;$('meanMae').textContent=rd.stepsMae.toFixed(1);$('policyMoves').textContent=Number.isFinite(longest.policyMoves)?Math.round(longest.policyMoves):'—';$('heuristicMoves').textContent=Number.isFinite(longest.heuristicMoves)?Math.round(longest.heuristicMoves):'—';$('saEffort').textContent=(120*n).toLocaleString();$('metricLength').textContent=\`n=\${n} · successor graph + changed node → predicted steps-to-go · train \${trainedRange.min}…\${trainedRange.max} · depth \${MESSAGE_ROUNDS}\`;
  appendRankerProbe(trainedRange.max);appendTrace(trainedRange.max);logLine('END');statusEl.textContent='Done. Successor ranker is trained directly from oracle steps-to-go ordering.';trainBtn.disabled=false;benchmarkBtn.disabled=false;
}`;
  replaceBetween('async function runBenchmark(){', '\nfunction appendProbe(', benchmarkCode + '\nfunction appendProbe(', 'ranker benchmark');

  replaceBetween('function appendProbe(', '\nfunction appendTrace(',
`function appendRankerProbe(n){
  let base=makeRankerBaseState(n,mulberry32(9300+n)),i=selectRankerVariable(base,mulberry32(9301+n));
  if(i==null){base=makeRepairTrainingState(n,mulberry32(9302+n)).state;i=selectRankerVariable(base,mulberry32(9303+n));}
  const candidates=generateValueCandidates(base,i,mulberry32(9304+n),false).values,seedBase=(930500+n)>>>0,rows=[];
  for(const v of candidates){const next=propagateSuccessor(base,i,v);rows.push({v,pred:rankerStepsSnapshot(next,i),truth:meanTeacherSteps(next,seedBase,5)});}
  rows.sort((a,b)=>a.pred-b.pred);logLine(\`RANKER_PROBE n=\${n} variable=\${i} phase=\${base.assignedCount<n?'construct':'repair'} best=[\${rows.slice(0,10).map(x=>`\${x.v}:pred\${x.pred.toFixed(1)}/true\${x.truth.toFixed(1)}`).join(',')}]\`);
}
function appendProbe(n){appendRankerProbe(n);}`, 'ranker probe');

  replaceBetween('function appendTrace(', '\n\nfunction setupCanvas(',
`function appendTrace(n){
  const rng=mulberry32(9400+n);let state=emptyState(n),moves=0,maxMoves=MOVE_MULTIPLIER*n;
  for(let step=0;step<16&&moves<maxMoves;step++){
    let i,phase;if(state.assignedCount<n){const rem=[];for(let k=0;k<n;k++)if(!state.assigned[k])rem.push(k);i=rem[Math.floor(rng()*rem.length)];phase='construct';}else{if(isStrictChain(state.values))break;i=chooseRepairVariable(state,rng,true);phase='repair';}
    const candidates=generateValueCandidates(state,i,rng,false).values;let best=null;for(const v of candidates){const next=propagateSuccessor(state,i,v),steps=rankerStepsSnapshot(next,i);if(!best||steps<best.steps)best={v,next,steps};}
    const beforeObj=solverObjective(state);state=best.next;moves++;logLine(\`TRACE step=\${step} phase=\${phase} i=\${i} v=\${best.v} predSteps=\${best.steps.toFixed(2)} objective=\${beforeObj.toFixed(3)}->\${solverObjective(state).toFixed(3)} assigned=\${state.assignedCount}/\${n}\${state.assignedCount===n?` viol=\${violatedConstraintCount(state.values)} energy=\${violationEnergy(state.values)}`:''}\`);
  }
  logLine(\`TRACE_END solved=\${state.assignedCount===n&&isStrictChain(state.values)?1:0} movesShown=\${moves} assigned=\${state.assignedCount}/\${n}\${state.assignedCount===n?` violations=\${violatedConstraintCount(state.values)} energy=\${violationEnergy(state.values)}`:''}\`);
}`, 'ranker trace');

  src = src.replace('Successor-state critic: apply candidate, update graph, score immediate reward + γV(next).', 'Successor ranker: apply candidate, update graph, predict remaining solver steps.');
  document.title = 'Generic GNN Successor Steps-to-Go Ranker';
  const h1 = document.querySelector('h1'); if (h1) h1.textContent = 'Learn to rank real successor graphs by remaining solver steps';
  const lede = document.querySelector('.lede'); if (lede) lede.innerHTML = 'The variable selector and proposal are pretrained first and then frozen. A separate GNN ranker sees each temporarily applied successor graph plus the changed variable node, and learns directly from teacher-measured steps-to-go candidate rankings.';
  const footer = document.querySelector('footer');
  if (footer && !document.getElementById('rankerExplanation')) {
    const section = document.createElement('section'); section.id = 'rankerExplanation'; section.className = 'card explanation';
    section.innerHTML = `<h2>Directly imitate the successful oracle ranking</h2><p>For one state and selected variable, training generates the same candidate set used by search, temporarily applies every candidate, and measures teacher steps-to-go with shared rollout seeds. The ranker predicts normalized remaining steps and is trained with both Huber regression and a listwise softmax ranking loss.</p><p>The ranker has its own GNN encoder. Proposal and variable-selection weights are frozen before ranking training, so ranker gradients cannot degrade those already-working components.</p>`;
    footer.parentNode.insertBefore(section, footer);
  }

  const blobUrl = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
  try { await import(blobUrl); } finally { URL.revokeObjectURL(blobUrl); }
}

bootSuccessorStepsRanker().catch(err => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `Error loading successor steps ranker: ${err.message}`;
});
