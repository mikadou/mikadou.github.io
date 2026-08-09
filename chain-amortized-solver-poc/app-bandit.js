const MAX_TEST_N = 100;
const FEATURE_DIM = 2;
const STEPS_PER_EPISODE = 8;
const REPLAY_LIMIT = 6000;
const BATCH_HALF = 32;

let model = null;
let trained = false;
let trainedRange = { min: 4, max: 20 };
let positiveReplay = [];
let negativeReplay = [];

const $ = id => document.getElementById(id);
const trainMinN = $('trainMinN');
const trainMaxN = $('trainMaxN');
const episodes = $('episodes');
const trainBtn = $('trainBtn');
const benchmarkBtn = $('benchmarkBtn');
const statusEl = $('status');

function bindRange(input, labelId) {
  const update = () => $(labelId).textContent = input.value;
  input.addEventListener('input', update);
  update();
}
bindRange(trainMinN, 'trainMinNLabel');
bindRange(trainMaxN, 'trainMaxNLabel');
bindRange(episodes, 'episodesLabel');
trainMinN.addEventListener('input', () => {
  if (+trainMinN.value > +trainMaxN.value) {
    trainMaxN.value = trainMinN.value;
    $('trainMaxNLabel').textContent = trainMaxN.value;
  }
});
trainMaxN.addEventListener('input', () => {
  if (+trainMaxN.value < +trainMinN.value) {
    trainMinN.value = trainMaxN.value;
    $('trainMinNLabel').textContent = trainMinN.value;
  }
});

function mulberry32(seed) {
  return function () {
    let t = seed += 0x6D2B79F5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function actionFeatures(n, i, v) {
  const d = Math.max(1, n - 1);
  return Float32Array.from([i / d, v / d]);
}

// Generic bound propagation for x[0] < x[1] < ... < x[n-1], x[k] in 0..n-1,
// after fixing one variable x[i]=v. No x[i]=i rule appears here.
function fixIsFeasible(n, i, v) {
  const lb = new Int32Array(n);
  const ub = new Int32Array(n);
  ub.fill(n - 1);
  lb[i] = v;
  ub[i] = v;

  for (let pass = 0; pass < n; pass++) {
    let changed = false;
    for (let k = 1; k < n; k++) {
      const next = Math.max(lb[k], lb[k - 1] + 1);
      if (next !== lb[k]) { lb[k] = next; changed = true; }
      if (lb[k] > ub[k]) return false;
    }
    for (let k = n - 2; k >= 0; k--) {
      const next = Math.min(ub[k], ub[k + 1] - 1);
      if (next !== ub[k]) { ub[k] = next; changed = true; }
      if (lb[k] > ub[k]) return false;
    }
    if (!changed) break;
  }
  for (let k = 0; k < n; k++) if (lb[k] > ub[k]) return false;
  return true;
}

function buildModel() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ inputShape: [FEATURE_DIM], units: 32, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 1, activation: 'linear' }));
  m.compile({ optimizer: tf.train.adam(0.003), loss: 'meanSquaredError' });
  return m;
}

function scoreAll(n) {
  const data = new Float32Array(n * n * FEATURE_DIM);
  let p = 0;
  for (let i = 0; i < n; i++) for (let v = 0; v < n; v++) {
    data.set(actionFeatures(n, i, v), p);
    p += FEATURE_DIM;
  }
  const xs = tf.tensor2d(data, [n * n, FEATURE_DIM]);
  const pred = model.predict(xs);
  const values = Float32Array.from(pred.dataSync());
  xs.dispose(); pred.dispose();
  return values;
}

function greedyBanditAction(n) {
  const q = scoreAll(n);
  let best = 0;
  for (let k = 1; k < q.length; k++) if (q[k] > q[best]) best = k;
  return { i: Math.floor(best / n), v: best % n, q: q[best] };
}

function epsilonAt(ep, total) {
  const t = ep / Math.max(1, total - 1);
  return 0.05 + 0.95 * Math.pow(1 - t, 2);
}

function remember(features, reward) {
  const target = reward > 0 ? positiveReplay : negativeReplay;
  target.push(Float32Array.from(features));
  if (target.length > REPLAY_LIMIT) target.splice(0, target.length - REPLAY_LIMIT);
}

async function trainBalancedBatch(rng) {
  if (positiveReplay.length < 8 || negativeReplay.length < 8) return NaN;
  const half = Math.min(BATCH_HALF, positiveReplay.length, negativeReplay.length);
  const m = half * 2;
  const xData = new Float32Array(m * FEATURE_DIM);
  const yData = new Float32Array(m);
  for (let b = 0; b < half; b++) {
    const pos = positiveReplay[Math.floor(rng() * positiveReplay.length)];
    const neg = negativeReplay[Math.floor(rng() * negativeReplay.length)];
    xData.set(pos, b * FEATURE_DIM); yData[b] = 1;
    xData.set(neg, (half + b) * FEATURE_DIM); yData[half + b] = -1;
  }
  const xs = tf.tensor2d(xData, [m, FEATURE_DIM]);
  const ys = tf.tensor2d(yData, [m, 1]);
  try {
    const h = await model.fit(xs, ys, { epochs: 1, batchSize: Math.min(32, m), shuffle: true, verbose: 0 });
    return h.history.loss[0];
  } finally { xs.dispose(); ys.dispose(); }
}

function ruleAccuracy(n) {
  const q = scoreAll(n);
  let correct = 0;
  for (let i = 0; i < n; i++) {
    let bestV = 0;
    for (let v = 1; v < n; v++) if (q[i * n + v] > q[i * n + bestV]) bestV = v;
    if (bestV === i) correct++;
  }
  return correct / n;
}

async function trainRL() {
  if (!window.tf) throw new Error('TensorFlow.js did not load.');
  trainBtn.disabled = true; benchmarkBtn.disabled = true;
  const minN = +trainMinN.value, maxN = +trainMaxN.value, total = +episodes.value;
  trainedRange = { min: minN, max: maxN };
  positiveReplay = []; negativeReplay = []; trained = false;
  if (model) model.dispose();
  model = buildModel();
  const rng = mulberry32(1337);
  let lastLoss = NaN, accuracy = 0;

  for (let ep = 0; ep < total; ep++) {
    const epsilon = epsilonAt(ep, total);
    for (let s = 0; s < STEPS_PER_EPISODE; s++) {
      const n = minN + Math.floor(rng() * (maxN - minN + 1));
      let action;
      if (rng() < epsilon || positiveReplay.length < 8) {
        action = { i: Math.floor(rng() * n), v: Math.floor(rng() * n) };
      } else action = greedyBanditAction(n);

      const reward = fixIsFeasible(n, action.i, action.v) ? 1 : -1;
      remember(actionFeatures(n, action.i, action.v), reward);
    }
    lastLoss = await trainBalancedBatch(rng);

    if (ep % 20 === 0 || ep === total - 1) {
      accuracy = ruleAccuracy(maxN);
      statusEl.textContent = `Episode ${ep + 1}/${total} · ε=${epsilon.toFixed(3)} · rule accuracy ${(100 * accuracy).toFixed(0)}% at n=${maxN} · positive replay ${positiveReplay.length} · negative ${negativeReplay.length}${Number.isFinite(lastLoss) ? ` · loss ${lastLoss.toFixed(4)}` : ''}`;
      await tf.nextFrame();
    }
  }
  trained = true;
  trainBtn.disabled = false; benchmarkBtn.disabled = false;
  statusEl.textContent = `Training complete · learned rule accuracy ${(100 * accuracy).toFixed(0)}% at n=${maxN}. Running benchmark…`;
  await runBenchmark();
}

function constructPolicy(n) {
  const q = scoreAll(n);
  const x = new Int32Array(n); x.fill(-1);
  const unusedI = new Set(Array.from({ length: n }, (_, i) => i));
  const unusedV = new Set(Array.from({ length: n }, (_, i) => i));
  for (let move = 0; move < n; move++) {
    let bestI = -1, bestV = -1, bestQ = -Infinity;
    for (const i of unusedI) for (const v of unusedV) {
      const value = q[i * n + v];
      if (value > bestQ) { bestQ = value; bestI = i; bestV = v; }
    }
    x[bestI] = bestV; unusedI.delete(bestI); unusedV.delete(bestV);
  }
  return { x, solved: isStrictChain(x), moves: n };
}

function isStrictChain(x) {
  for (let i = 0; i + 1 < x.length; i++) if (x[i] >= x[i + 1]) return false;
  return true;
}

function gapEnergy(x) {
  let e = 0;
  for (let i = 0; i < x.length - 1; i++) for (let j = i + 1; j < x.length; j++) {
    e += Math.max(0, (j - i) - (x[j] - x[i]));
  }
  return e;
}

function shuffledPermutation(n, rng) {
  const x = Int32Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1)); [x[i], x[j]] = [x[j], x[i]];
  }
  return x;
}

function runSA(n, evaluations, rng) {
  const x = shuffledPermutation(n, rng);
  let e = gapEnergy(x), bestE = e, best = Int32Array.from(x);
  const T0 = Math.max(1, n / 2), Tend = 0.01;
  for (let step = 0; step < evaluations && bestE > 0; step++) {
    const a = Math.floor(rng() * n), b = Math.floor(rng() * n);
    if (a === b) continue;
    [x[a], x[b]] = [x[b], x[a]];
    const next = gapEnergy(x), delta = next - e;
    const t = step / Math.max(1, evaluations - 1);
    const temp = T0 * Math.pow(Tend / T0, t);
    if (delta <= 0 || rng() < Math.exp(-delta / temp)) {
      e = next;
      if (e < bestE) { bestE = e; best = Int32Array.from(x); }
    } else [x[a], x[b]] = [x[b], x[a]];
  }
  return { x: best, solved: bestE === 0, finalEnergy: bestE, evaluations };
}

function benchmarkLengths() {
  const m = trainedRange.max;
  return [...new Set([trainedRange.min, m, Math.min(MAX_TEST_N, Math.round(m * 1.5)), Math.min(MAX_TEST_N, m * 2), Math.min(MAX_TEST_N, m * 3), MAX_TEST_N])].sort((a,b)=>a-b);
}

async function runBenchmark() {
  if (!trained || !model) return;
  trainBtn.disabled = true; benchmarkBtn.disabled = true;
  const rows = [], rng = mulberry32(20260809), trials = 8;
  for (const n of benchmarkLengths()) {
    statusEl.textContent = `Benchmarking n=${n}…`;
    const policy = constructPolicy(n);
    let saSolved = 0;
    for (let t = 0; t < trials; t++) if (runSA(n, 100 * n, rng).solved) saSolved++;
    rows.push({ n, policySuccess: policy.solved ? 1 : 0, saSuccess: saSolved / trials });
    await tf.nextFrame();
  }
  drawScaling(rows);
  const longest = rows[rows.length - 1], n = longest.n;
  const policy = constructPolicy(n), sa = runSA(n, 100 * n, mulberry32(9002));
  drawAssignment(policy.x, sa.x);
  $('policySuccess').textContent = `${Math.round(100 * longest.policySuccess)}%`;
  $('saSuccess').textContent = `${Math.round(100 * longest.saSuccess)}%`;
  $('policyMoves').textContent = n.toString();
  $('saEffort').textContent = (100 * n).toLocaleString();
  $('metricLength').textContent = `n=${n} (trained on n=${trainedRange.min}–${trainedRange.max})`;
  statusEl.textContent = `Done. The learned scorer constructs a complete assignment in exactly ${n} joint (i,v) choices.`;
  trainBtn.disabled = false; benchmarkBtn.disabled = false;
}

function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1, rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(300, Math.floor(rect.width * dpr));
  canvas.height = Math.floor(Math.max(260, rect.width * 0.46) * dpr);
  const ctx = canvas.getContext('2d'); ctx.setTransform(dpr,0,0,dpr,0,0);
  return { ctx, w: canvas.width/dpr, h: canvas.height/dpr };
}
function axes(ctx,w,h,pad,xl,yl){ctx.clearRect(0,0,w,h);ctx.strokeStyle='#dbe1eb';ctx.beginPath();ctx.moveTo(pad,14);ctx.lineTo(pad,h-pad);ctx.lineTo(w-12,h-pad);ctx.stroke();ctx.fillStyle='#697386';ctx.font='12px system-ui';ctx.fillText(yl,pad+6,24);ctx.textAlign='right';ctx.fillText(xl,w-12,h-10);ctx.textAlign='left';}
function line(ctx,pts,color,width=2.4){if(!pts.length)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.stroke();ctx.fillStyle=color;for(const p of pts){ctx.beginPath();ctx.arc(p[0],p[1],2.8,0,Math.PI*2);ctx.fill();}}
function drawScaling(rows){const{ctx,w,h}=setupCanvas($('scalingChart')),pad=44;axes(ctx,w,h,pad,'chain length n','solve rate');const lo=Math.min(...rows.map(r=>r.n)),hi=Math.max(...rows.map(r=>r.n));const X=n=>pad+(n-lo)/Math.max(1,hi-lo)*(w-pad-20),Y=p=>h-pad-p*(h-pad-28);ctx.fillStyle='#7b8495';ctx.font='11px system-ui';for(const r of rows)ctx.fillText(String(r.n),X(r.n)-6,h-pad+17);for(let k=0;k<=4;k++)ctx.fillText(`${25*k}%`,5,Y(k/4)+4);const bx=X(trainedRange.max);ctx.save();ctx.setLineDash([5,5]);ctx.strokeStyle='#a9b1c2';ctx.beginPath();ctx.moveTo(bx,18);ctx.lineTo(bx,h-pad);ctx.stroke();ctx.restore();ctx.fillText('training max',Math.min(w-85,bx+5),28);line(ctx,rows.map(r=>[X(r.n),Y(r.policySuccess)]),'#5b67d6');line(ctx,rows.map(r=>[X(r.n),Y(r.saSuccess)]),'#dd6b55');}
function drawAssignment(policy,sa){const{ctx,w,h}=setupCanvas($('instanceChart')),pad=44;axes(ctx,w,h,pad,'node id i','assigned value');const n=policy.length,X=i=>pad+i/Math.max(1,n-1)*(w-pad-20),Y=v=>h-pad-v/Math.max(1,n-1)*(h-pad-28),exact=Int32Array.from({length:n},(_,i)=>i);line(ctx,Array.from(exact,(v,i)=>[X(i),Y(v)]),'#2e8b72',2.6);line(ctx,Array.from(policy,(v,i)=>[X(i),Y(v)]),'#5b67d6',2.2);line(ctx,Array.from(sa,(v,i)=>[X(i),Y(v)]),'#dd6b55',1.8);}

trainBtn.addEventListener('click',()=>trainRL().catch(err=>{console.error(err);statusEl.textContent=`Error: ${err.message}`;trainBtn.disabled=false;}));
benchmarkBtn.addEventListener('click',()=>runBenchmark().catch(err=>{console.error(err);statusEl.textContent=`Error: ${err.message}`;trainBtn.disabled=false;benchmarkBtn.disabled=false;}));
if(window.tf){tf.ready().then(()=>{statusEl.textContent=`TensorFlow.js ready · backend: ${tf.getBackend()}. Train the contextual-bandit action scorer to begin.`;});}else statusEl.textContent='TensorFlow.js failed to load.';