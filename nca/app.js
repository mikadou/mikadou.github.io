(() => {
  const IS_PHONE = matchMedia('(max-width: 900px)').matches || /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
  const CHANNELS = IS_PHONE ? 12 : 16;
  const HIDDEN = IS_PHONE ? 32 : 64;
  const MODEL_URL = 'indexeddb://mikadou-growing-nca-pool-v4';
  const META_KEY = 'mikadou-growing-nca-pool-v4-meta';
  const $ = id => document.getElementById(id);
  const els = {
    log:$('log'), step:$('stepStat'), rolloutStep:$('rolloutStepStat'), loss:$('lossStat'), best:$('bestStat'), worst:$('worstStat'), backend:$('backendStat'),
    train:$('trainBtn'), pause:$('pauseBtn'), one:$('oneBtn'), reset:$('resetBtn'), save:$('saveBtn'), load:$('loadBtn'), def:$('defaultTargetBtn'), phone:$('mobilePresetBtn'), file:$('fileInput'),
    loadRun:$('loadRunBtn'), newRun:$('newRunBtn'), stepRun:$('stepRunBtn'), animateRun:$('animateRunBtn'), stopRun:$('stopRunBtn'), run10:$('run10Btn'), run40:$('run40Btn'), clearRun:$('clearRunBtn'),
    size:$('sizeInput'), batch:$('batchInput'), lr:$('lrInput'), fire:$('fireInput'), minIter:$('minIterInput'), maxIter:$('maxIterInput'), pool:$('poolInput'), reseed:$('reseedInput')
  };
  const targetCanvas = $('targetCanvas');
  const previewCanvas = $('previewCanvas');
  const runCanvas = $('runCanvas');
  const tctx = targetCanvas.getContext('2d', { willReadFrequently:true });
  let model, optimizer, perceptionKernel, targetTensor, pool = [];
  let running = false, step = 0, lastLoss = NaN, lastBest = NaN, lastWorst = NaN;
  let rolloutState = null, rolloutStep = 0, animatingRun = false;

  function log(s){ els.log.textContent = s; }
  function logError(prefix, err){ console.error(err); log(prefix + '\n' + (err && err.message ? err.message : String(err))); }
  addEventListener('error', e => logError('JavaScript error:', e.error || e.message));
  addEventListener('unhandledrejection', e => logError('Async error:', e.reason));
  function val(el, fallback){ return Number(el.value) || fallback; }
  function size(){ return Math.max(16, Math.min(64, val(els.size, 24))); }
  function batch(){ return Math.max(1, Math.min(8, val(els.batch, IS_PHONE ? 2 : 4))); }
  function fire(){ return Math.max(.05, Math.min(1, val(els.fire, .5))); }
  function minIter(){ return Math.max(1, val(els.minIter, 16)); }
  function maxIter(){ return Math.max(minIter(), val(els.maxIter, 40)); }
  function poolSize(){ return Math.max(batch(), Math.min(128, val(els.pool, 64))); }
  function reseedProb(){ return Math.max(0, Math.min(1, val(els.reseed, 10) / 100)); }
  function rnd(a,b){ return a + Math.floor(Math.random() * (b - a + 1)); }
  function nextPaint(){ return new Promise(r => requestAnimationFrame(() => setTimeout(r, 0))); }
  function sleep(ms){ return new Promise(r => setTimeout(r, ms)); }
  function fmtLoss(v){ return Number.isFinite(v) ? v.toExponential(2) : '–'; }
  function stats(){
    els.step.textContent = String(step);
    els.rolloutStep.textContent = String(rolloutStep);
    els.loss.textContent = fmtLoss(lastLoss);
    els.best.textContent = fmtLoss(lastBest);
    els.worst.textContent = fmtLoss(lastWorst);
    els.backend.textContent = tf.getBackend();
  }
  function resizeCanvases(s){ for (const c of [targetCanvas, previewCanvas, runCanvas]) { c.width = s; c.height = s; } }
  function disposePool(){ for (const t of pool) t.dispose(); pool = []; }
  function initPool(){ disposePool(); for (let i=0;i<poolSize();i++) pool.push(seed(1)); }
  function ensurePool(){ if (pool.length !== poolSize() || !pool[0] || pool[0].shape[1] !== size()) initPool(); }
  function disposeRollout(){ if (rolloutState) rolloutState.dispose(); rolloutState = null; rolloutStep = 0; stats(); }
  function sampleIdxs(n){
    const a = Array.from({length: pool.length}, (_,i)=>i);
    for(let i=a.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [a[i],a[j]]=[a[j],a[i]]; }
    return a.slice(0,n);
  }
  function addTargetPresetButtons(){
    const parent = els.def && els.def.parentElement;
    if (!parent || $('carTargetBtn')) return;
    for (const [id, label, kind] of [['carTargetBtn','Car','car'], ['houseTargetBtn','House','house'], ['catTargetBtn','Cat','cat']]) {
      const b = document.createElement('button');
      b.id = id; b.type = 'button'; b.textContent = label;
      b.onclick = () => { drawPresetTarget(kind); resetModel(); };
      parent.appendChild(b);
    }
    if (els.def) els.def.textContent = 'Flower';
  }
  function phonePreset(){
    els.size.value=24; els.batch.value=2; els.lr.value=.002; els.minIter.value=16; els.maxIter.value=40; els.pool.value=64; els.reseed.value=10;
    drawDefaultTarget(); resetModel();
  }
  function clearTarget(){ const s=size(); resizeCanvases(s); tctx.clearRect(0,0,s,s); return s; }
  function drawDefaultTarget(){
    const s=clearTarget(); const cx=s/2, cy=s/2; tctx.save(); tctx.translate(cx,cy);
    for(let i=0;i<6;i++){
      tctx.rotate(Math.PI/3); tctx.fillStyle='rgba(255,170,80,.92)';
      tctx.beginPath(); tctx.ellipse(s*.17,0,s*.18,s*.075,0,0,Math.PI*2); tctx.fill();
    }
    tctx.restore(); tctx.fillStyle='rgba(70,255,215,1)';
    tctx.beginPath(); tctx.arc(cx,cy,s*.12,0,Math.PI*2); tctx.fill(); updateTarget();
  }
  function drawPresetTarget(kind){
    const s = clearTarget();
    const u = s / 24;
    tctx.save();
    tctx.lineCap = 'round'; tctx.lineJoin = 'round';
    if (kind === 'car') {
      tctx.fillStyle='rgba(70,180,255,1)'; tctx.fillRect(5*u,11*u,14*u,5*u);
      tctx.fillStyle='rgba(120,220,255,.95)'; tctx.fillRect(8*u,8*u,7*u,4*u);
      tctx.fillStyle='rgba(8,12,20,1)'; tctx.beginPath(); tctx.arc(8*u,17*u,2*u,0,Math.PI*2); tctx.arc(17*u,17*u,2*u,0,Math.PI*2); tctx.fill();
      tctx.fillStyle='rgba(255,245,120,1)'; tctx.fillRect(18*u,12*u,1.5*u,1.5*u);
    } else if (kind === 'house') {
      tctx.fillStyle='rgba(255,190,110,1)'; tctx.fillRect(6*u,11*u,12*u,9*u);
      tctx.fillStyle='rgba(255,95,95,1)'; tctx.beginPath(); tctx.moveTo(4*u,12*u); tctx.lineTo(12*u,5*u); tctx.lineTo(20*u,12*u); tctx.closePath(); tctx.fill();
      tctx.fillStyle='rgba(80,55,35,1)'; tctx.fillRect(11*u,15*u,3*u,5*u);
      tctx.fillStyle='rgba(115,210,255,1)'; tctx.fillRect(7.5*u,13*u,2.5*u,2.5*u); tctx.fillRect(15*u,13*u,2.5*u,2.5*u);
    } else if (kind === 'cat') {
      tctx.fillStyle='rgba(245,170,75,1)';
      tctx.beginPath(); tctx.arc(12*u,13*u,6*u,0,Math.PI*2); tctx.fill();
      tctx.beginPath(); tctx.moveTo(7.5*u,9.5*u); tctx.lineTo(8.5*u,4.8*u); tctx.lineTo(11*u,8.5*u); tctx.closePath(); tctx.fill();
      tctx.beginPath(); tctx.moveTo(16.5*u,9.5*u); tctx.lineTo(15.5*u,4.8*u); tctx.lineTo(13*u,8.5*u); tctx.closePath(); tctx.fill();
      tctx.fillStyle='rgba(20,25,35,1)'; tctx.beginPath(); tctx.arc(10*u,12.5*u,.8*u,0,Math.PI*2); tctx.arc(14*u,12.5*u,.8*u,0,Math.PI*2); tctx.fill();
      tctx.fillStyle='rgba(255,125,160,1)'; tctx.beginPath(); tctx.arc(12*u,15*u,1*u,0,Math.PI*2); tctx.fill();
    }
    tctx.restore(); updateTarget();
    log('Selected target: ' + kind + '. Model reset; train this target separately.');
  }
  function updateTarget(){
    if(targetTensor) targetTensor.dispose();
    targetTensor = tf.tidy(() => tf.browser.fromPixels(targetCanvas,4).toFloat().div(255).expandDims(0));
  }
  function uploaded(file){
    const url=URL.createObjectURL(file), img=new Image();
    img.onload=()=>{
      const s=clearTarget();
      const scale=Math.min(s/img.width,s/img.height); const w=img.width*scale,h=img.height*scale;
      tctx.drawImage(img,(s-w)/2,(s-h)/2,w,h); updateTarget(); initPool(); disposeRollout(); URL.revokeObjectURL(url);
      log('Loaded target image and reset training pool.');
    };
    img.src=url;
  }
  function makePerceptionKernel(){
    const id=[0,0,0,0,1,0,0,0,0], sx=[-1,0,1,-2,0,2,-1,0,1].map(x=>x/8), sy=[-1,-2,-1,0,0,0,1,2,1].map(x=>x/8);
    const data = new Float32Array(3*3*CHANNELS*3);
    for(let y=0;y<3;y++) for(let x=0;x<3;x++) for(let c=0;c<CHANNELS;c++){
      const k=y*3+x, base=(((y*3+x)*CHANNELS+c)*3); data[base]=id[k]; data[base+1]=sx[k]; data[base+2]=sy[k];
    }
    return tf.tensor4d(data, [3,3,CHANNELS,3]);
  }
  function createModel(){
    const m=tf.sequential();
    m.add(tf.layers.conv2d({inputShape:[null,null,CHANNELS*3],filters:HIDDEN,kernelSize:1,padding:'same',activation:'relu',kernelInitializer:'glorotNormal'}));
    m.add(tf.layers.conv2d({filters:CHANNELS,kernelSize:1,padding:'same',activation:'linear',useBias:false,kernelInitializer:'zeros'}));
    return m;
  }
  function seed(b){
    const s=size(), buf=tf.buffer([b,s,s,CHANNELS]), mid=Math.floor(s/2);
    for(let i=0;i<b;i++) buf.set(1,i,mid,mid,3);
    return buf.toTensor();
  }
  function perceive(x){ return tf.depthwiseConv2d(x, perceptionKernel, 1, 'same'); }
  function aliveMask(x){
    return tf.maxPool(x.slice([0,0,0,3],[-1,-1,-1,1]), 3, 1, 'same').greater(0.1).toFloat();
  }
  function stopGrad(x){ return tf.stopGradient ? tf.stopGradient(x) : x.clone(); }
  function caStep(x){
    const preAlive = stopGrad(aliveMask(x));
    const dx=model.apply(perceive(x));
    const mask=tf.randomUniform([x.shape[0],x.shape[1],x.shape[2],1]).less(fire()).toFloat();
    const y = x.add(dx.mul(mask)).clipByValue(-2, 2);
    const postAlive = stopGrad(aliveMask(y));
    return y.mul(preAlive.mul(postAlive));
  }
  function perSampleLosses(x){
    const rgba = x.slice([0,0,0,0],[-1,-1,-1,4]).clipByValue(0,1);
    const target = targetTensor.tile([x.shape[0],1,1,1]);
    const alpha = target.slice([0,0,0,3],[-1,-1,-1,1]);
    const weight = alpha.mul(4).add(.35);
    return rgba.sub(target).square().mul(weight).mean([1,2,3]);
  }
  async function drawStateToCanvas(state, canvas){
    const img = tf.tidy(() => state.slice([0,0,0,0],[1,-1,-1,4]).squeeze().clipByValue(0,1));
    await tf.browser.toPixels(img, canvas); img.dispose(); stats();
  }
  async function renderPreview(iter=maxIter()){
    const state=tf.tidy(()=>{ let x=seed(1); for(let i=0;i<iter;i++) x=caStep(x); return x; });
    await drawStateToCanvas(state, previewCanvas); state.dispose();
  }
  async function newRun(){
    animatingRun = false;
    disposeRollout();
    rolloutState = seed(1);
    rolloutStep = 0;
    await drawStateToCanvas(rolloutState, runCanvas);
    log('Run viewer: clean center seed created. Use Step or Animate.');
  }
  async function stepRun(count=1){
    if (!rolloutState) { rolloutState = seed(1); rolloutStep = 0; }
    for (let i=0; i<count; i++) {
      const next = tf.tidy(() => caStep(rolloutState));
      rolloutState.dispose(); rolloutState = next; rolloutStep++;
    }
    await drawStateToCanvas(rolloutState, runCanvas);
    log('Run viewer advanced to step ' + rolloutStep + '.');
  }
  async function animateRun(){
    if (animatingRun) return;
    if (!rolloutState) { rolloutState = seed(1); rolloutStep = 0; await drawStateToCanvas(rolloutState, runCanvas); }
    animatingRun = true;
    log('Run viewer animating. Tap Stop to pause.');
    try { while (animatingRun) { await stepRun(1); await sleep(140); } }
    catch (err) { animatingRun = false; logError('Run animation failed:', err); }
  }
  function stopRun(){ animatingRun = false; log('Run viewer stopped at step ' + rolloutStep + '.'); }
  function clearRun(){ animatingRun = false; disposeRollout(); const ctx=runCanvas.getContext('2d'); ctx.clearRect(0,0,runCanvas.width,runCanvas.height); log('Run viewer cleared.'); }
  async function trainOneStep(){
    ensurePool();
    const b=batch(), n=rnd(minIter(),maxIter());
    log('Training step ' + (step + 1) + '… rollout=' + n + ', batch=' + b + ', pool=' + pool.length);
    await nextPaint();
    const idxs = sampleIdxs(b);
    const parts = idxs.map(i => pool[i]);
    let fresh = null;
    if(Math.random() < reseedProb()){
      fresh = seed(1); parts[0] = fresh;
    }
    const x0 = tf.concat(parts, 0);
    if(fresh) fresh.dispose();
    let finalX = null;
    const varList = model.trainableWeights.map(w => w.val);
    const vg = tf.variableGrads(() => {
      let x = x0;
      for(let i=0;i<n;i++) x = caStep(x);
      finalX = tf.keep(x);
      return perSampleLosses(x).mean();
    }, varList);
    optimizer.applyGradients(vg.grads);
    const sampleLossTensor = tf.tidy(() => perSampleLosses(finalX));
    const sampleLosses = Array.from(await sampleLossTensor.data());
    sampleLossTensor.dispose();
    lastLoss = sampleLosses.reduce((a,v)=>a+v,0) / sampleLosses.length;
    lastBest = Math.min(...sampleLosses);
    lastWorst = Math.max(...sampleLosses);
    const worstLocalIndex = sampleLosses.indexOf(lastWorst);
    vg.value.dispose(); Object.values(vg.grads).forEach(g => g.dispose()); x0.dispose();
    for(let j=0;j<idxs.length;j++){
      let newState;
      if(j === worstLocalIndex) newState = seed(1);
      else newState = tf.tidy(() => finalX.slice([j,0,0,0],[1,-1,-1,-1]).clone());
      pool[idxs[j]].dispose(); pool[idxs[j]] = newState;
    }
    finalX.dispose();
    step++; localStorage.setItem(META_KEY + ':step', String(step)); stats();
    if(step % (IS_PHONE ? 3 : 5) === 0) await renderPreview(maxIter());
    if(step % 50 === 0) await save(false);
    log('Training completed step ' + step + '. Alive mask enabled; worst sample reseeded.');
  }
  async function loop(){
    if(running) return; running=true; els.train.disabled=true; els.pause.disabled=false;
    animatingRun = false;
    log('Training started with alive mask enabled.'); await nextPaint();
    try{ while(running){ await trainOneStep(); await nextPaint(); } }
    catch(err){ running=false; els.train.disabled=false; els.pause.disabled=true; logError('Training stopped:', err); }
  }
  function pause(){ running=false; els.train.disabled=false; els.pause.disabled=true; log('Training paused.'); }
  async function resetModel(){
    pause(); animatingRun = false; disposeRollout();
    if(model) model.dispose(); model=createModel(); optimizer=tf.train.adam(val(els.lr,.002));
    step=0; lastLoss=NaN; lastBest=NaN; lastWorst=NaN; initPool(); stats(); await renderPreview(1);
    const ctx=runCanvas.getContext('2d'); ctx.clearRect(0,0,runCanvas.width,runCanvas.height);
    log('Model and pool reset. Alive mask enabled.');
  }
  async function save(verbose=true){
    await model.save(MODEL_URL);
    localStorage.setItem(META_KEY, JSON.stringify({step,size:size(),channels:CHANNELS,pool:poolSize(),savedAt:new Date().toISOString()}));
    if(verbose) log('Saved model checkpoint in this browser.');
  }
  async function load(verbose=true){
    try{
      animatingRun = false; disposeRollout();
      if(model) model.dispose(); model=await tf.loadLayersModel(MODEL_URL); optimizer=tf.train.adam(val(els.lr,.002));
      const meta=JSON.parse(localStorage.getItem(META_KEY)||'{}');
      if(meta.size) els.size.value=meta.size; if(meta.pool) els.pool.value=meta.pool;
      drawDefaultTarget(); initPool(); step=Number(meta.step||localStorage.getItem(META_KEY+':step')||0); stats(); await renderPreview(maxIter());
      if(verbose) log('Loaded saved model. Use New seed to run a clean alive-masked rollout.');
      return true;
    } catch(e){ log('No saved model found in this browser. Train and Save first.'); return false; }
  }
  async function loadRun(){ const ok = await load(false); if(ok){ await newRun(); log('Loaded saved model and started a clean fresh run.'); } }
  async function init(){
    try{
      if(IS_PHONE){ els.size.value=24; els.batch.value=2; els.lr.value=.002; els.minIter.value=16; els.maxIter.value=40; els.pool.value=64; els.reseed.value=10; }
      addTargetPresetButtons();
      await tf.ready(); try{ await tf.setBackend('webgl'); await tf.ready(); }catch(e){}
      perceptionKernel=makePerceptionKernel(); model=createModel(); optimizer=tf.train.adam(val(els.lr,.002));
      drawDefaultTarget(); initPool(); stats(); await renderPreview(1); clearRun();
      log('Ready. Alive mask enabled. Presets added: Flower, Car, House, Cat.');
    } catch(err){ logError('Initialization failed:', err); }
  }
  els.train.onclick=loop;
  els.pause.onclick=pause;
  els.one.onclick=async()=>{ try{ await trainOneStep(); }catch(e){ logError('One-step training failed:',e); } };
  els.reset.onclick=resetModel;
  els.save.onclick=()=>save(true);
  els.load.onclick=()=>load(true);
  els.def.onclick=()=>{ drawDefaultTarget(); resetModel(); };
  els.phone.onclick=phonePreset;
  els.file.onchange=e=>{ if(e.target.files[0]) uploaded(e.target.files[0]); };
  els.size.onchange=()=>{ drawDefaultTarget(); resetModel(); };
  els.pool.onchange=()=>initPool();
  els.loadRun.onclick=()=>{ loadRun(); };
  els.newRun.onclick=()=>{ newRun(); };
  els.stepRun.onclick=()=>{ stepRun(1); };
  els.animateRun.onclick=()=>{ animateRun(); };
  els.stopRun.onclick=()=>{ stopRun(); };
  els.run10.onclick=()=>{ stepRun(10); };
  els.run40.onclick=()=>{ stepRun(40); };
  els.clearRun.onclick=()=>{ clearRun(); };
  init();
})();
