(() => {
  'use strict';
  const C = window.ChainPOC;
  const $ = id => document.getElementById(id);
  const trainBtn = $('trainBtn');
  const benchmarkBtn = $('benchmarkBtn');
  const sampleBtn = $('sampleBtn');
  const statusEl = $('status');
  const progressBar = $('progressBar');
  const resultsBody = $('resultsBody');
  const sampleN = $('sampleN');
  const sampleNOut = $('sampleNOut');
  const benchmarkCanvas = $('benchmarkChart');
  const sampleCanvas = $('sampleChart');
  const sampleMetrics = $('sampleMetrics');

  let policy = null;
  let trained = false;
  let sampleSeed = 42;

  const nextFrame = () => new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));
  const num = (id, fallback) => Number($(id).value) || fallback;
  const fmt = x => x < 10 ? x.toFixed(2) : x < 1000 ? x.toFixed(1) : Math.round(x).toLocaleString();
  function setStatus(text, p = null) {
    statusEl.textContent = text;
    if (p !== null) progressBar.style.width = `${Math.max(0, Math.min(100, p * 100))}%`;
  }
  function setBusy(busy) {
    trainBtn.disabled = busy;
    benchmarkBtn.disabled = busy || !trained;
    sampleBtn.disabled = busy || !trained;
  }

  trainBtn.addEventListener('click', train);
  benchmarkBtn.addEventListener('click', runBenchmark);
  sampleBtn.addEventListener('click', () => { sampleSeed += 101; renderSample(); });
  sampleN.addEventListener('input', () => { sampleNOut.value = sampleN.value; if (trained) renderSample(); });

  async function train() {
    setBusy(true); trained = false;
    const count = Math.max(100, Math.floor(num('trainCount', 700)));
    const nMin = Math.max(8, Math.floor(num('trainMin', 16)));
    const nMax = Math.min(C.MAX_N, Math.max(nMin, Math.floor(num('trainMax', 64))));
    const epochs = Math.max(1, Math.floor(num('epochs', 8)));
    policy = new C.MLPPolicy(new C.RNG(7));

    await nextFrame();
    setStatus(`Building ${count} exact teacher trajectories…`, 0.02);
    const t0 = performance.now();
    const dataset = C.buildImitationDataset(count, nMin, nMax, new C.RNG(12345), p => {
      if (p.done % 100 === 0 || p.done === p.total) setStatus(`Teacher data: ${p.done}/${p.total} instances · ${p.samples.toLocaleString()} decisions`, 0.28 * p.done / p.total);
    });
    await nextFrame();

    let loss = 0;
    for (let e = 0; e < epochs; e++) {
      loss = policy.trainSamples(dataset, { epochs: 1, batchSize: 128, lr: 0.003, rng: new C.RNG(900 + e) });
      setStatus(`Training epoch ${e + 1}/${epochs} · MSE ${loss.toFixed(5)}`, 0.28 + 0.66 * (e + 1) / epochs);
      await nextFrame();
    }
    const seconds = (performance.now() - t0) / 1000;
    trained = true;
    setBusy(false);
    setStatus(`Trained on ${dataset.length.toLocaleString()} variable decisions in ${seconds.toFixed(2)}s · final MSE ${loss.toFixed(5)}`, 1);
    renderSample();
    await runBenchmark();
  }

  async function runBenchmark() {
    if (!trained) return;
    setBusy(true);
    const saMultiplier = Math.max(5, Math.floor(num('saMultiplier', 50)));
    const sizes = [16, 32, 64, 96, 128];
    await nextFrame();
    setStatus('Running unseen scaling benchmark…', 0);
    const rows = C.benchmark(policy, sizes, {
      perSize: 10,
      saMultiplier,
      seed: 7701,
      onProgress: p => {
        if (p.done % 5 === 0 || p.done === p.total) setStatus(`Benchmark ${p.done}/${p.total} instances · currently n=${p.n}`, p.done / p.total);
      }
    });
    drawBenchmark(rows);
    fillTable(rows);
    setBusy(false);
    setStatus(`Benchmark complete · SA budget ${saMultiplier}×n evaluations per method.`, 1);
  }

  function fillTable(rows) {
    resultsBody.innerHTML = rows.map(r => `
      <tr>
        <td>${r.n}</td>
        <td><strong>${fmt(r.policyExcess)}</strong></td>
        <td>${fmt(r.greedyExcess)}</td>
        <td>${fmt(r.saExcess)}</td>
        <td>${fmt(r.blockSaExcess)}</td>
        <td>${r.policyMoves.toLocaleString()}</td>
        <td>${r.saMoves.toLocaleString()}</td>
      </tr>`).join('');
  }

  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || canvas.width;
    const cssH = Math.round(cssW * canvas.height / canvas.width);
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: cssW, h: cssH };
  }

  function drawBenchmark(rows) {
    const { ctx, w, h } = setupCanvas(benchmarkCanvas);
    const pad = { l: 60, r: 24, t: 35, b: 48 };
    const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
    ctx.clearRect(0, 0, w, h);
    const series = [
      ['Policy', 'policyExcess', '#7dd3fc'],
      ['Greedy', 'greedyExcess', '#94a3b8'],
      ['SA single', 'saExcess', '#fbbf24'],
      ['SA block', 'blockSaExcess', '#a78bfa']
    ];
    const vals = rows.flatMap(r => series.map(s => Math.log10(1 + r[s[1]])));
    const yMax = Math.max(1, ...vals) * 1.08;
    const xAt = i => pad.l + (rows.length === 1 ? 0 : i * plotW / (rows.length - 1));
    const yAt = y => pad.t + plotH - y / yMax * plotH;

    ctx.strokeStyle = '#253452'; ctx.lineWidth = 1; ctx.fillStyle = '#8ea2c4'; ctx.font = '12px system-ui';
    for (let k = 0; k <= 4; k++) {
      const yv = yMax * k / 4, y = yAt(yv);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillText(yv.toFixed(1), 12, y + 4);
    }
    rows.forEach((r, i) => { const x = xAt(i); ctx.fillText(String(r.n), x - 8, h - 18); });
    ctx.save(); ctx.translate(16, pad.t + plotH / 2); ctx.rotate(-Math.PI/2); ctx.fillText('log₁₀(mean excess / variable + 1)', -90, 0); ctx.restore();
    ctx.fillText('chain length n', pad.l + plotW / 2 - 35, h - 4);

    series.forEach(([name, key, color], si) => {
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = key === 'policyExcess' ? 3 : 2;
      ctx.beginPath();
      rows.forEach((r, i) => { const x=xAt(i), y=yAt(Math.log10(1+r[key])); i ? ctx.lineTo(x,y) : ctx.moveTo(x,y); });
      ctx.stroke();
      rows.forEach((r, i) => { const x=xAt(i), y=yAt(Math.log10(1+r[key])); ctx.beginPath(); ctx.arc(x,y,key==='policyExcess'?4:3,0,Math.PI*2); ctx.fill(); });
      const lx = pad.l + si * Math.min(150, plotW / 4);
      ctx.fillRect(lx, 12, 18, 3); ctx.fillStyle = '#dce7fb'; ctx.fillText(name, lx + 24, 17);
    });
  }

  function renderSample() {
    if (!trained) return;
    const n = Number(sampleN.value);
    const saMultiplier = Math.max(5, Math.floor(num('saMultiplier', 50)));
    const s = C.sampleComparison(policy, n, sampleSeed, saMultiplier);
    drawSample(s);
    const entries = [
      ['Optimal', s.opt.cost], ['Policy', s.policy.cost], ['Greedy', s.greedy.cost], ['SA single', s.sa.cost], ['SA block', s.blockSa.cost]
    ];
    sampleMetrics.className = 'metric-row';
    sampleMetrics.innerHTML = entries.map(([name, cost]) => {
      const excess = (cost - s.opt.cost) / n;
      return `<span class="metric"><b>${name}</b> · cost ${fmt(cost)}${name==='Optimal' ? '' : ` · excess/var ${fmt(excess)}`}</span>`;
    }).join('');
  }

  function drawSample(s) {
    const { ctx, w, h } = setupCanvas(sampleCanvas);
    const pad = { l: 48, r: 20, t: 38, b: 42 };
    const plotW = w - pad.l - pad.r, plotH = h - pad.t - pad.b;
    ctx.clearRect(0, 0, w, h);
    const n = s.instance.n;
    const xAt = i => pad.l + i * plotW / Math.max(1, n - 1);
    const yAt = v => pad.t + plotH - v / C.MAX_VALUE * plotH;
    ctx.strokeStyle = '#253452'; ctx.fillStyle = '#8ea2c4'; ctx.font = '12px system-ui';
    for (let v = 0; v <= C.MAX_VALUE; v += 64) {
      const y = yAt(v); ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(w-pad.r,y); ctx.stroke(); ctx.fillText(String(v), 9, y+4);
    }
    const line = (arr, color, width, dash=[]) => {
      ctx.strokeStyle=color; ctx.lineWidth=width; ctx.setLineDash(dash); ctx.beginPath();
      for(let i=0;i<n;i++){const x=xAt(i),y=yAt(arr[i]); i?ctx.lineTo(x,y):ctx.moveTo(x,y);} ctx.stroke(); ctx.setLineDash([]);
    };
    line(s.instance.targets, '#64748b', 1.5, [4,5]);
    line(s.opt.x, '#86efac', 2.6);
    line(s.policy.x, '#7dd3fc', 2.6);
    line(s.greedy.x, '#94a3b8', 1.5);
    line(s.blockSa.x, '#a78bfa', 1.5);
    const legends = [['targets','#64748b'],['optimal','#86efac'],['policy','#7dd3fc'],['greedy','#94a3b8'],['block SA','#a78bfa']];
    legends.forEach(([name,color],i)=>{const x=pad.l+i*105;ctx.fillStyle=color;ctx.fillRect(x,13,16,3);ctx.fillStyle='#dce7fb';ctx.fillText(name,x+21,17);});
    ctx.fillStyle='#8ea2c4'; ctx.fillText('variable index →', pad.l + plotW/2 - 35, h-9);
  }

  function initialDraw() {
    const ctx1 = benchmarkCanvas.getContext('2d');
    ctx1.fillStyle = '#0b1325'; ctx1.fillRect(0,0,benchmarkCanvas.width,benchmarkCanvas.height);
    ctx1.fillStyle = '#7183a4'; ctx1.font = '24px system-ui'; ctx1.fillText('Train the policy to generate benchmark results', 70, 210);
    const ctx2 = sampleCanvas.getContext('2d');
    ctx2.fillStyle = '#0b1325'; ctx2.fillRect(0,0,sampleCanvas.width,sampleCanvas.height);
    ctx2.fillStyle = '#7183a4'; ctx2.font = '24px system-ui'; ctx2.fillText('An unseen chain will appear here after training', 70, 230);
  }

  initialDraw();
})();
