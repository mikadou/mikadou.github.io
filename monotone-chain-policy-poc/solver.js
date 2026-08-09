(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.ChainPOC = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const MAX_VALUE = 255;
  const MAX_N = 128;
  const FEATURE_COUNT = 14;

  class RNG {
    constructor(seed = 123456789) { this.state = seed >>> 0 || 1; this._spare = null; }
    next() {
      let x = this.state;
      x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
      this.state = x >>> 0;
      return this.state / 4294967296;
    }
    int(lo, hi) { return lo + Math.floor(this.next() * (hi - lo + 1)); }
    uniform(lo, hi) { return lo + (hi - lo) * this.next(); }
    normal() {
      if (this._spare !== null) { const s = this._spare; this._spare = null; return s; }
      let u = 0, v = 0;
      while (u <= Number.EPSILON) u = this.next();
      while (v <= Number.EPSILON) v = this.next();
      const mag = Math.sqrt(-2 * Math.log(u));
      const z0 = mag * Math.cos(2 * Math.PI * v);
      this._spare = mag * Math.sin(2 * Math.PI * v);
      return z0;
    }
  }

  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  function generateInstance(n, rng = new RNG()) {
    if (n < 2 || n > MAX_N) throw new Error(`n must be in [2, ${MAX_N}]`);
    const targets = new Float64Array(n);
    const weights = new Float64Array(n);

    const base = rng.uniform(80, 205);
    const trend = rng.uniform(-0.25, 1.25);
    const amp1 = rng.uniform(5, 24);
    const amp2 = rng.uniform(0, 12);
    const period1 = rng.uniform(10, Math.max(14, n * 0.8));
    const period2 = rng.uniform(5, Math.max(8, n * 0.35));
    const phase1 = rng.uniform(0, Math.PI * 2);
    const phase2 = rng.uniform(0, Math.PI * 2);
    const stepAt = rng.int(Math.max(1, Math.floor(n * 0.2)), Math.max(1, Math.floor(n * 0.8)));
    const stepSize = rng.uniform(-18, 22);
    const noise = rng.uniform(0.5, 4.5);

    for (let i = 0; i < n; i++) {
      const step = i >= stepAt ? stepSize : 0;
      const t = base + trend * i
        + amp1 * Math.sin((2 * Math.PI * i) / period1 + phase1)
        + amp2 * Math.sin((2 * Math.PI * i) / period2 + phase2)
        + step + noise * rng.normal();
      targets[i] = clamp(t, 0, MAX_VALUE);
      const regime = 0.25 * Math.sin((2 * Math.PI * i) / Math.max(7, period2) + phase2);
      weights[i] = clamp(rng.uniform(0.65, 1.55) + regime, 0.35, 2.0);
    }

    return { n, targets, weights };
  }

  function objective(instance, x) {
    let sum = 0;
    for (let i = 0; i < instance.n; i++) {
      const d = x[i] - instance.targets[i];
      sum += instance.weights[i] * d * d;
    }
    return sum;
  }

  function isFeasible(x) {
    for (let i = 0; i < x.length; i++) {
      if (x[i] < 0 || x[i] > MAX_VALUE || !Number.isFinite(x[i])) return false;
      if (i > 0 && x[i - 1] < x[i] + 1) return false;
    }
    return true;
  }

  function exactSolve(instance) {
    const n = instance.n;
    const M = MAX_VALUE;
    let prev = new Float64Array(M + 1);
    let cur = new Float64Array(M + 1);
    const parents = Array.from({ length: n }, () => new Int16Array(M + 1));
    const INF = 1e300;

    for (let v = 0; v <= M; v++) {
      const d = v - instance.targets[0];
      prev[v] = instance.weights[0] * d * d;
      parents[0][v] = -1;
    }

    for (let i = 1; i < n; i++) {
      const suffixBest = new Float64Array(M + 2);
      const suffixArg = new Int16Array(M + 2);
      suffixBest[M + 1] = INF;
      suffixArg[M + 1] = -1;
      let best = INF, arg = -1;
      for (let u = M; u >= 0; u--) {
        if (prev[u] <= best) { best = prev[u]; arg = u; }
        suffixBest[u] = best;
        suffixArg[u] = arg;
      }
      for (let v = 0; v <= M; v++) {
        const minPrev = v + 1 <= M ? suffixBest[v + 1] : INF;
        const d = v - instance.targets[i];
        cur[v] = minPrev + instance.weights[i] * d * d;
        parents[i][v] = v + 1 <= M ? suffixArg[v + 1] : -1;
      }
      const tmp = prev; prev = cur; cur = tmp;
    }

    let bestV = 0, bestCost = prev[0];
    for (let v = 1; v <= M; v++) {
      if (prev[v] < bestCost) { bestCost = prev[v]; bestV = v; }
    }
    const x = new Int16Array(n);
    x[n - 1] = bestV;
    for (let i = n - 1; i > 0; i--) x[i - 1] = parents[i][x[i]];
    return { x, cost: bestCost };
  }

  function precomputeStats(instance) {
    const n = instance.n;
    const suffixMean = new Float64Array(n);
    const suffixStd = new Float64Array(n);
    const suffixMin = new Float64Array(n);
    const suffixMax = new Float64Array(n);
    let sum = 0, sumSq = 0, mn = Infinity, mx = -Infinity;
    for (let i = n - 1; i >= 0; i--) {
      const v = instance.targets[i];
      sum += v; sumSq += v * v; mn = Math.min(mn, v); mx = Math.max(mx, v);
      const k = n - i;
      const mean = sum / k;
      suffixMean[i] = mean;
      suffixStd[i] = Math.sqrt(Math.max(0, sumSq / k - mean * mean));
      suffixMin[i] = mn;
      suffixMax[i] = mx;
    }
    let globalMean = 0;
    for (const t of instance.targets) globalMean += t;
    globalMean /= n;
    const globalSlope = (instance.targets[n - 1] - instance.targets[0]) / Math.max(1, n - 1);
    return { suffixMean, suffixStd, suffixMin, suffixMax, globalMean, globalSlope };
  }

  function legalBounds(n, i, prevValue) {
    const lower = n - i - 1;
    const upper = i === 0 ? MAX_VALUE : prevValue - 1;
    return { lower, upper };
  }

  function makeFeatures(instance, stats, i, prevValue) {
    const n = instance.n;
    const { lower, upper } = legalBounds(n, i, prevValue);
    const t = instance.targets[i];
    const prevT = i > 0 ? instance.targets[i - 1] : t;
    const nextT = i + 1 < n ? instance.targets[i + 1] : t;
    return new Float64Array([
      n === 1 ? 0 : i / (n - 1),
      (n - i - 1) / Math.max(1, n - 1),
      t / MAX_VALUE,
      instance.weights[i] / 2,
      lower / MAX_VALUE,
      upper / MAX_VALUE,
      stats.suffixMean[i] / MAX_VALUE,
      stats.suffixStd[i] / 80,
      stats.suffixMin[i] / MAX_VALUE,
      stats.suffixMax[i] / MAX_VALUE,
      (t - prevT) / 64,
      (nextT - t) / 64,
      stats.globalMean / MAX_VALUE,
      stats.globalSlope / 8
    ]);
  }

  function greedySolve(instance) {
    const n = instance.n;
    const x = new Int16Array(n);
    let prev = MAX_VALUE + 1;
    for (let i = 0; i < n; i++) {
      const { lower, upper } = legalBounds(n, i, prev);
      x[i] = clamp(Math.round(instance.targets[i]), lower, upper);
      prev = x[i];
    }
    return { x, cost: objective(instance, x), moves: n };
  }

  class MLPPolicy {
    constructor(rng = new RNG(7), h1 = 32, h2 = 16) {
      this.h1 = h1; this.h2 = h2;
      this.W1 = new Float64Array(h1 * FEATURE_COUNT);
      this.b1 = new Float64Array(h1);
      this.W2 = new Float64Array(h2 * h1);
      this.b2 = new Float64Array(h2);
      this.W3 = new Float64Array(h2);
      this.b3 = 0;
      const init = (arr, fanIn) => { const s = Math.sqrt(2 / fanIn); for (let i = 0; i < arr.length; i++) arr[i] = rng.normal() * s * 0.45; };
      init(this.W1, FEATURE_COUNT); init(this.W2, h1); init(this.W3, h2);
      this._adam = null; this.step = 0;
    }

    forward(x, cache = false) {
      const h1 = new Float64Array(this.h1);
      for (let j = 0; j < this.h1; j++) {
        let z = this.b1[j];
        const off = j * FEATURE_COUNT;
        for (let k = 0; k < FEATURE_COUNT; k++) z += this.W1[off + k] * x[k];
        h1[j] = Math.tanh(z);
      }
      const h2 = new Float64Array(this.h2);
      for (let j = 0; j < this.h2; j++) {
        let z = this.b2[j];
        const off = j * this.h1;
        for (let k = 0; k < this.h1; k++) z += this.W2[off + k] * h1[k];
        h2[j] = Math.tanh(z);
      }
      let z = this.b3;
      for (let j = 0; j < this.h2; j++) z += this.W3[j] * h2[j];
      const y = 1 / (1 + Math.exp(-clamp(z, -30, 30)));
      return cache ? { y, h1, h2, x } : y;
    }

    predictValue(instance, stats, i, prevValue) {
      const { lower, upper } = legalBounds(instance.n, i, prevValue);
      if (upper <= lower) return lower;
      const p = this.forward(makeFeatures(instance, stats, i, prevValue));
      return clamp(Math.round(p * MAX_VALUE), lower, upper);
    }

    solve(instance) {
      const stats = precomputeStats(instance);
      const x = new Int16Array(instance.n);
      let prev = MAX_VALUE + 1;
      for (let i = 0; i < instance.n; i++) {
        x[i] = this.predictValue(instance, stats, i, prev);
        prev = x[i];
      }
      return { x, cost: objective(instance, x), moves: instance.n };
    }

    initAdam() {
      const zeros = n => new Float64Array(n);
      this._adam = {
        W1m: zeros(this.W1.length), W1v: zeros(this.W1.length), b1m: zeros(this.b1.length), b1v: zeros(this.b1.length),
        W2m: zeros(this.W2.length), W2v: zeros(this.W2.length), b2m: zeros(this.b2.length), b2v: zeros(this.b2.length),
        W3m: zeros(this.W3.length), W3v: zeros(this.W3.length), b3m: 0, b3v: 0
      };
    }

    trainSamples(samples, opts = {}) {
      const lr = opts.lr ?? 0.003;
      const epochs = opts.epochs ?? 3;
      const batchSize = opts.batchSize ?? 64;
      const rng = opts.rng ?? new RNG(99);
      const onEpoch = opts.onEpoch ?? (() => {});
      if (!this._adam) this.initAdam();
      const idx = new Int32Array(samples.length); for (let i = 0; i < idx.length; i++) idx[i] = i;
      let lastLoss = 0;

      for (let epoch = 0; epoch < epochs; epoch++) {
        for (let i = idx.length - 1; i > 0; i--) { const j = rng.int(0, i); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
        let sumLoss = 0, seen = 0;
        for (let start = 0; start < idx.length; start += batchSize) {
          const end = Math.min(idx.length, start + batchSize);
          const gW1 = new Float64Array(this.W1.length), gb1 = new Float64Array(this.b1.length);
          const gW2 = new Float64Array(this.W2.length), gb2 = new Float64Array(this.b2.length);
          const gW3 = new Float64Array(this.W3.length); let gb3 = 0;

          for (let q = start; q < end; q++) {
            const s = samples[idx[q]];
            const c = this.forward(s.features, true);
            const err = c.y - s.label;
            sumLoss += err * err; seen++;
            let dz3 = 2 * err * c.y * (1 - c.y);
            dz3 = clamp(dz3, -5, 5);
            for (let j = 0; j < this.h2; j++) gW3[j] += dz3 * c.h2[j];
            gb3 += dz3;

            const dh2 = new Float64Array(this.h2);
            for (let j = 0; j < this.h2; j++) dh2[j] = dz3 * this.W3[j] * (1 - c.h2[j] * c.h2[j]);
            for (let j = 0; j < this.h2; j++) {
              const off = j * this.h1;
              for (let k = 0; k < this.h1; k++) gW2[off + k] += dh2[j] * c.h1[k];
              gb2[j] += dh2[j];
            }
            const dh1 = new Float64Array(this.h1);
            for (let k = 0; k < this.h1; k++) {
              let z = 0;
              for (let j = 0; j < this.h2; j++) z += this.W2[j * this.h1 + k] * dh2[j];
              dh1[k] = z * (1 - c.h1[k] * c.h1[k]);
            }
            for (let j = 0; j < this.h1; j++) {
              const off = j * FEATURE_COUNT;
              for (let k = 0; k < FEATURE_COUNT; k++) gW1[off + k] += dh1[j] * c.x[k];
              gb1[j] += dh1[j];
            }
          }
          const scale = 1 / (end - start);
          this._adamUpdate(this.W1, gW1, this._adam.W1m, this._adam.W1v, lr, scale);
          this._adamUpdate(this.b1, gb1, this._adam.b1m, this._adam.b1v, lr, scale);
          this._adamUpdate(this.W2, gW2, this._adam.W2m, this._adam.W2v, lr, scale);
          this._adamUpdate(this.b2, gb2, this._adam.b2m, this._adam.b2v, lr, scale);
          this._adamUpdate(this.W3, gW3, this._adam.W3m, this._adam.W3v, lr, scale);
          this.step++;
          const b1 = 0.9, b2 = 0.999, eps = 1e-8;
          this._adam.b3m = b1 * this._adam.b3m + (1 - b1) * gb3 * scale;
          this._adam.b3v = b2 * this._adam.b3v + (1 - b2) * (gb3 * scale) ** 2;
          const mh = this._adam.b3m / (1 - b1 ** this.step);
          const vh = this._adam.b3v / (1 - b2 ** this.step);
          this.b3 -= lr * mh / (Math.sqrt(vh) + eps);
        }
        lastLoss = sumLoss / Math.max(1, seen);
        onEpoch({ epoch: epoch + 1, epochs, loss: lastLoss });
      }
      return lastLoss;
    }

    _adamUpdate(param, grad, m, v, lr, scale) {
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      const t = this.step + 1;
      for (let i = 0; i < param.length; i++) {
        const g = clamp(grad[i] * scale, -5, 5);
        m[i] = b1 * m[i] + (1 - b1) * g;
        v[i] = b2 * v[i] + (1 - b2) * g * g;
        const mh = m[i] / (1 - b1 ** t);
        const vh = v[i] / (1 - b2 ** t);
        param[i] -= lr * mh / (Math.sqrt(vh) + eps);
      }
    }
  }

  function buildImitationDataset(count, nMin, nMax, rng = new RNG(101), onProgress = () => {}) {
    const samples = [];
    for (let e = 0; e < count; e++) {
      const n = rng.int(nMin, nMax);
      const instance = generateInstance(n, rng);
      const exact = exactSolve(instance);
      const stats = precomputeStats(instance);
      let prev = MAX_VALUE + 1;
      for (let i = 0; i < n; i++) {
        const label = exact.x[i] / MAX_VALUE;
        samples.push({ features: makeFeatures(instance, stats, i, prev), label: clamp(label, 0, 1) });
        prev = exact.x[i];
      }
      if ((e + 1) % 25 === 0 || e + 1 === count) onProgress({ done: e + 1, total: count, samples: samples.length });
    }
    return samples;
  }

  function randomFeasible(instance, rng) {
    const x = new Int16Array(instance.n);
    let prev = MAX_VALUE + 1;
    for (let i = 0; i < instance.n; i++) {
      const { lower, upper } = legalBounds(instance.n, i, prev);
      x[i] = rng.int(lower, upper);
      prev = x[i];
    }
    return x;
  }

  function saSolve(instance, opts = {}) {
    const rng = opts.rng ?? new RNG(23);
    const iterations = opts.iterations ?? instance.n * 100;
    const block = !!opts.block;
    const start = opts.start === 'random' ? randomFeasible(instance, rng) : greedySolve(instance).x;
    const x = new Int16Array(start);
    let cost = objective(instance, x);
    let bestCost = cost; const bestX = new Int16Array(x);
    let accepted = 0;
    const initialTemp = opts.initialTemp ?? Math.max(2, cost / Math.max(1, instance.n) * 0.08);
    const finalTemp = opts.finalTemp ?? 0.05;

    for (let it = 0; it < iterations; it++) {
      const frac = iterations <= 1 ? 1 : it / (iterations - 1);
      const T = initialTemp * Math.pow(finalTemp / initialTemp, frac);
      let delta = 0;
      if (!block) {
        const i = rng.int(0, instance.n - 1);
        const lo = i === instance.n - 1 ? 0 : x[i + 1] + 1;
        const hi = i === 0 ? MAX_VALUE : x[i - 1] - 1;
        if (hi < lo || hi === lo && x[i] === lo) continue;
        let nv;
        if (rng.next() < 0.8) {
          const step = Math.max(1, Math.round(Math.exp(rng.uniform(0, Math.log(12)))));
          nv = clamp(x[i] + (rng.next() < 0.5 ? -step : step), lo, hi);
        } else nv = rng.int(lo, hi);
        if (nv === x[i]) continue;
        const oldD = x[i] - instance.targets[i], newD = nv - instance.targets[i];
        delta = instance.weights[i] * (newD * newD - oldD * oldD);
        if (delta <= 0 || rng.next() < Math.exp(-delta / Math.max(T, 1e-9))) {
          cost += delta; x[i] = nv; accepted++;
        }
      } else {
        const l = rng.int(0, instance.n - 1);
        const maxLen = Math.min(instance.n - l, Math.max(2, Math.round(Math.sqrt(instance.n) * 2)));
        const r = Math.min(instance.n - 1, l + rng.int(0, maxLen - 1));
        const loShift = r === instance.n - 1 ? -x[r] : (x[r + 1] + 1 - x[r]);
        const hiShift = l === 0 ? MAX_VALUE - x[l] : (x[l - 1] - 1 - x[l]);
        if (loShift > hiShift || (loShift === 0 && hiShift === 0)) continue;
        const span = Math.min(16, Math.max(Math.abs(loShift), Math.abs(hiShift)));
        let shift = rng.int(-span, span);
        shift = clamp(shift, loShift, hiShift);
        if (shift === 0) continue;
        for (let i = l; i <= r; i++) {
          const oldD = x[i] - instance.targets[i], newD = x[i] + shift - instance.targets[i];
          delta += instance.weights[i] * (newD * newD - oldD * oldD);
        }
        if (delta <= 0 || rng.next() < Math.exp(-delta / Math.max(T, 1e-9))) {
          cost += delta;
          for (let i = l; i <= r; i++) x[i] += shift;
          accepted++;
        }
      }
      if (cost < bestCost) { bestCost = cost; bestX.set(x); }
    }
    return { x: bestX, cost: bestCost, moves: iterations, accepted };
  }

  function benchmark(policy, sizes, opts = {}) {
    const perSize = opts.perSize ?? 12;
    const saMultiplier = opts.saMultiplier ?? 100;
    const seed = opts.seed ?? 7001;
    const onProgress = opts.onProgress ?? (() => {});
    const rows = [];
    let done = 0, total = sizes.length * perSize;
    for (const n of sizes) {
      const acc = {
        n, count: 0,
        policyExcess: 0, greedyExcess: 0, saExcess: 0, blockSaExcess: 0,
        policyHit: 0, greedyHit: 0, saHit: 0, blockSaHit: 0,
        policyMoves: n, greedyMoves: n, saMoves: n * saMultiplier, blockSaMoves: n * saMultiplier
      };
      for (let k = 0; k < perSize; k++) {
        const rng = new RNG(seed + n * 10007 + k * 7919);
        const instance = generateInstance(n, rng);
        const opt = exactSolve(instance);
        const p = policy.solve(instance);
        const g = greedySolve(instance);
        const sa = saSolve(instance, { iterations: n * saMultiplier, rng: new RNG(seed + 11 + n * 97 + k), block: false });
        const bsa = saSolve(instance, { iterations: n * saMultiplier, rng: new RNG(seed + 29 + n * 193 + k), block: true });
        const denom = Math.max(1, n);
        acc.policyExcess += (p.cost - opt.cost) / denom;
        acc.greedyExcess += (g.cost - opt.cost) / denom;
        acc.saExcess += (sa.cost - opt.cost) / denom;
        acc.blockSaExcess += (bsa.cost - opt.cost) / denom;
        const tol = 1e-7;
        if (p.cost - opt.cost <= tol) acc.policyHit++;
        if (g.cost - opt.cost <= tol) acc.greedyHit++;
        if (sa.cost - opt.cost <= tol) acc.saHit++;
        if (bsa.cost - opt.cost <= tol) acc.blockSaHit++;
        acc.count++;
        done++; onProgress({ done, total, n });
      }
      for (const key of ['policyExcess','greedyExcess','saExcess','blockSaExcess']) acc[key] /= acc.count;
      rows.push(acc);
    }
    return rows;
  }

  function sampleComparison(policy, n, seed = 42, saMultiplier = 100) {
    const instance = generateInstance(n, new RNG(seed));
    const opt = exactSolve(instance);
    const policyRes = policy.solve(instance);
    const greedy = greedySolve(instance);
    const sa = saSolve(instance, { iterations: n * saMultiplier, rng: new RNG(seed + 1), block: false });
    const blockSa = saSolve(instance, { iterations: n * saMultiplier, rng: new RNG(seed + 2), block: true });
    return { instance, opt, policy: policyRes, greedy, sa, blockSa };
  }

  return {
    MAX_VALUE, MAX_N, FEATURE_COUNT, RNG, generateInstance, objective, isFeasible,
    exactSolve, precomputeStats, legalBounds, makeFeatures, greedySolve, MLPPolicy,
    buildImitationDataset, saSolve, benchmark, sampleComparison
  };
});
