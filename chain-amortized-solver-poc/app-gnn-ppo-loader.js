// Select a TensorFlow.js backend before loading the neighbor-based imitation experiment.
async function boot() {
  const statusEl = document.getElementById('status');
  try {
    if (!window.tf) throw new Error('TensorFlow.js did not load.');
    const isAndroid = /Android/i.test(navigator.userAgent || '');
    let backend = 'cpu';
    if (isAndroid) {
      await tf.setBackend('cpu');
      await tf.ready();
      backend = tf.getBackend();
    } else {
      if (tf.wasm?.setWasmPaths) tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/');
      try {
        const ok = await tf.setBackend('wasm');
        if (!ok) throw new Error('WASM backend was not accepted.');
        await tf.ready();
        backend = tf.getBackend();
      } catch (err) {
        console.warn('WASM unavailable; using CPU.', err);
        await tf.setBackend('cpu');
        await tf.ready();
        backend = tf.getBackend();
      }
    }
    if (statusEl) statusEl.textContent = `TensorFlow.js ready · backend: ${backend}. Loading neighbor-based imitation solver…`;
    await import('./app-gnn-neighbor-imitation.js?v=20260813-47');
  } catch (err) {
    console.error(err);
    if (statusEl) statusEl.textContent = `Error initializing neighbor-based imitation solver: ${err.message}`;
  }
}
boot();
