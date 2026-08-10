// Force a non-WebGL TensorFlow.js backend before the PPO engine is evaluated.
// WASM is preferred for browser performance; CPU is the reliability fallback.
async function boot() {
  const statusEl = document.getElementById('status');
  try {
    if (!window.tf) throw new Error('TensorFlow.js did not load.');

    if (tf.wasm?.setWasmPaths) {
      tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/');
    }

    let backend = 'cpu';
    try {
      const ok = await tf.setBackend('wasm');
      if (!ok) throw new Error('WASM backend was not accepted.');
      await tf.ready();
      backend = tf.getBackend();
    } catch (wasmError) {
      console.warn('TensorFlow.js WASM backend unavailable; falling back to CPU.', wasmError);
      await tf.setBackend('cpu');
      await tf.ready();
      backend = tf.getBackend();
    }

    if (statusEl) statusEl.textContent = `TensorFlow.js ready · backend: ${backend}. Loading PPO GNN…`;
    await import('./app-gnn-ppo.js?v=20260810-19');
  } catch (err) {
    console.error(err);
    if (statusEl) statusEl.textContent = `Error initializing TensorFlow.js backend: ${err.message}`;
  }
}

boot();
