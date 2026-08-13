// Select a TensorFlow.js backend before the absolute-utility successor-state critic + oracle control is evaluated.
// Android Chrome uses CPU directly for reliability; other platforms prefer WASM.
async function boot() {
  const statusEl = document.getElementById('status');
  try {
    if (!window.tf) throw new Error('TensorFlow.js did not load.');

    const ua = navigator.userAgent || '';
    const isAndroid = /Android/i.test(ua);
    let backend = 'cpu';

    if (isAndroid) {
      await tf.setBackend('cpu');
      await tf.ready();
      backend = tf.getBackend();
    } else {
      if (tf.wasm?.setWasmPaths) {
        tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/');
      }
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
    }

    const platformNote = isAndroid ? ' · Android reliability mode' : '';
    if (statusEl) statusEl.textContent = `TensorFlow.js ready · backend: ${backend}${platformNote}. Loading absolute critic + oracle steps-to-go…`;
    await import('./app-gnn-oracle-steps.js?v=20260813-40');
  } catch (err) {
    console.error(err);
    if (statusEl) statusEl.textContent = `Error initializing TensorFlow.js backend: ${err.message}`;
  }
}

boot();
