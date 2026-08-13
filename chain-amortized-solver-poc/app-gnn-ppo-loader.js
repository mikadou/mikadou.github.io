// Select a TensorFlow.js backend, then load the simple imitation solver and its REINFORCE extension
// as classic scripts so the extension can reuse the exact same policy functions/parameters.
function loadClassicScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = false;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.head.appendChild(script);
  });
}

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
    if (statusEl) statusEl.textContent = `TensorFlow.js ready · backend: ${backend}${platformNote}. Loading imitation + REINFORCE solver…`;
    await loadClassicScript('./app-gnn-simple-imitation.js?v=20260813-45-base');
    await loadClassicScript('./app-gnn-simple-rl-extension.js?v=20260813-46');
  } catch (err) {
    console.error(err);
    if (statusEl) statusEl.textContent = `Error initializing imitation + RL solver: ${err.message}`;
  }
}

boot();
