// Small bootstrap around the ranker wrapper: fix its base URL for blob import and one source patch boundary.
async function bootFixedRanker() {
  const rankerUrl = new URL('./app-gnn-successor-ranker.js?v=20260813-42-source', import.meta.url);
  const response = await fetch(rankerUrl, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Failed to load successor ranker wrapper: HTTP ${response.status}`);
  let src = await response.text();

  const relativeBase = "  const baseUrl = new URL('./app-gnn-successor-critic.js?v=20260813-37-base', import.meta.url);";
  if (!src.includes(relativeBase)) throw new Error('Ranker bootstrap failed: base URL marker');
  const absoluteBase = new URL('./app-gnn-successor-critic.js?v=20260813-37-base', rankerUrl).href;
  src = src.replace(relativeBase, '  const baseUrl = new URL(' + JSON.stringify(absoluteBase) + ');');

  const badBoundary = "  replaceBetween('async function runBenchmark(){', '\\nfunction appendProbe(', benchmarkCode + '\\nfunction appendProbe(', 'ranker benchmark');";
  const goodBoundary = "  replaceBetween('async function runBenchmark(){', '\\nfunction appendProbe(', benchmarkCode, 'ranker benchmark');";
  if (!src.includes(badBoundary)) throw new Error('Ranker bootstrap failed: benchmark boundary marker');
  src = src.replace(badBoundary, goodBoundary);

  const blobUrl = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
  try { await import(blobUrl); }
  finally { URL.revokeObjectURL(blobUrl); }
}

bootFixedRanker().catch(err => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `Error loading fixed successor ranker: ${err.message}`;
});
