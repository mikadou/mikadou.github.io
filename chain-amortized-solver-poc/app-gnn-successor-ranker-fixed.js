// Bootstrap around the ranker wrapper: fix blob-relative URLs and sanitize source-patch template boundaries.
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

  // The wrapper builds source code with String.raw blocks. Remove nested template literals from
  // generated diagnostic expressions so the wrapper itself remains valid JavaScript.
  src = src.replace(
    "oracleVariable=\\${n<80?`\\${Math.round(100*oracleVarSolved/trials)}%`:'skip'}",
    "oracleVariable=\\${n<80?(Math.round(100*oracleVarSolved/trials)+'%'):'skip'}"
  );
  src = src.replace(
    "rows.slice(0,10).map(x=>`\\${x.v}:pred\\${x.pred.toFixed(1)}/true\\${x.truth.toFixed(1)}`).join(',')",
    "rows.slice(0,10).map(x=>String(x.v)+':pred'+x.pred.toFixed(1)+'/true'+x.truth.toFixed(1)).join(',')"
  );
  src = src.replace(
    "\\${state.assignedCount===n?` viol=\\${violatedConstraintCount(state.values)} energy=\\${violationEnergy(state.values)}`:''}",
    "\\${state.assignedCount===n?(' viol='+violatedConstraintCount(state.values)+' energy='+violationEnergy(state.values)):''}"
  );
  src = src.replace(
    "\\${state.assignedCount===n?` violations=\\${violatedConstraintCount(state.values)} energy=\\${violationEnergy(state.values)}`:''}",
    "\\${state.assignedCount===n?(' violations='+violatedConstraintCount(state.values)+' energy='+violationEnergy(state.values)):''}"
  );

  const blobUrl = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
  try { await import(blobUrl); }
  finally { URL.revokeObjectURL(blobUrl); }
}

bootFixedRanker().catch(err => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `Error loading fixed successor ranker: ${err.message}`;
});
