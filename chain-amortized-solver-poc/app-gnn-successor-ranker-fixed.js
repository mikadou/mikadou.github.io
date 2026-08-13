// Bootstrap around the successor steps-to-go ranker wrapper.
// It converts generated-code String.raw blocks into normal template literals so escaped
// backticks/interpolations become valid JavaScript, sanitizes nested templates, and validates
// the exact generated module before import.
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

  // These are template literals nested inside generated-code template literals. Replace them
  // with concatenation so the wrapper itself has only one template-literal level.
  const replacements = [
    [
      "oracleVariable=\\${n<80?`\\${Math.round(100*oracleVarSolved/trials)}%`:'skip'}",
      "oracleVariable=\\${n<80?(Math.round(100*oracleVarSolved/trials)+'%'):'skip'}"
    ],
    [
      "rows.slice(0,10).map(x=>`\\${x.v}:pred\\${x.pred.toFixed(1)}/true\\${x.truth.toFixed(1)}`).join(',')",
      "rows.slice(0,10).map(x=>String(x.v)+':pred'+x.pred.toFixed(1)+'/true'+x.truth.toFixed(1)).join(',')"
    ],
    [
      "\\${state.assignedCount===n?` viol=\\${violatedConstraintCount(state.values)} energy=\\${violationEnergy(state.values)}`:''}",
      "\\${state.assignedCount===n?(' viol='+violatedConstraintCount(state.values)+' energy='+violationEnergy(state.values)):''}"
    ],
    [
      "\\${state.assignedCount===n?` violations=\\${violatedConstraintCount(state.values)} energy=\\${violationEnergy(state.values)}`:''}",
      "\\${state.assignedCount===n?(' violations='+violatedConstraintCount(state.values)+' energy='+violationEnergy(state.values)):''}"
    ]
  ];
  for (const [from, to] of replacements) {
    if (!src.includes(from)) throw new Error('Ranker bootstrap failed: nested-template marker');
    src = src.replace(from, to);
  }

  // Critical fix: String.raw preserves the backslashes in \` and \${...}. Those blocks are
  // intended to generate executable JavaScript, so use ordinary template literals instead.
  const rawCount = (src.match(/String\.raw`/g) || []).length;
  if (rawCount < 3) throw new Error(`Ranker bootstrap failed: expected generated-code blocks, found ${rawCount}`);
  src = src.replace(/String\.raw`/g, '`');

  // Make the wrapper validate the exact generated source before importing the Blob module.
  const blobMarker = "  const blobUrl = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));";
  if (!src.includes(blobMarker)) throw new Error('Ranker bootstrap failed: generated-module marker');
  src = src.replace(
    blobMarker,
    "  try { new Function(src); } catch (syntaxError) { throw new Error('Generated ranker syntax error: ' + syntaxError.message); }\n" + blobMarker
  );

  // Validate the transformed wrapper itself before module import.
  // import.meta is module-only, so replace it only for this parse check.
  const parseCheck = src.replaceAll('import.meta.url', JSON.stringify(import.meta.url));
  try { new Function(parseCheck); }
  catch (syntaxError) { throw new Error('Ranker wrapper syntax error after bootstrap fixes: ' + syntaxError.message); }

  const blobUrl = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
  try { await import(blobUrl); }
  finally { URL.revokeObjectURL(blobUrl); }
}

bootFixedRanker().catch(err => {
  console.error(err);
  const statusEl = document.getElementById('status');
  if (statusEl) statusEl.textContent = `Error loading successor steps ranker: ${err.message}`;
});
