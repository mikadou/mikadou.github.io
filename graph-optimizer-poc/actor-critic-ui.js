function replaceButtonListener(key, handler) {
  const current = ui[key];
  const replacement = current.cloneNode(true);
  current.replaceWith(replacement);
  ui[key] = replacement;
  replacement.addEventListener('click', handler);
}

const resetBaseModel = resetModel;

function resetActorCriticModel() {
  resetBaseModel();
  ui.backendNotice.textContent =
    `TensorFlow.js ready · backend: ${tf.getBackend()} · direct actor–critic training`;
  ui.policyStateBadge.textContent = 'Untrained actor–critic';
  ui.policyStateBadge.classList.add('muted');
  ui.statusText.textContent =
    'The actor starts with uniform node probabilities and a broad value distribution; the critic starts at zero.';
}

const createProblemBase = createCurrentProblem;

function createActorCriticProblem() {
  createProblemBase();
  setStatus(
    `New N=${state.currentProblem.n} problem created. Actor–critic search and random SA share this initial assignment.`
  );
  ui.benchmarkSummary.className = 'benchmark-empty';
  ui.benchmarkSummary.textContent =
    'Train the actor–critic, compare once, or benchmark 30 paired instances.';
}

resetModel = resetActorCriticModel;
createCurrentProblem = createActorCriticProblem;

replaceButtonListener('newProblemBtn', createActorCriticProblem);
replaceButtonListener('trainBtn', trainActorCriticPolicy);
replaceButtonListener('compareBtn', runActorCriticComparison);
replaceButtonListener('benchmarkBtn', runActorCriticBenchmark);
replaceButtonListener('resetModelBtn', resetActorCriticModel);

const currentTestN = ui.testN;
const replacementTestN = currentTestN.cloneNode(true);
currentTestN.replaceWith(replacementTestN);
ui.testN = replacementTestN;
replacementTestN.addEventListener('change', createActorCriticProblem);

ui.policyStateBadge.textContent = state.trainedEpisodes > 0
  ? `${state.trainedEpisodes} actor–critic episodes`
  : 'Untrained actor–critic';
ui.backendNotice.textContent = tf && tf.getBackend
  ? `TensorFlow.js ready · backend: ${tf.getBackend()} · direct actor–critic training`
  : ui.backendNotice.textContent;
if (state.trainedEpisodes === 0) {
  ui.statusText.textContent =
    'The actor directly samples node/value proposals and the zero critic gives a hill-climb bootstrap.';
}
