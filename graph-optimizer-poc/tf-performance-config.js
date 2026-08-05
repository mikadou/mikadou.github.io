'use strict';

// Declare the critic-influence hook before later strict-mode modules replace it.
// The local actor module assigns the concrete implementation at load time.
var performanceCriticInfluence;

// Disable development-only TensorFlow.js checks before the first model is
// created. The default browser backend remains WebGL when it is available.
tf.enableProdMode();
