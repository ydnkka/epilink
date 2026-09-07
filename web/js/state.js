export const MAXIMUM_HIDDEN_DEPTH = 5;

export const DEFAULT_STATE = Object.freeze({
  adDepth: 0,
  caDepthA: 0,
  caDepthB: 0,
  parameterKey: "incubation_shape",
  sensitivityLevel: 2,
  mutationProcess: "stochastic",
  observedTime: 3,
  observedGenetic: 2,
});

export function constrainScenario(state) {
  state.adDepth = clamp(Math.round(state.adDepth), 0, MAXIMUM_HIDDEN_DEPTH);
  state.caDepthA = clamp(Math.round(state.caDepthA), 0, MAXIMUM_HIDDEN_DEPTH);
  state.caDepthB = clamp(Math.round(state.caDepthB), 0, MAXIMUM_HIDDEN_DEPTH - state.caDepthA);
}

export function selectedDefinition(data, state) {
  return data.parameter_catalog.find((definition) => definition.key === state.parameterKey);
}

export function configurationKey(data, state) {
  const definition = selectedDefinition(data, state);
  return state.sensitivityLevel === definition.default_index
    ? "baseline"
    : `${definition.key}:${state.sensitivityLevel}`;
}

export function currentParameters(data, state) {
  const parameters = { ...data.baseline_parameters };
  const definition = selectedDefinition(data, state);
  parameters[definition.key] = definition.values[state.sensitivityLevel];
  return parameters;
}

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
