import { drawSurface, renderHistogram, renderTree, surfacePosition } from "./charts.js";
import {
  DEFAULT_STATE,
  MAXIMUM_HIDDEN_DEPTH,
  configurationKey,
  constrainScenario,
  selectedDefinition,
} from "./state.js";

const state = {
  ...DEFAULT_STATE,
  modelScenario: "ad(0)",
  targets: new Set(["ad(0)"]),
};

const byId = (id) => document.getElementById(id);
let explorerData = null;

const PARAMETER_COPY = {
  incubation_shape: {
    label: "Variation in time to symptoms (shape)",
    description: "Changes the shape and spread of the infection-to-symptom timing distribution.",
  },
  incubation_scale: {
    label: "Time to symptoms (scale)",
    description: "Stretches or contracts the modelled time from infection to symptoms.",
  },
  latent_shape: {
    label: "Time before infectiousness (shape)",
    description: "Changes how the time to symptoms is divided between non-infectious and infectious stages.",
  },
  symptomatic_rate: {
    label: "End of infectiousness after symptoms",
    description: "Higher values shorten the expected infectious period after symptoms begin.",
  },
  symptomatic_shape: {
    label: "Variation after symptoms (shape)",
    description: "Changes variation in the infectious period after symptoms begin.",
  },
  transmission_rate_ratio: {
    label: "Infectiousness before versus after symptoms",
    description: "Changes the relative level of transmission before symptoms begin.",
  },
  testing_delay_shape: {
    label: "Variation in delay to testing (shape)",
    description: "Changes the shape and spread of the delay from symptoms to sampling.",
  },
  testing_delay_scale: {
    label: "Delay from symptoms to testing (scale)",
    description: "Stretches or contracts the modelled delay from symptoms to sampling.",
  },
  substitution_rate: {
    label: "Genome change rate",
    description: "Changes how quickly genome differences are expected to accumulate.",
  },
  relaxation: {
    label: "Variation in genome change rate",
    description: "Changes how much the genome change rate can vary between transmission routes.",
  },
  genome_length: {
    label: "Genome length",
    description: "Changes how many genome positions are available to accumulate differences.",
  },
};

function format(value, decimals = 2) {
  return Number(value).toFixed(decimals);
}

function plural(count, singular, pluralForm = `${singular}s`) {
  return Number(count) === 1 ? singular : pluralForm;
}

function hiddenPeopleText(count) {
  return count === 0 ? "None" : `${count} ${plural(count, "person", "people")}`;
}

function timeObservationText(value) {
  if (value === 0) return "Samples taken the same day";
  return `B is ${format(Math.abs(value), 0)} days ${value > 0 ? "later" : "earlier"}`;
}

function geneticObservationText(value) {
  return value === 0 ? "No differences" : `${value} ${plural(value, "difference")}`;
}

function scenarioParts(label) {
  if (label.startsWith("ad(")) {
    return { kind: "ad", a: Number(label.slice(3, -1)) };
  }
  const [a, b] = label.slice(3, -1).split(",").map(Number);
  return { kind: "ca", a, b };
}

function scenarioPlainLabel(label) {
  const scenario = scenarioParts(label);
  if (scenario.kind === "ad") {
    if (scenario.a === 0) return "A may have infected B directly";
    return `A may have led to B through ${scenario.a} unsampled ${plural(scenario.a, "person", "people")}`;
  }
  if (scenario.a === 0 && scenario.b === 0) {
    return "A and B may share a source, with no one else on either route";
  }
  return `A and B may share a source (${scenario.a} missing toward A; ${scenario.b} toward B)`;
}

function scenarioOptionLabel(label) {
  const scenario = scenarioParts(label);
  if (scenario.kind === "ad") {
    return scenario.a === 0
      ? `Direct from A to B — ${label}`
      : `${scenario.a} missing between A and B — ${label}`;
  }
  if (scenario.a === 0 && scenario.b === 0) return `Shared source, simple routes — ${label}`;
  return `Shared source; ${scenario.a} toward A, ${scenario.b} toward B — ${label}`;
}

function scenarioChipLabel(label) {
  const scenario = scenarioParts(label);
  if (scenario.kind === "ad") {
    return scenario.a === 0 ? "Direct A to B" : `${scenario.a} missing between A and B`;
  }
  if (scenario.a === 0 && scenario.b === 0) return "No missing people on either route";
  return `${scenario.a} toward A · ${scenario.b} toward B`;
}

function groupDisplay(group) {
  return {
    "Natural history": "Infection timing",
    Testing: "Testing timing",
    Clock: "Genome change",
  }[group] || group;
}

function parameterLabel(definition) {
  return PARAMETER_COPY[definition.key]?.label || definition.label;
}

function parameterDescription(definition) {
  return PARAMETER_COPY[definition.key]?.description || definition.description;
}

function parameterDisplay(definition, value) {
  if (definition.key === "substitution_rate") return `${format(value, 5)} per site per year`;
  if (definition.key === "genome_length") return `${Number(value).toLocaleString()} positions`;
  if (definition.unit === "days") return `${format(value, 2)} days`;
  if (definition.unit === "per day") return `${format(value, 2)} per day`;
  if (definition.key === "transmission_rate_ratio") return `${format(value, 2)}× the after-symptom rate`;
  return format(value, 2);
}

function activeModel() {
  return explorerData.models[configurationKey(explorerData, state)];
}

function comparisonModel() {
  return explorerData.models.baseline;
}

function renderScenarioExplorer() {
  constrainScenario(state);
  byId("adDepth").value = state.adDepth;
  byId("caDepthA").value = state.caDepthA;
  byId("caDepthB").max = MAXIMUM_HIDDEN_DEPTH - state.caDepthA;
  byId("caDepthB").value = state.caDepthB;
  byId("adDepthOut").value = hiddenPeopleText(state.adDepth);
  byId("caDepthAOut").value = hiddenPeopleText(state.caDepthA);
  byId("caDepthBOut").value = hiddenPeopleText(state.caDepthB);

  const ancestorDescendant = { kind: "ad", a: state.adDepth, label: `ad(${state.adDepth})` };
  const commonAncestor = {
    kind: "ca",
    a: state.caDepthA,
    b: state.caDepthB,
    label: `ca(${state.caDepthA},${state.caDepthB})`,
  };

  byId("adToken").textContent = ancestorDescendant.label;
  byId("caToken").textContent = commonAncestor.label;
  byId("adTreeCaption").textContent = renderTree(byId("adTree"), ancestorDescendant);
  byId("caTreeCaption").textContent = renderTree(byId("caTree"), commonAncestor);
  byId("adTreeDesc").textContent = `${scenarioPlainLabel(ancestorDescendant.label)}. Dots represent unsampled people.`;
  byId("caTreeDesc").textContent = `${scenarioPlainLabel(commonAncestor.label)}. The gold dot is the shared unsampled source.`;
}

function populateModelScenarioOptions() {
  const selector = byId("modelScenario");
  selector.replaceChildren();
  const oneRoute = document.createElement("optgroup");
  const sharedSource = document.createElement("optgroup");
  oneRoute.label = "A may have led to B";
  sharedSource.label = "A and B may share a source";

  explorerData.target_labels.forEach((label) => {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = scenarioOptionLabel(label);
    (label.startsWith("ad(") ? oneRoute : sharedSource).append(option);
  });

  selector.append(oneRoute, sharedSource);
  selector.value = state.modelScenario;
}

function updateAssumptionsPanel() {
  const definition = selectedDefinition(explorerData, state);
  const slider = byId("sensitivityLevel");
  const parameterFocus = byId("parameterFocus");
  slider.max = definition.values.length - 1;
  slider.value = state.sensitivityLevel;
  parameterFocus.replaceChildren();

  const groups = new Map();
  explorerData.parameter_catalog.forEach((item) => {
    if (!groups.has(item.group)) {
      const optionGroup = document.createElement("optgroup");
      optionGroup.label = groupDisplay(item.group);
      groups.set(item.group, optionGroup);
      parameterFocus.append(optionGroup);
    }
    const option = document.createElement("option");
    option.value = item.key;
    option.textContent = parameterLabel(item);
    groups.get(item.group).append(option);
  });

  parameterFocus.value = definition.key;
  byId("parameterGroupOut").value = groupDisplay(definition.group);
  byId("sensitivityLabel").textContent = parameterLabel(definition);
  byId("sensitivityValueOut").value = parameterDisplay(
    definition,
    definition.values[state.sensitivityLevel]
  );

  const baseline = state.sensitivityLevel === definition.default_index;
  byId("sensitivityDescription").textContent = baseline
    ? `Standard setting. ${parameterDescription(definition)}`
    : `Changed from the standard setting; every other assumption stays fixed. ${parameterDescription(definition)}`;

  const mode = state.mutationProcess === "stochastic"
    ? "stochastic random counts"
    : "deterministic expected counts";
  byId("clockSetting").textContent = definition.group === "Clock"
    ? `You are changing ${parameterLabel(definition).toLowerCase()}. Only the genome-difference chart should respond directly. It currently shows ${mode}.`
    : `Changing infection or testing timing can affect both when samples are taken and how much time genomes have to differ. The genome chart currently shows ${mode}.`;
}

function updateMutationModeCopy() {
  const stochastic = state.mutationProcess === "stochastic";
  byId("stochastic").checked = !stochastic;
  byId("mutationToggleHelp").textContent = stochastic
    ? "Off — stochastic variation in mutation counts is included."
    : "On — random variation is removed; deterministic expected counts are shown.";
  byId("geneticMode").textContent = stochastic
    ? "Stochastic variation"
    : "Expected count only";
  byId("geneticChartDescription").textContent = stochastic
    ? "Stochastic genome differences, including random variation in mutation counts."
    : "Deterministic expected differences between the two virus genome sequences.";
  byId("clockRateContext").textContent = stochastic
    ? "stochastic · random mutation counts included"
    : "deterministic · expected count only";
  byId("comparisonModeCopy").textContent = stochastic
    ? "Change the two observed values, choose the stories to compare, and see where the pair falls. This section uses the standard settings with stochastic genome-change variation included."
    : "Change the two observed values, choose the stories to compare, and see where the pair falls. This section uses the standard settings with deterministic expected genome changes.";
}

function selectedScenarioSurface(model, label, mutationProcess) {
  return {
    ...explorerData.surface,
    scores: model.surface_scores[mutationProcess][label],
  };
}

function combineTargetSurface(model, mutationProcess) {
  const scores = explorerData.surface.genetic_values.map(() =>
    explorerData.surface.time_values.map(() => 0)
  );
  state.targets.forEach((label) => {
    const scenarioSurface = model.surface_scores[mutationProcess][label];
    scores.forEach((row, rowIndex) => row.forEach((_, columnIndex) => {
      row[columnIndex] += scenarioSurface[rowIndex][columnIndex];
    }));
  });
  return { ...explorerData.surface, scores };
}

function closestIndex(values, target) {
  return values.reduce(
    (best, value, index) => Math.abs(value - target) < Math.abs(values[best] - target) ? index : best,
    0
  );
}

function scoreAtCurrentObservation(surface) {
  const geneticRow = closestIndex(surface.genetic_values, state.observedGenetic);
  const timeColumn = closestIndex(surface.time_values, state.observedTime);
  return surface.scores[geneticRow][timeColumn];
}

function quantile(summary, probability) {
  const index = summary.ecdf.probabilities.findIndex((value) => value >= probability);
  return summary.ecdf.values[index < 0 ? summary.ecdf.values.length - 1 : index];
}

function distributionValue(value, mode) {
  if (mode === "time") return `${format(value, 0)} days`;
  const rounded = Math.round(value);
  return `${rounded} ${plural(rounded, "difference")}`;
}

function renderDistributionSummary(element, summary, observed, mode) {
  const low = quantile(summary, .1);
  const high = quantile(summary, .9);
  let position = "inside the middle range";
  if (observed < low) position = "below the middle range";
  if (observed > high) position = "above the middle range";

  const rangeItem = document.createElement("div");
  const rangeLabel = document.createElement("span");
  const rangeValue = document.createElement("strong");
  rangeItem.className = "summary-item";
  rangeLabel.textContent = "Middle 80% of simulations";
  rangeValue.textContent = `${distributionValue(low, mode)} to ${distributionValue(high, mode)}`;
  rangeItem.append(rangeLabel, rangeValue);

  const observedItem = document.createElement("div");
  const observedLabel = document.createElement("span");
  const observedValue = document.createElement("strong");
  observedItem.className = "summary-item observation";
  observedLabel.textContent = "Current example";
  observedValue.textContent = `${distributionValue(observed, mode)} · ${position}`;
  observedItem.append(observedLabel, observedValue);
  element.replaceChildren(rangeItem, observedItem);

  return { low, high, position };
}

function scoreInterpretation(ratio) {
  if (ratio >= .67) {
    return "On average, the observed values sit relatively near the centres of the selected model distributions.";
  }
  if (ratio >= .33) {
    return "On average, the observed values have mixed compatibility with the selected model distributions.";
  }
  return "On average, one or both observed values sit toward the outer parts of the selected model distributions.";
}

function renderScore(combinedSurface) {
  const scenarioScores = [...state.targets].map((label) => ({
    label,
    score: scoreAtCurrentObservation(
      selectedScenarioSurface(comparisonModel(), label, state.mutationProcess)
    ),
  }));
  const overallScore = scoreAtCurrentObservation(combinedSurface);
  const maximum = scenarioScores.length;
  const ratio = Math.max(0, Math.min(1, overallScore / maximum));

  byId("overallScore").textContent = `${format(overallScore, 2)} / ${maximum}`;
  byId("overallScore").setAttribute(
    "aria-label",
    `${format(overallScore, 2)} out of a maximum ${maximum}`
  );
  byId("overallScoreLabel").textContent = `${Math.round(ratio * 100)}% of the maximum available score—not a probability`;
  byId("overallMeter").style.width = `${ratio * 100}%`;
  byId("overallScoreContext").textContent = scoreInterpretation(ratio);

  const breakdown = byId("scoreBreakdown");
  breakdown.replaceChildren();
  scenarioScores.forEach(({ label, score }) => {
    const row = document.createElement("div");
    const name = document.createElement("div");
    const scenarioCode = document.createElement("code");
    const plainName = document.createElement("span");
    const scoreValue = document.createElement("span");
    const meter = document.createElement("div");
    const meterFill = document.createElement("span");
    row.className = "breakdown-row";
    name.className = "breakdown-name";
    scoreValue.className = "breakdown-score";
    meter.className = "breakdown-meter";
    scenarioCode.textContent = label;
    plainName.textContent = scenarioChipLabel(label);
    scoreValue.textContent = `${format(score, 2)} / 1`;
    meterFill.style.width = `${Math.max(0, Math.min(1, score)) * 100}%`;
    name.append(scenarioCode, plainName);
    meter.append(meterFill);
    row.append(name, scoreValue, meter);
    breakdown.append(row);
  });

  const insightCopy = byId("insight").lastElementChild;
  const lead = document.createElement("strong");
  lead.textContent = `${format(overallScore, 2)} out of ${maximum} for the selected ${plural(maximum, "story")}. `;
  insightCopy.replaceChildren(lead, document.createTextNode(
    `${scoreInterpretation(ratio)} This describes fit to the model, not the chance that transmission occurred.`
  ));
}

function buildTargetGroup(title, labels) {
  const group = document.createElement("section");
  const heading = document.createElement("h4");
  const list = document.createElement("div");
  group.className = "target-group";
  list.className = "target-group-list";
  heading.textContent = title;

  labels.forEach((label) => {
    const chip = document.createElement("button");
    const code = document.createElement("code");
    const description = document.createElement("span");
    chip.type = "button";
    chip.className = `target-chip${state.targets.has(label) ? " selected" : ""}`;
    chip.setAttribute("aria-pressed", String(state.targets.has(label)));
    chip.setAttribute("aria-describedby", "selectedScenarioCount");
    chip.title = state.targets.has(label) ? "Remove this story" : "Add this story";
    code.textContent = label;
    description.textContent = scenarioChipLabel(label);
    chip.append(code, description);
    chip.addEventListener("click", () => {
      if (state.targets.has(label) && state.targets.size === 1) {
        byId("selectedScenarioCount").textContent = "Keep at least one story selected.";
        return;
      }
      if (state.targets.has(label)) state.targets.delete(label);
      else state.targets.add(label);
      renderTargetChips();
      render();
    });
    list.append(chip);
  });

  group.append(heading, list);
  return group;
}

function renderTargetChips() {
  const grid = byId("targetGrid");
  const adLabels = explorerData.target_labels.filter((label) => label.startsWith("ad("));
  const caLabels = explorerData.target_labels.filter((label) => label.startsWith("ca("));
  grid.replaceChildren(
    buildTargetGroup("A may have led to B", adLabels),
    buildTargetGroup("A and B may share a source", caLabels)
  );
  const count = state.targets.size;
  byId("selectedScenarioCount").textContent = `${count} ${plural(count, "story")} selected · maximum combined score ${count}`;
}

function setTargetPreset(labels) {
  state.targets = new Set(labels);
  renderTargetChips();
  render();
}

function useScenarioInModel(label) {
  state.modelScenario = label;
  byId("modelScenario").value = label;
  render();
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  byId("model").scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
}

function render() {
  renderScenarioExplorer();
  updateAssumptionsPanel();
  updateMutationModeCopy();
  byId("modelScenarioPlain").textContent = scenarioPlainLabel(state.modelScenario);
  byId("observedTimeOut").textContent = timeObservationText(state.observedTime);
  byId("observedGeneticOut").textContent = geneticObservationText(state.observedGenetic);
  byId("exampleMarkerCopy").textContent = `The blue bars show how often each whole-day or whole-mutation value occurs in the model. The pink line marks the current example: ${timeObservationText(state.observedTime)}, with ${geneticObservationText(state.observedGenetic).toLowerCase()}.`;

  const expectedPatternsModel = activeModel();
  const storedDistribution = expectedPatternsModel.scenario_distributions[state.modelScenario];
  const distribution = {
    ...storedDistribution,
    genetic: storedDistribution.genetic[state.mutationProcess],
  };
  renderHistogram(byId("timeChart"), distribution.time, state.observedTime, "days between samples", "time");
  renderHistogram(byId("geneticChart"), distribution.genetic, state.observedGenetic, "genome differences", "genetic");
  const timeSummary = renderDistributionSummary(byId("timeSummary"), distribution.time, state.observedTime, "time");
  const geneticSummary = renderDistributionSummary(byId("geneticSummary"), distribution.genetic, state.observedGenetic, "genetic");
  byId("timeChart").setAttribute("aria-label", `Simulated days between samples. The middle 80 percent runs from ${distributionValue(timeSummary.low, "time")} to ${distributionValue(timeSummary.high, "time")}. The current example is ${timeSummary.position}.`);
  const geneticMode = state.mutationProcess === "stochastic"
    ? "Stochastic mutation variation is included."
    : "Deterministic expected counts are shown.";
  byId("geneticChart").setAttribute("aria-label", `Simulated genome differences. ${geneticMode} The middle 80 percent runs from ${distributionValue(geneticSummary.low, "genetic")} to ${distributionValue(geneticSummary.high, "genetic")}. The current example is ${geneticSummary.position}.`);

  const combinedSurface = combineTargetSurface(comparisonModel(), state.mutationProcess);
  drawSurface(byId("surface"), combinedSurface, state.targets.size, state.observedTime, state.observedGenetic);
  byId("surfaceLow").textContent = "0 · less central";
  byId("surfaceHigh").textContent = `${state.targets.size} · maximum`;
  byId("surface").setAttribute("aria-label", `Compatibility map for ${state.targets.size} selected ${plural(state.targets.size, "story")}. Current pair: ${timeObservationText(state.observedTime)}, ${geneticObservationText(state.observedGenetic)}.`);
  renderScore(combinedSurface);

  byId("clockRate").textContent = `${format(expectedPatternsModel.derived.daily_clock_rate, 3)} genome changes per day`;
}

function bindEvents() {
  byId("parameterFocus").addEventListener("change", (event) => {
    state.parameterKey = event.target.value;
    state.sensitivityLevel = selectedDefinition(explorerData, state).default_index;
    render();
  });
  byId("sensitivityLevel").addEventListener("input", (event) => {
    state.sensitivityLevel = Number(event.target.value);
    render();
  });
  [["adDepth", "adDepth"], ["caDepthA", "caDepthA"], ["caDepthB", "caDepthB"]].forEach(([id, field]) => {
    byId(id).addEventListener("input", (event) => {
      state[field] = Number(event.target.value);
      render();
    });
  });
  byId("useAdScenario").addEventListener("click", () => useScenarioInModel(`ad(${state.adDepth})`));
  byId("useCaScenario").addEventListener("click", () => useScenarioInModel(`ca(${state.caDepthA},${state.caDepthB})`));
  byId("modelScenario").addEventListener("change", (event) => {
    state.modelScenario = event.target.value;
    render();
  });
  [["observedTime", "observedTime"], ["observedGenetic", "observedGenetic"]].forEach(([id, field]) => {
    byId(id).addEventListener("input", (event) => {
      state[field] = Number(event.target.value);
      render();
    });
  });
  byId("stochastic").addEventListener("change", (event) => {
    state.mutationProcess = event.target.checked ? "deterministic" : "stochastic";
    render();
  });
  byId("directPreset").addEventListener("click", () => setTargetPreset(["ad(0)"]));
  byId("commonPreset").addEventListener("click", () => setTargetPreset(["ca(0,0)"]));
  byId("starterPreset").addEventListener("click", () => setTargetPreset(["ad(0)", "ca(0,0)"]));

  byId("surface").addEventListener("pointermove", (event) => {
    const { rect, time, genetic } = surfacePosition(event, byId("surface"));
    const surface = combineTargetSurface(comparisonModel(), state.mutationProcess);
    const column = closestIndex(surface.time_values, time);
    const row = closestIndex(surface.genetic_values, genetic);
    const tooltip = byId("surfaceTooltip");
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.min(rect.width - 166, Math.max(2, event.clientX - rect.left + 10))}px`;
    tooltip.style.top = `${Math.min(rect.height - 56, Math.max(2, event.clientY - rect.top - 48))}px`;
    tooltip.textContent = `${timeObservationText(surface.time_values[column])} · ${geneticObservationText(surface.genetic_values[row])} · score ${format(surface.scores[row][column], 2)} of ${state.targets.size}`;
  });
  byId("surface").addEventListener("pointerleave", () => {
    byId("surfaceTooltip").style.display = "none";
  });
  byId("surface").addEventListener("click", (event) => {
    const { time, genetic } = surfacePosition(event, byId("surface"));
    state.observedTime = Math.round(time);
    state.observedGenetic = Math.max(0, Math.min(12, Math.round(genetic)));
    byId("observedTime").value = state.observedTime;
    byId("observedGenetic").value = state.observedGenetic;
    render();
  });
  window.addEventListener("resize", render);
}

async function initialize() {
  try {
    const response = await fetch("data/epilink-explorer-data.json");
    if (!response.ok) throw new Error(`data file returned ${response.status}`);
    explorerData = await response.json();
    state.parameterKey = explorerData.parameter_catalog[0].key;
    state.sensitivityLevel = explorerData.parameter_catalog[0].default_index;
    populateModelScenarioOptions();
    bindEvents();
    renderTargetChips();
    render();
  } catch (error) {
    const insightCopy = byId("insight").lastElementChild;
    insightCopy.textContent = `The teaching model could not be loaded: ${error.message}`;
  }
}

initialize();
