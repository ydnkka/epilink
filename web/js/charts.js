const NS = "http://www.w3.org/2000/svg";

export function renderTree(svg, scenario) {
  while (svg.children.length > 2) svg.removeChild(svg.lastChild);
  const append = (node) => svg.appendChild(node);
  const edge = (x1, y1, x2, y2, hidden = false) => append(svgNode("line", {
    x1, y1, x2, y2,
    stroke: hidden ? "#9baaae" : "#648a95",
    "stroke-width": hidden ? 1.5 : 2.2,
    "stroke-dasharray": hidden ? "4 4" : "",
  }));
  const node = (x, y, label, kind) => {
    const fill = kind === "sample" ? "#153c56" : kind === "source" ? "#d89e35" : "#ffffff";
    const stroke = kind === "sample" ? "#153c56" : kind === "source" ? "#c28a29" : "#9baaae";
    append(svgNode("circle", { cx: x, cy: y, r: kind === "sample" ? 16 : 12, fill, stroke, "stroke-width": 2 }));
    append(svgNode("text", { x, y: y + 4, "text-anchor": "middle", fill: kind === "sample" ? "#fff" : "#40536a", "font-size": kind === "sample" ? 11 : 10, "font-weight": 700 }, kind === "sample" ? label : "·"));
    if (kind === "sample") append(svgNode("text", { x, y: y + 37, "text-anchor": "middle", fill: "#40536a", "font-size": 11, "font-weight": 650 }, `sample ${label}`));
  };

  if (scenario.kind === "ad") {
    const count = scenario.a + 2;
    const start = 64;
    const end = 555;
    const step = (end - start) / (count - 1);
    const y = 136;
    for (let index = 0; index < count - 1; index += 1) {
      edge(start + step * index + 16, y, start + step * (index + 1) - (index + 1 === count - 1 ? 16 : 12), y, index > 0);
    }
    for (let index = 0; index < count; index += 1) {
      if (index === 0) node(start + step * index, y, "A", "sample");
      else if (index === count - 1) node(start + step * index, y, "B", "sample");
      else node(start + step * index, y, "", "hidden");
    }
    append(svgNode("text", { x: 310, y: 53, "text-anchor": "middle", fill: "#5b6879", "font-size": 12 }, scenario.a === 0 ? "direct route" : `${scenario.a} unsampled ${scenario.a === 1 ? "person" : "people"}`));
    return scenario.a === 0
      ? "One possible story is that A passed the infection directly to B. Their sample dates can still vary because people develop symptoms and get sampled at different times."
      : `This possible route passes through ${scenario.a} unsampled ${scenario.a === 1 ? "person" : "people"}. Each extra step usually adds time and opportunity for genome change.`;
  }

  const source = { x: 310, y: 52 };
  const sampleA = { x: 120, y: 216 };
  const sampleB = { x: 500, y: 216 };
  const branchNodes = (depth, end) => {
    const count = depth + 2;
    const points = [source];
    for (let index = 1; index < count - 1; index += 1) {
      const fraction = index / (count - 1);
      points.push({ x: source.x + (end.x - source.x) * fraction, y: source.y + (end.y - source.y) * fraction });
    }
    points.push(end);
    for (let index = 0; index < points.length - 1; index += 1) {
      edge(points[index].x, points[index].y + 12, points[index + 1].x, points[index + 1].y - (index + 1 === points.length - 1 ? 16 : 12), index > 0);
    }
    for (let index = 1; index < points.length - 1; index += 1) node(points[index].x, points[index].y, "", "hidden");
  };
  branchNodes(scenario.a, sampleA);
  branchNodes(scenario.b, sampleB);
  node(source.x, source.y, "", "source");
  node(sampleA.x, sampleA.y, "A", "sample");
  node(sampleB.x, sampleB.y, "B", "sample");
  append(svgNode("text", { x: 310, y: 23, "text-anchor": "middle", fill: "#7b5c1c", "font-size": 11, "font-weight": 700 }, "shared unsampled source"));
  append(svgNode("text", { x: 78, y: 121, "text-anchor": "middle", fill: "#5b6879", "font-size": 11 }, `${scenario.a + 1} transmission ${scenario.a ? "steps" : "step"}`));
  append(svgNode("text", { x: 542, y: 121, "text-anchor": "middle", fill: "#5b6879", "font-size": 11 }, `${scenario.b + 1} transmission ${scenario.b ? "steps" : "step"}`));
  return "A and B are on separate routes from the same unsampled source. Either person could be sampled first, while both routes provide time for genome differences to appear.";
}

export function renderHistogram(svg, summary, observed, xLabel, mode) {
  svg.replaceChildren();
  const W = Number(svg.getAttribute("viewBox")?.split(/\s+/)[2]) || 340;
  const H = Number(svg.getAttribute("viewBox")?.split(/\s+/)[3]) || 205;
  const left = 39;
  const right = 12;
  const top = 12;
  const bottom = 30;
  const { edges, counts } = summary.histogram;
  const min = edges[0];
  const max = edges[edges.length - 1];
  const maxCount = Math.max(...counts, 1);
  const x = (value) => left + ((value - min) / (max - min)) * (W - left - right);
  const y = (value) => H - bottom - value * (H - top - bottom);

  [0, .5, 1].forEach((value) => svg.append(svgNode("line", { class: "grid", x1: left, x2: W - right, y1: y(value), y2: y(value) })));
  svg.append(svgNode("line", { class: "axis", x1: left, x2: W - right, y1: H - bottom, y2: H - bottom }));
  svg.append(svgNode("line", { class: "axis", x1: left, x2: left, y1: top, y2: H - bottom }));

  counts.forEach((count, index) => {
    const binLeft = x(edges[index]);
    const binWidth = x(edges[index + 1]) - binLeft;
    const gap = Math.min(1.5, binWidth * .18);
    const binTop = y(count / maxCount);
    svg.append(svgNode("rect", {
      class: "bar",
      x: binLeft + gap / 2,
      y: binTop,
      width: Math.max(0, binWidth - gap),
      height: H - bottom - binTop,
    }));
  });
  if (observed !== null) {
    const markerValue = clamp(observed, min, max);
    const outside = observed < min || observed > max;
    const markerX = x(markerValue);
    svg.append(svgNode("line", { class: `marker${outside ? " outside" : ""}`, x1: markerX, x2: markerX, y1: top, y2: H - bottom }));
    svg.append(svgNode("text", {
      class: "marker-label",
      x: markerX + (observed < min ? 4 : observed > max ? -4 : 0),
      y: top + 10,
      "text-anchor": observed < min ? "start" : observed > max ? "end" : "middle",
    }, outside ? "example outside chart" : "example"));
  }

  const firstValue = (edges[0] + edges[1]) / 2;
  const lastValue = (edges[edges.length - 2] + edges[edges.length - 1]) / 2;
  const middleValue = mode === "time" || mode === "genetic"
    ? Math.round((firstValue + lastValue) / 2)
    : (firstValue + lastValue) / 2;
  [firstValue, middleValue, lastValue].forEach((value) => svg.append(svgNode("text", { x: x(value), y: H - 10, "text-anchor": "middle" }, format(value, 0))));
  svg.append(svgNode("text", { x: (left + W - right) / 2, y: H - 1, "text-anchor": "middle" }, xLabel));
  const yLabelX = 12;
  const yLabelY = (top + H - bottom) / 2;
  svg.append(svgNode("text", {
    x: yLabelX,
    y: yLabelY,
    "text-anchor": "middle",
    transform: `rotate(-90 ${yLabelX} ${yLabelY})`,
  }, "more common"));
}

export function drawSurface(canvas, surface, targetCount, observedTime, observedGenetic) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext("2d");
  const pad = { left: 44 * dpr, right: 12 * dpr, top: 12 * dpr, bottom: 32 * dpr };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const minTime = surface.time_values[0];
  const maxTime = surface.time_values[surface.time_values.length - 1];
  const maxGenetic = surface.genetic_values[0];
  const minGenetic = surface.genetic_values[surface.genetic_values.length - 1];
  ctx.clearRect(0, 0, width, height);
  const columns = surface.time_values.length;
  const rows = surface.genetic_values.length;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      ctx.fillStyle = colorForScore(surface.scores[row][column], Math.max(1, targetCount));
      ctx.fillRect(pad.left + column * plotWidth / columns, pad.top + row * plotHeight / rows, Math.ceil(plotWidth / columns) + 1, Math.ceil(plotHeight / rows) + 1);
    }
  }
  ctx.strokeStyle = "#b8c5ca";
  ctx.lineWidth = dpr;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, pad.top + plotHeight);
  ctx.lineTo(pad.left + plotWidth, pad.top + plotHeight);
  ctx.stroke();
  ctx.fillStyle = "#5b6879";
  ctx.font = `${10 * dpr}px Inter, system-ui, sans-serif`;
  ctx.textAlign = "center";
  [-15, 10, 35].forEach((value) => {
    const x = pad.left + (value - minTime) / (maxTime - minTime) * plotWidth;
    ctx.fillText(String(value), x, height - 10 * dpr);
  });
  ctx.fillText("days between samples (B minus A)", pad.left + plotWidth / 2, height - dpr);
  ctx.textAlign = "right";
  [0, 6, 12].forEach((value) => {
    const y = pad.top + (maxGenetic - value) / (maxGenetic - minGenetic) * plotHeight;
    ctx.fillText(String(value), pad.left - 6 * dpr, y + 3 * dpr);
  });
  ctx.save();
  ctx.translate(11 * dpr, pad.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("genome differences", 0, 0);
  ctx.restore();
  const markerX = pad.left + (observedTime - minTime) / (maxTime - minTime) * plotWidth;
  const markerY = pad.top + (maxGenetic - observedGenetic) / (maxGenetic - minGenetic) * plotHeight;
  ctx.beginPath();
  ctx.arc(markerX, markerY, 5.3 * dpr, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.lineWidth = 2 * dpr;
  ctx.strokeStyle = "#c6535c";
  ctx.stroke();
}

export function surfacePosition(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  const left = 44;
  const right = 12;
  const top = 12;
  const bottom = 32;
  const x = clamp((event.clientX - rect.left - left) / (rect.width - left - right), 0, 1);
  const y = clamp((event.clientY - rect.top - top) / (rect.height - top - bottom), 0, 1);
  return { rect, time: -15 + x * 50, genetic: 12 - y * 12 };
}

function svgNode(name, attributes = {}, text = "") {
  const node = document.createElementNS(NS, name);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  if (text) node.textContent = text;
  return node;
}

function colorForScore(value, max) {
  const ratio = clamp(value / Math.max(max, .01), 0, 1);
  const stops = [[237, 242, 242], [184, 222, 216], [30, 121, 145], [21, 60, 86]];
  const position = ratio * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(position));
  const fraction = position - index;
  const color = stops[index].map((valueAtIndex, channel) => Math.round(valueAtIndex + (stops[index + 1][channel] - valueAtIndex) * fraction));
  return `rgb(${color.join(",")})`;
}

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function format(value, decimals = 1) { return Number(value).toFixed(decimals); }
