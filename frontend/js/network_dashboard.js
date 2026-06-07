const DEFAULT_API_BASE = "/api/earningalz";

const signalColors = {
  demand_outlook: "#00E5FF",
  margin_outlook: "#00F5A0",
  supply_outlook: "#FFB703",
  inventory_outlook: "#9D4EDD",
  pricing_outlook: "#FF4D9D",
  capex_outlook: "#4CC9F0",
};

const relationColors = {
  upstream: "#FFB703",
  downstream: "#00E5FF",
  partner: "#9D4EDD",
  parent: "#4CC9F0",
  subsidiary: "#00F5A0",
  competitor: "#FF4D9D",
  related: "#94A3B8",
  customer: "#38BDF8",
  customer_group: "#38BDF8",
  supplier_group: "#F59E0B",
};

let rawData = null;
let networkList = [];
let selectedNetworkId = null;
let currentLinks = [];
let currentNodes = [];
let simulation = null;
let pulseTimer = null;

const svg = d3.select("#network");
const tooltip = d3.select("#tooltip");
const detailBox = d3.select("#detailBox");

function getApiBase() {
  const saved = localStorage.getItem("earningalz_api_base");

  // Backward compatibility:
  // If the browser cached the old value "/api", automatically migrate it.
  if (!saved || saved === "/api") {
    localStorage.setItem("earningalz_api_base", DEFAULT_API_BASE);
    return DEFAULT_API_BASE;
  }

  return saved;
}

function setApiBase(value) {
  const clean = String(value || "").trim().replace(/\/$/, "");

  if (!clean || clean === "/api") {
    localStorage.setItem("earningalz_api_base", DEFAULT_API_BASE);
    return;
  }

  localStorage.setItem("earningalz_api_base", clean);
}

function apiUrl(path) {
  const base = getApiBase().replace(/\/$/, "");
  const cleanPath = String(path || "").startsWith("/") ? path : `/${path}`;
  return `${base}${cleanPath}`;
}

function encodeNetworkId(id) {
  return String(id || "")
    .split("/")
    .map(encodeURIComponent)
    .join("/");
}

async function fetchJson(url) {
  const res = await fetch(url);

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }

  return res.json();
}

function formatRate(x) {
  return x == null || Number.isNaN(Number(x))
    ? "N/A"
    : `${(100 * Number(x)).toFixed(1)}%`;
}

function unique(v) {
  return Array.from(
    new Set(
      v.filter((x) => x != null && String(x).length > 0)
    )
  ).sort();
}

function quarterIndex(q) {
  const m = String(q).match(/^(\d{4})Q([1-4])$/);
  return m ? Number(m[1]) * 4 + Number(m[2]) : NaN;
}

function linkColor(d) {
  return signalColors[d.signal] || relationColors[d.relation_group] || "#00E5FF";
}

function nodeColor(d) {
  if (d.is_center) return "#FFFFFF";

  return (d.leader_score || 0) > (d.follower_score || 0)
    ? "#00E5FF"
    : "#9D4EDD";
}

function shortNode(x) {
  return String(x || "").replace("COMPANY::", "");
}

async function loadNetworkList() {
  const select = d3.select("#networkSelect");
  select.html("<option>Loading...</option>");

  try {
    const data = await fetchJson(apiUrl("/dynamic-networks"));
    networkList = data.networks || [];
  } catch (err) {
    console.warn("API network list failed, trying static data/manifest.json", err);

    try {
      const manifest = await fetchJson("data/manifest.json");
      networkList = Array.isArray(manifest)
        ? manifest
        : (manifest.networks || []);
    } catch (fallbackErr) {
      console.error(fallbackErr);
      networkList = [];
    }
  }

  select
    .selectAll("option")
    .data(networkList)
    .join("option")
    .attr("value", (d) => d.id)
    .text((d) => d.label || d.id);

  if (networkList.length > 0) {
    selectedNetworkId =
      selectedNetworkId && networkList.some((x) => x.id === selectedNetworkId)
        ? selectedNetworkId
        : networkList[0].id;

    select.property("value", selectedNetworkId);
  } else {
    select.html("<option>No generated networks found</option>");
  }
}

async function loadSelectedNetwork() {
  if (!selectedNetworkId) return;

  const meta = networkList.find((x) => x.id === selectedNetworkId) || {};
  rawData = null;

  try {
    rawData = await fetchJson(
      apiUrl(`/dynamic-networks/${encodeNetworkId(selectedNetworkId)}`)
    );
  } catch (err) {
    console.warn("API network load failed, trying static JSON", err);

    const filename = meta.filename || `${selectedNetworkId}.json`;
    rawData = await fetchJson(`data/${filename}`);
  }

  document.getElementById("headerBadge").textContent =
    `${rawData.metadata?.ticker || selectedNetworkId} · ${rawData.metadata?.node_count || 0} nodes · ${rawData.metadata?.link_count || 0} links`;

  d3.select("#metadataBox").html(metadataHtml(rawData.metadata || {}));

  fillControls(rawData);
  applyFilters();

  setTimeout(() => playPulses(), 500);
}

function metadataHtml(m) {
  return `
    <b>${m.ticker || "Unknown"}</b><br/>
    mode: ${m.mode || ""}<br/>
    signal: ${m.signal || "All"}<br/>
    quarter range: ${(m.start_quarter || "-")} → ${(m.end_quarter || "-")}<br/>
    hop depth: ${m.hop_depth ?? ""}<br/>
    nodes: ${m.node_count ?? 0}<br/>
    links: ${m.link_count ?? 0}<br/>
    events: ${(m.event_count ?? 0).toLocaleString?.() || m.event_count || 0}
  `;
}

function fillControls(data) {
  const signals = [
    "All",
    ...unique((data.links || []).map((d) => d.signal)),
  ];

  const relations = [
    "All",
    ...unique((data.links || []).map((d) => d.relation_group)),
  ];

  const quarters = [
    "All",
    ...unique(data.metadata?.quarters || []),
  ];

  d3.select("#signalSelect")
    .selectAll("option")
    .data(signals)
    .join("option")
    .attr("value", (d) => d)
    .text((d) => d);

  d3.select("#relationSelect")
    .selectAll("option")
    .data(relations)
    .join("option")
    .attr("value", (d) => d)
    .text((d) => d);

  d3.select("#quarterSelect")
    .selectAll("option")
    .data(quarters)
    .join("option")
    .attr("value", (d) => d)
    .text((d) => d);

  d3.select("#directionSelect").property("value", "All");

  const defaultSignal =
    data.metadata?.signal && data.metadata.signal !== "All"
      ? data.metadata.signal
      : "All";

  if (signals.includes(defaultSignal)) {
    d3.select("#signalSelect").property("value", defaultSignal);
  }
}

function applyFilters() {
  if (!rawData) return;

  const signal = d3.select("#signalSelect").property("value");
  const direction = d3.select("#directionSelect").property("value");
  const relation = d3.select("#relationSelect").property("value");
  const quarter = d3.select("#quarterSelect").property("value");

  currentLinks = (rawData.links || []).filter((d) => {
    if (signal !== "All" && d.signal !== signal) return false;
    if (direction !== "All" && d.source_direction !== direction) return false;
    if (relation !== "All" && d.relation_group !== relation) return false;

    if (
      quarter !== "All" &&
      d.first_source_quarter !== quarter &&
      d.last_target_quarter !== quarter
    ) {
      return false;
    }

    return true;
  });

  const nodeIds = new Set();

  currentLinks.forEach((d) => {
    nodeIds.add(d.source);
    nodeIds.add(d.target);
  });

  currentNodes = (rawData.nodes || []).filter((d) => nodeIds.has(d.id));

  updateMetrics();
  drawNetwork();
  drawTimeline();
  updateTopLinks();

  const emptyState = document.getElementById("emptyState");
  if (emptyState) {
    emptyState.classList.toggle("hidden", currentLinks.length > 0);
  }
}

function updateMetrics() {
  const avg = d3.mean(currentLinks, (d) => d.success_rate || 0);
  const events = d3.sum(currentLinks, (d) => d.event_count || 0);

  d3.select("#mNodes").text(currentNodes.length.toLocaleString());
  d3.select("#mLinks").text(currentLinks.length.toLocaleString());
  d3.select("#mEvents").text(events.toLocaleString());
  d3.select("#mSuccess").text(formatRate(avg));
}

function drawNetwork() {
  const width = svg.node().clientWidth;
  const height = svg.node().clientHeight;

  svg.selectAll("*").remove();

  if (!currentNodes.length || !currentLinks.length) return;

  const defs = svg.append("defs");

  defs
    .append("filter")
    .attr("id", "glow")
    .html(`
      <feGaussianBlur stdDeviation="3.5" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    `);

  const zoomLayer = svg.append("g");
  const linkLayer = zoomLayer.append("g");
  const pulseLayer = zoomLayer.append("g");
  const nodeLayer = zoomLayer.append("g");
  const labelLayer = zoomLayer.append("g");

  const nodes = currentNodes.map((d) => ({ ...d }));
  const links = currentLinks.map((d) => ({ ...d }));

  simulation = d3
    .forceSimulation(nodes)
    .force(
      "link",
      d3
        .forceLink(links)
        .id((d) => d.id)
        .distance((d) => 90 + 35 / Math.max(0.1, d.success_rate || 0.1))
        .strength(0.48)
    )
    .force("charge", d3.forceManyBody().strength(-440))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .force(
      "collision",
      d3.forceCollide().radius((d) =>
        16 + Math.log1p(d.outgoing_exposures || 0) * 2.5
      )
    );

  const link = linkLayer
    .selectAll("line")
    .data(links)
    .join("line")
    .attr("class", "link")
    .attr("stroke", (d) => linkColor(d))
    .attr("stroke-width", (d) => 1.2 + Math.sqrt(d.event_count || 1) * 0.22)
    .on("mousemove", (e, d) => showTooltip(e, edgeHtml(d)))
    .on("mouseleave", hideTooltip)
    .on("click", (e, d) => detailBox.html(edgeHtml(d)));

  const node = nodeLayer
    .selectAll("circle")
    .data(nodes)
    .join("circle")
    .attr("class", (d) => (d.is_center ? "node center" : "node"))
    .attr("r", (d) =>
      d.is_center
        ? 13
        : 7 + Math.log1p((d.outgoing_exposures || 0) + (d.incoming_exposures || 0)) * 1.3
    )
    .attr("fill", (d) => nodeColor(d))
    .attr("filter", "url(#glow)")
    .call(drag(simulation))
    .on("mousemove", (e, d) => showTooltip(e, nodeHtml(d)))
    .on("mouseleave", hideTooltip)
    .on("click", (e, d) => detailBox.html(nodeHtml(d)));

  const label = labelLayer
    .selectAll("text")
    .data(nodes)
    .join("text")
    .attr("class", "node-label")
    .text((d) => d.label || shortNode(d.id))
    .attr("dy", -13)
    .attr("text-anchor", "middle");

  simulation.on("tick", () => {
    link
      .attr("x1", (d) => d.source.x)
      .attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x)
      .attr("y2", (d) => d.target.y);

    node
      .attr("cx", (d) => d.x)
      .attr("cy", (d) => d.y);

    label
      .attr("x", (d) => d.x)
      .attr("y", (d) => d.y);
  });

  svg.call(
    d3.zoom()
      .scaleExtent([0.15, 5])
      .on("zoom", (e) => zoomLayer.attr("transform", e.transform))
  );

  window.currentPulseLayer = pulseLayer;
  window.currentLinksForPulse = links;
}

function edgeHtml(d) {
  return `
    <b>${shortNode(d.source)}</b> → <b>${shortNode(d.target)}</b><br/>
    signal: <span style="color:${linkColor(d)}">${d.signal}</span><br/>
    relation: ${d.relation_group}<br/>
    direction: ${d.source_direction}<br/>
    events: ${d.event_count}<br/>
    success rate: ${formatRate(d.success_rate)}<br/>
    direction match: ${formatRate(d.direction_match_rate)}<br/>
    target active: ${formatRate(d.target_active_rate)}<br/>
    avg gap days: ${d.avg_gap_days == null ? "N/A" : Number(d.avg_gap_days).toFixed(1)}<br/>
    window: ${d.first_source_quarter} → ${d.last_target_quarter}
  `;
}

function nodeHtml(d) {
  return `
    <b>${d.label || shortNode(d.id)}</b><br/>
    node: ${d.id}<br/>
    leader score: ${Number(d.leader_score || 0).toFixed(3)}<br/>
    follower score: ${Number(d.follower_score || 0).toFixed(3)}<br/>
    outgoing exposures: ${d.outgoing_exposures || 0}<br/>
    incoming exposures: ${d.incoming_exposures || 0}
  `;
}

function showTooltip(e, html) {
  tooltip
    .style("opacity", 1)
    .style("left", `${e.clientX + 14}px`)
    .style("top", `${e.clientY + 14}px`)
    .html(html);
}

function hideTooltip() {
  tooltip.style("opacity", 0);
}

function drag(sim) {
  function started(e, d) {
    if (!e.active) sim.alphaTarget(0.25).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(e, d) {
    d.fx = e.x;
    d.fy = e.y;
  }

  function ended(e, d) {
    if (!e.active) sim.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }

  return d3
    .drag()
    .on("start", started)
    .on("drag", dragged)
    .on("end", ended);
}

function playPulses() {
  if (pulseTimer) {
    clearInterval(pulseTimer);
    pulseTimer = null;
  }

  const speed = Number(d3.select("#speedRange").property("value"));
  const pulseLayer = window.currentPulseLayer;
  const links = (window.currentLinksForPulse || []).slice(0, 120);

  if (!pulseLayer || links.length === 0) return;

  pulseLayer.selectAll("*").remove();

  function spawn(d) {
    if (!d.source || !d.target) return;

    const c = pulseLayer
      .append("circle")
      .attr("class", "pulse")
      .attr("r", 4 + Math.sqrt(d.event_count || 1) * 0.22)
      .attr("fill", linkColor(d))
      .attr("opacity", 0.95);

    c.transition()
      .duration(1800 / speed)
      .ease(d3.easeLinear)
      .attrTween("cx", () => (t) => d.source.x + (d.target.x - d.source.x) * t)
      .attrTween("cy", () => (t) => d.source.y + (d.target.y - d.source.y) * t)
      .attr("opacity", 0.05)
      .remove();
  }

  let i = 0;

  pulseTimer = setInterval(() => {
    for (let k = 0; k < 4; k++) {
      spawn(links[i % links.length]);
      i++;
    }
  }, 180 / speed);
}

function drawTimeline() {
  const tsvg = d3.select("#timeline");
  const width = tsvg.node().clientWidth;
  const height = tsvg.node().clientHeight;

  tsvg.selectAll("*").remove();

  if (!rawData) return;

  const signal = d3.select("#signalSelect").property("value");
  let timeline = rawData.timeline || [];

  if (signal !== "All") {
    timeline = timeline.filter((d) => d.signal === signal);
  }

  if (!timeline.length) return;

  const quarters = unique(timeline.map((d) => d.quarter))
    .sort((a, b) => quarterIndex(a) - quarterIndex(b));

  const x = d3.scalePoint()
    .domain(quarters)
    .range([45, width - 28]);

  const y = d3.scaleLinear()
    .domain([0, d3.max(timeline, (d) => d.success_rate || 0) || 1])
    .nice()
    .range([height - 35, 18]);

  const line = d3.line()
    .x((d) => x(d.quarter))
    .y((d) => y(d.success_rate || 0))
    .curve(d3.curveMonotoneX);

  const bySignal = d3.group(timeline, (d) => d.signal);

  tsvg
    .append("g")
    .attr("transform", `translate(0,${height - 35})`)
    .call(
      d3.axisBottom(x).tickValues(
        quarters.filter((d, i) => i % Math.ceil(quarters.length / 10) === 0)
      )
    )
    .selectAll("text")
    .attr("fill", "#9FB3C8");

  tsvg
    .append("g")
    .attr("transform", "translate(45,0)")
    .call(d3.axisLeft(y).ticks(4))
    .selectAll("text")
    .attr("fill", "#9FB3C8");

  for (const [sig, values] of bySignal) {
    tsvg
      .append("path")
      .datum(
        values.sort((a, b) => quarterIndex(a.quarter) - quarterIndex(b.quarter))
      )
      .attr("fill", "none")
      .attr("stroke", signalColors[sig] || "#00E5FF")
      .attr("stroke-width", 2.2)
      .attr("d", line);
  }
}

function updateTopLinks() {
  const table = d3.select("#topLinksTable");

  const rows = currentLinks
    .slice()
    .sort((a, b) => (b.success_count || 0) - (a.success_count || 0))
    .slice(0, 12);

  table.html("");

  table
    .append("thead")
    .append("tr")
    .selectAll("th")
    .data(["Path", "Signal", "Rate"])
    .join("th")
    .text((d) => d);

  const tr = table
    .append("tbody")
    .selectAll("tr")
    .data(rows)
    .join("tr")
    .on("click", (e, d) => detailBox.html(edgeHtml(d)));

  tr.append("td").text((d) => `${shortNode(d.source)} → ${shortNode(d.target)}`);
  tr.append("td").text((d) => d.signal);
  tr.append("td").text((d) => formatRate(d.success_rate));
}

function resetView() {
  if (simulation) simulation.alpha(0.8).restart();
}

function setupEvents() {
  d3.select("#signalSelect").on("change", applyFilters);
  d3.select("#directionSelect").on("change", applyFilters);
  d3.select("#relationSelect").on("change", applyFilters);
  d3.select("#quarterSelect").on("change", applyFilters);

  d3.select("#playBtn").on("click", playPulses);
  d3.select("#resetBtn").on("click", resetView);

  d3.select("#networkSelect").on("change", async function () {
    selectedNetworkId = this.value;
    await loadSelectedNetwork();
  });

  d3.select("#reloadNetworksBtn").on("click", async () => {
    await loadNetworkList();
    await loadSelectedNetwork();
  });

  d3.select("#saveApiBtn").on("click", async () => {
    const input = document.getElementById("apiBaseInput");
    setApiBase(input.value);
    input.value = getApiBase();

    await loadNetworkList();
    await loadSelectedNetwork();
  });
}

async function init() {
  const apiBaseInput = document.getElementById("apiBaseInput");

  if (apiBaseInput) {
    apiBaseInput.value = getApiBase();
  }

  setupEvents();

  await loadNetworkList();
  await loadSelectedNetwork();
}

init().catch((err) => {
  console.error(err);

  document.getElementById("headerBadge").textContent = "Load failed";

  detailBox.html(
    `<span class="error">${err.message || err}</span>`
  );
});