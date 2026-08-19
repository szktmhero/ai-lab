"""HTML visualization for organism simulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any


def render_html(
    snapshots: List[Dict[str, Any]],
    metrics: List[Dict[str, Any]],
    resource_field: List[List[float]],
    output_path: Path,
    world_w: int = 32,
    world_h: int = 32,
):
    """Render simulation as interactive HTML with canvas."""

    snapshots_json = json.dumps(snapshots)
    metrics_json = json.dumps(metrics)
    resource_json = json.dumps(resource_field)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Organism Simulation</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: monospace; background: #111; color: #eee; display: flex; flex-direction: column; align-items: center; padding: 20px; }}
  h1 {{ margin-bottom: 10px; font-size: 18px; }}
  #controls {{ margin: 10px 0; display: flex; gap: 10px; align-items: center; }}
  #controls button {{ padding: 5px 15px; cursor: pointer; background: #333; color: #eee; border: 1px solid #555; }}
  #controls button:hover {{ background: #444; }}
  #step-display {{ font-size: 14px; }}
  canvas {{ border: 1px solid #333; }}
  #metrics {{ display: flex; gap: 20px; margin-top: 15px; flex-wrap: wrap; justify-content: center; }}
  .metric-box {{ background: #1a1a1a; border: 1px solid #333; padding: 10px; min-width: 150px; }}
  .metric-label {{ font-size: 10px; color: #888; }}
  .metric-value {{ font-size: 16px; font-weight: bold; }}
  #chart-container {{ margin-top: 20px; width: 100%; max-width: 800px; }}
  canvas.chart {{ width: 100%; height: 150px; }}
</style>
</head>
<body>
<h1>128-Cell Emergent Organism</h1>

<div id="controls">
  <button onclick="stepBack()">&#9664; Back</button>
  <button onclick="togglePlay()" id="playBtn">Play</button>
  <button onclick="stepForward()">Forward &#9654;</button>
  <span id="step-display">Step: 0 / 0</span>
  <label>Speed: <input type="range" id="speed" min="1" max="30" value="10"></label>
</div>

<canvas id="world" width="{world_w * 16}" height="{world_h * 16}"></canvas>

<div id="metrics">
  <div class="metric-box"><div class="metric-label">Alive</div><div class="metric-value" id="m-alive">-</div></div>
  <div class="metric-box"><div class="metric-label">Avg Energy</div><div class="metric-value" id="m-energy">-</div></div>
  <div class="metric-box"><div class="metric-label">Largest Cluster</div><div class="metric-value" id="m-cluster">-</div></div>
  <div class="metric-box"><div class="metric-label">Clusters</div><div class="metric-value" id="m-clusters">-</div></div>
  <div class="metric-box"><div class="metric-label">Avg Neighbors</div><div class="metric-value" id="m-neighbors">-</div></div>
  <div class="metric-box"><div class="metric-label">Signal Diversity</div><div class="metric-value" id="m-signal">-</div></div>
  <div class="metric-box"><div class="metric-label">Spatial Entropy</div><div class="metric-value" id="m-entropy">-</div></div>
  <div class="metric-box"><div class="metric-label">Resource</div><div class="metric-value" id="m-resource">-</div></div>
</div>

<div id="chart-container">
  <canvas id="chart" class="chart" width="800" height="150"></canvas>
</div>

<script>
const SNAPSHOTS = {snapshots_json};
const METRICS = {metrics_json};
const FALLBACK_RESOURCE = {resource_json};
const W = {world_w};
const H = {world_h};
const CELL_SIZE = 16;

let currentStep = 0;
let playing = false;
let timer = null;

const canvas = document.getElementById('world');
const ctx = canvas.getContext('2d');

function drawResource(resource) {{
  for (let y = 0; y < H; y++) {{
    for (let x = 0; x < W; x++) {{
      const val = resource[y][x] / 10.0;
      const r = Math.floor(20 + val * 30);
      const g = Math.floor(40 + val * 60);
      const b = Math.floor(20 + val * 30);
      ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
      ctx.fillRect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE);
    }}
  }}
}}

function energyColor(energy) {{
  const t = Math.min(energy / 100.0, 1.0);
  const r = Math.floor(255 * (1 - t));
  const g = Math.floor(200 * t);
  const b = Math.floor(100 * t);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function signalColor(signal) {{
  const mag = Math.sqrt(signal.reduce((s, v) => s + v * v, 0));
  const t = Math.min(mag / 3.0, 1.0);
  const h = 200 + t * 160;
  return `hsl(${{h}}, 70%, ${{40 + t * 30}}%)`;
}}

function draw() {{
  const snap = SNAPSHOTS[currentStep];
  if (!snap) return;
  drawResource(snap.resource_field || FALLBACK_RESOURCE);

  const cells = Object.values(snap.cells);
  for (const c of cells) {{
    if (!c.alive) continue;
    const x = c.pos[0];
    const y = c.pos[1];

    ctx.beginPath();
    ctx.arc(x * CELL_SIZE + CELL_SIZE / 2, y * CELL_SIZE + CELL_SIZE / 2, CELL_SIZE / 2.5, 0, Math.PI * 2);
    ctx.fillStyle = energyColor(c.energy);
    ctx.fill();

    const sigMag = Math.sqrt(c.signal_output.reduce((s, v) => s + v * v, 0));
    if (sigMag > 0.3) {{
      ctx.beginPath();
      ctx.arc(x * CELL_SIZE + CELL_SIZE / 2, y * CELL_SIZE + CELL_SIZE / 2, CELL_SIZE / 2.2, 0, Math.PI * 2);
      ctx.strokeStyle = signalColor(c.signal_output);
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }}
  }}
}}

function updateMetrics() {{
  const snap = SNAPSHOTS[currentStep];
  const m = METRICS.find(item => item.step === snap.step) || METRICS[METRICS.length - 1];
  if (!m) return;
  document.getElementById('m-alive').textContent = m.alive_cells;
  document.getElementById('m-energy').textContent = m.average_energy.toFixed(2);
  document.getElementById('m-cluster').textContent = m.largest_cluster_size;
  document.getElementById('m-clusters').textContent = m.number_of_clusters;
  document.getElementById('m-neighbors').textContent = m.average_neighbor_count.toFixed(2);
  document.getElementById('m-signal').textContent = m.signal_diversity.toFixed(3);
  document.getElementById('m-entropy').textContent = m.spatial_entropy.toFixed(3);
  document.getElementById('m-resource').textContent = m.resource_consumption.toFixed(1);
  document.getElementById('step-display').textContent = `Step: ${{snap.step}} / ${{SNAPSHOTS[SNAPSHOTS.length - 1].step}}`;
}}

function drawChart() {{
  const chartCanvas = document.getElementById('chart');
  const cctx = chartCanvas.getContext('2d');
  cctx.clearRect(0, 0, 800, 150);

  const keys = ['alive_cells', 'average_energy', 'largest_cluster_size', 'signal_diversity'];
  const colors = ['#4f4', '#ff4', '#4af', '#f4a'];

  for (let k = 0; k < keys.length; k++) {{
    const key = keys[k];
    const vals = METRICS.map(m => m[key]);
    const max = Math.max(...vals) || 1;
    cctx.beginPath();
    cctx.strokeStyle = colors[k];
    cctx.lineWidth = 1.5;
    for (let i = 0; i < vals.length; i++) {{
      const x = (i / (vals.length - 1)) * 800;
      const y = 140 - (vals[i] / max) * 130;
      if (i === 0) cctx.moveTo(x, y);
      else cctx.lineTo(x, y);
    }}
    cctx.stroke();
  }}

  // Current step indicator
  const x = (SNAPSHOTS[currentStep].step / Math.max(METRICS.length, 1)) * 800;
  cctx.beginPath();
  cctx.strokeStyle = '#fff';
  cctx.lineWidth = 1;
  cctx.moveTo(x, 0);
  cctx.lineTo(x, 150);
  cctx.stroke();
}}

function render() {{
  draw();
  updateMetrics();
  drawChart();
}}

function stepForward() {{
  currentStep = Math.min(currentStep + 1, SNAPSHOTS.length - 1);
  render();
}}

function stepBack() {{
  currentStep = Math.max(currentStep - 1, 0);
  render();
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
  if (playing) {{
    timer = setInterval(() => {{
      if (currentStep >= SNAPSHOTS.length - 1) {{
        playing = false;
        document.getElementById('playBtn').textContent = 'Play';
        clearInterval(timer);
        return;
      }}
      stepForward();
    }}, 1000 / document.getElementById('speed').value);
  }} else {{
    clearInterval(timer);
  }}
}}

document.getElementById('speed').addEventListener('input', () => {{
  if (playing) {{
    clearInterval(timer);
    timer = setInterval(() => {{
      if (currentStep >= SNAPSHOTS.length - 1) {{
        playing = false;
        document.getElementById('playBtn').textContent = 'Play';
        clearInterval(timer);
        return;
      }}
      stepForward();
    }}, 1000 / document.getElementById('speed').value);
  }}
}});

render();
</script>
</body>
</html>"""

    output_path.write_text(html)
