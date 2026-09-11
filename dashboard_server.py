"""Live monitoring dashboard — FastAPI + polling + background training.

Run:  python dashboard_server.py
Open: http://localhost:8080
"""

from __future__ import annotations

import collections
import json
import threading
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from flyecon.avatar.svg import to_html as svg_to_html
from flyecon.dashboard.renderer import _get_plotly_js
from flyecon.harness import Harness

log = structlog.get_logger()

_stop_event = threading.Event()
_harness: Harness | None = None
_train_thread: threading.Thread | None = None
_metrics_lock = threading.Lock()
_metrics: dict[str, Any] = {
    "step": collections.deque(maxlen=1000),
    "fci": collections.deque(maxlen=1000),
    "reward": collections.deque(maxlen=1000),
    "policy_loss": collections.deque(maxlen=1000),
    "value_loss": collections.deque(maxlen=1000),
    "entropy": collections.deque(maxlen=1000),
    "value_ratio": collections.deque(maxlen=100),
    "value_ratio_step": collections.deque(maxlen=100),
    # Feature 1: Economy decisions
    "agreement_rate": collections.deque(maxlen=100),
    "policy_mean_money": collections.deque(maxlen=100),
    "oracle_mean_money": collections.deque(maxlen=100),
    "decision_trace": [],  # last trace (list of round dicts)
    # Feature 2: Neural activity
    "neural_layers": {},  # latest layer→rate dict
    "neural_activity": 0.0,  # latest mean_output_rate scalar
}
_last_fci: float = 0.0
_last_neural_activity: float = 0.0


def _training_loop(harness: Harness, stop: threading.Event) -> None:
    import torch
    torch.set_num_threads(1)

    global _last_fci, _last_neural_activity

    orig_emit = harness._emit

    def patched_emit(event_type: str, data: dict) -> None:
        global _last_fci, _last_neural_activity
        orig_emit(event_type, data)
        with _metrics_lock:
            if event_type == "training_step":
                _metrics["step"].append(data.get("iteration", 0))
                _metrics["reward"].append(data.get("mean_reward", 0))
                _metrics["policy_loss"].append(data.get("policy_loss", 0))
                _metrics["value_loss"].append(data.get("value_loss", 0))
                _metrics["entropy"].append(data.get("entropy", 0))
            elif event_type == "fci":
                _metrics["fci"].append(data.get("fci", 0))
                _last_fci = data.get("fci", 0)
            elif event_type == "eval":
                _metrics["value_ratio"].append(data.get("value_ratio", 0))
                _metrics["value_ratio_step"].append(
                    _metrics["step"][-1] if _metrics["step"] else 0
                )
                # Feature 1: capture previously-dropped eval fields
                _metrics["agreement_rate"].append(
                    data.get("agreement_rate", 0)
                )
                _metrics["policy_mean_money"].append(
                    data.get("policy_mean_money", 0)
                )
                _metrics["oracle_mean_money"].append(
                    data.get("oracle_mean_money", 0)
                )
            elif event_type == "decision_trace":
                # Feature 1: store latest decision trace
                _metrics["decision_trace"] = data.get("rounds", [])
            elif event_type == "neural_activity":
                # Feature 2: store layer-aggregated spike rates
                _metrics["neural_layers"] = data.get("layers", {})
                _metrics["neural_activity"] = data.get(
                    "mean_output_rate", 0.0
                )
                _last_neural_activity = data.get("mean_output_rate", 0.0)

    harness._emit = patched_emit

    try:
        harness.boot()
        while not stop.is_set():
            harness.train_iterations(4)
    except Exception:
        log.exception("training_thread_error")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _harness, _train_thread
    _harness = Harness()
    _stop_event.clear()
    _train_thread = threading.Thread(
        target=_training_loop, args=(_harness, _stop_event),
        daemon=True, name="fly-train",
    )
    _train_thread.start()
    log.info("dashboard.training_thread_started")
    yield
    _stop_event.set()
    if _train_thread:
        _train_thread.join(timeout=10)


app = FastAPI(title="FLY//ECON Dashboard", lifespan=_lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    plotly_js = _get_plotly_js()
    avatar_html = svg_to_html(
        fci=_last_fci,
        event="",
        neural_activity=_last_neural_activity,
    )
    return _build_html(plotly_js, avatar_html)


@app.get("/api/metrics")
async def api_metrics() -> dict[str, Any]:
    with _metrics_lock:
        return {
            "step": list(_metrics["step"]),
            "fci": list(_metrics["fci"]),
            "reward": list(_metrics["reward"]),
            "policy_loss": list(_metrics["policy_loss"]),
            "value_loss": list(_metrics["value_loss"]),
            "entropy": list(_metrics["entropy"]),
            "value_ratio": list(_metrics["value_ratio"]),
            "value_ratio_step": list(_metrics["value_ratio_step"]),
            "last_fci": _last_fci,
            "current_step": int(_metrics["step"][-1]) if _metrics["step"] else 0,
            "stage": _harness.mission_state.stage.value if _harness else "?",
            "neural_activity": _last_neural_activity,
        }


@app.get("/api/decisions")
async def api_decisions() -> dict[str, Any]:
    """Return the latest decision trace and action distribution."""
    with _metrics_lock:
        trace = list(_metrics["decision_trace"])
        # Compute action distribution from latest trace
        action_dist: dict[str, int] = {}
        for rd in trace:
            action = rd.get("policy_action", "")
            action_dist[action] = action_dist.get(action, 0) + 1
        return {
            "rounds": trace,
            "action_distribution": action_dist,
        }


@app.get("/api/neural")
async def api_neural() -> dict[str, Any]:
    """Return the latest layer-aggregated neural activity."""
    with _metrics_lock:
        return {
            "layers": dict(_metrics["neural_layers"]),
            "mean_output_rate": float(_metrics["neural_activity"]),
        }


def _build_html(plotly_js: str, avatar_html: str) -> str:
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FLY//ECON Live Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Courier New', monospace;
    background: #0a0a0a; color: #e0e0e0;
  }}
  .banner {{
    background: #1a1a2e; padding: 12px 20px;
    display: flex; flex-wrap: wrap; gap: 20px;
    border-bottom: 2px solid #16213e;
    align-items: center;
  }}
  .banner h1 {{ color: #00ff88; font-size: 1.4em; }}
  .banner .stat {{ color: #aaa; font-size: 0.9em; }}
  .banner .stat strong {{ color: #fff; }}
  .banner .stat .live {{ color: #00ff88; animation: blink 1s infinite; }}
  @keyframes blink {{ 50% {{ opacity: 0.4; }} }}
  .avatar-container {{
    display: flex; justify-content: center;
    padding: 20px 0;
    background: #0d0d0d;
    border-bottom: 1px solid #222;
  }}
  .chart-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px; padding: 12px;
  }}
  .chart-panel {{
    background: #111; border: 1px solid #333;
    border-radius: 6px; padding: 16px;
    min-height: 280px;
  }}
  .chart-panel h2 {{
    color: #00ff88; font-size: 1em;
    margin-bottom: 10px; border-bottom: 1px solid #333;
    padding-bottom: 6px;
  }}
  .full-width-panel {{
    background: #111; border: 1px solid #333;
    border-radius: 6px; padding: 16px;
    margin: 0 12px 12px 12px;
    min-height: 280px;
  }}
  .full-width-panel h2 {{
    color: #00ff88; font-size: 1em;
    margin-bottom: 10px; border-bottom: 1px solid #333;
    padding-bottom: 6px;
  }}
  /* ── Decision log table ──────────────── */
  .decision-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85em;
  }}
  .decision-table th {{
    background: #1a1a2e;
    color: #00ff88;
    padding: 6px 8px;
    text-align: left;
    border-bottom: 1px solid #333;
  }}
  .decision-table td {{
    padding: 5px 8px;
    border-bottom: 1px solid #222;
  }}
  .decision-table tr:hover {{ background: #1a1a1a; }}
  .match-yes {{ color: #00ff88; }}
  .match-no {{ color: #ff4444; }}
  @media (max-width: 800px) {{
    .chart-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
<script>{plotly_js}</script>
</head>
<body>

<div class="banner">
  <h1>FLY//ECON — LIVE</h1>
  <div class="stat"><span class="live">&#9679;</span> Step: <strong id="stat-step">0</strong></div>
  <div class="stat">FCI: <strong id="stat-fci">—</strong></div>
  <div class="stat">Reward: <strong id="stat-reward">—</strong></div>
  <div class="stat">Stage: <strong id="stat-stage">booting</strong></div>
</div>

<div class="avatar-container" id="avatar-panel">
{avatar_html}
</div>

<div class="chart-grid">
  <div class="chart-panel">
    <h2>Fly Confidence Index</h2>
    <div id="chart-fci" style="width:100%;height:240px;"></div>
  </div>
  <div class="chart-panel">
    <h2>Mean Reward per Step</h2>
    <div id="chart-reward" style="width:100%;height:240px;"></div>
  </div>
  <div class="chart-panel">
    <h2>Policy Loss</h2>
    <div id="chart-loss" style="width:100%;height:240px;"></div>
  </div>
  <div class="chart-panel">
    <h2>Value Ratio (Fly vs Oracle)</h2>
    <div id="chart-ratio" style="width:100%;height:240px;"></div>
  </div>
</div>

<!-- ══ Feature 1: Decision panels ══ -->
<div class="chart-grid">
  <div class="chart-panel">
    <h2>Decision Log (Latest MR12 Episode)</h2>
    <div id="decision-table-wrap" style="max-height:300px;overflow-y:auto;">
      <table class="decision-table">
        <thead><tr>
          <th>Round</th><th>Half</th><th>Money</th>
          <th>Fly Choice</th><th>Oracle Choice</th><th>Match</th>
        </tr></thead>
        <tbody id="decision-tbody">
          <tr><td colspan="6" style="color:#666;">Waiting for eval...</td></tr>
        </tbody>
      </table>
    </div>
  </div>
  <div class="chart-panel">
    <h2>Action Distribution</h2>
    <div id="chart-actions" style="width:100%;height:260px;"></div>
  </div>
</div>

<!-- ══ Feature 2: Connectome Neural Activity ══ -->
<div class="full-width-panel">
  <h2>Connectome Neural Activity</h2>
  <canvas id="neural-canvas" width="800" height="300"
          style="width:100%;height:300px;background:#0a0a0a;border-radius:4px;"></canvas>
</div>

<script>
(function() {{
  var dark = {{
    paper_bgcolor: '#111', plot_bgcolor: '#111',
    font: {{color: '#888', size: 11}},
    margin: {{t: 10, b: 35, l: 50, r: 10}},
  }};
  var lineConf = {{ width: 2 }};

  // ── Action color map ──
  var actionColors = {{
    'FULL_BUY': '#00ff88',
    'FORCE_BUY': '#ff8844',
    'HALF_BUY': '#ffcc00',
    'ECO': '#888888',
    'SAVE': '#ff4444'
  }};

  Plotly.newPlot('chart-fci', [{{x:[], y:[], type:'scatter', mode:'lines',
    fill:'tozeroy', line:Object.assign({{color:'#00ff88'}}, lineConf),
    fillcolor:'rgba(0,255,136,0.1)', name:'FCI'}}],
    Object.assign({{}}, dark, {{xaxis:{{title:'Step'}}, yaxis:{{title:'FCI'}}}}));

  Plotly.newPlot('chart-reward', [{{x:[], y:[], type:'scatter', mode:'lines',
    fill:'tozeroy', line:Object.assign({{color:'#4488ff'}}, lineConf),
    fillcolor:'rgba(68,136,255,0.1)', name:'Reward'}}],
    Object.assign({{}}, dark, {{xaxis:{{title:'Step'}}, yaxis:{{title:'Reward'}}}}));

  Plotly.newPlot('chart-loss', [{{x:[], y:[], type:'scatter', mode:'lines',
    fill:'tozeroy', line:Object.assign({{color:'#ff6644'}}, lineConf),
    fillcolor:'rgba(255,102,68,0.1)', name:'Loss'}}],
    Object.assign({{}}, dark, {{xaxis:{{title:'Step'}}, yaxis:{{title:'Loss'}}}}));

  Plotly.newPlot('chart-ratio', [{{x:[], y:[], type:'bar',
    marker:{{color:'#ffcc00'}}, name:'Ratio'}}],
    Object.assign({{}}, dark, {{xaxis:{{title:'Step'}}, yaxis:{{title:'Fly / Oracle'}}}}));

  Plotly.newPlot('chart-actions', [{{labels:[], values:[], type:'pie',
    marker:{{colors:[]}}, textfont:{{color:'#fff'}},
    hole: 0.35}}],
    Object.assign({{}}, dark, {{showlegend: true,
      legend: {{font: {{color: '#aaa', size: 10}}}}}})
  );

  // ── Neural canvas state ──
  var neuralSmoothed = {{}};
  var neuralLastTime = Date.now();
  var neuralLayers = ['Input', 'KC', 'MBON', 'DN', 'Other'];
  var neuralLayerColors = {{
    'Input': [51,68,85], 'KC': [0,255,136],
    'MBON': [255,204,0], 'DN': [255,102,68], 'Other': [136,136,136]
  }};
  // Initialize smoothed values
  for (var li = 0; li < neuralLayers.length; li++) {{
    neuralSmoothed[neuralLayers[li]] = 0.0;
  }}

  function drawNeural() {{
    var canvas = document.getElementById('neural-canvas');
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    var nCols = neuralLayers.length;
    var colW = W / (nCols + 1);
    var nodesPerCol = 8;
    var rowH = (H - 60) / nodesPerCol;

    // Draw connector ribbons between adjacent columns
    for (var c = 0; c < nCols - 1; c++) {{
      var x1 = (c + 1) * colW;
      var x2 = (c + 2) * colW;
      var rate1 = neuralSmoothed[neuralLayers[c]] || 0;
      var rate2 = neuralSmoothed[neuralLayers[c + 1]] || 0;
      var avgRate = (rate1 + rate2) / 2.0;
      var thickness = 1 + Math.min(avgRate * 3, 8);
      var alpha = 0.05 + Math.min(avgRate * 0.15, 0.3);
      ctx.beginPath();
      ctx.moveTo(x1 + 10, H / 2 - thickness);
      ctx.lineTo(x2 - 10, H / 2 - thickness);
      ctx.lineTo(x2 - 10, H / 2 + thickness);
      ctx.lineTo(x1 + 10, H / 2 + thickness);
      ctx.closePath();
      ctx.fillStyle = 'rgba(0,255,136,' + alpha.toFixed(2) + ')';
      ctx.fill();
    }}

    // Draw nodes per column
    for (var c = 0; c < nCols; c++) {{
      var layer = neuralLayers[c];
      var rate = neuralSmoothed[layer] || 0;
      var cx = (c + 1) * colW;
      var baseColor = neuralLayerColors[layer] || [136,136,136];

      for (var n = 0; n < nodesPerCol; n++) {{
        var cy = 30 + n * rowH + rowH / 2;
        // Radius proportional to rate
        var r = 2 + Math.min(rate * 3, 6);
        // Color: blend base toward bright with rate
        var bright = Math.min(rate * 2, 1.0);
        var cr = Math.round(baseColor[0] * (1 - bright) + baseColor[0] * bright);
        var cg = Math.round(baseColor[1] * (1 - bright * 0.3) + 255 * bright * 0.3);
        var cb = Math.round(baseColor[2] * (1 - bright) + baseColor[2] * bright);
        // Idle pulse: subtle opacity oscillation
        var t = Date.now() / 1000.0;
        var pulse = 0.9 + 0.1 * Math.sin(t * 2 + n * 0.5 + c * 1.3);

        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(' + cr + ',' + cg + ',' + cb + ',' + pulse.toFixed(2) + ')';
        ctx.fill();
      }}

      // Layer label
      ctx.fillStyle = '#666';
      ctx.font = '11px Courier New';
      ctx.textAlign = 'center';
      ctx.fillText(layer, cx, H - 8);
    }}

    requestAnimationFrame(drawNeural);
  }}
  requestAnimationFrame(drawNeural);

  // ── Decision table update ──
  function updateDecisionTable(rounds) {{
    var tbody = document.getElementById('decision-tbody');
    if (!tbody || !rounds || rounds.length === 0) return;
    var html = '';
    for (var i = 0; i < rounds.length; i++) {{
      var r = rounds[i];
      var matchClass = r.agreed ? 'match-yes' : 'match-no';
      var matchSymbol = r.agreed ? '&#10003;' : '&#10007;';
      html += '<tr>'
        + '<td>' + r.round + '</td>'
        + '<td>' + r.half + '</td>'
        + '<td>$' + r.money + '</td>'
        + '<td>' + r.policy_action + '</td>'
        + '<td>' + r.oracle_action + '</td>'
        + '<td class="' + matchClass + '">' + matchSymbol + '</td>'
        + '</tr>';
    }}
    tbody.innerHTML = html;
  }}

  // ── Action distribution chart update ──
  function updateActionChart(dist) {{
    if (!dist) return;
    var labels = [], values = [], colors = [];
    var keys = Object.keys(dist);
    for (var i = 0; i < keys.length; i++) {{
      labels.push(keys[i]);
      values.push(dist[keys[i]]);
      colors.push(actionColors[keys[i]] || '#888888');
    }}
    if (labels.length > 0) {{
      Plotly.react('chart-actions', [{{
        labels: labels, values: values, type: 'pie',
        marker: {{colors: colors}},
        textfont: {{color: '#fff'}},
        hole: 0.35
      }}], Object.assign({{}}, dark, {{
        showlegend: true,
        legend: {{font: {{color: '#aaa', size: 10}}}}
      }}));
    }}
  }}

  function poll() {{
    fetch('/api/metrics')
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        // Banner
        document.getElementById('stat-step').textContent = d.current_step;
        document.getElementById('stat-fci').textContent = d.last_fci.toFixed(4);
        document.getElementById('stat-stage').textContent = d.stage;
        if (d.reward.length > 0) {{
          document.getElementById('stat-reward').textContent = d.reward[d.reward.length-1].toFixed(4);
        }}

        // Update avatar
        if (typeof updateAvatar === 'function') {{ updateAvatar(d.last_fci); }}
        // Update avatar glow
        if (typeof updateAvatarActivity === 'function') {{
          updateAvatarActivity(d.neural_activity || 0);
        }}

        // Charts
        if (d.step.length > 0) {{
          Plotly.react('chart-fci', [{{
            x: d.step.slice(0, d.fci.length), y: d.fci,
            type:'scatter', mode:'lines', fill:'tozeroy',
            line:{{color:'#00ff88', width:2}}, fillcolor:'rgba(0,255,136,0.1)', name:'FCI'
          }}], Object.assign({{}}, dark, {{xaxis:{{title:'Step'}}, yaxis:{{title:'FCI'}}}}));

          Plotly.react('chart-reward', [{{
            x: d.step, y: d.reward,
            type:'scatter', mode:'lines', fill:'tozeroy',
            line:{{color:'#4488ff', width:2}}, fillcolor:'rgba(68,136,255,0.1)', name:'Reward'
          }}], Object.assign({{}}, dark, {{xaxis:{{title:'Step'}}, yaxis:{{title:'Reward'}}}}));

          Plotly.react('chart-loss', [{{
            x: d.step, y: d.policy_loss,
            type:'scatter', mode:'lines', fill:'tozeroy',
            line:{{color:'#ff6644', width:2}}, fillcolor:'rgba(255,102,68,0.1)', name:'Loss'
          }}], Object.assign({{}}, dark, {{xaxis:{{title:'Step'}}, yaxis:{{title:'Loss'}}}}));
        }}

        if (d.value_ratio.length > 0) {{
          var colors = d.value_ratio.map(function(v) {{
            return v > 1.0 ? '#00ff88' : '#ffcc00';
          }});
          Plotly.react('chart-ratio', [{{
            x: d.value_ratio_step, y: d.value_ratio,
            type:'bar', marker:{{color:colors}}, name:'Ratio'
          }}], Object.assign({{}}, dark, {{xaxis:{{title:'Eval Step'}}, yaxis:{{title:'Fly / Oracle'}}}}));
        }}
      }})
      .catch(function(err) {{ console.error('poll error', err); }});

    // Fetch decisions
    fetch('/api/decisions')
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        updateDecisionTable(d.rounds);
        updateActionChart(d.action_distribution);
      }})
      .catch(function(err) {{ console.error('decisions poll error', err); }});

    // Fetch neural activity
    fetch('/api/neural')
      .then(function(r) {{ return r.json(); }})
      .then(function(d) {{
        // Exponential smoothing for neural canvas
        var now = Date.now();
        var dt = (now - neuralLastTime) / 1000.0;
        neuralLastTime = now;
        var tau = 0.5;
        var alpha = 1.0 - Math.exp(-dt / tau);
        var layers = d.layers || {{}};
        for (var k in layers) {{
          if (neuralSmoothed[k] === undefined) neuralSmoothed[k] = 0;
          neuralSmoothed[k] += (layers[k] - neuralSmoothed[k]) * alpha;
        }}
      }})
      .catch(function(err) {{ console.error('neural poll error', err); }});
  }}

  // Poll every 1.5 seconds
  setInterval(poll, 1500);
  setTimeout(poll, 500);
}})();
</script>

</body>
</html>"""


if __name__ == "__main__":
    print("Dashboard running at http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)
