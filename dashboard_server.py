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
_metrics: dict[str, collections.deque] = {
    "step": collections.deque(maxlen=1000),
    "fci": collections.deque(maxlen=1000),
    "reward": collections.deque(maxlen=1000),
    "policy_loss": collections.deque(maxlen=1000),
    "value_loss": collections.deque(maxlen=1000),
    "entropy": collections.deque(maxlen=1000),
    "value_ratio": collections.deque(maxlen=100),
    "value_ratio_step": collections.deque(maxlen=100),
}
_last_fci: float = 0.0


def _training_loop(harness: Harness, stop: threading.Event) -> None:
    import torch
    torch.set_num_threads(1)

    global _last_fci

    orig_emit = harness._emit

    def patched_emit(event_type: str, data: dict) -> None:
        global _last_fci
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
    avatar_html = svg_to_html(fci=0.5, event="")
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

<script>
(function() {{
  var dark = {{
    paper_bgcolor: '#111', plot_bgcolor: '#111',
    font: {{color: '#888', size: 11}},
    margin: {{t: 10, b: 35, l: 50, r: 10}},
  }};
  var lineConf = {{ width: 2 }};

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

        // Update avatar wing speed
        if (typeof updateAvatar === 'function') {{ updateAvatar(d.last_fci); }}

        // Charts — use Plotly.react for full re-render each poll
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
