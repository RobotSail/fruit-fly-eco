"""Live monitoring dashboard server — FastAPI + SSE + background training.

Run with::

    python dashboard_server.py

Opens http://localhost:8080 with a self-contained HTML page featuring an
animated SVG fruit fly avatar and real-time Plotly charts fed by SSE.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent

from flyecon.avatar.svg import to_html as svg_to_html
from flyecon.dashboard.renderer import _get_plotly_js
from flyecon.dashboard.telemetry import TelemetryStore
from flyecon.harness import Harness

log = structlog.get_logger()

# ── Global state shared between the lifespan and routes ──────────────────
_stop_event = threading.Event()
_harness: Harness | None = None
_train_thread: threading.Thread | None = None


def _training_loop(harness: Harness, stop: threading.Event) -> None:
    """Background training thread entry point."""
    import torch

    torch.set_num_threads(1)

    try:
        harness.boot()
        while not stop.is_set():
            harness.train_iterations(4)
    except Exception:
        log.exception("training_thread_error")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start the training thread on startup; join on shutdown."""
    global _harness, _train_thread  # noqa: PLW0603

    _harness = Harness()
    _stop_event.clear()

    _train_thread = threading.Thread(
        target=_training_loop,
        args=(_harness, _stop_event),
        daemon=True,
        name="fly-train",
    )
    _train_thread.start()
    log.info("dashboard.training_thread_started")

    yield  # ── app is running ──

    log.info("dashboard.shutting_down")
    _stop_event.set()
    if _train_thread is not None:
        _train_thread.join(timeout=10)
    log.info("dashboard.shutdown_complete")


app = FastAPI(title="FLY//ECON Dashboard", lifespan=_lifespan)


# ── GET / ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Return the self-contained live dashboard HTML page."""
    plotly_js = _get_plotly_js()
    avatar_html = svg_to_html(fci=0.5, event="")

    return _build_dashboard_html(plotly_js, avatar_html)


# ── GET /events (SSE) ───────────────────────────────────────────────────
@app.get("/events", response_class=EventSourceResponse)
async def events(request: Request) -> AsyncGenerator[ServerSentEvent, None]:
    """Stream telemetry events via SSE, tailing the JSONL by byte offset."""
    if _harness is None:
        return

    store: TelemetryStore = _harness._telemetry
    offset = 0

    while True:
        if await request.is_disconnected():
            break

        new_events, offset = store.read_from_offset(offset)
        for ev in new_events:
            yield ServerSentEvent(
                data=json.dumps(ev.data, default=str),
                event=ev.event_type,
                id=str(ev.timestamp),
            )

        await asyncio.sleep(0.5)


# ── GET /api/status ──────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status() -> dict[str, Any]:
    """Return current mission state as JSON."""
    if _harness is None:
        return {"stage": "starting", "training_step": 0}

    ms = _harness.mission_state
    return {
        "stage": ms.stage.value,
        "training_step": ms.training_step,
        "ladder_level": ms.ladder_level,
        "last_eval_score": ms.last_eval_score,
        "cycles_completed": ms.cycles_completed,
        "uptime_seconds": ms.uptime_seconds,
        "candidate_id": ms.candidate_id,
    }


# ── HTML builder ─────────────────────────────────────────────────────────
def _build_dashboard_html(plotly_js: str, avatar_html: str) -> str:
    """Build the self-contained dashboard HTML string.

    Inlines Plotly.js, SVG avatar, and EventSource JS — no external URLs.
    """
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
  <div class="stat">Step: <strong id="stat-step">0</strong></div>
  <div class="stat">FCI: <strong id="stat-fci">0.50</strong></div>
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
    <h2>Mean Reward</h2>
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
    font: {{color: '#888'}},
    margin: {{t: 10, b: 30, l: 50, r: 10}},
  }};

  Plotly.newPlot('chart-fci',
    [{{x: [], y: [], type: 'scatter', mode: 'lines',
       line: {{color: '#00ff88'}}, name: 'FCI'}}],
    Object.assign({{}}, dark, {{yaxis: {{title: 'FCI', range: [0, 1]}}}})
  );
  Plotly.newPlot('chart-reward',
    [{{x: [], y: [], type: 'scatter', mode: 'lines',
       line: {{color: '#4488ff'}}, name: 'Mean Reward'}}],
    Object.assign({{}}, dark, {{yaxis: {{title: 'Reward'}}}})
  );
  Plotly.newPlot('chart-loss',
    [{{x: [], y: [], type: 'scatter', mode: 'lines',
       line: {{color: '#ff6644'}}, name: 'Policy Loss'}}],
    Object.assign({{}}, dark, {{yaxis: {{title: 'Loss'}}}})
  );
  Plotly.newPlot('chart-ratio',
    [{{x: ['Value Ratio'], y: [0], type: 'bar',
       marker: {{color: '#ffcc00'}}, name: 'Ratio'}}],
    Object.assign({{}}, dark, {{yaxis: {{title: 'Ratio', range: [0, 1.5]}}}})
  );

  // ── SSE live updates ────────────────────────────────
  var step = 0;
  var es = new EventSource('/events');

  es.addEventListener('fci', function(e) {{
    var d = JSON.parse(e.data);
    var fci = d.fci || 0;
    document.getElementById('stat-fci').textContent = fci.toFixed(4);
    Plotly.extendTraces('chart-fci', {{x: [[step]], y: [[fci]]}}, [0], 500);
    if (typeof updateAvatar === 'function') {{ updateAvatar(fci); }}
  }});

  es.addEventListener('training_step', function(e) {{
    var d = JSON.parse(e.data);
    step = d.iteration || step + 1;
    document.getElementById('stat-step').textContent = step;
    var reward = d.mean_reward || 0;
    var loss = d.policy_loss || 0;
    Plotly.extendTraces('chart-reward', {{x: [[step]], y: [[reward]]}}, [0], 500);
    Plotly.extendTraces('chart-loss', {{x: [[step]], y: [[loss]]}}, [0], 500);
  }});

  es.addEventListener('eval', function(e) {{
    var d = JSON.parse(e.data);
    var ratio = d.value_ratio || 0;
    Plotly.react('chart-ratio',
      [{{x: ['Value Ratio'], y: [ratio], type: 'bar',
         marker: {{color: ratio > 0.8 ? '#00ff88' : '#ffcc00'}}}}],
      Object.assign({{}}, dark, {{yaxis: {{title: 'Ratio', range: [0, 1.5]}}}})
    );
  }});

  es.addEventListener('mission_state', function(e) {{
    var d = JSON.parse(e.data);
    document.getElementById('stat-stage').textContent = d.stage || '?';
  }});

}})();
</script>

</body>
</html>"""


if __name__ == "__main__":
    print("Dashboard running at http://localhost:8080")  # noqa: T201
    uvicorn.run(app, host="0.0.0.0", port=8080)
