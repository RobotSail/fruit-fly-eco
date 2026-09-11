"""Dashboard renderer — single self-contained HTML via Jinja2 + Plotly.

Produces a 10-panel dashboard with uptime banner and inline plotly.js.
No external URLs — fully offline, fully self-contained.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Template

from flyecon.avatar.ascii import ASCIIAvatar
from flyecon.dashboard.telemetry import TelemetryEvent, TelemetryStore

log = structlog.get_logger()


# ── Minimal inline Plotly stub ───────────────────────────────────────────────
# In production this would be the full plotly.min.js (~3 MB) inlined.
# For initial builds we embed a stub that provides the Plotly.newPlot API
# with a no-op implementation so the HTML is valid and self-contained.
_PLOTLY_JS_STUB = """
/* Plotly.js stub — replace with full plotly.min.js for real charts */
var Plotly = {
  newPlot: function(id, data, layout) {
    var el = document.getElementById(id);
    if (el) {
      el.innerHTML = '<p style="color:#888;padding:1em;">Chart: ' + id + '</p>';
    }
  }
};
"""


def _get_plotly_js() -> str:
    """Return the plotly.js source to inline.

    Tries to use the installed plotly package's bundled JS; falls back
    to the stub if plotly is not installed or the bundle is missing.
    """
    try:
        import plotly  # noqa: F401
        from plotly.offline import get_plotlyjs

        return get_plotlyjs()
    except Exception:
        return _PLOTLY_JS_STUB


# ── HTML template ────────────────────────────────────────────────────────────
_DASHBOARD_TEMPLATE = Template(
    """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FLY//ECON Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Courier New', monospace;
    background: #0a0a0a; color: #e0e0e0;
  }
  .banner {
    background: #1a1a2e; padding: 12px 20px;
    display: flex; flex-wrap: wrap; gap: 20px;
    border-bottom: 2px solid #16213e;
    align-items: center;
  }
  .banner h1 { color: #00ff88; font-size: 1.4em; }
  .banner .stat { color: #aaa; font-size: 0.9em; }
  .banner .stat strong { color: #fff; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 12px; padding: 12px;
  }
  .panel {
    background: #111; border: 1px solid #333;
    border-radius: 6px; padding: 16px;
    min-height: 200px;
  }
  .panel h2 {
    color: #00ff88; font-size: 1em;
    margin-bottom: 10px; border-bottom: 1px solid #333;
    padding-bottom: 6px;
  }
  .panel-hero {
    grid-column: 1 / -1;
    text-align: center; min-height: 300px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
  }
  .panel-hero pre {
    font-size: 24px; line-height: 1.3;
    color: #00ff88;
  }
  .health-ok { color: #00ff88; }
  .health-warn { color: #ffcc00; }
  .health-dead { color: #ff4444; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #222; }
  th { color: #888; }
</style>
<script>
{{ plotly_js }}
</script>
</head>
<body>

<!-- Uptime Banner -->
<div class="banner">
  <h1>FLY//ECON</h1>
  <div class="stat">Mission Clock: <strong>{{ uptime }}</strong></div>
  <div class="stat">Ladder Level: <strong>{{ ladder_level }}</strong></div>
  <div class="stat">Candidates: <strong>{{ n_candidates }}</strong></div>
  <div class="stat">Cycles: <strong>{{ cycles_completed }}</strong></div>
  <div class="stat">Last Check-In: <strong>{{ last_checkin }}</strong></div>
  <div class="stat">FCI: <strong>{{ fci_value }}</strong></div>
</div>

<div class="grid">

  <!-- Panel 0: Behavioral Avatar (hero) -->
  <div id="panel-avatar" class="panel panel-hero">
    <h2>Panel 0 — Behavioral Avatar</h2>
    {{ avatar_html }}
  </div>

  <!-- Oracle Duel Visualization (stub) -->
  <div id="oracle-duel-stub" style="display:none;text-align:center;padding:10px;">
    <p style="color:#888;font-style:italic;">Oracle duel: two flies face across a table.</p>
  </div>

  <!-- Panel 1: FCI -->
  <div id="panel-fci" class="panel">
    <h2>Panel 1 — Fly Confidence Index</h2>
    <div id="fci-chart"></div>
    <p>Current FCI: {{ fci_value }} {{ fci_status }}</p>
  </div>

  <!-- Panel 2: Dopamine Ledger -->
  <div id="panel-dopamine" class="panel">
    <h2>Panel 2 — Dopamine Ledger</h2>
    <div id="dopamine-chart"></div>
    <p>Cumulative PPL dopamine-edge activation per round.</p>
  </div>

  <!-- Panel 3: Eco-Round Distress -->
  <div id="panel-distress" class="panel">
    <h2>Panel 3 — Eco-Round Distress Telemetry</h2>
    <div id="distress-chart"></div>
    <p>MBON activation variance during forced-save rounds.</p>
  </div>

  <!-- Panel 4: The Ladder -->
  <div id="panel-ladder" class="panel">
    <h2>Panel 4 — The Ladder</h2>
    <div id="ladder-chart"></div>
    <p>Loss-bonus ladder: current rung marked.</p>
  </div>

  <!-- Panel 5: Connectome ROI -->
  <div id="panel-roi" class="panel">
    <h2>Panel 5 — Connectome ROI</h2>
    <div id="roi-chart"></div>
    <p>Per-neuron-type spike rate / synapse count ratio.</p>
  </div>

  <!-- Panel 6: Subagent Sentiment -->
  <div id="panel-sentiment" class="panel">
    <h2>Panel 6 — Subagent Sentiment</h2>
    {% for comp in components %}
    <div>
      <span class="{{ comp.css_class }}">●</span>
      {{ comp.name }}: {{ comp.status }}
      <span style="color:#666">({{ comp.detail }})</span>
    </div>
    {% endfor %}
    {% if not components %}
    <p style="color:#666">No components reporting.</p>
    {% endif %}
  </div>

  <!-- Panel 7: Wall of Champions -->
  <div id="panel-champions" class="panel">
    <h2>Panel 7 — Wall of Champions</h2>
    <table>
      <tr><th>Candidate</th><th>Best Score</th><th>Step</th></tr>
      {% for c in champions %}
      <tr><td>{{ c.id }}</td><td>{{ c.score }}</td><td>{{ c.step }}</td></tr>
      {% endfor %}
      {% if not champions %}
      <tr><td colspan="3" style="color:#666">No champions yet.</td></tr>
      {% endif %}
    </table>
  </div>

  <!-- Panel 8: Grief Counsel -->
  <div id="panel-grief" class="panel">
    <h2>Panel 8 — Grief Counsel</h2>
    <p style="color:#888;font-style:italic">They are not gone. They are versioned.</p>
    <table>
      <tr><th>Candidate</th><th>Final Score</th><th>Retired At</th></tr>
      {% for r in retired %}
      <tr><td>{{ r.id }}</td><td>{{ r.score }}</td><td>{{ r.step }}</td></tr>
      {% endfor %}
      {% if not retired %}
      <tr><td colspan="3" style="color:#666">No retirements yet.</td></tr>
      {% endif %}
    </table>
  </div>

  <!-- Panel 9: Perpetuity Meter -->
  <div id="panel-perpetuity" class="panel">
    <h2>Panel 9 — Perpetuity Meter</h2>
    <div style="font-size:2em;color:#00ff88;text-align:center;padding:20px;">
      {{ uptime }}
    </div>
    <p style="text-align:center;color:#888">The mission does not sleep.</p>
  </div>

  <!-- Panel 10: Nanny Panel -->
  <div id="panel-nanny" class="panel">
    <h2>Panel 10 — Nanny Panel</h2>
    <table>
      <tr><td>Check-in interval</td><td>{{ nanny_interval_s }}s</td></tr>
      <tr><td>Last fire</td><td>{{ last_checkin }}</td></tr>
      <tr><td>Wakings delivered</td><td>{{ wakings }}</td></tr>
    </table>
  </div>

</div>

<script>
// Initialize Plotly charts (stubs with real structure)
Plotly.newPlot('fci-chart',
  [{x: {{ fci_times }}, y: {{ fci_values }}, type: 'scatter', mode: 'lines',
    line: {color: '#00ff88'}, name: 'FCI'}],
  {margin: {t:10,b:30,l:40,r:10}, paper_bgcolor:'#111', plot_bgcolor:'#111',
   font: {color:'#888'}, xaxis:{title:'Time'}, yaxis:{title:'FCI',range:[0,1]}}
);
Plotly.newPlot('dopamine-chart',
  [{x: {{ dopamine_x | tojson }}, y: {{ dopamine_ppl | tojson }}, type: 'scatter',
    mode: 'lines', line:{color:'#ff6644'}, name:'PPL Rate'},
   {x: {{ dopamine_x | tojson }}, y: {{ dopamine_gate | tojson }}, type: 'scatter',
    mode: 'lines', line:{color:'#44ff66', dash:'dash'}, name:'DA Gate', yaxis:'y2'}],
  {margin:{t:10,b:30,l:40,r:40}, paper_bgcolor:'#111', plot_bgcolor:'#111',
   font:{color:'#888'}, yaxis:{title:'PPL Spike Rate'},
   yaxis2:{title:'Gate Value', overlaying:'y', side:'right', range:[0.4,1.6]}}
);
Plotly.newPlot('distress-chart',
  [{y: [0], type: 'box', name: 'MBON Variance', marker:{color:'#ffcc00'}}],
  {margin:{t:10,b:30,l:40,r:10}, paper_bgcolor:'#111', plot_bgcolor:'#111',
   font:{color:'#888'}}
);
Plotly.newPlot('ladder-chart',
  [{x: ['$1400','$1900','$2400','$2900','$3400'],
    y: [1400,1900,2400,2900,3400], type: 'bar',
    marker:{color:['#333','#333','#333','#333','#333']}, name:'Ladder'}],
  {margin:{t:10,b:30,l:40,r:10}, paper_bgcolor:'#111', plot_bgcolor:'#111',
   font:{color:'#888'}}
);
Plotly.newPlot('roi-chart',
  [{x:['KC','MBON','DN','PPL'], y:[0,0,0,0], type:'bar',
    marker:{color:'#4488ff'}, name:'ROI'}],
  {margin:{t:10,b:30,l:40,r:10}, paper_bgcolor:'#111', plot_bgcolor:'#111',
   font:{color:'#888'}}
);
</script>

</body>
</html>
"""
)


def _format_uptime(seconds: float) -> str:
    """Format seconds as Xd Xh Xm Xs."""
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts: list[str] = []
    if d:
        parts.append(f"{d}d")
    if h or d:
        parts.append(f"{h}h")
    parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def _extract_dashboard_data(events: list[TelemetryEvent]) -> dict[str, Any]:
    """Extract structured data from telemetry events for template rendering."""
    fci_times: list[float] = []
    fci_values: list[float] = []
    components: list[dict[str, str]] = []
    champions: list[dict[str, Any]] = []
    retired: list[dict[str, Any]] = []
    uptime_s: float = 0.0
    ladder_level: int = 0
    n_candidates: int = 0
    cycles: int = 0
    last_fci: float = 0.5
    last_event: str = ""
    wakings: int = 0

    for ev in events:
        if ev.event_type == "fci":
            fci_times.append(ev.timestamp)
            val = ev.data.get("fci", 0.5)
            fci_values.append(val)
            last_fci = val

        elif ev.event_type == "mission_state":
            uptime_s = ev.data.get("uptime_seconds", uptime_s)
            ladder_level = ev.data.get("ladder_level", ladder_level)
            n_candidates = ev.data.get("n_candidates", n_candidates)
            cycles = ev.data.get("cycles_completed", cycles)

        elif ev.event_type == "component_health":
            status = ev.data.get("status", "UNKNOWN")
            css = "health-ok" if status == "ALIVE" else (
                "health-dead" if status == "DEAD" else "health-warn"
            )
            components.append({
                "name": ev.data.get("component", "?"),
                "status": status,
                "detail": ev.data.get("detail", ""),
                "css_class": css,
            })

        elif ev.event_type == "champion":
            champions.append({
                "id": ev.data.get("candidate_id", "?"),
                "score": ev.data.get("score", 0.0),
                "step": ev.data.get("step", 0),
            })

        elif ev.event_type == "retirement":
            retired.append({
                "id": ev.data.get("candidate_id", "?"),
                "score": ev.data.get("final_score", 0.0),
                "step": ev.data.get("step", 0),
            })

        elif ev.event_type == "economy_event":
            last_event = ev.data.get("event", "")

        elif ev.event_type == "waking":
            wakings += 1

    # Extract dopamine telemetry for real chart data
    dopamine_events = [e for e in events if e.event_type == "dopamine_activity"]
    dopamine_x = [f"Step {i}" for i in range(len(dopamine_events))]
    dopamine_ppl = [e.data.get("ppl_rate", 0) for e in dopamine_events]
    dopamine_gate = [e.data.get("dopamine_gate", 1.0) for e in dopamine_events]

    return {
        "fci_times": fci_times,
        "fci_values": fci_values,
        "components": components,
        "champions": champions,
        "retired": retired,
        "uptime_s": uptime_s,
        "ladder_level": ladder_level,
        "n_candidates": n_candidates,
        "cycles": cycles,
        "last_fci": last_fci,
        "last_event": last_event,
        "wakings": wakings,
        "dopamine_x": dopamine_x,
        "dopamine_ppl": dopamine_ppl,
        "dopamine_gate": dopamine_gate,
    }


def _build_avatar_html(
    fci: float,
    event: str,
    ladder_level: int,
    avatar_mode: str = "ascii",
) -> str:
    """Build the avatar HTML using the specified tier.

    Degrade chain: threejs → sprite → ascii.
    If a higher tier fails to import, falls back to the next.
    """
    if avatar_mode == "threejs":
        try:
            from flyecon.avatar.threejs import ThreeJSAvatar
            return ThreeJSAvatar().to_html(fci=fci, event=event)
        except Exception:
            avatar_mode = "sprite"  # fall through to sprite

    if avatar_mode == "sprite":
        try:
            from flyecon.avatar.sprite import SpriteAvatar
            return SpriteAvatar().to_html(fci=fci, event=event)
        except Exception:
            pass  # fall through to ascii

    # ASCII is the bottom tier — always works
    return ASCIIAvatar().to_html(
        fci=fci, event=event, ladder_level=ladder_level,
    )


def render_dashboard(
    telemetry: TelemetryStore,
    output_path: Path,
    n_events: int = 1000,
    avatar_mode: str = "ascii",
) -> Path:
    """Render the 10-panel dashboard as a single self-contained HTML file.

    Args:
        telemetry: TelemetryStore to read events from.
        output_path: Path to write the HTML file.
        n_events: Number of latest events to read.
        avatar_mode: Avatar tier — 'ascii', 'sprite', or 'threejs'.

    Returns:
        Path to the rendered HTML file.
    """
    events = telemetry.read_latest(n=n_events)
    data = _extract_dashboard_data(events)

    # Build avatar with degrade chain
    avatar_html = _build_avatar_html(
        fci=data["last_fci"],
        event=data["last_event"],
        ladder_level=data["ladder_level"],
        avatar_mode=avatar_mode,
    )

    # FCI status label
    fci = data["last_fci"]
    if fci > 0.7:
        fci_status = "🟢 Confident"
    elif fci > 0.4:
        fci_status = "🟡 Neutral"
    else:
        fci_status = "🔴 Distressed"

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = _DASHBOARD_TEMPLATE.render(
        plotly_js=_get_plotly_js(),
        uptime=_format_uptime(data["uptime_s"]),
        ladder_level=data["ladder_level"],
        n_candidates=data["n_candidates"],
        cycles_completed=data["cycles"],
        last_checkin=now_str,
        fci_value=f"{fci:.2f}",
        fci_status=fci_status,
        avatar_html=avatar_html,
        components=data["components"],
        champions=data["champions"],
        retired=data["retired"],
        fci_times=data["fci_times"],
        fci_values=data["fci_values"],
        dopamine_x=data["dopamine_x"],
        dopamine_ppl=data["dopamine_ppl"],
        dopamine_gate=data["dopamine_gate"],
        nanny_interval_s=60,
        wakings=data["wakings"],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    log.info("dashboard_rendered", path=str(output_path), events=len(events))
    return output_path
