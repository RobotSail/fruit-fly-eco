"""CS2 Esports Dashboard — FLY//ECON × CS2.

A CS2-themed esports dashboard that feeds economy states from 16,464 real
HLTV pro match rounds through a 164,587-neuron fly brain (FlyReservoir),
presenting the reservoir's buy-plan predictions alongside pro players'
actual decisions and Oracle-optimal strategy.
"""

import sys
from pathlib import Path

# Worktree import hack: flyecon/ lives in the parent repo, not this worktree.
_PARENT_REPO = Path(__file__).resolve().parent.parent.parent  # /home/osilkin/fruit-fly
if str(_PARENT_REPO) not in sys.path:
    sys.path.insert(0, str(_PARENT_REPO))

import json
import threading
from typing import Any

import numpy as np
import pandas as pd
import structlog
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Template
from scipy import sparse

from flyecon.avatar.svg import to_html as avatar_to_html
from flyecon.dashboard.fci import compute_fci
from flyecon.dashboard.renderer import _get_plotly_js
from flyecon.etl.controls import random_sparse
from flyecon.oracle.mdp import EconomyMDP
from flyecon.oracle.solver import solve as solve_mdp
from flyecon.reservoir import FlyReservoir
from flyecon.state.economy import BuyPlan, EconomyState

log = structlog.get_logger()

# ── BuyPlan colors (reuse established palette from prior dashboard cycle) ──
BUYPLAN_COLORS = {
    "FULL_BUY": "#00ff88",   # green — aggressive/confident
    "FORCE_BUY": "#ff8844",  # orange — risky
    "HALF_BUY": "#ffcc00",   # yellow — moderate
    "ECO": "#888888",        # grey — passive
    "SAVE": "#ff4444",       # red — danger/save
}

# ── CS2 side colors (T-orange / CT-blue) ──
SIDE_COLORS = {"t": "#de9b35", "ct": "#5e98d9"}

# ── Map display names ──
MAP_DISPLAY = {
    "de_ancient": "Ancient", "de_anubis": "Anubis", "de_dust2": "Dust II",
    "de_inferno": "Inferno", "de_mirage": "Mirage", "de_nuke": "Nuke",
    "de_overpass": "Overpass",
}

# ── Load parquet IMMEDIATELY at import time (tiny file, <1s) ──
_DATA_PATH = _PARENT_REPO / "data" / "cs2_pro_eval.parquet"
DF = pd.read_parquet(_DATA_PATH)

# Build match index for the match browser (available instantly, no reservoir needed)
MATCHES = []
for _match_id, _group in DF.groupby("match_id"):
    _teams = sorted(_group["team_name_resolved"].unique())
    _map_name = _group["map_name"].iloc[0]
    _n_rounds = len(_group) // 2  # two teams per round
    MATCHES.append({
        "match_id": str(_match_id),
        "teams": _teams,
        "team1": _teams[0] if len(_teams) > 0 else "?",
        "team2": _teams[1] if len(_teams) > 1 else "?",
        "map_name": _map_name,
        "map_display": MAP_DISPLAY.get(_map_name, _map_name),
        "n_rounds": _n_rounds,
    })
MATCHES.sort(key=lambda m: m["match_id"])


# ═══════════════════════════════════════════════════════════════════════════════
# Background thread — reservoir build + precomputation
# ═══════════════════════════════════════════════════════════════════════════════

_state_lock = threading.Lock()
_precomputed: dict[str, Any] = {"ready": False, "progress": "initializing"}


def _background_init():
    """Build reservoir, solve Oracle, precompute all predictions."""
    try:
        # Step 1: Build reservoir via random_sparse (NOT build_reservoir — it's BROKEN)
        with _state_lock:
            _precomputed["progress"] = "building reservoir (164,587 neurons)..."
        log.info("cs2dashboard.reservoir_building", n_neurons=164587)

        connectome = random_sparse(n_neurons=164587, density=0.0001, seed=42)

        # Fan-in normalize the weight matrix (mirrors _build_weight_matrix logic)
        W_torch = connectome.weight_matrix
        crow = W_torch.crow_indices().numpy()
        col = W_torch.col_indices().numpy()
        vals = W_torch.values().numpy().copy()
        n = W_torch.shape[0]
        W_scipy = sparse.csr_matrix((vals, col, crow), shape=(n, n))
        abs_row_sums = np.array(np.abs(W_scipy).sum(axis=1)).flatten()
        abs_row_sums = np.maximum(abs_row_sums, 1.0)
        inv_sums = sparse.diags(1.0 / abs_row_sums, format="csr")
        W_norm = (inv_sums @ W_scipy).astype(np.float32)

        reservoir = FlyReservoir(n_neurons=n, W_sparse=W_norm, device="cpu")
        log.info("cs2dashboard.reservoir_ready", n_neurons=n, nnz=W_norm.nnz)

        # Step 2: Solve Oracle MDP (pure NumPy, ~96,600 states, fast)
        with _state_lock:
            _precomputed["progress"] = "solving Oracle MDP..."
        mdp = EconomyMDP()
        oracle_policy = solve_mdp(mdp)
        log.info("cs2dashboard.oracle_ready", iterations=oracle_policy.iterations)

        # Step 3: Precompute fly + oracle predictions for all in-range rows
        with _state_lock:
            _precomputed["progress"] = "precomputing predictions..."

        results_by_match: dict[str, list[dict]] = {}
        total_agree_fly = 0
        total_agree_oracle = 0
        total_in_range = 0

        for match_id, match_df in DF.groupby("match_id"):
            match_id_str = str(match_id)
            match_rounds = []

            # Group by team for sequential reservoir processing
            for team_name, team_df in match_df.groupby("team_name_resolved"):
                team_rows = team_df.sort_values(["half", "round_in_half"])
                reservoir.reset()

                for _, row in team_rows.iterrows():
                    round_in_half = int(row["round_in_half"])
                    is_overtime = round_in_half > 12
                    pro_decision = str(row["buy_decision"])

                    round_data = {
                        "round_num": int(row["round_num"]),
                        "round_in_half": round_in_half,
                        "half": int(row["half"]),
                        "team": str(team_name),
                        "side": str(row["side"]),
                        "money": int(round(row["avg_balance"])),
                        "team_money": int(round(row["team_money"])),
                        "avg_equip": int(round(row["avg_equip"])),
                        "pro_decision": pro_decision,
                        "round_won": bool(row["round_won"]),
                        "loss_streak": int(row["loss_streak"]),
                        "opponent_loss_streak": int(row["opponent_loss_streak"]),
                        "is_overtime": is_overtime,
                        "fly_decision": None,
                        "oracle_decision": None,
                        "fly_agrees_pro": None,
                        "oracle_agrees_pro": None,
                        "fly_telemetry": None,
                    }

                    if not is_overtime:
                        # Build EconomyState — use avg_balance NOT team_money
                        econ_state = EconomyState(
                            money=int(round(row["avg_balance"])),
                            loss_streak=int(row["loss_streak"]),
                            round_number=round_in_half,
                            half=int(row["half"]),
                            opponent_loss_streak=int(row["opponent_loss_streak"]),
                        )

                        # Fly prediction
                        with torch.no_grad():
                            logits = reservoir.step(econ_state)
                            fly_idx = int(torch.argmax(logits).item())
                            fly_plan = list(BuyPlan)[fly_idx]
                        telem = reservoir.telemetry()

                        # Oracle prediction
                        oracle_plan = oracle_policy.decide(econ_state)

                        round_data["fly_decision"] = fly_plan.value
                        round_data["oracle_decision"] = oracle_plan.value
                        round_data["fly_agrees_pro"] = (fly_plan.value == pro_decision)
                        round_data["oracle_agrees_pro"] = (oracle_plan.value == pro_decision)
                        round_data["fly_telemetry"] = {
                            "state_rms": telem["state_rms"],
                            "n_active": telem["n_active"],
                        }

                        total_in_range += 1
                        if fly_plan.value == pro_decision:
                            total_agree_fly += 1
                        if oracle_plan.value == pro_decision:
                            total_agree_oracle += 1

                    match_rounds.append(round_data)

            # Sort rounds within match
            match_rounds.sort(key=lambda r: (r["half"], r["round_in_half"], r["team"]))
            results_by_match[match_id_str] = match_rounds

        # Compute aggregate stats
        fly_agreement = total_agree_fly / max(total_in_range, 1)
        oracle_agreement = total_agree_oracle / max(total_in_range, 1)

        # Per-map stats
        map_stats = {}
        for match_id_str, rounds in results_by_match.items():
            match_info = next((m for m in MATCHES if m["match_id"] == match_id_str), None)
            if not match_info:
                continue
            map_name = match_info["map_name"]
            if map_name not in map_stats:
                map_stats[map_name] = {"fly_agree": 0, "oracle_agree": 0, "total": 0}
            for r in rounds:
                if not r["is_overtime"] and r["fly_decision"]:
                    map_stats[map_name]["total"] += 1
                    if r["fly_agrees_pro"]:
                        map_stats[map_name]["fly_agree"] += 1
                    if r["oracle_agrees_pro"]:
                        map_stats[map_name]["oracle_agree"] += 1

        with _state_lock:
            _precomputed["ready"] = True
            _precomputed["progress"] = "ready"
            _precomputed["results"] = results_by_match
            _precomputed["fly_agreement"] = fly_agreement
            _precomputed["oracle_agreement"] = oracle_agreement
            _precomputed["total_in_range"] = total_in_range
            _precomputed["total_overtime"] = len(DF) // 2 - total_in_range
            _precomputed["map_stats"] = map_stats

        log.info("cs2dashboard.precompute_done",
                 total_rounds=total_in_range,
                 fly_agreement=f"{fly_agreement:.1%}",
                 oracle_agreement=f"{oracle_agreement:.1%}")

    except Exception:
        log.exception("cs2dashboard.background_init_failed")
        with _state_lock:
            _precomputed["progress"] = "error during initialization"


# START BACKGROUND THREAD AT MODULE IMPORT TIME
_bg_thread = threading.Thread(target=_background_init, daemon=True)
_bg_thread.start()


# ═══════════════════════════════════════════════════════════════════════════════
# Jinja2 Templates (NOT f-strings — CSS {} braces would cause KeyError)
# ═══════════════════════════════════════════════════════════════════════════════

MAIN_TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FLY//ECON × CS2 Dashboard</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #0a0a0a;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    min-height: 100vh;
  }

  .mono { font-family: 'Courier New', monospace; }

  /* ── Header banner ── */
  .header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-bottom: 2px solid #00ff88;
    padding: 1.2rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
  }
  .header h1 {
    color: #00ff88;
    font-size: 1.8rem;
    letter-spacing: 3px;
    text-transform: uppercase;
  }
  .header .stats-bar {
    display: flex;
    gap: 2rem;
    font-size: 0.85rem;
    color: #aaa;
  }
  .header .stats-bar .num { color: #00ff88; font-weight: bold; }

  /* ── Layout ── */
  .dashboard-grid {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 1rem;
    padding: 1rem;
    max-width: 1800px;
    margin: 0 auto;
  }

  /* ── Panels ── */
  .panel {
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    border-radius: 8px;
    padding: 1rem;
  }
  .panel h2 {
    color: #00ff88;
    font-size: 1rem;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid #2a2a4e;
    padding-bottom: 0.5rem;
  }

  /* ── Match browser ── */
  .match-list {
    max-height: 70vh;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #00ff88 #0a0a0a;
  }
  .match-item {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid #16213e;
    cursor: pointer;
    transition: background 0.15s;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .match-item:hover { background: #16213e; }
  .match-item.active { background: #16213e; border-left: 3px solid #00ff88; }
  .match-teams { font-size: 0.9rem; font-weight: 600; }
  .match-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    color: #888;
  }
  .map-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    color: #fff;
  }
  .map-de_inferno { background: #b33; }
  .map-de_mirage { background: #c90; }
  .map-de_dust2 { background: #a86; }
  .map-de_nuke { background: #369; }
  .map-de_anubis { background: #639; }
  .map-de_ancient { background: #396; }
  .map-de_overpass { background: #696; }

  /* ── Right panel ── */
  .right-panel {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  /* ── Avatar panel ── */
  .avatar-panel {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
  }
  .avatar-container {
    flex-shrink: 0;
  }
  .avatar-container .fly-svg {
    width: 120px;
    height: auto;
  }

  /* ── Status / Loading ── */
  .status-banner {
    background: #16213e;
    border: 1px solid #00ff88;
    border-radius: 6px;
    padding: 1rem;
    text-align: center;
  }
  @keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1.0; }
    100% { opacity: 0.6; }
  }
  .status-banner .loading-text {
    color: #00ff88;
    animation: pulse 1.5s ease-in-out infinite;
  }
  .spinner {
    display: inline-block;
    width: 1em;
    height: 1em;
    border: 2px solid #00ff8844;
    border-top-color: #00ff88;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 0.5em;
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* ── Round table ── */
  .round-table-container { overflow-x: auto; }
  .round-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }
  .round-table th {
    background: #16213e;
    color: #00ff88;
    padding: 0.5rem 0.6rem;
    text-align: left;
    font-weight: 600;
    position: sticky;
    top: 0;
    z-index: 1;
  }
  .round-table td {
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid #16213e;
  }
  .round-table tr:hover td { background: #0f0f23; }

  /* ── Buy plan chips ── */
  .bp-chip {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    font-weight: 700;
    font-size: 0.75rem;
    color: #000;
  }

  /* ── Side badge ── */
  .side-badge {
    display: inline-block;
    padding: 0.1rem 0.4rem;
    border-radius: 2px;
    font-weight: 700;
    font-size: 0.7rem;
    text-transform: uppercase;
  }
  .side-t { background: #de9b35; color: #000; }
  .side-ct { background: #5e98d9; color: #fff; }

  /* ── OT badge ── */
  .ot-badge {
    display: inline-block;
    padding: 0.1rem 0.4rem;
    border-radius: 2px;
    background: #555;
    color: #ccc;
    font-size: 0.7rem;
    font-weight: 600;
  }

  /* ── Agreement icons ── */
  .agree-yes { color: #00ff88; font-weight: bold; }
  .agree-no { color: #ff4444; }

  /* ── Charts ── */
  .charts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  .chart-box {
    background: #0f0f23;
    border-radius: 6px;
    padding: 0.5rem;
    min-height: 280px;
  }

  /* ── Round strip (HLTV-style) ── */
  .round-strip {
    display: flex;
    gap: 3px;
    flex-wrap: wrap;
    margin-bottom: 1rem;
  }
  .round-pip {
    width: 20px;
    height: 20px;
    border-radius: 3px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.55rem;
    font-weight: 700;
    transition: transform 0.1s;
  }
  .round-pip:hover { transform: scale(1.3); }
  .round-pip.t-win { background: #de9b35; color: #000; }
  .round-pip.ct-win { background: #5e98d9; color: #fff; }

  /* ── Aggregate stats footer ── */
  .aggregate-footer {
    background: #16213e;
    border-top: 2px solid #00ff88;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-around;
    flex-wrap: wrap;
    gap: 1rem;
  }
  .stat-card {
    text-align: center;
  }
  .stat-card .stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #00ff88;
  }
  .stat-card .stat-label {
    font-size: 0.75rem;
    color: #888;
    margin-top: 0.2rem;
  }

  .footnote {
    padding: 0.8rem 2rem;
    font-size: 0.7rem;
    color: #666;
    text-align: center;
    font-style: italic;
  }
</style>
</head>
<body>

<!-- ═══════════ HEADER BANNER ═══════════ -->
<div class="header">
  <h1>FLY//ECON × CS2</h1>
  <div class="stats-bar">
    <span><span class="num">{{ n_matches }}</span> matches</span>
    <span><span class="num">{{ "{:,}".format(n_neurons) }}</span> neurons</span>
    <span>Status: <span id="header-status" class="num">{{ status }}</span></span>
  </div>
</div>

<!-- ═══════════ MAIN GRID ═══════════ -->
<div class="dashboard-grid">

  <!-- LEFT: Match browser -->
  <div class="panel">
    <h2>Match Browser</h2>
    <div class="match-list" id="match-list">
      {% for m in matches %}
      <div class="match-item" data-match-id="{{ m.match_id }}" onclick="selectMatch('{{ m.match_id }}')">
        <div class="match-teams">{{ m.team1 }} vs {{ m.team2 }}</div>
        <div class="match-meta">
          <span class="map-badge map-{{ m.map_name }}">{{ m.map_display }}</span>
          <span>{{ m.n_rounds }} rounds</span>
        </div>
      </div>
      {% endfor %}
    </div>
  </div>

  <!-- RIGHT: Detail panel -->
  <div class="right-panel">

    <!-- Avatar + status -->
    <div class="panel avatar-panel">
      <div class="avatar-container">
        {{ avatar_html }}
      </div>
      <div>
        {% if not ready %}
        <div class="status-banner">
          <p class="loading-text"><span class="spinner"></span>Brain warming up...</p>
          <p style="color:#888;font-size:0.8rem;margin-top:0.5rem;" id="status-detail">{{ status }}</p>
        </div>
        {% else %}
        <div>
          <p style="color:#00ff88;font-weight:bold;">✓ Reservoir Ready</p>
          <p style="color:#888;font-size:0.8rem;">164,587-neuron connectome loaded</p>
        </div>
        {% endif %}
      </div>
    </div>

    <!-- Match detail area -->
    <div class="panel" id="match-detail">
      <h2>Select a match to explore</h2>
      <p style="color:#888;">Click a match from the browser panel to see round-by-round predictions.</p>
    </div>

    <!-- Charts -->
    <div class="charts-grid" id="charts-area" style="display:none;">
      <div class="chart-box" id="chart-economy"></div>
      <div class="chart-box" id="chart-decisions"></div>
    </div>

  </div>
</div>

<!-- ═══════════ AGGREGATE STATS FOOTER ═══════════ -->
{% if stats %}
<div class="aggregate-footer">
  <div class="stat-card">
    <div class="stat-value">{{ "%.1f"|format(stats.fly_agreement * 100) }}%</div>
    <div class="stat-label">Fly–Pro Agreement</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{{ "%.1f"|format(stats.oracle_agreement * 100) }}%</div>
    <div class="stat-label">Oracle–Pro Agreement</div>
  </div>
  <div class="stat-card">
    <div class="stat-value mono">{{ "{:,}".format(stats.total_in_range) }}</div>
    <div class="stat-label">Rounds Analyzed</div>
  </div>
  <div class="stat-card">
    <div class="stat-value mono">{{ "{:,}".format(stats.total_overtime) }}</div>
    <div class="stat-label">OT Rounds Excluded</div>
  </div>
</div>
{% endif %}

<div class="footnote">
  Reservoir Prediction uses an untrained forward pass through a 164,587-neuron
  random sparse connectome — agreement rate reflects random projection baseline
  (~20% expected for 5 balanced classes), not trained accuracy.
  Oracle uses MDP value-iteration optimal policy. OT rounds excluded from
  fly/oracle predictions (round_number &gt; 12 outside model range).
</div>

<!-- ═══════════ PLOTLY JS ═══════════ -->
<script>{{ plotly_js }}</script>

<!-- ═══════════ APP JS ═══════════ -->
<script>
(function() {
  var matchCache = {};
  var matchStatsCache = {};
  var ready = {{ "true" if ready else "false" }};
  var BUYPLAN_COLORS = {{ buyplan_colors_json }};

  function bpChip(decision) {
    if (!decision) return '<span class="ot-badge">OT</span>';
    var color = BUYPLAN_COLORS[decision] || '#888';
    return '<span class="bp-chip" style="background:' + color + ';">' + decision + '</span>';
  }

  function sideSpan(side) {
    var cls = side === 't' ? 'side-t' : 'side-ct';
    return '<span class="side-badge ' + cls + '">' + side.toUpperCase() + '</span>';
  }

  function agreeIcon(val) {
    if (val === null || val === undefined) return '—';
    return val ? '<span class="agree-yes">✓</span>' : '<span class="agree-no">✗</span>';
  }

  window.selectMatch = function(matchId) {
    // Highlight active
    document.querySelectorAll('.match-item').forEach(function(el) {
      el.classList.toggle('active', el.getAttribute('data-match-id') === matchId);
    });

    if (!ready) {
      document.getElementById('match-detail').innerHTML =
        '<h2>Match Selected</h2><p class="loading-text"><span class="spinner"></span>Waiting for brain to finish loading...</p>';
      return;
    }

    if (matchCache[matchId]) {
      renderMatch(matchId, matchCache[matchId]);
    } else {
      document.getElementById('match-detail').innerHTML =
        '<h2>Loading...</h2><p class="loading-text"><span class="spinner"></span></p>';
      fetch('/api/matches/' + matchId + '/rounds')
        .then(function(r) { return r.json(); })
        .then(function(data) {
          matchCache[matchId] = data;
          renderMatch(matchId, data);
        });
    }

    if (matchStatsCache[matchId]) {
      renderCharts(matchId, matchStatsCache[matchId]);
    } else {
      fetch('/api/matches/' + matchId + '/stats')
        .then(function(r) { return r.json(); })
        .then(function(data) {
          matchStatsCache[matchId] = data;
          renderCharts(matchId, data);
        });
    }
  };

  function renderMatch(matchId, rounds) {
    var detail = document.getElementById('match-detail');
    var teams = [];
    rounds.forEach(function(r) { if (teams.indexOf(r.team) === -1) teams.push(r.team); });

    // Round strip
    var strip = '<div class="round-strip">';
    var seenRounds = {};
    rounds.forEach(function(r) {
      var key = r.half + '-' + r.round_in_half;
      if (!seenRounds[key]) {
        seenRounds[key] = true;
        var cls = r.round_won ? (r.side === 't' ? 't-win' : 'ct-win') : (r.side === 't' ? 'ct-win' : 't-win');
        strip += '<div class="round-pip ' + cls + '">' + r.round_in_half + '</div>';
      }
    });
    strip += '</div>';

    // Table
    var html = '<h2>Match: ' + teams.join(' vs ') + '</h2>';
    html += strip;
    html += '<div class="round-table-container"><table class="round-table">';
    html += '<thead><tr><th>Rnd</th><th>Half</th><th>Team</th><th>Side</th>';
    html += '<th>Money ($)</th><th>Pro Decision</th><th>Fly Prediction</th>';
    html += '<th>Oracle Optimal</th><th>Fly=Pro</th><th>Oracle=Pro</th><th>Won</th></tr></thead>';
    html += '<tbody>';

    rounds.forEach(function(r) {
      html += '<tr>';
      html += '<td class="mono">' + r.round_in_half + '</td>';
      html += '<td class="mono">' + (r.half + 1) + '</td>';
      html += '<td>' + r.team + '</td>';
      html += '<td>' + sideSpan(r.side) + '</td>';
      html += '<td class="mono">$' + r.money.toLocaleString() + '</td>';
      html += '<td>' + bpChip(r.pro_decision) + '</td>';
      html += '<td>' + (r.is_overtime ? '<span class="ot-badge">OT</span>' : bpChip(r.fly_decision)) + '</td>';
      html += '<td>' + (r.is_overtime ? '<span class="ot-badge">OT</span>' : bpChip(r.oracle_decision)) + '</td>';
      html += '<td>' + (r.is_overtime ? '—' : agreeIcon(r.fly_agrees_pro)) + '</td>';
      html += '<td>' + (r.is_overtime ? '—' : agreeIcon(r.oracle_agrees_pro)) + '</td>';
      html += '<td>' + (r.round_won ? '<span class="agree-yes">W</span>' : '<span class="agree-no">L</span>') + '</td>';
      html += '</tr>';

      // Update avatar activity on last row
      if (r.fly_telemetry) {
        updateAvatarActivity(r.fly_telemetry.state_rms);
      }
    });

    html += '</tbody></table></div>';
    detail.innerHTML = html;
  }

  function renderCharts(matchId, stats) {
    var area = document.getElementById('charts-area');
    area.style.display = 'grid';

    var darkLayout = {
      paper_bgcolor: '#1a1a2e',
      plot_bgcolor: '#0f0f23',
      font: { color: '#e0e0e0', size: 11 },
      margin: { t: 40, r: 20, b: 40, l: 50 },
      legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#e0e0e0' } }
    };

    // Economy chart
    var ecoTraces = [];
    var teamColors = ['#de9b35', '#5e98d9'];
    var ti = 0;
    for (var team in stats.economy) {
      var pts = stats.economy[team];
      ecoTraces.push({
        x: pts.map(function(p) { return 'H' + (p.half+1) + 'R' + p.round; }),
        y: pts.map(function(p) { return p.money; }),
        name: team,
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: teamColors[ti % 2], width: 2 },
        marker: { size: 4 }
      });
      ti++;
    }
    Plotly.newPlot('chart-economy', ecoTraces, Object.assign({}, darkLayout, {
      title: { text: 'Economy Over Rounds', font: { color: '#00ff88' } },
      xaxis: { title: 'Round' },
      yaxis: { title: 'Avg Balance ($)' }
    }));

    // Decision distribution chart
    var decisions = ['FULL_BUY', 'FORCE_BUY', 'HALF_BUY', 'ECO', 'SAVE'];
    var decTraces = [];
    ['pro', 'fly', 'oracle'].forEach(function(who) {
      var counts = decisions.map(function(d) { return (stats.decision_distribution[who] || {})[d] || 0; });
      decTraces.push({
        x: decisions,
        y: counts,
        name: who.charAt(0).toUpperCase() + who.slice(1),
        type: 'bar'
      });
    });
    Plotly.newPlot('chart-decisions', decTraces, Object.assign({}, darkLayout, {
      title: { text: 'Decision Distribution', font: { color: '#00ff88' } },
      barmode: 'group',
      xaxis: { title: 'Buy Plan' },
      yaxis: { title: 'Count' }
    }));
  }

  // Status polling
  if (!ready) {
    var poll = setInterval(function() {
      fetch('/api/status')
        .then(function(r) { return r.json(); })
        .then(function(data) {
          var el = document.getElementById('status-detail');
          if (el) el.textContent = data.progress;
          document.getElementById('header-status').textContent = data.progress;
          if (data.ready) {
            clearInterval(poll);
            ready = true;
            location.reload();
          }
        });
    }, 500);
  }
})();
</script>

</body>
</html>
""")


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI app + routes
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="FLY//ECON × CS2 Dashboard")


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the main dashboard page."""
    with _state_lock:
        ready = _precomputed["ready"]
        status = _precomputed["progress"]
        stats = {
            "fly_agreement": _precomputed.get("fly_agreement", 0),
            "oracle_agreement": _precomputed.get("oracle_agreement", 0),
            "total_in_range": _precomputed.get("total_in_range", 0),
            "total_overtime": _precomputed.get("total_overtime", 0),
        } if ready else None

    plotly_js = _get_plotly_js()
    avatar_html = avatar_to_html(fci=0.5, event="idle", neural_activity=0.3)

    html = MAIN_TEMPLATE.render(
        plotly_js=plotly_js,
        avatar_html=avatar_html,
        matches=MATCHES,
        matches_json=json.dumps(MATCHES),
        ready=ready,
        status=status,
        stats=stats,
        buyplan_colors=BUYPLAN_COLORS,
        buyplan_colors_json=json.dumps(BUYPLAN_COLORS),
        side_colors=SIDE_COLORS,
        map_display=MAP_DISPLAY,
        n_matches=len(MATCHES),
        n_neurons=164587,
    )
    return HTMLResponse(content=html)


@app.get("/api/status")
def api_status():
    """Return background initialization status."""
    with _state_lock:
        return JSONResponse({"ready": _precomputed["ready"], "progress": _precomputed["progress"]})


@app.get("/api/matches")
def api_matches():
    """Return list of all matches (lightweight, no round data)."""
    return JSONResponse(MATCHES)


@app.get("/api/matches/{match_id}/rounds")
def api_match_rounds(match_id: str):
    """Return round-by-round data for a specific match."""
    with _state_lock:
        if not _precomputed["ready"]:
            return JSONResponse({"error": "still loading", "progress": _precomputed["progress"]}, status_code=503)
        rounds = _precomputed["results"].get(match_id, [])
    if not rounds:
        return JSONResponse({"error": "match not found"}, status_code=404)
    return JSONResponse(rounds)


@app.get("/api/stats")
def api_stats():
    """Return aggregate stats."""
    with _state_lock:
        if not _precomputed["ready"]:
            return JSONResponse({"error": "still loading"}, status_code=503)
        return JSONResponse({
            "fly_agreement": _precomputed["fly_agreement"],
            "oracle_agreement": _precomputed["oracle_agreement"],
            "total_in_range": _precomputed["total_in_range"],
            "total_overtime": _precomputed["total_overtime"],
            "map_stats": _precomputed["map_stats"],
        })


@app.get("/api/matches/{match_id}/stats")
def api_match_stats(match_id: str):
    """Return per-match aggregate for Plotly charts."""
    with _state_lock:
        if not _precomputed["ready"]:
            return JSONResponse({"error": "still loading"}, status_code=503)
        rounds = _precomputed["results"].get(match_id, [])
    if not rounds:
        return JSONResponse({"error": "match not found"}, status_code=404)

    # Compute per-match stats for charts
    teams = sorted(set(r["team"] for r in rounds))
    economy_data = {t: [] for t in teams}
    decision_dist = {"pro": {}, "fly": {}, "oracle": {}}
    agree_fly = agree_oracle = total = 0

    for r in rounds:
        economy_data.get(r["team"], []).append({"round": r["round_in_half"], "half": r["half"], "money": r["money"]})
        if not r["is_overtime"] and r["fly_decision"]:
            for key, field in [("pro", "pro_decision"), ("fly", "fly_decision"), ("oracle", "oracle_decision")]:
                val = r[field]
                decision_dist[key][val] = decision_dist[key].get(val, 0) + 1
            total += 1
            if r["fly_agrees_pro"]:
                agree_fly += 1
            if r["oracle_agrees_pro"]:
                agree_oracle += 1

    return JSONResponse({
        "economy": economy_data,
        "decision_distribution": decision_dist,
        "agreement": {
            "fly": agree_fly / max(total, 1),
            "oracle": agree_oracle / max(total, 1),
            "total": total,
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
