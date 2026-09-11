---
tags: [factory, strategy, archived]
project: fruit-fly
date: 2026-09-11
source: factory-archivist
status: approved
---

# Three-Panel Dashboard Enhancement — Archived Strategy

**Approved for Implementation**

## Overview

Single backlog item: enhance `dashboard_server.py` with three interactive dashboard features:
1. **CS Economy Decision Visualization** — show fly's economy choices vs. oracle baseline
2. **Connectome Neural Activity Visualization** — layered neural pipeline diagram with real-time activity
3. **Better Fly Avatar with Glow/Pulse** — responsive glow keyed to neural activity

## Source Code Location

- Base codebase: `/home/osilkin/fruit-fly/.factory-worktrees/run-b7b1625d/`
- Files to copy: `dashboard_server.py`, `pyproject.toml`, entire `flyecon/` package
- Modified files: `dashboard_server.py`, `flyecon/harness.py`, `flyecon/policy/ppo.py`, `flyecon/avatar/svg.py`

## Current Dashboard State

- **4 flat scalar metrics**: FCI, reward, policy_loss, value_ratio
- **Avatar**: Hardcoded `fci=0.5` at page load, only wing-beat animation
- **Limitations**: No economy decisions visible, no neural activity, no dynamic glow

## Key Research Findings

### Data Sources (Already Exist!)
- `evaluate_against_oracle()` emits `agreement_rate`, `policy_mean_money`, `oracle_mean_money` → currently dropped by `patched_emit`
- `LIFNetwork.spike_counts` available at eval checkpoints → can sample via `policy.inspect()`
- SVG infrastructure ready for glow filters and animation

### Performance
- LIF simulate at 200 neurons: ~5.2ms per call → negligible at 1.5s poll rate
- Plotly.js already inlined (4.1MB) — pie/bar/scatter charts free
- Canvas 2D native — zero dependencies
- SVG filters native — zero dependencies

### Critical Constraints

**Thread Safety (MANDATORY)**
- `LIFNetwork` mutates `V`, `refractory`, `spike_counts` in-place
- ALL calls to `policy.forward()`, `policy.inspect()` MUST happen inside `_training_loop` thread
- NEVER call from FastAPI handlers or async tasks — causes silent training corruption
- Request handlers ONLY read from `_metrics_lock`-guarded plain Python values

**SVG Animation**
- Animate ONLY `opacity` and `transform` (compositor-friendly)
- NEVER animate `stdDeviation` or other filter attributes (forces main-thread repaints)
- Pre-filter elements, then animate the filtered composite

## Implementation Roadmap

### Feature 1: CS Economy Decision Visualization

**In `harness.py:train_iterations()` at eval checkpoints:**
- Play one full MR12 episode (2 halves × 12 rounds max)
- Collect per-round: `(round_number, half, money, policy_action_name, oracle_action_name, agreed: bool)`
- Emit as `"decision_trace"` event via `self._emit("decision_trace", {...})`

**In `dashboard_server.py:patched_emit()`:**
- For `"eval"` events: capture and store `agreement_rate`, `policy_mean_money`, `oracle_mean_money` (already computed, currently dropped)
- For `"decision_trace"` events: store rounds list in `_metrics["decision_trace"]`

**New endpoints & UI:**
- `GET /api/decisions` → latest trace + action distribution
- **Decision Log table**: Show latest MR12 episode (Round, Half, Money, Fly Choice, Oracle Choice, Match ✓/✗)
- **Action Distribution chart**: Plotly pie/bar showing BuyPlan frequency (ECO/FORCE_BUY/FULL_BUY/HALF_BUY/SAVE) with color coding

**BuyPlan colors:**
- `FULL_BUY`: #00ff88 (green)
- `FORCE_BUY`: #ff8844 (orange)
- `HALF_BUY`: #ffcc00 (yellow)
- `ECO`: #888888 (grey)
- `SAVE`: #ff4444 (red)

### Feature 2: Connectome Neural Activity Visualization

**In `ppo.py`:**
- Add `FlyPolicy.inspect(state: EconomyState) -> dict` method (non-mutating, `torch.no_grad()`)
- Returns: `spike_counts` tensor, action, value, features
- Dashboard-only tap point, called from training thread only

**In `harness.py`:**
- Add `@property connectome(self) -> Connectome` (read-only, mirrors existing `.policy` pattern)
- Add `neuron_layer_map(self) -> dict[int, str]` utility mapping neuron index → layer label
  - For real connectome: use `neuron_types` (KC → "KC", MBON → "MBON", DN → "DN", PPL → "PPL", else "Input")
  - For synthetic (n=200): index ranges → [0%, 20%] Input, [20%, 60%] KC, [60%, 80%] MBON, [80%, 90%] DN, [90%, 100%] Other
- At eval checkpoints: call `policy.inspect()`, aggregate spike rates by layer
- Emit as `"neural_activity"` event: `{"layers": {layer: rate, ...}, "mean_output_rate": float}`

**In `dashboard_server.py`:**
- Store layer dict and `mean_output_rate` in `_metrics_lock`-guarded state
- `GET /api/neural` → latest layer-aggregated rates dict

**UI: Canvas 2D Connectome Diagram**
- 5-column layered view (Input, KC, MBON, DN, Other)
- Nodes: circles, radius ∝ layer's mean spike rate (2–8px), color gradient (low=#334455 dark, high=#00ff88 green, max=#ffcc00 amber)
- Connections: thin translucent ribbons between adjacent layers
- Idle pulse: requestAnimationFrame with ±10% opacity oscillation
- Client-side exponential smoothing: `smoothed = smoothed + (target - smoothed) * (1 - Math.exp(-dt/tau))` with `tau=0.5s`
- Layer labels below each column

### Feature 3: Better Fly Avatar with Glow/Pulse

**In `svg.py:to_html()`:**
- Extend signature: `to_html(fci: float, event: str, neural_activity: float | None = None)` (backward-compatible)
- Increase scale: `viewBox="0 0 400 520"`, `width="400" height="520"`, all coords ~1.33x
- Add SVG `<filter id="neuralGlow">` in `<defs>`:
  - `<feGaussianBlur stdDeviation="8"/>`
  - `<feColorMatrix/>` (green-amber tint based on activity tier)
  - `<feMerge/>` composite
- Add glow halo `<ellipse>` behind thorax (same cx/cy, larger rx/ry, filtered, pre-rendered before body)
- CSS animation: `@keyframes pulseGlow` — opacity and transform only, with `--glow-pulse-duration` var
- Compute `--glow-intensity` (0.0–1.0) and `--glow-pulse-duration` from `neural_activity`:
  - High activity: intensity=0.9, duration=0.4s (fast)
  - Low: intensity=0.2, duration=3s (slow)
  - None: intensity=0.0 (hidden)

**In `dashboard_server.py`:**
- Pass `neural_activity=_last_neural_activity` (not hardcoded) to `svg.to_html()` in `index()` handler
- Also pass actual `_last_fci` instead of hardcoded 0.5
- In poll JS `poll()`: call `updateAvatarActivity(d.neural_activity)` alongside `updateAvatar(d.last_fci)`
- Include `neural_activity: _last_neural_activity` in `/api/metrics` response

**New JS function:**
- `window.updateAvatarActivity(activity)` — sets CSS vars, applies exponential smoothing client-side

## Critical Anti-patterns to Avoid

1. **Never call `policy`/`_lif`/`LIFNetwork` from FastAPI handlers** — causes silent training corruption via shared mutable tensor state. GIL does NOT prevent races on PyTorch in-place ops.
2. **Never animate SVG filter attributes** — `stdDeviation`, `feColorMatrix values` force main-thread repaints. Animate only `opacity` and `transform` on pre-filtered elements.
3. **Never send raw per-neuron arrays over `/api/neural`** — at 200 neurons OK, but at 166K = 3MB per poll. Always aggregate to layer means (5 scalars).
4. **Don't fix the FCI bug** — `kc_mean_rate = mean_reward * 10` is out of scope. Create independent, honest `neural_activity` signal instead.
5. **Don't introduce a second lock** — extend existing `_metrics_lock` dict with new keys.
6. **Don't modify fixed surfaces**: `LIFNetwork.step()`, `PPOTrainer.update()`, `EconomyMDP`, `eval/score.py`.
7. **Don't add external script references** — all JS inline in HTML f-string template (single self-contained document).

## Metrics Dict Extension

New keys to add to existing `_metrics` and protect with `_metrics_lock`:
- `"agreement_rate"` (float)
- `"policy_mean_money"` (float)
- `"oracle_mean_money"` (float)
- `"decision_trace"` (list of dicts: round, half, money, policy_action, oracle_action, agreed)
- `"neural_layers"` (dict: layer name → rate)
- `"neural_activity"` (float: mean output rate)

## Expected Impact

- **capability_surface**: +0.3 (three new interactive features)
- **observability**: +0.2 (neural activity and decision trace visibility)
- **eval score**: no movement expected (eval/score.py doesn't test dashboard_server.py)
- **User-visible**: Transforms "useless" dashboard into decision/neural/avatar feedback system

## Future Backlog Items

1. **Fix FCI three-part bug**: `kc_mean_rate`, `recent_reward`, `consecutive_wins` all derive from `mean_reward`. Requires new tracked state + threshold recalibration.
2. **Full-connectome dashboard support**: Handle 166K neurons (layer-only aggregation mandatory, throttle inspect frequency, cap payload).

## References

- cs2-meta-engine: economy decision visualization patterns
- ngxson/fly-llm-demo: neural sweep animation
- Xenova/fruit-fly-simulation: exponential smoothing (`relax()` pattern)
- Existing flyecon codebase: agreement loop pattern (ppo.py:393-409), emit pattern (harness.py:417-422), SVG design principles

---

**Archived**: 2026-09-11 by factory-archivist
**Status**: Ready for builder implementation
