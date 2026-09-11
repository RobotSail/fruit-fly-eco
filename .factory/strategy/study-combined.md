# Interaction Study — run-be7a5cbc

No interaction logs found.

## Similar Projects
No similar projects found.

## SPEC
No SPEC.md found. Run 'factory spec generate <path>' to generate one.

## Open GitHub Issues

### Your Issues (1) — actionable, may generate fix hypotheses

- **#1** Phase 1: Project scaffold + eval harness (by @RobotSail)
  > ## Phase 1: Project scaffold + eval harness  ### Description Bootstrap project layout, dependencies, and eval framework for PROJECT FLY//ECON.  ### Deliverables 1. Create the full project directory tree matching the 9-team architecture under flyecon/ 2. Create pyproject.toml with dependencies 3. Cre

## Backlog

**TARGETED MODE** — building exactly one item: The live dashboard server (dashboard_server.py) works but is useless. The user wants THREE specific things added: 1. CS ECONOMY DECISION VISUALIZATION 2. CONNECTOME NEURAL ACTIVITY visualization 3. BETTER FLY avatar with glow/pulse effects keyed to neural activity

- The live dashboard server (dashboard_server.py) works but is useless. The user wants THREE specific things added: 1. CS ECONOMY DECISION VISUALIZATION 2. CONNECTOME NEURAL ACTIVITY visualization 3. BETTER FLY avatar with glow/pulse effects keyed to neural activity

## Observability Coverage
- **Score:** 0.0%
- **Function coverage:** 0/0 functions have logging (0%)
- **Total log statements:** 0
- **Structured logging:** No
- **Request tracing:** No

### Observability Recommendations
- No source files found to analyze.

## Prior Knowledge (Obsidian)
No prior notes found.

## Hypothesis Budget

**TARGETED MODE — single-item budget**

**Backlog items: 1** (the focus target only)
**New items: at most 0** (do not add new items)
**Growth minimum: 0** (growth constraints suspended for targeted mode)

### Rules

- Generate exactly ONE hypothesis for the focus target.
- Do NOT clear other backlog items this cycle.
- Do NOT add new items.
- FEEC category still applies for classifying the single hypothesis.

## Memory Context (MemPalace)

## Episodic Memory (Task-Relevant)

  No results found for: "The live dashboard server (dashboard_server.py) works but is useless. The user wants THREE specific things added: 1. CS ECONOMY DECISION VISUALIZATION 2. CONNECTOME NEURAL ACTIVITY visualization 3. BETTER FLY avatar with glow/pulse effects keyed to neural activity"


## Past QA Findings

  No results found for: "The live dashboard server (dashboard_server.py) works but is useless. The user wants THREE specific things added: 1. CS ECONOMY DECISION VISUALIZATION 2. CONNECTOME NEURAL ACTIVITY visualization 3. BETTER FLY avatar with glow/pulse effects keyed to neural activity"


## Design Rationale

  No results found for: "The live dashboard server (dashboard_server.py) works but is useless. The user wants THREE specific things added: 1. CS ECONOMY DECISION VISUALIZATION 2. CONNECTOME NEURAL ACTIVITY visualization 3. BETTER FLY avatar with glow/pulse effects keyed to neural activity"


## Anti-Patterns & Past Failures

  No results found for: "failed reverted broken The live dashboard server (dashboard_server.py) works but is useless. The user wants THREE specific things added: 1. CS ECONOMY DECISION VISUALIZATION 2. CONNECTOME NEURAL ACTIVITY visualization 3. BETTER FLY avatar with glow/pulse effects keyed to neural activity"


## Knowledge Graph Facts


## Timeline


## Experiment Outcomes

  No results found for: "The live dashboard server (dashboard_server.py) works but is useless. The user wants THREE specific things added: 1. CS ECONOMY DECISION VISUALIZATION 2. CONNECTOME NEURAL ACTIVITY visualization 3. BETTER FLY avatar with glow/pulse effects keyed to neural activity"
# Graph Context — FLY//ECON Dashboard Enhancement

**Source of truth for code:** `/home/osilkin/fruit-fly/.factory-worktrees/run-b7b1625d/`
(this worktree, `run-be7a5cbc`, only has `README.md` on `main` — the graph.json
here is a 2-node stub. The real knowledge graph — 1,200 nodes / 2,347 edges,
built at commit `46302d0` — lives in the sibling worktree and was used for
all analysis below, cross-checked against direct source reads.)

**Focus target (from observations.md, TARGETED MODE):**
`dashboard_server.py` works but is "useless" — needs THREE enhancements:
1. CS Economy Decision Visualization
2. Connectome Neural Activity visualization
3. Better Fly avatar with glow/pulse tied to neural activity

---

## 1. Architectural Layers

The codebase is a strict 9-team pipeline, each team = one `flyecon/` subpackage.
Data flows in one direction (ETL → Sim → Encode → Policy → Readout → Oracle),
with the Harness as sole orchestrator and Dashboard as a pure read-only consumer.

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐
│ flyecon.etl │──▶│ flyecon.sim  │   │flyecon.state│──▶│flyecon.encoding│
│  loader.py  │   │   lif.py     │   │ economy.py  │   │ population.py │
│ subcircuit  │   │ calibration  │   │ constants.py│   └───────┬────────┘
└─────────────┘   └──────┬───────┘   └──────┬──────┘           │
                          │                  │                  │
                          ▼                  ▼                  ▼
                  ┌───────────────────────────────────────────────┐
                  │        flyecon.policy.ppo.FlyPolicy            │
                  │  encoder → map_to_connectome_inputs → LIF sim  │
                  │  → _extract_features → readout.W/b → logits    │
                  │  + value_head → values                         │
                  └───────────────────┬─────────────────────────────┘
                                       │
                       ┌───────────────┼────────────────┐
                       ▼                                 ▼
              ┌─────────────────┐             ┌───────────────────┐
              │ flyecon.oracle   │             │ PPOTrainer         │
              │  mdp.py/solver.py│◀── compared │ (rollouts, GAE,    │
              │ (value iteration)│  in eval    │  PPO update loop)  │
              └─────────────────┘             └─────────┬──────────┘
                                                          │
                                                          ▼
                                              ┌─────────────────────┐
                                              │  flyecon.harness     │
                                              │  Harness.boot()/     │
                                              │  train_iterations()  │
                                              │  — THE orchestrator  │
                                              └──────────┬───────────┘
                                                          │ _emit() →
                                                          ▼
                                        ┌───────────────────────────────┐
                                        │ flyecon.dashboard.telemetry    │
                                        │ TelemetryStore (JSONL append)  │
                                        └───────────────┬─────────────────┘
                                                          │ read_latest()/
                                                          │ read_from_offset()
                                    ┌─────────────────────┼─────────────────────┐
                                    ▼                                           ▼
                      ┌───────────────────────────┐                ┌────────────────────┐
                      │ flyecon.dashboard.renderer │                │ dashboard_server.py │
                      │ render_dashboard() → HTML  │                │ FastAPI live poller │
                      │ (10-panel static snapshot) │                │ (THE TARGET FILE)   │
                      └───────────────────────────┘                └──────────┬───────────┘
                                                                                │ imports
                                                                                ▼
                                                                   flyecon.avatar.svg.to_html()
```

**Key architectural fact:** there are **two separate dashboard code paths** that
currently diverge:
- `flyecon/dashboard/renderer.py` — the "real" 10-panel static HTML renderer,
  reads `TelemetryStore`, has named panel stubs for Connectome ROI (Panel 5),
  Dopamine Ledger (Panel 2), Eco-Round Distress (Panel 3) — **but these are all
  hardcoded zero-data placeholders** (`x:['KC','MBON','DN','PPL'], y:[0,0,0,0]`).
- `dashboard_server.py` — the LIVE FastAPI server (the actual target of this
  work). It does NOT use `renderer.py` at all. It hand-rolls its own minimal
  HTML in `_build_html()`, polls `/api/metrics` every 1.5s, and only surfaces
  4 flat scalar metrics (FCI, reward, policy_loss, value_ratio) plus the raw
  `svg_to_html(fci=0.5, event="")` avatar — called ONCE at page load with a
  **hardcoded fci=0.5**, never updated server-side. Client-side JS calls
  `updateAvatar(fci)` which only touches `--wing-beat-duration`.

This means: enhancing `dashboard_server.py` requires (a) new telemetry capture
in `_training_loop`'s `patched_emit`, (b) new `/api/*` endpoints, (c) new
client-side JS/HTML panels, and (d) avatar module changes — it is fully
decoupled from `renderer.py`/`_extract_dashboard_data()`, so no need to touch
the static-render path unless we want parity.

---

## 2. Entry Points & Hotspots (by graph degree centrality)

| Degree | Node | File | Role |
|---|---|---|---|
| 76 | `EconomyState` | `state/economy.py:L34` | Central data type — frozen dataclass, flows through encoder→policy→oracle→env-step |
| 57 | `Connectome` | `etl/loader.py:L46` | Connectome container — weight_matrix (CSR), neuron_types, body_ids |
| 51 | `PopulationEncoder` | `encoding/population.py:L62` | State→current-vector encoder (trainable) |
| 41 | `__main__` | `flyecon/__main__.py` | CLI entry point |
| 41 | `TelemetryStore` | `dashboard/telemetry.py:L55` | JSONL append/read — the only channel from Harness → Dashboard |
| 39 | `Harness` | `harness.py:L75` | **THE** orchestrator — single entry point for boot/train/eval/dashboard loop |
| 37 | `EconomyMDP` | `oracle/mdp.py:L124` | Oracle's ground-truth MDP (96,600 states) |
| 34 | `ASCIIAvatar` | `avatar/ascii.py:L86` | Bottom-tier avatar (always works) |
| 30 | `LIFNetwork` | `sim/lif.py:L41` | Spiking sim — holds `spike_counts` tensor |
| 30 | `ThreeJSAvatar` | `avatar/threejs.py:L385` | Top-tier avatar — has `STATE_POSE_MAP` w/ wing_freq/body_scale/head_yaw per pose (no neural-activity hook yet) |
| 29 | `FlyPolicy` | `policy/ppo.py:L80` | Encoder→LIF→Readout wrapper; **the tap point for both economy AND neural viz** |

`dashboard_server.py` itself is a relatively low-degree leaf node (imports
`Harness`, `svg_to_html`, `_get_plotly_js`) — it is the thinnest, least-invasive
place to add new read-only taps, which is good: changes here can't destabilize
the training loop as long as we only *read* from objects the Harness already
exposes (or extend `_emit`/`patched_emit` payloads, which is additive).

---

## 3. Data Tap Points Per Visualization

### 3.1 CS Economy Decision Visualization

**Goal:** show what buy-plan decisions the fly policy is making, state context,
and how it compares to Oracle's optimal choice — round-by-round.

**Data flow to tap:**
```
EconomyState (money, loss_streak, round_number, half, opponent_loss_streak)
   │
   ▼ PPOTrainer._step_env()          [ppo.py:214]
   │   picks BuyPlan via FlyPolicy.act(state) → action_idx
   │   affordability clamp: BUY_PLAN_COSTS[bp.value] > state.money → SAVE
   ▼
reward = (nxt.money - state.money) / MAX_MONEY
```

- **Currently NOT emitted anywhere.** `collect_rollouts()` builds a
  `RolloutBuffer` (states, actions, rewards, values, log_probs, dones,
  features) purely in-memory, per PPO iteration, then discards it after
  `update()`. Nothing about *which BuyPlan was chosen* or *what state it was
  chosen in* ever reaches telemetry today.
- **Oracle comparison already exists** but only as an aggregate:
  `evaluate_against_oracle()` (`ppo.py:368`) runs `n_episodes` complete MR12
  matches for policy vs Oracle independently and returns `EvalMetrics`
  (`policy_mean_money`, `oracle_mean_money`, `value_ratio`, `agreement_rate`).
  Note it does NOT do round-by-round side-by-side comparison — Oracle plays
  its own separate trajectory, not the same states as the policy. To get a
  true "per-round decision + oracle recommendation" comparison you'd want
  `oracle.decide(state)` called on the *same* state the policy just saw
  (this pattern already exists inside `evaluate_against_oracle`'s agreement
  loop at `ppo.py:396-401`: `dist.forward([st])`, `obp = oracle.decide(st)`).
- **Tap point recommendation:** Add a `decision` telemetry event type. Two
  options, from least to most invasive:
  1. **Cheapest / non-invasive:** In `dashboard_server.py`'s `patched_emit`,
     nothing new is available today because `Harness.train_iterations()`
     never surfaces per-step decisions — only aggregate `metrics` dict
     (`policy_loss`, `mean_reward`, etc. — see `ppo.py:330-338`) and,
     periodically, `EvalMetrics` via the `"eval"` event
     (`harness.py:417-422`, already captured by `dashboard_server.py:65-69`
     into `_metrics["value_ratio"]`). This is the **already-available tap**
     — `agreement_rate` and `policy_mean_money` vs `oracle_mean_money` are
     computed every `eval_interval` (default 50) iterations and are sitting
     unused in the eval payload (`harness.py:417-422`) — `dashboard_server.py`
     only currently extracts `value_ratio`, ignoring `agreement_rate`,
     `policy_mean_money`, `oracle_mean_money` that are already in the dict!
     **Quick win: surface these 3 already-computed-but-dropped fields.**
  2. **Richer / requires harness/PPO change:** Modify `PPOTrainer.collect_rollouts`
     or add a lightweight sampling hook that, once per N iterations, replays
     one pistol-round-to-round-12 trajectory using
     `FlyPolicy.forward([state])` + `oracle.decide(state)` on matched states
     (mirroring `evaluate_against_oracle`'s agreement loop but capturing the
     *sequence* of (state, policy_action, oracle_action, money, round) tuples)
     and emits it as a `"decision_trace"` telemetry event — this gives the
     dashboard enough to render a round-by-round bar/timeline of
     BuyPlan choices with oracle-agreement highlighting.
  - Either way, the **BuyPlan enum** (`state/economy.py:23-30`: FULL_BUY,
    FORCE_BUY, HALF_BUY, ECO, SAVE) and **BUY_PLAN_COSTS**
    (`state/constants.py:94-100`) are the vocabulary for rendering; `MDPState`
    conversion helpers (`oracle/mdp.py: economy_to_mdp_state`, `money_from_idx`)
    are available if bucketed/discretized state display is wanted.
  - `Policy.decide_and_value(state)` (`oracle/solver.py:57`) gives both the
    Oracle's action AND its value estimate for a state in one call — useful
    for a "confidence gap" style visualization (policy value vs oracle value
    at the same state).

**Mutable surfaces for this feature:** `dashboard_server.py` (`patched_emit`,
new `/api/decisions` endpoint, new chart panel), optionally
`flyecon/harness.py::train_iterations` (to emit richer eval data — low risk,
additive dict keys), optionally `flyecon/policy/ppo.py` (new helper function
for decision-trace sampling — should NOT touch `update()`/`collect_rollouts()`
hot path).

---

### 3.2 Connectome Neural Activity Visualization

**Goal:** show live spiking activity per neuron / neuron-layer (KC, MBON, PPL,
DN) as the fly "thinks" through a buy decision.

**Data flow to tap:**
```
FlyPolicy.forward(states)                              [ppo.py:123]
  │
  ├─▶ encoder.encode_batch(states)                      [population.py:175] → [batch, ~44]
  ├─▶ map_to_connectome_inputs(encoded, input_ids, N)    [population.py:190] → [batch, N] (full connectome width, mostly zero)
  ├─▶ with torch.no_grad():
  │     spike_counts = self._lif.batched_simulate(full_input, duration_ms)   [lif.py:190]
  │        → LIFNetwork.simulate() per batch item      [lif.py:161]
  │           → LIFNetwork.step() called n_steps=duration_ms/DT_MS times    [lif.py:106]
  │              → returns per-step bool spike mask, accumulates spike_counts[n_neurons]
  ├─▶ features = self._extract_features(spike_counts)   [ppo.py:114] — reads ONLY output_neuron_ids subset, baseline-normalized
  ├─▶ logits = F.linear(features, readout.W, readout.b)
  └─▶ values = value_head(features)
```

- **`spike_counts` is the raw per-neuron activity tensor** — shape
  `[n_neurons]` (or `[batch, n_neurons]` from `batched_simulate`), full
  connectome width (e.g. 200 synthetic / ~2-10K mushroom-body / 166,700 full).
  This is computed **inside `torch.no_grad()`** and is presently **discarded
  entirely** after `_extract_features()` slices out only the `output_neuron_ids`
  (≤20 by default, or all DN-class neurons in full-connectome mode via
  `_find_dn_neuron_indices` in `harness.py:56`). No neuron-level activity data
  reaches telemetry today — `FlyPolicy.forward()` returns
  `(Categorical, values, features)` — `features` are ALREADY
  baseline-normalized rates for output neurons only, not raw per-neuron
  spike counts, and not neuron-type-tagged.
- **Neuron-type metadata IS available** via `Connectome.neuron_types: dict[int,
  str]` (body_id → type label, e.g. "KC...", "MBON...", "PPL...", "DN...") and
  `Connectome.body_ids: np.ndarray` (sorted, maps dense index → body ID). To
  group spike activity by layer for visualization, you need:
  `dense_idx → body_id (via body_ids[idx]) → neuron_type (via neuron_types[body_id])`.
  Layer prefixes already used elsewhere in the code:
  `_MB_PREFIXES = ("KC", "MBON", "PPL")` (`etl/subcircuit.py:19`),
  `_DN_PREFIXES = ("DN",)` (`harness.py:53`). No single canonical mapping
  utility exists yet — would need to build a `neuron_id → layer_name` lookup
  table once at boot and reuse it.
- **`SpikeReadout.output_neuron_ids`** (registered buffer, `readout/linear.py:60-67`)
  and **`FlyPolicy.input_neuron_ids`** are the two neuron-index sets already
  tracked on the policy object — accessible as `policy.readout.output_neuron_ids`
  / `policy.input_neuron_ids` for highlighting "input" vs "output" neurons in
  a connectome graph view.
- **Baseline rates** (`SpikeReadout.baseline_rates`, computed once in
  `build_fly_policy` from a neutral 10.0-current probe, `ppo.py:468-471`) give
  a reference point for "how excited is this neuron relative to baseline" —
  same normalization already used for readout features, reusable for a
  "z-score heatmap" style neural activity view.
- **Tap point recommendation:** `FlyPolicy.forward()` is the single choke
  point where `spike_counts` exists but is thrown away. Cheapest non-invasive
  approach:
  1. Add an optional `return_spike_counts: bool = False` param to
     `FlyPolicy.forward()` (or a separate `FlyPolicy.act_verbose()` /
     `inspect()` method that duplicates the forward pass under `no_grad()`
     purely for dashboard display — does NOT touch the hot training path if
     kept as an additive, opt-in method called only from `dashboard_server.py`
     at a throttled interval (e.g. once every few seconds, not every PPO
     step), since a full LIF sim (`duration_ms/DT_MS` steps, default 400/5=80
     steps for a single state) is somewhat expensive to run at 1.5s poll
     rate but the state is fixed at buy-decision time so it's a single-item
     batch — cheap enough for periodic display.
  2. In `dashboard_server.py`, after boot, periodically call this
     `inspect()`-style method with `harness._policy` (readable via the
     already-imported `Harness` object, and the `policy` property is already
     public — `harness.policy` at `harness.py:538-541`) on a representative
     state (e.g. current or pistol-round state) to get fresh `spike_counts`,
     then bucket by neuron_type using `harness._connectome.neuron_types` (note:
     `_connectome` is currently private — would need a small getter, e.g.
     `Harness.connectome` property, mirroring the existing `.policy` property
     pattern at `harness.py:538-541`).
  3. Emit a `"neural_activity"` telemetry event (or serve directly via a new
     `/api/neural` FastAPI endpoint) with per-layer aggregate spike rates
     (mean rate per neuron-type prefix) plus optionally a downsampled
     per-neuron array for a heatmap/graph viz.
- **Layer-level aggregation** is the practical approach for large connectomes
  (166,700 neurons in full mode) — per-neuron rendering only makes sense for
  the synthetic (200) or mushroom-body (2K-10K) modes. Recommend aggregating
  by `neuron_types` prefix (KC/MBON/PPL/DN/other) → mean firing rate, matching
  the exact "Panel 5 — Connectome ROI" stub already sketched in
  `renderer.py:266-271` (`x:['KC','MBON','DN','PPL'], y:[0,0,0,0]`) — this
  panel design already exists conceptually, just needs real data wired in,
  and could be ported into `dashboard_server.py`'s custom HTML.

**Mutable surfaces for this feature:** `flyecon/policy/ppo.py` (additive
inspect/verbose-forward method on `FlyPolicy`, no change to `forward()`
signature needed if implemented as a separate method), `flyecon/harness.py`
(new `connectome` read-only property, mirroring `.policy`), `dashboard_server.py`
(new periodic sampling call + `/api/neural` endpoint + new chart panel).
**Do not modify** `LIFNetwork.step()`/`simulate()` internals — they are
performance/correctness-critical and already return everything needed
(`spike_counts` tensor) via existing return values.

---

### 3.3 Better Fly Avatar (glow/pulse tied to neural activity)

**Goal:** replace/extend the current wing-speed-only FCI mapping with
glow/pulse effects keyed to actual neural (spike) activity.

**Current interface** (`flyecon/avatar/svg.py`):
```python
def to_html(fci: float, event: str) -> str
```
- Pure function, no neural-activity parameter today. Takes only `fci` (float
  ∈[0,1]) and `event` (string, "reserved"/currently unused visually — see
  docstring at `svg.py:38-39`, confirmed unused in the body).
- Internally: `_wing_beat_duration(fci)` → CSS var `--wing-beat-duration`
  (0.06s @ fci=1 to 0.8s @ fci=0), and `_resolve_body_color(fci)` → 3-tier
  abdomen stripe color threshold (fci>0.7 dark goldenrod, >0.4 amber, else
  dull brown).
- Produces a self-contained `<style>+<svg>+<script>` string. The `<script>`
  defines `window.updateAvatar(fci)` which is called client-side from
  `dashboard_server.py`'s poll loop (`dashboard_server.py:252`:
  `if (typeof updateAvatar === 'function') { updateAvatar(d.last_fci); }`)
  — this is the **only** live-update path today; it just re-sets the CSS var.
  No color/glow/pulse update happens client-side after initial page load —
  `_resolve_body_color` only runs once, server-side, at page-render time
  with `fci=0.5` (dashboard_server.py:104) and is baked into the initial SVG
  fill color — never refreshed.
- **Sibling avatar tiers** (`ascii.py`, `sprite.py`, `threejs.py`) share the
  same `to_html(fci, event, ...)` calling convention — `ThreeJSAvatar` goes
  further with a `STATE_POSE_MAP` (wing_freq, body_scale, leg_phase, head_yaw
  per pose name) driven by `(fci, event)` via `_resolve_pose_name()`, but
  still has **no neural-activity input** — poses map only from FCI thresholds
  and discrete `event` strings (round_win, forced_eco, oracle_duel, etc, see
  `threejs.py:24-113`). `renderer.py::_build_avatar_html()` has a degrade
  chain (threejs → sprite → ascii) that `dashboard_server.py` does NOT use —
  it imports `svg.py` directly, bypassing the tier system entirely.
- **FCI itself** is computed in `flyecon/dashboard/fci.py::compute_fci()`
  from `(kc_mean_rate, recent_reward, consecutive_wins)` — note the parameter
  name `kc_mean_rate` implies KC (Kenyon cell) firing rate was intended as a
  neural-activity input to FCI, but in practice `harness.py:383-387` calls it
  with a **proxy**: `kc_mean_rate=max(metrics.get("mean_reward",0.0)*10, 0)`
  — i.e. it's NOT real KC firing rate, it's a scaled reward proxy! This is a
  latent bug/gap: the "neural" component of FCI is currently fake. If real
  spike data becomes available (per section 3.2 tap point), this is the
  natural place to wire it in for a more honest FCI, which would also
  organically improve the avatar's neural-groundedness even before touching
  `svg.py` itself.

- **Tap point recommendation for glow/pulse tied to neural activity:**
  1. Extend `svg.py::to_html()` signature additively:
     `to_html(fci: float, event: str, neural_activity: float | None = None)`
     — keep backward compatible default so `renderer.py`/other callers don't
     break.
  2. Add SVG `<filter>` (e.g. `feGaussianBlur` + `feColorMatrix` for glow) on
     the body/thorax group, driven by a new CSS var like
     `--glow-intensity`/`--pulse-duration` computed from `neural_activity`
     (e.g. mean output-neuron firing rate, normalized 0-1 — reuse the
     baseline-normalized `features` tensor already computed in
     `FlyPolicy.forward()`, see section 3.2 — `features.mean()` or `.abs().mean()`
     is a cheap scalar summary already sitting in memory after each `act()`
     call, no extra LIF sim needed if you tap the existing per-step `feat`
     returned by `PPOTrainer.collect_rollouts()`'s `self.policy.act(state)`
     call at `ppo.py:241`, OR the periodic `inspect()` call from 3.2).
  3. Add a `window.updateAvatarActivity(activity)` JS function analogous to
     the existing `updateAvatar(fci)` (`svg.py:287-291`) so
     `dashboard_server.py`'s poll loop can drive glow/pulse live without a
     full page reload — mirrors the exact pattern already in place for wing
     speed, just add a second CSS custom property and a second small JS
     setter.
  4. Simplest data source to start with: reuse `_last_fci`-style pattern —
     add `_last_activity` to `dashboard_server.py`'s global metrics state,
     populated either from the new neural-activity tap (3.2) or, as an
     interim/cheap proxy, from `metrics.get("mean_value")` (already computed
     every PPO update at `ppo.py:338`, already flows into
     `_metrics_lock`-guarded dict) until real per-neuron data is wired.

**Mutable surfaces for this feature:** `flyecon/avatar/svg.py` (additive
param + new CSS/SVG filter + new JS setter — low risk, pure presentational),
`dashboard_server.py` (new metric key + pass-through to `updateAvatarActivity`
in the poll callback). No changes needed to `ascii.py`/`sprite.py`/`threejs.py`
unless parity across tiers is desired (out of scope for `dashboard_server.py`
since it hardcodes `svg.py` directly, bypassing the tier system).

---

## 4. Summary — Recommended Implementation Order (least → most invasive)

1. **Quick win (economy viz, ~free):** Surface `agreement_rate`,
   `policy_mean_money`, `oracle_mean_money` from the existing `"eval"`
   telemetry event — they're already computed in `evaluate_against_oracle()`
   and already emitted by `harness.py:417-422`, just dropped by
   `dashboard_server.py`'s `patched_emit` (`dashboard_server.py:65-69`).
2. **Avatar glow (low risk):** Additive `neural_activity` param on
   `svg.py::to_html()` + new CSS var + new JS setter, fed initially by
   `mean_value` (already flowing) as a stand-in signal, upgradeable later to
   real spike data.
3. **Neural activity panel (medium):** Add `Harness.connectome` read-only
   property + `FlyPolicy` verbose-inspect method that exposes `spike_counts`
   bucketed by `neuron_types` prefix (KC/MBON/PPL/DN/other); new
   `/api/neural` endpoint + panel in `dashboard_server.py`; throttle sampling
   (not every poll) since it runs a real LIF simulation.
4. **Decision trace panel (medium-high):** New helper in `ppo.py` that plays
   one full episode logging `(state, policy_action, oracle_action, round,
   money)` tuples per round; emit as `"decision_trace"` telemetry event;
   render as a timeline/table in `dashboard_server.py`.

All four items are additive to existing modules — none require modifying
`LIFNetwork.step()`, `PPOTrainer.update()`, `EconomyMDP`/`solver.py`
(training/oracle correctness-critical code), keeping risk low while directly
serving the three requested visualizations.
