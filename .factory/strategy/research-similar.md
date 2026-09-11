# Research: Similar Projects — Live Neural + Economy Dashboard Prior Art

**Scope:** Targeted research for the single backlog item — enhancing
`dashboard_server.py` with (1) CS economy decision visualization, (2)
connectome neural activity visualization, (3) a better fly avatar with
glow/pulse tied to neural activity. Cross-referenced against
`study-combined.md`'s tap-point analysis (section 3) so every finding below
maps to a concrete file/function in this codebase.

**Methodology note:** No `WebSearch`/`WebFetch` tool was available in this
session. General search engines (DuckDuckGo, Bing) returned bot-detection
challenge pages / decoy results when scraped via `curl`, so I fetched the
user-cited reference directly (its actual source files, since it's an open
static HF Space) and used the GitHub public search API (`api.github.com`,
unauthenticated, rate-limited) for everything else. All links below were
verified to resolve; source code was read directly, not inferred.

---

## 1. `ngxson/fly-llm-demo` — the reference the user cited

**Link:** https://huggingface.co/spaces/ngxson/fly-llm-demo
**Source (it's a `static` HF Space, fully inspectable):**
https://huggingface.co/spaces/ngxson/fly-llm-demo/tree/main

**What it is:** "Fly brain LLM" — a language model whose transformer blocks
are replaced by the actual wiring diagram of a real fruit fly connectome
(49,393 central-brain neurons, 9M synapses, MaleCNS dataset). Each generated
token is injected into 14,069 sensory neurons; the frozen connectome mixes
state once per token (`x = 0.1x + 0.9·tanh(g(Wx+input)+b)`); a linear readout
over all neurons gives next-token logits. Runs entirely client-side
(onnxruntime-web, WebGPU/WASM).

### What it does well (the "good fly-brain viz" the user means)

1. **Real anatomical layout, not a force-directed graph.** Neuron soma
   `(x,y,z)` positions come straight from the connectome dataset
   (`positions.u16`, quantized to 16-bit per axis, decoded in `brain.js`
   against a bounding box). This is why it looks like an actual brain and
   not an abstract node-link diagram — **the single biggest differentiator**
   from generic "neural network visualizer" tools.
2. **GPU point-cloud rendering via a custom shader**, not per-neuron DOM/SVG
   elements. `three.js` `THREE.Points` + a hand-written vertex/fragment
   shader (`brain.js`) renders tens of thousands of neurons at 60fps: circular
   sprites via `discard` in the fragment shader based on distance-from-center,
   size and color driven by per-vertex attributes (`size`, `color`, `slot`,
   `isModel`). This is the only way to hit these neuron counts in-browser —
   directly relevant if we ever go beyond the ~200-neuron synthetic connectome
   in flyecon to the 2-10K mushroom-body or 166K full-connectome modes.
3. **Non-model neurons rendered too, just dimmed grey** (`GREY = [0.30, 0.33,
   0.38]`, alpha 0.35 vs 0.85 for model neurons). This gives brain-shape
   context "for free" without cluttering the signal — cheap trick worth
   stealing even at flyecon's small scale.
4. **Four selectable color modes** (`activation`, `class`, `hop`, `delta`) —
   warm/cool bipolar coloring for signed activation, categorical coloring by
   neuron superclass (sensory/intrinsic/ascending/descending), hop-distance-
   from-input coloring, and delta-magnitude coloring. Cheap to add (just a
   `paintNeuron()` branch) and gives the same data multiple useful lenses.
5. **The "sweep" animation is the killer trick, and it's fake in the best
   way.** The comment in `brain.js` is explicit: *"Animation is visual only:
   the real state update is instant, the colour sweep is ordered by the
   number of synapses from the input neurons so you can see where the signal
   enters and where it goes."* Concretely: `showStep(prev, next, durationMs)`
   interpolates each neuron's displayed value from `prev`→`next` on a
   per-neuron delay keyed by precomputed hop-distance (`hop.u8`, 0-3+), using
   a smoothstep easing window `[0.15+0.2h, 0.45+0.2h]` per hop `h`. Input
   neurons additionally get a `flashInputs()` pulse (bright slot-colored,
   oversized) during the first 30% of the animation. **This is exactly the
   "make it feel alive" technique we need for the connectome panel** —
   flyecon already computes a similar structural quantity implicitly via
   `map_to_connectome_inputs`/output-neuron distance, and even without true
   hop-precomputation, a fake fixed-duration "input flash → propagate outward
   → settle" tween is cheap to fake with existing FCI/spike-rate data.
6. **Data shipped as compact typed arrays, not JSON.** `positions.u16`,
   `superclass.u8`, `sub2global.i32`, `in_slot.u8`, `hop.u8` — all raw binary,
   fetched once, decoded into typed arrays client-side. Not directly
   applicable to flyecon's live-polling dashboard (payload is per-tick, not
   static), but the *lesson* transfers: keep the wire format for a
   per-neuron-type activity array as a flat `Float32Array`-friendly JSON list
   (e.g. `{types: [...], rates: [...]}`), not a heavier nested structure.

### What's missing / not applicable to flyecon

- No live/streaming update loop — it's a one-shot-per-token static compute,
  not a training dashboard. flyecon's polling architecture (1.5s interval)
  is a different problem shape; adopt the *rendering* technique, not the
  data-loading architecture.
- No economy/decision/game-state panel — out of scope for that demo,
  irrelevant to borrow beyond visual language.
- Requires full anatomical XYZ coordinates. flyecon's synthetic/small
  connectomes (per `etl/loader.py`) may not have this; **the "Panel 5"
  approach already stubbed in `renderer.py` (grouped bars by neuron type:
  KC/MBON/PPL/DN) is the right scope-appropriate analog** — a force/grid
  layout by neuron-type cluster instead of true anatomical coordinates is a
  reasonable middle ground if we want a "cloud" look without real 3D
  positions.

**Inspired-by chain worth knowing:** the README states it was "Inspired by
[Xenova/fruit-fly-simulation](https://huggingface.co/spaces/Xenova/fruit-fly-simulation),
which also provided the packaged connectome data" — see §2 below, which is
even more directly relevant to the avatar work.

---

## 2. `Xenova/fruit-fly-simulation` ("Neural Canvas") — rigged 3D fly + brain

**Link:** https://huggingface.co/spaces/Xenova/fruit-fly-simulation
**Source:** https://huggingface.co/spaces/Xenova/fruit-fly-simulation/tree/main

**What it is:** "Paint neurons, stimulate the network, and watch an
articulated Three.js fly respond." Full MaleCNS connectome (166,700 neurons,
25.6M directed connections) drives a Shiu-et-al.-style LIF model running on
WebGPU kernels (via `@huggingface/kernels`); computed firing rates feed a
hand-authored motion controller that animates a rigged NeuroMechFly-derived
3D fly body (real STL meshes: thorax, abdomen segments, wings, per-leg
segments — coxa/femur/tibia/tarsus 1-5 — with joint-limited inverse
kinematics and a crafted tripod gait).

### What it does well

1. **This is the "desktop-fly rigged insect 3D model" reference** — real
   NeuroMechFly body meshes (from NeLy-EPFL's neuromechanical fly model,
   attributed to https://github.com/NeLy-EPFL/fly-svg-maker in their README)
   with actual joint hierarchies (`src/body/fk.js` forward kinematics,
   `src/gait.js` tripod gait, `src/body/stl.js` mesh loading). This is the
   ceiling of "rigged insect avatar" sophistication — likely far beyond
   flyecon's scope (flyecon's avatar tiers are SVG/ASCII/simple-three.js, not
   anatomically rigged), but useful to know the state-of-the-art exists and
   to consciously scope down rather than accidentally under-shoot obvious
   wins.
2. **Neural-activity → motion mapping pattern (`controller.js`) is directly
   reusable as a design pattern**, even without the 3D rig:
   - Six decoded "rate" channels (`wl, wr, tl, tr, back, escape`) are
     exponentially relaxed toward raw neural input every frame:
     `relax(value, target, dt, tau) = value + (target - value) * (1 -
     exp(-dt/tau))` — a simple one-line low-pass filter that turns spiky
     neural output into smooth, physically plausible motion. **This exact
     `relax()` pattern is the missing piece in flyecon's avatar** — currently
     `svg.py::to_html()` takes a raw scalar `fci` with no smoothing, so any
     wired-in neural-activity signal will visibly jitter every 1.5s poll
     tick unless client-side JS applies the same kind of exponential
     smoothing before setting CSS vars.
   - A `groundSpeed = 8·tanh(walking/30) - 4·tanh(reverse/90)` style
     saturating nonlinearity turns unbounded neural firing rates into
     bounded, designer-tunable visual intensity — same idea as flyecon's own
     `_wing_beat_duration(fci)` tiered thresholds, just smoother (tanh vs.
     3-bucket if/else).
   - Discrete behavior states (`ground` / `escape` / `landing`) triggered by
     threshold-crossing + cooldown/hysteresis (`escapeArmed`, `cooldown`)
     avoid flickering between states — directly applicable to
     `ThreeJSAvatar`'s existing `STATE_POSE_MAP`/`_resolve_pose_name()`
     pattern (`threejs.py:24-113`) if we ever want debounced pose transitions
     tied to noisy neural-activity input.
3. **Procedural glow/shader techniques** (`fly-appearance.js`) show how to
   add cheap surface detail (hex-tiled compound-eye facets, procedural wing
   veins, bump-mapped cuticle noise) via `onBeforeCompile` GLSL injection on
   stock Three.js materials rather than hand-authored textures — a lighter-
   weight alternative to full custom shaders if the `ThreeJSAvatar` tier ever
   wants more visual fidelity.

### What's missing / not applicable

- Massive asset footprint (STL meshes, WebGPU kernel manifests, ~100s of
  files) — total overkill for flyecon's dashboard, which needs a lightweight
  glow/pulse on an existing SVG, not a new rigged body. **Do not port the
  rig** — port only the `relax()`/tanh-smoothing *pattern* from `controller.js`.
- No economy/game-state layer, no direct dashboard-polling relevance.

---

## 3. D3.js / Three.js neural & connectome visualizers (GitHub survey)

| Project | Stars | What it offers |
|---|---|---|
| [neuro-bit/Neural-Nematode](https://github.com/neuro-bit/Neural-Nematode) | — | Interactive D3+React visualization of the *C. elegans* connectome — closest same-domain-scale precedent (small, fully-mapped nervous system, ~300 neurons) to flyecon's synthetic/small connectome mode. Worth a closer look if we want a literal node-link graph instead of the point-cloud/bar-chart approach. |
| [SripadaLab/d3-connectomes](https://github.com/SripadaLab/d3-connectomes) | — | "Realtime visualization of connectome graphs" — validates that D3 force-directed graphs are a known-workable approach for connectome-scale (not just abstract NN) data, for teams that want live-updating node-link rather than point-cloud rendering. |
| [RichardGeorgeDavis/Neural-Network-Prototype](https://github.com/RichardGeorgeDavis/Neural-Network-Prototype) | 5 | "3D neural network brain built with three.js, shader-based rendering" — another data point for shader-based (not DOM-based) rendering being the standard approach once neuron counts exceed a few hundred. |
| [Carranza55/NeuroPudding](https://github.com/Carranza55/NeuroPudding) | — | "Live visualization and debugging tool for Spiking Neural Networks... attach to any SpikingJelly model with one line of Python and get a real-time dashboard of spike activity, neuron health, and emergent dynamics." **Closest architectural analog to what flyecon needs**: a Python-side hook that taps a live SNN and streams to a dashboard — same shape as the recommended `FlyPolicy.inspect()` tap point in `study-combined.md` §3.2. |
| [erojasoficial-byte/fly-brain](https://github.com/erojasoficial-byte/fly-brain) | 42 | Full embodied whole-brain FlyWire connectome sim (138,639 neurons) + **`brain_monitor.py`**: a pygame-based live neural monitor with explicit "sci-fi / Black Mirror aesthetic: gaussian glow, animated particles, dashed connections, hex grid, scanlines, **pulsing regions**." Also ships a `visualizer.py` matplotlib/Tkinter spike-raster + binned-firing-rate heatmap browser (neuron × time bins, sorted by total spike count, custom black→blue→cyan→white colormap). **This is the single most relevant prior-art for "glow/pulse tied to neural activity"** — see §5 below for direct technique extraction. |

**Cross-cutting takeaway:** every serious live-neural-activity dashboard in
this survey uses one of two rendering strategies — (a) shader-based GPU
point/particle clouds for >1K neurons, or (b) small-multiple bar/heatmap
panels aggregated by neuron type/region for anything meant to be readable at
a glance. Given flyecon's connectome sizes (200 synthetic / 2-10K MB / 166K
full) and that `dashboard_server.py` is a simple FastAPI+vanilla-JS+Plotly
stack (no Three.js dependency currently), **strategy (b) — a per-neuron-type
bar/heatmap panel, exactly as already stubbed in `renderer.py`'s Panel 5 — is
the pragmatic choice**, with the `ngxson` "sweep" easing trick borrowed for
liveliness rather than any true GPU particle system.

---

## 4. CS:GO / CS2 economy analysis & visualization tools

| Project | Stars | What it offers |
|---|---|---|
| [Twoos123/cs2-meta-engine](https://github.com/Twoos123/cs2-meta-engine) | 6 | Full pro-level CS2 demo analysis platform. **Its "Economy Tracker" module is a near-exact template for flyecon's CS Economy Decision Visualization**: (1) bar chart of T-vs-CT equipment value per round with hover details, (2) buy-type classification into **Eco / Force / Half / Full** badges from equipment thresholds — this is *literally* flyecon's `BuyPlan` enum (`FULL_BUY, FORCE_BUY, HALF_BUY, ECO, SAVE`, `state/economy.py:23-30`) with different naming, (3) a round-by-round table: winner, buy types, equipment values, cash spent per side, (4) consecutive **loss-bonus tracking** ($1400 base + $500/loss, capped $3400) — flyecon has the isomorphic concept in `EconomyState.loss_streak`/`opponent_loss_streak` and `BUY_PLAN_COSTS`. |
| [whuang37/csgo-market](https://github.com/whuang37/csgo-market) | 5 | Market-analysis-focused (skin economy, not round economy) — less directly relevant, but confirms "CS:GO economy" as a genre with existing tooling precedent worth citing for domain credibility. |

### Direct mapping: `cs2-meta-engine`'s Economy Tracker → flyecon's data model

| cs2-meta-engine concept | flyecon equivalent | Where to tap (per study-combined.md §3.1) |
|---|---|---|
| Equipment value bar (T vs CT) | `EconomyState.money` (policy) vs. Oracle's counterfactual money at the same state | `oracle.decide(state)` + `Policy.decide_and_value(state)` (`oracle/solver.py:57`), already exercised in `evaluate_against_oracle()`'s agreement loop (`ppo.py:396-401`) |
| Buy-type badge (Eco/Force/Half/Full) | `BuyPlan` enum directly | `state/economy.py:23-30`, `state/constants.py:94-100` (`BUY_PLAN_COSTS`) — zero translation needed, same vocabulary |
| Round-by-round table (winner, buy, cash) | Per-round `(state, policy_action, oracle_action, round, money)` tuple | New `"decision_trace"` telemetry event, per study-combined.md's recommendation #4 (medium-high effort) |
| Loss-bonus tracking | `loss_streak` / `opponent_loss_streak` fields | Already on `EconomyState`, just needs to be included in the emitted trace tuple |
| "Quick win" equivalent | Agreement rate / oracle-vs-policy money already computed but dropped | `harness.py:417-422` emits `agreement_rate`, `policy_mean_money`, `oracle_mean_money` in the `"eval"` event; `dashboard_server.py`'s `patched_emit` (lines 65-69) currently extracts only `value_ratio` and silently drops the other three — **this is a genuinely free win, no new computation, just stop discarding fields already in the payload** |

**Differentiation opportunity:** cs2-meta-engine's economy tracker shows two
independent teams' economies. flyecon's unique angle — which no CS economy
tool has, because no CS tool has an "oracle" — is showing the **policy's
choice next to the game-theoretically optimal choice at the identical
state**, i.e. a real-time agreement/disagreement highlight (already computed
via `evaluate_against_oracle`'s per-state comparison), not just two teams'
raw spend. This is flyecon's genuine novelty over all economy-viz prior art
surveyed and should be the headline of the economy panel, not an afterthought.

---

## 5. Reference Implementations — Direct Technique Extraction Per Feature

### 5.1 CS Economy Decision Visualization
- **Primary reference:** `Twoos123/cs2-meta-engine` Economy Tracker (bar +
  badge + table pattern, see §4 table above).
- **Concrete plan:** (a) quick win — surface the 3 already-computed-but-
  dropped `agreement_rate`/`policy_mean_money`/`oracle_mean_money` fields
  from the existing `"eval"` telemetry event (near-zero cost, per
  study-combined.md §3.1 item 1); (b) richer — new `/api/decisions` endpoint
  + Plotly bar/timeline of BuyPlan choices per round with a
  policy-vs-oracle agreement highlight column, using the exact color-coded
  badge idiom (Eco=grey/red, Force=orange, Half=yellow, Full=green) that
  cs2-meta-engine uses for buy-type classification.

### 5.2 Connectome Neural Activity Visualization
- **Primary reference:** `ngxson/fly-llm-demo`'s `brain.js` for the
  *animation/liveliness* technique (sweep-by-hop-distance tween,
  `flashInputs()` pulse on input neurons); `renderer.py`'s existing Panel 5
  stub (`x:['KC','MBON','DN','PPL'], y:[0,0,0,0]`) for the *scope-appropriate
  data shape* — small-multiple bar chart aggregated by neuron-type prefix,
  not a full 3D point cloud (flyecon lacks real anatomical XYZ coordinates
  for its synthetic connectome anyway).
- **Secondary reference:** `erojasoficial-byte/fly-brain`'s `visualizer.py`
  for a fallback/alternate rendering idea — binned firing-rate heatmap
  (neuron-type × recent-time-bins) with a custom low→high colormap — a
  richer alternative to a flat bar chart if there's appetite for a
  time-history view rather than a single live snapshot.
- **Concrete plan (per study-combined.md §3.2):** add `FlyPolicy.inspect()`
  method (opt-in, throttled) exposing `spike_counts` bucketed by
  `neuron_types` prefix via a new `Harness.connectome` read-only property;
  new `/api/neural` endpoint; render as a small-multiple bar/gauge panel;
  optionally borrow the ngxson "sweep" easing (CSS transition or JS
  requestAnimationFrame tween between polls) purely for visual liveliness
  even though the real update is instant, exactly as ngxson's own comment
  admits their propagation animation is.

### 5.3 Better Fly Avatar (glow/pulse tied to neural activity)
- **Primary reference:** `Xenova/fruit-fly-simulation`'s `controller.js`
  `relax()` exponential-smoothing pattern — apply this client-side (in the
  JS injected by `svg.py::to_html()`) to whatever neural-activity scalar
  `dashboard_server.py` starts polling, so glow/pulse intensity eases
  between polls instead of stepping abruptly every 1.5s.
- **Secondary reference:** `erojasoficial-byte/fly-brain`'s `brain_monitor.py`
  — "gaussian glow... pulsing regions" and its explicit CI (consciousness
  index) → color gradient (`black → blue → green → white`) is a directly
  reusable palette idea for FCI-driven body coloring, an upgrade over
  `svg.py`'s current 3-tier discrete `_resolve_body_color()` threshold
  scheme.
- **Concrete plan (per study-combined.md §3.3):** additive
  `to_html(fci, event, neural_activity=None)` param; SVG `<filter>`
  (`feGaussianBlur`+`feColorMatrix`) on the thorax/body group driven by a new
  `--glow-intensity`/`--pulse-duration` CSS var; new
  `window.updateAvatarActivity(activity)` JS setter mirroring the existing
  `updateAvatar(fci)`; smooth the incoming value with a `relax()`-style
  filter before applying to avoid jitter at the 1.5s poll cadence; initial
  data source can be the already-flowing `mean_value` metric (interim proxy)
  per study-combined.md, upgraded later to real per-neuron-type spike data
  once §5.2 is wired in.

---

## 6. Prior Knowledge (`.factory/archive/`)

**No archive available.** `.factory/archive/memory/*.md` exists but every
file returned "No results found" for both the task query and general
anti-pattern/failure queries — this appears to be the first cycle touching
this dashboard-visualization work; there is no prior cross-cycle knowledge to
reconcile against. `facts.md` was present but empty. Nothing to carry
forward or contradict.

---

## 7. Differentiation Opportunities — Summary

1. **Oracle-agreement overlay is flyecon's unique angle** — no surveyed CS
   economy tool has a game-theoretically optimal baseline to compare against
   in real time; this should be the headline feature of the economy panel,
   not a secondary metric (see §4).
2. **Honest FCI** — `study-combined.md` flags that FCI's "neural" input is
   currently a fake proxy (`kc_mean_rate=metrics["mean_reward"]*10`, not real
   KC firing rate). Wiring in real spike data (§5.2) doesn't just feed the
   avatar glow — it fixes a latent correctness gap in the dashboard's own
   headline metric. This is a second-order win worth flagging to the
   Strategist even though it's not one of the three literal ask items.
3. **Scope discipline vs. the reference projects:** both HF Spaces (ngxson,
   Xenova) are client-side, one-shot/static-connectome, no-training-loop
   demos with enormous asset pipelines (binary connectome dumps, WebGPU
   kernels, STL rigs). flyecon's dashboard is a *live training monitor* for
   a *small* connectome — the right move is to borrow rendering/animation
   *techniques* (sweep easing, exponential smoothing, glow filters, neuron-
   type aggregation) while explicitly not chasing their asset scale or
   physical realism. This should be stated plainly in the hypothesis so the
   Builder doesn't over-engineer toward a Three.js rig flyecon doesn't need.

## Recommended Focus Areas (ranked)

1. **Quick win, do first regardless of hypothesis scope:** surface
   `agreement_rate`/`policy_mean_money`/`oracle_mean_money` — already
   computed, already emitted, currently dropped by `dashboard_server.py`.
   Zero risk, immediate visible improvement to the economy panel.
2. **Neural activity panel** using neuron-type aggregation (KC/MBON/PPL/DN)
   — matches existing `renderer.py` Panel-5 design intent, has a ready data
   tap (`FlyPolicy.forward`'s discarded `spike_counts`), and directly enables
   #3.
3. **Avatar glow/pulse**, fed initially by `mean_value` as an interim proxy
   and upgraded to real spike aggregate once #2 lands — low risk, purely
   presentational, reuses the exact CSS-var/JS-setter pattern already proven
   by the existing `updateAvatar(fci)` wing-speed mechanism.
4. **Decision-trace panel** (round-by-round BuyPlan + oracle comparison) —
   highest value per §7.1 but also highest implementation cost (new
   sampling helper in `ppo.py`); sequence after 1-3 if the targeted-mode
   single-hypothesis budget requires picking one.
