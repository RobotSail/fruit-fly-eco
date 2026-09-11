# Research Report — Tech Stack for Dashboard Visualization Enhancements

## Context

Targeted-mode research for `dashboard_server.py` (FastAPI + poll-based live
training dashboard for FLY//ECON). Per `study-combined.md` / `graph-context.md`,
the backlog item requires three additive features:

1. CS Economy Decision Visualization (pie/bar/timeline of BuyPlan choices vs
   Oracle)
2. Connectome Neural Activity visualization (per-layer/per-neuron spike rate,
   200–10K nodes depending on connectome mode)
3. Fly avatar glow/pulse tied to neural activity

**Hard constraint (factory guard):** the dashboard must remain a **single
self-contained HTML file** — everything (CSS, JS, chart library code) must be
string-inlined into the FastAPI response, exactly as `dashboard_server.py`
already does today by inlining the full `plotly.js` bundle via
`_get_plotly_js()` / `plotly.offline.get_plotlyjs()`.

I verified the current inlining baseline directly in this environment:

| Asset already inlined today | Size (minified, uncompressed) |
|---|---|
| `plotly.js` full bundle (`get_plotlyjs()`, plotly 7.0.0, confirmed by running it) | **4,192.7 KB (~4.1 MB)** |
| Hand-written `svg.py` fly avatar (SVG+CSS+JS, zero deps) | ~7 KB |

This is the single most important fact for scoping this research: **the
project already ships a 4+ MB inlined JS blob on every page load.** Bundle
size is not a meaningful constraint for anything discussed below — every
option considered here is 1–2 orders of magnitude smaller than what's already
being inlined.

---

## 1. Neural Activity Visualization: D3 vs Canvas vs WebGL

### What's actually being rendered

Per `graph-context.md` §3.2, the practical data shape is **not** an arbitrary
graph-layout problem — it's a **fixed 4–5 column layered diagram**
(Input → KC → MBON → PPL → DN), where node **position is determined by layer
membership**, not by graph topology. Node count depends on connectome mode:
synthetic (~200), mushroom-body (~2K–10K), full (~166,700, but the graph
context doc explicitly recommends **layer-aggregation only** at that scale —
not per-neuron rendering). So the real design target is: **render up to ~10K
positioned nodes + their edges, at 60fps, with color/size bound to a
scalar (spike rate) that updates every 1.5–5s.**

### Rendering technology comparison

| Approach | Node capacity @60fps | Bundle cost to inline | Fit for "layered" (not organic) layout | Verdict |
|---|---|---|---|---|
| **D3 + SVG DOM** (one `<circle>`/`<line>` per neuron, D3 handling data-join) | SVG DOM rendering degrades sharply past ~1,000–3,000 live elements because every node is a real DOM node subject to style recalc/layout/paint on each update. This is precisely why D3-ecosystem force-graph libraries deliberately moved to canvas. | ~45 KB (d3-selection+d3-transition only, no d3-force needed) | Fine, but node count ceiling makes it unsuitable at the 2K–10K end of the stated range | Only acceptable for the 200-node synthetic connectome; will visibly stutter at 2K–10K |
| **Canvas 2D, hand-rolled (no library)** | Comfortably handles tens of thousands of simple shapes per frame — canvas draws raw pixels with no per-element DOM/layout overhead. Reference: `vasturiano/force-graph` (canvas+d3-force) ships a public demo at **~75,000 elements** running smoothly in-browser (verified via its README example list, fetched directly from GitHub). 10K nodes + a few thousand edges is well inside this library's comfortably-supported range, and is trivial for raw Canvas 2D with no framework at all. | **0 KB** if hand-rolled directly against the native Canvas API (exactly the same "zero-dependency, hand-write it" philosophy the project already uses for `svg.py`) | Best fit — deterministic column layout (x = layer index, y = sorted position within layer) is a ~30-line static layout function; no physics simulation needed since layer membership fixes x and a simple sort/bucket fixes y | **Recommended** |
| **D3-force (physics) + Canvas** (`vasturiano/force-graph` or hand-rolled `d3-force` + own canvas draw loop) | Same canvas-class capacity as above; d3-force just supplies the physics tick. | `force-graph` full bundle: **177.6 KB** minified (verified download). Hand-rolled minimal `d3-force` + `d3-quadtree` + `d3-dispatch` + `d3-timer` (the actual physics dependencies, no selection/DOM layer needed since drawing is manual canvas): **~17 KB** (8.3+5.3+1.9+1.9 KB, verified via individual package downloads). | Organic force layout actively fights the desired "layers flow left→right" semantic — nodes will drift into a topology-driven blob unless you pin x-per-layer and only let the force sim jiggle y (`forceX` fixed, `forceCollide` for y jitter) | Viable but adds a dependency + tuning complexity for a layout problem that's actually deterministic (layer membership + rate ranking), not organic |
| **WebGL** (`three.js`/`sigma.js`) | Millions of points — massive headroom, but the headroom is wasted at 200–10K nodes | `three.js` r160 full minified bundle: **669.9 KB** (verified download) — still smaller than the already-inlined Plotly, so bundle size isn't the blocker; `sigma.js` requires `graphology` as a second dependency and is designed for large *organic* network exploration (pan/zoom/community layout), which is not this use case | Justified only if the full 166,700-neuron connectome needs live per-neuron rendering (explicitly *not* recommended per the graph-context doc, which calls for layer-level aggregation at that scale) | **Not recommended for this scope** — adds real engineering complexity (shaders, WebGL context management, scene graph) for a problem that's a 2D column diagram, not a 3D scene |

### Recommendation

**Hand-rolled Canvas 2D, zero external dependencies**, using a deterministic
layered layout:
- x-position = fixed per layer (Input | KC | MBON | PPL | DN columns)
- y-position = neurons sorted within their layer (e.g., by index or by
  current rate, so high-activity neurons cluster visually)
- node radius/opacity = spike rate (normalized against `baseline_rates`,
  already computed server-side per `graph-context.md` §3.2)
- edges = optionally skip entirely, or draw only a downsampled subset
  (e.g., top-K by weight) as thin translucent lines — full edge rendering
  at 10K nodes × real connectome density would be visually unreadable
  anyway; per-layer aggregate flow (bundled "ribbon" between layers,
  thickness = mean inter-layer activity) reads better than a hairball and
  is cheap to draw (one `lineTo` per layer-pair, not per synapse)
- redraw driven by `requestAnimationFrame` for smooth idle animation
  (subtle idle jitter/pulse), with actual *data* refreshed only on each
  poll tick (1.5–5s) — same "cheap tempo update, continuous local
  animation" pattern the project already uses for wing-beat

This mirrors the project's own established pattern (zero-dependency
hand-written `svg.py`) and needs **no new library inlined at all** — the
Canvas 2D API is native to every browser. For the 200-node synthetic mode,
this same code path renders individual circles per neuron; for 2K–10K it
naturally degrades gracefully (canvas doesn't care); for full-connectome
mode, feed it pre-aggregated per-layer values (5 bars, not 166,700 nodes) —
same renderer, different granularity of input data.

If team appetite exists for a slightly richer force-directed feel (nodes
gently repel within their layer column rather than a rigid sort), add
**`d3-force` only** (not full D3, not `force-graph`) at ~17 KB inlined, using
`forceX(layerX).strength(1)` to pin columns and `forceCollide` + weak
`forceY` for organic vertical jitter — this is a genuinely small, surgical
addition if wanted, but is not required to meet the stated goal.

---

## 2. Chart Libraries: Is Plotly.js Enough?

**Yes — no new charting library is needed.** I verified directly (running
`plotly.offline.get_plotlyjs()` in this environment and grepping the output)
that the full bundle **already inlined today** includes `pie`, `bar`,
`scatter`, `histogram`, and `sunburst` trace types, among all standard Plotly
chart types. This is the *full* bundle (not a trimmed/partial one), so
nothing is missing:

- **Decision distribution (pie/bar):** `type: 'pie'` for BuyPlan share of
  choices; `type: 'bar'` (grouped or stacked, policy vs Oracle
  side-by-side) for per-round agreement — both natively supported, zero
  marginal bytes since Plotly is already loaded.
- **Economy decision timeline:** Plotly has no dedicated "Gantt/timeline"
  trace type, but the standard idioms both work with what's already
  inlined:
  - `type: 'scatter'`, `mode: 'markers'`, x = round number, y = categorical
    BuyPlan (via `yaxis: {type: 'category'}`), marker color = agreement
    with Oracle (green/red) — this is the natural fit for
    "round-by-round decision trace" from `graph-context.md` §3.1.
  - `type: 'bar'`, `orientation: 'h'`, `base` = round-start offset for a
    literal Gantt-style bar-per-round view, if a denser visual is wanted.
- No need for `plotly.js-dist` variants, no need for Chart.js/ECharts/etc.
  — introducing a second charting library would mean inlining a second
  bundle for capability that's already present, plus inconsistent visual
  styling (dark theme is currently hand-tuned once, in the `dark` config
  object in `dashboard_server.py`'s poll script) across two different
  libraries' theming APIs.

**Action:** extend the existing `dark` Plotly layout object and add 2 new
`chart-panel` divs + `Plotly.newPlot`/`Plotly.react` calls, following the
exact pattern already used for `chart-fci`/`chart-reward`/`chart-loss`/
`chart-ratio`. This is the lowest-risk, most consistent path.

---

## 3. Real-Time Data: SSE vs Polling

### Current state (verified by reading `dashboard_server.py`)

- Client polls `GET /api/metrics` every 1.5s via `setInterval`, fully
  replacing chart data with `Plotly.react` each tick.
- Server holds all metrics in a `threading.Lock`-guarded dict of
  `collections.deque`, mutated from a background training thread's
  monkey-patched `harness._emit` (`patched_emit`).
- FastAPI + Starlette natively support `StreamingResponse` with
  `media_type="text/event-stream"` for SSE (confirmed against the FastAPI
  docs — `fastapi>=0.135` in `pyproject.toml` supports this out of the box,
  no new dependency needed), and the browser-side `EventSource` API
  (confirmed against MDN) auto-reconnects on drop and requires materially
  less client code than the current `fetch`+`setInterval` loop.

### Does SSE actually help here?

**No, not as the primary transport — for three independently-verified
reasons specific to this codebase:**

1. **The neural-activity data source itself is throttled, not the
   transport.** Per `graph-context.md` §3.2, a live LIF spiking simulation
   is expensive to run per query (`duration_ms/DT_MS` ≈ 80 steps per call);
   the explicit recommendation there is to sample it "once every few
   seconds, not every PPO step / not every poll." SSE would let the *server*
   push updates the instant new data exists, but new neural-activity data
   literally doesn't exist more often than the current 1.5s poll interval
   already captures — there is no latency to save on this specific panel.
2. **Perceived "liveliness" is already solved architecturally by CSS, not
   by network cadence.** The existing `updateAvatar(fci)` /
   `--wing-beat-duration` pattern (and the recommended glow/pulse extension
   in §4 below) works by having a **continuous CSS `@keyframes` loop run at
   60fps on the compositor thread**, while network updates only adjust the
   loop's *tempo* parameter occasionally. This means the avatar and
   neural-activity visuals can *look* real-time even on a slow poll,
   because the animation itself never stops between polls — a classic
   "server sets the rate, client renders the motion" pattern that
   decouples visual smoothness from network frequency entirely.
3. **Single local viewer, no scale requirement.** This is a `localhost`
   admin dashboard for one training run, not a multi-tenant service —
   polling's main downsides (N×connections, server fan-out cost) don't
   apply. SSE's main benefit over polling (avoiding N wasted round-trips
   when nothing changed) is a non-issue at n=1 client.

### Recommendation

**Keep polling as the primary transport.** It is simple, already
implemented, already thread-safe against the training thread, and matched
to the actual data-generation cadence. Two low-risk refinements:

- Optionally **shorten the poll interval only for the metrics endpoint**
  (e.g., 1.5s → 1s) if PPO iterations complete faster than 1.5s and users
  want tighter chart granularity — this is a one-line `setInterval` change,
  not an architecture change.
- If genuinely instantaneous *discrete event* notification is wanted later
  (e.g., "round won" triggering an immediate avatar pose change rather than
  waiting for the next poll), that is the **one legitimate SSE use case**
  here: a narrow, separate `/api/events/stream` endpoint that only pushes
  discrete named events (not continuous metrics), leaving the existing
  metrics-polling loop untouched. This is explicitly **out of scope** for
  the current backlog item (which asks for visualization additions, not
  transport changes) and should be flagged as a possible future
  enhancement rather than bundled into this change — it also introduces
  new failure modes (dropped SSE connections, async-generator lifecycle
  vs. the existing synchronous polling handlers) that aren't worth taking
  on for a targeted, additive backlog item.

---

## 4. SVG Filter Effects for Avatar Glow/Pulse

I confirmed the relevant primitives directly against MDN source (fetched
`mdn/content` markdown for both elements):

- **`<feGaussianBlur>`**: blurs `SourceGraphic`/`SourceAlpha` by
  `stdDeviation`. Key gotcha confirmed from the spec text: the default
  filter region (`x`/`y`/`width`/`height` on the parent `<filter>`) is only
  `-10%`/`120%` — **too tight for a visible glow halo**; MDN's own example
  explicitly recommends widening to e.g. `x="-30%" y="-30%" width="160%"
  height="160%"` for larger `stdDeviation` values, or the blur gets clipped
  at the element's bounding box edge. For a glow radius that needs to
  extend ~15–20px beyond a ~70px-wide thorax ellipse, budget generously
  (e.g. `-50%`/`200%`).
- **`<feColorMatrix>`**: full 5×5 matrix transform of `[R,G,B,A]` per pixel.
  Useful here for a "hue-shift toward hot amber/red as activity rises"
  effect via `type="hueRotate"` or `type="saturate"`, applied to a
  duplicated/blurred copy of the body shape used purely as a glow-halo
  layer (not the crisp foreground body).
- **Recommended structure**, concretely tied to the actual SVG the project
  ships today (`svg.py`, verified layout: thorax `<ellipse cx="150" cy="168"
  rx="35" ry="30">`, head `<ellipse cx="150" cy="105" rx="32" ry="28">`):
  1. Add one `<filter id="neuralGlow">` containing a **fixed**
     `feGaussianBlur stdDeviation="8"` (baked at render time, not animated
     continuously — see performance note below) feeding a `feColorMatrix`
     tuned per-tier (reuse the existing 3-tier threshold pattern from
     `_resolve_body_color`: >0.7 hot/bright, >0.4 amber, else dim).
  2. Add a **duplicate halo ellipse** behind the existing thorax/head
     group (same cx/cy, slightly larger rx/ry, `filter="url(#neuralGlow)"`,
     solid fill in the tier color) — inserted *before* the real body
     elements in the `<g class="fly-body-group">` so it paints underneath.
  3. Pulse via a **new CSS `@keyframes pulseGlow`** animating `opacity`
     (e.g. 0.25 → 0.85) and/or `transform: scale(...)` on that halo
     element only — **not** the `stdDeviation`/filter attributes
     themselves.

### Performance note (why animate opacity/transform, not the filter itself)

This directly extends a principle the project has already adopted — the
top of `svg.py` states outright: *"CSS @keyframes on transform properties
only (compositor-thread friendly)."* This is standard browser
rendering-pipeline behavior: `transform` and `opacity` changes can be
handled by the compositor thread without a full repaint, whereas changing
`filter` values (including animating `stdDeviation` or `feColorMatrix`
`values`) forces the browser to re-rasterize that element's filter
graph on the main thread on every change. Practically:

- **Do:** bake the blur radius and color-matrix values once per poll tick
  (they change slowly — once every 1.5s+ is trivial repaint cost), then
  drive the *visible pulsing* via `@keyframes` on `opacity`/`transform` of
  the (now-static) filtered halo layer, running continuously at 60fps on
  the compositor thread between polls — same pattern as the existing
  wing-beat duration mechanism.
- **Don't:** put `stdDeviation` or `feColorMatrix` `values` inside a
  `@keyframes` block that runs every animation frame — this is the classic
  SVG-filter-animation performance trap and would fight the project's
  stated compositor-friendly design goal.
- **Data source for the pulse rate/intensity:** reuse the exact interim
  proxy already identified as the pragmatic first step in
  `graph-context.md` §3.3 — `mean_value` (already computed every PPO
  update, already flowing through the metrics lock) — with a clean
  upgrade path to real per-neuron `features.mean()`/`.abs().mean()` once
  the neural-activity tap point from §3.2 lands. Ship the CSS/SVG
  mechanism now; swap the data source later without touching markup.

**New JS setter:** add `window.updateAvatarActivity(activity)` alongside
the existing `updateAvatar(fci)`, setting two new CSS custom properties
(`--glow-intensity`, `--pulse-duration`) that the new `@keyframes` block
reads — exactly mirroring the existing `--wing-beat-duration` mechanism, so
reviewers/future agents see one consistent idiom for "server sets tempo,
CSS drives motion" throughout the file.

---

## 5. Architecture: Staying Single-File / Self-Contained

Confirmed inlining precedent already in the codebase
(`flyecon/dashboard/renderer.py::_get_plotly_js()`): the *actual* JS source
of a third-party library (`plotly.offline.get_plotlyjs()`) is read from the
installed Python package at render time and interpolated directly into a
`<script>{plotly_js}</script>` tag — no CDN `<script src=...>` reference, no
build step, no bundler. This is the established, working pattern for "how
to inline a JS library" in this project, and it should be the template for
any additional inlined code:

| Item | Inlining approach |
|---|---|
| Canvas-based neural activity renderer (recommended, §1) | **No library to inline** — write plain JS directly into the existing f-string HTML template, same as all the current chart-init/poll JS. Zero new build/packaging concerns. |
| `d3-force` (only if the optional organic-jitter enhancement from §1 is later wanted) | Vendor the minified `d3-force`+deps bundle as a small static string constant in Python (same shape as `_PLOTLY_JS_STUB`/`_get_plotly_js()` in `renderer.py`), read via `Path.read_text()` or hardcoded as a triple-quoted string, then interpolated the same way `plotly_js` is today. ~17 KB — trivial to vendor as a checked-in file (e.g. `flyecon/dashboard/vendor/d3-force.min.js`) read once at import time. |
| SVG filter glow (§4) | No inlining question at all — `<filter>`/`feGaussianBlur`/`feColorMatrix` are native SVG markup, already generated the same way `svg.py` generates every other SVG element (an f-string). |
| Plotly.js (existing) | Already solved — no change needed. |

**No architecture change is required to satisfy the single-file guard for
any of the three requested features.** The two genuinely new code
surfaces (canvas neural renderer, SVG glow filter) are both pure
hand-written markup/JS with zero external library dependencies, and the
one optional library addition (`d3-force`) follows the exact vendor-and-
inline pattern the codebase already uses for Plotly, just at ~1/250th the
size.

---

## Recommended Tech Stack Summary

| Feature | Technology | New dependency? | Inlined size added |
|---|---|---|---|
| Neural activity (Input→KC→MBON→PPL→DN) | Hand-rolled Canvas 2D, deterministic layered layout, `requestAnimationFrame` loop | None | ~0 KB (pure hand-written JS, same as existing poll script) |
| Decision distribution (pie/bar) | Plotly.js (already inlined) — `type: 'pie'`, `type: 'bar'` | None | 0 KB marginal |
| Decision timeline | Plotly.js (already inlined) — `type: 'scatter'` categorical y-axis, or horizontal `bar` w/ `base` | None | 0 KB marginal |
| Real-time transport | Keep existing 1.5s `fetch`+`setInterval` polling; optionally tighten interval | None | 0 KB |
| Avatar glow/pulse | Native SVG `<filter>` (`feGaussianBlur`+`feColorMatrix`) + CSS `@keyframes` on `opacity`/`transform` only | None | ~1–2 KB of new SVG/CSS text |
| (Optional, not required) organic node jitter | `d3-force` + deps, vendored as static file | New (optional) | ~17 KB |

**Net new inlined bytes for the full backlog item: on the order of a few KB**
— negligible next to the 4.1 MB Plotly bundle already shipped today.

---

## Performance Trade-offs (condensed)

- **SVG DOM for 10K live-updating nodes:** rejected — per-element DOM
  overhead makes this the only option in this research that risks visibly
  missing 60fps at the upper end of the stated 200–10K node range.
- **WebGL:** rejected as unnecessary complexity — its scaling headroom
  (millions of points) is not needed for a ≤10K-node layered diagram, and
  it would be the first WebGL context in a project that has otherwise
  stayed with plain SVG/CSS/Canvas throughout.
- **Canvas 2D (recommended):** matches the actual node-count ceiling with
  large headroom (community reference examples run smoothly at ~75K
  elements, ~7.5× the stated upper bound), costs zero new inlined bytes,
  and reuses the "hand-write it" philosophy already established by
  `svg.py`.
- **Filter-attribute animation vs opacity/transform animation:** the single
  most important performance decision for the avatar glow — animating
  `stdDeviation`/`feColorMatrix` directly forces main-thread repaint on
  every frame; animating `opacity`/`transform` of a pre-filtered layer
  keeps the pulse on the compositor thread, consistent with the project's
  documented design principle.
- **SSE vs polling:** the bottleneck for the neural-activity panel is the
  server-side LIF simulation cost, not transport latency — switching
  transports would add complexity without measurably reducing perceived
  latency, since the underlying data doesn't refresh faster than the
  current poll interval anyway.

---

## References

- `plotly.offline.get_plotlyjs()` executed directly in this environment
  (plotly 7.0.0 installed) — bundle size 4,192.7 KB; confirmed `pie`,
  `bar`, `scatter`, `histogram`, `sunburst` trace types present in output.
- `flyecon/dashboard/renderer.py::_get_plotly_js()` and
  `dashboard_server.py` (this repo) — read directly for existing inlining
  pattern and poll-loop implementation.
- `flyecon/avatar/svg.py` (this repo) — read directly for existing SVG
  structure, CSS animation pattern, and stated compositor-friendly design
  principle.
- https://raw.githubusercontent.com/vasturiano/force-graph/master/README.md
  — canvas-based force graph, confirmed "~75k elements" public example,
  confirmed "Uses HTML5 canvas for rendering... [because of] d3-force"
  physics + canvas draw split.
- https://raw.githubusercontent.com/vasturiano/3d-force-graph/master/README.md
  — WebGL/Three.js variant, confirmed dependency on Three.js + d3-force-3d.
- https://raw.githubusercontent.com/jacomyal/sigma.js/main/README.md —
  confirmed WebGL graph library requires separate `graphology` dependency,
  designed for large organic network exploration.
- https://raw.githubusercontent.com/plotly/plotly.js/master/README.md —
  confirmed script-tag / bundle loading conventions.
- https://d3js.org/d3.v7.min.js — downloaded directly, 279,706 bytes
  (full D3 bundle).
- https://cdn.jsdelivr.net/npm/d3-force@3/dist/d3-force.min.js (8,300
  bytes), d3-quadtree@3 (5,279 bytes), d3-dispatch@3 (1,901 bytes),
  d3-timer@3 (1,947 bytes), d3-selection@3 (13,522 bytes), d3-drag@3
  (4,186 bytes), d3-zoom@3 (9,984 bytes) — all downloaded directly to
  verify minimal-subset bundle sizes.
- https://unpkg.com/three@0.160.0/build/three.min.js — downloaded
  directly, 669,884 bytes.
- https://unpkg.com/force-graph/dist/force-graph.min.js →
  `force-graph@1.51.4` — downloaded directly, 177,599 bytes.
- MDN (`mdn/content` GitHub source, fetched directly):
  `Web/SVG/Reference/Element/feGaussianBlur` — confirmed default filter
  region (`-10%`/`120%`) and the documented need to widen it for larger
  blur radii; `Web/SVG/Reference/Element/feColorMatrix` — confirmed 5×5
  matrix transform semantics.
- MDN `Web/API/Server-sent_events/Using_server-sent_events` — confirmed
  `EventSource` API shape and one-way server→client semantics.
- FastAPI docs (`fastapi/fastapi` GitHub source,
  `docs/en/docs/advanced/custom-response.md`) — confirmed native
  `StreamingResponse` support for SSE-style streaming, no new dependency
  needed beyond the already-pinned `fastapi>=0.135`.
- `.factory/strategy/study-combined.md` (this repo, includes
  `graph-context.md` content) — primary source for data tap points,
  architecture layers, and the 200/2K–10K/166,700-neuron connectome mode
  breakdown that scoped the node-count analysis in §1.

## Notes on Search Method

Standard search engines (DuckDuckGo, Bing) returned bot-challenge pages or
JS-rendered result shells that could not be scraped reliably via `curl` in
this environment (no `WebSearch`/`WebFetch` tool was available in this
session's toolset). To compensate, research was conducted by **directly
fetching primary sources** (raw GitHub READMEs, MDN's markdown source
repository, actual npm-distributed bundle files) via `curl`, and by
**directly executing the project's own code** (`get_plotlyjs()`) to obtain
ground-truth figures rather than relying on secondhand claims. All
quantitative claims in this report (bundle sizes, trace-type availability)
were verified this way rather than asserted from memory.
