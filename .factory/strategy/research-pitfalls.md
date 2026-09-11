# Research — Pitfalls & Scope for Dashboard Enhancement

## Context

**Target:** `dashboard_server.py` (live FastAPI training dashboard). Backlog item
(TARGETED MODE, single hypothesis): add (1) CS Economy Decision Visualization,
(2) Connectome Neural Activity visualization, (3) Better Fly avatar with
glow/pulse tied to neural activity.

**Method:** Read `study-combined.md` / `graph-context.md` for the architectural
survey, then cross-checked every claim against direct source reads in the
sibling worktree (`run-b7b1625d`, commit `46302d0` — the actual code, since
this worktree only carries a stub graph) and ran live micro-benchmarks of
`LIFNetwork.simulate()` on this machine (CPU, `torch.set_num_threads(1)`,
matching production config in `_training_loop`).

**Headline finding that changes the risk calculus:** `dashboard_server.py`'s
`_lifespan()` calls `Harness()` with **zero arguments** →
`n_neurons=200, connectome_mode="synthetic"` (`harness.py:118`,
`dashboard_server.py:84`). There is currently **no CLI flag or code path that
lets the live dashboard boot the full 166,700-neuron MaleCNS connectome** —
that only happens via `flyecon/__main__.py`'s `_run_harness(connectome_name=...)`,
which `dashboard_server.py` never calls. This means the "200 vs 166K neurons"
question is not hypothetical-both-equally-likely; **200 neurons is the actual
default runtime state of the file we are modifying**, and 166K is an
aspirational future mode that isn't wired up yet. Scope and perf budgets
below are set accordingly, with the full-scale case flagged as an explicit
non-goal / documented constraint rather than a thing to solve now.

---

## 1. PERFORMANCE — cost of sampling `spike_counts` for the dashboard

### Measured cost (this machine, single-threaded torch, `duration_ms=400` → 80 LIF steps, matching `DEFAULT_BIN_MS`)

| n_neurons | density | edges (nnz) | single `simulate()` (batch=1, 80 steps) | as % of a 1.5s poll |
|---|---|---|---|---|
| 200 (**current dashboard default**) | 0.10 | 3,980 | **5.2 ms** | 0.35% |
| 2,000 | 0.05 | 199,900 | **12.7 ms** | 0.85% |
| 10,000 | 0.01 | 999,900 | **46.5 ms** | 3.1% |
| 50,000 | 0.005 | 12,499,750 | **629 ms** | 42% |
| 166,700 (MaleCNS full, ~125M synapses, README) | ~0.0045 | ~125,000,000 | **extrapolated ~6 s** (O(nnz) scaling holds cleanly across the measured range: 10x nnz ≈ 10-13x time) | **400%+ — exceeds the poll interval outright** |

`LIFNetwork.step()`'s cost is dominated by the sparse CSR mat-vec
(`torch.mv(self._W_csr, self._prev_spikes)`), run once per of the 80 steps;
everything else (`clamp`, `where`) is O(n) and negligible even at 166K. Cost
scales with **synapse count, not neuron count** — this is the right variable
to budget against, not "how many neurons."

### Implications

- **At the current default (n=200):** a full extra `simulate()` call costs
  ~5ms. Even called every single 1.5s poll tick, this is free (0.3% duty
  cycle). No throttling is strictly required for the synthetic/default mode,
  but throttle anyway (see below) since it costs nothing and future-proofs
  against mode changes.
- **At mushroom-body scale (~2K-10K, per graph-context.md's stated middle
  tier):** 13-47ms per call — still cheap enough for the 1.5s poll rate
  without throttling, but this is where you'd want to start being
  deliberate rather than casual about call frequency.
- **At full-connectome scale (166,700 neurons / 125M synapses):** a single
  extra forward-simulate call costs an estimated ~6 seconds on this CPU —
  more than 4x the entire poll interval. This is not a "sample less often"
  problem, it's a "this call is fundamentally too slow to run inline in a
  background thread that must also make training progress" problem. Running
  it even once every 30-60s would still visibly stall training throughput
  (recall `collect_rollouts` already does 128 sequential single-item
  simulates per iteration for training — at full scale that's ~13 minutes
  per PPO iteration from `batched_simulate`'s per-item Python loop alone,
  independent of anything the dashboard adds).
- **Recommendation:** Design the sampling hook to be **frequency-configurable
  and connectome-size-aware**, but do not attempt to make full-connectome
  mode performant as part of this task — that would require batching
  `batched_simulate`'s Python-level loop into true batched sparse ops (a
  `simulation/` change, which is a fixed/hot-path surface per graph-context's
  own guidance: "Do not modify `LIFNetwork.step()`/`simulate()` internals").
  State this explicitly as an out-of-scope constraint in the PR/commit
  description so nobody is surprised the neural panel is slow if someone
  later flips `dashboard_server.py` to `connectome_mode="full"`.

### Concrete throttling design

- Sample neural activity **once every N passes through the training loop**
  (e.g., every 4th call to `harness.train_iterations(4)`, i.e. every 16
  PPO iterations), not on a wall-clock timer and not on every `/api/*`
  poll. This decouples sampling cost from poll cadence entirely.
- Use a **fixed, cached representative state** (e.g. a single pistol-round
  state, or the last state seen in the training loop) for the inspect call
  — a single-item batch, not a full rollout-sized batch. This is what keeps
  the cost at "single `simulate()` call" rather than "128x that."

---

## 2. THREAD SAFETY — the risk graph-context.md doesn't cover

The established `_metrics_lock` pattern (a `threading.Lock` guarding a plain
`dict`/`deque` of already-computed Python floats, read by async FastAPI
handlers, written by the `fly-train` background thread) is **safe and
sufficient for scalar/aggregate data** — reuse it for the new panels'
publish side without modification.

**What it does NOT cover, and what graph-context.md's "cheapest tap point"
suggestion glosses over:** `LIFNetwork` mutates shared instance state
in-place on every step — `self.V`, `self.refractory`, `self.spike_counts`,
`self._prev_spikes` (`lif.py:154-157`). `FlyPolicy.forward()` calls
`self._lif.batched_simulate(...)` which calls `self.reset()` then loops
`self.step()` — all mutating that same `LIFNetwork` instance that the
training thread's `collect_rollouts()` is *also* driving via
`policy.act(state)` on every PPO step.

**The trap:** if a new "inspect for dashboard" call is invoked from anywhere
*other than* the training thread — e.g. as an `asyncio.create_task(...)`
scheduled from the FastAPI `lifespan` handler, or directly inside an
`/api/neural` request handler — you get two threads calling into the *same*
mutable `LIFNetwork`/`FlyPolicy` object concurrently. Python's GIL prevents
bytecode-level corruption but does **not** prevent semantic races: thread A's
`reset()` can zero `spike_counts` mid-way through thread B's step loop,
`V`/`refractory` reads and writes can interleave arbitrarily, and PyTorch's
in-place tensor ops (`.zero_()`, `.fill_()`, `self.V = V_new` reassignment)
are not synchronized across threads. Worse: this doesn't just corrupt the
dashboard's read — it can corrupt **training data mid-rollout**, which is a
silent correctness bug in the model being trained, far worse than a dashboard
glitch. The same applies more mildly to `PopulationEncoder`'s trainable
gains and `readout.W`/`value_head` parameters: reading them via a forward
pass on one thread while `optimizer.step()` mutates them in-place on another
is a known-unsafe PyTorch pattern (no crash, but can produce reads of
partially-updated parameters).

**Mitigation — do the sampling *inside* the training thread, never from the
request-handler / event-loop thread:**

1. Add the periodic "inspect" call as a step inside `_training_loop()`'s
   `while not stop.is_set(): harness.train_iterations(4)` loop (or inside
   `Harness.train_iterations`/`patched_emit` itself), so it runs
   **sequentially interleaved with, never concurrently with**, the training
   calls that also touch `_lif`/`policy`. This is the same trick already
   used for `_metrics` — the async endpoints never touch torch objects, they
   only read a lock-guarded dict of plain Python values that the training
   thread already computed and published.
2. Publish the result (per-layer mean rates, decision-trace tuples, a scalar
   activity summary for the avatar) as plain Python floats/lists into the
   same `_metrics_lock`-guarded structure (extend the existing dict, add new
   keys — don't introduce a second lock unless profiling shows contention,
   which is very unlikely given how fast this all is).
3. **Explicit anti-pattern to flag for the Builder:** it will be tempting to
   add a `@app.get("/api/neural")` handler that calls `harness.policy(...)`
   or a fresh `inspect()` method directly, since that reads cleanly and
   "FastAPI is async so it's probably fine." It is not fine — it is a
   different OS thread (or the event loop thread) touching the same
   `LIFNetwork` the training thread owns. The rule: **request handlers only
   ever read the lock-guarded dict; anything that calls into `policy`/`_lif`
   happens exclusively inside `_training_loop`.**
4. If, later, true on-demand (not periodic) inspection is wanted (e.g. a
   "sample now" button), the safe way is a **second, independent
   `LIFNetwork` instance** dedicated to inspection, sharing the read-only
   `Connectome`/weight matrix (never mutated by `step()`) but with its own
   `V`/`refractory`/`spike_counts`/`_prev_spikes` tensors — this avoids the
   shared-mutable-state race by construction. Not needed for the MVP polling
   design above; note it as the pattern to reach for if scope grows.

---

## 3. SINGLE HTML CONSTRAINT — already blown past, in a good way

Measured directly: `flyecon.dashboard.renderer._get_plotly_js()` (imported
and used by `dashboard_server.py` today) inlines the **full installed
Plotly.js bundle via `plotly.offline.get_plotlyjs()`, which is 4,293,280
bytes (~4.19 MB)** of raw JS, embedded directly into the `<script>` tag of
the single HTML response. `plotly>=5.18` is a real `pyproject.toml`
dependency, not an optional extra — this is what ships today, right now, in
what factory.md calls "a single self-contained HTML file."

**Implication for the 250KB-D3 question:** the "can we afford to inline a
library" debate is already settled by precedent — a self-contained HTML file
in this project already carries ~4.2MB of embedded JS and satisfies the
guard (the guard is about *self-containment* — no external `<script src=...>`
network fetches — not about byte budget). A minified D3 bundle (~250KB) is
**~17x smaller** than what's already inlined. There is no real constraint
being violated by adding D3 if the team wants a connectome node-link graph.

**Recommendation — but avoid D3 anyway, for scope reasons, not size reasons:**
Plotly is already loaded and already supports bar charts, heatmaps, and
grouped/stacked charts — everything the layer-level neural-activity
aggregation (KC/MBON/PPL/DN/other → mean rate) and the economy
decision-timeline panel need, **at zero additional inlined bytes**. Introduce
D3 only if a literal force-directed connectome graph (nodes = neurons, edges
= synapses) is explicitly wanted — which is a "nice-to-have" stretch goal,
not required by the backlog wording ("CONNECTOME NEURAL ACTIVITY
visualization" is satisfied by a per-layer bar/heatmap, it does not
literally require a graph-drawing library). Recommend: **MVP uses Plotly
bar/heatmap panels for both new visualizations; defer D3/Canvas
node-link graph rendering as an explicit stretch item**, not blocked by
any size constraint, just deprioritized for a first pass.

For the avatar glow/pulse: this is SVG (`avatar/svg.py`), not Canvas. SVG
`<filter>` primitives (`feGaussianBlur` + `feColorMatrix`/`feFlood`) achieve
glow with zero additional library bytes and stay consistent with the
existing self-contained `<style>+<svg>+<script>` string pattern — no reason
to introduce a `<canvas>` element or any drawing library for this piece.

---

## 4. DATA SIZE — per-neuron payload over `/api/neural`

Given finding #1 (default is n=200, and full-connectome mode isn't wired
into the live dashboard), the raw "200 to 166K floats" range in the prompt
is really "200 floats today, up to 166K only if someone later flips the
connectome mode" — so size the payload strategy for both but implement/test
against the real default.

- **200 floats as JSON**: trivially small (a Python list of 200 floats
  serializes to roughly 2-4 KB as text) — sending the raw per-neuron array
  every poll is not a bandwidth problem at n=200 or even n=10,000 (still
  only tens of KB).
- **166,700 floats as JSON**: ~1.5-3 MB per response depending on float
  precision/formatting — this *is* a real payload problem if ever hit
  (166K-element JSON array, re-serialized and re-sent every poll, plus
  React/Plotly re-render cost client-side), independent of the ~6s
  server-side compute cost already noted in #1.
- **Recommendation (applies regardless of connectome size, so it's free
  insurance):** Always send **layer-level aggregates** (mean/std spike rate
  per neuron-type prefix: KC, MBON, PPL, DN, other — 5-ish scalars, matching
  the existing `renderer.py` Panel 5 stub design already sketched with
  `x:['KC','MBON','DN','PPL']`) as the primary payload. Do **not** add a
  raw per-neuron array to the JSON payload as MVP scope — it provides
  little visual value at 200 neurons (a bar chart of 5 layers is more
  readable than 200 individual bars) and is a latent scaling hazard if
  connectome mode ever changes. If a heatmap/sampled per-neuron view is
  wanted later, cap it explicitly (e.g. random/stratified sample of ≤200
  neurons regardless of true `n_neurons`) rather than sending the full
  array — build this cap in from day one so it's not a silent landmine.

---

## 5. FCI LATENT BUG — `kc_mean_rate` is not real KC rate (and it's worse than that)

Confirmed directly in `harness.py:382-387` and `dashboard/fci.py`. The bug is
actually **two of the three `compute_fci()` inputs are proxies derived from
the same single underlying scalar**, not just one:

```python
fci = compute_fci(
    kc_mean_rate=max(metrics.get("mean_reward", 0.0) * 10, 0),        # proxy #1
    recent_reward=max(metrics.get("mean_reward", 0.0) * 16000, 0),    # proxy #2 — same source metric!
    consecutive_wins=min(int(metrics.get("mean_value", 0) * 5), 5),   # proxy #3, from mean_value
)
```

`kc_mean_rate` (weight 0.4) and `recent_reward` (weight 0.4) are **both**
scaled copies of `metrics["mean_reward"]` — 80% of FCI's weight currently
collapses to one signal wearing two hats. `consecutive_wins` (weight 0.2)
isn't an actual win-streak counter either, it's `mean_value` rescaled.
None of the three documented semantics (KC firing rate, cumulative reward,
win streak) are what's actually being measured today.

**Should you fix it?** Recommend **no, not as part of this task** — for
three concrete reasons:

1. **Scope creep risk.** A real fix isn't "swap in real KC rate" — it's
   "also fix `recent_reward` to track actual cumulative money instead of
   re-deriving from `mean_reward`, and fix `consecutive_wins` to track an
   actual streak counter instead of deriving from `mean_value`." That's a
   3-part change to `harness.py`'s training loop state (needs new tracked
   state: cumulative reward window, actual win/loss streak), well beyond
   "wire in a neural-activity number for the avatar."
2. **Recalibration risk.** `compute_fci`'s thresholds (`kc_rate_max=20.0`,
   `reward_max=16_000.0`) and the avatar's tier thresholds (`fci > 0.7`,
   `fci > 0.4` in `svg.py::_resolve_body_color`) were implicitly tuned
   against the *current* proxy's numeric range and behavior over training.
   Swapping in real (likely much noisier, differently-scaled) KC firing
   rates without re-tuning those constants risks making the avatar's
   glow/color *less* meaningful, not more — exactly the opposite of the
   backlog's intent ("BETTER fly avatar").
3. **Not required by the backlog.** The three requested items are additive
   visualizations; none of them require FCI's internal semantics to be
   correct — the avatar enhancement backlog item asks for glow/pulse "keyed
   to neural activity," which per graph-context.md's own recommendation
   (§3.3, item 4) can and should be wired as a **new, separate**
   `neural_activity` scalar (e.g. mean output-neuron firing rate from
   `features` — already computed, already in memory) driving a **new** CSS
   var (`--glow-intensity`), independent of the existing (buggy) FCI →
   wing-speed path. This sidesteps the bug entirely rather than fixing it.

**Recommended action:** Leave `compute_fci()` and its call site untouched.
Feed the avatar's new glow/pulse effect from a **new, honestly-named**
signal (real neural activity from the inspect tap, or `mean_value`/`features`
as an interim stand-in per graph-context.md §3.3.4) rather than reusing or
patching the mislabeled `kc_mean_rate` parameter. Optionally, as a
zero-risk, zero-behavior-change cleanup, rename the call-site keyword/add a
one-line comment clarifying `kc_mean_rate` is currently a reward proxy, so
the next person doesn't get misled the way this research pass initially
would have been. Log the full three-part fix as a distinct backlog item for
a future (non-targeted) cycle rather than folding it into this one.

---

## 6. SCOPE MANAGEMENT — MVP per feature, and what to defer

**Cross-cutting risk not yet mentioned:** `eval/score.py::dashboard_renders`
(weight 0.10) only tests `flyecon.dashboard.renderer.render_dashboard()` —
the *static* renderer path — via a temp-dir `TelemetryStore` and checking
`<html` appears in the output. **It never imports, boots, or hits
`dashboard_server.py` at all.** This means whatever ships for this task has
**zero automated eval coverage** in the existing harness — a broken
`/api/neural` endpoint or a JS error in the new poll code will not move the
eval score in either direction, and won't be caught by the harness. This
isn't a reason to gold-plate the static renderer instead (backlog explicitly
targets `dashboard_server.py`), but it is a reason to (a) keep the diff
small/reviewable since there's no safety net, and (b) consider a minimal
manual smoke-test path (start the server, `curl` each new `/api/*` endpoint,
assert 200 + expected JSON shape) as part of the implementation PR, since
`eval/score.py` won't do this for you.

### Per-feature MVP recommendation (least → most invasive, per graph-context.md §4, endorsed here after verification)

| # | Feature | MVP (do this) | Defer (explicit stretch goal) |
|---|---|---|---|
| 0 | Quick win, not one of the 3 but nearly free | Surface `agreement_rate`, `policy_mean_money`, `oracle_mean_money` from the existing `"eval"` telemetry event into `_metrics` — already computed in `evaluate_against_oracle`, already emitted by `harness.py:417-422`, currently dropped by `dashboard_server.py`'s `patched_emit`. Zero new compute. | — |
| 1 | CS Economy Decision Viz | Add the 3 already-computed eval fields above to a new chart panel (bar or line, agreement rate over eval checkpoints). | Full round-by-round decision-trace timeline (state, policy action, oracle action per round) — requires a new sampling helper in `ppo.py` that replays a full episode; real value but bigger diff. Ship as v2. |
| 2 | Connectome Neural Activity Viz | Layer-level aggregate (KC/MBON/PPL/DN/other mean rate) from one throttled `simulate()` call per N training-loop passes, run **inside** `_training_loop` (§2 thread-safety), rendered as a Plotly bar chart. Requires: `Harness.connectome` read-only property (mirrors existing `.policy` pattern), a small `neuron_id → layer_name` lookup built once at boot. | Per-neuron heatmap/graph view, D3 force-directed connectome graph, full-connectome-mode support. |
| 3 | Better Fly Avatar (glow/pulse) | Additive `neural_activity: float \| None = None` param on `svg.py::to_html()` (backward-compatible default), one new SVG `<filter>` for glow, one new CSS var (`--glow-intensity`), one new JS setter (`updateAvatarActivity`) mirroring the existing `updateAvatar(fci)` pattern. Feed it from `mean_value` (already flowing, zero new compute) initially, upgrade to the real per-layer rate from feature #2 once that lands in the same PR/cycle. | Parity across `ascii.py`/`sprite.py`/`threejs.py` tiers (out of scope — `dashboard_server.py` hardcodes `svg.py` directly, bypassing the tier system entirely, so no user-visible benefit to touching the other tiers here). Fixing the underlying FCI bug (§5) — explicitly deferred. |

### General scope guardrails

- All three MVP paths are **additive**: new optional params with
  backward-compatible defaults, new properties, new endpoints, new dict
  keys. None require touching `LIFNetwork.step()/simulate()`,
  `PPOTrainer.update()/collect_rollouts()`'s core logic, or
  `EconomyMDP`/`solver.py` — consistent with graph-context.md's fixed/hot
  surfaces and factory.md's guards (LIF params, MR12 rules, neurotransmitter
  signing must not change).
- Single hypothesis budget (TARGETED MODE) means this should land as **one
  cohesive change**, not three separate PRs — but internally sequence the
  implementation in the MVP order above (economy quick-win → avatar glow →
  neural panel) since each step is cheap, testable in isolation via
  `curl`/manual load, and independently revertable if the thread-safety
  pattern in §2 needs iteration.
- Do not expand into full-connectome-mode support, D3/Canvas connectome
  graphs, decision-trace timelines, or the FCI three-part fix in this cycle
  — all four are real, valuable, and explicitly out of scope for a
  single-hypothesis targeted cycle. Recommend the Strategist note these as
  candidate backlog items for a future (non-targeted) cycle rather than
  silently dropping them.

---

## Summary of Risk Mitigations

1. **Performance:** Default runtime is 200 neurons (~5ms/call, free at any
   poll rate); throttle sampling to once per few training-loop passes anyway
   as free insurance; explicitly document full-connectome mode
   (~6s/call, estimated from measured O(nnz) scaling) as unsupported/slow by
   design, not a bug to fix here.
2. **Thread safety:** Never call `policy`/`_lif` from a FastAPI request
   handler or a separately-scheduled asyncio task. All new sampling happens
   inside `_training_loop`, sequential with existing training calls;
   publish only plain-Python aggregates through the existing
   `_metrics_lock` pattern.
3. **Single HTML file:** Non-issue — a 4.2MB Plotly bundle is already
   inlined and satisfies the guard today. Prefer reusing Plotly (already
   loaded, zero marginal bytes) over adding D3 for the MVP bar/heatmap
   panels; SVG filters (no library) for avatar glow.
4. **Data size:** Always send layer-level aggregates, never a raw
   per-neuron array, regardless of connectome size — cheap now, safe later.
5. **FCI bug:** Leave `compute_fci()` untouched; feed the avatar's new
   glow effect from a new, honestly-named signal instead of reusing/patching
   the mislabeled `kc_mean_rate` proxy. Log the full 3-part fix as a
   separate future backlog item.
6. **Scope:** Ship the additive MVP table above as one cohesive change;
   defer per-neuron heatmaps, D3 graphs, decision-trace timelines,
   full-connectome support, and the FCI fix explicitly. No automated eval
   coverage exists for `dashboard_server.py` (`dashboard_renders` only tests
   the static `renderer.py` path) — keep the diff small and manually
   smoke-test the new endpoints since the harness won't catch regressions.
