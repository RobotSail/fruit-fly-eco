## CEO Review: Research Agents (Parallel)
- **Verdict:** PROCEED
- **Rationale:** All three research outputs are exceptionally thorough, grounded in actual source code analysis rather than speculation, and directly address the three dashboard enhancement features.
- **Issues found:** None significant.
- **Key findings to carry forward to Strategist:**

### Similar Projects (research-similar.md)
- **ngxson/fly-llm-demo** reference studied in depth — key technique: GPU point-cloud with fake "sweep" animation ordered by synaptic hop-distance. Applicable: fake propagation animation is cheap to implement.
- **Xenova/fruit-fly-simulation** — NeuroMechFly 3D fly with exponential smoothing for smooth avatar motion (solves poll jitter)
- **Twoos123/cs2-meta-engine** — CS2 economy tracker with buy-type badges mapping to flyecon BuyPlan enum

### Tech Stack (research-techstack.md)
- **Recommendation: Hand-rolled Canvas 2D, zero dependencies** for neural activity — deterministic layered layout (Input → KC → MBON → PPL → DN columns)
- Plotly.js already inlined at 4.1MB — can handle pie/bar/scatter for economy viz
- SSE not needed — polling is fine since bottleneck is LIF sim rate, not transport
- SVG filters (feGaussianBlur + feColorMatrix) for fly glow — animate opacity/transform only (compositor-friendly)

### Pitfalls (research-pitfalls.md)
- **Critical: dashboard_server.py boots with n_neurons=200 (synthetic) by default** — full 166K connectome is NOT wired up. Design for 200, not 166K.
- LIF simulate cost: 5.2ms at 200 neurons — effectively free at 1.5s poll rate
- **Thread safety: NEVER call policy/LIF from FastAPI handler** — sample inside _training_loop, publish through existing _metrics_lock
- FCI "neural" input is fake (scaled reward, not KC rate) — leave it, create honest new signal instead
- eval/score.py doesn't test dashboard_server.py — zero automated eval coverage for this task

- **Instructions for next step:** Strategist should produce ONE hypothesis covering all three features as a single build. Key constraints to embed in the plan: (1) all sampling inside _training_loop thread, (2) Canvas 2D for connectome viz, (3) Plotly.js for economy charts, (4) SVG filter for glow, (5) single self-contained HTML file constraint, (6) target 200-neuron synthetic mode.
