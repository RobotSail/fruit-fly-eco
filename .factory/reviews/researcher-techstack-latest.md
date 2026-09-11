# Researcher Agent Output

- **timestamp:** 2026-09-11T15:22:11Z
- **exit_code:** 0

---

The research report has been written to `.factory/strategy/research-techstack.md`. Summary of what I did and found:

## Approach
Since this session's toolset didn't include `WebSearch`/`WebFetch`, I compensated by:
1. Reading `study-combined.md` (which embeds the full `graph-context.md` architectural analysis) for exact data tap points, connectome node-count modes (200 / 2K-10K / 166,700), and existing dashboard code.
2. Reading the actual source files (`dashboard_server.py`, `flyecon/avatar/svg.py`, `flyecon/dashboard/renderer.py`) to ground every recommendation in real code structure (exact SVG element coordinates, existing poll loop, existing CSS-var pattern).
3. **Executing** `plotly.offline.get_plotlyjs()` directly to get ground-truth bundle size (4.19MB already inlined!) and confirm `pie`/`bar`/`scatter` trace types are present.
4. Fetching primary sources via `curl` (raw GitHub READMEs, MDN's markdown source, actual npm bundle files for d3-force, three.js, force-graph, sigma.js) since search engines blocked scraping.

## Key findings
- **Neural viz**: Hand-rolled Canvas 2D (zero dependencies) beats D3/SVG-DOM (degrades past ~1-3K DOM nodes) and WebGL (unneeded complexity for ≤10K nodes) — verified via `vasturiano/force-graph`'s public ~75K-element canvas demo.
- **Charts**: Plotly.js is *already* fully capable (verified `pie`/`bar`/`scatter` in the inlined bundle) — no new library needed.
- **SSE vs polling**: Polling wins — the bottleneck is the LIF simulation's own throttled sampling rate, not transport latency; "liveliness" should come from CSS `@keyframes` running continuously between polls (matching the existing wing-beat pattern).
- **SVG filters**: `feGaussianBlur`+`feColorMatrix` for the glow halo, but critically — animate `opacity`/`transform` (compositor-friendly, matches the project's own stated design principle in `svg.py`), never animate `stdDeviation` directly (forces repaint).
- **Single-file constraint**: Non-issue — the project already inlines a 4.1MB Plotly bundle; every option here is 1-2 orders of magnitude smaller.
---

> **⚠ CEO IDENTITY RE-ANCHOR (Sacred Rule 8)**
> You are the Factory CEO. You orchestrate, delegate, and decide. You do NOT implement.
> If you are about to write code, run tests, do research, or fix bugs — STOP and spawn the appropriate agent.
> Re-read your Permitted/Forbidden Actions lists in the Identity section above.
