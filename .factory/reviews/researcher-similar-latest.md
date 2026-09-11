# Researcher Agent Output

- **timestamp:** 2026-09-11T15:22:19Z
- **exit_code:** 0

---

The research report has been written to `.factory/strategy/research-similar.md`. Summary of what I found and how I got there:

**Methodology note:** No `WebSearch`/`WebFetch` tools were available in this session — only `Bash`, `Edit`, `Read`. General search engines (DuckDuckGo, Bing) served bot-detection challenge/decoy pages when scraped via `curl` (verified — Bing returned "Hospital - Wikipedia" for a totally unrelated query). I pivoted to fetching the user-cited HF Space's actual source directly (it's a `static` Space, fully inspectable) and to GitHub's public search API, which worked reliably.

**Key findings:**
1. **`ngxson/fly-llm-demo`** (the cited reference) — read its actual `brain.js`/`Brain.svelte` source. It's a GPU point-cloud (Three.js shader) using real anatomical neuron coordinates, with a "fake but convincing" propagation animation (colors sweep outward by precomputed synaptic hop-distance, even though the real state update is instant) — the single most useful technique to steal for the connectome panel.
2. **`Xenova/fruit-fly-simulation`** — its inspiration source, a fully rigged NeuroMechFly 3D fly body with a neural-activity→motion controller using exponential smoothing (`relax()`) — directly solves the "avatar will jitter every 1.5s poll" problem.
3. **`erojasoficial-byte/fly-brain`** (42★, GitHub) — an academic whole-connectome sim with a `brain_monitor.py` sci-fi "glow/pulse" neural monitor — closest prior art for the glow/pulse avatar ask.
4. **`Twoos123/cs2-meta-engine`** — a real CS2 economy tracker whose buy-type badges (Eco/Force/Half/Full) map almost 1:1 onto flyecon's existing `BuyPlan` enum.
5. **No archive/prior knowledge** existed for this project — confirmed empty.

Every finding is mapped back to `study-combined.md`'s concrete tap points (`dashboard_server.py`, `svg.py`, `ppo.py`, `harness.py`) so the Strategist can turn this directly into an implementable hypothesis.
---

> **⚠ CEO IDENTITY RE-ANCHOR (Sacred Rule 8)**
> You are the Factory CEO. You orchestrate, delegate, and decide. You do NOT implement.
> If you are about to write code, run tests, do research, or fix bugs — STOP and spawn the appropriate agent.
> Re-read your Permitted/Forbidden Actions lists in the Identity section above.
