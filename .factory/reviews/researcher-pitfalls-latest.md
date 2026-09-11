# Researcher Agent Output

- **timestamp:** 2026-09-11T15:25:26Z
- **exit_code:** 0

---

## Summary

I've written `.factory/strategy/research-pitfalls.md` (374 lines) covering all six requested risk areas. Key findings, all empirically verified against the actual source (in the sibling worktree `run-b7b1625d`, since this worktree only has a stub graph):

1. **Performance**: `dashboard_server.py` boots `Harness()` with **zero args** → hardcoded `n_neurons=200`. Full 166K-neuron mode isn't wired into the live dashboard at all today. I ran live benchmarks: 200 neurons costs **5.2ms/call** (free at any poll rate); scaling is O(synapse-count), and extrapolating from measured data, the full MaleCNS connectome (~125M synapses) would cost an estimated **~6 seconds per call** — flagged as an explicit non-goal rather than something to solve.

2. **Thread safety**: Found a race condition beyond the `_metrics_lock` pattern — `LIFNetwork` mutates `V`/`refractory`/`spike_counts` in-place, shared between the training thread and any naive dashboard-triggered inspect call. The fix: all new sampling must happen *inside* `_training_loop`, publishing only plain Python values through the existing lock — never call `policy`/`_lif` from a FastAPI request handler or asyncio task.

3. **Single HTML constraint**: Measured directly — the dashboard **already inlines a 4.19MB Plotly.js bundle** to satisfy "self-contained HTML." A 250KB D3 bundle would be trivial by comparison. Recommended reusing already-loaded Plotly instead of adding D3, for scope reasons (not size reasons).

4. **Data size**: Recommend layer-level aggregates only, never raw per-neuron arrays, regardless of connectome size.

5. **FCI bug**: Found it's actually worse than described — 2 of 3 `compute_fci()` inputs (`kc_mean_rate` and `recent_reward`) are both derived from the same `mean_reward` scalar. Recommended leaving it untouched and feeding the avatar from a new, honestly-named signal instead.

6. **Scope**: Discovered `eval/score.py`'s `dashboard_renders` dimension only tests the static `renderer.py` path, never `dashboard_server.py` — zero automated eval coverage for this entire task. Provided a per-feature MVP table with explicit deferred items.
---

> **⚠ CEO IDENTITY RE-ANCHOR (Sacred Rule 8)**
> You are the Factory CEO. You orchestrate, delegate, and decide. You do NOT implement.
> If you are about to write code, run tests, do research, or fix bugs — STOP and spawn the appropriate agent.
> Re-read your Permitted/Forbidden Actions lists in the Identity section above.
