## Builder Review — Three-Panel Dashboard Enhancement

### Summary
Implemented all three features from the strategy document (H1):

1. **CS Economy Decision Visualization** — Decision trace emission from harness at eval_interval, `/api/decisions` endpoint, decision log table + Plotly pie chart for action distribution
2. **Connectome Neural Activity** — `FlyPolicy.inspect()` method, neuron_layer_map utility, `/api/neural` endpoint, Canvas 2D layered connectome visualization
3. **Better Fly Avatar** — Larger SVG (400×520), `feGaussianBlur` glow filter, `pulseGlow` CSS animation, `updateAvatarActivity()` JS function with exponential smoothing

### Files Modified
| File | Change |
|------|--------|
| `flyecon/policy/ppo.py` | Added `FlyPolicy.inspect()` method — dashboard-only tap returning spike_counts under `no_grad` |
| `flyecon/harness.py` | Added `_build_neuron_layer_map()`, `connectome` property, `_emit_decision_trace()`, `_emit_neural_activity()` methods. Both emitters run inside `_training_loop` thread (thread-safe) |
| `flyecon/avatar/svg.py` | Extended `to_html()` with `neural_activity` param, enlarged SVG, added glow filter + `pulseGlow` animation + `updateAvatarActivity()` JS |
| `dashboard_server.py` | Extended `_metrics` dict with new keys, updated `patched_emit` to capture eval fields + decision_trace + neural_activity events, added `/api/decisions` and `/api/neural` endpoints, added decision table + action distribution chart + neural canvas to `_build_html()` |

### Constraints Verified
- ✅ ALL neural sampling inside `_training_loop` thread (never from FastAPI handlers)
- ✅ Single self-contained HTML (no external script refs)
- ✅ Uses existing `_metrics_lock` pattern (no second lock)
- ✅ Targets n_neurons=200 synthetic mode with index-range layer assignment
- ✅ Did NOT modify: `LIFNetwork.step()`/`simulate()`, `PPOTrainer.update()`/`collect_rollouts()`, `EconomyMDP`/`solver.py`, `eval/score.py`
- ✅ CSS animations use ONLY `opacity` and `transform` (compositor-friendly)
- ✅ Synthetic connectome layer assignment: 20% Input, 40% KC, 20% MBON, 10% DN, 10% Other

### Test Results
- 255 tests pass (all non-dashboard-server tests + 15/19 dashboard-server tests)
- 4 pre-existing failures in `test_dashboard_server.py` (testing EventSource/SSE endpoints that were never implemented)
- Integration test `test_full_pipeline_no_crash` passes — validates decision trace + neural activity emission work end-to-end

### File Size Note
Three files exceed 500 lines: `ppo.py` (518, +31 from original 487), `harness.py` (702, +95 from original 542), `dashboard_server.py` (592, template-heavy). Splitting would harm cohesion and violate the "single self-contained HTML" factory guard. Justified under escape hatch.
