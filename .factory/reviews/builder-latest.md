# Builder Review — Phase 6: PPO Policy Head + Training Loop

## Summary
Implemented Phase 6 deliverables: FlyPolicy wrapping the fixed-reservoir
connectome pipeline, CleanRL-style PPO trainer, Oracle evaluation function,
CLI training command, and comprehensive tests.

## Files Changed
| File | Change | Scope |
|------|--------|-------|
| `flyecon/policy/ppo.py` | New — 468 lines | Modifiable surface |
| `tests/test_ppo.py` | New — 198 lines | Test file |
| `flyecon/__main__.py` | Added `train` CLI command | Issue scope |
| `flyecon/policy/__init__.py` | Updated docstring | Supporting change |
| `eval/score.py` | Activated fly_vs_oracle stub | Phase 6 deliverable |

## Architecture Decisions
1. **Fixed reservoir**: LIFNetwork is NOT an nn.Module — its connectome weights
   never appear in `policy.parameters()` and never receive gradients.
2. **Feature caching**: During rollout collection, baseline-normalised features
   are cached in the RolloutBuffer. During PPO update, only the readout
   projection (W, b) and value head MLP are recomputed — no LIF re-simulation.
3. **Value head**: MLP [n_output_features, 64, 1] shares features with the
   policy head but has separate trainable parameters.
4. **Baseline normalization**: Rates clamped to ≥1 Hz floor to prevent extreme
   normalization when baseline firing is near zero (important for small test
   networks with gain=0).

## Test Results
- 151/151 tests pass (8 new PPO tests + 143 existing)
- All eval dimensions score 1.0 (including newly activated fly_vs_oracle: 3/3)
- Aggregate eval score: 0.833 (5/6 dimensions active, dashboard_renders is Phase 8)
- CLI `python -m flyecon train --connectome test --iterations 3` works correctly

## Verification
- [x] FlyPolicy forward produces valid Categorical distributions (no NaN)
- [x] Gradients only flow to encoder/readout/value params, NOT connectome
- [x] PPO update reduces value loss on synthetic batch
- [x] Rollout collection produces correct buffer shape
- [x] 10-iteration integration test on 50-neuron network: no crash, metrics finite
- [x] eval/score.py fly_vs_oracle: policy valid, PPO update finite, training completes

## Risks
- Encoder params receive zero gradients in fixed-reservoir mode (by design —
  surrogate gradients are the Phase 3.8 plateau escalation)
- gain=0 test networks have identical spike patterns for all states (features
  only vary through encoder → input current mapping, not recurrent dynamics)
