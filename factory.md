# factory.md — PROJECT FLY//ECON

## Project

- **Name:** flyecon
- **Goal:** Build a harness that runs CS MR12 economy decisions through a Drosophila MaleCNS v1.0 connectome (166,700 neurons). Fly must beat the Baseline Oracle on held-out economy states.
- **Language:** Python 3.12
- **Package manager:** uv
- **Test framework:** pytest
- **Linter:** ruff
- **Type checker:** mypy

## Guards

- Economy simulator must match exact MR12 rules (loss bonus ladder, kill rewards, halftime reset)
- LIF parameters: V_rest=-52mV, V_thresh=-45mV, tau=20ms
- Neurotransmitter signing: ACh=excitatory, GABA=inhibitory, Glutamate=inhibitory (Drosophila)
- Dashboard must be a single self-contained HTML file

## Constraints

- No GPU reservation active — all development uses CPU
- Python 3.12, numpy 2.5, scipy 1.18, PyTorch 2.13 (CPU)
- Large data files go in ~/data/
- Use `uv` for package management
- Type-hinted Python, no classes where functions suffice
- Keep simulation hot-path clean — no allocations in spike loop

## Eval Dimensions

| Dimension | File | Status |
|---|---|---|
| economy_mdp | eval/score.py | ✅ Active (Phase 1) |
| oracle_solver | eval/score.py | ⬚ Stub (Phase 2) |
| connectome_etl | eval/score.py | ⬚ Stub (Phase 3) |
| simulation_runs | eval/score.py | ⬚ Stub (Phase 4) |
| fly_vs_oracle | eval/score.py | ⬚ Stub (Phase 6) |
| dashboard_renders | eval/score.py | ⬚ Stub (Phase 8) |

## Modifiable Surfaces

- `flyecon/encoding/population.py` — encoding scheme, population code parameters
- `flyecon/readout/linear.py` — readout architecture, normalization
- `flyecon/policy/ppo.py` — PPO hyperparameters, reward shaping
- `flyecon/sim/calibration.py` — gain calibration targets
- `flyecon/state/constants.py` — LIF parameters (within biologically plausible ranges)
- `flyecon/avatar/ascii.py` — state→pose map
- `flyecon/dashboard/fci.py` — FCI weights and thresholds

## Fixed Surfaces (do not modify)

- `eval/score.py` — factory eval harness
- `flyecon/oracle/mdp.py` — economy MDP definition (ground truth)
- `flyecon/oracle/solver.py` — value iteration solver (incorruptible baseline)
- `flyecon/state/economy.py` — MR12 economy mechanics
- `flyecon/eval/harness.py` — evaluation harness
- `cache/connectome/*.feather` — raw connectome data
- `.factory/` — factory state

## Architecture

```
flyecon/
├── __init__.py          # structlog init, version
├── __main__.py          # CLI entry point
├── etl/                 # ETL-Team: Feather table ingestion
├── sim/                 # Simulation-Team: LIF stepper, sparse CSR ops
├── encoding/            # Encoding-Team: state → spike-current mapping
├── readout/             # Readout-Team: DN spike bins → action features
├── policy/              # Policy-Team: PPO + action head
├── oracle/              # Oracle-Team: value-iteration MDP solver
├── avatar/              # Avatar-Team: fly rendering (ASCII/2D/3D)
├── dashboard/           # Dashboard-Team: 10-panel HTML generator
├── resilience/          # Resilience-Team: checkpoint, heartbeat, watchdog
├── state/               # Shared state definitions
│   ├── constants.py     # LIF constants, NT sign map, economy rules
│   └── economy.py       # EconomyState dataclass, MR12 step() function
└── eval/                # Eval harness
    └── harness.py       # evaluate_policy() → EvalResult
tests/
├── conftest.py
├── test_constants.py
└── test_economy.py
eval/
└── score.py             # Factory eval: 6 dimensions
```
