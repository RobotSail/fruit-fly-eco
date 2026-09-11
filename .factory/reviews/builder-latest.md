# Builder Review — Phase 1: Project scaffold + eval harness

## Status: ✅ COMPLETE

## Deliverables

### 1. Project directory tree
- Created full 9-team architecture under `flyecon/`
- All modules have `__init__.py` stubs: etl, sim, encoding, readout, policy, oracle, avatar, dashboard, resilience, state, eval
- `flyecon/__init__.py` initializes structlog with JSON output
- `flyecon/__main__.py` CLI entry point stub

### 2. Core implementations
- **`flyecon/state/constants.py`** — All LIF parameters (V_REST_MV=-52.0, V_THRESH_MV=-45.0, TAU_MS=20.0, DT_MS=5.0), NT sign map (ACh:+1, GABA:-1, Glu:-1, His:-1, dopamine separate), MIN_CONFIDENCE=0.5, economy constants (money range, loss-bonus ladder, kill rewards, MR12 rules)
- **`flyecon/state/economy.py`** — EconomyState frozen dataclass, BuyPlan enum (5 plans), pistol_round_state() factory, step() MR12 transition function with full mechanics
- **`flyecon/eval/harness.py`** — evaluate_policy() with ScenarioResult and EvalResult dataclasses

### 3. pyproject.toml
- Dependencies: torch>=2.0, numpy>=1.24, pyarrow>=14.0, plotly>=5.18, jinja2>=3.1, structlog>=23.0
- Dev deps: pytest, mypy, ruff
- Python >=3.12, hatchling build system

### 4. eval/score.py
- 6 eval dimensions: economy_mdp (active, 12/12 checks), oracle_solver (stub), connectome_etl (stub), simulation_runs (stub), fly_vs_oracle (stub), dashboard_renders (stub)
- economy_mdp scores 1.0 — all MR12 rules verified

### 5. Tests
- `tests/conftest.py` — shared fixtures
- `tests/test_constants.py` — 19 tests: units consistency, subthreshold range, NT sign map coverage
- `tests/test_economy.py` — 27 tests: pistol start, loss-bonus ladder progression, win/loss mechanics, halftime reset, buy plan costs, state validation
- **46/46 tests pass**

### 6. factory.md
- Project configuration, guards, constraints, eval dimensions, modifiable/fixed surfaces, architecture diagram

## Verification
- `pytest`: 46/46 passed
- `ruff check`: clean
- `eval/score.py`: economy_mdp 1.0, aggregate 0.167 (5 stubs at 0.0)
- No fixed surface files modified
- All files within declared scope
