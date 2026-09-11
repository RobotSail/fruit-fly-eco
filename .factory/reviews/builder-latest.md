# Builder Review — Phase 2: Baseline Oracle

## Summary
Implemented the Baseline Oracle as a NumPy value-iteration solver over the
MR12 economy MDP. The Oracle is a standalone, zero-connectome-dependency
component that solves the full state space (~96,600 states) and produces
optimal buy-plan decisions.

## Changes

### flyecon/oracle/mdp.py (new)
- `EconomyMDP` class with full state space: money (161 buckets) × loss_streak (5) × round (12) × half (2) × opp_loss_streak (5) = 96,600 states
- Parameterised win-probability model (`DEFAULT_WIN_PROBS`) — tunable, documented
- `transition()` delegates to `flyecon.state.economy.step()` — Oracle shares exact MR12 rules
- Affordability masking: unaffordable actions return empty transitions (solver uses -inf Q)
- Terminal state detection: round 12 half 1 is absorbing (match over)
- Reward: 1.0 for win, 0.0 for loss (Oracle maximises expected round wins)

### flyecon/oracle/solver.py (new)
- `solve(mdp, gamma=0.99, tol=1e-8)` → `Policy` via NumPy value iteration
- Per-state gamma: terminal states use γ=0, ensuring fast convergence (25 iterations)
- Vectorised VI loop: precomputed transition arrays for O(1) per-step
- `Policy` class with `decide(state)`, `value(state)`, `summary()`
- `simulate_episode()` stochastic simulation with win-probability model
- `verify_against_random()` → `VerificationResult` with wins and money metrics

### tests/test_oracle.py (new, 16 tests)
- State space size = 96,600
- Index roundtrip consistency
- Transition probabilities sum to 1 for affordable actions
- Convergence in <100 iterations (actual: 25)
- Policy never chooses FULL_BUY when money < $4,750
- Pistol round ($800) never picks FULL_BUY
- Rich state ($10,000) never picks SAVE
- Oracle dominates random by ≥1.0 rounds/match (actual: ~1.7)

### eval/score.py (updated oracle_solver stub)
- 4 checks: convergence, affordability, pistol sanity, Oracle vs random

### flyecon/__main__.py (updated CLI)
- `python -m flyecon oracle solve` — solve MDP, print summary, cache to checkpoints/
- `python -m flyecon oracle eval --episodes N` — Oracle vs random evaluation

## Key Design Decisions

1. **Round-win reward** (not money): Using raw money as reward makes SAVE dominant
   (not spending = more money). Round-win reward correctly incentivises the
   Oracle to manage money as a means to winning rounds.

2. **Terminal states**: Round 12 half 1 creates a self-loop in economy.step().
   Without γ=0 at terminal states, VI took 2,800+ iterations. With per-state
   gamma, convergence is 25 iterations.

3. **Affordability masking**: Unaffordable actions map to -inf Q values, ensuring
   the policy never recommends buying equipment the team can't afford.

4. **VerificationResult**: Returns both money and wins metrics. The Oracle
   wins ~1.7 more rounds/match than random but accumulates less money (it
   spends on equipment). The wins advantage × WIN_REWARD ≈ $5,500 money-equivalent.

## Metrics
- All 62 tests pass (46 Phase 1 + 16 Phase 2)
- eval/score.py: economy_mdp 1.0, oracle_solver 1.0
- Solve time: ~5 seconds (including precomputation)
- Convergence: 25 iterations
- Oracle wins advantage: ~1.7 rounds/match over random
