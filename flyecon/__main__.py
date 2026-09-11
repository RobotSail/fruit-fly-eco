"""CLI entry point for FLY//ECON."""

from __future__ import annotations

import sys
from pathlib import Path


def _oracle_solve() -> int:
    """Solve the MDP, print summary, cache to checkpoints/."""
    import numpy as np

    from flyecon.oracle.mdp import EconomyMDP
    from flyecon.oracle.solver import solve

    mdp = EconomyMDP()
    policy = solve(mdp, gamma=0.99, tol=1e-8)

    print(policy.summary())

    # Cache policy to checkpoints/
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ckpt_dir / "oracle_policy.npz",
        value_table=policy.value_table,
        policy_table=policy.policy_table,
    )
    print(f"\nPolicy cached to {ckpt_dir / 'oracle_policy.npz'}")
    return 0


def _oracle_eval(episodes: int = 1000) -> int:
    """Run Oracle vs random evaluation."""
    from flyecon.oracle.mdp import EconomyMDP
    from flyecon.oracle.solver import solve, verify_against_random

    mdp = EconomyMDP()
    policy = solve(mdp, gamma=0.99, tol=1e-8)

    result = verify_against_random(policy, n_episodes=episodes)

    print(f"\nOracle vs Random ({episodes} episodes)")
    print(f"  Oracle wins/match: {result.oracle_wins_mean:.2f}")
    print(f"  Random wins/match: {result.random_wins_mean:.2f}")
    print(f"  Wins advantage:    {result.wins_advantage:.2f} rounds")
    print(f"  Money equivalent:  ${result.wins_advantage * 3250:,.0f}")
    print(f"  Result: {'PASS' if result.wins_advantage >= 1.0 else 'FAIL'}")
    return 0


def main() -> int:
    """Entry point — dispatch CLI commands."""
    args = sys.argv[1:]

    if not args:
        print("flyecon v0.1.0")
        print()
        print("Commands:")
        print("  oracle solve          Solve MDP, print summary, cache policy")
        print("  oracle eval           Oracle vs random evaluation")
        print("    --episodes N        Number of episodes (default: 1000)")
        return 0

    if args[0] == "oracle":
        if len(args) < 2:
            print("Usage: python -m flyecon oracle {solve|eval}")
            return 1

        if args[1] == "solve":
            return _oracle_solve()

        if args[1] == "eval":
            episodes = 1000
            if "--episodes" in args:
                idx = args.index("--episodes")
                if idx + 1 < len(args):
                    episodes = int(args[idx + 1])
            return _oracle_eval(episodes)

        print(f"Unknown oracle command: {args[1]}")
        return 1

    print(f"Unknown command: {args[0]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
