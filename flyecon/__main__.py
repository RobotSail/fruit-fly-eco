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


def _etl_download(cache_dir: str = "cache/connectome/") -> int:
    """Download Feather tables from GCS."""
    from flyecon.etl.loader import download_feather_tables

    paths = download_feather_tables(cache_dir=cache_dir)
    print("Downloaded Feather tables:")
    for key, path in paths.items():
        size_mb = path.stat().st_size / 1e6 if path.exists() else 0
        print(f"  {key}: {path} ({size_mb:.1f} MB)")
    return 0


def _etl_load(cache_dir: str = "cache/connectome/", subcircuit: str | None = None) -> int:
    """Load connectome, optionally extract subcircuit."""
    from flyecon.etl.loader import load_connectome

    conn = load_connectome(cache_dir=cache_dir)
    print(f"Loaded connectome: {conn.n_neurons} neurons, {conn.n_synapses} synapses")
    print(f"  Weight matrix: {conn.weight_matrix.shape}")
    print(f"  Dopamine matrix: {conn.dopamine_matrix.shape}")
    print(f"  Fast edges: {conn.metadata.get('fast_edges', '?')}")
    print(f"  Dopamine edges: {conn.metadata.get('dopamine_edges', '?')}")

    if subcircuit == "mushroom-body":
        from flyecon.etl.subcircuit import extract_mushroom_body

        sub = extract_mushroom_body(conn, hops=1)
        print(f"\nMushroom body subcircuit: {sub.n_neurons} neurons, {sub.n_synapses} synapses")
        print(f"  Weight matrix: {sub.weight_matrix.shape}")
        print(f"  Neuron types: {len(sub.neuron_types)}")
    elif subcircuit is not None:
        from flyecon.etl.subcircuit import extract_by_roi

        sub = extract_by_roi(conn, subcircuit)
        print(
            f"\nROI '{subcircuit}' subcircuit: "
            f"{sub.n_neurons} neurons, {sub.n_synapses} synapses"
        )

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
        print("  etl download          Download Feather tables from GCS")
        print("    --cache-dir DIR     Cache directory (default: cache/connectome/)")
        print("  etl load              Load connectome from cached Feather tables")
        print("    --subcircuit NAME   Extract subcircuit (e.g. mushroom-body)")
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

    if args[0] == "etl":
        if len(args) < 2:
            print("Usage: python -m flyecon etl {download|load}")
            return 1

        cache_dir = "cache/connectome/"
        if "--cache-dir" in args:
            idx = args.index("--cache-dir")
            if idx + 1 < len(args):
                cache_dir = args[idx + 1]

        if args[1] == "download":
            return _etl_download(cache_dir=cache_dir)

        if args[1] == "load":
            subcircuit = None
            if "--subcircuit" in args:
                idx = args.index("--subcircuit")
                if idx + 1 < len(args):
                    subcircuit = args[idx + 1]
            return _etl_load(cache_dir=cache_dir, subcircuit=subcircuit)

        print(f"Unknown etl command: {args[1]}")
        return 1

    print(f"Unknown command: {args[0]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
