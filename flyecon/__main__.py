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


def _load_connectome_by_name(name: str) -> object:
    """Load connectome, optionally extracting a subcircuit."""
    from flyecon.etl.loader import load_connectome

    conn = load_connectome()

    if name == "mushroom-body":
        from flyecon.etl.subcircuit import extract_mushroom_body

        return extract_mushroom_body(conn, hops=1)
    elif name == "full":
        return conn
    else:
        from flyecon.etl.subcircuit import extract_by_roi

        return extract_by_roi(conn, name)


def _sim_calibrate(connectome_name: str) -> int:
    """Run gain calibration on the specified connectome."""
    from flyecon.sim.calibration import calibrate_gain

    print(f"Loading connectome: {connectome_name}")
    conn = _load_connectome_by_name(connectome_name)

    print("Running gain calibration...")
    result = calibrate_gain(conn, target_rate_hz=(1.0, 10.0), duration_ms=1000.0)

    print(f"\nCalibration result:")
    print(f"  Gain:       {result.gain:.8f}")
    print(f"  Mean rate:  {result.mean_rate_hz:.2f} Hz")
    print(f"  Healthy:    {result.is_healthy}")
    print(f"  Trials:     {len(result.trials)}")
    for i, trial in enumerate(result.trials):
        print(f"    [{i}] gain={trial['gain']:.8f}  rate={trial['rate_hz']:.2f} Hz")
    return 0


def _sim_preflight(connectome_name: str, gain: float | None) -> int:
    """Run pre-flight gate on the specified connectome."""
    from flyecon.sim.calibration import calibrate_gain, preflight_gate

    print(f"Loading connectome: {connectome_name}")
    conn = _load_connectome_by_name(connectome_name)

    if gain is None:
        print("No gain specified — running calibration first...")
        cal = calibrate_gain(conn, target_rate_hz=(1.0, 10.0), duration_ms=1000.0)
        gain = cal.gain
        print(f"  Calibrated gain: {gain:.8f} (rate: {cal.mean_rate_hz:.2f} Hz)")

    print(f"\nRunning pre-flight gate with gain={gain:.8f}...")
    result = preflight_gate(conn, gain=gain)

    print(f"\nPre-flight results:")
    print(f"  Gate 1 (zero-input stability):  {'PASS' if result.gate1_pass else 'FAIL'}")
    print(f"  Gate 2 (non-degenerate response): {'PASS' if result.gate2_pass else 'FAIL'}")
    print(f"  Gate 3 (state discrimination):  {'PASS' if result.gate3_pass else 'FAIL'}")
    print(f"  Overall: {'ALL PASS' if result.all_pass else 'FAILED'}")

    for k, v in result.diagnostics.items():
        print(f"    {k}: {v}")
    return 0 if result.all_pass else 1


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


def _encoding_test_discrimination(connectome_name: str, gain: float | None) -> int:
    """Encode 10 economy states, simulate, read out, report pairwise distances."""
    import torch

    from flyecon.encoding.population import (
        N_INPUT_NEURONS,
        PopulationEncoder,
        map_to_connectome_inputs,
    )
    from flyecon.readout.linear import SpikeReadout
    from flyecon.sim.calibration import calibrate_gain
    from flyecon.sim.lif import LIFNetwork
    from flyecon.state.economy import EconomyState

    # Load connectome
    print(f"Loading connectome: {connectome_name}")
    conn = _load_connectome_by_name(connectome_name)

    # Calibrate gain if not provided
    if gain is None:
        print("No gain specified — running calibration first...")
        cal = calibrate_gain(conn, target_rate_hz=(1.0, 10.0), duration_ms=1000.0)
        gain = cal.gain
        print(f"  Calibrated gain: {gain:.8f} (rate: {cal.mean_rate_hz:.2f} Hz)")

    # Build encoder
    encoder = PopulationEncoder()

    # Pick input neuron IDs (first N_INPUT_NEURONS neurons in the connectome)
    n_inputs = min(N_INPUT_NEURONS, conn.n_neurons)
    input_ids = list(range(n_inputs))

    # Define 10 diverse economy states
    test_states = [
        EconomyState(money=800, loss_streak=0, round_number=1, half=0, opponent_loss_streak=0),
        EconomyState(money=16000, loss_streak=0, round_number=12, half=0, opponent_loss_streak=4),
        EconomyState(money=4000, loss_streak=2, round_number=5, half=0, opponent_loss_streak=1),
        EconomyState(money=8000, loss_streak=0, round_number=8, half=1, opponent_loss_streak=0),
        EconomyState(money=2000, loss_streak=4, round_number=3, half=0, opponent_loss_streak=0),
        EconomyState(money=6000, loss_streak=1, round_number=6, half=0, opponent_loss_streak=2),
        EconomyState(money=10000, loss_streak=0, round_number=10, half=1, opponent_loss_streak=3),
        EconomyState(money=1400, loss_streak=3, round_number=2, half=0, opponent_loss_streak=0),
        EconomyState(money=12000, loss_streak=0, round_number=11, half=1, opponent_loss_streak=1),
        EconomyState(money=5000, loss_streak=1, round_number=7, half=0, opponent_loss_streak=4),
    ]

    # Encode and simulate each state
    duration_ms = 400.0
    net = LIFNetwork(conn, gain=gain)
    readout_vectors: list[torch.Tensor] = []

    # Get baseline rates for readout
    baseline_input = torch.zeros(conn.n_neurons, dtype=torch.float32)
    baseline_counts = net.simulate(baseline_input, duration_ms, jitter=True)
    baseline_rates = baseline_counts / (duration_ms / 1000.0)

    # Pick output neuron IDs (last min(20, n_neurons) neurons)
    n_outputs = min(20, conn.n_neurons)
    output_ids = list(range(conn.n_neurons - n_outputs, conn.n_neurons))
    output_baselines = baseline_rates[output_ids]

    readout = SpikeReadout(
        output_neuron_ids=output_ids,
        baseline_rates=output_baselines,
    )

    print(f"\nEncoding {len(test_states)} economy states through the pipeline...")
    print(f"  Connectome: {conn.n_neurons} neurons")
    print(f"  Input neurons: {n_inputs}, Output neurons: {n_outputs}")
    print(f"  Duration: {duration_ms} ms, Gain: {gain:.8f}")
    print()

    for i, state in enumerate(test_states):
        encoded = encoder.encode(state)
        # Pad encoded if fewer input IDs than N_INPUT_NEURONS
        if n_inputs < N_INPUT_NEURONS:
            encoded = encoded[:n_inputs]
        full_input = map_to_connectome_inputs(encoded, input_ids, conn.n_neurons)
        spike_counts = net.simulate(full_input, duration_ms, jitter=True)

        with torch.no_grad():
            logits = readout(spike_counts, duration_ms)
        readout_vectors.append(logits)

        print(
            f"  State {i}: money=${state.money:>5}, streak={state.loss_streak}, "
            f"round={state.round_number:>2}, half={state.half} "
            f"→ logits=[{', '.join(f'{v:.3f}' for v in logits.tolist())}]"
        )

    # Compute pairwise distances
    print("\nPairwise L2 distances between readout vectors:")
    stacked = torch.stack(readout_vectors)
    n = len(readout_vectors)
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(torch.norm(stacked[i] - stacked[j]).item())
            print(f"  states({i},{j}): {dist:.4f}")

    mean_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            mean_dist += float(torch.norm(stacked[i] - stacked[j]).item())
            count += 1
    mean_dist /= max(count, 1)
    print(f"\nMean pairwise distance: {mean_dist:.4f}")
    print(f"Result: {'PASS (>0)' if mean_dist > 0 else 'FAIL (no discrimination)'}")

    return 0


def _train(
    connectome_name: str,
    iterations: int,
    eval_interval: int,
) -> int:
    """Run PPO training."""
    from flyecon.policy.ppo import PPOTrainer, build_fly_policy

    if connectome_name == "test":
        from flyecon.etl.controls import random_sparse

        conn = random_sparse(n_neurons=50, density=0.1, seed=42)
        gain = 0.0
        print("Using 50-neuron random test connectome (gain=0.0)")
    else:
        conn = _load_connectome_by_name(connectome_name)
        from flyecon.sim.calibration import calibrate_gain

        print(f"Calibrating gain for {connectome_name}...")
        cal = calibrate_gain(conn, target_rate_hz=(1.0, 10.0), duration_ms=1000.0)
        gain = cal.gain
        print(f"  Gain: {gain:.8f} (rate: {cal.mean_rate_hz:.2f} Hz)")

    print("Building FlyPolicy...")
    policy = build_fly_policy(conn, gain=gain)

    # Optionally solve Oracle for periodic evaluation
    oracle = None
    if eval_interval > 0:
        from flyecon.oracle.mdp import EconomyMDP
        from flyecon.oracle.solver import solve

        print("Solving Oracle MDP...")
        mdp = EconomyMDP()
        oracle = solve(mdp, gamma=0.99, tol=1e-8)
        print(f"  Oracle solved in {oracle.iterations} iterations")

    print(f"\nTraining for {iterations} iterations "
          f"(eval every {eval_interval})...\n")
    trainer = PPOTrainer(policy, n_steps=128)
    tlog = trainer.train(
        n_iterations=iterations,
        eval_interval=eval_interval,
        oracle=oracle,
    )

    print(f"\n{'='*50}")
    print(f"Training complete: {iterations} iterations")
    if tlog.metrics:
        last = tlog.metrics[-1]
        for k, v in last.items():
            print(f"  {k}: {v:.4f}")
    if tlog.eval_results:
        last_eval = tlog.eval_results[-1]
        print("\nLast evaluation:")
        print(f"  Value ratio: {last_eval.value_ratio:.4f}")
        print(f"  Agreement:   {last_eval.agreement_rate:.4f}")
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
        print("  sim calibrate         Run gain calibration on a connectome")
        print("    --connectome NAME   Connectome to use (default: mushroom-body)")
        print("  sim preflight         Run pre-flight gate checks")
        print("    --connectome NAME   Connectome to use (default: mushroom-body)")
        print("    --gain VALUE        Gain value (auto-calibrates if omitted)")
        print("  encoding test-discrimination")
        print("                        Encode states, simulate, report discrimination")
        print("    --connectome NAME   Connectome to use (default: mushroom-body)")
        print("    --gain VALUE        Gain value (auto-calibrates if omitted)")
        print("  train                 Run PPO training")
        print("    --connectome NAME   Connectome to use (default: mushroom-body, or 'test')")
        print("    --iterations N      Number of training iterations (default: 100)")
        print("    --eval-interval N   Evaluate against Oracle every N iterations (default: 50)")
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

    if args[0] == "sim":
        if len(args) < 2:
            print("Usage: python -m flyecon sim {calibrate|preflight}")
            return 1

        connectome_name = "mushroom-body"
        if "--connectome" in args:
            idx = args.index("--connectome")
            if idx + 1 < len(args):
                connectome_name = args[idx + 1]

        if args[1] == "calibrate":
            return _sim_calibrate(connectome_name)

        if args[1] == "preflight":
            gain_val: float | None = None
            if "--gain" in args:
                idx = args.index("--gain")
                if idx + 1 < len(args):
                    gain_val = float(args[idx + 1])
            return _sim_preflight(connectome_name, gain_val)

        print(f"Unknown sim command: {args[1]}")
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

    if args[0] == "encoding":
        if len(args) < 2:
            print("Usage: python -m flyecon encoding {test-discrimination}")
            return 1

        connectome_name = "mushroom-body"
        if "--connectome" in args:
            idx = args.index("--connectome")
            if idx + 1 < len(args):
                connectome_name = args[idx + 1]

        if args[1] == "test-discrimination":
            gain_val: float | None = None
            if "--gain" in args:
                idx = args.index("--gain")
                if idx + 1 < len(args):
                    gain_val = float(args[idx + 1])
            return _encoding_test_discrimination(connectome_name, gain_val)

        print(f"Unknown encoding command: {args[1]}")
        return 1

    if args[0] == "train":
        connectome_name = "mushroom-body"
        if "--connectome" in args:
            idx = args.index("--connectome")
            if idx + 1 < len(args):
                connectome_name = args[idx + 1]
        iterations = 100
        if "--iterations" in args:
            idx = args.index("--iterations")
            if idx + 1 < len(args):
                iterations = int(args[idx + 1])
        eval_interval = 50
        if "--eval-interval" in args:
            idx = args.index("--eval-interval")
            if idx + 1 < len(args):
                eval_interval = int(args[idx + 1])
        return _train(connectome_name, iterations, eval_interval)

    print(f"Unknown command: {args[0]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
