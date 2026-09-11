"""CleanRL-style discrete-action PPO with fixed-reservoir connectome.

FlyPolicy wraps PopulationEncoder + LIFNetwork + SpikeReadout.
Connectome weights are FROZEN — only encoder gains, readout weights,
and value head participate in PPO updates.  The LIF simulation runs
under ``torch.no_grad`` so gradients never flow through the spiking
network.  Features are cached during rollout collection and reused
during PPO update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from flyecon.encoding.population import (
    N_INPUT_NEURONS,
    PopulationEncoder,
    map_to_connectome_inputs,
)
from flyecon.readout.linear import DEFAULT_BIN_MS, SpikeReadout
from flyecon.sim.lif import LIFNetwork
from flyecon.state.constants import BUY_PLAN_COSTS, MAX_MONEY
from flyecon.state.economy import BuyPlan, EconomyState, pistol_round_state, step

if TYPE_CHECKING:
    from flyecon.etl.loader import Connectome

log = structlog.get_logger()

_EPS: float = 1e-6


# ── Data containers ──────────────────────────────────────────────────────────


@dataclass
class RolloutBuffer:
    """Collected rollout data for one PPO iteration."""

    states: list[EconomyState]
    actions: torch.Tensor       # [n_steps] long
    rewards: torch.Tensor       # [n_steps]
    values: torch.Tensor        # [n_steps]
    log_probs: torch.Tensor     # [n_steps]
    dones: torch.Tensor         # [n_steps]
    features: torch.Tensor      # [n_steps, n_features]


@dataclass
class EvalMetrics:
    """Results from :func:`evaluate_against_oracle`."""

    policy_mean_money: float
    oracle_mean_money: float
    value_ratio: float
    agreement_rate: float
    n_episodes: int


@dataclass
class TrainingLog:
    """Accumulated metrics from a full training run."""

    iterations: int
    metrics: list[dict[str, float]] = field(default_factory=list)
    eval_results: list[EvalMetrics] = field(default_factory=list)


# ── FlyPolicy ─────────────────────────────────────────────────────────────────


class FlyPolicy(nn.Module):
    """Fixed-reservoir policy: Encoder → LIF → Readout → actions + value.

    ``LIFNetwork`` is **not** an ``nn.Module``, so connectome weights
    never appear in ``parameters()`` and never receive gradients.
    """

    def __init__(
        self,
        encoder: PopulationEncoder,
        readout: SpikeReadout,
        lif_network: LIFNetwork,
        input_neuron_ids: list[int],
        duration_ms: float = DEFAULT_BIN_MS,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.readout = readout
        self._lif = lif_network  # NOT nn.Module → out of parameters()
        self.input_neuron_ids = input_neuron_ids
        self.duration_ms = duration_ms

        # Value head: [n_output_neurons, 64, 1]
        n_feat = readout.n_output_neurons
        self.value_head = nn.Sequential(
            nn.Linear(n_feat, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        for layer in self.value_head:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def _extract_features(self, spike_counts: torch.Tensor) -> torch.Tensor:
        """Baseline-normalised readout features from spike counts."""
        ids = self.readout.output_neuron_ids
        out = spike_counts[:, ids] if spike_counts.dim() > 1 else spike_counts[ids]
        rates = out / (self.duration_ms / 1000.0)
        return (rates - self.readout.baseline_rates) / (
            self.readout.baseline_rates + _EPS
        )

    def forward(
        self, states: list[EconomyState],
    ) -> tuple[Categorical, torch.Tensor, torch.Tensor]:
        """States → (action distribution, values, features)."""
        encoded = self.encoder.encode_batch(states)
        n_in = len(self.input_neuron_ids)
        if n_in < encoded.shape[-1]:
            encoded = encoded[:, :n_in]
        full_input = map_to_connectome_inputs(
            encoded, self.input_neuron_ids, self._lif.n_neurons,
        )
        with torch.no_grad():
            spike_counts = self._lif.batched_simulate(
                full_input, self.duration_ms,
            )
        features = self._extract_features(spike_counts)
        logits = F.linear(features, self.readout.W, self.readout.b)
        values = self.value_head(features).squeeze(-1)
        return Categorical(logits=logits), values, features

    def forward_from_features(
        self, features: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        """Cached features → (distribution, values).  Used in PPO update."""
        logits = F.linear(features, self.readout.W, self.readout.b)
        values = self.value_head(features).squeeze(-1)
        return Categorical(logits=logits), values

    def inspect(
        self, state: EconomyState,
    ) -> dict:
        """Dashboard-only tap: run a single forward pass and return internals.

        Called exclusively from the training thread (never from FastAPI
        handlers).  Returns spike_counts, action, value, and features
        under ``torch.no_grad()``.
        """
        encoded = self.encoder.encode_batch([state])
        n_in = len(self.input_neuron_ids)
        if n_in < encoded.shape[-1]:
            encoded = encoded[:, :n_in]
        full_input = map_to_connectome_inputs(
            encoded, self.input_neuron_ids, self._lif.n_neurons,
        )
        with torch.no_grad():
            spike_counts = self._lif.batched_simulate(
                full_input, self.duration_ms,
            )
        features = self._extract_features(spike_counts)
        logits = F.linear(features, self.readout.W, self.readout.b)
        values = self.value_head(features).squeeze(-1)
        dist = Categorical(logits=logits)
        action = int(dist.probs.argmax().item())
        return {
            "spike_counts": spike_counts.squeeze(0),
            "action": action,
            "value": float(values.item()),
            "features": features.detach().squeeze(0),
        }

    def act(
        self, state: EconomyState,
    ) -> tuple[int, float, float, torch.Tensor]:
        """Sample an action.  Returns (action_idx, log_prob, value, feats)."""
        dist, values, features = self.forward([state])
        action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(values.item()),
            features.detach().squeeze(0),
        )


# ── PPOTrainer ────────────────────────────────────────────────────────────────


class PPOTrainer:
    """CleanRL-style PPO.  Defaults: lr=3e-4, γ=0.99, λ=0.95, ε=0.2,
    c_ent=0.01, c_vf=0.5, grad_clip=0.5, n_steps=128, epochs=4, bs=32.
    """

    def __init__(
        self,
        policy: FlyPolicy,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_steps: int = 128,
        n_epochs: int = 4,
        batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self._buy_plans = list(BuyPlan)
        from flyecon.oracle.mdp import DEFAULT_WIN_PROBS
        self._win_probs = dict(DEFAULT_WIN_PROBS)

    def _random_state(self) -> EconomyState:
        return EconomyState(
            money=int(self.rng.integers(0, 161)) * 100,
            loss_streak=int(self.rng.integers(0, 5)),
            round_number=int(self.rng.integers(1, 13)),
            half=int(self.rng.integers(0, 2)),
            opponent_loss_streak=int(self.rng.integers(0, 5)),
        )

    def _step_env(
        self, state: EconomyState, action_idx: int,
    ) -> tuple[EconomyState, float, bool]:
        bp = self._buy_plans[action_idx]
        if BUY_PLAN_COSTS[bp.value] > state.money:
            bp = BuyPlan.SAVE
        won = float(self.rng.random()) < self._win_probs[bp.value]
        nxt = step(state, bp, round_won=won)
        reward = (nxt.money - state.money) / MAX_MONEY
        done = state.round_number == 12 and state.half == 1
        return nxt, reward, done

    def collect_rollouts(self, n_steps: int | None = None) -> RolloutBuffer:
        """Collect *n_steps* transitions from the economy environment."""
        n_steps = n_steps or self.n_steps
        states_l: list[EconomyState] = []
        acts_l: list[int] = []
        rews_l: list[float] = []
        vals_l: list[float] = []
        lps_l: list[float] = []
        dones_l: list[float] = []
        feats_l: list[torch.Tensor] = []

        state = self._random_state()
        self.policy.eval()
        with torch.no_grad():
            for _ in range(n_steps):
                a, lp, v, feat = self.policy.act(state)
                nxt, rew, done = self._step_env(state, a)
                states_l.append(state)
                acts_l.append(a)
                rews_l.append(rew)
                vals_l.append(v)
                lps_l.append(lp)
                dones_l.append(float(done))
                feats_l.append(feat)
                state = self._random_state() if done else nxt
        self.policy.train()

        return RolloutBuffer(
            states=states_l,
            actions=torch.tensor(acts_l, dtype=torch.long),
            rewards=torch.tensor(rews_l, dtype=torch.float32),
            values=torch.tensor(vals_l, dtype=torch.float32),
            log_probs=torch.tensor(lps_l, dtype=torch.float32),
            dones=torch.tensor(dones_l, dtype=torch.float32),
            features=torch.stack(feats_l),
        )

    @staticmethod
    def _compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        last_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE → *(advantages, returns)*."""
        n = len(rewards)
        adv = torch.zeros(n, dtype=torch.float32)
        gae = 0.0
        for t in reversed(range(n)):
            nxt_val = last_value if t == n - 1 else values[t + 1].item()
            mask = 1.0 - dones[t].item()
            delta = rewards[t].item() + gamma * nxt_val * mask - values[t].item()
            gae = delta + gamma * gae_lambda * mask * gae
            adv[t] = gae
        return adv, adv + values

    def update(self, rollouts: RolloutBuffer) -> dict[str, float]:
        """Run one PPO update over the rollout buffer, return metrics."""
        adv, returns = self._compute_gae(
            rollouts.rewards, rollouts.values, rollouts.dones,
            self.gamma, self.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(rollouts.actions)
        idx = np.arange(n)
        tot = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "cf": 0.0}
        n_up = 0

        for _ in range(self.n_epochs):
            self.rng.shuffle(idx)
            for s in range(0, n, self.batch_size):
                b = idx[s : s + self.batch_size]
                bf, ba = rollouts.features[b], rollouts.actions[b]
                bolp, badv, bret = rollouts.log_probs[b], adv[b], returns[b]

                dist, vals = self.policy.forward_from_features(bf)
                nlp = dist.log_prob(ba)
                ent = dist.entropy()

                ratio = torch.exp(nlp - bolp)
                pg1 = -badv * ratio
                pg2 = -badv * torch.clamp(
                    ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef,
                )
                pg_loss = torch.max(pg1, pg2).mean()
                vf_loss = F.mse_loss(vals, bret)
                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * (-ent.mean())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    tot["pg"] += pg_loss.item()
                    tot["vf"] += vf_loss.item()
                    tot["ent"] += ent.mean().item()
                    tot["kl"] += ((ratio - 1) - ratio.log()).mean().item()
                    tot["cf"] += ((ratio - 1).abs() > self.clip_coef).float().mean().item()
                n_up += 1

        d = max(n_up, 1)
        return {
            "policy_loss": tot["pg"] / d,
            "value_loss": tot["vf"] / d,
            "entropy": tot["ent"] / d,
            "approx_kl": tot["kl"] / d,
            "clip_fraction": tot["cf"] / d,
            "mean_reward": float(rollouts.rewards.mean()),
            "mean_value": float(rollouts.values.mean()),
        }

    def train(
        self,
        n_iterations: int = 1000,
        eval_interval: int = 50,
        oracle: object | None = None,
    ) -> TrainingLog:
        """Collect → update → log → eval loop."""
        tlog = TrainingLog(iterations=n_iterations)
        for it in range(1, n_iterations + 1):
            rollouts = self.collect_rollouts()
            metrics = self.update(rollouts)
            metrics["iteration"] = float(it)
            tlog.metrics.append(metrics)
            log_m = {k: round(v, 4) for k, v in metrics.items() if k != "iteration"}
            log.info("ppo.update", iteration=it, **log_m)
            if oracle is not None and eval_interval > 0 and it % eval_interval == 0:
                ev = evaluate_against_oracle(self.policy, oracle, n_episodes=50)
                tlog.eval_results.append(ev)
                log.info("ppo.eval", iteration=it,
                         value_ratio=round(ev.value_ratio, 4),
                         agreement=round(ev.agreement_rate, 4))
        return tlog


# ── Oracle evaluation ─────────────────────────────────────────────────────────


def evaluate_against_oracle(
    policy: FlyPolicy,
    oracle: object,
    n_episodes: int = 200,
    seed: int = 123,
) -> EvalMetrics:
    """Compare policy vs Oracle over full MR12 episodes.

    Each side plays independently with stochastic outcomes.  Agreement
    rate is measured on the policy's trajectory: what fraction of the
    policy's decisions match the Oracle's recommendation for that state.
    """
    from flyecon.oracle.mdp import DEFAULT_WIN_PROBS

    rng = np.random.default_rng(seed)
    buy_plans = list(BuyPlan)
    win_probs = DEFAULT_WIN_PROBS

    policy_money = 0.0
    oracle_money = 0.0
    agreements = 0
    total_rounds = 0

    policy.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            for half in range(2):
                st = pistol_round_state(half=half)
                for rnd in range(1, 13):
                    dist, _, _ = policy.forward([st])
                    pa = int(dist.probs.argmax().item())
                    obp = oracle.decide(st)  # type: ignore[union-attr]
                    if pa == buy_plans.index(obp):
                        agreements += 1
                    total_rounds += 1
                    pbp = buy_plans[pa]
                    eff = pbp if BUY_PLAN_COSTS[pbp.value] <= st.money else BuyPlan.SAVE
                    nxt = step(st, eff, round_won=float(rng.random()) < win_probs[eff.value])
                    policy_money += nxt.money
                    st = nxt
                    if st.round_number == 1 and rnd < 12:
                        break
            for half in range(2):
                st = pistol_round_state(half=half)
                for rnd in range(1, 13):
                    obp = oracle.decide(st)  # type: ignore[union-attr]
                    eff = obp if BUY_PLAN_COSTS[obp.value] <= st.money else BuyPlan.SAVE
                    nxt = step(st, eff, round_won=float(rng.random()) < win_probs[eff.value])
                    oracle_money += nxt.money
                    st = nxt
                    if st.round_number == 1 and rnd < 12:
                        break
    policy.train()

    total_rounds = max(total_rounds, 1)
    pm = policy_money / max(n_episodes, 1)
    om = oracle_money / max(n_episodes, 1)
    return EvalMetrics(
        policy_mean_money=pm, oracle_mean_money=om,
        value_ratio=pm / (om + _EPS),
        agreement_rate=agreements / total_rounds,
        n_episodes=n_episodes,
    )


# ── Convenience builder ──────────────────────────────────────────────────────


def build_fly_policy(
    connectome: Connectome,
    gain: float,
    duration_ms: float = DEFAULT_BIN_MS,
    output_neuron_ids: list[int] | None = None,
) -> FlyPolicy:
    """Build a :class:`FlyPolicy` from a connectome and calibrated gain.

    Picks the first ``min(N_INPUT_NEURONS, n)`` neurons as inputs and
    the last ``min(20, n)`` as outputs (or *output_neuron_ids* if
    provided — used for full-connectome DN-class readout expansion).
    Baseline rates are clamped to ≥1 Hz to prevent extreme normalisation.

    Parameters
    ----------
    connectome : Connectome
        Signed sparse connectome.
    gain : float
        Calibrated synaptic gain.
    duration_ms : float
        Readout bin duration (ms).
    output_neuron_ids : list[int] | None
        Explicit output neuron indices. When ``None``, uses last-N.
        When provided (e.g. all DN-class neurons for full connectome),
        those indices are used directly.
    """
    encoder = PopulationEncoder()
    n = connectome.n_neurons
    n_inputs = min(N_INPUT_NEURONS, n)
    input_ids = list(range(n_inputs))

    lif = LIFNetwork(connectome, gain=gain)
    baseline_input = torch.full((n,), 10.0, dtype=torch.float32)
    baseline_counts = lif.simulate(baseline_input, duration_ms, jitter=True)
    baseline_rates = baseline_counts / (duration_ms / 1000.0)
    baseline_rates = torch.clamp(baseline_rates, min=1.0)

    if output_neuron_ids is not None and len(output_neuron_ids) > 0:
        output_ids = output_neuron_ids
    else:
        n_outputs = min(20, n)
        output_ids = list(range(n - n_outputs, n))

    readout = SpikeReadout(
        output_neuron_ids=output_ids,
        baseline_rates=baseline_rates[output_ids],
    )
    return FlyPolicy(
        encoder=encoder, readout=readout, lif_network=lif,
        input_neuron_ids=input_ids, duration_ms=duration_ms,
    )
