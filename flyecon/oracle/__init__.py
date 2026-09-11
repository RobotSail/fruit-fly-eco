"""FLY//ECON oracle module — MDP value-iteration solver.

Provides the Baseline Oracle: an exact value-iteration solver over the
MR12 economy MDP. The Oracle is the incorruptible yardstick — its
transition model IS the MR12 rules, not a separate reimplementation.
"""

from flyecon.oracle.mdp import EconomyMDP
from flyecon.oracle.solver import Policy, VerificationResult, solve, verify_against_random

__all__ = ["EconomyMDP", "Policy", "VerificationResult", "solve", "verify_against_random"]
