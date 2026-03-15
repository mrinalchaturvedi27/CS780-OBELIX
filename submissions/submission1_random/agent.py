"""Submission 1 — Biased random walk (baseline).

This is the simplest possible agent: it mostly goes forward with small
probability of turning left or right. Use this as your first Codabench
submission to verify the full pipeline works before improving the policy.

No training or weight files are needed.

Observation layout (18 binary bits):
    obs[0-3]   : left sonar (near/far for two left-facing sensors)
    obs[4-11]  : forward sonar (near/far for four forward-facing sensors)
    obs[12-15] : right sonar (near/far for two right-facing sensors)
    obs[16]    : IR sensor (box directly ahead, very close range)
    obs[17]    : stuck flag (1 = robot cannot move forward)

Action space: "L45", "L22", "FW", "R22", "R45"
"""

from typing import Sequence
import numpy as np

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

# Action probabilities: strongly biased toward forward motion.
_PROBS = np.array([0.05, 0.10, 0.70, 0.10, 0.05], dtype=float)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Return an action sampled from a forward-biased distribution."""
    return ACTIONS[int(rng.choice(len(ACTIONS), p=_PROBS))]
