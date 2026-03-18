"""Submission 3 — Tabular Q-learning agent.

This agent loads a Q-table (qtable.npy) trained offline by train_qlearning.py
and uses it to pick the action with the highest Q-value for the current
observation.

Prerequisites
-------------
1. Run the training script to produce qtable.npy:
       python train_qlearning.py --episodes 8000 \\
           --out_qtable submissions/submission3_qlearning/qtable.npy
2. Zip for Codabench:
       python package_submission.py submissions/submission3_qlearning

Observation → state index
-------------------------
The 18-bit binary observation vector is treated as a single integer index
(obs[0] = MSB, obs[17] = LSB), giving a state space of 2^18 = 262 144
distinct states.

Algorithm: Q-learning (off-policy TD control)
    Q(s, a) ← Q(s, a) + α [ r + γ·max_a Q(s', a) − Q(s, a) ]
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

_Q: np.ndarray | None = None


def _obs_to_idx(obs: np.ndarray) -> int:
    """Pack 18-bit binary vector into an integer index (obs[0] = bit 17)."""
    bits = obs.astype(np.uint8)
    return int(bits.dot(np.int32(1) << np.arange(17, -1, -1, dtype=np.int32)))


def _load_once() -> None:
    global _Q
    if _Q is not None:
        return
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtable.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Q-table not found at {path}. "
            "Train first: python train_qlearning.py --out_qtable <path>"
        )
    _Q = np.load(path)


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Return the greedy action from the loaded Q-table."""
    _load_once()
    state_idx = _obs_to_idx(obs)
    return ACTIONS[int(np.argmax(_Q[state_idx]))]
