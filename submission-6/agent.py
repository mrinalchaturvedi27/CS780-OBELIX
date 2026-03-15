"""Evaluation agent for OBELIX — Submission 6.

Uses the DuelingDQN network with LayerNorm (trained by train_ddqn_v4.py).
Weights must be placed next to this file as weights.pth.

Architecture:
  • Input:  4-frame stack of 18-dim observations = 72-dim vector
  • Shared: Linear(72→256) → LayerNorm(256) → ReLU
            Linear(256→256) → LayerNorm(256) → ReLU
  • Value:  Linear(256→1)
  • Adv:    Linear(256→5)
  • Output: Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)

Changes vs submission-5:
  1. Larger network (256 units, was 128) matching train_ddqn_v4.py.
  2. LayerNorm after each hidden layer for inference stability.
  3. Episode-boundary detection: resets frame buffer when the agent
     has been inactive long enough to indicate a new evaluation episode,
     preventing last-episode context leaking into the next.
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

_FRAME_STACK = 4
_OBS_DIM     = 18
_IN_DIM      = _FRAME_STACK * _OBS_DIM   # 72
_HIDDEN      = 256


# ── Network ────────────────────────────────────────────────────────────────────

class DuelingDQN(nn.Module):

    def __init__(self, in_dim: int = _IN_DIM, n_actions: int = 5, hidden: int = _HIDDEN):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.value_stream = nn.Linear(hidden, 1)
        self.adv_stream   = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        v = self.value_stream(h)
        a = self.adv_stream(h)
        return v + a - a.mean(dim=1, keepdim=True)


# ── Module-level state ─────────────────────────────────────────────────────────

_model:       Optional[DuelingDQN] = None
_frame_buf:   deque                = deque(maxlen=_FRAME_STACK)
_last_action: Optional[int]        = None
_repeat_cnt:  int                  = 0
_ep_steps:    int                  = 0

# Hyper-params for inference-time action smoothing.
_MAX_REPEAT    = 3
_CLOSE_Q_DELTA = 0.1

# Episode-boundary heuristic: reset frame buffer and state after inferred
# episode end so the next episode starts with clean context.
# Uses max_steps (default 1000 to match evaluate.py's default).
_MAX_EPISODE_STEPS = 1000


def _load_model() -> None:
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"weights.pth not found at {wpath}")
    m  = DuelingDQN()
    sd = torch.load(wpath, map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    _model = m


def _init_frame_buf(obs: np.ndarray) -> None:
    for _ in range(_FRAME_STACK):
        _frame_buf.append(obs.copy())


def _get_stacked() -> np.ndarray:
    return np.concatenate(list(_frame_buf)).astype(np.float32)


def _reset_episode_state() -> None:
    """Clear per-episode state so a fresh episode starts cleanly."""
    global _last_action, _repeat_cnt, _ep_steps
    _frame_buf.clear()
    _last_action = None
    _repeat_cnt  = 0
    _ep_steps    = 0


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_cnt, _ep_steps

    _load_model()

    # Frame-stack management: append obs before using it.
    if not _frame_buf:
        _init_frame_buf(obs)
    else:
        _frame_buf.append(obs.copy())

    stacked = _get_stacked()

    # Greedy action from Q-network.
    x = torch.tensor(stacked).unsqueeze(0)
    q = _model(x).squeeze(0).numpy()
    best = int(np.argmax(q))

    # Action smoothing: avoid rapid oscillation when top-2 Q-values are close.
    if _last_action is not None:
        order    = np.argsort(-q)
        q_best   = float(q[order[0]])
        q_second = float(q[order[1]])
        if (q_best - q_second) < _CLOSE_Q_DELTA:
            if _repeat_cnt < _MAX_REPEAT:
                best = _last_action
                _repeat_cnt += 1
            else:
                _repeat_cnt = 0
        else:
            _repeat_cnt = 0

    _last_action = best

    # Increment step counter and reset at episode boundary.
    _ep_steps += 1
    if _ep_steps >= _MAX_EPISODE_STEPS:
        _reset_episode_state()

    return ACTIONS[best]
