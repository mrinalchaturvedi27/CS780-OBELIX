"""
Submission 5: Dueling DDQN with Frame Stacking (k=4) for OBELIX.

Loads weights trained by train_ddqn_v2.py (DuelingDQN, 72-dim input).
Greedy policy (ε=0) at eval time.
"""

from __future__ import annotations
import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# Constants must match what train_ddqn_v2.py used
_FRAME_STACK = 4
_OBS_DIM     = 18
_IN_DIM      = _FRAME_STACK * _OBS_DIM   # 72


class DuelingDQN(nn.Module):
    """Dueling DDQN: shared encoder → Value stream + Advantage stream."""

    def __init__(self, in_dim: int = _IN_DIM, n_actions: int = 5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_stream = nn.Linear(128, 1)
        self.adv_stream   = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        v = self.value_stream(h)
        a = self.adv_stream(h)
        return v + a - a.mean(dim=1, keepdim=True)


# ── Module-level state ─────────────────────────────────────────────────────────
_model       : Optional[DuelingDQN] = None
_frame_buf   : deque                = deque(maxlen=_FRAME_STACK)
_last_action : Optional[int]        = None
_repeat_cnt  : int                  = 0
_ep_steps    : int                  = 0

_MAX_EPISODE_STEPS = 2100
_CLOSE_Q_DELTA     = 0.1   # threshold for action smoothing
_MAX_REPEAT        = 3     # max repeated actions before forcing switch


def _load_model() -> None:
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"weights.pth not found at {wpath}")
    m = DuelingDQN()
    state_dict = torch.load(wpath, map_location="cpu")
    m.load_state_dict(state_dict)
    m.eval()
    _model = m


def _init_frame_buf(obs: np.ndarray) -> None:
    for _ in range(_FRAME_STACK):
        _frame_buf.append(obs.copy())


def _get_stacked() -> np.ndarray:
    return np.concatenate(list(_frame_buf)).astype(np.float32)


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _last_action, _repeat_cnt, _ep_steps

    _load_model()

    # ── Episode reset detection ────────────────────────────────────────────────
    _ep_steps += 1
    if _ep_steps == 1 or _ep_steps >= _MAX_EPISODE_STEPS:
        _init_frame_buf(obs)
        _last_action = None
        _repeat_cnt  = 0
        if _ep_steps >= _MAX_EPISODE_STEPS:
            _ep_steps = 1

    # ── Update frame stack ────────────────────────────────────────────────────
    _frame_buf.append(obs.copy())
    stacked = _get_stacked()

    # ── Greedy Q-inference ────────────────────────────────────────────────────
    x = torch.tensor(stacked).unsqueeze(0)   # (1, 72)
    q = _model(x).squeeze(0).numpy()         # (5,)
    best = int(np.argmax(q))

    # ── Action smoothing: prevent flip-flopping when top-2 Q-values are close ─
    if _last_action is not None:
        order       = np.argsort(-q)
        q_best      = float(q[order[0]])
        q_second    = float(q[order[1]])
        if (q_best - q_second) < _CLOSE_Q_DELTA:
            if _repeat_cnt < _MAX_REPEAT:
                best = _last_action
                _repeat_cnt += 1
            else:
                _repeat_cnt = 0
        else:
            _repeat_cnt = 0

    _last_action = best
    return ACTIONS[best]
