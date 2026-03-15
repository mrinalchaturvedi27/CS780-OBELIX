"""
Submission 6: PPO (Proximal Policy Optimization) with Frame Stacking for OBELIX.

Critique of Submission 5 (Dueling DDQN + Frame Stacking):
  • Off-policy DDQN: stale replay data introduces lag between behaviour and learning.
  • Q-value overestimation persists when rewards are highly sparse (success events
    are rare vs dense -18 open-space penalties).
  • Episode reset in agent.py relied on a hardcoded step counter (_MAX_EPISODE_STEPS)
    rather than the environment's done signal — fragile across different max_steps.
  • torch.load() without weights_only=True raises DeprecationWarning in PyTorch ≥2.0
    and is a latent security risk.
  • Training always used the same seed sequence → network memorised layout → poor
    generalisation across new arena seeds/difficulties.

Improvements in Submission 6:
  1. On-policy PPO: every gradient step uses data from the *current* policy —
     no stale-replay issue, more faithful credit assignment.
  2. Clipped surrogate objective: prevents catastrophically large policy updates
     without needing a hard KL constraint (cf. TRPO).
  3. Shared Actor-Critic encoder: V(s) guides advantage estimation while π(a|s)
     is learned jointly → better sample efficiency.
  4. Entropy bonus: built-in exploration pressure prevents premature convergence
     in sparse-reward settings.
  5. GAE (λ=0.95): lower-variance advantage estimates → more stable gradient.
  6. Deterministic evaluation (argmax of policy logits): eliminates stochastic
     variance at test time.
  7. weights_only=True in torch.load: fixes the security/deprecation issue.
  8. Frame stacking k=4 retained for temporal context (consistent with sub-5).

Trained with: train_ppo.py
Input:  frame-stacked observation (k=4 × obs_dim=18 → 72-dim float32 vector)
Output: one of {"L45", "L22", "FW", "R22", "R45"}

Training command (overnight on Colab / local machine):
  python submission-6/train_ppo.py \\
      --obelix_py ./obelix.py \\
      --out submission-6/weights.pth \\
      --epochs 800 --rollout_len 2048 --max_steps 2000 \\
      --difficulty 0 --scaling_factor 5 --arena_size 500 \\
      --ent_coef 0.01 --seed 42
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from collections import deque

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# Must match train_ppo.py
_FRAME_STACK = 4
_OBS_DIM     = 18
_IN_DIM      = _FRAME_STACK * _OBS_DIM   # 72


# ── Actor-Critic (identical architecture to train_ppo.py) ─────────────────────

class ActorCritic(nn.Module):
    """Shared encoder → policy logits (actor) + scalar state-value (critic)."""

    def __init__(self, in_dim: int = _IN_DIM, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head  = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


# ── Module-level state ─────────────────────────────────────────────────────────

_model     : Optional[ActorCritic] = None
_frame_buf : deque                 = deque(maxlen=_FRAME_STACK)
_ep_steps  : int                   = 0

# Fallback step-count guard.  Must be >= the max_steps used during training
# (train_ppo.py default: 2000) plus a small safety margin.
_MAX_EPISODE_STEPS = 2100


def _load_model() -> None:
    global _model
    if _model is not None:
        return
    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        raise FileNotFoundError(
            f"weights.pth not found at {wpath}. "
            "Train with train_ppo.py and place the output next to agent.py."
        )
    m = ActorCritic()
    # weights_only=True: security fix over submission-5's bare torch.load call
    state_dict = torch.load(wpath, map_location="cpu", weights_only=True)
    m.load_state_dict(state_dict)
    m.eval()
    _model = m


def _init_frame_buf(obs: np.ndarray) -> None:
    """Fill the frame buffer by repeating the first observation k times."""
    for _ in range(_FRAME_STACK):
        _frame_buf.append(obs.copy())


def _get_stacked() -> np.ndarray:
    return np.concatenate(list(_frame_buf)).astype(np.float32)


@torch.no_grad()
def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Return a greedy action for the current (stacked) observation.

    Uses argmax of policy logits — deterministic at evaluation time.
    """
    global _ep_steps

    _load_model()

    # Episode reset detection (step-count fallback; real done signal not exposed here)
    _ep_steps += 1
    is_new_episode = (_ep_steps == 1) or (_ep_steps >= _MAX_EPISODE_STEPS)
    if is_new_episode:
        _init_frame_buf(obs)          # fills buffer with obs × k; no extra append needed
        if _ep_steps >= _MAX_EPISODE_STEPS:
            _ep_steps = 1
    else:
        # Update frame stack with the latest observation
        _frame_buf.append(obs.copy())
    stacked = _get_stacked()

    # Greedy action: argmax of policy logits (no sampling, no action smoothing)
    x      = torch.tensor(stacked).unsqueeze(0)   # (1, 72)
    logits, _ = _model(x)
    best   = int(logits.squeeze(0).argmax().item())
    return ACTIONS[best]
