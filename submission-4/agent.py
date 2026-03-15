"""Submission 4: REINFORCE Policy Gradient agent for OBELIX.

At evaluation the agent loads a small policy network (PolicyNet) trained
offline with train_reinforce.py.  If weights.pth is absent (e.g. before
training has been run) it transparently falls back to the hand-coded FSM
heuristic so the module is always runnable.

Critique of the original FSM heuristic and why REINFORCE is better
--------------------------------------------------------------------
The original submission-4 used a finite-state machine (SEARCH → APPROACH →
PUSH) with entirely hand-crafted thresholds.  Its shortcomings:

  1. No learning – thresholds never improve, behaviour cannot adapt to
     different difficulties, arena sizes, or moving boxes.
  2. Global mutable state – fragile across multi-episode evaluations; a
     single missed reset corrupts future episodes.
  3. Rigid state transitions – the 3-state machine cannot discover
     behaviours that a learned policy might find (e.g. circling around a
     stuck box to find a better push angle).
  4. No generalisation – the hand-coded sweep direction and stuck-recovery
     logic is specific to the default difficulty and arena configuration.

REINFORCE directly optimises the expected return by following the policy
gradient:
    ∇_θ J(θ) = E[ ∇_θ log π_θ(a|s) · (G_t − b) ]
where G_t is the discounted return from time t and b is a variance-reducing
baseline.  The agent therefore discovers the best action distribution for
each observation purely from environment interaction, without hard-coded
rules.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

_OBS_DIM  = 18
_N_ACTIONS = len(ACTIONS)

# ── Policy network (must match train_reinforce.py) ─────────────────────────────

class PolicyNet(nn.Module):
    """Tiny MLP: obs (18,) → log-probabilities over 5 actions."""

    def __init__(self, in_dim: int = _OBS_DIM, n_actions: int = _N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.net(x), dim=-1)

    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


# ── Module-level state ─────────────────────────────────────────────────────────

_model: Optional[PolicyNet] = None
_use_neural: bool = False          # set to True once weights load successfully

# FSM fallback state (used only when weights.pth is absent)
_LEFT_REAR  = slice(0, 4)
_FWD_LEFT   = slice(4, 8)
_FWD_RIGHT  = slice(8, 12)
_RIGHT_REAR = slice(12, 16)
_IR         = 16
_STUCK      = 17

_IR_ATTACH_THRESH  = 3
_SWEEP_FLIP_STEPS  = 25
_MAX_EPISODE_STEPS = 2100

_fsm_state   : str = "SEARCH"
_ir_count    : int = 0
_stuck_count : int = 0
_last_turn   : str = "R22"
_sweep_dir   : str = "R22"
_sweep_ticks : int = 0
_ep_steps    : int = 0


# ── Model loader ───────────────────────────────────────────────────────────────

def _load_model() -> None:
    """Try to load weights.pth once; silently fall back to FSM if absent."""
    global _model, _use_neural
    if _model is not None:
        return
    here  = os.path.dirname(os.path.abspath(__file__))
    wpath = os.path.join(here, "weights.pth")
    if not os.path.exists(wpath):
        return                          # no weights → stay with FSM heuristic
    m = PolicyNet()
    m.load_state_dict(torch.load(wpath, map_location="cpu"))
    m.eval()
    _model     = m
    _use_neural = True


# ── Neural policy (greedy) ─────────────────────────────────────────────────────

@torch.no_grad()
def _neural_policy(obs: np.ndarray) -> str:
    x    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, 18)
    prob = _model.action_probs(x).squeeze(0).numpy()            # (5,)
    return ACTIONS[int(np.argmax(prob))]


# ── FSM heuristic fallback ─────────────────────────────────────────────────────

def _reset_fsm() -> None:
    global _fsm_state, _ir_count, _stuck_count, _last_turn
    global _sweep_dir, _sweep_ticks, _ep_steps
    _fsm_state   = "SEARCH"
    _ir_count    = 0
    _stuck_count = 0
    _last_turn   = "R22"
    _sweep_dir   = "R22"
    _sweep_ticks = 0
    _ep_steps    = 0


def _fsm_policy(obs: np.ndarray) -> str:
    global _fsm_state, _ir_count, _stuck_count, _last_turn
    global _sweep_dir, _sweep_ticks, _ep_steps

    _ep_steps += 1
    if _ep_steps >= _MAX_EPISODE_STEPS:
        _reset_fsm()

    ir    = int(obs[_IR])
    stuck = int(obs[_STUCK])

    left_score  = int(np.sum(obs[_LEFT_REAR]))
    fwd_left    = int(np.sum(obs[_FWD_LEFT]))
    fwd_right   = int(np.sum(obs[_FWD_RIGHT]))
    right_score = int(np.sum(obs[_RIGHT_REAR]))
    fwd_total   = fwd_left + fwd_right

    if ir == 1:
        _ir_count += 1
    else:
        _ir_count = 0

    if _fsm_state == "SEARCH" and ir == 1:
        _fsm_state = "APPROACH"

    if _fsm_state == "APPROACH" and _ir_count >= _IR_ATTACH_THRESH:
        _fsm_state   = "PUSH"
        _stuck_count = 0

    if _fsm_state == "PUSH":
        if stuck:
            _stuck_count += 1
            if _stuck_count % 8 < 4:
                action = "L22" if _last_turn != "L22" else "R22"
            else:
                action = "L45" if _last_turn != "L45" else "R45"
            _last_turn = action
            return action
        else:
            _stuck_count = 0
            return "FW"

    if _fsm_state == "APPROACH":
        if ir == 1:
            return "FW"
        _fsm_state = "SEARCH"

    if fwd_total > 0:
        if fwd_left > fwd_right:
            return "R22"
        elif fwd_right > fwd_left:
            return "L22"
        else:
            return "FW"

    if left_score > right_score:
        return "L45"
    if right_score > left_score:
        return "R45"

    _sweep_ticks += 1
    if _sweep_ticks >= _SWEEP_FLIP_STEPS:
        _sweep_ticks = 0
        _sweep_dir = "R22" if _sweep_dir == "L22" else "L22"

    return _sweep_dir


# ── Public policy function ─────────────────────────────────────────────────────

def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Return an action string for the current observation.

    Uses the trained PolicyNet when weights.pth is available; otherwise
    falls back to the FSM heuristic.
    """
    _load_model()
    if _use_neural:
        return _neural_policy(obs)
    return _fsm_policy(obs)