from typing import Sequence
import numpy as np

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

_LEFT_REAR  = slice(0, 4)   
_FWD_LEFT   = slice(4, 8)    
_FWD_RIGHT  = slice(8, 12)   
_RIGHT_REAR = slice(12, 16)  
_IR         = 16             
_STUCK      = 17           

_IR_ATTACH_THRESH  = 3
_SWEEP_FLIP_STEPS  = 25
_MAX_EPISODE_STEPS = 2100

_state       : str = "SEARCH"
_ir_count    : int = 0
_stuck_count : int = 0
_last_turn   : str = "R22"
_sweep_dir   : str = "R22"
_sweep_ticks : int = 0
_ep_steps    : int = 0


def _reset_state() -> None:
    global _state, _ir_count, _stuck_count, _last_turn
    global _sweep_dir, _sweep_ticks, _ep_steps
    _state       = "SEARCH"
    _ir_count    = 0
    _stuck_count = 0
    _last_turn   = "R22"
    _sweep_dir   = "R22"
    _sweep_ticks = 0
    _ep_steps    = 0


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _state, _ir_count, _stuck_count, _last_turn
    global _sweep_dir, _sweep_ticks, _ep_steps

    _ep_steps += 1
    if _ep_steps >= _MAX_EPISODE_STEPS:
        _reset_state()

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

    if _state == "SEARCH" and ir == 1:
        _state = "APPROACH"

    if _state == "APPROACH" and _ir_count >= _IR_ATTACH_THRESH:
        _state = "PUSH"
        _stuck_count = 0

    if _state == "PUSH":
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

    if _state == "APPROACH":
        if ir == 1:
            return "FW"
        _state = "SEARCH"

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