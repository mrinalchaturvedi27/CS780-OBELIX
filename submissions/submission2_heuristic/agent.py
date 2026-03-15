"""Submission 2 — Sensor-driven heuristic agent.

This agent uses hand-crafted rules derived from the sensor layout to navigate
toward the box and push it to the boundary. No training or weight files are
needed.

Observation layout (18 binary bits):
    obs[0]  sonar-0 near  (left extreme, near range)
    obs[1]  sonar-0 far   (left extreme, far range)
    obs[2]  sonar-1 near  (left, near range)
    obs[3]  sonar-1 far   (left, far range)
    obs[4]  sonar-2 near  (forward-left, near range)
    obs[5]  sonar-2 far   (forward-left, far range)
    obs[6]  sonar-3 near  (forward slight-left, near range)
    obs[7]  sonar-3 far   (forward slight-left, far range)
    obs[8]  sonar-4 near  (forward slight-right, near range)
    obs[9]  sonar-4 far   (forward slight-right, far range)
    obs[10] sonar-5 near  (forward-right, near range)
    obs[11] sonar-5 far   (forward-right, far range)
    obs[12] sonar-6 near  (right, near range)
    obs[13] sonar-6 far   (right, far range)
    obs[14] sonar-7 near  (right extreme, near range)
    obs[15] sonar-7 far   (right extreme, far range)
    obs[16] IR sensor     (box directly ahead, very close)
    obs[17] stuck flag    (1 = robot cannot move forward)

Decision rules (priority-ordered):
    1. Stuck → rotate to get unstuck
    2. IR fires → box is directly in front; move forward (attach / push)
    3. Forward near sensors fire → move forward
    4. Forward far sensors fire → move forward
    5. Box only on right → turn left toward it
    6. Box only on left  → turn right toward it
    7. Box on both sides (symmetric) → forward or slight random turn
    8. No sensor signal  → systematic exploration (spin then surge)
"""

from typing import Sequence
import numpy as np

ACTIONS: Sequence[str] = ("L45", "L22", "FW", "R22", "R45")

# Internal exploration counter (module-level state, reset each episode call).
_explore_counter: int = 0
_explore_turn_dir: str = "R45"


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    global _explore_counter, _explore_turn_dir

    stuck = bool(obs[17])

    # Left sonar group
    left_near  = bool(obs[0] or obs[2])
    left_far   = bool(obs[1] or obs[3])
    left_any   = left_near or left_far

    # Forward sonar group
    fwd_near = bool(obs[4] or obs[6] or obs[8] or obs[10])
    fwd_far  = bool(obs[5] or obs[7] or obs[9] or obs[11])
    fwd_any  = fwd_near or fwd_far

    # Left vs right asymmetry within forward sonars
    fwd_left_any  = bool(obs[4] or obs[5] or obs[6] or obs[7])
    fwd_right_any = bool(obs[8] or obs[9] or obs[10] or obs[11])

    # Right sonar group
    right_near = bool(obs[12] or obs[14])
    right_far  = bool(obs[13] or obs[15])
    right_any  = right_near or right_far

    ir = bool(obs[16])

    # --- Rule 1: stuck — rotate to free the robot ---
    if stuck:
        _explore_counter = 0
        return _explore_turn_dir  # alternate handled below

    # --- Rule 2: IR fires — box is directly in front, very close ---
    if ir:
        _explore_counter = 0
        return "FW"

    # --- Rule 3 & 4: box is ahead — approach it ---
    if fwd_near:
        _explore_counter = 0
        # Fine-tune heading toward the stronger forward side
        if fwd_left_any and not fwd_right_any:
            return "R22"
        if fwd_right_any and not fwd_left_any:
            return "L22"
        return "FW"

    if fwd_far:
        _explore_counter = 0
        if fwd_left_any and not fwd_right_any:
            return "R22"
        if fwd_right_any and not fwd_left_any:
            return "L22"
        return "FW"

    # --- Rule 5: box only on right — turn left ---
    if right_any and not left_any:
        _explore_counter = 0
        return "L22" if right_near else "L45"

    # --- Rule 6: box only on left — turn right ---
    if left_any and not right_any:
        _explore_counter = 0
        return "R22" if left_near else "R45"

    # --- Rule 7: box on both sides — go forward (box may be ahead, sensors
    #     overlapping) or use a small random turn to break symmetry ---
    if left_any and right_any:
        _explore_counter = 0
        return "FW"

    # --- Rule 8: no sensor signal — systematic exploration ---
    # Alternate between a burst of rotation and a surge forward.
    _explore_counter += 1
    # Every 12 steps flip the exploration turn direction.
    if _explore_counter % 24 == 0:
        _explore_turn_dir = "L45" if _explore_turn_dir == "R45" else "R45"

    if _explore_counter % 6 < 4:
        return _explore_turn_dir
    return "FW"
