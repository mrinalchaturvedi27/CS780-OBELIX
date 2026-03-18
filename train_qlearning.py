"""Tabular Q-learning training script for the OBELIX environment.

Usage
-----
    python train_qlearning.py                          # defaults
    python train_qlearning.py --episodes 8000 \\
        --out_qtable submissions/submission3_qlearning/qtable.npy

After training, copy or move qtable.npy into the submission folder and
package it with package_submission.py.

Algorithm
---------
Q-learning (off-policy TD control):
    Q(s, a) ← Q(s, a) + α [ r + γ · max_a Q(s', a) − Q(s, a) ]

State representation
--------------------
The 18-bit binary observation is interpreted as an integer index in
[0, 2^18 − 1], giving 262 144 possible states. The Q-table is therefore
a float32 array of shape (262144, 5) — about 5 MB on disk.

Exploration
-----------
ε-greedy with linear decay from ε_start down to ε_min over the first
ε_decay_episodes episodes, then held constant at ε_min.
"""

import argparse
import os
from typing import List

import numpy as np

from obelix import OBELIX

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Observation → state index
# ---------------------------------------------------------------------------
N_STATES = 2 ** 18   # 262 144
N_ACTIONS = 5
ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_BIT_WEIGHTS = np.int32(1) << np.arange(17, -1, -1, dtype=np.int32)


def obs_to_idx(obs: np.ndarray) -> int:
    """Pack 18-bit binary observation into a scalar index (obs[0] = MSB)."""
    return int(obs.astype(np.uint8).dot(_BIT_WEIGHTS))


# ---------------------------------------------------------------------------
# Hyper-parameters (tune these for better performance)
# ---------------------------------------------------------------------------
ALPHA = 0.1           # learning rate
GAMMA = 0.99          # discount factor
EPS_START = 1.0       # initial exploration probability
EPS_MIN = 0.05        # minimum exploration probability
MAX_STEPS = 500       # episode horizon for training (shorter = faster iteration)


def train(
    n_episodes: int,
    difficulty: int,
    wall_obstacles: bool,
    box_speed: int,
    eps_decay_episodes: int,
    out_qtable: str,
    max_steps: int,
) -> None:
    # Initialize Q-table to zeros.
    Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float32)
    visited_states: set = set()

    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=box_speed,
        seed=SEED,
    )

    episode_rewards: List[float] = []

    for episode in range(1, n_episodes + 1):
        # Linearly decay epsilon over eps_decay_episodes, then fix at EPS_MIN.
        frac = min(1.0, (episode - 1) / max(1, eps_decay_episodes))
        epsilon = EPS_START + frac * (EPS_MIN - EPS_START)

        obs = env.reset()
        state = obs_to_idx(obs)

        total_reward = 0.0
        done = False

        while not done:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action_idx = np.random.randint(N_ACTIONS)
            else:
                action_idx = int(np.argmax(Q[state]))

            action_str = ACTIONS[action_idx]
            next_obs, reward, done = env.step(action_str, render=False)
            next_state = obs_to_idx(next_obs)

            # Q-learning update
            best_next = float(np.max(Q[next_state]))
            td_target = float(reward) + GAMMA * best_next * (1.0 - float(done))
            Q[state, action_idx] += ALPHA * (td_target - Q[state, action_idx])

            visited_states.add(state)
            state = next_state
            total_reward += float(reward)

        episode_rewards.append(total_reward)

        if episode % 500 == 0:
            recent_mean = float(np.mean(episode_rewards[-500:]))
            print(
                f"Episode {episode:5d}/{n_episodes}  "
                f"ε={epsilon:.3f}  "
                f"mean_reward(last500)={recent_mean:.1f}"
            )

    # Save Q-table
    os.makedirs(os.path.dirname(os.path.abspath(out_qtable)), exist_ok=True)
    np.save(out_qtable, Q)
    print(f"\nTraining complete. Q-table saved to: {out_qtable}")
    print(
        f"Unique states visited: {len(visited_states)} / {N_STATES}"
    )
    print(f"Mean reward (last 500 episodes): {float(np.mean(episode_rewards[-500:])):.1f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tabular Q-learning agent on the OBELIX environment."
    )
    parser.add_argument(
        "--episodes", type=int, default=6000, help="number of training episodes"
    )
    parser.add_argument(
        "--eps_decay_episodes",
        type=int,
        default=None,
        help=f"number of episodes over which epsilon decays (default: 70%% of --episodes)",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        default=0,
        help="difficulty level: 0=static box, 2=blinking, 3=moving+blinking",
    )
    parser.add_argument(
        "--wall_obstacles",
        action="store_true",
        help="enable wall obstacle in the arena",
    )
    parser.add_argument(
        "--box_speed",
        type=int,
        default=2,
        help="box speed in pixels/step for difficulty >= 3",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_STEPS,
        help=f"episode horizon during training (default: {MAX_STEPS})",
    )
    parser.add_argument(
        "--out_qtable",
        type=str,
        default=os.path.join("submissions", "submission3_qlearning", "qtable.npy"),
        help="path to save the trained Q-table (.npy file)",
    )
    args = parser.parse_args()

    decay_ep = args.eps_decay_episodes or int(0.70 * args.episodes)

    print(
        f"Training Q-learning  episodes={args.episodes}  "
        f"difficulty={args.difficulty}  "
        f"wall_obstacles={args.wall_obstacles}  "
        f"eps_decay_over={decay_ep} episodes"
    )
    train(
        n_episodes=args.episodes,
        difficulty=args.difficulty,
        wall_obstacles=args.wall_obstacles,
        box_speed=args.box_speed,
        eps_decay_episodes=decay_ep,
        out_qtable=args.out_qtable,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
