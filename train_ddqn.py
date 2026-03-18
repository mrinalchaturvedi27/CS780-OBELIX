"""DDQN (Double Deep Q-Network) training script for the OBELIX environment.

Usage
-----
    python train_ddqn.py                           # defaults
    python train_ddqn.py --episodes 5000 \\
        --out_weights submissions/submission4_ddqn/weights.pth

After training, copy or move weights.pth into the submission folder and
package it with package_submission.py.

Algorithm — Double DQN
-----------------------
Vanilla DQN overestimates Q-values because it uses the *same* network for
both selecting and evaluating the next action. DDQN separates these roles:

    Online network  →  selects the best next action (argmax Q_online(s', ·))
    Target network  →  evaluates that action's value  Q_target(s', a*)

TD target:
    y = r + γ · Q_target(s', argmax_a Q_online(s', a))

The target network is periodically hard-updated from the online network every
``target_update`` gradient steps.

Reference: Hasselt et al., "Deep Reinforcement Learning with Double
Q-learning" (https://arxiv.org/pdf/1509.06461)
"""

import argparse
import collections
import os
import random
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from obelix import OBELIX

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBS_DIM = 18
ACTIONS = ("L45", "L22", "FW", "R22", "R45")
N_ACTIONS = len(ACTIONS)

# ---------------------------------------------------------------------------
# Hyper-parameters (tune for better performance)
# ---------------------------------------------------------------------------
GAMMA = 0.99          # discount factor
LR = 1e-3             # Adam learning rate
BATCH_SIZE = 64       # replay mini-batch size
BUFFER_SIZE = 50_000  # replay buffer capacity
MIN_BUFFER = 1_000    # start learning only after this many transitions
TARGET_UPDATE = 500   # hard-copy online → target every N gradient steps

EPS_START = 1.0       # initial ε
EPS_END = 0.05        # minimum ε
EPS_DECAY = 0.995     # multiplicative decay per episode

MAX_STEPS = 500       # episode horizon for training (shorter = faster iteration)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Fully-connected Q-network: 18 → 64 → ReLU → 64 → ReLU → 5."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Fixed-size circular replay buffer storing (s, a, r, s', done) tuples."""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    n_episodes: int,
    difficulty: int,
    wall_obstacles: bool,
    box_speed: int,
    out_weights: str,
    max_steps: int,
) -> None:
    device = torch.device("cpu")  # CPU-only for Codabench compatibility

    online_net = QNetwork().to(device)
    target_net = QNetwork().to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    buffer = ReplayBuffer(BUFFER_SIZE)

    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=max_steps,
        wall_obstacles=wall_obstacles,
        difficulty=difficulty,
        box_speed=box_speed,
        seed=SEED,
    )

    epsilon = EPS_START
    global_step = 0
    episode_rewards: List[float] = []

    for episode in range(1, n_episodes + 1):
        # No fixed seed per episode so the environment's own RNG drives diversity.
        obs = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # ε-greedy action selection
            if random.random() < epsilon:
                action_idx = random.randrange(N_ACTIONS)
            else:
                state_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                with torch.no_grad():
                    action_idx = int(online_net(state_t).argmax(dim=1).item())

            action_str = ACTIONS[action_idx]
            next_obs, reward, done = env.step(action_str, render=False)

            buffer.push(obs, action_idx, float(reward), next_obs, done)
            obs = next_obs
            total_reward += float(reward)
            global_step += 1

            # --- Learn ---
            if len(buffer) >= MIN_BUFFER:
                batch = buffer.sample(BATCH_SIZE)
                obs_b, act_b, rew_b, next_obs_b, done_b = zip(*batch)

                obs_t = torch.tensor(np.array(obs_b, dtype=np.float32), device=device)
                act_t = torch.tensor(act_b, dtype=torch.long, device=device)
                rew_t = torch.tensor(rew_b, dtype=torch.float32, device=device)
                nobs_t = torch.tensor(np.array(next_obs_b, dtype=np.float32), device=device)
                done_t = torch.tensor(done_b, dtype=torch.float32, device=device)

                # Q-values for actions taken
                q_values = online_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)

                # DDQN: online net selects next action, target net evaluates it
                with torch.no_grad():
                    best_actions = online_net(nobs_t).argmax(dim=1, keepdim=True)
                    q_next = target_net(nobs_t).gather(1, best_actions).squeeze(1)
                    targets = rew_t + GAMMA * q_next * (1.0 - done_t)

                loss = loss_fn(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Hard update target network periodically
            if global_step % TARGET_UPDATE == 0:
                target_net.load_state_dict(online_net.state_dict())

        # Decay ε after each episode
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        episode_rewards.append(total_reward)

        if episode % 100 == 0:
            recent = float(np.mean(episode_rewards[-100:]))
            print(
                f"Episode {episode:5d}/{n_episodes}  "
                f"ε={epsilon:.3f}  "
                f"mean_reward(last100)={recent:.1f}"
            )

    # Save online network weights
    os.makedirs(os.path.dirname(os.path.abspath(out_weights)), exist_ok=True)
    torch.save(online_net.state_dict(), out_weights)
    print(f"\nTraining complete. Weights saved to: {out_weights}")
    print(f"Mean reward (last 100 episodes): {float(np.mean(episode_rewards[-100:])):.1f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DDQN agent on the OBELIX environment."
    )
    parser.add_argument(
        "--episodes", type=int, default=3000, help="number of training episodes"
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
        "--out_weights",
        type=str,
        default=os.path.join("submissions", "submission4_ddqn", "weights.pth"),
        help="path to save trained model weights (.pth)",
    )
    args = parser.parse_args()

    print(
        f"Training DDQN  episodes={args.episodes}  "
        f"difficulty={args.difficulty}  "
        f"wall_obstacles={args.wall_obstacles}"
    )
    train(
        n_episodes=args.episodes,
        difficulty=args.difficulty,
        wall_obstacles=args.wall_obstacles,
        box_speed=args.box_speed,
        out_weights=args.out_weights,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
