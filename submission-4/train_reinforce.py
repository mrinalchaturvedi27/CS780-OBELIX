"""REINFORCE (Monte Carlo Policy Gradient) Trainer for OBELIX — Submission 4.

Why REINFORCE instead of DQN?
  • DQN (submission-1) is value-based: it learns Q(s,a) then acts greedily.
  • REINFORCE is policy-based: it directly optimises the parametrised policy
    π_θ(a|s), updating θ in the direction that increases expected return.
  • This makes it easier to learn stochastic policies and is a natural fit
    for continuous/partially observable environments.

Key algorithmic improvements over the naive REINFORCE baseline:
  1. Baseline subtraction  – subtract a running mean return from the MC
     return G_t to reduce variance without changing the gradient in
     expectation (advantage estimate: G_t − b).
  2. Entropy bonus        – adds -β·H(π) to the loss to encourage
     exploration and prevent premature convergence to a deterministic policy.
  3. Reward normalisation – scales raw rewards into a stable training range
     (divide by 200, clip to [-1, 10]).
  4. Gradient clipping    – clips gradient norms to 5.0 for stability.
  5. Best-checkpoint save – persists the weights that achieved the highest
     50-episode rolling mean, not just the final epoch.

Algorithm sketch
  for each episode:
      collect trajectory (s_0, a_0, r_0, …, s_T) by sampling π_θ
      compute discounted returns G_t = Σ_{k≥t} γ^{k-t} r_k
      normalise: G_t ← (G_t − mean) / (std + ε)
      loss = -mean_t[ log π_θ(a_t|s_t) · G_t ] - β·H(π_θ(·|s_t))
      θ ← θ - α · ∇_θ loss

Reference: Williams (1992) "Simple Statistical Gradient-Following
Algorithms for Connectionist Reinforcement Learning".

Example run (overnight on Colab):
  python submission-4/train_reinforce.py \\
      --obelix_py ./obelix.py \\
      --out submission-4/weights.pth \\
      --episodes 5000 --max_steps 2000 \\
      --difficulty 0 --scaling_factor 5 --arena_size 500
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from typing import Deque, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS: List[str] = ["L45", "L22", "FW", "R22", "R45"]
_N_ACTIONS = len(ACTIONS)
_OBS_DIM = 18


# ── Policy network ─────────────────────────────────────────────────────────────

class PolicyNet(nn.Module):
    """Small MLP policy: obs → action probabilities (softmax).

    Kept deliberately small (18→64→64→5) so that it trains on CPU / Colab
    within a reasonable time budget.
    """

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
        """Return log-probabilities for numerical stability."""
        return torch.log_softmax(self.net(x), dim=-1)

    def action_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


# ── Reward normalisation ───────────────────────────────────────────────────────

def normalise_reward(r: float) -> float:
    """Scale raw reward to a stable training range.

    Raw range: stuck=-200, open=-18, sensors+1 to +5, attach=+100,
    success=+2000.  After scale+clip: (-1.0, 10.0).
    """
    return float(np.clip(r / 200.0, -1.0, 10.0))


# ── Discounted return computation ──────────────────────────────────────────────

def discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """Compute G_t = Σ_{k≥t} γ^{k-t} r_k for each time step t."""
    g = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        g = r + gamma * g
        returns.append(g)
    returns.reverse()          # O(n) reversal instead of O(n²) repeated insert
    return torch.tensor(returns, dtype=torch.float32)


# ── Env loader ─────────────────────────────────────────────────────────────────

def import_obelix(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OBELIX


# ── Training loop ──────────────────────────────────────────────────────────────

def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    OBELIX = import_obelix(args.obelix_py)

    # Create the environment once (re-used across episodes via reset())
    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed,
    )

    policy = PolicyNet()
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    recent_returns: Deque[float] = deque(maxlen=50)
    baseline: float = 0.0                 # running mean return for variance reduction
    best_mean: float = float("-inf")

    for ep in range(args.episodes):
        # ── Collect one episode ────────────────────────────────────────────────
        obs = env.reset(seed=args.seed + ep)
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        rewards:   List[float]        = []

        for _ in range(args.max_steps):
            x    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            lp   = policy(x).squeeze(0)                       # log-probs (5,)
            prob = lp.exp()                                    # probs     (5,)

            # Sample action from the policy distribution
            a = int(torch.multinomial(prob, num_samples=1).item())

            obs2, raw_r, done = env.step(ACTIONS[a], render=False)

            log_probs.append(lp[a])
            entropies.append(-torch.sum(prob * lp))            # H(π) = -Σ p log p
            rewards.append(normalise_reward(float(raw_r)))

            obs = obs2
            if done:
                break

        ep_ret = float(sum(rewards))
        recent_returns.append(ep_ret)

        # ── Compute advantage-weighted policy gradient loss ────────────────────
        G = discounted_returns(rewards, args.gamma)

        # Normalise returns (reduces variance within the episode)
        if G.std() > 1e-8:
            G = (G - G.mean()) / (G.std() + 1e-8)

        # Subtract running baseline from each return (further variance reduction)
        baseline = 0.9 * baseline + 0.1 * ep_ret

        lp_tensor  = torch.stack(log_probs)                        # (T,)
        ent_tensor = torch.stack(entropies)                        # (T,)
        pg_loss    = -(lp_tensor * (G - baseline)).mean()
        ent_loss   = -args.entropy_coef * ent_tensor.mean()
        loss       = pg_loss + ent_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

        # ── Logging & checkpoint saving ────────────────────────────────────────
        if (ep + 1) % 25 == 0:
            mean_ret = float(np.mean(recent_returns))
            print(
                f"ep={ep+1:5d} | ret={ep_ret:8.1f} | "
                f"mean50={mean_ret:8.1f} | baseline={baseline:8.1f} | "
                f"loss={loss.item():7.4f}"
            )
            if mean_ret > best_mean:
                best_mean = mean_ret
                torch.save(policy.state_dict(), args.out)
                print(f"  → New best mean50={best_mean:.1f}. Saved to {args.out}")

    # Always persist at least the final weights so the file exists even for
    # very short runs (e.g. smoke tests with <25 episodes).
    if best_mean == float("-inf"):
        torch.save(policy.state_dict(), args.out)
        print(f"  → Saved final weights to {args.out}")

    print(f"\nTraining done. Best mean50={best_mean:.1f}. Weights: {args.out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="REINFORCE Policy Gradient Trainer for OBELIX (Submission 4)"
    )
    # Environment
    ap.add_argument("--obelix_py",      type=str,   required=True,
                    help="Path to obelix.py")
    ap.add_argument("--out",            type=str,   default="weights.pth",
                    help="Output path for best weights")
    ap.add_argument("--episodes",       type=int,   default=5000)
    ap.add_argument("--max_steps",      type=int,   default=2000)
    ap.add_argument("--difficulty",     type=int,   default=0,
                    help="0=static box, 2=blinking, 3=moving+blinking")
    ap.add_argument("--wall_obstacles", action="store_true")
    ap.add_argument("--box_speed",      type=int,   default=2)
    ap.add_argument("--scaling_factor", type=int,   default=5)
    ap.add_argument("--arena_size",     type=int,   default=500)
    # REINFORCE hyperparameters
    ap.add_argument("--gamma",          type=float, default=0.99,
                    help="Discount factor")
    ap.add_argument("--lr",             type=float, default=3e-4,
                    help="Adam learning rate")
    ap.add_argument("--entropy_coef",   type=float, default=0.01,
                    help="Entropy bonus coefficient (encourages exploration)")
    ap.add_argument("--seed",           type=int,   default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
