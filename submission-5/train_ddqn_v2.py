"""Improved DDQN Trainer for OBELIX — v2 (for Submission 5+).

Key improvements over v1:
  1. Frame stacking (k=4): 72-dim input, gives temporal context for POMDP
  2. Dueling DDQN architecture: separates V(s) and A(s,a) for better learning
  3. Reward normalisation: clips reward to [-1, 10] to stabilise Q-values
  4. Single env instance (no per-episode recreation)
  5. Longer epsilon decay over 500K steps
  6. Best checkpoint saving (by mean episode return)
  7. Logging every 25 episodes

Example run (start tonight, train overnight):
  cd /Users/mrinalchaturvedi/Documents/sem6/CS780/CS780-OBELIX
  python ../submission-5/train_ddqn_v2.py \\
      --obelix_py ./obelix.py \\
      --out ../submission-5/weights.pth \\
      --episodes 6000 --max_steps 2000 \\
      --difficulty 0 --scaling_factor 5 --arena_size 500
"""

from __future__ import annotations
import argparse
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# ── Frame stacking ─────────────────────────────────────────────────────────────

class FrameStack:
    """Stacks last k observations into a single 1-D vector."""

    def __init__(self, k: int = 4, obs_dim: int = 18):
        self.k = k
        self.obs_dim = obs_dim
        self._buf: deque = deque(maxlen=k)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        for _ in range(self.k):
            self._buf.append(obs.copy())
        return self._get()

    def step(self, obs: np.ndarray) -> np.ndarray:
        self._buf.append(obs.copy())
        return self._get()

    def _get(self) -> np.ndarray:
        return np.concatenate(list(self._buf)).astype(np.float32)

    @property
    def dim(self) -> int:
        return self.k * self.obs_dim


# ── Dueling DDQN network ───────────────────────────────────────────────────────

class DuelingDQN(nn.Module):
    """Dueling network: shared encoder → separate Value + Advantage streams."""

    def __init__(self, in_dim: int, n_actions: int = 5):
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
        h   = self.shared(x)
        v   = self.value_stream(h)              # (B, 1)
        a   = self.adv_stream(h)               # (B, n_actions)
        # Q = V + (A - mean(A))
        return v + a - a.mean(dim=1, keepdim=True)


# ── Replay buffer ──────────────────────────────────────────────────────────────

@dataclass
class Transition:
    s:    np.ndarray
    a:    int
    r:    float
    s2:   np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self._buf: Deque[Transition] = deque(maxlen=capacity)

    def add(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch: int):
        idx   = np.random.choice(len(self._buf), size=batch, replace=False)
        items = [self._buf[i] for i in idx]
        s  = torch.tensor(np.stack([it.s  for it in items]))
        a  = torch.tensor(np.array([it.a  for it in items], dtype=np.int64))
        r  = torch.tensor(np.array([it.r  for it in items], dtype=np.float32))
        s2 = torch.tensor(np.stack([it.s2 for it in items]))
        d  = torch.tensor(np.array([it.done for it in items], dtype=np.float32))
        return s, a, r, s2, d

    def __len__(self) -> int:
        return len(self._buf)


# ── Reward normalisation ───────────────────────────────────────────────────────

def normalise_reward(r: float) -> float:
    """Map raw per-step reward to a stable training range.

    Raw range: stuck=-200, open=-18, sensors+1 to +5, attach=+100, success=+2000.
    After clip: (-1.0, 10.0). Keeps relative signal while preventing Q explosion.
    """
    r = r / 200.0          # scale: stuck → -1, success → +10
    return float(np.clip(r, -1.0, 10.0))


# ── Env loader (dynamic import of obelix.py) ───────────────────────────────────

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
    fs     = FrameStack(k=args.frame_stack, obs_dim=18)

    # Create env ONCE (fixed in v2 — v1 recreated each episode)
    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=args.seed,
    )

    in_dim = fs.dim  # 72 for k=4, obs_dim=18

    q   = DuelingDQN(in_dim=in_dim)
    tgt = DuelingDQN(in_dim=in_dim)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt    = optim.Adam(q.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay)

    total_steps = 0
    best_mean   = float("-inf")

    def eps(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    recent_returns = deque(maxlen=50)   # track recent episode returns

    for ep in range(args.episodes):
        raw_obs = env.reset()
        s       = fs.reset(raw_obs)
        ep_ret  = 0.0

        for _ in range(args.max_steps):
            # ε-greedy action selection
            if np.random.rand() < eps(total_steps):
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            raw_obs2, raw_r, done = env.step(ACTIONS[a], render=False)
            s2  = fs.step(raw_obs2)
            r   = normalise_reward(float(raw_r))
            ep_ret += float(raw_r)

            replay.add(Transition(s=s, a=a, r=r, s2=s2, done=bool(done)))
            s = s2
            total_steps += 1

            # ── Learning step ──────────────────────────────────────────────────
            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db = replay.sample(args.batch)

                with torch.no_grad():
                    # Double DQN: online net selects action, target net evaluates
                    next_a   = q(s2b).argmax(dim=1, keepdim=True)
                    next_val = tgt(s2b).gather(1, next_a).squeeze(1)
                    y        = rb + args.gamma * (1.0 - db) * next_val

                pred = q(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                if total_steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        recent_returns.append(ep_ret)

        # ── Logging ────────────────────────────────────────────────────────────
        if (ep + 1) % 25 == 0:
            mean_ret = float(np.mean(recent_returns))
            print(
                f"ep={ep+1:5d} | ret={ep_ret:8.1f} | "
                f"mean50={mean_ret:8.1f} | ε={eps(total_steps):.3f} | "
                f"buf={len(replay)} | steps={total_steps}"
            )
            # Save best model (by 50-episode average return)
            if mean_ret > best_mean:
                best_mean = mean_ret
                torch.save(q.state_dict(), args.out)
                print(f"  → New best! Saved to {args.out}")

    print(f"\nTraining done. Best mean50={best_mean:.1f}. Weights: {args.out}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Improved DDQN Trainer for OBELIX v2")
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights.pth")
    ap.add_argument("--episodes",        type=int,   default=6000)
    ap.add_argument("--max_steps",       type=int,   default=2000)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--frame_stack",     type=int,   default=4)
    # DQN hyperparams
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=3e-4)
    ap.add_argument("--batch",           type=int,   default=256)
    ap.add_argument("--replay",          type=int,   default=100_000)
    ap.add_argument("--warmup",          type=int,   default=2_000)
    ap.add_argument("--target_sync",     type=int,   default=1_000)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int,   default=500_000)
    ap.add_argument("--seed",            type=int,   default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
