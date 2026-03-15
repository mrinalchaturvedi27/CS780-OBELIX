"""DDQN with Prioritized Experience Replay for OBELIX — Submission 6.

Critique of submission-5:
  - Training converged at ep=1050 with mean100=-3301.7 (very poor performance).
  - Epsilon decayed too quickly: eps_decay_steps=150K / 8 envs = only ~18K
    parallel iterations before ε reaches 0.05 — far too little exploration.
  - Fixed reward clipping (r/200 clamped to [-1, 10]) distorts relative
    magnitudes; stuck=-200→-1 same scale as open-space=-18→-0.09.
  - Vanilla uniform replay: rare success events (+2000 raw) get overwritten by
    thousands of mediocre transitions and are never replicated.
  - Network had no normalisation layers, making learning slow.

Key improvements in v4:
  1. Prioritized Experience Replay (PER) — sum-tree for O(log n) sampling.
     High TD-error transitions (successes, attachments) are replayed more often.
  2. Running reward normalisation (EMA mean/std) preserves relative signal.
  3. DuelingDQN with LayerNorm after each hidden layer for training stability.
  4. Slower epsilon decay (500K steps), more training (10 000 episodes).
  5. Warm-start: optionally load weights from a previous run (--warmstart).
  6. PER importance-sampling (IS) weights applied to loss.

Example run:
  python submission-6/train_ddqn_v4.py \\
      --obelix_py ./obelix.py \\
      --out ./submission-6/weights.pth \\
      --episodes 10000 --max_steps 1000 \\
      --difficulty 0 --scaling_factor 5 --arena_size 500 \\
      --warmstart ./submission-5/weights.pth
"""

from __future__ import annotations
import argparse
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# ── Frame stacking ─────────────────────────────────────────────────────────────

class FrameStack:
    """Stacks last k observations into a single 1-D vector."""

    def __init__(self, k: int = 4, obs_dim: int = 18):
        self.k       = k
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


# ── Improved DuelingDQN with LayerNorm ─────────────────────────────────────────

class DuelingDQN(nn.Module):
    """Dueling DDQN: shared encoder with LayerNorm → Value + Advantage streams.

    LayerNorm stabilises activations across training steps, helping in
    environments with very different reward magnitudes.
    """

    def __init__(self, in_dim: int, n_actions: int = 5, hidden: int = 256):
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
        v = self.value_stream(h)            # (B, 1)
        a = self.adv_stream(h)              # (B, n_actions)
        return v + a - a.mean(dim=1, keepdim=True)


# ── Sum-Tree for Prioritized Replay ────────────────────────────────────────────

class SumTree:
    """Binary sum-tree for O(log n) priority updates and proportional sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: List[Optional[object]] = [None] * capacity
        self._write   = 0
        self.n_entries = 0

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: object) -> None:
        idx = self._write + self.capacity - 1
        self.data[self._write] = data
        self.update(idx, priority)
        self._write = (self._write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


# ── Prioritized Experience Replay ──────────────────────────────────────────────

@dataclass
class Transition:
    s:    np.ndarray
    a:    int
    r:    float
    s2:   np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    """PER with IS-weight correction.

    Parameters follow Schaul et al. (2015):
      α  — exponent for priority (0 = uniform, 1 = fully prioritised)
      β  — IS weight exponent (anneals from β_start → 1.0 over training)
    """

    _EPSILON = 1e-5   # Avoid zero priority

    def __init__(
        self,
        capacity:     int   = 100_000,
        alpha:        float = 0.6,
        beta_start:   float = 0.4,
        beta_end:     float = 1.0,
        beta_steps:   int   = 500_000,
    ):
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self._step      = 0
        self._max_p     = 1.0

        self.tree       = SumTree(capacity)

    # ── Beta annealing ─────────────────────────────────────────────────────────

    def _beta(self) -> float:
        frac = min(self._step / self.beta_steps, 1.0)
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    # ── Priority helpers ───────────────────────────────────────────────────────

    def _priority(self, error: float) -> float:
        return (abs(error) + self._EPSILON) ** self.alpha

    # ── Buffer operations ──────────────────────────────────────────────────────

    def add(self, t: Transition, error: Optional[float] = None) -> None:
        p = self._priority(error) if error is not None else self._max_p
        self.tree.add(p, t)

    def sample(self, batch: int):
        self._step += batch

        items:      List[Transition] = []
        idxs:       List[int]        = []
        priorities: List[float]      = []
        beta        = self._beta()

        segment = self.tree.total / batch
        for i in range(batch):
            lo = segment * i
            hi = segment * (i + 1)
            s  = np.random.uniform(lo, hi)
            idx, p, data = self.tree.get(s)
            items.append(data)
            idxs.append(idx)
            priorities.append(max(p, self._EPSILON))

        # IS weights
        probs   = np.array(priorities) / self.tree.total
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        s_t  = torch.tensor(np.stack([it.s    for it in items]))
        a_t  = torch.tensor(np.array([it.a    for it in items], dtype=np.int64))
        r_t  = torch.tensor(np.array([it.r    for it in items], dtype=np.float32))
        s2_t = torch.tensor(np.stack([it.s2   for it in items]))
        d_t  = torch.tensor(np.array([it.done for it in items], dtype=np.float32))
        w_t  = torch.tensor(weights)

        return s_t, a_t, r_t, s2_t, d_t, w_t, idxs

    def update_priorities(self, idxs: List[int], errors: np.ndarray) -> None:
        for idx, err in zip(idxs, errors):
            p = self._priority(float(err))
            self._max_p = max(self._max_p, p)
            self.tree.update(idx, p)

    def __len__(self) -> int:
        return self.tree.n_entries


# ── Running reward normalisation ───────────────────────────────────────────────

class RunningNorm:
    """Online EMA mean/variance normaliser for reward.

    Keeps a running estimate so the normalisation adapts as the agent improves.
    """

    def __init__(self, alpha: float = 0.001, clip: float = 5.0):
        self.alpha = alpha
        self.clip  = clip
        self._mean: float = 0.0
        self._var:  float = 1.0

    def normalise(self, r: float) -> float:
        self._mean = (1 - self.alpha) * self._mean + self.alpha * r
        self._var  = (1 - self.alpha) * self._var  + self.alpha * (r - self._mean) ** 2
        std = max(float(np.sqrt(self._var)), 1e-4)
        return float(np.clip((r - self._mean) / std, -self.clip, self.clip))


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
    fs     = FrameStack(k=args.frame_stack, obs_dim=18)
    in_dim = fs.dim  # 72 for k=4

    # ── Networks ───────────────────────────────────────────────────────────────
    q   = DuelingDQN(in_dim=in_dim, hidden=args.hidden)
    tgt = DuelingDQN(in_dim=in_dim, hidden=args.hidden)

    if args.warmstart and os.path.exists(args.warmstart):
        try:
            sd = torch.load(args.warmstart, map_location="cpu", weights_only=True)
            missing, unexpected = q.load_state_dict(sd, strict=False)
            loaded = len(sd) - len(unexpected)
            print(
                f"Warm-started {loaded}/{len(sd)} keys from {args.warmstart}"
                + (f" (missing: {missing})" if missing else "")
                + (f" (unexpected: {unexpected})" if unexpected else "")
            )
        except Exception as e:
            print(f"Warm-start failed ({type(e).__name__}: {e}), training from scratch.")

    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt    = optim.Adam(q.parameters(), lr=args.lr)
    replay = PrioritizedReplayBuffer(
        capacity   = args.replay,
        alpha      = args.per_alpha,
        beta_start = args.per_beta,
        beta_steps = args.eps_decay_steps,
    )
    rnorm = RunningNorm(alpha=0.001, clip=5.0)

    # ── Single persistent environment ──────────────────────────────────────────
    env = OBELIX(
        scaling_factor = args.scaling_factor,
        arena_size     = args.arena_size,
        max_steps      = args.max_steps,
        wall_obstacles = args.wall_obstacles,
        difficulty     = args.difficulty,
        box_speed      = args.box_speed,
        seed           = args.seed,
    )

    total_steps    = 0
    best_mean      = float("-inf")
    recent_returns: deque = deque(maxlen=100)

    def eps_fn(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        frac = t / args.eps_decay_steps
        return args.eps_start + frac * (args.eps_end - args.eps_start)

    for ep in range(args.episodes):
        raw_obs = env.reset(seed=args.seed + ep)
        s       = fs.reset(raw_obs)
        ep_ret  = 0.0

        for _ in range(args.max_steps):
            epsilon = eps_fn(total_steps)

            # ε-greedy
            if np.random.rand() < epsilon:
                a = np.random.randint(len(ACTIONS))
            else:
                with torch.no_grad():
                    qs = q(torch.tensor(s).unsqueeze(0)).squeeze(0).numpy()
                a = int(np.argmax(qs))

            raw_obs2, raw_r, done = env.step(ACTIONS[a], render=False)
            s2    = fs.step(raw_obs2)
            r_raw = float(raw_r)
            r     = rnorm.normalise(r_raw)
            ep_ret += r_raw

            replay.add(Transition(s=s, a=a, r=r, s2=s2, done=bool(done)))
            s = s2
            total_steps += 1

            # ── Learning ───────────────────────────────────────────────────────
            if len(replay) >= max(args.warmup, args.batch):
                sb, ab, rb, s2b, db, wb, idxs = replay.sample(args.batch)

                with torch.no_grad():
                    next_a   = q(s2b).argmax(dim=1, keepdim=True)
                    next_val = tgt(s2b).gather(1, next_a).squeeze(1)
                    y        = rb + args.gamma * (1.0 - db) * next_val

                pred           = q(sb).gather(1, ab.unsqueeze(1)).squeeze(1)
                td_errors_abs  = (pred - y).detach().abs().numpy()

                # IS-weighted smooth-L1 loss
                elementwise = nn.functional.smooth_l1_loss(pred, y, reduction="none")
                loss        = (wb * elementwise).mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()

                # Update priorities with absolute TD errors
                replay.update_priorities(idxs, td_errors_abs)

                if total_steps % args.target_sync == 0:
                    tgt.load_state_dict(q.state_dict())

            if done:
                break

        recent_returns.append(ep_ret)

        # ── Logging + saving ───────────────────────────────────────────────────
        if (ep + 1) % 25 == 0:
            mean_ret = float(np.mean(recent_returns))
            print(
                f"ep={ep+1:6d} | ret={ep_ret:9.1f} | "
                f"mean100={mean_ret:9.1f} | ε={epsilon:.3f} | "
                f"buf={len(replay):6d} | steps={total_steps}"
            )
            if mean_ret > best_mean:
                best_mean = mean_ret
                torch.save(q.state_dict(), args.out)
                print(f"  → New best mean100={best_mean:.1f}. Saved to {args.out}")

    print(
        f"\nTraining done. Best mean100={best_mean:.1f}. Weights: {args.out} "
        f"ep={ep+1} | mean100={float(np.mean(recent_returns)):.1f} | ε={epsilon:.3f}"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="DDQN+PER Trainer for OBELIX v4 (Submission 6)")
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights.pth")
    ap.add_argument("--warmstart",       type=str,   default=None,
                    help="Path to previous weights.pth to warm-start training from")
    ap.add_argument("--episodes",        type=int,   default=10_000)
    ap.add_argument("--max_steps",       type=int,   default=1_000)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--frame_stack",     type=int,   default=4)
    ap.add_argument("--hidden",          type=int,   default=256,
                    help="Hidden layer size (default: 256, was 128 in v2/v3)")
    # DQN hyperparams
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=1e-4)
    ap.add_argument("--batch",           type=int,   default=512)
    ap.add_argument("--replay",          type=int,   default=200_000)
    ap.add_argument("--warmup",          type=int,   default=5_000)
    ap.add_argument("--target_sync",     type=int,   default=2_000)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int,   default=500_000,
                    help="Total env-steps over which epsilon decays (default: 500K)")
    # PER hyperparams
    ap.add_argument("--per_alpha",       type=float, default=0.6,
                    help="PER priority exponent α (0=uniform, 1=full priority)")
    ap.add_argument("--per_beta",        type=float, default=0.4,
                    help="PER IS-weight initial β (anneals to 1.0 over training)")
    ap.add_argument("--seed",            type=int,   default=0)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
