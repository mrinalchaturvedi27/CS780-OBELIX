"""Ultra-Fast DDQN Trainer for OBELIX (Submission 5+).

Uses Python multiprocessing to run multiple environments in parallel,
collecting experience ~4-8x faster than the single-threaded v2 script.

Run with:
  python train_ddqn_v3.py --obelix_py ./obelix.py --num_envs 8 ...
"""

from __future__ import annotations
import argparse
import random
import multiprocessing as mp
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["L45", "L22", "FW", "R22", "R45"]

# ── Frame stacking ─────────────────────────────────────────────────────────────

class FrameStack:
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
        h = self.shared(x)
        v = self.value_stream(h)
        a = self.adv_stream(h)
        return v + a - a.mean(dim=1, keepdim=True)

# ── Multiprocessing Environment Workers ────────────────────────────────────────

def worker_process(worker_id, obelix_path, args, cmd_pipe, result_pipe):
    """Worker process running a single OBELIX environment."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("obelix_env", obelix_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    OBELIX = mod.OBELIX

    # Unique seed per worker
    seed = args.seed + (worker_id * 10000)
    np.random.seed(seed)
    random.seed(seed)

    env = OBELIX(
        scaling_factor=args.scaling_factor,
        arena_size=args.arena_size,
        max_steps=args.max_steps,
        wall_obstacles=args.wall_obstacles,
        difficulty=args.difficulty,
        box_speed=args.box_speed,
        seed=seed,
    )

    while True:
        cmd, data = cmd_pipe.recv()
        if cmd == "reset":
            obs = env.reset()
            result_pipe.send((obs, 0.0, False))
        elif cmd == "step":
            obs, r, done = env.step(ACTIONS[data], render=False)
            result_pipe.send((obs, float(r), bool(done)))
        elif cmd == "close":
            break


class VecEnv:
    """Manages multiple environment workers to step them in parallel."""
    def __init__(self, num_envs, obelix_path, args):
        self.num_envs = num_envs
        self.workers = []
        self.cmd_pipes = []
        self.result_pipes = []

        for i in range(num_envs):
            p_cmd, c_cmd = mp.Pipe()
            p_res, c_res = mp.Pipe()
            w = mp.Process(target=worker_process, args=(i, obelix_path, args, c_cmd, p_res))
            w.daemon = True
            w.start()
            self.workers.append(w)
            self.cmd_pipes.append(p_cmd)
            self.result_pipes.append(c_res)

    def reset(self):
        for p in self.cmd_pipes:
            p.send(("reset", None))
        results = [p.recv() for p in self.result_pipes]
        # returns [obs, r, done] ... r and done are dummy here
        return [res[0] for res in results]

    def step_async(self, actions):
        for p, a in zip(self.cmd_pipes, actions):
            p.send(("step", a))

    def step_wait(self):
        results = [p.recv() for p in self.result_pipes]
        observations = [res[0] for res in results]
        rewards = [res[1] for res in results]
        dones = [res[2] for res in results]
        return observations, rewards, dones

    def close(self):
        for p in self.cmd_pipes:
            p.send(("close", None))
        for w in self.workers:
            w.join()

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

def normalise_reward(r: float) -> float:
    return float(np.clip(r / 200.0, -1.0, 10.0))

# ── Main Training Loop ─────────────────────────────────────────────────────────

def train(args) -> None:
    mp.set_start_method("spawn", force=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Starting {args.num_envs} environment workers in parallel...")
    vec_env = VecEnv(args.num_envs, args.obelix_py, args)

    fs_list = [FrameStack(k=args.frame_stack, obs_dim=18) for _ in range(args.num_envs)]
    in_dim = fs_list[0].dim

    q   = DuelingDQN(in_dim=in_dim)
    tgt = DuelingDQN(in_dim=in_dim)
    tgt.load_state_dict(q.state_dict())
    tgt.eval()

    opt    = optim.Adam(q.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.replay)

    total_steps = 0
    best_mean   = float("-inf")
    episodes_completed = 0

    def eps(t: int) -> float:
        if t >= args.eps_decay_steps:
            return args.eps_end
        return args.eps_start + (t / args.eps_decay_steps) * (args.eps_end - args.eps_start)

    recent_returns = deque(maxlen=100)
    current_ep_returns = np.zeros(args.num_envs)
    current_ep_steps = np.zeros(args.num_envs, dtype=int)

    # Initial reset
    raw_obs_list = vec_env.reset()
    s_list = [fs.reset(obs) for fs, obs in zip(fs_list, raw_obs_list)]

    while episodes_completed < args.episodes:
        # Determine actions for all envs
        epsilon = eps(total_steps)
        actions = []
        
        # Batch inference for greedy actions
        with torch.no_grad():
            s_tensor = torch.tensor(np.stack(s_list))
            qs_batch = q(s_tensor).numpy()

        for i in range(args.num_envs):
            if np.random.rand() < epsilon:
                actions.append(np.random.randint(len(ACTIONS)))
            else:
                actions.append(int(np.argmax(qs_batch[i])))

        # Step all environments simultaneously
        vec_env.step_async(actions)
        raw_obs2_list, raw_r_list, done_list = vec_env.step_wait()

        # Process results
        for i in range(args.num_envs):
            s2 = fs_list[i].step(raw_obs2_list[i])
            r_val = float(raw_r_list[i])
            r_norm = normalise_reward(r_val)
            done = bool(done_list[i])

            current_ep_returns[i] += r_val
            current_ep_steps[i] += 1
            total_steps += 1

            # Auto-done if max steps reached
            if current_ep_steps[i] >= args.max_steps:
                done = True

            replay.add(Transition(s=s_list[i], a=actions[i], r=r_norm, s2=s2, done=done))
            s_list[i] = s2

            if done:
                recent_returns.append(current_ep_returns[i])
                episodes_completed += 1
                
                # Check metrics & save every 25 episodes
                if episodes_completed % 25 == 0:
                    mean_ret = float(np.mean(recent_returns))
                    print(
                        f"ep={episodes_completed:5d} | "
                        f"mean100={mean_ret:8.1f} | ε={epsilon:.3f} | "
                        f"buf={len(replay)} | steps={total_steps}"
                    )
                    if mean_ret > best_mean and episodes_completed > 50:
                        best_mean = mean_ret
                        torch.save(q.state_dict(), args.out)
                        print(f"  → New best! Saved to {args.out}")
                
                # Manual reset for this individual environment
                vec_env.cmd_pipes[i].send(("reset", None))
                raw_obs, _, _ = vec_env.result_pipes[i].recv()
                s_list[i] = fs_list[i].reset(raw_obs)
                current_ep_returns[i] = 0.0
                current_ep_steps[i] = 0

                if episodes_completed >= args.episodes:
                    break

        # ── Learning step ──────────────────────────────────────────────────
        if len(replay) >= max(args.warmup, args.batch):
            sb, ab, rb, s2b, db = replay.sample(args.batch)

            with torch.no_grad():
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

    vec_env.close()
    print(f"\nTraining done. Best mean100={best_mean:.1f}. Weights: {args.out}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obelix_py",       type=str,   required=True)
    ap.add_argument("--out",             type=str,   default="weights.pth")
    ap.add_argument("--episodes",        type=int,   default=6000)
    ap.add_argument("--max_steps",       type=int,   default=500)
    ap.add_argument("--difficulty",      type=int,   default=0)
    ap.add_argument("--wall_obstacles",  action="store_true")
    ap.add_argument("--box_speed",       type=int,   default=2)
    ap.add_argument("--scaling_factor",  type=int,   default=5)
    ap.add_argument("--arena_size",      type=int,   default=500)
    ap.add_argument("--frame_stack",     type=int,   default=4)
    ap.add_argument("--num_envs",        type=int,   default=8, help="Number of parallel workers")
    
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--lr",              type=float, default=3e-4)
    ap.add_argument("--batch",           type=int,   default=256)
    ap.add_argument("--replay",          type=int,   default=100_000)
    ap.add_argument("--warmup",          type=int,   default=2_000)
    ap.add_argument("--target_sync",     type=int,   default=1_000)
    ap.add_argument("--eps_start",       type=float, default=1.0)
    ap.add_argument("--eps_end",         type=float, default=0.05)
    ap.add_argument("--eps_decay_steps", type=int,   default=150_000)
    ap.add_argument("--seed",            type=int,   default=0)
    args = ap.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()
