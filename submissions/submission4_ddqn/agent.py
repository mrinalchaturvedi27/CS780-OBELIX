"""Submission 4 — Double Deep Q-Network (DDQN) agent.

This agent loads a neural network (weights.pth) trained offline by
train_ddqn.py and uses it to pick the action with the highest Q-value for
the current observation.

Prerequisites
-------------
1. Run the training script to produce weights.pth:
       python train_ddqn.py --episodes 5000 \\
           --out_weights submissions/submission4_ddqn/weights.pth
2. Zip for Codabench:
       python package_submission.py submissions/submission4_ddqn

Network architecture: 18 → 64 → ReLU → 64 → ReLU → 5

IMPORTANT: all inference runs on CPU so the submission is compatible with
Codabench's CPU-only evaluation environment.

Reference: Hasselt et al., "Deep Reinforcement Learning with Double
Q-learning" (https://arxiv.org/pdf/1509.06461)
"""

import os
import numpy as np

ACTIONS = ("L45", "L22", "FW", "R22", "R45")

_MODEL = None  # lazy-loaded on first call


def _build_model():
    import torch
    import torch.nn as nn

    class QNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(18, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, len(ACTIONS)),
            )

        def forward(self, x):
            return self.net(x)

    return QNetwork()


def _load_once() -> None:
    global _MODEL
    if _MODEL is not None:
        return

    import torch

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Weights not found at {path}. "
            "Train first: python train_ddqn.py --out_weights <path>"
        )

    model = _build_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    _MODEL = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """Return the greedy action from the trained DDQN network."""
    _load_once()

    import torch

    x = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        q_values = _MODEL(x).squeeze(0).numpy()

    return ACTIONS[int(np.argmax(q_values))]
