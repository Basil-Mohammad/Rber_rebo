"""Tabular softmax planner (one logit table per task and step) trained with group-baseline policy gradients.

The update is REINFORCE with a group-normalized baseline, one on-policy step per batch (identical to GRPO
with a single optimisation step, where the importance ratio is 1 and clipping is inactive).
"""
from __future__ import annotations

import numpy as np

from .domain import N_OPT
from .rewards import SkillPosterior, batch_rewards, group_advantage


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


def greedy_value(world, L, tasks=None):
    tasks = world.train if tasks is None else tasks
    return float(np.mean([world.success(t, L[i].argmax(-1)) for i, t in enumerate(tasks)]))


def train(world, method: str, lr: float, seed: int, n_updates: int = 2000, G: int = 8, eval_every: int = 50,
          prior=(1.0, 1.0)):
    """Returns a dict with learning curves (update, executions, normalized greedy success)."""
    rng = np.random.default_rng(seed)
    exec_rng = np.random.default_rng(seed + 10_000)      # separate stream for execution outcomes
    T, K = len(world.train), world.K
    L = np.zeros((T, K, N_OPT))
    n_groups = T if method == "RBER-L" else 1
    post = SkillPosterior(len(world.lib.p), n_groups, *prior) if method.startswith("RBER") else None
    opt = np.mean([world.optimum(t) for t in world.train])
    curve = [(0, 0, greedy_value(world, L) / opt)]
    execs = 0
    for u in range(1, n_updates + 1):
        i = int(rng.integers(T)); t = world.train[i]
        P = softmax(L[i])                                                  # [K, N_OPT]
        u01 = rng.random((G, K, 1))
        ch = (u01 > np.cumsum(P, -1)[None]).sum(-1)                        # inverse-CDF sampling [G, K]
        ch = np.minimum(ch, N_OPT - 1)
        r, n = batch_rewards(method, world, t, ch, exec_rng, post, group=i if method == "RBER-L" else 0)
        execs += n
        A = group_advantage(r)
        onehot = np.eye(N_OPT)[ch]                                         # [G, K, N_OPT]
        L[i] += lr * (A[:, None, None] * (onehot - P[None])).mean(0)
        if u % eval_every == 0:
            curve.append((u, execs, greedy_value(world, L) / opt))
    return dict(method=method, lr=lr, seed=seed, curve=curve, optimum=opt,
                final=curve[-1][2], held=None)
