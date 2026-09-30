"""Reward signals for training a planner, all computed from the same sampled plans.

VERIF   offline verification: plan valid and agreement with a reliability-blind reference plan (no execution)
STEP    execution, dense progress: fraction of steps completed before the first failure
EXEC    execution, binary success of the plan
RBER    execution, Rao-Blackwellized execution reward: product of posterior-mean skill success probabilities.
        The posterior is shared by all tasks and is *lagged*: the reward of a batch uses the posterior
        before that batch's outcomes are added (Section IV; makes the reward independent of the sample's own
        outcome given the history).
RBER-L  RBER with one posterior per task (no sharing across tasks)                 [ablation]
RBER-NL RBER whose posterior already includes the current batch (non-lagged; used in our pilot) [ablation]
RB*     oracle Rao-Blackwellization with the true success probability R(a)     [upper reference]
"""
from __future__ import annotations

import numpy as np

METHODS = ("VERIF", "STEP", "EXEC", "RBER", "RBER-L", "RBER-NL", "RB*")
EXECUTES = {"STEP", "EXEC", "RBER", "RBER-L", "RBER-NL", "RB*"}


class SkillPosterior:
    """Independent Beta(a0, b0) posteriors over skill success probabilities, optionally one set per task."""

    def __init__(self, n_skills: int, n_groups: int = 1, a0: float = 1.0, b0: float = 1.0):
        self.a = np.full((n_groups, n_skills), a0, float)
        self.b = np.full((n_groups, n_skills), b0, float)

    def mean(self, g: int = 0) -> np.ndarray:
        return self.a[g] / (self.a[g] + self.b[g])

    def count(self, g: int = 0) -> np.ndarray:
        return self.a[g] + self.b[g]

    def update(self, sk: np.ndarray, outcomes: list, g: int = 0):
        """Censored observation: steps up to and including the first failure."""
        for s, z in zip(sk, outcomes):
            self.a[g, s] += z; self.b[g, s] += 1 - z


def batch_rewards(method: str, world, task, choices: np.ndarray, rng: np.random.Generator,
                  post: SkillPosterior | None = None, group: int = 0, reward_post: SkillPosterior | None = None):
    """Rewards for G sampled plans of one task. Returns (rewards[G], executions used).

    ``post`` is updated with the censored outcomes of the executed plans. For RBER/RBER-L the reward uses
    ``reward_post`` if given (a snapshot taken before the whole batch, used when a batch contains several
    tasks), otherwise ``post`` before this call's updates."""
    G = len(choices)
    r = np.zeros(G)
    plans = [world.skills(task, c) for c in choices]
    if method == "VERIF":
        for i, (c, sk) in enumerate(zip(choices, plans)):
            r[i] = 0.0 if sk is None else float(np.mean(np.asarray(c) == task.ref))
        return r, 0
    if method in ("RBER", "RBER-L"):  # lagged: reward from the posterior *before* this batch
        m = (reward_post if reward_post is not None else post).mean(group)
        for i, sk in enumerate(plans):
            r[i] = 0.0 if sk is None else float(np.prod(m[sk]))
    if method == "RB*":
        for i, c in enumerate(choices):
            r[i] = world.success(task, c)
    n_exec = 0
    for i, sk in enumerate(plans):
        if sk is None:
            continue                      # an invalid plan fails at once and costs no execution
        n_exec += 1
        ok, out = world.execute(sk, rng)
        if post is not None:
            post.update(sk[:len(out)], out, group)
        if method == "EXEC":
            r[i] = float(ok)
        elif method == "STEP":
            r[i] = sum(out[:len(out) - (not ok)]) / world.K if not ok else 1.0
    if method == "RBER-NL":  # posterior including this batch
        m = post.mean(group)
        for i, sk in enumerate(plans):
            r[i] = 0.0 if sk is None else float(np.prod(m[sk]))
    return r, n_exec


def group_advantage(r: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Group-normalized advantage (GRPO-style); zero when all rewards in the group are equal."""
    return (r - r.mean()) / (r.std() + eps)
