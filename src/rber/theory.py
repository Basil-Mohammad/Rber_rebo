"""Exact (enumeration-based) quantities used to verify the theory on the tabular planner.

For one task with a factorized softmax policy pi(a) = prod_k softmax(L_k)[a_k], the score function is
grad_L log pi(a) = onehot(a) - P  (a K x N_OPT matrix), with squared norm sum_k ||e_{a_k} - P_k||^2.
"""
from __future__ import annotations

import itertools

import numpy as np

from .domain import N_OPT


def enumerate_plans(world, task, P):
    """All plans with their probability, true success R(a) and squared score norm."""
    K = world.K
    rows = []
    for c in itertools.product(range(N_OPT), repeat=K):
        c = np.array(c)
        prob = float(np.prod(P[np.arange(K), c]))
        g2 = float(sum(np.sum((np.eye(N_OPT)[c[k]] - P[k]) ** 2) for k in range(K)))
        rows.append((prob, world.success(task, c), g2))
    return np.array(rows)  # columns: prob, R, ||score||^2


def second_moments(world, task, P, baseline: float | None = None):
    """E||g||^2 for the execution estimator (Y-b)*score and the oracle Rao-Blackwellized (R-b)*score,
    and the Theorem-1 gap E[R(1-R)||score||^2]. baseline=None uses b = J (the policy's success rate)."""
    tab = enumerate_plans(world, task, P)
    w, R, g2 = tab[:, 0], tab[:, 1], tab[:, 2]
    J = float(np.sum(w * R))
    b = J if baseline is None else baseline
    exec_m2 = float(np.sum(w * (R * (1 - b) ** 2 + (1 - R) * b ** 2) * g2))   # E[(Y-b)^2 ||s||^2]
    rb_m2 = float(np.sum(w * (R - b) ** 2 * g2))                               # E[(R-b)^2 ||s||^2]
    gap = float(np.sum(w * R * (1 - R) * g2))
    # exact mean gradient (same for both estimators): sum_a pi(a) (R(a)-b) score(a)
    return dict(J=J, exec=exec_m2, rb=rb_m2, gap=gap, Rmax=float(R.max()))


def trace_cov(world, task, P, baseline=None):
    """Trace of the covariance of both estimators (second moment minus squared norm of the mean)."""
    K = world.K
    mean = np.zeros((K, N_OPT))
    tab = []
    for c in itertools.product(range(N_OPT), repeat=K):
        c = np.array(c)
        prob = float(np.prod(P[np.arange(K), c]))
        s = np.eye(N_OPT)[c] - P
        R = world.success(task, c)
        tab.append((prob, R, s))
    J = sum(p * R for p, R, _ in tab)
    b = J if baseline is None else baseline
    for p, R, s in tab:
        mean += p * (R - b) * s
    m = second_moments(world, task, P, baseline)
    mn = float(np.sum(mean ** 2))
    return dict(exec=m["exec"] - mn, rb=m["rb"] - mn, gap=m["gap"], mean_sq=mn)
