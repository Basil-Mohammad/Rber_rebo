"""Unit tests of the estimator, the posterior, and the theory (run: python tests/test_core.py, or pytest)."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rber import theory                                              # noqa: E402
from rber.domain import N_OPT, build_world, load_library             # noqa: E402
from rber.rewards import SkillPosterior, batch_rewards, group_advantage  # noqa: E402
from rber.tabular import softmax                                     # noqa: E402

LIB = load_library()


def test_theorem1_identity_and_theorem2_bound():
    """tr Cov[EXEC] - tr Cov[RB*] = E[R(1-R)||psi||^2] exactly; E||g_EX||^2 / E||g_RB||^2 >= 1/R_max (b=0)."""
    rng = np.random.default_rng(0)
    for K in (2, 3, 4):
        w = build_world(LIB, K=K, seed=K)
        for t in w.train[:3]:
            P = softmax(rng.normal(0, 1.5, (K, N_OPT)))
            c = theory.trace_cov(w, t, P)
            assert abs(c["exec"] - c["rb"] - c["gap"]) < 1e-12
            m0 = theory.second_moments(w, t, P, 0.0)
            assert m0["exec"] / m0["rb"] * m0["Rmax"] >= 1 - 1e-12


def test_collapse_probability():
    """Proposition 1: P(all binary outcomes equal) = (1-J)^G + J^G for i.i.d. plans (Monte Carlo check)."""
    w = build_world(LIB, K=2, seed=1); t = w.train[0]; rng = np.random.default_rng(1)
    P = np.full((2, N_OPT), 0.25); G = 4
    J = sum(np.prod(P[np.arange(2), c]) * w.success(t, c) for c in itertools.product(range(N_OPT), repeat=2))
    hits, n = 0, 40_000
    for _ in range(n):
        ch = rng.integers(0, N_OPT, (G, 2))
        y = [rng.random() < w.success(t, c) for c in ch]
        hits += all(y) or not any(y)
    assert abs(hits / n - ((1 - J) ** G + J ** G)) < 0.01


def test_posterior_is_censored_and_rber_is_lagged():
    """Steps after the first failure are not observed; RBER uses the posterior before the batch."""
    w = build_world(LIB, K=3, seed=2); t = w.train[0]
    post = SkillPosterior(len(LIB.p))
    valid = np.array([np.flatnonzero(t.valid[k])[0] for k in range(3)])
    ch = np.tile(valid, (8, 1))
    r, n = batch_rewards("RBER", w, t, ch, np.random.default_rng(0), post)
    assert n == 8 and np.allclose(r, 0.5 ** 3)          # prior means only: lag
    sk = w.skills(t, valid)
    obs = post.a[0, sk] + post.b[0, sk] - 2            # observations per step
    assert obs[0] == 8 and obs[0] >= obs[1] >= obs[2]   # censoring: later steps observed less often


def test_invalid_plans_cost_nothing_and_group_advantage():
    w = build_world(LIB, K=3, seed=3); t = w.train[0]
    bad = np.array([np.flatnonzero(~t.valid[k])[0] for k in range(3)])
    r, n = batch_rewards("EXEC", w, t, np.tile(bad, (8, 1)), np.random.default_rng(0))
    assert n == 0 and np.all(r == 0) and np.all(group_advantage(r) == 0)


def test_batch_level_lag_with_snapshot():
    """With several tasks per update, all rewards of the batch use the posterior from before the batch."""
    import copy
    w = build_world(LIB, K=3, seed=4); post = SkillPosterior(len(LIB.p)); snap = copy.deepcopy(post)
    rng = np.random.default_rng(0)
    for t in w.train[:4]:
        valid = np.array([np.flatnonzero(t.valid[k])[0] for k in range(3)])
        r, _ = batch_rewards("RBER", w, t, np.tile(valid, (8, 1)), rng, post, reward_post=snap)
        assert np.allclose(r, 0.5 ** 3)                    # snapshot = prior, although post was updated
    assert post.count().sum() > snap.count().sum()          # ... and the real posterior did learn


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("passed:", name)
    print("ALL CORE TESTS PASSED")
