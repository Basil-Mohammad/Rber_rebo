"""Unit tests of the extension methods (run: python tests/test_extension.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rber.domain import build_world, load_library          # noqa: E402
from rber import tabular                                     # noqa: E402
from rber.extension import ContextPosterior, SuccessPredictor, loo_batch, train_ext  # noqa: E402

LIB = load_library()


def test_reproduces_original_methods():
    w = build_world(LIB, K=3, seed=5)
    for m in ("EXEC", "RBER", "STEP"):
        a = tabular.train(w, m, 1.0, seed=3, n_updates=200, eval_every=50)
        b = train_ext(w, m, 1.0, seed=3, n_updates=200, eval_every=50)
        assert a["curve"] == b["curve"], m


def test_loo_equals_posterior_without_own_plan():
    w = build_world(LIB, K=3, seed=6); t = w.train[0]; rng = np.random.default_rng(0)
    for method in ("RBER-LOO", "RBER-C"):
        post = ContextPosterior(len(LIB.p))
        ch = np.array([[np.flatnonzero(t.valid[k])[j % 3] for k in range(3)] for j in range(8)])
        r, n = loo_batch(method, w, t, ch, np.random.default_rng(1), post)
        # rebuild: replay the same outcomes but skip plan 0, then evaluate plan 0
        rng2 = np.random.default_rng(1); post2 = ContextPosterior(len(LIB.p)); outs = []
        for c in ch:
            sk = w.skills(t, c); ok, out = w.execute(sk, rng2); outs.append((sk[:len(out)], [int(z) for z in out]))
        for o in outs[1:]:
            post2.add(*o)
        v = post2.plan_value(w.skills(t, ch[0]), contextual=(method == "RBER-C"))
        assert abs(v - r[0]) < 1e-12, (method, v, r[0])


def test_lsp_lin_recovers_product_model():
    rng = np.random.default_rng(0); S, K = 6, 2; p = np.array([.9, .5, .2, .8, .6, .3])
    m = SuccessPredictor("lin", S, K, 0.05, rng)
    for _ in range(4000):
        sk = rng.choice(S, K, replace=False); m.add(sk, rng.random() < np.prod(p[sk]))
    for _ in range(500):
        m.fit()
    est = m.predict([[0, 3], [2, 5]])
    assert np.allclose(est, [p[0] * p[3], p[2] * p[5]], atol=0.06), est


def test_all_extension_methods_learn():
    w = build_world(LIB, K=3, seed=7)
    for m in ("RBER-LOO", "RBER-C", "EXEC-RLOO", "EXEC-PPO", "LSP-lin", "LSP-mlp", "TS-plan"):
        r = train_ext(w, m, 1.0, seed=0, n_updates=600, eval_every=200)
        assert r["final"] > r["curve"][0][2], (m, r["curve"])


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("passed:", name)
    print("ALL EXTENSION TESTS PASSED")
