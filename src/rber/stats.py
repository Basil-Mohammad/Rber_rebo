"""Statistics for reporting results over seeds (following Agarwal et al., 2021).

IQM = interquartile mean (mean of the middle 50%); confidence intervals are percentile bootstrap intervals
over seeds; the probability of improvement P(X > Y) is the Mann-Whitney statistic; paired comparisons on the
same random worlds use the Wilcoxon signed-rank test with Holm correction.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st


def iqm(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def boot_ci(x, fn=np.mean, n_boot=10_000, alpha=0.05, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    b = np.array([fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    return float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def summary(x):
    x = np.asarray(x, float)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                iqm=iqm(x), iqm_ci=boot_ci(x, iqm), mean_ci=boot_ci(x, np.mean))


def prob_improvement(x, y, n_boot=5000, seed=0):
    """P(X > Y) + 0.5 P(X = Y) with a bootstrap CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def pi(a, b):
        return float(((a[:, None] > b[None]) + 0.5 * (a[:, None] == b[None])).mean())
    rng = np.random.default_rng(seed)
    bs = [pi(x[rng.integers(0, len(x), len(x))], y[rng.integers(0, len(y), len(y))]) for _ in range(n_boot)]
    return pi(x, y), (float(np.quantile(bs, .025)), float(np.quantile(bs, .975)))


def wilcoxon_paired(x, y):
    """Two-sided Wilcoxon signed-rank p-value for paired samples (1.0 if all differences are zero)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    if np.allclose(d, 0):
        return 1.0
    return float(st.wilcoxon(x, y, zero_method="zsplit").pvalue)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj
