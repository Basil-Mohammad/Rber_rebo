"""Extension study (docs/PREREGISTRATION_extension.md): leave-one-out and contextual RBER, and stronger baselines.

Methods trained by :func:`train_ext` (tabular planner, identical random streams to ``rber.tabular.train``):

RBER-LOO   product of posterior means, posterior = history + all *other* plans of the batch (own outcomes excluded)
RBER-C     first-order contextual model m_{s|c} (c = preceding skill or START) shrunk to m_s, leave-one-out
EXEC-RLOO  binary reward, leave-one-out mean baseline, no std normalization
EXEC-PPO   binary reward, group-normalized advantage, 4 clipped epochs per batch (eps = 0.2)
LSP-lin    learned success predictor prod_k sigmoid(theta_{s_k}) fitted on final outcomes only (replay + Adam)
LSP-mlp    learned success predictor, MLP on per-step one-hot skills, fitted on final outcomes only
TS-plan    Thompson sampling over plans with the shared Beta posterior (model-based reference, not a policy)
Also accepts the original methods (EXEC, STEP, RBER, ...) and then reproduces ``rber.tabular.train`` exactly.
"""
from __future__ import annotations

import itertools

import numpy as np

from .domain import N_OPT
from .rewards import SkillPosterior, batch_rewards, group_advantage
from .tabular import greedy_value, softmax

EXT_METHODS = ("RBER-LOO", "RBER-C", "EXEC-RLOO", "EXEC-PPO", "LSP-lin", "LSP-mlp", "TS-plan")


# ----------------------------------------------------------------------------------------------- posteriors
class ContextPosterior:
    """Marginal Beta(a0, b0) counts per skill plus success/attempt counts per (context, skill), where the context
    is the preceding skill, or a START token (index n_skills) for the first step of a plan."""

    def __init__(self, n_skills: int, a0: float = 1.0, b0: float = 1.0, kappa: float = 2.0):
        self.S = n_skills
        self.a = np.full(n_skills, a0); self.b = np.full(n_skills, b0)
        self.ps = np.zeros((n_skills + 1, n_skills)); self.pn = np.zeros((n_skills + 1, n_skills))
        self.kappa = kappa

    def _ctx(self, sk, k):
        return self.S if k == 0 else sk[k - 1]

    def add(self, sk, out, sign=1.0):
        for k, z in enumerate(out):
            s = sk[k]; self.a[s] += sign * z; self.b[s] += sign * (1 - z)
            c = self._ctx(sk, k); self.ps[c, s] += sign * z; self.pn[c, s] += sign

    def plan_value(self, sk, own=None, contextual=True):
        """Posterior-predictive success of a plan; `own` = (skills, outcomes) of this plan, removed first
        (leave-one-out). Contextual: m_{s|c} = (kappa m_s + S_{c,s}) / (kappa + N_{c,s})."""
        a, b = self.a, self.b
        if own is not None:
            osk, oout = own
            a = a.copy(); b = b.copy()
            for k, z in enumerate(oout):
                a[osk[k]] -= z; b[osk[k]] -= 1 - z
        m = a[sk] / (a[sk] + b[sk])
        if not contextual:
            return float(np.prod(m))
        v = 1.0
        for k in range(len(sk)):
            c = self._ctx(sk, k); S, N = self.ps[c, sk[k]], self.pn[c, sk[k]]
            if own is not None and k < len(own[1]):          # own attempt of step k used exactly this (c, s)
                S -= own[1][k]; N -= 1
            v *= (self.kappa * m[k] + S) / (self.kappa + N)
        return float(v)


def loo_batch_multi(method, world, tasks, choices_list, rng, post: ContextPosterior):
    """Execute every valid plan of a batch (several tasks), add all outcomes to the posterior, then reward each
    plan with the posterior that excludes only its own outcomes (leave-one-out over the whole batch)."""
    plans = [[world.skills(t, c) for c in ch] for t, ch in zip(tasks, choices_list)]
    obs, n = [], 0
    for P in plans:
        o_t = []
        for sk in P:
            if sk is None:
                o_t.append(None); continue
            ok, out = world.execute(sk, rng); n += 1
            o = (sk[:len(out)], [int(z) for z in out]); post.add(*o); o_t.append(o)
        obs.append(o_t)
    rewards = []
    for P, o_t in zip(plans, obs):
        r = np.zeros(len(P))
        for i, sk in enumerate(P):
            if sk is not None:
                r[i] = post.plan_value(sk, own=o_t[i], contextual=(method == "RBER-C"))
        rewards.append(r)
    return rewards, n


def loo_batch(method, world, task, choices, rng, post: ContextPosterior):
    r, n = loo_batch_multi(method, world, [task], [choices], rng, post)
    return r[0], n


# ----------------------------------------------------------------------------------------------- learned predictors
class Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(x) for x in params]; self.v = [np.zeros_like(x) for x in params]

    def step(self, grads):
        self.t += 1
        for x, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
            x += self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)  # ascent


class SuccessPredictor:
    """P(Y=1 | plan) learned from final binary outcomes only (no step outcomes)."""

    def __init__(self, kind, n_skills, K, lr, rng, hidden=64):
        self.kind, self.S, self.K, self.rng = kind, n_skills, K, rng
        if kind == "lin":
            self.theta = np.zeros(n_skills); self.opt = Adam([self.theta], lr)
        else:
            d = K * n_skills
            self.W1 = rng.normal(0, 1 / np.sqrt(K), (hidden, d)); self.b1 = np.zeros(hidden)
            self.w2 = rng.normal(0, 1 / np.sqrt(hidden), hidden); self.b2 = np.zeros(1)
            self.opt = Adam([self.W1, self.b1, self.w2, self.b2], lr)
        self.X, self.Y = [], []

    def _x(self, SK):
        X = np.zeros((len(SK), self.K * self.S))
        for i, sk in enumerate(SK):
            X[i, np.arange(self.K) * self.S + sk] = 1
        return X

    def predict(self, SK):
        SK = np.asarray(SK)
        if self.kind == "lin":
            return np.prod(1 / (1 + np.exp(-self.theta[SK])), axis=1)
        h = np.tanh(self._x(SK) @ self.W1.T + self.b1)
        return 1 / (1 + np.exp(-(h @ self.w2 + self.b2[0])))

    def add(self, sk, y):
        self.X.append(np.asarray(sk)); self.Y.append(float(y))

    def fit(self, steps=4, batch=64):
        if not self.X:
            return
        for _ in range(steps):
            idx = self.rng.integers(0, len(self.X), batch)
            SK = np.stack([self.X[i] for i in idx]); y = np.array([self.Y[i] for i in idx])
            if self.kind == "lin":
                sg = 1 / (1 + np.exp(-self.theta[SK]))                       # [B, K]
                P = np.clip(np.prod(sg, 1), 1e-9, 1 - 1e-6)
                coef = np.where(y == 1, 1.0, -P / (1 - P))                  # d loglik / d log P
                g = np.zeros_like(self.theta)
                np.add.at(g, SK, coef[:, None] * (1 - sg))
                self.opt.step([g / batch])
            else:
                X = self._x(SK); h = np.tanh(X @ self.W1.T + self.b1)
                p = 1 / (1 + np.exp(-(h @ self.w2 + self.b2[0])))
                d = (y - p) / batch                                         # d loglik / d logit
                gw2 = h.T @ d; gb2 = np.array([d.sum()])
                dh = np.outer(d, self.w2) * (1 - h ** 2)
                self.opt.step([dh.T @ X, dh.sum(0), gw2, gb2])


# ----------------------------------------------------------------------------------------------- training loop
def _valid_plans(world, t):
    rows = []
    for c in itertools.product(range(N_OPT), repeat=world.K):
        sk = world.skills(t, np.array(c))
        if sk is not None:
            rows.append((np.array(c), sk))
    return rows


def train_ext(world, method, lr, seed, n_updates=3000, G=8, eval_every=50, prior=(1.0, 1.0), model_lr=0.05,
              kappa=2.0, ppo_epochs=4, clip=0.2):
    rng = np.random.default_rng(seed)
    exec_rng = np.random.default_rng(seed + 10_000)
    aux_rng = np.random.default_rng(seed + 20_000)        # predictor minibatches / Thompson samples
    T, K, S = len(world.train), world.K, len(world.lib.p)
    L = np.zeros((T, K, N_OPT))
    opt = np.mean([world.optimum(t) for t in world.train])
    ctx = ContextPosterior(S, *prior, kappa=kappa) if method in ("RBER-LOO", "RBER-C", "TS-plan") else None
    lsp = SuccessPredictor(method[4:], S, K, model_lr, aux_rng) if method.startswith("LSP") else None
    base_method = {"EXEC-RLOO": "EXEC", "EXEC-PPO": "EXEC"}.get(method, method)
    post = SkillPosterior(S, T if method == "RBER-L" else 1, *prior) if base_method.startswith("RBER") and ctx is None else None
    vp = [_valid_plans(world, t) for t in world.train] if method == "TS-plan" else None

    def evaluate():
        if method == "TS-plan":
            m = ctx.a / (ctx.a + ctx.b)
            return float(np.mean([world.success(t, max(vp[i], key=lambda r: np.prod(m[r[1]]))[0])
                                  for i, t in enumerate(world.train)])) / opt
        return greedy_value(world, L) / opt

    curve = [(0, 0, evaluate())]; execs = 0
    for u in range(1, n_updates + 1):
        i = int(rng.integers(T)); t = world.train[i]
        P = softmax(L[i])
        u01 = rng.random((G, K, 1))
        ch = np.minimum((u01 > np.cumsum(P, -1)[None]).sum(-1), N_OPT - 1)
        if method == "TS-plan":
            for _ in range(G):
                th = aux_rng.beta(ctx.a, ctx.b)
                c, sk = max(vp[i], key=lambda r: np.prod(th[r[1]]))
                ok, out = world.execute(sk, exec_rng); execs += 1
                ctx.add(sk[:len(out)], [int(z) for z in out])
        else:
            if method in ("RBER-LOO", "RBER-C"):
                r, n = loo_batch(method, world, t, ch, exec_rng, ctx)
            elif lsp is not None:
                plans = [world.skills(t, c) for c in ch]
                r = np.zeros(G); n = 0
                valid = [j for j, sk in enumerate(plans) if sk is not None]
                if valid:
                    r[valid] = lsp.predict([plans[j] for j in valid])     # lagged: model before this batch
                for j in valid:
                    ok, out = world.execute(plans[j], exec_rng); n += 1; lsp.add(plans[j], ok)
                lsp.fit()
            else:
                r, n = batch_rewards(base_method, world, t, ch, exec_rng, post, group=i if method == "RBER-L" else 0)
            execs += n
            onehot = np.eye(N_OPT)[ch]
            if method == "EXEC-RLOO":
                A = r - (r.sum() - r) / (G - 1)
                L[i] += lr * (A[:, None, None] * (onehot - P[None])).mean(0)
            elif method == "EXEC-PPO":
                A = group_advantage(r); lp_old = np.log(P[np.arange(K), ch]).sum(1)
                for _ in range(ppo_epochs):
                    Pn = softmax(L[i]); ratio = np.exp(np.log(Pn[np.arange(K), ch]).sum(1) - lp_old)
                    act = ~(((A > 0) & (ratio > 1 + clip)) | ((A < 0) & (ratio < 1 - clip)))
                    L[i] += lr * ((act * A * ratio)[:, None, None] * (onehot - Pn[None])).mean(0)
            else:
                A = group_advantage(r)
                L[i] += lr * (A[:, None, None] * (onehot - P[None])).mean(0)
        if u % eval_every == 0:
            curve.append((u, execs, evaluate()))
    return dict(method=method, lr=lr, seed=seed, curve=curve, optimum=opt, final=curve[-1][2], held=None)
