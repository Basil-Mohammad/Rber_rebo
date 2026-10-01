"""All CPU experiments of the paper (tabular planner + exact theory checks). Resumable; results in results/cpu/.

    python scripts/run_cpu.py tune          # learning-rate selection on tuning seeds
    python scripts/run_cpu.py main          # E1: main comparison (K in {3,4}, independent / interaction)
    python scripts/run_cpu.py horizon       # E2: horizon sweep K = 2..6
    python scripts/run_cpu.py variance      # E3: exact gradient-variance computations (Theorem 1)
    python scripts/run_cpu.py interaction   # E4: interaction-strength sweep (misspecification)
    python scripts/run_cpu.py ablations     # E5: prior strength, group size, lag
    python scripts/run_cpu.py all
"""
import json, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rber.domain import build_world, load_library          # noqa: E402
from rber.tabular import train, softmax                     # noqa: E402
from rber import theory                                      # noqa: E402
from rber.extension import EXT_METHODS, train_ext            # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "cpu")
LIB = load_library()
EVAL_SEEDS = list(range(20))
TUNE_SEEDS = list(range(1000, 1005))
LRS = [0.3, 1.0, 3.0, 10.0]
N_UPD, G, EVAL_EVERY = 3000, 8, 50
MAIN_METHODS = ["VERIF", "STEP", "EXEC", "RBER", "RBER-L", "RB*"]
WORKERS = int(os.environ.get("WORKERS", "2"))


def auc(curve):
    return float(np.mean([v for _, _, v in curve]))


def execs_to(curve, thr=0.9):
    for u, e, v in curve:
        if v >= thr:
            return e, True
    return curve[-1][1], False


def run_one(cfg):
    w = build_world(LIB, K=cfg["K"], seed=cfg["seed"], interaction=cfg.get("inter", 0.0))
    if cfg["method"] in EXT_METHODS:
        r = train_ext(w, cfg["method"], cfg["lr"], seed=cfg["seed"], n_updates=cfg.get("U", N_UPD), G=cfg.get("G", G),
                      eval_every=EVAL_EVERY, model_lr=cfg.get("mlr", 0.05), kappa=cfg.get("kappa", 2.0))
    else:
        r = train(w, cfg["method"], cfg["lr"], seed=cfg["seed"], n_updates=cfg.get("U", N_UPD), G=cfg.get("G", G),
                  eval_every=EVAL_EVERY, prior=tuple(cfg.get("prior", (1.0, 1.0))))
    e, hit = execs_to(r["curve"])
    return dict(cfg, final=r["final"], auc=auc(r["curve"]), execs90=e, reached=hit, optimum=r["optimum"],
                curve=r["curve"], total_execs=r["curve"][-1][1])


def run_many(name, cfgs):
    path = os.path.join(OUT, f"{name}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            d = json.loads(line); done.add(json.dumps({k: d[k] for k in sorted(cfgs[0])}, sort_keys=True))
    todo = [c for c in cfgs if json.dumps({k: c[k] for k in sorted(cfgs[0])}, sort_keys=True) not in done]
    print(f"[{name}] {len(cfgs)} runs, {len(todo)} to do", flush=True)
    t0 = time.time()
    with Pool(WORKERS) as p, open(path, "a") as f:
        for i, res in enumerate(p.imap_unordered(run_one, todo), 1):
            f.write(json.dumps(res) + "\n"); f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    return [json.loads(l) for l in open(path)]


def tuned_lrs():
    path = os.path.join(OUT, "tuned_lrs.json")
    if os.path.exists(path):
        return json.load(open(path))
    raise SystemExit("run `tune` first")


def cmd_tune():
    cfgs = [dict(K=3, inter=inter, seed=s, method=m, lr=lr) for inter in (0.0, 0.3) for s in TUNE_SEEDS
            for m in MAIN_METHODS + ["RBER-NL"] for lr in LRS]
    rows = run_many("tune", cfgs)
    best = {}
    for inter in (0.0, 0.3):
        for m in MAIN_METHODS + ["RBER-NL"]:
            sc = {lr: np.mean([r["auc"] for r in rows if r["inter"] == inter and r["method"] == m and r["lr"] == lr])
                  for lr in LRS}
            best[f"{m}|{inter}"] = max(sc, key=sc.get)
            print(f"  inter={inter} {m:8s} " + "  ".join(f"lr={k}: {v:.3f}" for k, v in sc.items()), "->", best[f"{m}|{inter}"])
    json.dump(best, open(os.path.join(OUT, "tuned_lrs.json"), "w"), indent=1)


def lr_of(best, m, inter):
    return best[f"{m}|{0.3 if inter > 0 else 0.0}"]


def cmd_main():
    best = tuned_lrs()
    cfgs = [dict(K=K, inter=inter, seed=s, method=m, lr=lr_of(best, m, inter))
            for K in (3, 4) for inter in (0.0, 0.3) for s in EVAL_SEEDS for m in MAIN_METHODS]
    run_many("main", cfgs)


def cmd_horizon():
    best = tuned_lrs()
    cfgs = [dict(K=K, inter=0.0, seed=s, method=m, lr=lr_of(best, m, 0.0))
            for K in (2, 3, 4, 5, 6) for s in EVAL_SEEDS for m in ["STEP", "EXEC", "RBER", "RB*"]]
    run_many("horizon", cfgs)


def cmd_interaction():
    best = tuned_lrs()
    cfgs = [dict(K=3, inter=d, seed=s, method=m, lr=lr_of(best, m, d))
            for d in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5) for s in EVAL_SEEDS for m in ["EXEC", "RBER", "RB*"]]
    rows = run_many("interaction", cfgs)
    # interaction gap eps_int = max over plans |R(a) - prod_k p_{s_k}| (base probabilities), per world
    gaps = {}
    for d in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        vals = []
        for s in EVAL_SEEDS:
            w = build_world(LIB, K=3, seed=s, interaction=d); w0 = build_world(LIB, K=3, seed=s, interaction=0.0)
            import itertools
            e = max(abs(w.success(t, np.array(c)) - w0.success(t0, np.array(c)))
                    for t, t0 in zip(w.train, w0.train) for c in itertools.product(range(4), repeat=3))
            vals.append(e)
        gaps[str(d)] = vals
    json.dump(gaps, open(os.path.join(OUT, "interaction_gaps.json"), "w"))


def cmd_ablations():
    best = tuned_lrs()
    cfgs = []
    for inter in (0.0, 0.3):
        for s in EVAL_SEEDS:
            for pr in ((0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)):
                cfgs.append(dict(K=3, inter=inter, seed=s, method="RBER", lr=lr_of(best, "RBER", inter), prior=list(pr), G=8, U=N_UPD))
            cfgs.append(dict(K=3, inter=inter, seed=s, method="RBER-NL", lr=lr_of(best, "RBER-NL", inter), prior=[1.0, 1.0], G=8, U=N_UPD))
            for g in (4, 16):  # same sample budget: U * G = 24000 plans
                for m in ("EXEC", "RBER"):
                    cfgs.append(dict(K=3, inter=inter, seed=s, method=m, lr=lr_of(best, m, inter), prior=[1.0, 1.0], G=g, U=N_UPD * 8 // g))
    run_many("ablations", cfgs)


def _var_job(args):
    K, s = args
    w = build_world(LIB, K=K, seed=s)
    rng = np.random.default_rng(s)
    out = []
    for ti, t in enumerate(w.train[:8]):
        for tag, P in (("uniform", np.full((K, 4), 0.25)), ("random", softmax(rng.normal(0, 1.5, (K, 4))))):
            c = theory.trace_cov(w, t, P)
            m = theory.second_moments(w, t, P)
            out.append(dict(K=K, seed=s, task=ti, policy=tag, exec=c["exec"], rb=c["rb"], gap=c["gap"],
                            identity_err=abs(c["exec"] - c["rb"] - c["gap"]), J=m["J"], Rmax=m["Rmax"],
                            m2_exec0=theory.second_moments(w, t, P, 0.0)["exec"], m2_rb0=theory.second_moments(w, t, P, 0.0)["rb"]))
    return out


def cmd_variance():
    rows = []
    with Pool(WORKERS) as p:
        for r in p.imap_unordered(_var_job, [(K, s) for K in (2, 3, 4, 5, 6) for s in EVAL_SEEDS]):
            rows += r
    json.dump(rows, open(os.path.join(OUT, "variance.json"), "w"))
    print("max identity error:", max(r["identity_err"] for r in rows))


# ================================================================ extension study (docs/PREREGISTRATION_extension.md)
EXT_POLICY = ["RBER-LOO", "RBER-C", "EXEC-RLOO", "EXEC-PPO"]
EXT_LSP = ["LSP-lin", "LSP-mlp"]
MLRS = [0.003, 0.01, 0.05, 0.2]      # 0.003 and policy lr 0.1 added by amendment (edge of grid), before evaluation
LRS_LSP = [0.1] + LRS


def cmd_ext_tune():
    cfgs = [dict(K=3, inter=inter, seed=s, method=m, lr=lr, mlr=0.05) for inter in (0.0, 0.3) for s in TUNE_SEEDS
            for m in EXT_POLICY for lr in LRS]
    cfgs += [dict(K=3, inter=inter, seed=s, method=m, lr=lr, mlr=mlr) for inter in (0.0, 0.3) for s in TUNE_SEEDS
             for m in EXT_LSP for lr in LRS_LSP for mlr in MLRS]
    rows = run_many("ext_tune", cfgs)
    best = {}
    for inter in (0.0, 0.3):
        for m in EXT_POLICY + EXT_LSP:
            grid = [(lr, mlr) for lr in (LRS_LSP if m in EXT_LSP else LRS) for mlr in (MLRS if m in EXT_LSP else [0.05])]
            sc = {g: np.mean([r["auc"] for r in rows if r["inter"] == inter and r["method"] == m and r["lr"] == g[0]
                              and r["mlr"] == g[1]]) for g in grid}
            b = max(sc, key=sc.get); best[f"{m}|{inter}"] = dict(lr=b[0], mlr=b[1])
            print(f"  inter={inter} {m:9s} best lr={b[0]} mlr={b[1]} auc={sc[b]:.3f}", flush=True)
    json.dump(best, open(os.path.join(OUT, "ext_tuned.json"), "w"), indent=1)


def ext_cfg(m, K, inter, seed, **kw):
    if m == "TS-plan":
        return dict(K=K, inter=inter, seed=seed, method=m, lr=0.0, mlr=0.0, **kw)
    b = json.load(open(os.path.join(OUT, "ext_tuned.json")))[f"{m}|{0.3 if inter > 0 else 0.0}"]
    return dict(K=K, inter=inter, seed=seed, method=m, lr=b["lr"], mlr=b["mlr"], **kw)


def cmd_ext_main():
    run_many("ext_main", [ext_cfg(m, K, inter, s) for K in (3, 4) for inter in (0.0, 0.3) for s in EVAL_SEEDS
                          for m in EXT_POLICY + EXT_LSP + ["TS-plan"]])


def cmd_ext_interaction():
    run_many("ext_interaction", [ext_cfg(m, 3, d, s) for d in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5) for s in EVAL_SEEDS
                                 for m in ("RBER-LOO", "RBER-C")])


def cmd_ext_horizon():
    run_many("ext_horizon", [ext_cfg("RBER-LOO", K, 0.0, s) for K in (2, 3, 4, 5, 6) for s in EVAL_SEEDS])


def cmd_ext_kappa():
    run_many("ext_kappa", [ext_cfg("RBER-C", 3, inter, s, kappa=k) for inter in (0.0, 0.3) for s in EVAL_SEEDS
                           for k in (0.5, 8.0)])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmds = dict(tune=cmd_tune, main=cmd_main, horizon=cmd_horizon, variance=cmd_variance,
                interaction=cmd_interaction, ablations=cmd_ablations, ext_tune=cmd_ext_tune, ext_main=cmd_ext_main,
                ext_interaction=cmd_ext_interaction, ext_horizon=cmd_ext_horizon, ext_kappa=cmd_ext_kappa)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k in (cmds if which == "all" else [which]):
        t0 = time.time(); print(f"=== {k}", flush=True); cmds[k](); print(f"=== {k} done in {time.time() - t0:.0f}s", flush=True)
