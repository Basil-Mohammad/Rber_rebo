"""Measure success probabilities of 54 Meta-World v3 skills (18 scripted experts x 3 perturbation levels).

Each skill = the scripted expert policy of a Meta-World v3 task with a random, episode-level bias added to its
actions (2 phases x [xyz bias, gripper bias]) drawn from N(0, sigma^2), sigma in {0.3, 0.8, 1.5}.
Success = the environment's own success predicate at the last of 200 steps. 100 rollouts per skill,
deterministic seeds (independent of Python's hash randomization). Output: data/skills_metaworld.json
with counts and 95% Wilson intervals.

Usage:  MUJOCO_GL=egl python scripts/measure_skills.py [--rollouts 100] [--workers 2]
"""
import argparse, json, os, sys, time, warnings, zlib
os.environ.setdefault("MUJOCO_GL", "egl"); warnings.filterwarnings("ignore")
from multiprocessing import Pool
import numpy as np

TASKS = ["bin-picking-v3", "assembly-v3", "window-open-v3", "button-press-v3", "hammer-v3", "soccer-v3",
         "coffee-push-v3", "sweep-into-v3", "drawer-open-v3", "reach-v3", "push-v3", "pick-place-v3",
         "plate-slide-v3", "faucet-open-v3", "handle-press-v3", "lever-pull-v3", "drawer-close-v3", "door-close-v3"]
LEVELS = [0.3, 0.8, 1.5]
HORIZON = 200


def wilson(k, n, z=1.959964):
    if n == 0:
        return 0.0, 1.0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def job(args):
    name, sigma, n = args
    import metaworld, metaworld.policies as P
    cls = "Sawyer" + "".join(w.capitalize() for w in name.replace("-v3", "").split("-")) + "V3Policy"
    pol = getattr(P, cls)()
    mt = metaworld.MT1(name, seed=0); env = mt.train_classes[name]()
    rng = np.random.default_rng(zlib.crc32(f"{name}|{sigma}".encode()))
    out = []
    for i in range(n):
        env.set_task(mt.train_tasks[i % len(mt.train_tasks)])
        obs, _ = env.reset(seed=i); th = rng.normal(0, sigma, 8); info = {}
        for t in range(HORIZON):
            a = pol.get_action(obs).copy(); ph = 0 if t < HORIZON // 2 else 1
            a[:3] += 0.3 * th[4 * ph:4 * ph + 3]; a[3] += 0.8 * th[4 * ph + 3]
            obs, _, _, _, info = env.step(np.clip(a, -1, 1))
        out.append(int(info["success"]))
    return name, sigma, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--rollouts", type=int, default=100); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(); t0 = time.time()
    with Pool(a.workers) as p:
        res = p.map(job, [(t, s, a.rollouts) for t in TASKS for s in LEVELS])
    skills = []
    for name, sigma, o in res:
        k, n = int(sum(o)), len(o); lo, hi = wilson(k, n)
        skills.append(dict(task=name, sigma=sigma, successes=k, rollouts=n, p=k / n, ci95=[lo, hi], outcomes=o))
        print(f"{name:18s} sigma={sigma:<4} p={k / n:.2f}  [{lo:.2f}, {hi:.2f}]", flush=True)
    import metaworld, mujoco
    meta = dict(metaworld_version=getattr(metaworld, "__version__", "3.x"), mujoco_version=mujoco.__version__,
                horizon=HORIZON, levels=LEVELS, rollouts=a.rollouts, seconds=round(time.time() - t0))
    os.makedirs("data", exist_ok=True)
    json.dump(dict(meta=meta, skills=skills), open("data/skills_metaworld.json", "w"), indent=1)
    print("done in", meta["seconds"], "s")
