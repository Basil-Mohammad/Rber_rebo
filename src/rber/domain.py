"""Planning domain: skills with measured success probabilities, tasks built from semantic step categories.

A *skill* is a (Meta-World scripted expert, perturbation level) pair whose success probability was measured
in simulation (``data/skills_metaworld.json``). A *task* is a sequence of K semantic steps (e.g. "open the
container"); every step offers ``M`` valid skills of the matching category plus ``N_INVALID`` skills of other
categories. A *plan* chooses one option per step. Executing a plan runs its skills in order and stops at the
first failure; plan success is the product of per-step successes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

CATEGORIES = {
    "move the object to the target": ["push-v3", "pick-place-v3", "plate-slide-v3", "soccer-v3", "sweep-into-v3",
                                      "coffee-push-v3", "bin-picking-v3"],
    "open the container": ["drawer-open-v3", "window-open-v3", "faucet-open-v3"],
    "press the control": ["button-press-v3", "handle-press-v3", "lever-pull-v3"],
    "close the container": ["drawer-close-v3", "door-close-v3"],
    "use the tool on the part": ["assembly-v3", "hammer-v3"],
    "move the gripper to the marked point": ["reach-v3"],
}
MODES = ("alpha", "beta", "gamma")
N_VALID, N_INVALID = 3, 1
N_OPT = N_VALID + N_INVALID
INVALID = -1
DEFAULT_SKILLS = os.path.join(os.path.dirname(__file__), "..", "..", "data", "skills_metaworld.json")


@dataclass
class Library:
    names: list            # human-readable skill names, e.g. "push[beta]"
    p: np.ndarray          # measured success probability per skill
    category: np.ndarray   # category index per skill
    categories: list       # category names
    ci95: np.ndarray       # Wilson intervals of the measurement


def load_library(path: str = DEFAULT_SKILLS, seed: int = 0) -> Library:
    """Skills from the measurement file. The mode label (alpha/beta/gamma) is assigned by a fixed random
    permutation per base skill, so the name carries no information about the perturbation level."""
    data = json.load(open(path))
    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    by_task: dict = {}
    for s in data["skills"]:
        by_task.setdefault(s["task"], []).append(s)
    names, p, cat, ci = [], [], [], []
    for task, rows in by_task.items():
        rows = sorted(rows, key=lambda r: r["sigma"])
        modes = rng.permutation(MODES)
        c = next(i for i, (k, v) in enumerate(CATEGORIES.items()) if task in v)
        for m, r in zip(modes, rows):
            names.append(f"{task.replace('-v3', '')}[{m}]"); p.append(r["p"]); cat.append(c); ci.append(r["ci95"])
    return Library(names, np.array(p), np.array(cat), cats, np.array(ci))


@dataclass
class Task:
    id: int
    steps: list                 # category index per step
    opts: np.ndarray            # [K, N_OPT] skill index per option
    valid: np.ndarray           # [K, N_OPT] bool
    ref: np.ndarray             # [K] option index of the reliability-blind reference plan (VERIF)


@dataclass
class World:
    lib: Library
    K: int
    train: list
    held: list
    pen: np.ndarray = field(default=None)  # [S, S] multiplicative context factor on the 2nd skill of a pair

    # ---- true execution model -------------------------------------------------------------------------
    def skills(self, t: Task, choice) -> np.ndarray | None:
        """Skill indices of a plan, or None if any chosen option is invalid for its step."""
        c = np.asarray(choice)
        if not t.valid[np.arange(self.K), c].all():
            return None
        return t.opts[np.arange(self.K), c]

    def step_probs(self, sk: np.ndarray) -> np.ndarray:
        q = self.lib.p[sk].astype(float).copy()
        if self.pen is not None:
            q[1:] *= self.pen[sk[:-1], sk[1:]]
        return q

    def success(self, t: Task, choice) -> float:
        """True success probability R(a) of a plan (0 for invalid plans)."""
        sk = self.skills(t, choice)
        return 0.0 if sk is None else float(np.prod(self.step_probs(sk)))

    def optimum(self, t: Task) -> float:
        import itertools
        return max(self.success(t, c) for c in itertools.product(range(N_OPT), repeat=self.K))

    def execute(self, sk: np.ndarray, rng: np.random.Generator):
        """Run skills in order, stop at the first failure. Returns (success, outcomes of attempted steps)."""
        q = self.step_probs(sk)
        out = []
        for k in range(self.K):
            z = bool(rng.random() < q[k]); out.append(z)
            if not z:
                return False, out
        return True, out


def build_world(lib: Library, K: int = 3, n_train: int = 24, n_held: int = 12, seed: int = 0,
                interaction: float = 0.0, interaction_frac: float = 0.25) -> World:
    """Random tasks over distinct step categories. ``interaction`` > 0 multiplies the success of the second
    skill of a random ``interaction_frac`` of ordered skill pairs by (1 - interaction): the per-skill
    independence model used by RBER is then misspecified."""
    rng = np.random.default_rng(seed)
    C, S = len(lib.categories), len(lib.p)
    if K > C:
        raise ValueError(f"K={K} exceeds the number of categories ({C})")

    def make(i):
        steps = list(rng.choice(C, K, replace=False))
        opts = np.zeros((K, N_OPT), int); valid = np.zeros((K, N_OPT), bool)
        for k, c in enumerate(steps):
            good = rng.choice(np.flatnonzero(lib.category == c), N_VALID, replace=False)
            bad = rng.choice(np.flatnonzero(lib.category != c), N_INVALID, replace=False)
            perm = rng.permutation(N_OPT); o = np.r_[good, bad][perm]
            opts[k] = o; valid[k] = perm < N_VALID
        ref = np.array([rng.choice(np.flatnonzero(valid[k])) for k in range(K)])
        return Task(i, steps, opts, valid, ref)

    tasks = [make(i) for i in range(n_train + n_held)]
    pen = None
    if interaction > 0:
        pen = np.where(rng.random((S, S)) < interaction_frac, 1.0 - interaction, 1.0)
    return World(lib, K, tasks[:n_train], tasks[n_train:], pen)


def prompt_text(w: World, t: Task) -> str:
    """Natural-language task description shown to the language-model planner."""
    cats = w.lib.categories
    order = ["first"] + ["then"] * (w.K - 2) + ["finally"] if w.K > 1 else ["first"]
    head = "Robot task: " + ", ".join(f"{o} {cats[c]}" for o, c in zip(order, t.steps)) + "."
    lines = [head, "Choose one skill for each step from its options."]
    for k in range(w.K):
        opts = "  ".join(f"{'ABCD'[j]}) {w.lib.names[s]}" for j, s in enumerate(t.opts[k]))
        lines.append(f"Step {k + 1} ({cats[t.steps[k]]}): {opts}")
    return "\n".join(lines)
