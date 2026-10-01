"""Builds the self-contained Kaggle notebooks for the language-model experiments (notebooks/kaggle_llm_*.ipynb).

Each notebook embeds the exact source of rber/domain.py, rber/rewards.py, rber/llm.py and the measured skill
file, so it runs without cloning the repository. Sessions split the 10 seeds so that each fits one Kaggle
GPU session (< 6 h on a T4); seeds are the outer loop so a partial session still yields balanced data.
"""
import json, os

ROOT = os.path.join(os.path.dirname(__file__), "..")
SRC = {f: open(os.path.join(ROOT, "src", "rber", f)).read()
       for f in ("__init__.py", "domain.py", "rewards.py", "tabular.py", "extension.py", "llm.py")}
SKILLS = open(os.path.join(ROOT, "data", "skills_metaworld.json")).read()
BASE_METHODS = ["VERIF", "STEP", "EXEC", "RBER", "RBER-NL"]
# study 1 (done): A-C. Extension study (docs/PREREGISTRATION_extension.md, H12-H13): D-F.
SESSIONS = {"A": ([0, 1, 2, 3], BASE_METHODS), "B": ([4, 5, 6], BASE_METHODS), "C": ([7, 8, 9], BASE_METHODS),
            "D": (list(range(10)), ["RBER-LOO"]),
            "E": ([10, 11, 12, 13, 14], ["EXEC", "RBER", "RBER-LOO"]),
            "F": ([15, 16, 17, 18, 19], ["EXEC", "RBER", "RBER-LOO"])}
MINUTES = {"VERIF": 16, "STEP": 36, "EXEC": 24, "RBER": 16, "RBER-NL": 15, "RBER-LOO": 16}


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def notebook(name, seeds, methods):
    cells = [
        md(f"""# RBER — language-model planner experiments, session {name} (seeds {seeds})

**Settings:** Accelerator → **GPU T4** (x1 is enough), Internet → **On**. Then *Save Version → Save & Run All*.
Expected time: about {len(seeds) * sum(MINUTES[m] for m in methods) / 60:.1f} h. Output: **`llm_results_session{name}.zip`** (Output tab).

What runs: Qwen2.5-0.5B-Instruct planners trained with {", ".join(methods)} rewards
(250 updates × 4 tasks × 8 plans, LR 3e-6), seeds {seeds}. Protocol: `docs/PREREGISTRATION_final.md` in the repository.
The run is resumable: finished runs are skipped if the notebook is restarted in the same session."""),
        code("""!pip -q install -U accelerate "transformers>=4.51"
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
        code("import os\nos.makedirs('rber', exist_ok=True)"),
    ]
    for f, s in SRC.items():
        assert s.strip(), f"{f} is empty: %%writefile needs a non-empty cell body"
        cells.append(code(f"%%writefile rber/{f}\n" + s))
    cells.append(code("%%writefile skills_metaworld.json\n" + SKILLS))
    cells.append(code(f"""import os, json, time, traceback, shutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np, torch
from rber.domain import load_library, build_world
from rber import llm

SEEDS = {seeds}
METHODS = {methods}
OUT = "/kaggle/working/llm_results_session{name}" if os.path.exists("/kaggle") else "llm_results_session{name}"
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "log.txt"), "a")
def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\\n"); LOG.flush()

lib = load_library("skills_metaworld.json")
W = build_world(lib, K=3, n_train=24, n_held=12, seed=0)
log("=== session {name}", time.ctime(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
log("skills:", len(lib.p), "| optimum train/held:", np.mean([W.optimum(t) for t in W.train]), np.mean([W.optimum(t) for t in W.held]))"""))
    cells.append(code(f"""for s in SEEDS:
    for m in METHODS:
        try:
            llm.train(W, m, s, OUT, lr=3e-6, n_updates=250, tasks_per_update=4, G=8, eval_every=25, log=log)
        except Exception:
            log("!! run", m, s, "crashed (continuing):\\n" + traceback.format_exc())
            import gc; gc.collect(); torch.cuda.empty_cache()
        log(f"GPU memory after run: {{torch.cuda.memory_allocated() / 2**30:.2f}} GiB")
zp = shutil.make_archive(OUT, "zip", OUT)
log("Download:", zp)"""))
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}, "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
    for name, (seeds, methods) in SESSIONS.items():
        p = os.path.join(ROOT, "notebooks", f"kaggle_llm_session{name}.ipynb")
        json.dump(notebook(name, seeds, methods), open(p, "w"), indent=1, ensure_ascii=False)
        print("wrote", p)
