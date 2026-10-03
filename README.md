# Rao–Blackwellized Execution Rewards (RBER)

Code, data and experiment logs for **Rao–Blackwellized Execution Rewards (RBER)**: variance-reduced reinforcement learning of agentic language-model planners for robots.

Agentic planners that sequence robot skills are often trained with the binary success of executed plans. On long plans this reward is very noisy. RBER replaces the binary outcome of each sampled plan with its posterior-predictive success, ∏ₖ m_{sₖ}. The posterior is a Beta distribution over the reliability of every skill:
- it is learned from the censored step outcomes of all executions;
- it is shared by all tasks;
- it is lagged by one batch.

RBER needs no extra executions, no critic and no annotation.

## Main results (tabular planner, 20 seeds per cell)

| Method | K=3 AUC | K=3 E₀.₉ | K=4 AUC | K=4 E₀.₉ |
|---|---|---|---|---|
| VERIF (no execution) | 0.217 | n/a | 0.120 | n/a |
| STEP | 0.792 | 16.2k | 0.652 | not reached |
| EXEC (binary success) | 0.875 | 5.0k | 0.745 | 16.0k |
| **RBER (ours)** | **0.951** | **1.7k** | **0.926** | **2.2k** |
| RB* (oracle, true R) | 0.964 | 1.5k | 0.936 | 2.3k |

The table shows the independent-skill condition. AUC is the interquartile mean. E₀.₉ is the median number of robot executions needed to reach 90% of the optimal success. Running `scripts/make_report.py` generates the full tables, including the skill-interaction condition.

- **Statistics:** RBER beats STEP and EXEC in every condition (paired Wilcoxon with Holm correction, p ≤ 0.002).
- **Theory check:** the variance identity of Theorem 1 holds to 1.1e-16.
- **Limitation:** with strong skill interactions (δ ≥ 0.4), RBER's final performance falls below EXEC. The theory predicts this bias floor.

## Language-model planner (Kaggle, 10 seeds, Qwen2.5-0.5B-Instruct)

| Method | AUC (train) | Final train | Final held-out | Executions |
|---|---|---|---|---|
| VERIF | 0.195 | 0.196 | 0.080 | 0 |
| STEP | 0.678 | 0.768 | 0.521 | 7329 |
| EXEC | 0.676 | 0.789 | 0.418 | 6663 |
| **RBER (ours)** | **0.774** | **0.905** | 0.500 | 7694 |
| RBER-NL (ablation) | 0.802 | 0.917 | 0.562 | 7735 |

All values are normalized to the optimum; the table shows means, and `make_report.py` adds 95% confidence intervals.

- **Pre-registered primary hypothesis H6: supported.** At half of EXEC's executions, RBER reaches 0.823, against EXEC's final 0.789.
- **AUC versus EXEC:** +0.098, with RBER ahead in 8 of 10 seeds. The uncorrected p is 0.027; after Holm correction it is 0.137, so not significant.
- **Held-out success (H7):** +0.082 versus EXEC, not significant.
- **Raw results:** `results/llm/{A,B,C}`.

## Extension study (pre-registered in `docs/PREREGISTRATION_extension.md`; 20 seeds)

**New methods:**
- **RBER-LOO:** leave-one-out credit. Each plan's reward uses the outcomes of all other plans in the batch, but never its own.
- **RBER-C:** contextual credit. A first-order skill model with shrinkage, which removes the bias under skill interactions.

**New baselines:** EXEC-RLOO, EXEC-PPO (critic-free clipped multi-epoch), LSP-lin / LSP-mlp (learned success predictors fitted to final outcomes), and TS-plan (a model-based planning reference, not a policy).

| Hypothesis | Outcome |
|---|---|
| H8: RBER-LOO > RBER | not supported (indistinguishable on CPU) |
| H9: RBER-C removes the interaction bias floor | supported (final success above EXEC at every δ ≤ 0.5) |
| H10: RBER-LOO > all four stronger baselines | supported (16/16 tests, Holm p < 1e-4) |
| H11: RBER-C costs ≤ 0.01 AUC without interactions | not supported (cost 0.015–0.020) |
| H12: LLM, 20 seeds, RBER-LOO AUC > EXEC | **supported**: +0.172 [+0.113, +0.231], p = 0.0001, ahead in 16/20 seeds |
| H13: LLM held-out success | RBER-LOO +0.087 (p = 0.08), RBER +0.025 (p = 0.43): not significant |

```bash
python scripts/run_cpu.py ext_tune && python scripts/run_cpu.py ext_main      # also: ext_interaction, ext_horizon, ext_kappa
python tests/test_extension.py
```

## Repository layout

```
src/rber/            library
  domain.py          skills, tasks, execution model, skill interactions
  rewards.py         VERIF / STEP / EXEC / RBER / RBER-L / RBER-NL / RB* rewards, Beta posterior
  tabular.py         tabular softmax planner + group-normalized policy gradient
  llm.py             Qwen2.5-0.5B-Instruct planner (skill-name scoring) + training loop
  theory.py          exact enumeration of gradient moments (Theorems 1-2)
  extension.py       RBER-LOO, RBER-C, EXEC-RLOO, EXEC-PPO, LSP-lin/mlp, TS-plan
  stats.py           IQM, bootstrap CIs, probability of improvement, Wilcoxon, Holm
scripts/
  measure_skills.py  measures the 54 Meta-World v3 skills (100 rollouts each)      [executed]
  run_cpu.py         all CPU experiments: tune, main, horizon, variance, interaction, ablations [executed]
  make_report.py     every figure and table from results/            [executed]
  build_kaggle_notebooks.py  builds the self-contained Kaggle notebooks
data/skills_metaworld.json   measured skill success rates (+ raw outcomes, Wilson CIs)
results/cpu/                 raw logs of every CPU run (JSONL) and run logs
results/llm/                 put the Kaggle outputs here (see below)
notebooks/                   kaggle_llm_session{A..F}.ipynb  (GPU; A-C original study, D-F extension)
docs/                        PREREGISTRATION_final.md, PILOT_HISTORY.md, pilots/ (all pilot pre-registrations)
tests/                       test_core.py (estimator/theory), test_llm_mechanics.py (tiny random LM)
```

## Reproducing

```bash
pip install -r requirements.txt
python tests/test_core.py                 # estimator and theory unit tests (seconds)
python tests/test_llm_mechanics.py        # LM planner mechanics with a tiny random model (CPU, ~2 min)

# optional: re-measure the skills (about 7 min on 2 cores; data/ already contains the result)
MUJOCO_GL=egl python scripts/measure_skills.py --rollouts 100 --workers 2

python scripts/run_cpu.py all             # all CPU experiments (about 20 min on 2 cores; resumable)
python scripts/make_report.py             # figures -> paper/figures (PDF + PNG), tables -> paper/tables (LaTeX)
```

All figures and tables are generated from the raw logs in `results/`.

## Language-model experiments (Kaggle GPU)

The three notebooks in `notebooks/` are self-contained: each one embeds the exact source code and skill data. Each notebook trains Qwen2.5-0.5B-Instruct planners with VERIF, STEP, EXEC, RBER and RBER-NL for its seeds:

| Session | Seeds | Expected time |
|---|---|---|
| A | 0–3 | ~6 h |
| B | 4–6 | ~4.5 h |
| C | 7–9 | ~4.5 h |

To run them:
1. On kaggle.com, go to *Create → New Notebook → File → Import Notebook* and upload `kaggle_llm_sessionA.ipynb`.
2. In the right panel, set *Accelerator* to **GPU T4 ×1** and turn *Internet* **On**. Internet access requires a phone-verified account.
3. Click **Save Version → Save & Run All (Commit)**. The run continues after you close the browser.
4. When it finishes, open the version's **Output** tab and download `llm_results_sessionA.zip`.
5. Repeat for sessions B and C. All three together fit in Kaggle's weekly GPU quota.
6. Unzip the three archives into `results/llm/`; sub-folders are fine. Then run:
   ```bash
   python scripts/make_report.py
   ```
   This generates the LLM figure and table.

The runs are resumable at the run level. If a session stops, re-running the same notebook in the same session skips the finished runs. A run that crashes is logged, and the loop continues.

## Protocol

- **Pre-registration:** hypotheses H1–H7, metrics and tests were fixed before any evaluation run (`docs/PREREGISTRATION_final.md`). The single amendment, a wider learning-rate grid, was made before evaluation. `docs/PILOT_HISTORY.md` summarizes every pilot, including negative and inconclusive ones.
- **Seeds:**
  - Evaluation uses seeds 0–19 (CPU) and 0–9 (GPU).
  - Tuning uses seeds 1000–1004, which are disjoint from the evaluation seeds.
  - Execution outcomes use a separate random stream.

## License

MIT (see `LICENSE`). Meta-World and MuJoCo are used under their own licenses.
