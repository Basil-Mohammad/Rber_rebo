# Pilot history

This project ran pilot studies before it committed to a research direction. Each pilot was
pre-registered before any run. The original pre-registrations, with every amendment, are in `docs/pilots/`.
Negative and inconclusive outcomes are reported here as well as positive ones.

| Stage | Question | Outcome | Consequence |
|---|---|---|---|
| Pilot 1 (toy gridworld, CPU) | Does RL exploit a learned success verifier? Can selected audits and a certificate fix this with few labels? | The problem is real: the policy exploits a verifier that looks accurate. "Smart" audit selection (H2) and sub-linear audit scaling (H3) were **not** supported. | The audit-selection direction was reframed. |
| Pilot 2 (Meta-World + Qwen3-VL-4B, GPU) | Does a real VLM success detector get exploited? Does an embodied probe (act, then look again) separate false positives? | **Inconclusive** by the pre-registered rule: the VLM was biased towards "No", so tasks failed the threshold-based credibility gate. A post-hoc analysis (not pre-registered) showed exploitation. | This led to pilot 3. |
| Pilot 3 (existing pilot-2 data, CPU) | Can a verifier's precision on the optimized tail be estimated without labels, from agreement between views? | **Failed**: F1 did not hold, e.g. mean absolute error 0.58 on assembly-v3 against a threshold of 0.15. | The direction was dropped. A systematic gap search followed. |
| Pilot 4, stage 1 (tabular planner over measured Meta-World skills, CPU, 10 seeds) | Execution-grounded RL for a planner over stochastic skills: does a Rao-Blackwellized reward help? | K1–K3 passed. The Rao-Blackwellized reward (non-lagged) reached 90% of the optimum about 3.5× faster than the best tuned baseline. It stayed ahead when its independence assumption was violated. | Go to stage 2. |
| Pilot 4, stage 2, run 1 (Qwen2.5-0.5B, free-text answers) | Does the result survive with a real LM planner? | **Inconclusive** (S0 failed): the answer format collapsed under GRPO. | Amendment 1: constrained letter actions. |
| Pilot 4, stage 2, run 2 (letter actions) | Same | **Inconclusive** (S0 failed): gradients moved a letter prior that carries no meaning. | Amendment 2: actions are skill names scored by length-normalized likelihood. Amendment 3 made defensive changes after an external code review. Amendment 4 fixed a GPU-memory leak. |
| Pilot 4, stage 2, run 3 (skill-name scoring, 3 seeds) | Same | **GO**: S0 and the primary criterion S2 held; S1 and S3 did not (details below). | The final, pre-registered study (`docs/PREREGISTRATION_final.md`). |

**Stage 2, run 3 in detail** (3 seeds for EXEC and RB, 1 for VERIF; train optimum 0.191, held-out optimum 0.243):

| Method | Final train success | Final held-out success |
|---|---|---|
| RB | 0.187 | 0.182 |
| EXEC | 0.153 | 0.170 |
| VERIF | 0.056 | 0.009 |

- **S2 (primary) held:** RB reached 0.178 at half of EXEC's executions, versus EXEC's final 0.153.
- **S1 failed:** VERIF was not 0.15 below the optimum.
- **S3 failed:** the held-out gain was below +0.05.
- **Variant:** RB in this pilot was the non-lagged variant. It is RBER-NL in the paper.

## What changed between the pilots and the final study
- **Lag:** the reward now uses the posterior before the current batch, which is the RBER in the paper. The non-lagged pilot version is kept as the RBER-NL ablation.
- **Seeds and tuning:** 20 evaluation seeds on the CPU. Learning rates are tuned per method on separate tuning seeds, over {0.3, 1, 3, 10}.
- **Skill measurements:** 100 rollouts per skill instead of 20, with Wilson intervals.
- **Language-model study:** 10 seeds per method, five methods including RBER-NL, and LR 3e-6. The LR was chosen by the VERIF-only rule of stage 2.
