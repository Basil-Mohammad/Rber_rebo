# Pre-registration of the final experiments (written 2026-09-30, before any final run)

The pilot history (four stages, all amendments) is in `docs/PILOT_HISTORY.md`. This document fixes the
hypotheses, protocol and analysis of the experiments reported in the paper.

## Method under test
RBER: policy-gradient training of a planner with the Rao-Blackwellized execution reward
R_hat(a) = prod_k mean_posterior(p_{s_k}), where the Beta(1,1) skill posteriors are shared across tasks,
updated with censored step outcomes of every executed plan, and *lagged* (the reward of a batch uses the
posterior before that batch). Baselines: VERIF (offline verification against a reliability-blind reference),
STEP (fraction of steps completed), EXEC (binary success). References/ablations: RB* (true R(a)), RBER-L (no
sharing), RBER-NL (non-lagged).

## Protocol (CPU, tabular planner)
- Skills: 54 Meta-World v3 skills, success probabilities from 100 rollouts each (`data/skills_metaworld.json`).
- Worlds: 24 training tasks; step categories distinct within a task; 3 valid + 1 invalid option per step.
  Each seed draws a new world (tasks) and new sampling/execution streams.
- Optimizer: REINFORCE with group-normalized baseline, G = 8 plans per update, 3000 updates.
- Learning rates: tuned per method on 5 tuning seeds (1000-1004, disjoint from evaluation seeds) over
  {0.3, 1, 3} by mean normalized area under the learning curve, separately for the independent and the
  interaction condition at K = 3; the tuned values are then fixed for every experiment.
- Evaluation: 20 seeds (0-19). Metric: normalized success = true success of the greedy plan / optimum,
  averaged over tasks. Reported: final value, AUC over updates, executions to reach 0.9 (censored at the budget).

## Hypotheses
- H1 (efficiency): RBER needs fewer executions than EXEC and than STEP to reach 0.9 normalized success
  (K in {3, 4}, independent and interaction conditions). Test: paired Wilcoxon signed-rank, Holm correction
  over all RBER-vs-baseline comparisons, alpha = 0.05.
- H2 (final performance): RBER's final normalized success is not lower than EXEC's by more than 0.02 in any
  main condition (mean difference and 95% bootstrap CI).
- H3 (theory): the exact variance identity of Theorem 1 holds to numerical precision; the ratio
  Tr Cov[EXEC] / Tr Cov[RB*] at the uniform policy grows with the horizon K.
- H4 (misspecification): RBER's AUC advantage over EXEC remains positive for interaction strength <= 0.3.
- H5 (ablations): sharing the posterior across tasks increases AUC (RBER vs RBER-L); lagging does not
  reduce AUC by more than 0.02 (RBER vs RBER-NL).

## Protocol (GPU, language-model planner; run on Kaggle)
Qwen2.5-0.5B-Instruct, skills chosen by name (length-normalized log-likelihood), full fine-tuning, AdamW,
LR 3e-6 (selected in the pilot by a VERIF-only rule), 250 updates x 4 tasks x 8 plans, 10 seeds per method
(VERIF, STEP, EXEC, RBER). K = 3, 24 training + 12 held-out tasks.
- H6: RBER's train success at half of EXEC's execution budget >= EXEC's final (primary, as in the pilot);
  paired comparison of AUC over seeds (Wilcoxon, Holm).
- H7 (secondary): held-out success RBER vs EXEC (reported with CI; no directional claim pre-committed).

Any deviation from this plan will be reported in the paper.

## Amendment (2026-09-30, after tuning, before any evaluation run)
The first tuning pass selected lr = 3 for EXEC and RB* in the independent condition, the edge of the grid
{0.3, 1, 3}. To avoid under-tuning a baseline, the grid was extended to {0.3, 1, 3, 10} for *all* methods and
selection was repeated with the same rule on the same tuning seeds. No evaluation seed had been run.
