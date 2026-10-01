# Pre-registration of the extension study (written 2026-10-01, before any extension run)

This document extends `PREREGISTRATION_final.md`. The results of the original study are known and unchanged:
- CPU study: H1–H5 were supported.
- Language-model study: H6 was supported. The AUC and held-out differences were not significant after Holm correction, and the non-lagged RBER-NL was numerically ahead of the lagged RBER.

The extension adds new methods and new baselines. All hypotheses below were fixed before any of these methods was run on any seed, including the tuning seeds.

## New methods
- **RBER-LOO (leave-one-out).** Each plan's reward uses the posterior built from the history plus the step outcomes of all *other* plans in the current batch, excluding the plan's own outcomes. Its reward therefore does not depend on its own outcome, which is the property the lag provides, but it uses the information the non-lagged variant uses.
- **RBER-C (contextual, leave-one-out).** A first-order context model of step success:
  - Each (context, skill) pair (c, s) has its own posterior mean, shrunk toward the shared marginal mean of s:
    m_{s|c} = (κ·m_s + successes(c, s)) / (κ + attempts(c, s)).
  - The context c is the preceding skill, or a START token for the first step.
  - The strength is fixed at κ = 2.
  - The plan reward is ∏ₖ m_{s_k | c_k}.
  - Leave-one-out as in RBER-LOO.
  - (Revision note: the first draft used the marginal m_s for the first step. While writing the consistency proof, about one minute after the tuning job had started, I noticed this pools first-step outcomes with outcomes observed in other contexts, so the model is not consistent under interactions. The job was stopped and its partial log deleted unread, and the START context was introduced. No result of any extension method had been inspected.)

## New baselines (all trained with the same planner, optimizer and budget)
- **EXEC-RLOO:** the binary reward with the leave-one-out mean baseline and no standard-deviation normalization (REINFORCE leave-one-out).
- **EXEC-PPO:** the binary reward with 4 optimization epochs per batch and a clipped importance ratio (ε = 0.2).
- **LSP-lin:** a learned success predictor with the same product structure as RBER, P(Y=1|a) = ∏ₖ σ(θ_{s_k}). It is fitted by stochastic maximum likelihood on *final binary outcomes only*: 4 Adam steps per update on 64 samples drawn from a replay buffer of all executions. Its reward is the predicted success. This isolates the value of the step-level censored observations together with the conditional-expectation form.
- **LSP-mlp:** the same, with an unstructured MLP (one hidden layer of 64 units; input = the per-step one-hot skill vectors).
- **TS-plan (reference, not a policy):** Thompson sampling over plans with the shared Beta posterior. For each of the G plans it samples p, picks the best valid plan by enumeration, executes it, and updates. Its greedy plan is the argmax of the posterior-mean product. No hypothesis is attached to it; it is reported as a model-based planning reference.

## Tuning (same rule as the original study)
- **Planner learning rate:** selected per method and per condition (independent / δ = 0.3, at K = 3) on tuning seeds 1000–1004, from {0.3, 1, 3, 10}, by mean normalized AUC.
- **LSP model learning rate:** tuned jointly over {0.01, 0.05, 0.2}.
- **No learning rate:** κ is fixed at 2, and TS-plan has no learning rate.

## Experiments (evaluation seeds 0–19)
- **E6:** main conditions (K ∈ {3, 4} × independent / δ = 0.3) for RBER-LOO, RBER-C, EXEC-RLOO, EXEC-PPO, LSP-lin, LSP-mlp and TS-plan. RBER, EXEC and STEP are reused from the original study, which uses the same seeds and worlds.
- **E7:** interaction sweep (K = 3, δ ∈ {0, 0.1, …, 0.5}) for RBER-LOO and RBER-C.
- **E8:** a horizon sweep (K = 2…6) for RBER-LOO.
- **E9:** a κ ablation for RBER-C (κ ∈ {0.5, 2, 8}, K = 3, δ ∈ {0, 0.3}).

## Hypotheses
- **H8 (LOO):** RBER-LOO has a higher AUC than RBER in at least 3 of the 4 main conditions, by paired Wilcoxon test (Holm-corrected within H8).
- **H9 (context removes the floor):** In E7, RBER-C's final normalized success is not below EXEC's by more than 0.01 at δ = 0.4 and δ = 0.5 (mean difference and 95% bootstrap CI lower bound ≥ −0.01). Its AUC exceeds EXEC's at every δ (paired Wilcoxon, Holm).
- **H10 (baselines):** The better of RBER and RBER-LOO is pre-specified as **RBER-LOO**. RBER-LOO has a higher AUC than each of EXEC-RLOO, EXEC-PPO, LSP-lin and LSP-mlp in all four main conditions (paired Wilcoxon, Holm over the 16 tests).
- **H11 (independence case):** Without interactions, RBER-C's AUC is not lower than RBER-LOO's by more than 0.01; the context model costs little when it is not needed.

## Language-model extension (Kaggle; written before any of these runs)
- **Sessions:**
  - D: RBER-LOO, seeds 0–9.
  - E: EXEC, RBER and RBER-LOO, seeds 10–14.
  - F: EXEC, RBER and RBER-LOO, seeds 15–19.
- **Settings:** identical to the original study (Qwen2.5-0.5B-Instruct, LR 3e-6, 250 updates × 4 tasks × 8 plans, K = 3, the same world).
- **Pooling:** seeds 0–9 of EXEC and RBER come from sessions A–C.
- **H12 (primary):** over 20 seeds, RBER-LOO has a higher AUC (over updates) on the training tasks than EXEC (paired two-sided Wilcoxon, α = 0.05, a single primary test).
- **H13 (secondary):** over 20 seeds, held-out final success, RBER-LOO vs. EXEC and RBER vs. EXEC (paired Wilcoxon, Holm over the two tests); no direction is pre-committed. RBER vs. EXEC AUC over 20 seeds is also reported.
- If session F cannot be run, the analysis uses the seeds available. This is reported as a deviation.

## Statistics
Statistics follow the original study:
- AUC over updates, final normalized success, and E₀.₉ censored at the final execution count;
- interquartile means and means with 95% bootstrap CIs;
- paired two-sided Wilcoxon tests with Holm correction within each hypothesis family.

Anything not listed here is exploratory and will be labelled as such.

## Theory added in the same revision
- **Leave-one-out estimator:** unbiased for a fixed surrogate with no self-confirmation.
- **Contextual model:** consistent under first-order interactions.
- **Separation theorem:** outcome-only learning needs a number of executions exponential in K to rank two plans that differ in one skill. With shared step-level posteriors, the number of *attempts* of the two skills needed is independent of K.

## Amendment 1 (2026-10-01, after tuning, before any evaluation run)
- **What happened:** tuning selected edge values of the grid for the learned-predictor baselines. LSP-mlp chose planner lr 0.3 and model lr 0.01 in both conditions. LSP-lin chose planner lr 0.3 with interactions.
- **Change:** to avoid under-tuning a baseline, the LSP grids were extended to planner lr {0.1, 0.3, 1, 3, 10} × model lr {0.003, 0.01, 0.05, 0.2}, with the same rule and the same tuning seeds. This is the same rule as the amendment of the original study.
- **Main run:** an evaluation run of E6 had started with the old values. It was stopped after about one minute and its partial log was deleted unread.
- **Untouched:** no other method's tuning changed.
- **Result of the extended tuning:** LSP-mlp with interactions selected the new lowest model lr (0.003), with a tuning AUC gain of only 0.004 over the previous best. The grid was not extended again. All other choices were interior or unchanged.
- **Procedural note:** the second tuning pass (with the START context) was interrupted after 272 of 400 runs to set single-threaded BLAS, a pure speed fix, and was resumed. The finished runs were kept, and no code or setting that affects results changed.
