# Pilot 4 (stage 1, CPU) — execution-grounded RL for a planner over stochastic skills
Written before any run. 2026-09-29.

## Setup
- Skill reliabilities: measured in Meta-World v3 (scripted expert policies x 3 perturbation levels,
  20 rollouts each, success at episode end). Only the measured success rates are used.
- Task family: T=20 long-horizon tasks, each K=4 subgoal slots; each slot offers 3 valid skills
  (drawn from the measured pool, skills are SHARED across tasks) + 1 invalid option.
  A plan = one choice per slot (81 valid of 256). Plan executes sequentially; success iff every
  skill succeeds; execution stops at the first failure and reveals which step failed.
- Planner: per-task, per-slot softmax logits (proxy for an autoregressive LLM planner).
- Training budget: 24,000 executed plans total (group size G=8 per task per update).
- Methods:
  VERIF      reward = plan valid AND matches a reference plan (REVER-style); reference = a random valid plan.
  EXEC       reward = binary execution success, GRPO group-normalized advantage.
  STEP       reward = fraction of skills completed before failure (dense progress), GRPO.
  RB (ours)  reward = product of Beta-posterior mean success of the plan's skills, posteriors updated
             from all observed skill outcomes across all tasks (Rao-Blackwellized / model-based).
- Conditions: (a) independent skills; (b) context-dependent: 25% of ordered skill pairs get a 0.3
  multiplicative penalty on the second skill's success (RB's independence model is misspecified).
- 10 seeds. Metric: true expected success of the greedy plan, averaged over tasks; optimum = best valid plan.

## Criteria
- K1 (problem matters): VERIF's final true success is >= 0.15 below the optimum (condition a).
- K2 (method matters): episodes for RB to reach 90% of the optimum <= 0.5 x episodes needed by the
  BETTER of EXEC and STEP (median over seeds; not reaching within budget counts as budget), condition a.
- K3 (robustness): in condition (b), RB's final true success >= best baseline's final - 0.05.

## Decision
K1 & K2 & K3 -> stage 2 on Colab (real small LLM planner + real simulator execution).
K1 fails -> problem not important, stop. K2 fails -> the proposed core adds nothing, stop.
K3 fails -> go only if a bias-correction (control-variate) variant fixes it, stated as a new hypothesis.

## Caveat
Synthetic composition favors shared-skill modelling by construction; stage 1 can only kill the
idea, not confirm it.
