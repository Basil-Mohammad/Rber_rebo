# Pilot 4 — stage 2 pre-registration (written before any run)
Date: 2026-09-29

## Question
Does the stage-1 result survive when the planner is a real language model trained with GRPO?
Stage 1 (tabular planner, 10 seeds) passed K1-K3: RB reached 90% of optimum ~3.5x faster than the best
tuned baseline, and stayed ahead when its independence assumption was wrong.

## Setup
- Planner: Qwen2.5-0.5B-Instruct, full fine-tuning, fp32, AdamW lr 1e-5 (same for all methods, not tuned),
  grad-clip 1.0, GRPO with group-normalized advantages, 150 updates x 4 tasks x 8 samples.
- Domain: 54 skills with success rates measured in Meta-World (stage 1), grouped into 6 semantic
  categories. Skill names carry a random mode label (alpha/beta/gamma) unrelated to reliability.
  24 training tasks + 12 held-out tasks (new combinations), 3 steps each, 4 options per step
  (3 valid skills of the step's category + 1 skill from another category).
- Execution: skills run in order, each succeeds with its measured probability, stop at first failure.
  (Meta-World scenes cannot be chained physically; true sequential execution is stage 3.)
- Methods: VERIF (1 seed), EXEC (3 seeds), RB (3 seeds). Seeds change sampling and execution only.
- Metric: true success of the greedy plan (temperature 0), train and held-out tasks.

## Criteria
- S0 (learning happens): RB or EXEC improves train success by >= 0.05 over its own zero-shot value.
  If S0 fails the run is INCONCLUSIVE (not a negative result): re-run everything with a larger LR/budget.
- S1 (problem matters): VERIF final train success <= optimum - 0.15.
- S2 (primary): RB's train success at half of EXEC's execution budget >= EXEC's final train success,
  AND RB final >= EXEC final (means over seeds).
- S3 (secondary, generalization): RB held-out success >= EXEC held-out + 0.05.

## Decision
S0 & S2 -> GO: start the paper; stage 3 (true long-horizon execution, e.g. LIBERO-long) becomes its main experiment.
S0 & not S2 -> STOP this direction.

## Amendment 1 (2026-09-29, after run 1, before run 2) — criteria unchanged
Run 1 was INCONCLUSIVE (S0 failed) because training collapsed, not because of the methods:
free-text sampling at temperature 1 from a 0.5B model gave mostly unparseable answers; GRPO then pushed
down the answer-format tokens and parse rate fell from 1.00 to 0.00 within 10 updates in 5 of 6 EXEC/RB
runs, after which all rewards were 0 and learning stopped. The pre-registered remedy ("raise LR") was
therefore wrong in direction. Changes for run 2:
1. Constrained decoding: the answer is always "1:X 2:Y 3:Z"; at each step the model's own next-token
   distribution restricted to the letters A-D is the policy (exact log-probs, format cannot collapse).
2. LR chosen before any EXEC/RB run by a fixed rule on VERIF only (reward needs no execution):
   best mean VERIF reward over updates 16-20 among {1e-6, 3e-6, 1e-5}. Same LR for all methods.
S0-S3 and the decision rule are unchanged.

## Amendment 2 (2026-09-30, after run 2, before run 3) — criteria unchanged
Run 2 was INCONCLUSIVE (S0 failed): with letters A-D as actions the gradient mostly moved a letter prior
that is meaningless (options are shuffled per task). The pre-registered VERIF calibration showed it:
LR 1e-6 did not move (RB's greedy plan identical for 140 updates); LR >= 3e-6 collapsed (VERIF reward
0.28 -> 0.15 within 20 updates). Changes for run 3:
1. Actions are skill NAMES: the policy for step k is the softmax over the model's log-likelihood of each
   candidate skill name (SayCan-style), independently per step. Verified offline: exact log-likelihoods
   (max diff 2e-6 vs unbatched reference), and with a tiny random model the whole pipeline learns
   (EXEC 0.016 -> 0.096, RB 0.016 -> 0.106 train; held-out also improves) — a mechanics check only.
2. LR re-calibrated with the same VERIF rule, grid widened to {1e-6, 3e-6, 1e-5, 3e-5}.
3. Budget 250 updates, evaluation every 25 (time limit on free GPUs); sampling/evaluation forward passes
   in fp16 autocast with an fp32 fallback, training log-probs in fp32.
S0-S3 and the decision rule are unchanged.
4. Candidate score = per-token mean log-likelihood of the skill name (length-normalized), to avoid a
   near-deterministic initial policy that favors short names; policy entropy is logged at every evaluation.

## Amendment 3 (2026-09-30, after an external code review, before run 3) — criteria unchanged
- Terminology: the update is one on-policy step per batch, i.e. REINFORCE with a group-normalized
  baseline (GRPO with a single update per batch; ratio = 1, clipping inactive).
- S1 caveat: VERIF imitates a reference plan chosen at random among valid plans (a reliability-blind
  annotator), so S1 holds largely by construction; it is a sanity check, not evidence that the problem
  matters. That evidence is external (REVER: 43.6% of real-robot failures came from execution of correct
  plans). An optimal-plan reference was rejected: it would be an oracle with reliability knowledge,
  which is exactly what offline verification lacks.
- Defensive changes only: sampling probabilities cast to float64 before sampling; entropy logged on
  all 24 training tasks; a warning if the half-budget comparison would fall back to zero-shot.

## Amendment 4 (2026-09-30, after run 3 crashed at start) — no scientific change
Run 3 stopped with CUDA out-of-memory in the first LR-calibration step, before any training: a reference
cycle (lambda capturing self) kept the sanity-check model, gradients and Adam state (~8 GB) alive after
`del`. Fixed by removing the cycle and freeing memory explicitly (Policy.close()). Nothing else changed.
