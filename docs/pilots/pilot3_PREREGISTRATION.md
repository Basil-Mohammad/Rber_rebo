# Pilot 3 (feasibility, CPU, existing data) — written before running

Direction: label-free estimation of a robot verifier's precision on the optimization-selected tail,
from agreement among several cheap checks (camera views), under a latent-class model.
Gap: HedgeTune (Khalaf et al. 2025) locates the best-of-n hacking peak but needs true labels;
RoboRMBench (Sep 2026) gives a static fragility metric, not an estimate of the true success level.

Data: pilot-2 CEM populations (Qwen3-VL-4B, 2 seeds x 10 iters x 24 = 480 per task), with truth.
Checks: main view m, extra views e1, e2 (primary, passive only); + probe frame (secondary, confounded
with the truth definition, reported separately).

Estimator: 2-class latent model, each check's logit score Gaussian given class, conditionally
independent; fit by EM on all 480 rows WITHOUT labels. Tail = top-K rows by m, K in
{240,120,60,30,15,8}. Estimate = mean posterior P(y=1) over the tail. Baseline = same model fit on m alone.

Criteria (tasks assembly-v3 and bin-picking-v3; window-open-v3 reported as negative control):
- F1: mean |estimate - true tail success| over K <= 0.15 in both tasks.
- F2: that error <= 0.5 x the m-only baseline error in both tasks.
- F3 (assembly): estimate falls from K=120 to K=8 when true success falls.

Decision: F1-F3 hold -> proceed to a real pilot on fresh data (new tasks, new VLM prompts).
Otherwise -> the direction lacks signal; report and pick another.
Caveat: I have seen these data before (not this analysis); n = 2 usable tasks; this only
tests whether a signal exists, not the method.
