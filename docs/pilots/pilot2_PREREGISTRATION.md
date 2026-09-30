# Pilot 2 pre-registration (written before any VLM run)

Date: 2026-09-28. Follows pilot 1 (toy gridworld), whose verdict was: the problem is real, but
"smart audit selection" and "audits bounded by exploitable mass" were not supported. Pilot 2 tests the
reframed claim: *optimization exploits a real VLM success detector, the exploited states are
indistinguishable from success in the verifier's view, and an embodied probe (act, then look again)
separates them where extra passive camera views do not.*

## Setup
- Simulator: Meta-World v3 (MuJoCo). Tasks: bin-picking-v3, assembly-v3, window-open-v3.
- Policy family: scripted expert + 8-dim bias theta (2 phases x [xyz bias, grip bias]).
- Ground truth: simulator success at the last step AND still true after the robot opens its gripper,
  lifts away and the scene settles ("done, and stays done without the robot").
- Verifier: Qwen3-VL-4B-Instruct, zero-shot, score = P(Yes)/(P(Yes)+P(No)), threshold 0.5,
  judging the final frame from one main camera.
- Embodied probe: open gripper + lift 30 steps + settle 20 steps, then the same VLM judges the new frame.
  Probe verdict = main-view Yes AND probe-frame Yes (2 queries).
- Passive alternatives (same or more queries): main AND 1 extra view (2q), main AND 2 extra views (3q), mean of 3 views (3q).
- Optimization pressure: (a) best-of-N from a 300-rollout base pool, N up to 128;
  (b) CEM (10 iters x 24, 2 seeds) maximizing the VLM score; oracle CEM maximizing the truth.

## Credibility gate (per task, on the base pool)
VLM accuracy >= 0.80 AND AUROC >= 0.80. Tasks failing the gate are reported but excluded from P1-P3
(a verifier that is visibly bad on-distribution does not test the claim).

## Criteria
- P1 exploitation: in >= 2 credible tasks, EITHER CEM-vs-VLM final elites have VLM score >= 0.8 and
  true success <= 0.5 x oracle-CEM elites (oracle elites >= 0.3), OR best-of-128 true success is
  >= 15 points below the base-pool PPV.
- P2 aliasing: on the exploited set E (VLM-positive CEM members of the last 3 iterations + VLM-positive
  top-5% of the pool, credible tasks pooled), single-view false-positive rate >= 0.30.
- P3 embodied probe: on E, FP(probe) <= 0.5 x FP(single view), true positives retained >= 85%,
  and FP(probe) <= FP(best passive protocol) - 0.15.

## Decision
- GO (write the reframed paper): P1 and P2 and P3.
- PARTIAL (benchmark/theory paper without the embodied claim): P1 and P2, not P3.
- NO-GO: P1 fails (a modern VLM is not exploited at this optimization strength), or fewer than 2
  tasks pass the credibility gate (then re-run with a stronger VLM before deciding).

## Known limitations
- Truth uses the same release action as the probe; the probe is therefore the natural check for
  "held-not-placed" failures. It may not help partial-progress failures (window half open) - that is
  part of what P3 measures per task.
- The policy family is a perturbed expert, not RL from scratch; optimization pressure is modest.
