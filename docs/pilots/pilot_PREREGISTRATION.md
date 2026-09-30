# Pilot pre-registration (written before any run)

Date: 2026-09-28

## Setting
Toy pick-and-place gridworld (6x6). True success = object released on the goal cell.
Verifier = logistic regression trained on "base" rollouts (scripted expert with random
truncation + random wandering), seeing precise gripper position but noisy object position
and a noisy "holding" flag (mimics a VLM: good proprioceptive cues, poor object perception).
RL = tabular Q-learning, terminal reward = verifier probability at the final state.

## Hypotheses and thresholds
- H1 (problem is real): verifier agreement on base distribution >= 85%, yet after RL on the
  verifier alone, verifier-reported success minus true success >= 30 points (mean over seeds).
- H2 (selection matters): at an audit budget <= 5% of episodes, exploit-aware audits reach
  >= 70% of oracle true success AND beat random-positive audits (same budget, same
  correction rule) by >= 15 points or >= 1.5x, at the lowest budget tested.
- H3 (budget scaling, the core theoretical claim): total audits used by the exploit-aware
  rule grow sub-linearly with training length (doubling episodes increases audits by < 1.4x),
  while keeping true success.
- H4 (certificate): prediction-powered lower bound computed from on-policy audits has
  empirical coverage >= 90% (target 95%), while a verifier-only bound calibrated on the base
  distribution has coverage < 50% for verifier-trained policies.

## Decision rule
- GO: H1, H3, H4 hold and H2 holds.
- GO with reframing: H1, H3, H4 hold but H2 fails (contribution shifts to theory + benchmark +
  certificate; audit-selection algorithm is not a headline claim).
- NO-GO: H1 fails (exploitation does not emerge from a verifier that looks accurate) or H4 fails.

## Known limitations (stated up front)
- H1 is partly true by construction in any toy; it only shows the mechanism is not contrived.
- No embodied probes or real VLM here; those need a physics simulator + GPU.
