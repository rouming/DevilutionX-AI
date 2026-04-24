# Healthy training ranges for PPO metrics.
# Used for scale bar display and, potentially, early-stop heuristics.
# Each metric has a good_hi value: upper bound of the "good zone".
# Values below 0 (good_lo) show [<---], above good_hi show [--->].

# KL divergence: PPO target ~0.01, anything above 0.02 is worth watching
KL_GOOD_HI = 0.02

# Clip fraction: up to 10% clipping is normal, above that the policy is
# changing too fast
CLIP_FRAC_GOOD_HI = 0.10

# Gradient norm: below 0.5 is healthy, above suggests instability
# (note: max_grad_norm is typically set to 0.5, so clipping kicks in there)
GRAD_NORM_GOOD_HI = 0.5
