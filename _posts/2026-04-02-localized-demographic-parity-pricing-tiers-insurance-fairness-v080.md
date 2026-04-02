---
layout: post
title: "Your GLM Passes Mean Fairness. Does It Pass the Tier Test?"
date: 2026-04-02
categories: [insurance-fairness, pricing-governance]
tags: [fairness, demographic-parity, fca-consumer-duty, equality-act, insurance-fairness, actuarial, arXiv-2603.25224, post-processing, pricing-tiers]
description: "insurance-fairness v0.8.0 adds LocalizedParityCorrector and LocalizedParityAudit — enforcement of demographic parity at pricing tier boundaries, not just on portfolio averages. Built from Charpentier et al. (arXiv:2603.25224)."
author: burning-cost
---

Most pricing actuaries have, at some point in the last two years, run a demographic parity check. You compute mean premium by protected group, compute the ratio, confirm it is within an acceptable band, and tick a box in the Consumer Duty evidence pack.

That check is necessary. It is not sufficient. And the FCA's multi-firm review in 2024 found that most firms' monitoring stopped exactly there.

[insurance-fairness v0.8.0](https://pypi.org/project/insurance-fairness/) adds `LocalizedParityCorrector` and `LocalizedParityAudit` — tools that enforce parity at every pricing tier boundary, not just at the portfolio mean. The theoretical basis is Charpentier, Denis, Elie, Hebiri & HU (arXiv:2603.25224, March 2026). The regulatory hook is Equality Act 2010 section 19 and FCA Consumer Duty Outcome 4.

---

## The problem with mean parity

Consider two pricing models for motor insurance — call them Model A and Model B. Both produce identical mean premiums for male and female policyholders: £650 for each group. Mean demographic parity ratio: 1.0. Both pass the standard audit.

Now look at the tier distribution. Model A places 45% of female policyholders in the top pricing band (above £900); Model B places only 22% there. Both models have identical means, but Model B concentrates one group disproportionately in the expensive tier. If that tier boundary is the threshold between affordable and unaffordable renewal, Section 19 exposure is not in the portfolio average — it is in the tier.

This is not a theoretical edge case. Any model with differential variance across groups produces this outcome. A GBM that correctly captures that claim severity variance differs by gender (a legitimate risk factor) will, as a side effect, produce unequal tier distributions even when mean premiums are equal. The tier disparity can be substantial while the mean ratio stays close to 1.0.

---

## Localized demographic parity

Charpentier et al. formalise the problem via an (ell, Z)-fair predictor. Given M pricing tier boundaries Z = {z_1, ..., z_M} and target CDF levels ell = {ell_1, ..., ell_M}, the constraint is:

```
F_{f|S=s}(z_m) = ell_m  for all groups s and all m in {1,...,M}
```

This reads: for every group s, the fraction of that group's predictions at or below tier boundary z_m must equal ell_m. If ell is set to the portfolio CDF values — the marginal variant — you are requiring each group to have the same tier distribution as the overall portfolio.

This is strictly weaker than full distributional demographic parity, which would require every quantile to match across groups. Full distributional parity, implemented via optimal transport in `DiscriminationFreePrice`, forces the entire predicted distribution to be identical between groups. That destroys accuracy wherever groups have genuinely different risk distributions. Localized DP enforces equality only at the M policy-relevant thresholds — the tier boundaries you actually report against in your governance pack.

The paper proves three things we care about:

1. **The optimal correction has a closed-form Lagrangian characterisation.** For each group s, the post-processing transformation T_s is the solution to an M-point dual program — finite-dimensional, convex, and computable via a standard LP solver.
2. **The accuracy gap vs continuous-DP-optimal is O(1/M).** Adding more tier boundaries reduces the accuracy you sacrifice to achieve fairness. At five tiers, the gap is small; at twenty, it is negligible.
3. **Constraint violation converges at O(1/sqrt(n)) in calibration sample size.** For a calibration set of 10,000 policies, constraint violation is bounded at roughly 1%.

---

## The audit: LocalizedParityAudit

Run this before any correction to see where your tiers are exposed:

```python
from insurance_fairness import LocalizedParityAudit

# UK motor: five pricing bands
audit = LocalizedParityAudit(
    thresholds=[350.0, 550.0, 800.0, 1200.0, 2000.0],
)
report = audit.audit(predictions, gender_codes)

print(f"Max disparity: {report.max_disparity:.4f}")
print(report.group_cdf_table)
```

`group_cdf_table` is a Polars DataFrame with columns `group`, `threshold`, `empirical_cdf`, `target_cdf`, and `deviation`. One row per (group, threshold) pair. Positive deviation means the group is overrepresented below that tier boundary; negative means underrepresented.

The `max_disparity` value is what a supervisor would ask about. If it exceeds your tolerance threshold (we use 0.03 as a starting point — 3 percentage points of tier population), you have documented tier-level disparity.

---

## The correction: LocalizedParityCorrector

Post-process any model's predictions to satisfy the tier constraints:

```python
from insurance_fairness import LocalizedParityCorrector

corrector = LocalizedParityCorrector(
    thresholds=[350.0, 550.0, 800.0, 1200.0, 2000.0],
    mode='quantile',   # enforce common ell_m targets across groups
)
corrector.fit(predictions_train, gender_codes_train)

# At prediction time
fair_predictions = corrector.transform(predictions_test, gender_codes_test)

# Verify constraint satisfaction
post_report = corrector.audit()
print(f"Post-correction disparity: {post_report.max_disparity:.4f}")
```

The corrector constructs a per-group piecewise-linear transformation T_s that maps each group's prediction distribution to satisfy the CDF constraints at each tier boundary. Between tier boundaries, predictions are interpolated linearly — rank order is preserved, so the correction is monotone.

For the case where gender (or another protected attribute) is unavailable at prediction time — the correct UK posture post-Gender Directive — use `mode='marginal'`:

```python
corrector = LocalizedParityCorrector(
    thresholds=[350.0, 550.0, 800.0, 1200.0, 2000.0],
    mode='marginal',  # match portfolio CDF; no group label at inference
)
corrector.fit(predictions_train, gender_codes_train)
fair_predictions = corrector.transform(predictions_test)  # no group arg needed
```

Marginal mode learns separate per-group transformations during calibration, then applies a single portfolio-level transformation at inference. No sensitive attribute is required post-fit.

---

## Lagrange multipliers as an FCA evidence pack

The correction's dual variables — `corrector.lagrange_multipliers_` — tell you which tiers required the largest adjustment and for which groups. These are a direct answer to the question any reviewer will ask: "Where was the model unfair, and how much did you change it?"

```python
import numpy as np

lm = corrector.lagrange_multipliers_  # shape (n_groups, M)
groups = corrector.groups_
thresholds = corrector.thresholds_

for g, group in enumerate(groups):
    for m, thresh in enumerate(thresholds):
        if abs(lm[g, m]) > 0.1:
            print(f"Group {group}, tier ≤£{thresh:.0f}: λ = {lm[g,m]:.3f}")
```

A large positive lambda at a particular (group, tier) means that group was significantly underrepresented below that tier boundary — and a meaningful correction was applied. Near-zero lambda means that constraint was essentially satisfied by the base model and the correction there was free. This is the narrative structure a pricing actuary needs when writing a regulatory submission.

---

## Where this fits in the insurance-fairness toolkit

The library already contains mean parity (`BiasMetrics.demographic_parity_ratio`), full distributional parity via optimal transport (`DiscriminationFreePrice`), and multicalibration (`MulticalibrationAudit`). These are not competing tools.

The workflow we recommend is:

1. **Check mean parity** with `BiasMetrics` — lowest cost, required for any Consumer Duty evidence pack.
2. **Check tier-level parity** with `LocalizedParityAudit` at your actual pricing band boundaries. This is the step most firms are missing.
3. **If tier-level violations exist**, apply `LocalizedParityCorrector` post-model. No refit needed.
4. **Verify calibration post-correction** with `MulticalibrationAudit` — confirm the correction has not degraded calibration within each group.
5. **If you need distributional parity across the entire prediction range** — typically for a regulatory deep-dive rather than routine monitoring — apply `DiscriminationFreePrice`.

The `MulticalibrationAudit` step is important. The localized DP correction is a monotone transformation of model scores, not a recalibration. In our testing on synthetic motor data, running `MulticalibrationAudit` after `LocalizedParityCorrector` showed negligible calibration degradation at M=5 tiers, rising to a detectable but small effect at M=20. For operational pricing models, five to eight tiers is the right range.

---

## A note on the Equality Act framing

Section 19 of the Equality Act 2010 asks whether a provision, criterion, or practice puts a protected group at a "particular disadvantage." The localized DP framework maps to this directly: each tier boundary is a concrete threshold at which you can document whether a particular group is disproportionately represented. A disparity of 8 percentage points at the £800 tier boundary — meaning 38% of female policyholders are priced above £800 versus 30% of male policyholders — is exactly the kind of evidence a section 19 claim would rely on.

The corrector gives you a documented, auditable, model-agnostic response: here is the disparity we measured, here is the transformation we applied, here is the constraint violation post-correction. Lagrange multipliers quantify the cost of compliance. That is the structure of a defensible FCA evidence pack.

---

**Paper:** [arXiv:2603.25224](https://arxiv.org/abs/2603.25224) | **Library:** [insurance-fairness on PyPI](https://pypi.org/project/insurance-fairness/) | **GitHub:** [insurance-fairness](https://github.com/insurance-fairness/insurance-fairness)
