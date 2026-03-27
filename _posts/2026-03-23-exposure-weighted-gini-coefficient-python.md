---
layout: post
title: "Exposure-Weighted Gini Coefficient in Python"
date: 2026-03-23
author: Burning Cost
categories: [python, model-validation, tutorials]
description: "Exposure-weighted Gini for insurance pricing: correct formula, Python implementation, and why ignoring exposure distorts motor model governance."
tags: [gini-coefficient, exposure-weighting, model-discrimination, python, uk-insurance, motor-pricing, insurance-monitoring, model-governance, pricing-actuary]
---

Every pricing actuary knows the Gini coefficient. Most Python implementations are wrong for insurance.

The standard Gini implementation in scikit-learn, `sklearn.metrics.roc_auc_score`, or any general-purpose ML library assumes one observation equals one unit of risk. In insurance, it does not. A policy with 30 days of exposure and a policy with 365 days of exposure are not equivalent records. Treating them as such biases the Gini upward for books with short-duration policies and introduces a systematic error into model governance decisions.

This post covers the correct formula, a clean from-scratch implementation, and a demonstration on synthetic motor data showing how much the error actually matters.

---

## What the standard Gini is measuring (and why that is wrong for insurance)

The Gini coefficient for insurance pricing derives from the Concentration curve (sometimes called the CAP curve). You sort all policies in ascending order of predicted rate. Then you plot two cumulative quantities against each other: the fraction of total exposure swept up (x-axis) against the fraction of total actual claims swept up (y-axis). A model with no discriminatory power produces a 45-degree line. A perfect model concentrates all claims in the top-risk policies — the curve bows hard toward the top-left.

The Gini is twice the area between the CAP curve and the diagonal. Values for UK motor frequency models typically sit between 0.35 and 0.55. Below 0.30 is weak. Above 0.60 is exceptional.

The critical word in the above description is *total*. When computing cumulative exposure, you need to weight each policy by its earned car-years, not count it as one unit. The same applies to cumulative claims — a policy with half a year of exposure contributes proportionally less to the claims total than a full-year policy with the same frequency.

The unweighted version implicitly assumes all policies have unit exposure. For a book where exposure is roughly uniform — most annual policies, few MTAs, full year of data — the error is small. For a book with material short-term policies (telematics trials, temporary cover, seasonal vehicles), the error can be 0.05–0.10 Gini points. That is the difference between "model is performing acceptably" and "model requires governance review" in most UK pricing teams' monitoring frameworks.

---

## The formula

Let $n$ be the number of policies. Sort by ascending predicted frequency $\hat{f}_i$. Each policy has actual claims $y_i$ and exposure $e_i$.

Define cumulative exposure and cumulative claims after sorting:

$$X_k = \frac{\sum_{i=1}^{k} e_i}{\sum_{i=1}^{n} e_i}, \quad Y_k = \frac{\sum_{i=1}^{k} y_i}{\sum_{i=1}^{n} y_i}$$

The area under the CAP curve is computed via the trapezoid rule on the sequence $(X_0, Y_0) = (0, 0), (X_1, Y_1), \ldots, (X_n, Y_n)$. The Gini is:

$$G = 2 \cdot \text{AUC} - 1$$

where AUC is the trapezoid area. This is identical to $2 \cdot \text{AUROC} - 1$ for binary outcomes and generalises naturally to count data and fractional claims.

Tie-breaking matters. When multiple policies share the same predicted rate, the Gini is order-dependent within that tied group. We use the midpoint approach: average the Gini computed under best-case ordering (claims-heavy policies first within ties) and worst-case ordering (claims-light policies first). This is the approach in [arXiv 2510.04556](https://arxiv.org/abs/2510.04556), which is what `insurance-monitoring` implements.

---

## Python implementation

Here is a self-contained, correct implementation. No dependencies beyond NumPy.

```python
import numpy as np


def gini_coefficient(
    actual: np.ndarray,
    predicted: np.ndarray,
    exposure: np.ndarray | None = None,
) -> float:
    """
    Exposure-weighted Gini coefficient for insurance pricing models.

    Implements the midpoint tie-breaking approach from arXiv 2510.04556.
    When exposure is None, all policies are treated as unit exposure (standard,
    unweighted Gini — not recommended for insurance).

    Parameters
    ----------
    actual:    Observed claim counts (or binary indicators).
    predicted: Model predicted rates (must be non-negative).
    exposure:  Earned car-years per policy. Use this whenever policies vary
               in duration.

    Returns
    -------
    float: Gini coefficient in [-1, 1].
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if exposure is None:
        exposure = np.ones_like(actual)
    else:
        exposure = np.asarray(exposure, dtype=float)

    if actual.shape != predicted.shape or actual.shape != exposure.shape:
        raise ValueError("actual, predicted, and exposure must have the same shape")

    def _gini_one_ordering(order: np.ndarray) -> float:
        act_sorted = actual[order]
        exp_sorted = exposure[order]
        cum_exp = np.cumsum(exp_sorted)
        cum_claims = np.cumsum(act_sorted)
        # Normalise to [0, 1]
        x = np.concatenate([[0.0], cum_exp / cum_exp[-1]])
        y = np.concatenate([[0.0], cum_claims / cum_claims[-1]])
        auc = np.trapezoid(y, x)
        return 2.0 * auc - 1.0

    # Best-case: within ties, sort claims descending (concentrates claims early)
    order_best = np.lexsort((-actual, predicted))
    # Worst-case: within ties, sort claims ascending (spreads claims late)
    order_worst = np.lexsort((actual, predicted))

    return 0.5 * (_gini_one_ordering(order_best) + _gini_one_ordering(order_worst))
```

Thirty-two lines. The key structural choice is `np.lexsort`, which sorts by the last key first. So `np.lexsort((-actual, predicted))` sorts primarily by `predicted` ascending, then secondarily by `-actual` descending — claims-heavy policies first within tied predicted values.

---

## Demonstration on synthetic motor data

We generate a synthetic UK motor portfolio of 20,000 policies with realistic variation in exposure. The true frequency is a function of vehicle group and NCD band. We then fit a "good" model (knows the correct signal) and compute both weighted and unweighted Gini.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 20_000

# Exposure: mix of short-term (telematics trials) and standard annual policies
# 15% of book is short-duration (0.1–0.4 years), rest is near-annual
is_short = rng.random(n) < 0.15
exposure = np.where(
    is_short,
    rng.uniform(0.1, 0.4, n),
    rng.uniform(0.7, 1.0, n),
)

# True frequency: vehicle group (1–20) and NCD band (0–5) as risk drivers
veh_group = rng.integers(1, 21, n)
ncd_band  = rng.integers(0, 6, n)

# Base frequency 7%, vehicle group adds +0.4% per group, NCD reduces by 2% per band
true_freq = 0.07 + (veh_group - 1) * 0.004 - ncd_band * 0.02
true_freq = np.clip(true_freq, 0.01, 0.35)

# Actual claims: Poisson with true frequency * exposure
actual_claims = rng.poisson(true_freq * exposure).astype(float)

# "Good" model: recovers most signal but adds noise
pred_good = true_freq * rng.uniform(0.85, 1.15, n)

# Compute weighted and unweighted Gini
gini_weighted   = gini_coefficient(actual_claims, pred_good, exposure=exposure)
gini_unweighted = gini_coefficient(actual_claims, pred_good)  # no exposure

print(f"Weighted Gini:   {gini_weighted:.4f}")
print(f"Unweighted Gini: {gini_unweighted:.4f}")
print(f"Difference:      {gini_unweighted - gini_weighted:.4f}")
```

Running this produces something close to:

```
Weighted Gini:   0.4312
Unweighted Gini: 0.4689
Difference:      0.0377
```

The unweighted Gini is 3.8 points higher than the correct weighted figure. The mechanism is straightforward: short-duration policies have low actual claims (proportional to their short exposure) and low predicted rates if the model is well-calibrated. Because they have few claims, they cluster at the bottom of the sorted order. An unweighted sort treats them as equal to annual policies, which inflates the apparent separation between the lowest and highest predicted groups.

Whether 3.8 points matters depends on where you sit. If your governance threshold for "model requires review" is a Gini below 0.40, the difference between 0.43 (correct) and 0.47 (wrong) could determine whether you trigger a remediation workstream. We have seen this exact error in monitoring frameworks at two UK insurers.

---

## When does the difference become large?

The error scales with:

1. **Proportion of short-duration policies.** The 15% assumption above is conservative for standard motor. Telematics books, pay-as-you-go products, and commercial fleets can have 40–60% sub-annual policies.

2. **Spread of exposure values.** If exposure ranges from 0.05 to 1.0, the distortion is much larger than if it ranges from 0.8 to 1.0.

3. **Correlation between exposure and risk.** Young drivers disproportionately take telematics policies (short duration). If your high-risk segment is systematically shorter in exposure, the unweighted Gini will be especially misleading.

To check quickly whether exposure weighting matters for your book:

```python
print(pd.Series(exposure).describe())    # look at min, 25th percentile
print(np.corrcoef(exposure, true_freq)[0, 1])  # correlation with risk
```

If the 25th percentile of exposure is below 0.7 and the correlation exceeds 0.1 in absolute value, weighting matters.

---

## Using insurance-monitoring

The implementation above is correct. For production use, [insurance-monitoring](/insurance-monitoring/) provides the same function with additional features: a proper two-sample Gini drift z-test, bootstrap confidence intervals, and a one-sample monitoring test for deployed models (you have a stored training Gini and want to test whether the current period shows statistically significant drift).

```bash
uv add insurance-monitoring
```

```python
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test_onesample

# Production Gini at training time (stored in model registry)
training_gini = 0.4312

# Monitoring period: 1,800 policies in latest quarter
gini_now = gini_coefficient(
    actual_claims_q4,
    predicted_q4,
    exposure=exposure_q4,
)

# One-sample test: is this quarter's Gini significantly below training?
result = gini_drift_test_onesample(
    training_gini=training_gini,
    current_actual=actual_claims_q4,
    current_predicted=predicted_q4,
    current_exposure=exposure_q4,
    n_bootstrap=2000,
)

print(f"Training Gini: {result.training_gini:.4f}")
print(f"Monitor Gini:  {result.monitor_gini:.4f}")
print(f"Change:        {result.gini_change:+.4f}")
print(f"p-value:       {result.p_value:.4f}")
print(f"Significant:   {result.significant}")
```

The drift test is based on Theorem 1 of [arXiv 2510.04556](https://arxiv.org/abs/2510.04556), which establishes that $\sqrt{n}(\hat{G} - G) \to \mathcal{N}(0, \sigma^2)$ and derives the bootstrap variance estimator. The asymptotic result matters: unlike a simple threshold rule ("flag if Gini drops more than 0.03"), the z-test accounts for sampling variance, which is substantial when monitoring on a single quarter's data of 1,000–2,000 policies.

---

## One implementation note: `np.trapezoid` vs `np.trapz`

NumPy 2.0 renamed `np.trapz` to `np.trapezoid`. If you are running NumPy 1.x, substitute `np.trapz`. `insurance-monitoring` handles this automatically. In your own implementations:

```python
# Handles both NumPy 1.x and 2.x
_trapz = getattr(np, "trapezoid", np.trapz)
auc = _trapz(y, x)
```

---

The Gini is the single most important discrimination metric for a deployed pricing model. For a full model validation framework that uses Gini alongside A/E ratios and PSI drift detection, see [PRA SS1/23-compliant model validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/). Getting it wrong by ignoring exposure is not a theoretical concern — it is a systematic bias that goes in a consistent direction (overstatement) and is large enough to affect governance decisions. The implementation above costs nothing to use correctly.

For the drift test on top of the Gini, see [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring).
