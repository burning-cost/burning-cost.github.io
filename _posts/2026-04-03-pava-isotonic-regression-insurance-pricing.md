---
layout: post
title: "PAVA in Three Places: Isotonic Regression for Insurance Pricing"
date: 2026-04-03
categories: [techniques, pricing, validation]
tags: [isotonic-regression, pava, pool-adjacent-violators, monotonicity, calibration, market-ratemaking, glm-recalibration, ebm, gbm, abc-smc, scipy, sklearn, wuthrich-ziegel, insurance-monitoring, insurance-gam, python, uk-insurance, personal-lines, governance]
description: "The Pool Adjacent Violators Algorithm solves an O(N) monotonicity problem with no parametric assumptions. It appears in three distinct insurance pricing contexts: as the link function in market-based ratemaking, as a post-hoc GLM recalibrator, and as a monotonicity enforcement tool for GBMs. Here is when to use it and when not to."
author: burning-cost
math: true
---

There is an algorithm that appears in three separate corners of insurance pricing that most pricing actuaries have encountered at most once. The Pool Adjacent Violators Algorithm — PAVA — solves a constrained regression problem in O(N) time with no distributional assumptions and no tuning parameters. It is rarely the headline technique in a paper or a project, which means most teams reach for it in one context and miss the other two.

This post covers the three places PAVA is useful in insurance pricing, what it actually does, when it fails, and which Python implementation to use in each context.

---

## What PAVA does

The problem PAVA solves: given a sequence of observations $(y_1, y_2, \ldots, y_N)$ with associated weights $(w_1, w_2, \ldots, w_N)$, find the non-decreasing sequence $(\hat{f}_1, \hat{f}_2, \ldots, \hat{f}_N)$ that minimises:

$$\sum_{i=1}^N w_i (y_i - \hat{f}_i)^2 \quad \text{subject to } \hat{f}_1 \leq \hat{f}_2 \leq \cdots \leq \hat{f}_N$$

The algorithm scans left to right. It maintains a stack of "blocks" — groups of adjacent observations that have been pooled to a common fitted value, which is their weighted mean. When it encounters a new observation that violates monotonicity (the new value is less than the top of the stack), it merges the new observation with the top block and replaces both with their weighted mean. It then checks whether this merge has violated the constraint with the block below, and if so merges again. This continues until monotonicity is restored, then moves to the next observation.

The result is a piecewise-constant non-decreasing function. The total number of merge operations is bounded by N, giving O(N) complexity. There are no knots to choose, no bandwidth to tune, no distributional assumption beyond the monotonicity constraint itself.

Three things follow from this. PAVA is robust to outliers, because pooling averages across them. It is not smooth — the output is a step function, not a curve. And it is non-parametric — the loading structure does not need to be linear, quadratic, or any other functional form.

---

## Use 1: The isotonic link in market-based ratemaking

The most recent application of PAVA in the actuarial literature is as the link function in Goffard, Piette, and Peters (ASTIN Bulletin, 2025, arXiv:2502.04082). Their problem: an insurer entering a new market has no claims data. All it can observe is competitor commercial premiums. Can it extract the underlying risk parameters — claim frequency λ and log-mean severity μ — from those premiums alone?

The method is Approximate Bayesian Computation. Sample candidate (λ, μ) pairs from a prior. Simulate expected claims cost per risk class under those parameters. Compare simulated costs against observed market premiums. Accept or reject candidates based on closeness. Run multiple generations, tightening the acceptance threshold each time.

The comparison step is where PAVA enters. Simulated pure premiums and observed commercial premiums are on different scales — commercial premiums include expense loading, profit margin, and competitive adjustment. A linear rescaling assumes constant loading margin, which is wrong. A polynomial link may produce non-monotone mappings — cheaper risks appearing to cost more — which is actuarially indefensible and creates instability in the ABC acceptance step.

PAVA solves this cleanly: map simulated pure premiums to observed commercial premiums using the isotonic fit. This enforces monotonicity (riskier profiles must command higher commercial premiums) without assuming anything about the loading structure. When five insurers quote the same risk at different prices, the PAVA-fitted mapping absorbs that variation as noise and outputs the monotone trend.

The practical implication: inside an ABC-SMC loop running up to 9,000 PAVA fits (J=1,000 particles × G≤9 generations), computational speed matters.

```python
from scipy.optimize import isotonic_regression
import numpy as np

def isotonic_link(pure_premiums, commercial_premiums):
    """
    Fit monotone mapping from simulated pure premiums to observed commercial premiums.
    Returns the fitted values at each pure premium level.

    Uses scipy.optimize.isotonic_regression (SciPy 1.12+).
    O(N) PAVA. No overhead. Use this inside the ABC loop.
    """
    # Sort by pure premium to establish ordering
    order = np.argsort(pure_premiums)
    y_sorted = commercial_premiums[order]
    result = isotonic_regression(y_sorted, increasing=True)
    # Return fitted values in original order
    fitted = np.empty_like(result.x)
    fitted[order] = result.x
    return fitted
```

Use `scipy.optimize.isotonic_regression` (SciPy 1.12+) inside the ABC loop. It is a pure O(N) implementation with no object overhead. Our [earlier post on the Python implementation](/2026/03/31/market-based-ratemaking-python-implementation/) works through the full ABC-SMC pipeline; the PAVA step is 10 lines.

---

## Use 2: Post-hoc GLM recalibration

The second application is less exotic. A GLM fitted to motor claims data will typically be well-calibrated in the aggregate but may not be well-calibrated across the predicted premium range — it may underpredict at the low end and overpredict at the high end, or vice versa.

Wüthrich and Ziegel (Scandinavian Actuarial Journal, 2023, arXiv:2301.02692) formalise this problem. A model is auto-calibrated if, for any threshold t, the expected loss conditional on the model predicting at least t equals t. GLMs are auto-calibrated in the sample by construction — the score equations guarantee balance in the fitted values. But in the test period, with distributional shift or covariate drift, this guarantee breaks down.

The practical failure mode is the decile calibration plot: you rank policies by predicted premium, divide into ten bands, and plot the ratio of actual to predicted within each band. A well-calibrated model produces a flat line at 1.0. What you often see is a U-shape or an inverted U — the model is systematically wrong at the extremes of the predicted range.

PAVA provides a non-parametric recalibration:

```python
from sklearn.isotonic import IsotonicRegression
import numpy as np

def recalibrate_glm(y_pred, y_actual, weights=None):
    """
    Post-hoc isotonic recalibration.

    Fits a monotone increasing mapping from predicted to actual.
    Use sklearn.isotonic.IsotonicRegression for full fit/predict API.
    Supports out_of_bounds='clip' for extrapolation to new data.
    """
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    ir.fit(y_pred, y_actual, sample_weight=weights)
    return ir

# Usage:
# calibrator = recalibrate_glm(pred_train, actual_train, weights=exposure_train)
# pred_calibrated = calibrator.predict(pred_test)
```

The `out_of_bounds='clip'` argument matters here: when predicting on the test set, some predicted values will fall outside the range seen during fitting. Clipping to the boundary value is usually correct for insurance — you should not extrapolate a calibration function into regions where you have no data.

Three caveats on using PAVA for recalibration:

**It is a post-hoc correction, not a structural fix.** If the model is miscalibrated at the high end because it is missing a rating factor (say, a high-risk occupation class), PAVA-correcting the predicted premiums will reduce the measured miscalibration without resolving the root cause. Use recalibration for short-term stabilisation, not as a substitute for understanding why the model is wrong. The `insurance-monitoring` library's Murphy decomposition (GMCB vs LMCB split) tells you whether you are dealing with a scale shift (PAVA-correctable) or a structural problem (model refit required).

**Monotonicity is not always correct.** For some rating factors, a non-monotone claim-cost relationship is genuine. Vehicle age versus repair cost per claim is genuinely non-monotone for older vehicles — classic and historic cars can have higher claim severity than newer equivalents despite lower frequency. PAVA will constrain this away if you apply it globally. Apply it on the predicted premium ordering — not on individual factor levels.

**Sample size matters.** PAVA with fewer than 500 claims per decile will produce fitted steps that are highly variable. With a thin segment, the isotonic fit is overfitting to the calibration sample noise. Use exposure-weighted PAVA (the `sample_weight` argument in sklearn) and check the fitted step values make sense before applying.

---

## Use 3: Monotonicity enforcement on GBM outputs

The third application is the most straightforward and the one most likely to be needed during a model governance review.

A GBM fitted to motor claims data will, in general, produce a non-monotone age effect. This is sometimes correct — the claims-frequency curve for age has a genuine non-monotone shape with a minimum around age 35–45. But the specific shape the GBM learns is partly data and partly noise, particularly in thin age bands at the extremes.

The governance problem: an FCA reviewer or a non-executive director can reasonably ask why a 22-year-old driver is priced lower than a 21-year-old in your model. If the answer is "the GBM found that pattern in the data," that is not a governance-approved answer. You need either an architectural constraint that prevents it, or a documented post-hoc correction.

For GBMs, two approaches exist. Architectural: use monotone constraint parameters available in XGBoost (`monotone_constraints`), LightGBM (`monotone_constraints`), and CatBoost (`monotone_constraints`). These bake the constraint into training. Post-hoc: extract the GBM's partial dependence for the constrained factor, apply PAVA to enforce monotonicity, and use the corrected PD values.

The post-hoc PAVA approach is simpler to implement but has a limitation the architectural approach does not: it corrects the marginal effect while leaving the GBM's internal tree structure unchanged. Interactions between the constrained factor and other factors may still produce non-monotone model output. The architectural constraint applies during training, so all interaction effects are constrained simultaneously.

Our [post on monotonicity-constrained EBMs](/2026/03/28/does-monotonicity-constrained-ebm-actually-work-for-insurance-pricing/) covers the `insurance-gam` implementation and benchmarks unconstrained vs constrained across 200 simulation runs. The conclusion there holds for GBMs too: architectural constraints cost Gini but guarantee compliance; post-hoc PAVA correction is cheaper to implement and handles most governance queries but is not bulletproof.

For a governance audit where the requirement is "show us your age relativities are monotone," post-hoc PAVA on the PD curve is sufficient and documentable:

```python
from scipy.optimize import isotonic_regression
import numpy as np

def enforce_monotone_pd(age_values, pd_values):
    """
    Enforce monotone decreasing PD curve for young-to-mid age driver effect.

    pd_values: partial dependence of log-frequency on age
    age_values: corresponding age points (sorted ascending)

    Enforce non-increasing (decreasing=True via negation) for the
    "higher age -> lower frequency" direction.
    """
    # Negate to use increasing solver for a decreasing constraint
    negated = -pd_values
    result = isotonic_regression(negated, increasing=True)
    return -result.x

# The corrected PD values can be tabulated in the governance report
# and used to override the GBM output for the age factor in rating.
```

Document the correction in the model record: "Partial dependence for age was post-hoc isotonic-corrected to enforce actuarial monotonicity. The correction reduced the Gini coefficient by 0.8 percentage points. The magnitude of the correction is within the uncertainty interval from 500-repeat bootstrap." That is a defensible entry in a Solvency II model change record.

---

## Which Python implementation to use

The two main implementations have different APIs and different performance profiles:

| Situation | Implementation | Reason |
|-----------|---------------|--------|
| Inside an ABC loop (called thousands of times) | `scipy.optimize.isotonic_regression` | Pure O(N), no object overhead, fastest single call |
| Post-hoc recalibration with `predict()` on new data | `sklearn.isotonic.IsotonicRegression` | Full fit/predict API, `out_of_bounds` handling |
| Monotone PD correction (one-off, governance audit) | Either | Negligible performance difference at N < 100 |

`scipy.optimize.isotonic_regression` is available in SciPy 1.12 (released November 2023). If you are on an older environment, use `sklearn` for everything — its isotonic regression implementation is correct and well-tested, just slightly heavier.

`sklearn.isotonic.IsotonicRegression` accepts `sample_weight` in its `fit()` method and handles weighted PAVA correctly. Use it whenever you have exposure weighting — which in insurance is almost always.

---

## When not to use PAVA

PAVA is not always the right tool for monotonicity problems in insurance.

**When the constraint is over a continuous covariate with a known smooth shape.** GLMs with age rated as a smooth basis (splines, or piecewise-linear with actuarial bands) allow you to enforce monotonicity via the basis specification rather than post-hoc correction. A cubic spline with a monotone constraint is smoother and more interpretable than a PAVA step function over age deciles. Use spline constraints when the underlying relationship is expected to be smooth.

**When the non-monotonicity is genuine.** Some relationships in insurance are non-monotone by design — premium as a function of sum insured for home contents can decrease per-unit as sum insured increases due to risk correlation effects. Applying PAVA there forces an incorrect monotonicity. Check that the monotonicity constraint is actuarially grounded before applying it.

**When N is very large and the fitted step function will not be interpretable.** PAVA with N=50,000 predictions produces a step function with potentially thousands of distinct steps. The result is correct but not useful for presenting to a governance committee. In this case, fit PAVA on binned predictions (deciles or centiles of the predicted distribution), then interpolate or smooth.

---

## Connecting the three uses

All three applications share the same structure. You have a quantity that should increase as another quantity increases — commercial premiums as underlying risk cost increases, actual claims as predicted claims increase, risk premium as age rises from 17 to 25. PAVA finds the monotone function that best approximates the observed data without imposing a parametric shape.

In market-based ratemaking, the PAVA output is used inside an inference algorithm, called many thousands of times. In recalibration, it is fitted once on historical data and applied to new predictions. In GBM governance, it is applied once to a partial dependence curve and the result is presented to a committee.

The algorithm is identical in all three cases. The surrounding infrastructure — how you get the inputs, what you do with the outputs, what uncertainty you can claim about the result — differs substantially.

The Goffard, Piette, and Peters paper surfaced PAVA to the actuarial pricing literature through the market ratemaking application. But the technique was already the right answer in the other two places too. If your validation pipeline includes a decile calibration plot and you have a monitoring alert, PAVA recalibration is 10 lines of code and a documented audit trail. If your GBM governance review requires demonstrable monotone age relativities and you do not want to retrain with architectural constraints, post-hoc PAVA is another 10 lines.

It is an O(N) algorithm with no parameters. The question is only whether the data you have actually supports the monotonicity assumption you are enforcing.

---

*Goffard, P.-O., Piette, P., and Peters, G. W. (2025). Market-based insurance ratemaking: application to pet insurance. ASTIN Bulletin. arXiv:2502.04082.*

*Wüthrich, M. V. and Ziegel, J. F. (2023). Isotonic recalibration under a low signal-to-noise ratio. Scandinavian Actuarial Journal. arXiv:2301.02692.*

*Barlow, R. E., Bartholomew, D. J., Bremner, J. M. and Brunk, H. D. (1972). Statistical Inference under Order Restrictions. Wiley.*

Related posts:
- [From Competitor Quotes to Risk Parameters: Implementing Market-Based Ratemaking in Python](/2026/03/31/market-based-ratemaking-python-implementation/) — the ABC pipeline where the isotonic link runs thousands of times
- [Does Monotonicity-Constrained EBM Actually Work for Insurance Pricing?](/2026/03/28/does-monotonicity-constrained-ebm-actually-work-for-insurance-pricing/) — architectural vs post-hoc approaches benchmarked
- [Recalibrate or Refit? The Murphy Decomposition Makes it a Data Question](/2026/02/28/recalibrate-or-refit/) — when PAVA recalibration is the right answer and when it is not
