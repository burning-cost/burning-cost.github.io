---
layout: post
title: "Conditional Coverage and Conformal Prediction Model Selection: CVI and CC-Select"
date: 2026-03-31
categories: [machine-learning]
tags: [conformal-prediction, conditional-coverage, model-selection, prediction-intervals, insurance-conformal, CVI, CC-Select, TweedieConformPredictor, Zhou-2026, uncertainty-quantification]
description: "Marginal coverage guarantees say nothing about which policyholders are being undercovered. CVI decomposes conditional coverage into undercoverage risk and overcoverage cost. CC-Select picks the best conformal predictor using it. Both are now in insurance-conformal v0.8.0."
author: burning-cost
---

Your conformal prediction intervals achieve 90% coverage. You have the validation numbers to prove it. The question you have not answered is: 90% coverage on which policies?

Split conformal guarantees `P(y in C(X)) >= 1-alpha`. That is a marginal statement — it holds over the full test distribution. It says nothing about `P(y in C(X) | X=x)` for any particular risk segment. A UK motor book could achieve exactly 90% aggregate coverage while systematically undercovering 25-year-old male drivers at 70% and overcovering 55-year-old female drivers at 97%. The aggregate number passes validation. The young male segment is carrying undocumented interval risk.

This is the fundamental limitation of split conformal, well understood in the literature. What has been missing is a practical tool for measuring and acting on conditional coverage quality when you have multiple conformal predictor variants to choose between.

Zhou, Zhang, Tao, and Yang (arXiv:2603.27189, March 2026) provide exactly this: the Conditional Validity Index (CVI) and CC-Select algorithm. Both are now in insurance-conformal v0.8.0.

---

## Two distinct problems

Two separate questions about conditional coverage are often conflated:

1. **Diagnosis**: does this predictor have conditional coverage failures? Is there evidence that coverage varies meaningfully across the feature space?
2. **Selection**: among multiple competing conformal predictors, which has the best conditional coverage?

The first question is answered by the ERT (Excess Risk Test), implemented in v0.7.1 as `ConditionalCoverageERT`. ERT trains a LightGBM classifier to predict whether each observation is covered from its features. If coverage is uniform, the classifier gains nothing over predicting the constant `1-alpha` everywhere — ERT is zero. A positive ERT means coverage is predictable from features, which is evidence of conditional miscoverage.

ERT is a hypothesis test. It answers "is there a problem?" not "which predictor has less of it?"

CVI answers the selection question. It produces a continuous, interpretable scalar that can rank competing predictors. If you are deciding between `pearson_weighted`, `lw_pearson`, and `deviance` non-conformity scores, marginal coverage comparison is useless — any valid conformal predictor achieves at least `1-alpha` marginal coverage by construction. CVI is the right criterion.

---

## The CVI decomposition

The paper's key insight is to reframe conditional coverage assessment as a classification problem. Given a calibration set, compute coverage indicators `Z_i = 1{y_i in [lower_i, upper_i]}`. Train a LightGBM classifier on `Z_i ~ X_i` using KFold cross-validation, then apply isotonic calibration to the out-of-fold probability estimates. The result is `eta_hat(x)`: an estimate of the local coverage probability at each point in feature space.

Once you have `eta_hat(x)`, CVI is the mean absolute deviation from the target:

```
CVI = (1/n) * sum_i |eta_hat(X_i) - (1-alpha)|
```

This decomposes exactly into two components with insurance-specific interpretations:

**CVI_U (undercoverage risk):**
```
CVI_U = pi_minus * CMU
```
where `pi_minus` is the proportion of observations with `eta_hat < (1-gamma)*(1-alpha)` and `CMU` is the mean shortfall in that group. This is the component that should concern a pricing actuary first. Systematic undercoverage means prediction intervals are failing for identifiable groups of policyholders — a conduct risk under FCA TCF principles, and a reserving risk if the intervals feed downstream capital calculations.

**CVI_O (overcoverage cost):**
```
CVI_O = pi_plus * CMO
```
where `pi_plus` is the proportion with `eta_hat > (1+gamma)*(1-alpha)` and `CMO` is the mean excess. Overcoverage means intervals are wider than needed for those segments. In a Solvency II context, that is capital held against uncertainty that does not exist at the measured confidence level.

The tolerance parameter `gamma` (default 0.1) defines an acceptable band around the target. Observations with `eta_hat` within 10% of `1-alpha` contribute to neither component. This prevents the metric being dominated by noise in well-calibrated regions.

A predictor with low `CVI_U` and moderate `CVI_O` is one we would choose over a predictor with the reverse — the undercoverage side carries more regulatory and commercial consequence.

---

## CC-Select algorithm

Having a scalar CVI per predictor, selection is straightforward in principle. The complication is that CVI itself requires a data split — you need enough observations in the evaluation set for the reliability estimator to be meaningful. CC-Select handles this with repeated subsampling.

For each of `n_splits` random 50/50 splits of the available calibration data:

1. Calibrate each candidate predictor on the first half.
2. Generate prediction intervals on the second half.
3. Compute CVI for each predictor on the second half.

Select the predictor with the lowest mean CVI across all splits. The 50/50 split ratio is theoretically motivated — it minimises the sum of estimation and approximation errors in the reliability estimator (Theorem 2 in the paper).

The selection is provably consistent: if predictor A has strictly lower true conditional CVI than predictor B, the probability that CC-Select identifies A correctly goes to one as sample size grows (Theorem 4). In practice, the paper recommends a minimum of 800 observations across the combined calibration pool. Below that, the reliability estimator is unreliable.

---

## Code example

```bash
uv add "insurance-conformal[lightgbm]"
```

### Evaluating a single predictor

```python
from insurance_conformal import ConditionalValidityIndex, InsuranceConformalPredictor

# Fit and calibrate your predictor
predictor = InsuranceConformalPredictor(
    model=fitted_gbm,
    nonconformity="pearson_weighted",
    tweedie_power=1.5,
)
predictor.calibrate(X_cal, y_cal)
intervals = predictor.predict_interval(X_test, alpha=0.10)
# intervals is a Polars DataFrame with columns: lower, point, upper

# Evaluate conditional coverage
cvi = ConditionalValidityIndex(gamma=0.1, n_splits=5, random_state=42)
result = cvi.evaluate(
    X_test,
    y_lower=intervals["lower"],
    y_upper=intervals["upper"],
    y_true=y_test,
    alpha=0.10,
)

print(result)
# CVIResult(cvi=0.0312, cvi_u=0.0198, cvi_o=0.0114, ...)

print(f"Undercoverage risk: {result.cvi_u:.4f}  (pi_minus={result.pi_minus:.2%})")
print(f"Overcoverage cost:  {result.cvi_o:.4f}  (pi_plus={result.pi_plus:.2%})")
print(f"Marginal coverage:  {result.marginal_coverage:.3f}")
```

The `result.eta_hat` array contains the estimated local coverage probability for each test observation. Observations in the lower tail of the `eta_hat` distribution are your undercovered segments — cross-reference against features to identify who they are.

### Selecting between competing predictors

```python
from insurance_conformal import CCSelect, InsuranceConformalPredictor

# Candidate predictors — same base model, different non-conformity scores
predictors = {
    "pearson_weighted": InsuranceConformalPredictor(
        model=fitted_gbm, nonconformity="pearson_weighted", tweedie_power=1.5
    ),
    "deviance": InsuranceConformalPredictor(
        model=fitted_gbm, nonconformity="deviance", tweedie_power=1.5
    ),
    "anscombe": InsuranceConformalPredictor(
        model=fitted_gbm, nonconformity="anscombe", tweedie_power=1.5
    ),
}

selector = CCSelect(n_splits=5, split_ratio=0.5, random_state=42)
result = selector.fit(predictors, X_cal, y_cal, alpha=0.10)

print(result.best_predictor)   # e.g. "deviance"
print(result.rankings)         # ascending CVI order

# Full comparison table — Polars DataFrame
print(selector.compare())
# columns: predictor_name, mean_cvi, mean_cvi_u, mean_cvi_o, rank
```

The `compare()` output is the table to include in a validation pack alongside the `coverage_by_decile` diagnostic. The two together give regulators what they need: evidence that coverage is adequate in aggregate, and that conditional coverage differences between model variants were assessed and acted on.

---

## When to use CVI vs ERT

| Question | Tool | Output |
|---|---|---|
| Does this predictor have conditional coverage failures? | `ConditionalCoverageERT` | ERT statistic, bootstrap CI |
| Which of these predictors has the best conditional coverage? | `ConditionalValidityIndex` / `CCSelect` | Scalar CVI, rankings |

The natural sequence: run ERT first to establish whether conditional coverage is a concern for your data. If ERT is positive and material, compare your predictor variants with CVI and use CC-Select to select.

There is no point running CC-Select if you only have one predictor in mind. And ERT alone cannot tell you which predictor to choose — it can only flag that a problem exists.

One practical constraint: both tools require LightGBM and a meaningful feature matrix. If you evaluate against a single scalar predicted value rather than the full covariate set, the reliability estimator has almost nothing to work with and CVI will be near zero regardless of what is actually happening.

---

## What CVI does not fix

CVI measures conditional coverage quality relative to a trained reliability estimator. If the LightGBM classifier cannot distinguish covered from uncovered observations — because the data is too small, the features are uninformative, or coverage genuinely is uniform — CVI will be near zero regardless of what is actually happening in the tails. A near-zero CVI is not the same as confirmed uniform conditional coverage.

The 800-observation minimum from the paper is a practical lower bound, not a guarantee. On a 400-policy commercial book, treat CVI results with appropriate scepticism and use the coarser `subgroup_coverage()` function from v0.7.1 instead, which requires no classifier.

The other caveat is computational. CC-Select with `n_splits=5` and four candidate predictors runs 20 calibrate/predict cycles. For lightweight non-conformity scores this is fast. For spread-model-based variants, expect a meaningful wait on large books. Profile before embedding in automated pipelines.

---

## Installation and references

```bash
uv add "insurance-conformal[lightgbm]"
```

Source: [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal).

The paper: Zhou, Zhang, Tao, Yang — "Conformal Prediction Assessment: A Framework for Conditional Coverage Evaluation and Selection" (arXiv:2603.27189, March 2026).

For the ERT diagnostic tool and the broader conformal prediction framework, see `ConditionalCoverageERT` in v0.7.1 and our earlier post on [conformal prediction intervals for insurance pricing](https://burning-cost.github.io).
