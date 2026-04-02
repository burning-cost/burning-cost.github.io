---
layout: post
title: "Your Conformal Intervals Lie About Some Policyholders"
date: 2026-04-02
categories: [machine-learning, libraries]
tags: [conformal-prediction, conditional-coverage, CVI, consumer-duty, fairness, prediction-intervals, insurance-conformal, LightGBM, undercoverage, Zhou-2026, python]
description: "Marginal 90% coverage can hide severe undercoverage for specific risk profiles. ConditionalCoverageAssessor — new in insurance-conformal v1.2.0 — quantifies it with CVI, decomposes it into undercoverage risk and overcoverage waste, and picks the best predictor via CC-Select. Based on Zhou et al. (arXiv:2603.27189, March 2026)."
author: burning-cost
---

A conformal predictor with 90% marginal coverage achieves its guarantee. The finite-sample theorem is satisfied. Your validation table looks clean.

It does not follow that every segment of your portfolio is covered at 90%. The marginal guarantee is an average. Averages hide distributions. If your youngest-driver decile is covered at 68% and your middle-age decile at 97%, the number washes out to roughly 90% and nobody notices until claims start landing outside the intervals consistently.

This matters in two ways. First, if prediction intervals feed downstream decisions — reserving ranges, capacity allocation, reinsurance pricing — systematic undercoverage for specific cohorts introduces correlated estimation error that your overall coverage metric will not flag. Second, under the FCA's Consumer Duty, pricing tools that produce materially worse outcomes for identifiable customer segments carry conduct risk, regardless of how well they perform in aggregate.

`ConditionalCoverageAssessor` is the tool for measuring this. It is in `insurance-conformal` v1.2.0, alongside the `LCPModelSelector` released in the same version.

---

## Why this is separate from the March 31 release

We [covered `ConditionalValidityIndex` and `CCSelect`](/2026/03/31/conditional-coverage-conformal-prediction-model-selection-cvi/) when they shipped in v0.8.0. Those tools operate with a single-call `evaluate()` pattern: pass in features, actuals, and intervals, get a `CVIResult` back.

`ConditionalCoverageAssessor` is a redesign for production use cases where you want to fit the diagnostic once and query it repeatedly. The LightGBM classifier that estimates local coverage probabilities is the expensive step — O(n_splits × n_eval × n_estimators) — and refitting it for every gamma value or every candidate predictor comparison is wasteful. The fit/query split lets you calibrate once and then explore.

The practical difference:

- `cvi(gamma=0.1)` vs `cvi(gamma=0.05)` vs `cvi(gamma=0.2)` — no refit needed
- `cc_select([pred_a, pred_b, pred_c])` — reuses the stored evaluation data
- `cvp_curve()` — available immediately after `fit()`, without re-running anything

---

## What the CVI decomposition tells you

The Conditional Validity Index measures how much conditional coverage varies from the target. The assessor fits a LightGBM binary classifier — trained to predict whether each observation is covered from its features — using 5-fold cross-validation with isotonic calibration. The out-of-fold probability estimates are `eta_hat(x)`: the local coverage probability at each point in feature space.

CVI is then the mean absolute deviation of those estimates from the target:

```
CVI = (1/n) * sum |eta_hat(X_i) - (1 - alpha)|
```

A CVI of zero means every observation has `eta_hat(x) = 0.9` exactly — uniformly valid intervals. In practice, any real predictor has non-zero CVI. The question is whether it is tolerable.

The decomposition separates the CVI into two components with distinct implications:

**CVI_U = pi_minus × CMU**

`pi_minus` is the proportion of observations where `eta_hat(x) < (1 - gamma) × (1 - alpha)` — outside the acceptable tolerance band on the low side. `CMU` is the mean shortfall: how far below target those observations sit. CVI_U is the product. High CVI_U means a meaningful fraction of the portfolio is being systematically undercovered. That is the safety risk.

**CVI_O = pi_plus × CMO**

`pi_plus` is the proportion above the tolerance band. CMO is the mean excess. CVI_O is overcoverage cost: intervals wider than needed for those segments, which translates to capital held against uncertainty that the model has already accounted for.

The tolerance `gamma` (default 0.1) defines the acceptable band around target. At alpha=0.10, gamma=0.1 means the acceptable zone runs from 81% to 99% coverage. Observations inside that band contribute to neither component. This is deliberately generous — we do not want the decomposition dominated by rounding noise in well-calibrated regions.

---

## FCA Consumer Duty: what "identifiable segment" means here

Consumer Duty requires firms to deliver good outcomes for retail customers. The FCA has been explicit that this includes not just average outcomes but outcomes across different customer groups. If you use prediction intervals to set reserves, capacity limits, or acceptability decisions, and those intervals are systematically less reliable for customers with specific characteristics, that is a potential Duty failure.

The practical risk is not necessarily intentional. A conformal predictor that was calibrated on a portfolio mix that overrepresents standard risks and underrepresents unusual ones will have high `pi_minus` precisely in those underrepresented segments — because the calibration quantile is too tight for them. You have not designed it to fail young drivers. The data mix did it for you.

`eta_hat(x)` gives you the tool to check this before it becomes a problem. Cross-reference the low-`eta_hat` observations against your protected characteristics. If the bottom quintile of `eta_hat` is disproportionately concentrated in a particular age band, occupation class, or postcode cluster, you have a segment-level coverage problem that your aggregate metrics have masked.

There is no regulatory guidance yet on what a "acceptable" CVI_U is for a pricing model. We would treat any `pi_minus > 0.10` at `gamma=0.1` as requiring investigation — that means more than 10% of policyholders have estimated local coverage below 81% on a 90% interval. That is not a marginal failure.

---

## Code example

```bash
uv add "insurance-conformal[lightgbm]"
```

The workflow: fit conformal intervals separately, then assess conditional coverage.

```python
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.assessment import ConditionalCoverageAssessor
import numpy as np

# Fit and calibrate your predictor as usual
predictor = InsuranceConformalPredictor(
    model=fitted_gbm,
    nonconformity="pearson_weighted",
    tweedie_power=1.5,
)
predictor.calibrate(X_cal, y_cal)

# Generate intervals on evaluation set
intervals_df = predictor.predict_interval(X_eval, alpha=0.10)
intervals_np = intervals_df[["lower", "upper"]].to_numpy()

# Fit the assessor — this is the expensive step
assessor = ConditionalCoverageAssessor(alpha=0.10, n_splits=5, random_state=42)
assessor.fit(X_eval, y_eval, intervals_np)

# CVI at default gamma=0.1
result = assessor.cvi(gamma=0.1)
print(result)
# CVIResult(cvi=0.0421, cvi_u=0.0312, cvi_o=0.0109,
#           pi_minus=0.147, pi_plus=0.063, alpha=0.100, gamma=0.100, n_eval=1842)
```

Requery at different gamma values without refitting:

```python
# Tighter tolerance
result_tight = assessor.cvi(gamma=0.05)
print(f"pi_minus at gamma=0.05: {result_tight.pi_minus:.3f}")

# Full decomposition including CMU, CMO, marginal coverage
decomp = assessor.cvi_decomposition()
print(f"CMU (mean shortfall of undercovered obs): {decomp['cmu']:.4f}")
print(f"CMO (mean excess of overcovered obs):     {decomp['cmo']:.4f}")
print(f"Marginal coverage:                        {decomp['marginal_coverage']:.4f}")
```

A result where `marginal_coverage` is close to 0.90 but `cmu` is 0.12 means: overall the intervals hit their target, but the undercovered group misses by 12 percentage points on average. That is a substantial conditional failure.

---

## The CVP curve

`cvp_curve()` returns the empirical CDF of `eta_hat` values. Sorted ascending, plotted against cumulative proportion, this shows the full distribution of local coverage estimates across the portfolio.

```python
import matplotlib.pyplot as plt

proportions, eta_sorted = assessor.cvp_curve()

fig, ax = plt.subplots(figsize=(7, 4))
ax.step(proportions, eta_sorted, where="post", color="#1a4e8c", linewidth=1.5)
ax.axhline(1 - assessor.alpha_, color="#c0392b", linestyle="--", linewidth=1,
           label=f"Target {1 - assessor.alpha_:.0%}")
ax.axhspan(
    (1 - 0.1) * (1 - assessor.alpha_),
    (1 + 0.1) * (1 - assessor.alpha_),
    alpha=0.08, color="grey", label="Tolerance band (gamma=0.1)"
)
ax.set_xlabel("Cumulative proportion of observations")
ax.set_ylabel("Estimated local coverage probability")
ax.set_title("Coverage-Validity Profile (CVP)")
ax.legend()
```

A well-behaved predictor has a CVP curve that is mostly flat, hovering near the target line. The area below the target line in the lower-left portion of the curve integrates to CVI_U. The area above in the upper-right integrates to CVI_O.

A predictor with a long tail dipping below 0.70 in the first 15% of observations has a concentration problem: those 15% are severely undercovered, and the rest of the distribution is doing nothing to fix it. This shape is what you get when the calibration set was drawn from a different risk distribution than the tail you are now being asked to predict.

---

## CC-Select with the stateful interface

The `cc_select()` method evaluates multiple conformal predictors using the evaluation data stored from `fit()`. Predictors are passed as callables — functions that take a feature matrix and return `(n, 2)` interval arrays. This decouples the assessor from any particular predictor class.

```python
def pearson_predictor(X):
    df = predictor_pearson.predict_interval(X, alpha=0.10)
    return df[["lower", "upper"]].to_numpy()

def deviance_predictor(X):
    df = predictor_deviance.predict_interval(X, alpha=0.10)
    return df[["lower", "upper"]].to_numpy()

def anscombe_predictor(X):
    df = predictor_anscombe.predict_interval(X, alpha=0.10)
    return df[["lower", "upper"]].to_numpy()

best_idx = assessor.cc_select([
    pearson_predictor,
    deviance_predictor,
    anscombe_predictor,
])
names = ["pearson_weighted", "deviance", "anscombe"]
print(f"Best predictor: {names[best_idx]}")
```

`cc_select()` runs `n_splits=5` random subsamples of the stored evaluation set, fits a sub-assessor on each, and returns the index of the predictor with the lowest mean CVI across all subsamples. No additional data required.

---

## Limitations and honest caveats

**The classifier can fail silently.** `eta_hat(x)` is only as good as LightGBM's ability to distinguish covered from uncovered observations in feature space. If your evaluation set has 500 observations, a 5-fold CV leaves 100 per fold for training the classifier. With 30 features, that is not much. You will get an `eta_hat` array. It may be largely noise. The library warns at `n_eval < 800` for a reason — treat results below that threshold with genuine scepticism.

**CVI near zero is not confirmation of uniform coverage.** It might mean your predictor genuinely covers uniformly. It might also mean the classifier could not find the structure. These look identical in the output. Run ERT (`ConditionalCoverageERT` from v0.7.1) as a prior diagnostic — it tests whether coverage is predictable from features at all, and its null distribution is interpretable.

**Feature choice matters.** If you pass in only model-predicted values rather than the full feature matrix, the classifier has minimal signal. The assessor does not expose the underlying classifier directly, so LightGBM feature importances are not surfaced in the public API — you need to cross-tabulate `assessor.eta_hat_` against your feature columns yourself to identify which characteristics predict miscoverage.

**`cc_select()` compares CVI, not coverage.** A predictor with slightly higher CVI but lower CVI_U might be the better regulatory choice. `cc_select()` returns the index of the globally lowest-CVI predictor. If the undercoverage decomposition is what you care about, compare `result.cvi_u` across predictors after calling `cvi()` for each.

**Memory cost.** `fit()` stores copies of `X_eval` and `y_eval` internally for `cc_select()`. On a large commercial book with many features, that is non-trivial. If you do not intend to use `cc_select()`, the stored data is wasted. There is currently no option to disable storage.

---

## Installation

```bash
uv add "insurance-conformal[lightgbm]"
# or
uv add "insurance-conformal[lightgbm]"
```

`ConditionalCoverageAssessor` and `CVIResult` are importable from both `insurance_conformal.assessment` and the top-level `insurance_conformal` namespace in v1.2.0.

Source: [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal). PyPI: [pypi.org/project/insurance-conformal](https://pypi.org/project/insurance-conformal/).

The paper: Zhou, Y., Zhang, X., Tao, C., Yang, X. — "Conformal Prediction Assessment" (arXiv:2603.27189, March 2026).

---

Related:
- [Conditional Coverage and Conformal Prediction Model Selection: CVI and CC-Select](/2026/03/31/conditional-coverage-conformal-prediction-model-selection-cvi/) — the v0.8.0 release covering `ConditionalValidityIndex` and `CCSelect`; `ConditionalCoverageAssessor` is the stateful alternative added in v1.2.0
- [Locally Adaptive Score Selection for Conformal Intervals: insurance-conformal v1.2.0](/2026/04/02/lcp-model-selector-insurance-conformal-v120/) — the other v1.2.0 addition: per-prediction score selection for tighter intervals
- [Conformal Prediction Intervals for Insurance Pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the baseline split conformal implementation
- [FCA MS24/1 Pure Protection Review: Occupation Class as Proxy Discrimination](/2026/04/02/fca-ms241-pure-protection-occupation-class-proxy-discrimination/) — related conduct risk context
