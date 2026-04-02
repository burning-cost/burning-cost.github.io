---
layout: post
title: "Locally Adaptive Score Selection for Conformal Intervals: insurance-conformal v1.2.0"
date: 2026-04-02
categories: [machine-learning, libraries]
tags: [conformal-prediction, prediction-intervals, model-selection, tweedie, insurance-conformal, LCP, kernel-localisation, uncertainty-quantification, python, Wang-Wang-2026]
description: "insurance-conformal v1.2.0 adds LCPModelSelector: locally adaptive conformal model and score selection that gives per-prediction tighter intervals while maintaining coverage guarantees. Based on Wang & Wang (arXiv:2602.19284, Feb 2026). 15-35% interval width reduction in experiments."
author: burning-cost
---

Conformal prediction gives you valid intervals regardless of which non-conformity score you choose. The marginal coverage guarantee — P(y in C(X)) >= 1-alpha — holds whether you use raw residuals, Pearson residuals, or deviance residuals. The catch is that interval *width* depends heavily on the score choice, and the right score varies across the feature space.

A Tweedie GLM calibrated on a mix of urban and rural risks has no single optimal score. In the high-frequency urban segment, Pearson residuals are well-calibrated. In the heavy-tailed rural segment, deviance or Anscombe residuals might give tighter intervals. Pick one globally and you are leaving width on the table for half your portfolio.

Wang & Wang (arXiv:2602.19284, February 2026) show how to do per-prediction score selection within the conformal framework while keeping the coverage guarantee intact. The core difficulty is that naively picking the score that gives the shortest interval for each test point and then calibrating on that same score introduces selection bias — coverage collapses. Their construction avoids this by treating the selection symmetrically across calibration points.

`LCPModelSelector` is the implementation. It is in `insurance-conformal` v1.2.0.

---

## The problem with global score selection

`CCSelect`, added in v0.8.0, selects among conformal predictors globally: evaluate K candidate scores on random calibration splits, pick the one with the lowest CVI, apply it uniformly. That is the right approach if you want to optimise conditional coverage quality (and we wrote about it [here](/2026/03/31/conditional-coverage-conformal-prediction-model-selection-cvi/)).

`LCPModelSelector` asks a different question. Not "which score is globally best?" but "which score is locally best at this specific test point?" The globally best score for your full motor portfolio might not be the best score for a 23-year-old with a modified vehicle in Birmingham.

The gains from local selection are meaningful: Wang & Wang report 15-18% interval width reduction over the best single model in their parametric experiments (n=200-1000, 10 candidate models), rising to 25-35% in nonparametric settings with Nadaraya-Watson estimators. The gains vanish at high noise — when models are nearly indistinguishable, there is nothing for local selection to exploit. In heterogeneous portfolios with identifiable risk sub-segments, the reduction is real.

---

## How LCP-MS works

The algorithm is built on leave-one-out (LOO) surrogate intervals. At calibration time, for every combination of calibration point i and model k, the selector computes an approximate LOO prediction interval — what interval model k would give for point i using all other calibration points. These are the *validity proxies*: they tell us which models are covering the calibration labels well in the local neighbourhood of each test point.

At prediction time for a test point x:

1. Compute kernel weights from x to all calibration points. Points close to x get high weight; distant points get low weight.
2. For each model k, compute the weighted non-coverage fraction on the calibration LOO intervals. A model k is *locally valid* at test point x if its weighted non-coverage is at most alpha.
3. Among locally valid models, select the one with the narrowest interval for x.
4. If no model passes the validity check (degenerate case with tiny calibration sets or very tight alpha), fall back to the globally narrowest model with a warning.

The coverage guarantee holds because the selection is performed symmetrically — every calibration point i acts as a "test point" in the LOO construction, so the selection mechanism is exchangeable with the test observation.

---

## The rank-update trick

The LOO computation is the expensive part. For n calibration points and K models, brute-force LOO is O(K·n²) — removing each calibration point and recomputing the quantile from scratch. At n=2,000 and K=4 that is 16 million quantile computations.

For n > 1,000, `LCPModelSelector` uses an approximate LOO via rank-update. The key observation: removing score s_i from n sorted scores shifts the (1-alpha) quantile rank by at most 1. A binary search over the sorted score array gives the LOO quantile in O(log n) per calibration point. Total calibration cost drops to O(K·n·log n) — fast enough for production calibration sets of 5,000 to 10,000 policies.

For n <= 1,000, the brute-force path is used (O(K·n²) is fine at small n).

---

## Four kernel localisers

The local neighbourhood around a test point is defined by a kernel. Four options:

- **`gaussian`** (default): exp(-||x-x_i||² / 2h²). Smooth decay; most calibration points contribute with decreasing weight. Works well when risks vary continuously across covariate space.
- **`exponential`**: exp(-||x-x_i|| / h). Heavier tails than Gaussian; gives more weight to moderately distant points. Useful if your covariate space has pockets of unusual risk.
- **`uniform`**: indicator ||x-x_i|| <= h. Only calibration points within bandwidth h contribute. Equivalent to a hard neighbourhood. Falls back to equal-weight globally if no calibration points are within h.
- **`knn`**: uniform weight over the k nearest neighbours. Ignores the bandwidth parameter; useful when you want a fixed sample size for the local validity check rather than a fixed distance.

The bandwidth h (or k for knn) is a hyperparameter. The Wang & Wang paper suggests h = n^{-1/(d+4)} as a rule of thumb for Gaussian kernels, where d is the feature dimension. In practice, cross-validate on coverage efficiency if the portfolio has clear spatial structure; otherwise the default of 1.0 is a reasonable starting point for standardised features.

---

## Code example: three Tweedie scores on a single model

The most common insurance use case is not comparing different models — it is comparing different score functions applied to the same model. You have a Tweedie GLM. Should you use Pearson residuals, deviance residuals, or Anscombe residuals? The answer may depend on where in the risk distribution the test point sits.

`LocalScoreSelector` wraps this case as a one-liner:

```python
from insurance_conformal import LocalScoreSelector

# fitted_glm: a scikit-learn-compatible model with .predict()
# X_train, y_train: training data
# X_cal, y_cal: held-out calibration data
# X_test: test data

lss = LocalScoreSelector(
    model=fitted_glm,
    score_types=["pearson_weighted", "deviance", "anscombe"],
    tweedie_power=1.5,
    localizer="gaussian",
    bandwidth=2.0,
)

lss.calibrate(X_cal, y_cal)
intervals = lss.predict_interval(X_test, alpha=0.10)

print(intervals.head())
# shape: (n_test, 5)
# columns: lower, point_estimate, upper, ensemble_width, n_models_in_set
```

To see which score was selected for each test point:

```python
selection = lss.selected_models(X_test, alpha=0.10)
print(selection["score_name"].value_counts())
# e.g.:
# pearson_weighted    612
# deviance            287
# anscombe            101
```

On a heterogeneous portfolio, you would expect a spread across scores. If one score wins on nearly all points, the bandwidth may be too wide (everything looks locally the same) or the scores are genuinely equivalent on this data.

---

## Code example: three distinct Tweedie models

If you have fitted multiple models — for example, Tweedie(p=1.0), Tweedie(p=1.5), and Tweedie(p=2.0) — `LCPModelSelector` selects among them per prediction point:

```python
from sklearn.linear_model import TweedieRegressor
from insurance_conformal import LCPModelSelector

# Fit three models with different power parameters
m_poisson = TweedieRegressor(power=1.0, link="log").fit(X_train, y_train)
m_compound = TweedieRegressor(power=1.5, link="log").fit(X_train, y_train)
m_gamma    = TweedieRegressor(power=2.0, link="log").fit(X_train, y_train)

lcp = LCPModelSelector(
    models=[m_poisson, m_compound, m_gamma],
    score_fns=["pearson_weighted", "pearson_weighted", "pearson_weighted"],
    localizer="gaussian",
    bandwidth=1.5,
    tweedie_power=1.5,   # used for score inversion; set to your expected p
)

lcp.calibrate(X_cal, y_cal)
intervals = lcp.predict_interval(X_test, alpha=0.10)
```

For multi-model selection, check `coverage_diagnostics()` to verify that the local selection is actually distributing across models and that each model's selected-for segment achieves coverage close to 1-alpha:

```python
diag = lcp.coverage_diagnostics(X_test, y_test, alpha=0.10)
print(diag.filter(diag["slice_type"] == "by_model"))
# columns: slice_type, bin, mean_predicted, n_obs, coverage, target_coverage, dominant_model
```

A model that is only selected for 3% of test points but achieves 88% coverage on those points (target: 90%) is worth investigating — that segment may be systematically harder than the calibration set suggested.

---

## Relationship to CCSelect

`LCPModelSelector` and `CCSelect` (v0.8.0) are complementary, not competing.

`CCSelect` answers: among K candidate scores, which has the best conditional coverage? It selects globally, once, at configuration time. The selected score is applied uniformly. If you want a single score that performs consistently well across your portfolio's conditional structure, `CCSelect` is the right tool.

`LCPModelSelector` answers: for this specific test point, which score gives the tightest valid interval? It selects locally, per prediction. If your portfolio is heterogeneous enough that no single score dominates everywhere, local selection reduces interval width.

The natural composition: use `CCSelect` to narrow from ten candidate scores to three credible candidates. Use `LCPModelSelector` to select among those three locally at prediction time. You get conditional coverage quality from the global selection and interval efficiency from the local selection.

---

## Coverage guarantees and their limits

The finite-sample marginal coverage guarantee — P(Y_{n+1} in C(X_{n+1})) >= 1-alpha — holds under i.i.d. exchangeability. The same conditions as standard split conformal. LCP-MS does not weaken the guarantee; it inherits it from the symmetric LOO construction.

What it does not guarantee: conditional coverage at any particular x. Like all conformal methods without explicit conditional corrections, coverage can vary across the feature space. Use `coverage_diagnostics()` to check that coverage is not systematically below target in any predicted-value decile. If it is, that is a sign that the calibration set does not represent the test distribution in that region.

The fallback behaviour when no locally valid model is found — fall back to the globally narrowest model — preserves the coverage guarantee but issues a `UserWarning`. If you see this warning frequently in production, it typically means the bandwidth is too tight (the local neighbourhood is so small that the LOO validity check is noisy) or the calibration set is too small for the feature dimensionality.

---

## Installation

```bash
uv add insurance-conformal
```

`LCPModelSelector` and `LocalScoreSelector` are available from the top-level import in v1.2.0:

```python
from insurance_conformal import LCPModelSelector, LocalScoreSelector
```

Python 3.10+. Dependencies: numpy, polars, scikit-learn.

Source: [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal).

The paper: Wang, Y. & Wang, T. — "Localized conformal model selection" (arXiv:2602.19284, February 2026, Cambridge).

---

Related:
- [Conditional Coverage and Conformal Prediction Model Selection: CVI and CC-Select](/2026/03/31/conditional-coverage-conformal-prediction-model-selection-cvi/) — the global score selection tool that LCPModelSelector complements
- [Conformal Prediction Intervals for Insurance Pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the baseline split conformal implementation these tools extend
- [Conformal Prediction for Non-Exchangeable Claims Time Series](/2026/03/15/conformal-prediction-for-non-exchangeable-claims-time-series/) — handling temporal structure in the calibration set
