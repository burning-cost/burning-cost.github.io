---
layout: post
title: "Conformal Prediction Passes 90% Coverage, But Your Young Drivers Get 82%: ConditionalCoverageAssessor Fixes That"
date: 2026-04-05
author: Burning Cost
categories: [conformal-prediction, fairness]
tags: [conformal-prediction, fairness, consumer-duty, conditional-coverage, CVI, insurance-conformal, python, arXiv-2603.27189, model-validation, regulatory]
description: "insurance-conformal v1.3.1 adds ConditionalCoverageAssessor — a tool for detecting and decomposing conditional coverage failures in conformal prediction intervals. Here is the problem it solves, how it works, and how to use it."
math: true
---

Conformal prediction has one genuine guarantee: if the coverage target is 90%, at least 90% of intervals across the portfolio will contain the true value. That guarantee holds. It is also almost entirely beside the point for Consumer Duty purposes.

The problem is that 90% marginal coverage can coexist with 82% coverage for young drivers and 97% coverage for standard risks — simultaneously and without contradiction. The portfolio average is fine. Two specific groups are getting materially different outcomes, and nothing in the standard conformal machinery detects this.

v1.3.1 of [insurance-conformal](https://github.com/burning-cost/insurance-conformal) adds `ConditionalCoverageAssessor` to address exactly this. It implements the Conditional Validity Index framework from Zhou, Zhang, Tao, and Yang (arXiv:2603.27189, 2026), with an API designed for insurance validation workflows: fit once on calibration data, score multiple conformal predictors against the same estimator, pick the best one.

---

## The regulatory problem

Under FCA Consumer Duty, you are required to evidence fair outcomes at the customer level. The FCA's PS22/9 fair value framework and the January 2025 EP25/2 guidance on proxy discrimination both make clear that portfolio-level statistics are not evidence of fair individual outcomes. A pricing model that produces 90% interval coverage on average can still be systematically short-changing young drivers, high-crime-area policyholders, or any other segment the reliability estimator can distinguish from the background population.

This is not hypothetical. Any conformal predictor trained on an imbalanced dataset — or using a nonconformity score that is more error-prone for sparse segments — will produce conditional coverage that varies by subgroup. The marginal guarantee tells you nothing about which segments and by how much.

`ConditionalCoverageAssessor` makes the conditional coverage failure measurable as a single number that decomposes into two interpretable components.

---

## The CVI framework

The core idea from Zhou et al. is to train a binary classifier — a *reliability estimator* — that predicts, from features alone, whether a given observation will be covered by the prediction interval. Under perfect conditional coverage, coverage should be independent of features: the classifier should do no better than predicting the constant $$1 - \alpha$$ everywhere.

If the classifier can beat that baseline — if it finds segments where coverage is systematically higher or lower than the target — those predictions quantify the conditional coverage failure.

Formally, let $$\hat{\eta}(x)$$ be the estimated local coverage probability for observation $$x$$. The Conditional Validity Index decomposes deviations of $$\hat{\eta}$$ from the target $$(1 - \alpha)$$:

$$\text{CVI} = \text{CVI}_U + \text{CVI}_O$$

where:

- **CVI_U** (undercoverage) counts instances where $$\hat{\eta}(x) < (1 - \gamma)(1 - \alpha)$$ — segments receiving materially less coverage than promised. This is the dangerous component: it represents false confidence in the interval for specific risk segments.
- **CVI_O** (overcoverage) counts instances where $$\hat{\eta}(x) > (1 + \gamma)(1 - \alpha)$$ — segments with unnecessarily wide intervals. Inefficient rather than harmful, but still a model selection criterion.

The tolerance parameter $$\gamma$$ defines a band around the target within which coverage is considered "close enough." At $$\gamma = 0.1$$ and $$\alpha = 0.10$$, instances with local coverage between 81% and 99% contribute nothing to CVI.

The source code documentation suggests treating CVI_U > 0.02 at the portfolio level as a trigger for segment-level investigation. The full decomposition — $$\pi^-$$ (fraction of undercovered instances), $$\bar{c}_U$$ (average shortfall per undercovered instance), and their product — tells you both the prevalence and the severity of the problem.

---

## Install

```bash
pip install insurance-conformal
```

No LightGBM dependency. `ConditionalCoverageAssessor` uses scikit-learn's `GradientBoostingClassifier` throughout, which is included in the standard sklearn install.

---

## How to use it

The API separates training from scoring deliberately. Fitting the reliability estimator on calibration data is the expensive step. Once fitted, you can score any number of conformal predictors against the same estimator — each additional predictor requires only a forward pass through the classifier, not a new training run.

```python
import numpy as np
from insurance_conformal.assessment import ConditionalCoverageAssessor

rng = np.random.default_rng(42)
n_cal, n_test = 2000, 800

# --- Simulate calibration data ---
# Suppose a motor portfolio with 4 rating features.
# Coverage is uniform at 90% — no conditional failure.
X_cal = rng.normal(size=(n_cal, 4))
y_cal = rng.exponential(scale=2.0, size=n_cal)
covered_cal = rng.binomial(1, 0.90, size=n_cal).astype(bool)
lower_cal = np.where(covered_cal, y_cal - 1.0, y_cal + 1.0)
upper_cal = np.where(covered_cal, y_cal + 1.0, y_cal - 1.0)

# --- Fit the reliability estimator once ---
assessor = ConditionalCoverageAssessor(
    alpha=0.10,   # 90% coverage target
    gamma=0.10,   # 10% tolerance band
    random_state=0,
)
assessor.fit(X_cal, y_cal, (lower_cal, upper_cal))

# --- Score two candidate conformal predictors on the test set ---
# Predictor A: approximately uniform coverage (the good one)
X_test = rng.normal(size=(n_test, 4))
y_test = rng.exponential(scale=2.0, size=n_test)
covered_a = rng.binomial(1, 0.90, size=n_test).astype(bool)
lower_a = np.where(covered_a, y_test - 1.0, y_test + 1.0)
upper_a = np.where(covered_a, y_test + 1.0, y_test - 1.0)

# Predictor B: systematic undercoverage for X[:,0] > 0
prob_b = np.where(X_test[:, 0] > 0, 0.72, 1.00)
covered_b = rng.binomial(1, prob_b).astype(bool)
lower_b = np.where(covered_b, y_test - 1.0, y_test + 1.0)
upper_b = np.where(covered_b, y_test + 1.0, y_test - 1.0)

# Score each predictor individually
result_a = assessor.score(X_test, y_test, (lower_a, upper_a))
result_b = assessor.score(X_test, y_test, (lower_b, upper_b))

print(result_a)
# CVIAssessmentResult(cvi=0.0031, cvi_u=0.0000 [LOW], cvi_o=0.0031, ...)

print(result_b)
# CVIAssessmentResult(cvi=0.0614, cvi_u=0.0521 [HIGH], cvi_o=0.0093, ...)
```

The CVI_U flag for predictor B is `HIGH` — the `__repr__` applies thresholds directly from the assessment result, so you get an immediate signal in logs and validation reports without post-processing.

---

## Selecting the best predictor

When you have multiple candidate conformal predictors and want the one with the most uniform conditional coverage, use `select()`:

```python
sel = assessor.select(
    X_test,
    y_test,
    {
        "uniform_conformal": (lower_a, upper_a),
        "heteroscedastic_conformal": (lower_b, upper_b),
    }
)

print(sel.best_key)      # 'uniform_conformal'
print(sel.compare())
# shape: (2, 8) — key, cvi, cvi_u, cvi_o, pi_minus, pi_plus, marginal_coverage, rank
```

`compare()` returns a Polars DataFrame sorted by ascending CVI. If you are running a model selection exercise across CQR, split conformal, and Pearson-type nonconformity scores, this gives you a clean ranking with the full decomposition for each candidate.

---

## Interpreting the output

The `CVIAssessmentResult` dataclass contains everything you need for a governance report:

| Field | What it tells you |
|---|---|
| `cvi` | Total conditional coverage deviation — lower is better |
| `cvi_u` | Undercoverage component — the Consumer Duty risk |
| `cvi_o` | Overcoverage component — efficiency loss only |
| `pi_minus` | Fraction of test observations in undercovered segments |
| `cmu` | Average coverage shortfall per undercovered observation |
| `marginal_coverage` | Observed portfolio-level coverage — what standard validation sees |

A model with `marginal_coverage = 0.903` and `cvi_u = 0.000` is a very different regulatory proposition from one with `marginal_coverage = 0.901` and `cvi_u = 0.052`. Both pass the marginal coverage test. Only one passes a Consumer Duty fair outcomes assessment.

The `summary()` method returns a one-row Polars DataFrame suitable for concatenation into a multi-model comparison table:

```python
import polars as pl

comparison = pl.concat([
    result_a.summary().with_columns(pl.lit("uniform").alias("model")),
    result_b.summary().with_columns(pl.lit("heteroscedastic").alias("model")),
])
```

---

## When to use this

**Model validation.** Any conformal pricing model submitted for internal model committee sign-off should include a CVI decomposition alongside the standard marginal coverage check. The two metrics answer different questions: marginal coverage asks "does the portfolio meet the nominal guarantee?", CVI asks "is that guarantee distributed fairly across risk segments?"

**Consumer Duty fair value assessments.** If your pricing uses conformal intervals — for claims costs, for underwriting decisioning, or for pricing uncertainty quantification — CVI_U gives you a specific, quantified metric to include in the fair outcomes evidence pack. It is considerably stronger evidence than "marginal coverage is 90%."

**Conformal predictor selection.** When choosing between nonconformity score functions (CQR vs split conformal vs regularised adaptive), total CVI is a better selection criterion than interval width. A slightly wider interval with CVI_U = 0.001 is preferable to a tighter interval with CVI_U = 0.030 — the latter is producing false confidence for specific customer segments.

**Monitoring.** Fit the reliability estimator on your calibration set at deployment time. In subsequent quarters, call `score()` on the new test data with the same fitted estimator. CVI drift between quarters tells you whether conditional coverage is degrading — a more sensitive signal than marginal coverage drift for detecting segment-specific failures.

---

## A note on sample size

The Zhou et al. paper recommends a minimum of 800 calibration observations for reliable CVI estimation. `ConditionalCoverageAssessor` warns if you pass fewer than 800 to `fit()` and raises an error below 50. For typical UK motor or home books running calibration splits, 800 observations is not a binding constraint. For specialist lines with thin data, treat CVI estimates from smaller calibration sets as directional rather than precise — the ranking of predictors by CVI is more reliable than the absolute CVI values.

---

## Reference

Zhou, Z., Zhang, X., Tao, C. & Yang, Y. (2026). "Conformal Prediction Assessment: A Framework for Conditional Coverage Evaluation and Selection." arXiv:2603.27189.

---

*`ConditionalCoverageAssessor` is in [insurance-conformal v1.3.1](https://github.com/burning-cost/insurance-conformal). The full assessment module, including `CVIAssessmentResult`, `CVISelectResult`, and the `select()` CC-Select implementation, is in `src/insurance_conformal/assessment.py`.*
