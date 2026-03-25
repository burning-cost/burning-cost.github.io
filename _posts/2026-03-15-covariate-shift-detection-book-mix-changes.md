---
layout: post
title: "Monthly Covariate Shift Monitoring: When to Reweight and When to Retrain"
date: 2026-03-15
categories: [pricing, libraries, tutorials]
tags: [covariate-shift, density-ratio, importance-weighting, rulsif, kliep, catboost, conformal-prediction, model-monitoring, book-mix, insurance-covariate-shift, fca, sup-15-3, uk-motor, python, polars]
description: "How to run covariate shift detection as a recurring monthly check: monitoring cadence, ESS ratio trends, and the thresholds that trigger a retraining..."
---

The standard model monitoring workflow is A/E ratios by segment, PSI on score distributions, and a Gini trend. These are useful. They will catch a model that has gone badly wrong. The problem is the lag. A/E ratios need claims development. PSI needs a score distribution on the new book. Gini drift needs exposure to accumulate. By the time your monitoring dashboard is flashing, you have been writing business on a miscalibrated model for six months.

Density ratio estimation catches book mix drift months earlier, because it only needs the feature matrix of risks being scored - no claims labels required. The [foundational post on covariate shift](/2026/03/02/your-book-has-shifted-and-your-model-doesnt-know/) covers the full toolkit: how density ratio estimation works, importance-weighted retraining, and shift-robust conformal intervals. This post is about something different: running the check on a recurring monthly basis, reading the ESS ratio as a trend rather than a one-off diagnostic, and knowing which thresholds should trigger which conversations with governance.

`insurance-covariate-shift` gives you the tools to detect it, quantify it, correct for it, and document it for governance.

```bash
uv add insurance-covariate-shift
```

---

## Why covariate shift degrades a model that was never wrong

A Poisson frequency GBM fitted on Q1-Q4 2026 data learns the correct conditional expectation for the 2026 book. The model is not wrong in any fundamental sense. But if 2027 brings a demographic shift - say, your aggregator partnership starts generating 30% of new business from under-25s, a segment that was 8% of training data - the model's predictions for that segment are based on a thin slice of training data. The predictions may be directionally correct but the uncertainty is higher, and the model's implicit assumption about feature interactions was calibrated on a different distribution.

More concretely: imagine your training book had average driver age 42 with standard deviation 9. Your 2027 book has average age 35 with standard deviation 11. A GBM trained on the 2026 data has seen relatively few 24-year-olds in EVs with maximum NCB. It is extrapolating, not interpolating. Whether it extrapolates sensibly depends on how well the tree splits happen to represent that region.

The density ratio `w(x) = p_target(x) / p_source(x)` makes this precise. For a given risk profile `x`, it says how much more (or less) common that profile is in the current book than in the training book. A value of 3.0 means the model has seen roughly one-third as many similar risks as it needs to; a value of 0.4 means it has seen 2.5x as many as strictly necessary. Policies with `w(x) >> 1` are where the model is most likely to be miscalibrated and where monitoring should focus first.

---

## Detecting the shift: ESS ratio and KL divergence

The `CovariateShiftAdaptor` estimates the density ratio from the two unlabelled datasets. You do not need claims data from the target book. You only need the feature matrix of risks being scored.

```python
import polars as pl
from insurance_covariate_shift import CovariateShiftAdaptor

# Source: the training distribution (your 2026 book)
# Target: what you are scoring today (your 2027 book)
# Neither needs claims labels — this is unsupervised
adaptor = CovariateShiftAdaptor(
    method="catboost",        # handles postcode district, vehicle code natively
    categorical_cols=[3, 4],  # postcode_district, vehicle_abi_group (column indices)
    clip_quantile=0.99,       # cap extreme weights — important for stability
)
adaptor.fit(X_2026, X_2027)

report = adaptor.shift_diagnostic(
    source_label="Direct 2026",
    target_label="Direct + Aggregator 2027",
)
print(report.verdict)
# MODERATE
print(f"ESS ratio: {report.ess_ratio:.2f}")
# ESS ratio: 0.54
print(f"KL divergence: {report.kl_divergence:.3f} nats")
# KL divergence: 0.31 nats
print(report.feature_importance())
# {'driver_age': 0.38, 'postcode_district': 0.29, 'vehicle_age': 0.19, 'ncb': 0.14}
```

The ESS ratio (Effective Sample Size ratio) is the key summary statistic. If all risks in your new book look like risks in your old book, every source observation contributes equally to target-distribution estimates - ESS ratio is 1.0. If the new book is very different, most source observations have low weight, and the ESS ratio falls toward zero. The library's thresholds are:

- ESS >= 0.60 and KL <= 0.10 nats: NEGLIGIBLE. Deploy as-is, standard monitoring.
- ESS 0.30-0.60 or KL 0.10-0.50 nats: MODERATE. Apply importance weighting; monitor closely.
- ESS < 0.30 or KL > 0.50 nats: SEVERE. Retrain before deploying.

An ESS ratio of 0.54 means roughly half your source sample is effectively contributing to target-distribution estimates. The model is usable but biased. The feature importance breakdown tells you where to look: driver age and postcode are doing most of the work. The aggregator channel brought younger drivers from postcodes the direct book had seen rarely.

---

## Choosing the right method

The `method` parameter matters more than the defaults suggest.

**`catboost`** handles the real structure of UK motor data. Postcode district alone has several hundred distinct values. Vehicle make-model combinations run into the thousands. CatBoost's native categorical encoding deals with this without manual preprocessing or target encoding leakage. For most UK personal lines books, this is the right choice.

**`rulsif`** (Relative Unconstrained Least-Squares Importance Fitting) is a kernel method with a closed-form solution. Fast, no hyperparameter grid search needed, and the relative formulation (Yamada et al., 2013) is more numerically stable than the original LSIF. Use it when your feature set is all-continuous - score vectors, rating factors expressed as numerics, demographic indices. It breaks down on high-cardinality categoricals.

**`kliep`** (Kullback-Leibler Importance Estimation Procedure) explicitly enforces the normalisation constraint `E_source[w(x)] = 1`. Slower than RuLSIF and requires bandwidth selection, but useful as a sanity check. If RuLSIF and KLIEP agree on the verdict, you can be confident in it.

For a real motor book with postcodes and vehicle codes, start with `catboost`. If your governance committee wants a second opinion, run `rulsif` on the continuous features only and compare the ESS ratios.

---

## Correcting model evaluation metrics before retraining

The most immediate practical use is correcting the evaluation metrics you already compute. If you are reporting A/E on source-book holdout data to justify continued deployment on the target book, you are showing governance the wrong number.

```python
import numpy as np
from insurance_covariate_shift import CovariateShiftAdaptor

# Fit adaptor on source training data vs. target
adaptor = CovariateShiftAdaptor(method="catboost", categorical_cols=[3, 4])
adaptor.fit(X_train, X_target_current)

# Get importance weights for each source holdout observation
weights = adaptor.importance_weights(X_holdout)

# Standard A/E — what your current monitoring shows
ae_standard = y_holdout.sum() / predicted_holdout.sum()

# Importance-weighted A/E — what the target book looks like
ae_weighted = np.average(y_holdout, weights=weights) / np.average(predicted_holdout, weights=weights)

print(f"Standard A/E:  {ae_standard:.3f}")   # 0.983 — looks fine
print(f"Weighted A/E:  {ae_weighted:.3f}")   # 1.071 — 7% underpriced on the new mix
```

The weighted A/E does not require a single labelled observation from the new book. It reweights the source-book holdout to look like the target distribution. The only inputs to the correction are the feature matrices: what did you train on, what are you scoring now. The claims data is entirely from the source book.

This is the argument to make to your model governance committee: standard holdout performance is a statement about how the model performs on 2026 risks. Importance-weighted holdout performance is a statement about how the model performs on 2027 risks. These are different questions. If your book mix has changed materially, you should be answering the second one.

---

## Conformal intervals that hold on the new distribution

Standard conformal prediction calibrated on source data gives you prediction intervals with guaranteed coverage on the source distribution. If the target distribution differs, the coverage guarantee does not transfer. The intervals may be systematically too narrow for the demographic groups that are overrepresented in the new book.

`ShiftRobustConformal` gives you intervals with a finite-sample coverage guarantee on the target distribution. The default `weighted` method implements Tibshirani et al. (2019): importance-weighted empirical quantile on the calibration set, which inherits the density ratio correction.

```python
from insurance_covariate_shift import ShiftRobustConformal

cp = ShiftRobustConformal(
    model=freq_model,
    adaptor=adaptor,   # pre-fitted CovariateShiftAdaptor
    method="weighted", # Tibshirani 2019 — recommended default
    alpha=0.10,        # target miscoverage: 90% intervals
)
cp.calibrate(X_cal_source, y_cal_source)  # source-book holdout

lower, upper = cp.predict_interval(X_target_current)
```

For teams that want narrower intervals on low-risk profiles and wider on high-risk ones, the `lrqr` method (Marandon et al., 2025) learns a covariate-dependent threshold via likelihood-ratio regularised quantile regression. It requires at least 300 calibration observations and takes longer to fit, but the intervals are more informative. For a personal lines book where aggregate pricing is the priority, `weighted` is sufficient. For a commercial account where per-risk interval width matters to the underwriter, `lrqr` is worth the extra fitting time.

---

## FCA SUP 15.3: what to put in the governance note

Under FCA PRIN 2A.4, insurers must ensure models used in pricing are fit for purpose for the population being scored. SUP 15.3 requires notification of material changes to pricing methodology. Applying a model trained on a materially different book distribution without adjustment could constitute such a change.

The library's `fca_sup153_summary()` is designed to give the factual basis for a governance note - not to write the note for you. It outputs the verdict, ESS ratio, KL divergence, feature attribution, and the recommended action.

```python
print(report.fca_sup153_summary())
```

```
COVARIATE SHIFT DIAGNOSTIC REPORT
Source: Direct 2026 (n=45,231)
Target: Direct + Aggregator 2027 (n=61,840)
Generated: 2026-03-15

VERDICT: MODERATE
ESS ratio: 0.54 (threshold: 0.30 SEVERE, 0.60 NEGLIGIBLE)
KL divergence: 0.31 nats (threshold: 0.50 SEVERE, 0.10 NEGLIGIBLE)

Principal shift drivers:
  driver_age         38% of total shift
  postcode_district  29%
  vehicle_age        19%
  ncb                14%

Recommended action: Apply importance weighting to evaluation metrics.
Retrain within 6 months or when ESS ratio falls below 0.35.
Monitor driver_age and postcode_district distribution monthly.
```

The numbers are the numbers. What you add in the governance note is the actuarial judgement: what caused the shift (aggregator partnership starting Q3 2027), what the commercial implications are (younger, more urban risks are systematically underrepresented in the model's training data), and what the retraining timeline is. The library gives you the quantitative basis; you provide the context.

---

## When not to bother correcting

Importance weighting is not a free lunch. The correction is only as good as the density ratio estimate, and that estimate degrades when the shift is large. An ESS ratio below 0.30 means the weights are high-variance: the correction is directionally useful - it tells you the model is wrong and in which direction - but the corrected metrics should not be treated as precise estimates. Retrain.

The other case where correction adds little value: if the shift is driven entirely by features that do not enter the model. If your claim model uses age, NCB, and vehicle group, and the book mix shift is in a feature you are not modelling (say, payment frequency), then the density ratio correction on your model features will show NEGLIGIBLE and you are done. The shift is real but the model does not need to know about it.

---

## The monitoring cadence we recommend

Run the shift diagnostic monthly, against a fixed reference window (12 months of training data). Plot the ESS ratio over time. It is a single number. You can put it on a dashboard alongside your standard A/E and Gini monitoring. When it drops below 0.60, add importance-weighted metrics to your monitoring pack. When it drops below 0.40, start the retraining conversation with governance. When it hits 0.30, the conversation is over - start the retrain.

The point is not that you abandon the model. The point is that you know, months before the loss ratio catches up, that the model is operating outside its training distribution. That knowledge is worth having.

---

`insurance-covariate-shift` is open source under Apache 2.0 at [github.com/burning-cost/insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift). Requires Python 3.10+, CatBoost 1.2+, and NumPy 1.24+.

- [Your Year-End Large Loss Loading Is a Finger in the Air](/2026/03/14/year-end-large-loss-loading/) - per-segment TVaR loadings via quantile regression, for when the mix shift affects the tail as well as the mean
- [Model Validation Is a Checklist, Not a Test](/2026/03/11/model-validation-pra-ss123/) - building a defensible sign-off process and what the PRA's SS1/23 actually requires
- [Transfer Learning for Thin Segments](/2026/03/10/transfer-learning-for-thin-segments/) - when the target book has sparse data of its own, importance weighting pairs with transfer learning for best results

---

## See also

- [Correcting for Covariate Shift When You Acquire an MGA Book](/2026/03/13/insurance-covariate-shift/) - the library introduction: how density ratio estimation works and the LR-QR method for applying importance weights to conformal prediction intervals
- [Covariate Shift in Motor Pricing: Detection, Correction, and Conformal Intervals](/2026/03/02/your-book-has-shifted-and-your-model-doesnt-know/) - the full motor pricing case study covering detection, correction, and extending conformal intervals to a shifted book
