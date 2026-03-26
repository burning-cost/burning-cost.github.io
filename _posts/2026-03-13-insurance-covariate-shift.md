---
layout: post
title: "Correcting for Covariate Shift When You Acquire an MGA Book"
date: 2026-03-13
categories: [libraries, model-validation, pricing]
tags: [covariate-shift, density-ratio, conformal-prediction, catboost, m-and-a, fca, python]
description: "Correct covariate shift when acquiring an MGA book for UK motor pricing. Importance weighting, density ratio estimation, segment-level diagnostics - Python."
---

When a UK motor insurer acquires an MGA book, something predictable happens to the pricing model. Nobody notices immediately. The GBM loads without error. Predictions come out. Premiums get issued. Then, six months later, the loss ratio on the acquired cohort is 12 points worse than expected, and someone starts asking questions.

The model didn't break. It's doing exactly what it was trained to do: estimate expected loss cost for the distribution it was fitted on. The problem is that the acquired book is a different distribution. The model doesn't know that. It has no way of knowing.

This is covariate shift. The conditional relationship P(Y|X) -- frequency and severity given risk characteristics -- may be identical across both books. The marginal distribution P(X) is not. The acquired MGA skews older vehicles (mean age 9 years versus 5), more rural postcodes (55% South West and Wales versus 35%), and 40% telematics opt-in versus 12%. The source GBM was trained on sparse data in exactly the regions of feature space that are dense in the target book. It extrapolates where it should interpolate. The errors are systematic, not random.

We built [`insurance-covariate-shift`](https://github.com/burning-cost/insurance-covariate-shift) to correct this.

---

## Why the standard toolkit doesn't help

The usual responses to this problem are wrong in instructive ways.

**Platt scaling** fits a logistic regression on top of the model's outputs to correct marginal miscalibration. It will fix the overall loss ratio. It will not fix the segment-level bias. If the model is underestimating frequency for rural older vehicles specifically, a global rescaling makes the urban younger-vehicle segment worse while partially correcting the rural segment. You have redistributed the error, not removed it.

**PSI monitoring** (population stability index) will tell you that the book has shifted. AUC of 0.87 between source and target covariate distributions. That is useful but it stops at diagnosis. PSI detects; it doesn't correct.

**Full retraining** on the acquired book is the right answer eventually, but it requires target labels -- actual claims experience from the acquired portfolio. That takes 12 to 18 months to accumulate post-acquisition. You cannot wait that long to have valid predictions. You are pricing renewals at month three.

**Transfer learning** (our `insurance-thin-data` library) is a different tool: it adapts model parameters using some labeled target data. Useful when you have six months of experience and want to fine-tune. Not useful when you have zero claims from the acquired book and need to price now.

The gap this library fills is correction from covariates alone, before any target labels exist.

---

## The density ratio approach

The fix is conceptually clean. Define the importance weight:

```
w(x) = p_target(x) / p_source(x)
```

This is the density ratio between the target and source covariate distributions. For any region of feature space, w(x) tells you how much more (or less) common that region is in the target book relative to the source. Reweighting source predictions by w(x) makes them unbiased on the target distribution, under the covariate shift assumption.

Crucially, estimating w(x) requires only covariate data -- no loss labels. Pool your source and target covariate matrices, label them 0 and 1, train a classifier, extract class probabilities. The density ratio follows directly:

```
w(x) = (n_target / n_source) * P(target|x) / P(source|x)
```

The library supports three estimation methods. CatBoost classifier is the default and the one you should use for UK motor data. RuLSIF (relative unconstrained least-squares importance fitting) is the alternative for portfolios where all features are continuous. KLIEP is included as a reference implementation; it works but is slower with no material accuracy advantage.

---

## The postcode problem

UK motor data breaks kernel-based density ratio methods.

One-hot encoding 2,000 postcode districts creates a 2,000-dimensional binary vector. A Gaussian RBF kernel treats SW1A and SW1B as equally distant from LE1. The Gram matrices become near-singular. Bandwidth cross-validation produces meaningless results. RuLSIF and KLIEP both fail silently -- they converge to plausible-looking numbers that are nonetheless wrong.

CatBoost handles this natively. It processes postcode_district, vehicle_make, and occupation_code directly via ordered target statistics, learning that SW1A and SW1B are similar (both high-density London) and that rural Scottish postcodes cluster together. No encoding required. No geographic data required. The classifier learns the geographic structure from the data.

This is not a minor convenience feature. It is the reason the library exists. Every existing Python package for density ratio estimation -- densratio, ADAPT, SKADA -- requires one-hot encoding and therefore cannot handle postcode geography. They are also largely unmaintained: densratio 0.3.0 last updated October 2022, pykliep last updated 2019. ADAPT requires TensorFlow as a hard dependency -- a 1.2 GB install for an actuarial library -- which is not acceptable.

---

## Using the library

```python
from insurance_covariate_shift import CovariateShiftAdaptor, ShiftRobustConformal

# Density ratio estimation
adaptor = CovariateShiftAdaptor(
    method='catboost',
    categorical_cols=['postcode_district', 'vehicle_make', 'occupation_code'],
    exposure_col='exposure_years'
)
adaptor.fit(X_source, X_target)

weights = adaptor.importance_weights(X_source)
report = adaptor.shift_diagnostic()
print(report.verdict)  # 'MODERATE'
report.plot(kind='feature_psi')
```

`fit()` trains the CatBoost classifier on the pooled covariate matrix. `importance_weights()` returns w(x) for any feature matrix, clipped to prevent extreme values at distribution boundaries. `shift_diagnostic()` returns a `ShiftDiagnosticReport` with per-feature PSI, the classifier AUC, effective sample size ratio, and a three-level verdict.

The AUC is your primary diagnostic. AUC near 0.5 means the distributions are similar -- importance weighting will make a small difference. AUC near 1.0 means complete separation: the two books have essentially non-overlapping feature distributions. Importance weighting cannot recover valid predictions in that case; you need to retrain, and the ShiftDiagnosticReport will tell you so.

The effective sample size (ESS) ratio quantifies the concentration of weights:

```
ESS = (sum w_i)^2 / sum(w_i^2)
```

If ESS falls below 30% of n_source, a handful of observations are carrying most of the weight. Predictions on the weighted source sample are unreliable. SEVERE verdict. The library will not silently proceed and hand you a number with false precision.

---

## Prediction intervals on the target distribution

Correcting point predictions is half the problem. The other half is quantifying uncertainty under covariate shift.

Standard conformal prediction (split conformal, weighted conformal) relies on the exchangeability assumption: calibration and test data come from the same distribution. Under covariate shift, the calibration set comes from the source distribution and the test set comes from the target. The assumption is violated. The intervals are not valid.

Tibshirani et al. (2019) showed you can fix this by computing a weighted quantile of calibration scores, where weights are the importance weights w(x). This works when density ratio estimation is reliable -- that is, when the feature space is low-dimensional and all features are continuous. For UK motor data with postcodes and vehicle models, the density ratio estimates are too noisy for the weighted quantile to produce valid coverage.

LR-QR (arXiv:2502.13030, 2025) sidesteps this. Instead of estimating the density ratio explicitly and plugging it into a weighted quantile, it learns a threshold function h(x) directly, with implicit likelihood-ratio regularisation. The objective is:

```
L(h, beta) = E_source[pinball_alpha(h(X), S(X,Y))]
           + lambda * (E_source[beta^2 * h(X)^2] - E_target[2 * beta * h(X)])
```

The regularisation term -- the difference of source and target expectations -- is minimised when h(x) aligns with the density ratio through one-dimensional projections accessible from sample averages, without requiring full density ratio estimation in high dimensions. The prediction set is then C(x) = {y : S(x,y) <= h(x)}.

There is no existing Python implementation of LR-QR anywhere. We implement it from scratch.

```python
# Shift-robust conformal intervals
src = ShiftRobustConformal(
    model=fitted_model,
    method='lrqr',
    adaptor=adaptor,
    alpha=0.1
)
src.calibrate(X_cal, y_cal, X_target_unlabeled=X_target)
lo, hi = src.predict_interval(X_target_new)
```

`calibrate()` uses three data sources: the labeled calibration set from the source (X_cal, y_cal), unlabeled target covariates (X_target), and optionally unlabeled source covariates for the regularisation normalisation. If you only have X_cal from the source side, the library falls back to using it for both -- slightly less statistically efficient, but practical.

The coverage theorem (Theorem 4.3 of arXiv:2502.13030) gives an approximate guarantee: P_target[Y in C(X)] >= 1 - alpha - E_cov, where E_cov vanishes as sample sizes grow. For the finite-sample case, the error term depends on the regularisation strength lambda, which the library selects at the theoretical rate lambda* proportional to n^(-1/3).

---

## FCA and model governance

A UK motor insurer applying an existing model to a materially different acquired book is not straightforwardly compliant with FCA SUP 15.3. The firm's underwriting risk profile has changed. The pricing model's domain of validity has shifted. This should appear in model governance documentation and, depending on materiality, in a notification to the regulator.

The `ShiftDiagnosticReport` produces a structured FCA SUP 15.3 summary: which features shifted and by how much, overall distribution divergence, effective sample size, and a recommended action per verdict tier.

**MINOR** (ESS >= 70%, AUC <= 0.80): Standard monitoring. Importance weighting corrects predictions adequately. Document in model governance log.

**MODERATE** (30% <= ESS < 70% or 0.80 < AUC <= 0.95): Document in model governance. Quarterly review. Apply importance weighting and flag for revalidation at 12 months when target labels are available.

**SEVERE** (ESS < 30% or AUC > 0.95): Material change likely. Notify FCA. Importance weighting is insufficient -- the distributions do not overlap enough for correction to be valid. Consider whether the acquired book should be priced on a different model until target experience accumulates.

Consumer Duty (PS22/9) adds a second angle. Applying a model trained on direct business to an aggregator book creates systematic mispricing that is not intentional but produces discriminatory pricing outcomes. The ShiftDiagnosticReport documents that the issue was identified and addressed -- which is the governance response Consumer Duty requires.

---

## What this doesn't solve

Covariate shift assumes P(Y|X) is the same across source and target. In insurance, this assumption can fail. If the acquired MGA has different claims handling (faster settlement, different fraud detection, more generous reserve releases), then the conditional distribution of losses given features differs between books even for identical risks. That is not covariate shift; it is concept drift. This library cannot correct for it.

The diagnostic reports the features that shifted most strongly. If the shift is concentrated in features that are closely tied to claims handling rather than genuine risk characteristics, that is a signal that the covariate shift assumption may not hold. We flag this as HIGH_LEAKAGE_RISK in the report but cannot resolve it programmatically. That requires actuarial judgement.

---

## Install

```bash
uv add insurance-covariate-shift
```

The default installation pulls CatBoost and scikit-learn. RuLSIF support is included in the base install. No TensorFlow. No PyTorch.

```bash
uv add "insurance-covariate-shift[diagnostics]"
```

Adds matplotlib and the report plotting methods.

The library is on GitHub at [burning-cost/insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift). The LR-QR implementation is the part we expect the most interest in -- it is a 2025 method with no prior Python implementation and direct applicability to anyone running conformal intervals on a shifted distribution.

---

**Related reading:**
- [Why k-Fold CV Is Wrong for Insurance](/2026/03/21/why-k-fold-cv-is-wrong-for-insurance/) - the related problem of temporal leakage in model evaluation; covariate shift and temporal leakage are the two main reasons a model's training accuracy exceeds its live accuracy
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) - distribution-free prediction intervals; the LR-QR method in insurance-covariate-shift extends this to the shifted-distribution case
- [Three-Layer Pricing Model Monitoring: What PSI and A/E Ratios Miss](/2026/03/03/your-pricing-model-is-drifting/) - ongoing monitoring for the performance degradation that covariate shift causes in deployed models

---

## See also

- [Covariate Shift in Motor Pricing: Detection, Correction, and Conformal Intervals](/2026/03/02/your-book-has-shifted-and-your-model-doesnt-know/) - a fuller treatment covering all three stages: detecting the shift, applying importance weighting, and extending conformal intervals to the shifted distribution
- [Monthly Covariate Shift Monitoring: When to Reweight and When to Retrain](/2026/03/15/covariate-shift-detection-book-mix-changes/) - the operational monitoring question: how often to run the diagnostic, what ESS thresholds trigger action, and when retraining is the only answer
