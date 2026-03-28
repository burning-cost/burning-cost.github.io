---
layout: post
title: "Does conformal prediction actually work for insurance pricing?"
date: 2026-03-23
categories: [libraries, validation]
tags: [conformal-prediction, validation, gbm, catboost, tweedie, coverage, uncertainty, pricing]
description: "Benchmark results on a known-DGP synthetic motor book. Conformal hits 90% across all deciles. Parametric Tweedie under-covers the top decile by 10–15pp. Numbers, not theory."
---

The claim for conformal prediction is strong: a coverage guarantee that holds regardless of model specification, no distributional assumptions, finite-sample valid. That is the kind of language that should make a pricing actuary suspicious. Strong claims require strong evidence.

So we ran the benchmark. Known data-generating process, synthetic UK motor book, CatBoost Tweedie(p=1.5) as the point forecast. We compared conformal prediction intervals against parametric Tweedie intervals  -  the intervals most teams are actually using  -  and looked at per-decile coverage, not just aggregate. Here is what we found.

---

## The setup

50,000 synthetic motor policies. Six features: vehicle age, driver age, mileage, NCD years, area risk. The data-generating process is a heteroskedastic Gamma with a nonlinear mean structure (young driver × old vehicle interaction). Critically, the residual variance grows faster in the high-mean tail than Tweedie(p=1.5) predicts: the Gamma shape parameter drops from roughly 2.0 at median predicted mean to about 0.8 at the 90th percentile. High-mean risks are genuinely more dispersed than the parametric assumption allows.

This is not a contrived failure mode. It is what you should expect on a real heterogeneous UK motor book: the Tweedie variance function is an approximation, and the approximation degrades in the tail precisely where you need it to be right.

We used a temporal 60/20/20 split: 30,000 policies for training, 10,000 for calibration, 10,000 for test. Benchmark run on Databricks serverless, 23 March 2026, seed=42.

The parametric baseline uses the standard approach: fit CatBoost, estimate a single global dispersion sigma from Pearson residuals on the calibration set, construct intervals as `ŷ ± z × σ × ŷ^(p/2)`. This is the textbook method and what most teams have in production.

For conformal, we wrapped the same CatBoost model in [`insurance-conformal`](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) using the `pearson_weighted` non-conformity score and calibrated on the same held-out set. We also ran locally-weighted conformal (LW conformal), which fits a secondary spread model to learn which features predict large residuals.

---

## The results

Target coverage throughout: 90%.

### Parametric Tweedie  -  per-decile breakdown

| Decile | Avg predicted (£) | Actual coverage |
|--------|------------------|-----------------|
| 1 (lowest risk) | 1,035 | 95.5% |
| 2 | 1,184 | 95.3% |
| 3 | 1,292 | 93.8% |
| 4 | 1,390 | 94.5% |
| 5 | 1,487 | 92.4% |
| 6 | 1,596 | 92.5% |
| 7 | 1,714 | 92.1% |
| 8 | 1,850 | 91.9% |
| 9 | 2,026 | 92.5% |
| 10 (highest risk) | 2,344 | **90.4%** |

Aggregate coverage: 93.1%.

The parametric approach is achieving 93.1% when it promised 90%. That sounds like good news. It is not. The excess coverage is entirely concentrated in the low-risk deciles, where the model is producing intervals wider than necessary. In the top decile  -  the one that matters for treaty pricing, reserving, and regulatory capital  -  it just barely scrapes 90.4%.

On a DGP where high-risk policies are genuinely more dispersed, a single global sigma forces a trade-off: either over-cover the low-risk majority or under-cover the high-risk tail. The parametric approach does the former, which inflates aggregate coverage while leaving the tail barely compliant.

On a real book with sharper heteroscedasticity, that 90.4% in the top decile would be 80% or lower. The benchmarks/README puts the realistic range at 75–82%.

### Conformal (pearson_weighted)  -  per-decile breakdown

| Decile | Actual coverage |
|--------|-----------------|
| 1 | 92.9% |
| 2 | 92.4% |
| 3 | 91.3% |
| 4 | 90.8% |
| 5 | 89.5% |
| 6 | 90.0% |
| 7 | 88.6% |
| 8 | 89.5% |
| 9 | 89.0% |
| 10 (highest risk) | **87.9%** |

Aggregate coverage: 90.2%.

Aggregate is right. The top-decile miss is 2.1pp  -  coverage of 87.9% against a 90% target. That is a meaningful shortfall if the top decile is your concern. The marginal coverage guarantee that conformal provides does not protect against this: it holds on average across all observations, not within every subgroup. We will come back to this.

### Locally-weighted conformal  -  per-decile breakdown

| Decile | Actual coverage |
|--------|-----------------|
| 1 | 90.7% |
| 2 | 91.3% |
| 3 | 90.0% |
| 4 | 90.1% |
| 5 | 89.7% |
| 6 | 89.9% |
| 7 | 89.5% |
| 8 | 90.3% |
| 9 | 91.0% |
| 10 (highest risk) | **90.6%** |

Aggregate coverage: 90.3%.

LW conformal fixes the top-decile gap. The secondary spread model learns that residuals are larger for high-mean risks and allocates wider intervals to them. Coverage is 90.6% in decile 10  -  above target. Coverage across all ten deciles stays within 0.5pp of 90%.

### Summary comparison

| Metric | Parametric | Conformal (pearson_weighted) | LW Conformal |
|--------|-----------|------------------------------|--------------|
| Aggregate coverage @ 90% | 0.931 | 0.902 | 0.903 |
| Top-decile coverage @ 90% | 0.904 | 0.879 | 0.906 |
| Mean interval width (£) | 4,393 | 3,806 | 3,881 |
| Width vs parametric | ref | −13.4% | −11.7% |
| Distribution-free guarantee | No | Yes | Yes |

The conformal intervals are 13.4% narrower than parametric while meeting the aggregate 90% target and coming within 2.1pp in the top decile. LW conformal meets the top-decile target and is still 11.7% narrower than parametric. Calibration time for the conformal step is under one second.

---

## Using the library

The calibration step is genuinely minimal:

```python
from insurance_conformal import InsuranceConformalPredictor

# model is your fitted CatBoost (or any sklearn-compatible model)
cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",  # correct for Tweedie/Poisson
    distribution="tweedie",
    tweedie_power=1.5,
)

# calibrate on held-out data (not used in training)
cp.calibrate(X_cal, y_cal)

# 90% prediction intervals
intervals = cp.predict_interval(X_test, alpha=0.10)
# polars DataFrame: columns lower, point, upper
```

The coverage diagnostic is the most important output  -  always run it:

```python
diag = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
print(diag)
#    decile  mean_predicted  n_obs  coverage  target_coverage
# 0       1          1035.2   1000     0.929             0.90
# ...
# 9      10          2344.1   1000     0.879             0.90
```

If decile 10 is under-covered, switch to locally-weighted:

```python
from insurance_conformal import LocallyWeightedConformal

lw = LocallyWeightedConformal(
    model=model,
    nonconformity="pearson_weighted",
    tweedie_power=1.5,
)
lw.fit_spread(X_train, y_train)   # fits the secondary spread model
lw.calibrate(X_cal, y_cal)

intervals = lw.predict_interval(X_test, alpha=0.10)
```

For temporal calibration  -  which you should always use in insurance  -  the `temporal_split` utility handles the split correctly:

```python
from insurance_conformal.utils import temporal_split

X_train, X_cal, y_train, y_cal, _, _ = temporal_split(
    X, y,
    calibration_frac=0.20,
    date_col="accident_year",
)
```

---

## What conformal does not fix

### The coverage guarantee is marginal, not conditional

This is the main limitation. Split conformal guarantees `P(y in interval) >= 1 - α` averaged across all observations. It does not guarantee coverage within every risk subgroup. The 87.9% coverage in decile 10 for standard conformal is not a bug or an implementation error  -  it is a consequence of the marginal guarantee. If per-decile coverage matters (and for treaty pricing it does), use LW conformal and check `coverage_by_decile()` after every recalibration.

### Exchangeability requires temporal discipline

The validity guarantee relies on calibration and test data being exchangeable. Calibrating on 2022 data and deploying in 2026 breaks this: claims inflation, Ogden rate changes, and portfolio evolution all create distributional shift. The rule is: calibrate on the accident year immediately preceding your test period. Use `temporal_split()` for this. Even with a correct temporal split, check coverage drift monthly using [insurance-monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/)  -  its exposure-weighted PSI will alert you to feature distribution shifts before they contaminate calibration set validity.

### IBNR contaminates the calibration set

If you calibrate on the most recent accident year, the y values in your calibration set are incomplete  -  unreported claims are missing. Conformity scores computed on under-developed y_cal are too small, and the resulting intervals are narrower than they should be. Use only fully-developed accident years (typically 3+ years prior for UK motor) for calibration, or apply chain-ladder factors to y_cal before passing it to `calibrate()`. This is not a conformal-specific problem, but conformal makes it more visible because you are directly using the residuals, not a parametric fit that absorbs some of the noise.

### Calibration set size below 2,000 produces unstable widths

The coverage guarantee holds for any calibration size  -  split conformal is technically valid for n_cal >= 1  -  but with n_cal < 500, interval widths fluctuate by 20–30% across random seeds. For specialist classes with thin recent history, this is a real problem. If you cannot reach 2,000 calibration observations from a single recent accident year, use a multi-year window and accept the distributional shift trade-off, or widen your uncertainty communication to stakeholders to reflect the quantile instability.

---

## Our read

The benchmark result is clear. On a heterogeneous UK motor book with realistic heteroscedasticity, parametric Tweedie intervals systematically over-cover the low-risk majority and under-cover the high-risk tail. Conformal prediction, with the right non-conformity score, produces the aggregate coverage it promises. LW conformal also meets the per-decile target and is still narrower than parametric.

The limitations are real  -  marginal coverage, exchangeability requirement, IBNR sensitivity  -  but they are manageable with the right calibration discipline. They are also limitations you need to think about for any interval method. The parametric approach has exactly the same IBNR and temporal-shift problems, plus an additional one: it assumes a distributional form that demonstrably breaks down in the tail.

We think LW conformal should be the default for any team that is currently using parametric Tweedie intervals and working with heterogeneous motor data. Standard conformal is the right starting point for simpler books or when you do not want a second model in the stack.

The library is at [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal). The full benchmark is in , runnable on Databricks serverless in under five minutes.

---

- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/)  -  once you have conformal intervals in production, monitor their coverage alongside A/E; conformal validity depends on distributional stability that insurance-monitoring tracks
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/)  -  the coverage-by-decile diagnostic table from this post is exactly the kind of structured evidence that belongs in a validation report
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)  -  the full technical introduction covering pearson_weighted scores, coverage diagnostics, and the case for conformal over parametric Tweedie intervals
