---
layout: post
title: "Conformal Prediction Intervals for Insurance Pricing Models"
date: 2026-03-31
categories: [machine-learning]
tags: [conformal-prediction, prediction-intervals, Tweedie, GLM, GBM, uncertainty-quantification, PRA-SS1/23, insurance-pricing, insurance-conformal, Hong-2025, Graziadei-2023, Manna-2025]
description: "Parametric Tweedie intervals over-cover low-risk policies and under-cover the high-risk tail. Conformal prediction fixes this with a finite-sample guarantee that does not rely on your distributional assumptions being correct. Here is how to apply it in practice, why naive residual scores are wrong for insurance data, and what it means for model validation packs (SS1/23 for banks; Solvency II Article 120 for insurers)."
author: burning-cost
---

Your pricing model produces a number. That number is wrong — every point estimate is. The question is how wrong, and whether you can attach a range to it that you would stand behind in a model validation committee.

The standard answer is a parametric prediction interval: fit a Tweedie GLM, use the estimated dispersion parameter, and construct bounds from the assumed variance function Var(Y|X) ~ mu^p. This is fine in aggregate. On a heterogeneous UK motor book it is quietly broken in exactly the places that matter.

The problem is not that the Tweedie assumption is a bad one. It is that the assumption is uniform. Var ~ mu^p applies the same variance scaling across every cell — the standard young driver in SE postcode, the 72-year-old in rural Cumbria with three points, the restored classic that gets driven 1,800 miles a year. A single dispersion parameter phi is estimated from the full book. It will be approximately right on the modal risk and systematically wrong at the extremes. In aggregate, the parametric interval over-covers at 90% (produces 93.1% actual coverage on a realistic synthetic book — intervals wider than needed), while the top risk decile, which drives your reinsurance attachment, your reserving, and your SCR, gets under-covered.

Conformal prediction is the fix. The guarantee is `P(y_test in interval) >= 1 - alpha` for any data distribution, as long as the calibration and test observations are exchangeable. No distributional assumption. No correctly specified variance function. The interval is distribution-free in the strict sense.

---

## What conformal prediction actually does

The core idea is simple enough that you can explain it in a model governance committee without making statisticians uncomfortable.

Split the available data into three parts: training, calibration, and test. Train your model on the training set. On the calibration set — data the model has not seen — compute a non-conformity score for each observation: a measure of how surprising the actual outcome is given the model's prediction. Sort those scores. The (1-alpha) quantile of that sorted list is your threshold q*.

For any new test observation, the prediction interval is the set of outcomes y that would have a non-conformity score at most q*. For the standard Pearson score this inverts cleanly to a symmetric margin around the predicted mean.

The formal guarantee: for any alpha in (0,1), any data distribution, and any model (even a completely misspecified one), `P(y_test in [lower, upper]) >= 1 - alpha`. The over-coverage is at most 1/(n_cal + 1), which becomes negligible above n_cal = 500.

The one real assumption is exchangeability — calibration and test observations drawn from the same distribution. This is weaker than i.i.d., but it does get violated by portfolio drift, mid-year Ogden rate changes, or a CAT event mid-season. We will come back to that.

---

## Why naive residual scores are wrong for insurance

Most general conformal prediction libraries use the raw absolute residual as the non-conformity score: `s_i = |y_i - yhat_i|`. For a homoscedastic regression problem this is perfectly reasonable.

Insurance data is not homoscedastic. A prediction error of £500 on a policy with expected loss of £300 is very different from a prediction error of £500 on a policy with expected loss of £8,000. The Tweedie model tells you this explicitly: the expected variance scales as mu^p. For p=1.5 (the standard compound Poisson-Gamma value for UK motor), the standard deviation scales as mu^0.75. Raw residuals are not exchangeable across the risk distribution — policies with high predicted losses will systematically produce larger residuals, and the distribution of non-conformity scores will be skewed by risk level.

The correct score for a Tweedie model is the Pearson-weighted residual:

```
s_i = |y_i - yhat_i| / yhat_i^(p/2)
```

This normalises by the Tweedie standard deviation. The resulting scores are approximately exchangeable across risk levels, which is what the conformal guarantee requires. Using raw residuals in insurance is not conservative — it is wrong in a way that produces intervals that are too wide on cheap risks (unnecessary capital) and too narrow on expensive ones (the risks that actually matter).

The `insurance-conformal` library implements four score types for Tweedie models, ranging from fast to statistically optimal:

| Score | Formula | When to use |
|---|---|---|
| `pearson_weighted` | `\|y - yhat\| / yhat^(p/2)` | Default. Any Tweedie pricing model. |
| `pearson` | `\|y - yhat\| / sqrt(yhat)` | Pure Poisson frequency models (p=1). |
| `deviance` | Deviance residual | Exact statistical optimality; slower to invert. |
| `anscombe` | Anscombe variance-stabilising residual | Fast closed-form inversion; undefined at p=2. |
| `raw` | `\|y - yhat\|` | Baseline only. Not appropriate for insurance data. |

Width hierarchy (narrowest first, coverage identical): `pearson_weighted <= deviance <= anscombe < pearson < raw`. Use `pearson_weighted` unless you have a strong reason otherwise.

---

## The benchmark results

CatBoost Tweedie(p=1.5), 50,000 synthetic UK motor policies, heteroskedastic Gamma DGP, temporal 60/20/20 split:

| | Parametric Tweedie | Conformal (`pearson_weighted`) | Locally-weighted conformal |
|---|---|---|---|
| Distribution assumption | Tweedie Var ~ mu^p | None | None |
| Aggregate coverage @ 90% target | 93.1% (over-covers) | 90.2% | 90.3% |
| Top-decile coverage @ 90% target | 90.4% | 87.9% | 90.6% |
| Mean interval width | £4,393 | £3,806 (−13.4%) | £3,881 (−11.7%) |
| Width adapts per risk segment | No | Partial | Yes |
| Finite-sample valid guarantee | No | Yes | Yes |

The 13.4% width reduction from conformal over parametric is not a free lunch. The parametric baseline over-covers in aggregate (93.1% vs 90% target), and conformal earns its narrower intervals by being correctly calibrated rather than generous. The parametric baseline also only coincidentally meets the top-decile target on this dataset — on books with more pronounced tail heteroscedasticity it will miss.

The locally-weighted variant (`lw_pearson` score) fits a secondary spread model on top of the base predictions to estimate observation-level residual variability. It recovers the top-decile coverage gap that standard conformal misses, at the cost of fitting and maintaining an additional GBM.

---

## Practical usage

### The core predictor

```python
from insurance_conformal import InsuranceConformalPredictor

# Wrap any fitted sklearn-compatible model
cp = InsuranceConformalPredictor(
    model=fitted_gbm,
    nonconformity="pearson_weighted",
    tweedie_power=1.5,
)

# Calibrate on held-out data — must not overlap training
cp.calibrate(X_cal, y_cal)

# 90% prediction intervals — polars DataFrame: lower, point, upper
intervals = cp.predict_interval(X_test, alpha=0.10)

# Always check per-decile coverage
print(cp.coverage_by_decile(X_test, y_test, alpha=0.10))
```

Target n_cal >= 2,000 for stable production use. The guarantee holds for any n_cal >= 1, but below 500 the interval widths are materially wider and more variable.

### The new unified API: TweedieConformPredictor

v0.7.0 introduces `TweedieConformPredictor`, which consolidates the four score types, exposure weighting, and automatic score selection into a single class. For new code, prefer this over the older `InsuranceConformalPredictor`.

```python
from insurance_conformal.tweedie_conform import TweedieConformPredictor

tcp = TweedieConformPredictor(
    model=fitted_gbm,
    p=1.5,
    score="pearson",  # or "deviance", "anscombe", "lw_pearson"
)

# Basic usage: calibrate, then predict
tcp.calibrate(X_cal, y_cal)
intervals = tcp.predict_interval(X_test, alpha=0.10)
# intervals is a (n, 2) numpy array: [:, 0] lower, [:, 1] upper
```

For rate models with varying exposure — the common case in UK motor where the model predicts an annualised rate but policies have part-year exposure:

```python
tcp = TweedieConformPredictor(
    model=rate_model,
    p=1.5,
    exposure_weighted=True,
)
tcp.calibrate(X_cal, y_cal, exposure_cal=exposure_cal)
intervals = tcp.predict_interval(X_test, alpha=0.05, exposure_new=exposure_test)
```

The exposure adjustment modifies the score denominator to `(e * mu)^(p/2)`, which is correct when the model predicts the rate per unit exposure and policies have materially different exposure durations.

### Score selection

If you are unsure which score type to use, `select_score()` evaluates candidates on a held-out validation set and returns the one with the narrowest mean interval width at the target alpha:

```python
tcp.calibrate(X_cal, y_cal)
best = tcp.select_score(X_val, y_val, alpha=0.10)
print(best)  # e.g. "deviance" or "pearson"
```

This works by re-calibrating a copy of the predictor under each score type and measuring mean width. The validation set must be independent of the calibration set.

### Locally-weighted intervals

The standard conformal predictor produces intervals with width that varies only through the Tweedie variance function — policies with higher predicted means get wider intervals but through a fixed scaling. The locally-weighted variant also learns a secondary spread model rho_hat(X) = E[|Pearson residual| | X] from training data. This captures residual heteroscedasticity that the variance function misses.

```python
from insurance_conformal import LocallyWeightedConformal

lw = LocallyWeightedConformal(model=fitted_gbm, tweedie_power=1.5)
lw.fit(X_train, y_train)    # fits the spread model on training residuals
lw.calibrate(X_cal, y_cal)
intervals = lw.predict_interval(X_test, alpha=0.10)
```

Or via `TweedieConformPredictor` with `score="lw_pearson"`:

```python
tcp = TweedieConformPredictor(model=fitted_gbm, p=1.5, score="lw_pearson")
tcp.fit(X_train, y_train)   # only needed for lw_pearson
tcp.calibrate(X_cal, y_cal)
intervals = tcp.predict_interval(X_test, alpha=0.10)
```

### Coverage diagnostics

Aggregate coverage meeting the target does not mean the intervals are well-calibrated across the risk distribution. The guarantee is marginal, not conditional. Always run the decile diagnostic after calibration:

```python
diag = tcp.coverage_diagnostic(X_test, y_test, alpha=0.10)

print(f"Aggregate coverage: {diag['empirical_coverage']:.3f}")
print(f"Mean width: £{diag['mean_width']:,.0f}")

for row in diag["by_decile"]:
    print(
        f"Decile {row['decile']:2d} | "
        f"mean predicted £{row['mean_predicted']:,.0f} | "
        f"coverage {row['coverage']:.3f}"
    )
```

High-risk subgroups being systematically under-covered is the most common failure mode in practice. If the top two or three deciles are below the target, switch to locally-weighted conformal.

---

## Temporal calibration

The exchangeability assumption is violated by portfolio drift. If you calibrate on 2023 data and run on 2025 policies, claims inflation, book mix changes, and the model's view of the world may all have shifted. The interval widths will be wrong in an unpredictable direction.

The fix is straightforward: calibrate on recent data only.

```python
from insurance_conformal.utils import temporal_split

X_train, X_cal, y_train, y_cal, _, _ = temporal_split(
    X, y,
    calibration_frac=0.20,
    date_col="accident_year",
)

model.fit(X_train, y_train)
tcp.calibrate(X_cal, y_cal)
```

For books with active inflation or following a discrete shock (Ogden rate revision, CAT event), the `RetroAdj` online conformal adaptor recovers coverage within 15–30 steps rather than the 80–150 steps required by standard adaptive conformal inference.

One underappreciated calibration problem specific to insurance: IBNR. Calibrating on development-year 0 or 1 data means your non-conformity scores are computed against understated claim totals. The result is intervals that appear tight on calibration but will miss at test time on fully-developed years. Use only accident years with at least three years of development, or apply chain-ladder development factors to `y_cal` before calibration. This is not a conformal prediction problem — it is a data preparation problem that conformal prediction makes more visible.

---

## The frequency-severity case

A two-stage frequency-severity model has a calibration subtlety that most practitioners will miss. In a naive implementation you might calibrate the severity model using the observed claim count at calibration time, then predict using the model's estimated frequency at test time. This creates a distributional mismatch: the calibration scores are computed under a different input distribution than the prediction scores. The guarantee breaks.

`FrequencySeverityConformal` handles this correctly by feeding the predicted frequency (not the observed count) into the severity model at both calibration and test time.

```python
from insurance_conformal.claims import FrequencySeverityConformal

fs = FrequencySeverityConformal(
    freq_model=PoissonRegressor(),
    sev_model=GammaRegressor(),
)
fs.fit(X_train, d_train, y_train)   # d_train = observed claim counts
fs.calibrate(X_cal, d_cal, y_cal)
intervals = fs.predict_interval(X_test, alpha=0.10)
```

The underlying theory is in Graziadei, Janett, Embrechts & Bucher (arXiv:2307.13124, 2023). It is worth reading if you run a two-stage model in production.

---

## The regulatory case for conformal intervals

*(PRA SS1/23 applies to banks, not directly to insurers. For insurers, the equivalent requirements flow from Solvency II Articles 120–126. Many insurers adopt SS1/23's validation principles voluntarily. The framing below applies to both.)*

The SS1/23 model validation framework requires firms to demonstrate that their internal models produce predictions with appropriate uncertainty quantification, and that model limitations are documented and understood by users. The standard parametric bootstrap interval is difficult to defend under this framework: the interval width depends on the distributional assumption being correct, and you cannot write a finite-sample guarantee that is independent of model specification.

Conformal prediction changes this. The formal statement — `P(y_test in interval) >= 1 - alpha`, finite-sample valid, no distributional assumptions, holds regardless of model misspecification — is a stronger claim than any parametric interval can make. You can include it verbatim in a validation pack.

The `SCRReport` class produces per-risk 99.5% upper bounds with a coverage validation table in the format suitable for internal model stress testing:

```python
from insurance_conformal import InsuranceConformalPredictor, SCRReport

cp = InsuranceConformalPredictor(model=fitted_model)
cp.calibrate(X_cal, y_cal)

scr = SCRReport(predictor=cp)
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
val_table  = scr.coverage_validation_table(X_test, y_test)
print(scr.to_markdown())
```

One caveat that belongs in the validation pack alongside the output: `SCRReport` is an internal stress-testing tool. Solvency II SCR calculations for regulatory purposes require sign-off under an approved internal model or the standard formula. The conformal bounds are a sound complement to that process — a distribution-free check on whether your parametric SCR estimates are plausible — not a substitute for it.

For the model validation pack specifically — whether structured under SS1/23 (banks) or Solvency II Article 120 (insurers) — the defensible framing is:

1. The conformal coverage guarantee holds finite-sample, without distributional assumptions.
2. Per-decile coverage diagnostics are shown — aggregate coverage meeting the target is a necessary but not sufficient check.
3. Calibration data is drawn from recent accident years (specify years) with at least three development years to address IBNR.
4. Exchangeability monitoring is in place: `coverage_by_decile` is rerun quarterly; material deviations trigger recalibration.

---

## What conformal prediction does not fix

Three things worth being honest about.

**Conditional coverage.** The guarantee is marginal. It holds on average across the test distribution, not conditional on any particular covariate value or risk segment. Conformal prediction narrows the gap — and locally-weighted conformal narrows it further — but it does not eliminate it. A heterogeneous book with strong tail segments should always be checked with `coverage_by_decile`.

**Portfolio drift.** Exchangeability is the load-bearing assumption. If your calibration data no longer looks like your current book, the guarantee is void. This is not a limitation of conformal prediction specifically — any interval that relies on past data has the same problem — but conformal makes the assumption explicit rather than burying it in a distributional form.

**Model error.** Conformal prediction takes your point predictions as given. If the model is badly wrong on a subpopulation — systematic bias, missing features, structural break — the intervals will be centred in the wrong place. They will be wide enough to compensate, because the calibration scores on that subpopulation will be large, but an unbiased interval centred on a wrong prediction is not the same as an unbiased interval centred on the right one. Conformal prediction is not a substitute for monitoring model accuracy.

---

## Installation

```bash
uv add insurance-conformal

# With CatBoost support:
uv add "insurance-conformal[catboost]"

# With LightGBM support:
uv add "insurance-conformal[lightgbm]"

# With everything:
uv add "insurance-conformal[all]"
```

Source and documentation: [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal).

The library takes any fitted model — Tweedie GBM, GAM, GLM, or the output of [insurance-gam](https://github.com/burning-cost/insurance-gam) — and feeds distribution-free intervals into downstream tools including [insurance-governance](https://github.com/burning-cost/insurance-governance) for model governance validation packs (SS1/23-aligned for banks; Solvency II Article 120 for insurers).
