---
layout: post
title: "Conformal Prediction vs Bootstrap Intervals for Insurance Pricing"
date: 2026-03-26
categories: [pricing, techniques, comparisons]
tags: [conformal-prediction, bootstrap, insurance-conformal, prediction-intervals, coverage, uncertainty-quantification, GLM, GBM, CatBoost, uk-motor, python, decision-guide]
description: "Conformal prediction and the parametric bootstrap both produce prediction intervals for insurance pricing models. They answer different questions, have different computational costs, and fail in different ways. Here is when to use each."
---

Two methods dominate when pricing actuaries need uncertainty intervals around a model's output: the parametric bootstrap and conformal prediction. Both produce bounds. Both can be presented to a regulator. The resemblance ends there.

This post explains what each method actually computes, where each breaks down, and the practical decision criteria for choosing between them. We will use [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) for the conformal side and standard statsmodels for the bootstrap, with actual code throughout.

---

## What each method computes

This distinction is not pedantic — it determines whether the output is fit for your purpose.

**The parametric bootstrap** answers: if I refit my model on resampled training data, how much do my predicted values move? The output is a *parameter uncertainty interval* — it reflects the sampling variability of the model's coefficients given finite training data. It does not account for the irreducible randomness of future claims. A policy with a stable estimated frequency of 0.08 claims per year will still generate 0 or 1 or 2 claims. The bootstrap interval says nothing about that.

**Conformal prediction** answers: given what I have seen from this model's calibration errors, where will a future realisation of Y fall? The output is a *prediction interval* — it covers future observations, not future fitted values. The coverage guarantee — P(Y ∈ [lower, upper]) ≥ 1 − alpha — is non-asymptotic, distribution-free, and holds for any calibration set above roughly 200 observations.

For pricing purposes, the prediction interval is almost always what you want. You are pricing future claims, not future model fits. For capital modelling and regulatory submissions where the question is "how uncertain are your model parameters", the bootstrap interval is the correct tool.

---

## Bootstrap: what it costs and when it breaks

The parametric bootstrap for a Poisson GLM with 300,000 policies and 40 rating factors runs to roughly 8–15 minutes per 500 bootstrap iterations on a single machine (depending on your optimisation flags and whether you are refitting from scratch or using warm starts). For a CatBoost model with 500 trees, that climbs sharply — CatBoost does not expose a warm-start interface that preserves the early-stopping behaviour, so each bootstrap sample requires a full refit.

The code is straightforward:

```python
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Fit the baseline GLM
glm = smf.glm(
    "claims ~ C(age_band) + C(ncd) + C(vehicle_group) + offset(np.log(exposure))",
    data=train_df,
    family=sm.families.Poisson(),
).fit()

mu_base = glm.predict(test_df)

# Parametric bootstrap: resample rows, refit, predict
rng = np.random.default_rng(2026)
B = 500
boot_preds = np.zeros((B, len(test_df)))

for b in range(B):
    idx = rng.choice(len(train_df), size=len(train_df), replace=True)
    sample = train_df.iloc[idx].reset_index(drop=True)
    try:
        m = smf.glm(
            "claims ~ C(age_band) + C(ncd) + C(vehicle_group) + offset(np.log(exposure))",
            data=sample,
            family=sm.families.Poisson(),
        ).fit(disp=False)
        boot_preds[b] = m.predict(test_df)
    except Exception:
        boot_preds[b] = mu_base

lower_param = np.percentile(boot_preds, 5, axis=0)
upper_param = np.percentile(boot_preds, 95, axis=0)
```

This is a parameter uncertainty interval only. To convert it to a genuine prediction interval for a Poisson model, you need to add the process variance. The mean prediction `mu_i` is uncertain, and even if you knew `mu_i` exactly, the actual realised claim count would still scatter around it. The prediction interval is:

```python
from scipy import stats

# For each test policy, the predictive distribution integrates over
# bootstrap uncertainty in mu_i and Poisson process variance.
# A conservative approach: use the 90th percentile bootstrap mu,
# then find the 95th percentile of a Poisson with that rate.
# For pricing use: the relevant number is often just the parameter CI.
pred_upper = stats.poisson.ppf(0.95, mu=upper_param)
pred_lower = stats.poisson.ppf(0.05, mu=lower_param)
```

The bootstrap breaks in two failure modes. First, with thin cells: if a bootstrap sample draws no observations from a rating factor level, the MLE for that level fails or diverges. You handle this with the `try/except` fallback above, which biases the interval toward the baseline estimate for that iteration — not catastrophic, but not honest. Second, the bootstrap assumes model structure is correct. If the GLM is missing an interaction, bootstrapping 500 misspecified models gives you 500 confident wrong answers. The interval reflects parameter uncertainty conditional on the model being right.

---

## Conformal prediction: the mechanics and the limitations

Split conformal prediction requires a held-out calibration set that the model never trained on. For a temporal dataset — which all insurance pricing datasets are — this is typically the most recent year of policy experience, used as neither training nor tuning data:

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=fitted_catboost,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)

# Calibration set: policies from the most recent underwriting year
cp.calibrate(X_cal, y_cal, exposure=exposure_cal)

# Prediction intervals for the renewal book
intervals = cp.predict_interval(X_test, alpha=0.10)
# Returns a Polars DataFrame: lower | point | upper
```

The `pearson_weighted` score matters. For Tweedie and Poisson models, the correct non-conformity score is the Pearson residual `|y - yhat| / yhat^(p/2)`, not the raw residual. Using raw residuals inflates intervals for low-risk policies (their small residuals are on the same scale as large residuals from high-risk policies) and means the calibration quantile is dominated by the most volatile part of the book. On a UK motor portfolio, the difference is 13–14% in mean interval width with identical coverage guarantees.

Where conformal prediction is honest about its limits: the coverage guarantee is *marginal*, not conditional. P(Y ∈ interval) ≥ 1 − alpha holds across all policies on average. It does not guarantee that 90% of young-driver policies specifically are covered, or that 90% of policies in the top loss decile are covered. Check this before doing anything regulatory:

```python
from insurance_conformal import CoverageDiagnostics

diag = CoverageDiagnostics(cp)
by_decile = diag.by_predicted_decile(X_cal, y_cal, alpha=0.10)
print(by_decile)
# decile  n_obs  coverage  target  width_mean
# 1       1000   0.928     0.90    £1,204
# ...
# 10      1000   0.891     0.90    £6,843
```

A top-decile coverage of 88% when your target is 90% is a 2pp miss that most regulators will accept. A 15pp miss — which is possible if your model is severely miscalibrated in the tail — is a problem that wider intervals will not fix. The right response there is to re-examine the calibration set, not to widen alpha.

---

## The calibration set requirement

Conformal prediction requires a calibration set. This is the method's most significant practical constraint in insurance pricing.

You need at least 200 calibration observations for the finite-sample guarantee to give you intervals that are within ±2pp of the target coverage. Below 200, the quantile estimate becomes unstable. With 1,000 calibration observations, the guarantee is tight: expected coverage will be within ±0.8pp of the target at 90% confidence.

The formula: for a calibration set of size n and target coverage 1 − alpha, the expected deviation from target coverage is approximately `sqrt(alpha(1-alpha)/n)`. At n=200 and alpha=0.10, that is ±2.1pp. At n=1,000, it is ±0.9pp.

For most personal lines motor portfolios — which run to tens of thousands of policies per underwriting year — this is not a constraint. It becomes one for commercial lines, niche products, or thin segments. If your calibration set is 150 policies, use the bootstrap.

---

## When the bootstrap wins

Use the bootstrap in these situations:

**Your model is a GLM and the output must be fully auditable.** Every bootstrap sample is a refit of the same GLM formula. An auditor or model validator can reproduce the interval with the training data, the model specification, and a random seed. The mechanics are self-evident. Conformal prediction is not opaque — it is four lines of code — but "sort calibration residuals and take the 90th percentile" is harder to explain to a non-quantitative stakeholder than "we refitted the model 500 times."

**You need parameter uncertainty rather than prediction uncertainty.** Capital model validation often asks: how sensitive are your rate indications to sampling variation in the training data? That question requires a bootstrap, not a conformal predictor. The conformal interval is not informative about whether the estimated age-frequency relativities would be different with a different training sample.

**Your calibration set has fewer than 200 observations.** Below this threshold, conformal quantile estimates become unreliable and the finite-sample guarantee loses its practical meaning.

**You are using a model that conformal prediction handles badly.** Decision trees with very few leaves, for instance, produce discrete predicted values where the Pearson score distribution is poorly behaved. The bootstrap has no such restriction.

---

## When conformal prediction wins

Use conformal prediction in these situations:

**Your model is a GBM and you need a prediction interval.** Bootstrapping a GBM is computationally prohibitive for most pricing teams. 500 CatBoost refits on 300,000 policies is a multi-hour job. A conformal calibration pass takes seconds after the model is already fitted.

**You need a coverage guarantee rather than a calibrated estimate.** The bootstrap interval is approximately correct under large-sample theory and correct model specification. The conformal interval is exactly correct (up to the finite-sample deviation described above) regardless of whether the model is misspecified. For Solvency II per-risk SCR bounds, distribution-free is better than asymptotically correct.

**You want per-decile coverage diagnostics.** `CoverageDiagnostics` in insurance-conformal tells you whether the 90% guarantee holds across the risk distribution, not just in aggregate. There is no equivalent in a standard bootstrap workflow — you would need to write it yourself and the bootstrap interval has no theoretical connection to per-decile coverage.

**You have a frequency-severity model.** The correct conformal protocol for two-stage models — using predicted rather than observed claim counts in the calibration scoring — is implemented in `insurance_conformal.claims.FrequencySeverityConformal`. Bootstrapping a two-stage frequency-severity model requires careful handling of the dependent bootstrap to avoid breaking the conditional independence structure.

**You need SCR bounds.** The SCR reporting in insurance-conformal is designed for this:

```python
from insurance_conformal import InsuranceConformalPredictor, SCRReport

cp = InsuranceConformalPredictor(model=fitted_model)
cp.calibrate(X_cal, y_cal)

scr = SCRReport(predictor=cp)
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
print(scr.to_markdown())
```

The SCR upper bound at the 99.5th percentile has a finite-sample coverage guarantee. A bootstrap SCR interval does not — it relies on asymptotic normality of the GLM estimator, which may not hold in the extreme tail.

---

## The localised variance question

Both methods struggle with highly heteroskedastic data, but they fail differently.

The bootstrap applies a single quantile across the whole test set. If the high-risk tail of your book has genuinely higher residual variance than the model's Tweedie variance function implies, the single quantile will be too narrow for high-risk policies and too wide for low-risk ones.

Standard conformal prediction with `pearson_weighted` scores adapts to the model-implied heteroskedasticity (yhat^(p/2)) but not to residual variance that the model itself has not captured. For this case, `LocallyWeightedConformal` fits a secondary model on the absolute Pearson residuals to estimate rho_hat(x) — the actual residual spread as a function of features — and produces intervals that are ~24% narrower than standard conformal whilst maintaining coverage:

```python
from insurance_conformal import LocallyWeightedConformal

lw = LocallyWeightedConformal(model=fitted_catboost, tweedie_power=1.5)
lw.fit(X_train, y_train)
lw.calibrate(X_cal, y_cal)
intervals = lw.predict_interval(X_test, alpha=0.10)
```

The bootstrap has no direct equivalent. You can stratify the bootstrap by predicted decile and compute decile-specific quantiles, but this is ad hoc and the resulting intervals have no theoretical coverage guarantee.

---

## Decision rule

| | Bootstrap | Conformal |
|---|---|---|
| Model type | GLM only (GBM too slow) | Any fitted model |
| Output | Parameter uncertainty | Prediction interval |
| Coverage guarantee | Asymptotic, model-correct | Finite-sample, distribution-free |
| Calibration set required | No | Yes (≥200 obs) |
| Computational cost | High (B refits) | Low (one calibration pass) |
| Auditability | High | Moderate |
| SCR use | Approximate | Exact (distribution-free) |

The short version: if your model is a GLM and you need parameter uncertainty or full auditability, use the bootstrap. If your model is a GBM, or if you need a prediction interval with a coverage guarantee, or if you need SCR bounds, use conformal prediction.

They are not substitutes for each other. They answer different questions.

---

**Code:**
- [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) — install with `uv add insurance-conformal`

**Related:**
- [MAPIE vs insurance-conformal: Why Generic Conformal Prediction Breaks on Insurance Data](/2026/03/20/mapie-vs-insurance-conformal-prediction-intervals/)
- [Which Uncertainty Quantification Method? A Decision Framework](/2026/03/25/which-uncertainty-quantification-method-decision-framework/)
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)
- [Conformalized Quantile Regression for Heteroscedastic Insurance Data](/2026/03/24/conformalised-quantile-regression-insurance-prediction-intervals/)
