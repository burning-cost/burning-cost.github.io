---
layout: post
title: "Quantile GBMs for Insurance: TVaR, ILFs, and Large Loss Loadings"
date: 2026-03-07
categories: [techniques, libraries]
tags: [quantile-regression, tail-risk, tvar, large-loss-loading, ilf, catboost, gbm, motor, python]
description: "CatBoost MultiQuantile plus actuarial output layer: TVaR, ILFs, large loss loadings, exceedance probabilities for UK insurance pricing. insurance-quantile."
---

Your Tweedie GBM gives you an expected loss. That expected loss is one number. It summarises the centre of a conditional distribution that, for most insurance lines, has a right tail that will periodically destroy your loss ratio.

The expected loss is not the problem; you need it. The problem is using it as your only view of risk. A motor bodily injury risk with an expected cost of £800/year and a risk with the same expected cost but 40% more variance in the tail are not the same risk. Your burning cost model treats them identically. Your tail risk does not.

Quantile regression gives you the full conditional distribution, not just the mean. CatBoost has had MultiQuantile loss since version 1.0. What it has not had is an actuarial output layer: TVaR, large loss loadings, ILF curves, exceedance probabilities, calibration diagnostics. We built [`insurance-quantile`](https://github.com/burning-cost/insurance-quantile) to provide those.

---

## The gap in CatBoost

CatBoost's `MultiQuantile` loss function fits all quantile levels simultaneously in a single model, which is both efficient and theoretically sound: the shared feature representations are learned across all probability levels at once. Specify `alpha=0.5,0.75,0.9,0.95,0.99` and you get a model that predicts the 50th through 99th percentile for any new risk.

What you get out of `model.predict()` is an array of numbers. CatBoost does not know what an ILF is. It does not compute TVaR. It does not tell you whether the q_0.9 prediction for your validation set is actually covering 90% of observed losses, or 74% because the tail is heavier than the GBM learned.

There is also a practical problem: CatBoost's MultiQuantile does not guarantee monotone predictions across quantile levels at inference time (GitHub issue #2317). You can get q_0.9 < q_0.95 for individual risks. The fix (isotonic regression applied per row after prediction) is O(n × k) and negligible in practice.

`insurance-quantile` handles both: the isotonic correction is on by default, and the actuarial functions are the primary interface.

---

## What insurance-quantile adds

The library has five functional areas.

**QuantileGBM** wraps CatBoost MultiQuantile and returns predictions in Polars with actuarial column naming. `model.predict(X)` gives you a DataFrame with columns `q_0.5`, `q_0.9`, `q_0.99`, not `column_0`, `column_1`, `column_2`.

**TVaR**, per-risk and portfolio-level. TVaR (Tail Value at Risk, also called CVaR or CTE depending on who trained you) is E[Y | Y > VaR_alpha(Y)]. It is the expected loss given that the loss exceeds its alpha-quantile. Unlike VaR, it is a coherent risk measure, satisfying subadditivity, which means TVaR(portfolio) ≤ sum(TVaR(individual risks)). VaR does not have this property, which is why it is unsuitable for capital allocation.

**Large loss loading**: the additive supplement to burning cost. The loading is TVaR_alpha(i) − E[Y_i], where TVaR comes from the quantile model and E[Y] from a separate Tweedie model. This is the per-risk premium adjustment required to cover tail losses with alpha-level confidence.

**ILF**: Increased Limits Factors, estimated from the exceedance curve via numerical integration. E[min(Y, L)] = ∫₀ᴸ P(Y > x) dx, which is estimated from the quantile predictions. ILF(L1, L2) = E[min(Y, L2)] / E[min(Y, L1)]. This lets you price excess layers directly from the GBM.

**Calibration diagnostics**: coverage fraction per quantile level and pinball loss. A well-calibrated q_0.9 model should be exceeded by roughly 10% of validation observations. If it is exceeded by 22%, your tail risk estimates are materially understated.

---

## Getting started

```bash
uv add insurance-quantile
```

The motor property damage workflow:

```python
import polars as pl
from insurance_quantile import QuantileGBM, per_risk_tvar, large_loss_loading, ilf

# Fit a quantile model on severity (non-zero claims only)
# For a zero-inflated book, model frequency separately with Tweedie
model = QuantileGBM(
    quantiles=[0.5, 0.75, 0.9, 0.95, 0.99],
    iterations=500,
    learning_rate=0.05,
    depth=6,
)
model.fit(X_train, y_severity_train, exposure=exposure_train)

# Quantile predictions: DataFrame with columns q_0.5, q_0.75, q_0.9, q_0.95, q_0.99
preds = model.predict(X_val)
```

The `exposure` parameter weights each row's contribution to the loss function by its earned exposure. This is not offset modelling; there is no log-link adjustment. If your target is aggregate cost rather than severity, pass exposure here so that full-year policies weight more heavily in training.

---

## Per-risk TVaR and large loss loading

The primary use case is supplementing a burning cost model with a tail risk view per risk.

```python
from insurance_quantile import per_risk_tvar, large_loss_loading

# TVaR_0.95 per risk: expected severity given it exceeds the 95th percentile
tvar_result = per_risk_tvar(model, X_pricing, alpha=0.95)

print(tvar_result.values.describe())
# shape: (n_risks,)
# columns: ['tvar']
# mean: 4821.33
# 99th percentile: 18420.10

# The loading over VaR — how much tail risk sits above the 95th percentile
loading_over_var = tvar_result.loading_over_var  # TVaR - VaR per risk
```

`per_risk_tvar` approximates TVaR by taking the mean of quantile predictions at levels strictly above alpha. The approximation improves with more quantile levels above alpha — for TVaR_0.95, you want q_0.95, q_0.975, q_0.99, q_0.995 in the model. For TVaR_0.99, include q_0.99, q_0.995.

The additive large loss loading against a separate Tweedie mean model:

```python
# model_tweedie: any model with .predict(X) -> pl.Series of mean loss estimates
loading = large_loss_loading(
    model_mean=model_tweedie,
    model_quantile=model,
    X=X_pricing,
    alpha=0.95,
)

# loading is a Polars Series: TVaR_0.95(i) - E[Y_i]
# Positive values: risk where tail loading is warranted
# Negative values: unusual, check model consistency
```

The loading formula is additive: `technical_premium = burning_cost + large_loss_loading`. This is the appropriate form when the Tweedie model handles expected severity and the quantile model handles the tail supplement. For motor bodily injury and liability lines where large losses are the primary concern, use alpha=0.99.

One important consistency check: both models must be on the same scale. If the Tweedie model predicts severity and the quantile model was fitted on aggregate cost (including zero claims), the scales differ and the loading will be nonsense.

---

## Increased Limits Factors

ILFs price excess layers. If your basic policy limit is £100k and you want to write a risk up to £500k, the ILF is the factor by which you multiply the basic limit rate:

```python
from insurance_quantile import ilf

# Per-risk ILF from £100k basic to £500k
ilf_values = ilf(
    model=model,
    X=X_pricing,
    basic_limit=100_000,
    higher_limit=500_000,
    n_integration_points=200,
)

# ilf_values: Polars Series, one value per risk
# Value of 1.0 means no additional loading
# Value of 2.3 means the £500k layer costs 2.3x the £100k rate
```

The ILF is computed by numerically integrating the survival function derived from the quantile predictions: E[min(Y, L)] = ∫₀ᴸ S(x) dx. The integration uses 200 quadrature points by default, which is accurate for smooth exceedance curves. The survival function is interpolated from the quantile predictions — accuracy in the upper tail depends on having high quantile levels (q_0.99 or higher) in the model.

For motor bodily injury, where claims can run into millions and the tail beyond £500k is material, fitting expectile mode and including q_0.995 is the right approach. The expectile provides a better-estimated upper tail than the quantile for heavy-tailed lines (more on this below).

---

## Expectiles for heavy-tailed lines

`QuantileGBM` supports expectile regression via `use_expectile=True`. This matters for motor bodily injury and liability, where the tail behaviour is heavy and the choice of risk measure has real consequences.

The theoretical distinction is worth being clear about:

- **Quantiles** are *elicitable* (there exists a strictly proper scoring rule, the pinball loss) but not coherent. The 95th percentile quantile is the threshold that 5% of losses exceed; it tells you nothing about how bad those exceedances are.
- **Expectiles** are both *elicitable* (via an asymmetric squared loss) and *coherent*. They satisfy subadditivity. The expectile at level tau is the unique solution to a weighted squared error problem: it places weight tau on underprediction and (1 − tau) on overprediction. For tau > 0.5, it tilts toward the upper tail more aggressively than a quantile of the same nominal level, producing a risk measure that better captures severity in heavy-tailed distributions.

This is not a theoretical nicety. For motor BI severity, where the 99th percentile claim can be 50× the median and the distribution has no closed form, using a coherent risk measure for premium loading is the correct actuarial practice.

```python
# For motor bodily injury severity: expectile mode
model_bi = QuantileGBM(
    quantiles=[0.5, 0.75, 0.9, 0.95, 0.99],
    use_expectile=True,    # expectile regression: coherent and elicitable
    iterations=500,
    learning_rate=0.05,
)
model_bi.fit(X_bi_train, y_bi_severity)
```

The implementation cost: CatBoost has no MultiExpectile loss as of v1.2.x. Expectile mode fits one CatBoost model per alpha level, which is five times slower than quantile mode for five alpha levels. For a motor BI book with a full predictor set, that is a few minutes versus tens of seconds. The theoretical advantage justifies it for lines where it matters.

---

## Calibration before you rely on the loadings

Do not trust the tail risk estimates before checking calibration. The checks are a two-minute addition to any validation workflow:

```python
# Calibration report on held-out validation data
report = model.calibration_report(X_val, y_val)

print(report["coverage"])
# {'q_0.5': 0.503, 'q_0.75': 0.748, 'q_0.9': 0.871, 'q_0.95': 0.932, 'q_0.99': 0.981}

# q_0.9 observed coverage: 0.871 — 12.9% of validation losses exceed the predicted 90th percentile
# For a correctly calibrated model, it should be 10%.
# Undercoverage at the 90th: the model is underestimating tail severity.
```

Coverage below target at high quantile levels is the common failure mode. It means the model's tail predictions are too low: 90th percentile predictions are being exceeded more than 10% of the time. This matters most for the large loss loading calculation: if the q_0.95 prediction is systematically too low, the large loss loading is understated.

The standalone functions give more control:

```python
from insurance_quantile import coverage_check, pinball_loss

preds = model.predict(X_val)

# Coverage fraction per quantile level
calib = coverage_check(y_val, preds, quantiles=[0.5, 0.75, 0.9, 0.95, 0.99])
print(calib)
# Polars DataFrame: quantile, expected_coverage, observed_coverage, coverage_error

# Pinball loss at 95th percentile (the strictly proper scoring rule for quantile regression)
pb = pinball_loss(y_val, preds["q_0.95"], alpha=0.95)
```

The pinball loss is the correct diagnostic for comparing quantile models against each other at the same alpha level. Do not compare pinball loss across alpha levels, as the scaling differs. Use coverage_error to diagnose miscalibration, pinball_loss to select between competing model specifications.

If coverage is short at high quantile levels, the first things to check are: whether the training data includes enough extreme losses (is the 99th percentile of training claims actually extreme, or has the dataset been capped?), whether the quantile levels in the model are dense enough near the tail, and whether exposure weighting is being applied consistently between training and the Tweedie mean model. Tail miscalibration from data issues is more common than tail miscalibration from model misspecification.

---

## Zero-inflated data

Motor own damage and property books typically have large zero-claim fractions, often 70-85% of policies producing no claims in a year. This affects the quantile model materially.

For a book where 80% of policies claim nothing, the 50th through 79th quantile predictions will be zero. This is mathematically correct. It is also not useful for large loss loading, because the loading is derived from the severity distribution, not the full (zero-inclusive) aggregate distribution.

The recommended approach for zero-inflated books:

1. Fit a separate frequency model (Poisson GBM) on the full book.
2. Filter to non-zero claims only; fit `QuantileGBM` on severity.
3. Compute TVaR from the severity model; multiply by predicted frequency to get the expected per-risk large loss loading.

The library does not handle the frequency/severity split automatically; that decision needs to be made for your specific book and is model sign-off territory. For motor own damage with a 78% zero fraction, splitting is the right call. For motor BI where zeros are uncommon (most reported BI claims result in some payment), fitting the quantile model on the full aggregate is reasonable.

---

## Integration with insurance-conformal

`QuantileGBM` output feeds directly into [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) for Conformalized Quantile Regression (CQR). CQR takes the raw quantile predictions from the GBM and applies a data-driven calibration step that provides distribution-free coverage guarantees on the adjusted intervals.

If your calibration report shows systematic undercoverage, CQR is the fix. It does not improve the quantile model's accuracy — it adjusts the thresholds so that observed coverage matches the stated level on held-out calibration data.

```python
from insurance_conformal import InsuranceConformalPredictor

# Use the quantile GBM as the base model for CQR
cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="cqr",
    quantile_lower=0.05,
    quantile_upper=0.95,
)
cp.calibrate(X_cal, y_cal)
intervals = cp.predict_interval(X_test, alpha=0.10)
```

This is the right combination for any application where the coverage guarantee matters for regulatory or capital purposes: the GBM learns the conditional distribution efficiently, CQR provides the distributional-free coverage certificate.

---

## Practical recommendations by line

**Motor own damage.** Quantile mode, alpha=[0.5, 0.75, 0.9, 0.95, 0.99]. Model severity on non-zero claims. Use large_loss_loading at alpha=0.95 to supplement the Tweedie technical premium. Check coverage on each development year separately — own damage severity trends materially.

**Motor bodily injury.** Expectile mode, alpha=[0.5, 0.75, 0.9, 0.95, 0.99]. Use ILF curves for excess layer pricing. TVaR at alpha=0.99. BI claims have IBNR tails that extend years, so ensure your severity data is sufficiently developed before fitting.

**Home/property.** Quantile mode, alpha=[0.5, 0.75, 0.9, 0.95, 0.99, 0.995]. Use exceedance curves for cat accumulation monitoring. Include perils separately if you have enough volume per peril.

**Employer's liability / public liability.** Expectile mode. These lines are heavy-tailed enough that quantile TVaR understates the tail. The expectile's coherence matters for capital allocation.

---

## Getting started

```bash
uv add insurance-quantile

# With plotting:
uv add "insurance-quantile[plot]"
```

Source and tests on [GitHub](https://github.com/burning-cost/insurance-quantile). The library is 1,351 lines of source, 122 tests passing, and built around a single primary class.

The minimum viable workflow is three calls:

```python
model = QuantileGBM(quantiles=[0.5, 0.75, 0.9, 0.95, 0.99])
model.fit(X_train, y_train)
report = model.calibration_report(X_val, y_val)
```

Check the coverage report first, before touching the loading calculations. If q_0.9 observed coverage is below 0.87, the loading numbers will be unreliable; fix the calibration issue or use CQR on top before presenting the outputs to a sign-off committee.

The burning cost gives you the centre. The quantile model gives you the tail. Both are required. Using only the first is pricing for average experience. Average experience is not what determines whether you write a profitable book of motor BI.

---

**Related articles from Burning Cost:**
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/)
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)
