---
layout: post
title: "Does Conformal Prediction Actually Work for Insurance Claims?"
date: 2026-03-26
categories: [validation]
tags: [conformal-prediction, gbm, catboost, tweedie, uncertainty, pricing, python]
description: "Parametric Tweedie intervals undercover high-risk policies by 10–15 percentage points. We tested conformal prediction on 50,000 UK motor policies to find out whether the fix actually holds in practice."
---

The short answer is yes — but with one important condition: you have to use the right non-conformity score. Use the default absolute residual and you will reproduce exactly the coverage failure you were trying to fix, just for different deciles.

We ran [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) against a CatBoost Tweedie(p=1.5) model on 50,000 synthetic UK motor policies with a heteroskedastic Gamma DGP. The DGP introduces more variance in the high-mean tail than the Tweedie variance function predicts — which is what real motor books look like when large losses are not fully captured by the rating factors.

---

## Why parametric intervals fail in the tail

Parametric Tweedie intervals assume a single dispersion parameter across all risks. The dispersion parameter is estimated as a portfolio-level average. A standard motor risk and a high-value fleet risk both get intervals based on the same dispersion estimate, scaled by predicted mean.

On a heterogeneous book this is wrong, and wrong systematically. High-risk policies have residual variance that grows faster than the Tweedie variance function predicts. The single dispersion estimate is pulled toward the portfolio average, which means it overestimates dispersion for low-risk policies (intervals too wide) and underestimates it for high-risk policies (intervals too narrow).

The benchmark makes this concrete:

| Metric | Parametric Tweedie | Conformal (pearson\_weighted) | Locally-weighted conformal |
|---|---|---|---|
| Aggregate coverage (90% target) | ~88–91% | ≥90% | ≥90% |
| Coverage — top risk decile | ~75–82% | ~88–92% | ~90–93% |
| Coverage — bottom risk decile | ~94–96% | ~90–92% | ~90–92% |
| Assumes constant dispersion | Yes (fails in tail) | No | No |
| Valid coverage guarantee | No | Yes | Approximately |

The top decile coverage of 75–82% is the problem. These are the policies that matter most for reserving, treaty pricing, and regulatory capital. A 90% interval that is actually 78% is a systematic understatement of uncertainty for the worst risks. It is not a statistical curiosity — it is a number that could feed into capital model inputs.

---

## Why the non-conformity score matters

Standard conformal implementations use the absolute residual as the non-conformity score: `|y − ŷ|`. For Gaussian data this is fine. For insurance data it produces the same coverage failure as the parametric approach, just from the other direction: one calibration quantile is computed across all risks, but high-risk policies have fundamentally larger absolute residuals, so the quantile is too small for them.

The correct score for Tweedie models is the Pearson-weighted residual:

```
score(y, ŷ) = |y − ŷ| / ŷ^(p/2)
```

Dividing by `ŷ^(p/2)` normalises by the model's expected standard deviation at each risk level. The calibration quantile is now on a variance-stabilised scale. When inverting back to interval bounds the scaling is applied per-policy, so intervals automatically widen with risk size. This is based on Manna et al. (2025, arXiv:2507.06921).

The practical effect in the benchmark: 13–14% narrower intervals with identical aggregate coverage guarantees. No downside. The locally-weighted variant goes further and specifically targets the top decile, achieving 90–93% coverage there versus 88–92% for the standard weighted score.

---

## The code

```python
from catboost import CatBoostRegressor, Pool
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.utils import temporal_split

X_train, X_cal, y_train, y_cal, _, _ = temporal_split(
    X, y,
    calibration_frac=0.20,
    date_col="accident_year",
)

train_pool = Pool(data=X_train, label=y_train)
model = CatBoostRegressor(loss_function="Tweedie:variance_power=1.5", verbose=0)
model.fit(train_pool)

cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)
cp.calibrate(X_cal, y_cal)

intervals = cp.predict_interval(X_test, alpha=0.10)
# DataFrame: lower, point, upper

diag = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
```

The `coverage_by_decile()` call is the diagnostic that matters. If the top decile is more than 5 percentage points below target, your calibration data is not representative or the non-conformity score is wrong. A well-calibrated run with `pearson_weighted` should be flat across deciles within the Wilson score confidence bands.

---

## When to use a temporal split

The calibration set must be data the model has never seen, drawn from the same distribution as the test set. A random 20% holdout mixes all accident years, which means calibration and test are simultaneously from all periods. That is not strictly wrong, but it hides temporal trends in model residuals — exactly the trend you care about for an in-production pricing model.

Use a temporal split: calibrate on the most recent year of held-out experience, test on the year after. This correctly propagates any trend in residuals rather than averaging it away.

---

## Limitations that actually matter

Conformal prediction cannot make a bad model look good. If your CatBoost model has large residuals on high-risk policies because the features are insufficient, the calibration quantile will be large and the intervals will be wide. The guarantee is a coverage floor, not a quality floor.

The exchangeability assumption is the binding constraint. Calibration on 2024 data and test on 2021 data violates it. Major underwriting changes, mix shifts, or claims inflation breaks the assumption that calibration and test data were drawn from the same distribution. If your portfolio has shifted significantly since calibration, the coverage guarantee degrades in proportion to the shift. The `coverage_by_decile()` diagnostic will usually reveal this as non-monotone coverage across deciles.

For reserving applications where the calibration set is 2–3 years old, we would not rely on the formal guarantee without re-running the calibration.

---

## What it does not replace

Conformal prediction produces prediction intervals for individual risks. It is not a replacement for model calibration — a systematically biased model will produce intervals that are wide enough to achieve marginal coverage but centred on the wrong value. Use the calibration suite in [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) to check the point predictions first, then conformal prediction for the intervals around them.

---

## Verdict

Use conformal prediction if:
- You are producing per-risk uncertainty estimates for reserving, treaty pricing, or capital modelling and you need a guarantee that does not depend on the Tweedie dispersion assumption
- Your high-risk tail is heterogeneous enough that parametric intervals are likely to undercover it (which is most real motor books)
- You need a defensible coverage statement for PRA SS1/23 model validation documentation

Do not bother if:
- Your DGP is well-matched to Tweedie — parametric intervals work fine, as the simpler benchmark.py scenario demonstrates
- You only need marginal coverage, not per-decile coverage — the overhead of conformal calibration is not worth it for aggregate-level uncertainty
- Your calibration set is more than 18 months old relative to the test period without a re-calibration step

```bash
uv add "insurance-conformal[catboost]"
```

Source and benchmarks at [GitHub](https://github.com/burning-cost/insurance-conformal). The CatBoost benchmark (`benchmarks/benchmark_gbm.py`) is the one that produces the interesting result; `benchmarks/benchmark.py` shows the well-matched baseline where parametric intervals do fine.

- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the full library post covering the pearson_weighted non-conformity score, coverage-by-decile diagnostics, and design choices
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — if you need a full conditional distribution rather than a prediction interval, distributional GBMs model per-risk dispersion directly
- [Does Automated Model Monitoring Actually Work?](/2026/03/27/does-automated-model-monitoring-actually-work/)
