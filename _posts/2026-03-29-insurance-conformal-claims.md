---
layout: post
title: "Your GLM's Prediction Intervals Are Wrong (and We Can Prove It)"
date: 2026-03-29
categories: [libraries, conformal-prediction, solvency]
tags: [conformal-prediction, glm, tweedie, solvency-ii, scr, insurance-claims, python, catboost]
description: "Hong (2025) showed a parametric GLM produced 57.8% actual coverage against a 99.5% nominal prediction interval. Conformal prediction held at 99.6% with no distributional assumption. For Solvency UK SCR, this difference is material. We've built the first Python implementation."
---

Your GLM does two things. It predicts a mean claim amount. It also produces a prediction interval around that mean - a range within which future claims should fall with, say, 99.5% probability. We trust the mean. We largely ignore the interval.

Hong (2025, arXiv:2503.03659) ran the numbers on what happens when you take that interval seriously.

Using personal injury claims data from Jong and Heller (2008) - 22,036 settled UK claims from 1989 to 1999, 11 covariates - Hong fitted a GLM and asked a simple question: in a misspecification scenario where the true distribution is Pareto but the GLM assumes log-normal, how often does the 99.5% prediction interval actually contain the true claim value?

The answer was 57.8%.

You set your Solvency UK SCR at 99.5% VaR. Your prediction interval achieves 57.8% coverage. That gap is not a rounding error. It is a structural failure of the parametric approach.

Conformal prediction, under the same misspecification, held at 99.6%. And it does this with no distributional assumption whatsoever.

We have built [`insurance-conformal-claims`](https://github.com/burning-cost/insurance-conformal-claims) - the first Python implementation of Hong's order-statistic shortcut, the h-transformation extension, and insurance-specific Tweedie nonconformity scores from Manna et al. (2025, ASMBI asmb.70045).

---

## Why parametric intervals fail at the tail

A GLM prediction interval is derived from the assumed error distribution. For a log-normal GLM, the interval for a new observation is:

```
y_hat * exp(±z_{alpha/2} * sigma)
```

This interval is valid *if* the log-normal assumption holds. At the 95th or 99th percentile, a moderate distributional misspecification produces moderate coverage loss. At the 99.5th percentile - where SCR lives - the same misspecification can be catastrophic. The tail is precisely where distributional assumptions are least tested by data and most consequential for solvency.

The width ratio in Hong's Table 2 is the tell: the GLM interval at 99.5% coverage has width ratio 0.07 relative to the conformal interval. The parametric interval is seven times narrower than the distribution-free interval - and it achieves 57.8% actual coverage. The GLM is not being precise. It is being overconfident in the wrong direction.

Random forest does better at 98.8% coverage, but it still falls short of nominal. Conformal prediction holds at 99.6% regardless of what model generates the data.

---

## The order-statistic shortcut

Standard full conformal prediction requires a grid search over candidate values of y - in principle, infinitely many. Hong (2025) identified a property of a specific nonconformity measure that eliminates the grid search entirely.

The measure is linear in the new observation's value. That linearity means the plausibility function is a step function with a single jump. The prediction region is always an interval with an upper bound at a single order statistic. The computation is:

1. For each training observation i, compute the adjusted score:
   `W_i = y_i + sum_j (X_{new,j} - X_{i,j}) / n`
2. Sort the n scores: `W_{(1)} <= W_{(2)} <= ... <= W_{(n)}`
3. Set `k = min(n, floor((n+1) * (1-alpha) + 1))`
4. The prediction region is `(0, W_{(k)})`

That is the entire algorithm. O(n log n) - a sort and an index lookup. No model fitting, no grid, no iteration.

The coverage guarantee (Theorem 1 of arXiv:2503.03659) is finite-sample and distribution-free:

```
P(Y_new in C_alpha(X_new)) >= 1 - alpha
```

for all n and all exchangeable distributions. Not asymptotically. Not under model assumptions. For all n.

For SCR at 99.5%, set `alpha = 0.005`. The upper bound `W_{(k)}` is your distribution-free capital requirement. Hong's paper computed 544,093 for the personal injury dataset (Table 7). That number required no distributional assumption about injury claims. It required only that the 22,036 training claims and the new claim were drawn from the same generating process - the exchangeability condition.

---

## h-transformation: narrower intervals with a model

The model-free approach gives exact coverage. It can produce wide intervals precisely because it uses no information about the conditional mean structure of claims.

Hong (2026, arXiv:2601.21153) extends this via the h-transformation. Decompose claims as:

```
Y = h(X) + W
```

where h is any non-negative function - a fitted regression model, a log-linear predictor, anything. Because Y_i are iid, the residuals W_i = Y_i - h(X_i) are also iid regardless of h. The prediction interval becomes:

```
(h(X_new) + W_{(k_lo)}, h(X_new) + W_{(k_hi)})
```

where the quantiles are now computed on the residuals, not the raw claims. When h is a good fit, residuals have smaller variance, and intervals are narrower.

Hong's simulations show a well-fitted linear h reduces mean interval width by 25% relative to the model-free baseline while preserving finite-sample coverage validity. A CatBoost h will do better still.

The coverage guarantee does not depend on h being correctly specified. If h is poor, intervals are wide. If h is good, intervals are narrow. But coverage is guaranteed either way. You cannot lose coverage by choosing a bad model. You can only lose efficiency.

---

## Tweedie-specific nonconformity scores

The Hong approach is fully model-free. Manna et al. (ASMBI 2025, asmb.70045) take a different route: split conformal with nonconformity scores that respect the Tweedie structure of insurance claims.

For a Tweedie compound Poisson-Gamma distribution with power p in (1, 2) - the standard model for aggregate insurance claims - the variance function is V(mu) = mu^p. Three scores follow naturally:

**Pearson residual** - closed-form inverse, fastest:
```
R_Pear = |y - mu_hat| / mu_hat^{p/2}
```

**Deviance residual** - no closed-form inverse, requires root-finding per prediction:
```
R_Dev = sqrt(2 * [y * mu_hat^{1-p}/(p-1) - y^{2-p}/((p-1)(2-p)) + mu_hat^{2-p}/(2-p)])
```

**Anscombe residual** - a normalisation that achieves approximate Gaussianity for moderate p:
```
R_Ansc = |y^{1-p/3} - mu_hat^{1-p/3}| / mu_hat^{p/6}
```

The Pearson score is the practical choice. Its prediction interval has a closed form:

```
C(X_new) = [max(0, mu_hat - q * mu_hat^{p/2}),
             mu_hat + q * mu_hat^{p/2}]
```

where q is the empirical `ceil((1-alpha)(n_cal + 1))`-th quantile of calibration scores. Coverage is guaranteed to lie in `[1-alpha, 1-alpha + 1/(n_cal + 1)]`.

The best performer in Manna et al. is locally weighted Pearson - a two-stage procedure where a secondary model predicts the magnitude of Pearson residuals, and that predicted magnitude enters the denominator:

```
R*_Pear = R_Pear / rho_hat(X)
```

The `rho_hat` model is fitted on training Pearson residuals from the same data used to fit the mean model. `insurance-conformal-claims` implements this as `TwoStageLWConformal`, which auto-fits the spread model using LightGBM by default. You provide only the mean model and the Tweedie power p.

---

## Using the library

```bash
pip install insurance-conformal-claims
# or
uv add insurance-conformal-claims
```

### Model-free SCR from Hong's shortcut

```python
import polars as pl
from insurance_conformal_claims import HongConformal, SCRReport

df = pl.read_parquet("personal_injury_claims.parquet")
X = df.select(["age", "region", "severity_band", "vehicle_type", "ncd_years"]).to_numpy()
y = df["settled_amount"].to_numpy()

# No model. No distributional assumption.
hong = HongConformal()
hong.fit(X_train, y_train)

scr = SCRReport(hong)
capital = scr.solvency_capital_requirement(X_train, alpha=0.005)
print(f"SCR upper bound: £{capital:,.0f}")

# Full coverage table across alpha values
print(scr.coverage_table(X_test, y_test))
#    alpha  nominal_coverage  actual_coverage  mean_width  n_test
# 0  0.005             0.995            0.996  544093.0    4408
# 1  0.010             0.990            0.991  498721.0    4408
# 2  0.050             0.950            0.951  312440.0    4408
```

### Locally weighted Tweedie intervals with CatBoost

```python
from catboost import CatBoostRegressor
from insurance_conformal_claims import TwoStageLWConformal

# Fit your mean model first
mean_model = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=500,
    verbose=0,
)
mean_model.fit(X_train, y_train, cat_features=cat_idx)

# Two-stage conformal wraps it.
# p=1.5 matches the CatBoost Tweedie power
conf = TwoStageLWConformal(mean_model=mean_model, p=1.5)
conf.fit(X_train, y_train)
conf.calibrate(X_cal, y_cal)

intervals = conf.predict_interval(X_test, alpha=0.005)
# intervals.shape == (n_test, 2): [lower, upper] per claim
```

### h-transformation with CatBoost

```python
from insurance_conformal_claims import HongTransformConformal

# Use CatBoost as the h-function - narrows intervals vs model-free
htc = HongTransformConformal(h_model=mean_model)
htc.fit(X_train, y_train, X_cal=X_cal, y_cal=y_cal)

intervals = htc.predict_interval(X_test, alpha=0.005)
# Coverage guaranteed regardless of how good CatBoost is.
# If CatBoost fits well, intervals are ~25% narrower than HongConformal.
```

---

## The Solvency UK angle: PRA PS9/24

Under Solvency UK (PRA PS9/24, which replaces Solvency II post-Brexit for UK insurers), the SCR is computed at 99.5% VaR over a one-year horizon. The SCR is the capital required to survive a 1-in-200 year event.

The standard approach derives this from a fitted model's distributional assumption. If that assumption is wrong - which Hong demonstrates it can be, badly - the SCR is understated. A 57.8% coverage interval set at 99.5% nominal means the insurer holds capital for a 1-in-200 event but is actually only protected against roughly a 1-in-1.7 event.

The conformal approach is not a replacement for actuarial models. It is a validation layer. Run your GLM, compute your parametric SCR. Then run `HongConformal` on the same data, extract the conformal SCR, and compare. If they diverge materially, your distributional assumption is doing serious work. You should know that before your internal model validation team asks.

Three limitations are worth stating plainly.

**Marginal, not conditional, coverage.** The 99.5% guarantee is averaged over the covariate distribution. Conditioning on a specific policyholder's X, coverage may be lower. For portfolio-level SCR, this is fine. For per-policy capital allocation, you need mondrian conformal or conformalized quantile regression.

**Exchangeability.** The guarantee requires that training and test observations are exchangeable - drawn from the same distribution. UK personal injury claims from 1989 to 1999 are not exchangeable with 2025 claims. Restrict calibration data to recent years. We recommend no more than five years for motor and three years for lines with faster mix shift.

**Scaling.** `HongConformal.predict_interval()` computes a matrix of shape (n_new, n_train). For n_train = 50,000 and n_new = 10,000, this is 500 million floats. The library uses chunked batching with a configurable `batch_size` parameter (default: 1,000 new observations per batch).

---

## What is new here

There was no Python implementation of any of this. Hong's papers have no accompanying code. The Manna et al. R code at `alokesh17/conformal_LightGBM_tweedie` is research scripts - not installable as a package, no tests, does not implement Hong's method. MAPIE and crepes provide generic split conformal but with no Tweedie scores and no SCR reporting.

`insurance-conformal-claims` fills all of these gaps: Hong's O(n log n) order-statistic shortcut, h-transformation with any sklearn-compatible model, three Tweedie nonconformity scores, two-stage locally weighted conformal with auto-fitted spread model, and an `SCRReport` class that extracts and formats the 99.5% VaR in the form a Solvency UK filing expects.

The library is at [github.com/burning-cost/insurance-conformal-claims](https://github.com/burning-cost/insurance-conformal-claims).

```bash
pip install insurance-conformal-claims
```

Read Hong (2025) alongside it. The paper is twelve pages and Table 7 is worth the whole thing.

---

*Papers: Hong (2025) arXiv:2503.03659 - Hong (2026) arXiv:2601.21153 - Manna et al. (2025) ASMBI doi:10.1002/asmb.70045*

---

**Related articles from Burning Cost:**
- [Conformal Prediction for Insurance Solvency Capital Requirements](/2026/03/11/conformal-scr-solvency/)
- [Conformal Time-Series Prediction Intervals for Insurance](/2026/03/12/insurance-conformal-ts/)
- [Calibration Testing for Insurance Pricing Models](/2026/03/09/insurance-calibration/)
