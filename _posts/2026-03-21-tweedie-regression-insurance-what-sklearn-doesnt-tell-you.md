---
layout: post
title: "Tweedie Regression for Insurance: What sklearn Doesn't Tell You About Exposure"
date: 2026-03-21
author: Burning Cost
categories: [modelling, tutorials, libraries]
description: "sklearn's TweedieRegressor tutorial gets you to a fitted model in six lines. It also produces predictions that are wrong for any policy with non-annual exposure. Here is the correct approach."
canonical_url: "https://burning-cost.github.io/2026/03/21/tweedie-regression-insurance-what-sklearn-doesnt-tell-you/"
tags: [tweedie, GLM, exposure, offset, pure-premium, insurance-distributional, TweedieGBM, statsmodels, catboost, python, tutorial, motor-insurance, uk-insurance, compound-poisson-gamma]
---

Search "Tweedie regression insurance Python" and the top results will give you `sklearn.linear_model.TweedieRegressor`. It fits in six lines. The documentation is clean. The example works.

And then you try to use it on a real motor book — where policies have mid-term adjustments, monthly direct debits, and exposure periods anywhere from two weeks to fourteen months — and the predictions are wrong in a way that is not immediately obvious.

The problem is exposure handling. Or rather, the absence of it. `TweedieRegressor` has no mechanism for an exposure offset. This is not a minor gap you can work around with a feature transformation. It is a fundamental architectural difference between a Tweedie model and a Tweedie *rate* model, and getting it wrong produces systematic errors across your portfolio that are invisible unless you specifically test for them.

This post explains the difference, shows the correct implementation in statsmodels, compares that to gradient boosting with log-exposure handling, and then shows how `insurance-distributional`'s `TweedieGBM` handles all of this properly.

---

## What exposure actually is

A motor policy written for a full year has exposure 1.0. A policy written on 1 October and renewed on 1 October next year has exposure 1.0. A policy written on 1 October and cancelled on 1 April has exposure 0.5. A monthly direct debit customer in month three of their policy has exposure 0.083.

The aggregate claims cost you observe on each policy reflects both the underlying risk *and* how long the policy was in force. A policy at £400 pure premium per annum that runs for six months will cost, on average, £200. If you train a model on observed claims cost without accounting for exposure, you are teaching it to predict something that depends partly on how long the policy ran. That is not useful.

What you want is a model for the rate: expected claims cost per unit exposure. The canonical form of a GLM that achieves this is:

```
log(E[Y]) = log(exposure) + X * beta
```

The `log(exposure)` term is the offset. It is not a feature — you do not fit a coefficient to it. It is fixed at 1.0 for every observation, built into the linear predictor structure so that the model predicts a rate, and the expected total cost is rate × exposure.

`sklearn.linear_model.TweedieRegressor` does not support offsets. There is a long-standing GitHub issue about this. As of now, it remains unfixed. If you pass `log(exposure)` as a feature, sklearn will fit a coefficient to it. That coefficient will not be 1.0. The model will learn a distorted approximation that happens to have good aggregate statistics on annual policies but fails on any book with genuine exposure variation.

---

## Demonstrating the problem

We use 10,000 synthetic motor policies with genuine exposure variation — roughly 30% of the book has exposure below 0.7 (monthly payers, mid-term cancellations, new business late in the year). The true model is a Tweedie compound Poisson-Gamma with p=1.5, clean covariate structure, and a true offset of exactly log(exposure).

```python
import numpy as np
from scipy.stats import pearsonr

rng = np.random.default_rng(42)
n = 10_000

# Rating factors
vehicle_age   = rng.integers(0, 15, n).astype(float)
driver_age    = rng.integers(17, 75, n).astype(float)
ncd_years     = rng.integers(0, 9, n).astype(float)
vehicle_group = rng.integers(1, 5, n).astype(float)

# Exposure: genuine variation. 40% annual, 30% semi-annual, 30% fractional.
# This is typical for a UK direct writer with monthly DD customers.
u = rng.random(n)
exposure = np.where(u < 0.40, 1.0,
           np.where(u < 0.70, rng.uniform(0.4, 0.9, n),
                               rng.uniform(0.05, 0.4, n)))

X = np.column_stack([vehicle_age, driver_age, ncd_years, vehicle_group])

# True pure premium rate (annual basis)
log_mu_rate = (
    5.5
    + 0.025 * np.maximum(25 - driver_age, 0)
    + 0.03  * vehicle_age
    - 0.08  * ncd_years
    + 0.10  * vehicle_group
)
mu_rate = np.exp(log_mu_rate)   # annual pure premium
mu_obs  = mu_rate * exposure    # expected cost for this policy's actual exposure

# Simulate Tweedie compound Poisson-Gamma at p=1.5
# alpha = (2-p)/(p-1) = 1.0 for p=1.5 (Exponential severity)
p_tw     = 1.5
phi      = 0.8
alpha_sv = (2 - p_tw) / (p_tw - 1)          # = 1.0
lam      = mu_obs ** (2 - p_tw) / (phi * (2 - p_tw))
counts   = rng.poisson(lam)
beta_sv  = mu_obs / (lam * alpha_sv)

y = np.array([
    rng.gamma(alpha_sv, beta_sv[i], size=c).sum() if c > 0 else 0.0
    for i, c in enumerate(counts)
])

# Train/test split — 80/20
n_tr = 8_000
X_tr,  X_te  = X[:n_tr],        X[n_tr:]
y_tr,  y_te  = y[:n_tr],        y[n_tr:]
exp_tr, exp_te = exposure[:n_tr], exposure[n_tr:]
mu_rate_te = mu_rate[n_tr:]  # true annual rate for evaluation
```

---

## The wrong way: sklearn TweedieRegressor

sklearn has no offset parameter. The workaround most practitioners discover is to divide the response by exposure — fitting on `y / exposure` — or to add `log(exposure)` as a feature. Neither is correct.

**Dividing by exposure** produces a rate-per-exposure-unit target, which is correct conceptually. But the resulting observations have different variances — a policy with 0.05 exposure has a highly noisy `y/exposure` observation compared to a full-year policy. When you fit without weights, you weight all observations equally, which gives disproportionate influence to short-period policies that happened to have claims. This is wrong. (You can partially correct it by weighting by exposure, but you still cannot separate the offset from the mean function without a proper offset term.)

**Adding log(exposure) as a feature** lets sklearn fit a coefficient to it. In the true model, this coefficient is exactly 1.0 by construction. sklearn will estimate something close to 1.0 on a well-balanced portfolio, but it is not constrained to be 1.0, and on portfolios with unusual exposure distributions it drifts. More importantly, you are treating the offset as just another variable — it will interact with regularisation, affect feature importances, and contaminate anything downstream that uses the model structure.

```python
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import StandardScaler

# Approach 1: y/exposure target (common but wrong)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_te_sc = scaler.transform(X_te)

model_sk = TweedieRegressor(power=1.5, alpha=0.0, max_iter=1000)
model_sk.fit(X_tr_sc, y_tr / exp_tr, sample_weight=exp_tr)

pred_sk_rate = model_sk.predict(X_te_sc)

# Approach 2: log(exposure) as a feature (also wrong, but subtler)
X_tr_with_exp = np.column_stack([X_tr_sc, np.log(exp_tr)])
X_te_with_exp = np.column_stack([X_te_sc, np.log(exp_te)])

model_sk2 = TweedieRegressor(power=1.5, alpha=0.0, max_iter=1000)
model_sk2.fit(X_tr_with_exp, y_tr / exp_tr, sample_weight=exp_tr)

pred_sk2_rate = model_sk2.predict(X_te_with_exp)

# How wrong are they? Correlation with true annual rate.
r_sk,  _ = pearsonr(np.log(pred_sk_rate),  np.log(mu_rate_te))
r_sk2, _ = pearsonr(np.log(pred_sk2_rate), np.log(mu_rate_te))
print(f"sklearn (y/exp):          log-rate correlation {r_sk:.4f}")
print(f"sklearn (log-exp feature): log-rate correlation {r_sk2:.4f}")
```

The correlations will look reasonable — 0.85+. The issue is not that the model is badly wrong on average. It is that predictions for low-exposure policies are systematically biased. A policy with 0.05 exposure that happens to have a claim produces a `y/exposure` of 20× the claim amount. The model has no way to know this is a noisy observation; it tries to fit it.

---

## The correct way: statsmodels GLM with exposure offset

statsmodels supports proper offsets. The model structure is exactly what the maths requires.

```python
import statsmodels.api as sm
from statsmodels.genmod.families import Tweedie
from statsmodels.genmod.families.links import log as LogLink

X_tr_sm = sm.add_constant(X_tr)
X_te_sm = sm.add_constant(X_te)

glm = sm.GLM(
    y_tr,
    X_tr_sm,
    family=Tweedie(var_power=1.5, link=LogLink()),
    offset=np.log(exp_tr),   # <-- this is the key line
)
result = glm.fit()

# Predict on test set with correct exposure offset
pred_glm_rate = result.predict(X_te_sm, offset=np.log(exp_te)) / exp_te
# predict() returns E[Y] = mu_obs. Divide by exposure to recover annual rate.

r_glm, _ = pearsonr(np.log(pred_glm_rate), np.log(mu_rate_te))
print(f"statsmodels GLM (offset):  log-rate correlation {r_glm:.4f}")
print(result.summary())
```

The offset is not fitted. It enters the linear predictor as a fixed constant and ensures the model is estimating the *rate* rather than the total expected cost. The coefficient on `log(exposure)` is exactly 1.0 by construction — it does not need to be estimated.

This is what `sklearn.TweedieRegressor` cannot do.

---

## Why p=1.5 to 1.7 for motor, and what happens when you get it wrong

The Tweedie power parameter p controls the relationship between variance and mean: Var(Y) = phi * mu^p. For p between 1 and 2, Tweedie is the compound Poisson-Gamma distribution — a Poisson-distributed number of claims, each with Gamma-distributed severity. This is exactly the structure of insurance data.

For UK private motor, typical estimated values of p are 1.5 to 1.7. The exact value depends on the claims mix: bodily injury-heavy books tend toward higher p because severity is more dispersed (longer Gamma tail); predominantly accidental damage books sit lower.

You can estimate p empirically:

```python
from scipy.optimize import minimize_scalar
from scipy.stats import tweedie

def neg_ll(p):
    if p <= 1 or p >= 2:
        return 1e10
    # Approximate negative log-likelihood under Tweedie
    mu_hat = result.predict(X_tr_sm, offset=np.log(exp_tr))
    # Tweedie deviance as function of p
    d = 2 * np.mean(
        (y_tr ** (2 - p) / ((1 - p) * (2 - p)))
        - (y_tr * mu_hat ** (1 - p) / (1 - p))
        + (mu_hat ** (2 - p) / (2 - p))
    )
    return d

res_p = minimize_scalar(neg_ll, bounds=(1.1, 1.9), method='bounded')
print(f"Estimated Tweedie power: {res_p.x:.3f}")
```

Getting p wrong by 0.3 produces systematically miscalibrated predictions across the premium range. With p too low, the model underestimates variance for high-mean risks (and overestimates for low-mean risks). This biases safety loadings, distorts reinsurance pricing, and produces the kind of portfolio-level miscalibration that looks fine on a top-level A/E ratio but fails badly when you slice by claim severity band.

---

## Zero-inflation: why Tweedie handles this natively

On a typical UK motor book, 75–85% of policyholders file zero claims in any given year. This is not a modelling problem to be solved — it is baked into the compound Poisson-Gamma structure.

The Tweedie distribution with 1 < p < 2 has a point mass at zero. The probability of zero claims is:

```
P(Y = 0) = exp(-lambda)
```

where lambda = mu^(2-p) / (phi * (2-p)) is the Poisson claim count parameter. At typical motor parameters (mu ≈ 300, phi ≈ 0.8, p = 1.5), this gives P(Y=0) ≈ 0.83 — exactly the observed range.

You do not need a separate zero-inflation layer, a hurdle model, or a two-part frequency-severity approach to handle motor pure premium. Tweedie handles it. The zero-inflation is a consequence of the compound structure, not a separate modelling decision.

This is also why zero-inflated Tweedie (ZITE) models exist for lines with *excess* zeros beyond what the Poisson-Gamma structure implies — pet insurance with structural non-claimers, for instance. But for motor, standard Tweedie is correct.

---

## Gradient boosting: log(exposure) as feature vs offset

When you move from GLMs to gradient boosting (CatBoost, LightGBM, XGBoost), the offset question becomes more complicated.

**LightGBM and XGBoost** support an offset parameter natively. You can pass `offset=np.log(exposure)` and the boosting algorithm will treat it exactly as a GLM offset — fixed contribution to the linear predictor, not trained. This is the correct approach.

**CatBoost** does not support a native offset. The standard workaround is to pass log(exposure) as a feature with a fixed coefficient — which requires some additional machinery — or to use `y / exposure` as the target with sample weights. The `insurance-distributional` library handles this internally.

The critical point: passing `log(exposure)` as a free feature in any boosting model is wrong for the same reason it is wrong in sklearn. The tree structure can split on log(exposure), interact it with other features, and assign it whatever coefficient minimises the training loss. On a training set with systematic exposure patterns, the model will learn spurious relationships.

---

## insurance-distributional's TweedieGBM

`TweedieGBM` from `insurance-distributional` handles exposure correctly, fits both the mean and the dispersion (using the Smyth-Jorgensen double GLM approach), and gives you per-risk CoV on top.

```python
from insurance_distributional import TweedieGBM

model_tgbm = TweedieGBM(power=1.5)
model_tgbm.fit(X_tr, y_tr, exposure=exp_tr)

pred = model_tgbm.predict(X_te, exposure=exp_te)

# pred.mean is E[Y | X, exposure] — the expected cost for this specific exposure period
# Divide by exposure to recover annual rate
pred_tgbm_rate = pred.mean / exp_te

r_tgbm, _ = pearsonr(np.log(pred_tgbm_rate), np.log(mu_rate_te))
print(f"TweedieGBM (exposure-correct): log-rate correlation {r_tgbm:.4f}")

# Also outputs per-risk uncertainty — not available from GLMs or standard GBMs
print(f"Mean CoV across test set: {pred.volatility_score().mean():.3f}")
```

The `exposure` parameter is built into the model's objective function, not treated as a feature. The CatBoost objective uses the log(exposure) offset in the same way a GLM would — the model estimates the rate, and the predicted cost is rate × exposure.

The dispersion model is what separates this from a standard Tweedie GBM. By default, `TweedieGBM` fits a second gradient boosting model for phi, using squared Pearson residuals as the target (Smyth & Jorgensen, ASTIN 2002). This gives you `pred.phi` and `pred.volatility_score()` — the coefficient of variation per risk, which no commercial pricing tool currently outputs.

---

## Adding conformal prediction intervals

The `pred.mean` from `TweedieGBM` is a point estimate. For per-risk uncertainty — reinsurance attachment decisions, underwriter referral thresholds — you want a proper prediction interval.

`insurance-conformal` provides distribution-free intervals using the Pearson-weighted non-conformity score. Because Var(Y) ~ mu^p for Tweedie, the correct score is `|y - yhat| / yhat^(p/2)`. This produces intervals that are narrower for low-mean risks and wider for high-mean risks, rather than a uniform width that overcovers cheap policies and undercovers expensive ones.

```python
from insurance_conformal import InsuranceConformalPredictor

# Fit on a calibration split (not training data)
n_cal = 1_500
X_cal, X_val     = X_tr[-n_cal:], X_te
y_cal, y_val     = y_tr[-n_cal:], y_te
exp_cal, exp_val = exp_tr[-n_cal:], exp_te

# Re-fit TweedieGBM on reduced training set for clean calibration
model_conf = TweedieGBM(power=1.5)
model_conf.fit(X_tr[:-n_cal], y_tr[:-n_cal], exposure=exp_tr[:-n_cal])

cp = InsuranceConformalPredictor(
    model=model_conf,
    nonconformity="pearson_weighted",
    distribution="tweedie",
)
cp.calibrate(X_cal, y_cal, exposure=exp_cal)

intervals = cp.predict_interval(X_val, alpha=0.10)
# intervals[:, 0], intervals[:, 1] — 90% prediction interval per policy
```

On a heteroskedastic book, conformal intervals are typically 10–15% narrower than parametric Tweedie intervals at the same nominal coverage. The empirical coverage in the insurance-conformal benchmark is 0.902 vs 0.931 for parametric, with interval widths £3,806 vs £4,393 (50k synthetic UK motor policies, CatBoost Tweedie(1.5), temporal split). The intervals are wider where risk is genuinely higher, which is what you want when setting reinsurance attachments.

---

## What to use when

| Situation | Approach |
|-----------|----------|
| Regulatory GLM required (Solvency II, Lloyd's) | statsmodels GLM with `offset=np.log(exposure)` |
| Exploratory or prototype | statsmodels — clean, interpretable, fast |
| Production pricing, annual book with limited exposure variation | Either, with sample weights by exposure |
| Production pricing, mixed-duration or monthly-payment book | `TweedieGBM` from insurance-distributional |
| Need per-risk uncertainty (safety loading, RI, IFRS 17) | `TweedieGBM` + `pred.volatility_score()` |
| Need calibrated prediction intervals | `InsuranceConformalPredictor` on top of `TweedieGBM` |
| someone hands you sklearn code with `TweedieRegressor` | Replace it |

---

## The summary

`sklearn.TweedieRegressor` is fine for datasets where exposure is uniform across observations. Insurance pricing data is never uniform on exposure. The absence of an offset parameter is not a quirk — it is a structural limitation that makes the model wrong for the task.

The correct GLM implementation is statsmodels with `offset=np.log(exposure)`. For gradient boosting, `insurance-distributional`'s `TweedieGBM` handles exposure internally and adds per-risk dispersion estimation on top.

If you are building or inheriting a pricing model and the word "offset" does not appear anywhere in the codebase, that is the first thing to fix.

---

**Libraries used in this post:**

- `insurance-distributional` v0.2+ — [github.com/burning-cost/insurance-distributional](https://github.com/burning-cost/insurance-distributional)
- `insurance-conformal` — [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal)
- `statsmodels` — standard GLM with Tweedie family, `offset` parameter

**Related posts:**

- [Your Technical Price Ignores Variance](/2026/03/05/insurance-distributional/) — distributional GBM: mean + CoV per risk
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — distribution-free intervals, why parametric Tweedie intervals fail on heterogeneous books
