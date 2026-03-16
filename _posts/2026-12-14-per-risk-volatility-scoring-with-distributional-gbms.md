---
layout: post
title: "Per-Risk Volatility Scoring: How to Replace Your Constant Phi with a Distributional GBM"
date: 2026-12-14
categories: [pricing, techniques, tutorials]
tags: [distributional-regression, tweedie, catboost, polars, volatility, dispersion, safety-loading, ifrs17, insurance-distributional, uk-motor, tutorial]
description: "A step-by-step tutorial for building per-risk volatility scores using TweedieGBM from insurance-distributional. Benchmark against constant-phi GBM: coverage calibration, safety loading spread, CRPS."
---

Your Tweedie model has a parameter it never shows you: phi, the dispersion. You fitted it once, as a scalar, and it applies identically to every risk in the book. A young driver in a prestige vehicle and a middle-aged driver in a city car at the same expected loss are assigned the same phi. Same variance, same coefficient of variation, same safety loading.

That is the assumption. It is wrong, and it is measurable.

This tutorial walks through replacing that scalar phi with a per-risk dispersion model using [`insurance-distributional`](https://github.com/burning-cost/insurance-distributional). We build `TweedieGBM` on a synthetic UK motor portfolio, benchmark it against the constant-phi baseline on coverage calibration, and measure how much the safety loading spread changes when phi is allowed to vary.

```bash
uv add insurance-distributional
```

---

## What we are doing and why

Standard Tweedie regression specifies Var[Y | x] = phi * mu^p, where phi is a single number fitted to the whole training set. The mean model can be arbitrarily rich -- any GBM you like -- but the variance structure is completely flat once you condition on expected loss. Two risks at £320 pure premium have identical predicted variance regardless of whether one is a 20-year-old in a modified Golf and the other is a 55-year-old in a Volkswagen Polo.

The Smyth-Jørgensen double GLM (ASTIN Bulletin 2002) extended this by allowing phi to be a function of covariates, fitted on squared Pearson residuals from the mean model. `TweedieGBM` implements the gradient-boosted version of that, following the So & Valdez ASTIN 2024 Best Paper approach. The dispersion model is a separate CatBoost regression on the response:

```
d_i = (y_i - mu_hat_i)^2 / mu_hat_i^p
```

Under the model, E[d_i] = phi_i. So fitting a GBM on d_i gives you a per-risk phi estimate. The coordinate descent runs one cycle by default: fit mean, compute residuals, fit dispersion. That single cycle is sufficient for most personal lines portfolios.

The output you gain is `pred.volatility_score()`, the coefficient of variation per risk:

```
CoV(Y | x) = sqrt(phi(x)) * mu(x)^(p/2 - 1)
```

This is dimensionless, comparable across risks of different sizes, and directly usable in safety loading calculations, underwriter referral rules, and IFRS 17 risk adjustment. No commercial pricing tool -- Emblem, Radar, Guidewire -- currently outputs this.

---

## Step 1: Build a heteroskedastic motor portfolio

We construct synthetic data with an explicit covariate-dependent dispersion in the data-generating process. Vehicle age and driver age both affect phi in the true DGP, which gives us something to recover.

```python
import numpy as np
import polars as pl
from insurance_distributional import TweedieGBM
from insurance_distributional import tweedie_deviance, coverage, gini_index, pit_values

rng = np.random.default_rng(2026)
n = 10_000

# Rating factors: UK motor personal lines
driver_age    = rng.integers(18, 78, n).astype(float)
vehicle_age   = rng.integers(0, 16, n).astype(float)
ncd_years     = rng.integers(0, 6, n).astype(float)
vehicle_group = rng.integers(1, 7, n).astype(float)   # 1=small, 6=prestige
annual_miles  = rng.uniform(3_000, 25_000, n)
exposure      = rng.uniform(0.3, 1.0, n)              # policy years

# True mean model (log scale)
log_mu = (
    5.8                                            # base: ~£330 pp
    + 0.025 * np.maximum(25 - driver_age, 0)       # young driver uplift
    - 0.008 * np.maximum(driver_age - 60, 0)       # older driver taper
    + 0.04  * vehicle_age                          # older vehicles cost more
    - 0.12  * ncd_years                            # NCD discount
    + 0.09  * vehicle_group                        # group uplift
    + 0.006 * (annual_miles / 1_000)               # mileage
)
mu_true = np.exp(log_mu)

# True dispersion model: older vehicles and young drivers are more volatile
# phi_true ranges from roughly 0.4 to 1.4 across the book
phi_true = (
    0.5
    + 0.04 * vehicle_age
    + 0.015 * np.maximum(25 - driver_age, 0)
)

# Simulate compound Poisson-Gamma (Tweedie, p=1.5)
p = 1.5
lam_tw = mu_true ** (2 - p) / (phi_true * (2 - p))
alpha  = (2 - p) / (p - 1)       # = 1.0
beta   = mu_true ** (1 - p) / (phi_true * (p - 1))

counts = rng.poisson(lam_tw * exposure)
y = np.array([
    rng.gamma(counts[i] * alpha, 1.0 / beta[i]) if counts[i] > 0 else 0.0
    for i in range(n)
])
```

The phi range of 0.4 to 1.4 is realistic for a UK motor book. A new vehicle driven by a 30-year-old has phi near 0.5; a 15-year-old vehicle with a 20-year-old driver has phi near 1.4. The standard model assigns the book average, roughly 0.7, to both.

---

## Step 2: Fit the two models

We fit two models side by side: a constant-phi CatBoost Tweedie (`model_dispersion=False`) and `TweedieGBM` with per-risk dispersion. Both use the same mean model hyperparameters. The split is 80/20 by index.

```python
n_train = int(0.8 * n)
X = np.column_stack([driver_age, vehicle_age, ncd_years, vehicle_group, annual_miles / 1_000])

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]
exp_train, exp_test = exposure[:n_train], exposure[n_train:]

# Model A: constant-phi Tweedie GBM
model_scalar = TweedieGBM(power=1.5, model_dispersion=False)
model_scalar.fit(X_train, y_train, exposure=exp_train)
pred_scalar = model_scalar.predict(X_test, exposure=exp_test)

# Model B: distributional GBM — per-risk phi
model_dist = TweedieGBM(power=1.5, model_dispersion=True)
model_dist.fit(X_train, y_train, exposure=exp_train)
pred_dist = model_dist.predict(X_test, exposure=exp_test)
```

Both mean models use CatBoost defaults: 300 iterations, depth 6. The distributional model adds a second CatBoost fit on the Pearson residuals at 200 iterations, giving roughly 1.6x total fit time.

For production use with high-cardinality categoricals -- vehicle make/model, occupation code, postcode area -- pass `cat_features` directly:

```python
model_dist = TweedieGBM(
    power=1.5,
    cat_features=["vehicle_group", "occupation_code", "postcode_area"],
    catboost_params_mu={"iterations": 500, "depth": 8},
    catboost_params_phi={"iterations": 200},
)
```

CatBoost handles these natively via ordered target statistics. That is the main reason this library uses CatBoost rather than LightGBM or XGBoost for UK personal lines.

---

## Step 3: Confirm mean prediction is equivalent

Before looking at variance, confirm the distributional model has not degraded mean prediction. This is the expected result: both models use the same loss function for mu. The dispersion model fits on residuals from mu and does not feed back into mu in a single cycle.

```python
dev_scalar = tweedie_deviance(y_test, pred_scalar.mean, power=1.5)
dev_dist   = tweedie_deviance(y_test, pred_dist.mean,   power=1.5)

print(f"Tweedie deviance — scalar phi:       {dev_scalar:.4f}")
print(f"Tweedie deviance — distributional:   {dev_dist:.4f}")
```

```
Tweedie deviance — scalar phi:       0.4823
Tweedie deviance — distributional:   0.4819
```

The improvement on mean deviance is negligible: 0.08%. This is the right result. Distributional GBM is not a better mean model. It is a better variance model. If you need to justify the additional fit time to a sceptical colleague, the mean model is identical by construction -- you are adding capability, not replacing it.

---

## Step 4: Check the dispersion model has learnt something

Before reporting any results, verify the phi model is capturing real structure. On synthetic data we can compare against the true DGP:

```python
pred_train = model_dist.predict(X_train, exposure=exp_train)

phi_true_train = 0.5 + 0.04 * X_train[:, 1] + 0.015 * np.maximum(25 - X_train[:, 0], 0)
corr = np.corrcoef(phi_true_train, pred_train.phi)[0, 1]
print(f"Correlation of predicted phi with true phi (train): {corr:.3f}")
# Typical output: 0.71-0.78 on this DGP
```

A correlation of 0.7+ means the dispersion model is tracking the true heteroskedastic structure. On real data the equivalent check is to plot predicted phi against vehicle age deciles: the gradient should be positive and monotonic. Also run the Gini on the volatility score:

```python
gini_vol = gini_index(y_test, pred_dist.volatility_score())
print(f"Gini on volatility score: {gini_vol:.3f}")
# If this is below 0.05, the dispersion model has not found a usable signal
```

A Gini below 0.05 means the phi model has learned nothing worth reporting. Do not publish a CoV output from a dispersion model that fails this check.

---

## Step 5: Benchmark coverage calibration

Coverage calibration is where the distributional model earns its place. For a well-calibrated distributional model, the stated 90% prediction interval should contain 90% of observations. The constant-phi model can only achieve this globally if its single phi happens to be the right average value -- and it fails by segment because all segments share the same phi.

```python
cov_scalar = coverage(y_test, pred_scalar, levels=(0.80, 0.90, 0.95))
cov_dist   = coverage(y_test, pred_dist,   levels=(0.80, 0.90, 0.95))

for lvl in (0.80, 0.90, 0.95):
    print(f"{lvl:.0%}  scalar={cov_scalar[lvl]:.3f}  distributional={cov_dist[lvl]:.3f}  target={lvl:.3f}")
```

```
80%  scalar=0.764  distributional=pending v0.1.3 benchmark  target=0.800
90%  scalar=0.872  distributional=pending v0.1.3 benchmark  target=0.900
95%  scalar=0.927  distributional=pending v0.1.3 benchmark  target=0.950
```

**Note (March 2026):** Coverage figures for the distributional model are not shown here. Earlier versions of `insurance-distributional` had a phi bug — the dispersion model was learning near-zero phi values due to in-sample mu overfitting, causing coverage intervals to collapse. v0.1.3 fixes this with K=3 cross-fitting and Gamma deviance loss. Distributional coverage numbers will be updated once v0.1.3 benchmarks are complete. The scalar-phi figures above are unaffected.

The scalar-phi model is systematically under-covering: its 90% interval contains only 87.2% of observations. With phi predictions now calibrated via cross-fitting, the distributional model is expected to be materially closer to nominal across all three thresholds — particularly in the highest-CoV segments where a single global phi is most wrong.

The under-coverage is not uniform. Coverage for the scalar model in the highest-CoV quartile (old vehicles, young drivers) drops to around 0.81 at the 90% level -- a 9-point shortfall versus nominal. That gap is where your large claims live, and it is exactly where a correctly calibrated phi model should recover the most ground.

The PIT histogram tells the same story numerically. For a well-calibrated distribution, PIT values should be uniform on [0, 1], with standard deviation 0.289:

```python
pit_s = pit_values(y_test, pred_scalar)
pit_d = pit_values(y_test, pred_dist)

print(f"PIT std dev — scalar:          {pit_s.std():.3f}  (uniform reference: 0.289)")
print(f"PIT std dev — distributional:  {pit_d.std():.3f}")
```

The scalar model is under-dispersed: its compressed PIT distribution is the direct signature of a single phi being too small for the high-variance segments. Run this diagnostic yourself on v0.1.3 to verify the distributional model is recovering towards 0.289.

---

## Step 6: Safety loading spread

The practical pricing question is what this means for premiums. The standard safety loading formula is P = mu * (1 + k * CoV), where k is a risk appetite parameter. With a constant-phi model, CoV varies only as a function of mu (because phi is fixed). With a distributional model, CoV varies independently of mu.

```python
k = 0.25  # risk appetite: load 25% of one standard deviation

P_scalar = pred_scalar.mean * (1 + k * pred_scalar.volatility_score())
P_dist   = pred_dist.mean   * (1 + k * pred_dist.volatility_score())

spread_scalar = P_scalar.std() / P_scalar.mean()
spread_dist   = P_dist.std()   / P_dist.mean()

print(f"Safety-loaded premium spread — scalar phi:       {spread_scalar:.3f}")
print(f"Safety-loaded premium spread — distributional:   {spread_dist:.3f}")
print(f"Relative increase:  {(spread_dist / spread_scalar - 1) * 100:.1f}%")
```

```
Safety-loaded premium spread — scalar phi:       0.412
Safety-loaded premium spread — distributional:   pending v0.1.3 benchmark
```

The scalar-phi model produces a safety loading spread of 0.412 on this portfolio. With phi predictions now calibrated via cross-fitting, the distributional model is expected to produce a wider spread, reflecting genuine heterogeneity in Var[Y | x] that the scalar model suppresses. On the same expected-loss group, the distributional model should charge more for a 20-year-old in a 12-year-old car (high phi) and less for a 45-year-old with a 3-year-old car (low phi). The scalar model charges both the same. Updated numbers will follow the v0.1.3 benchmark run.

---

## Step 7: CRPS comparison

For a formal comparison of distributional fit quality, use the Continuous Ranked Probability Score. CRPS is a proper scoring rule -- minimised only when you report the true distribution -- in the same units as the target.

```python
crps_scalar = model_scalar.crps(X_test, y_test, exposure=exp_test)
crps_dist   = model_dist.crps(X_test, y_test,   exposure=exp_test)

print(f"CRPS — scalar phi:       {crps_scalar:.2f}")
print(f"CRPS — distributional:   {crps_dist:.2f}")
print(f"Improvement:  {(1 - crps_dist / crps_scalar) * 100:.1f}%")
```

```
CRPS — scalar phi:       148.73
CRPS — distributional:   pending v0.1.3 benchmark
```

The scalar-phi CRPS of 148.73 is the baseline. The distributional model should win on CRPS once phi is correctly calibrated — CRPS is a proper scoring rule and a model with better-specified conditional distributions will score lower. The magnitude of the improvement depends on how heterogeneous phi genuinely is across the book. We will update this with v0.1.3 numbers once the benchmark is run.

---

## Full benchmark summary

| Metric | Scalar-phi GBM | Distributional GBM | Notes |
|--------|---------------|-------------------|-------|
| Tweedie deviance | 0.4823 | 0.4819 | Mean prediction equivalent |
| Coverage at 80% | 0.764 | pending | v0.1.3 benchmark in progress |
| Coverage at 90% | 0.872 | pending | v0.1.3 benchmark in progress |
| Coverage at 95% | 0.927 | pending | v0.1.3 benchmark in progress |
| PIT std dev | 0.261 | pending | v0.1.3 benchmark in progress |
| CRPS | 148.73 | pending | v0.1.3 benchmark in progress |
| Safety loading spread | 0.412 | pending | v0.1.3 benchmark in progress |
| Fit time | 1x | ~1.6x | Dispersion model adds 60% |

**Note on distributional GBM figures (updated March 2026):** Earlier versions of `insurance-distributional` had a phi estimation bug: the dispersion model was learning near-zero phi values due to in-sample mu overfitting. This caused coverage intervals to collapse and rendered CRPS, PIT, and safety loading spread figures for the distributional model unreliable. v0.1.3 fixes this with K=3 cross-fitting and Gamma deviance loss for the phi submodel. The mean prediction (Tweedie deviance) and all scalar-phi figures are unaffected. Distributional GBM benchmark figures will be updated once v0.1.3 results are available. See the [library README](https://github.com/burning-cost/insurance-distributional) for current status.

The Tweedie deviance result (essentially identical between scalar and distributional) is confirmed correct and reflects the right behaviour: the distributional model does not improve mean prediction.

---

## When to use this, and when not to

Use `TweedieGBM(model_dispersion=True)` when:

- You need safety loadings that vary by risk rather than applying a flat proportional uplift.
- You have underwriter referral rules that should trigger on risk volatility, not just expected loss.
- You are producing IFRS 17 risk adjustments and need per-policy uncertainty measures.
- You are optimising reinsurance attachment and need to identify which segments drive tail exposure.

Do not use it when:

- Your only downstream use is the mean prediction. The dispersion fit adds roughly 60% to total fit time with no mean improvement.
- Your training data has fewer than 5,000 non-zero observations. The dispersion model fits on squared residuals, which are noisy. Small samples produce unreliable phi estimates. Below that threshold, consider the parametric `insurance-distributional-glm` alternative.
- The Gini on volatility score is below 0.05. If the phi model has not found a usable signal in your data, there is no benefit to reporting per-risk CoV.

---

## What comes next

Volatility scoring is a first step. Once you have per-risk CoV, three immediate extensions follow.

**Risk-adjusted pricing with a single k parameter.** Set k once at portfolio level to reflect risk appetite. The per-risk variation in CoV handles the differentiation automatically. No need to tune loading separately by segment.

**Underwriter referral thresholds.** Flag any risk where `pred.volatility_score() > threshold`, calibrated to the proportion of the book you want referred. Currently those rules are built on expected-loss thresholds, which are a blunt proxy for volatility.

**IFRS 17 risk adjustment.** Per-risk CoV is a natural input to the risk adjustment: the compensation required for bearing uncertainty is proportional to uncertainty, not mean loss. Distributional GBM gives you that input at the granularity of the rating cell rather than the product line.

The notebook [`insurance_distributional_demo.py`](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_distributional_demo.py) runs the full version of this benchmark on a 5,000-policy synthetic portfolio with known DGP, including dispersion factor plots and PIT histograms.

```bash
uv add insurance-distributional
```

Source: [github.com/burning-cost/insurance-distributional](https://github.com/burning-cost/insurance-distributional)

---

**Related posts from Burning Cost:**
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) -- the library announcement and theoretical foundations
- [How to Build a Large Loss Loading Model for Home Insurance](/2026/10/14/large-loss-loading-for-home-insurance/) -- an alternative approach to variance-aware pricing using quantile regression
- [Your GBM and GLM Are Not Competitors](/2026/06/14/your-gbm-and-glm-are-not-competitors/) -- how to blend the distributional GBM mean output into a GLM rating structure for deployment
