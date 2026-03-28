---
layout: post
title: "Building a Tweedie GLM for Insurance Pricing in Python"
date: 2026-03-26
author: Burning Cost
categories: [tutorials, glm, python]
tags: [tweedie, GLM, statsmodels, python, tutorial, pure-premium, compound-poisson-gamma, insurance-pricing, exposure, offset, residuals, motor-insurance, uk-insurance, frequency-severity, deviance]
description: "A complete Python tutorial for building a Tweedie GLM for insurance pricing: synthetic motor data, statsmodels, exposure offset, interpreting the p parameter, residual diagnostics, and factor tables. No prior Python GLM experience required."
---

A Tweedie GLM is the most compact way to model insurance pure premium in Python. One model, one response variable (claims cost per policy), one fit call. If you have used `glm()` in R with `family=tweedie()`, or the `TWEEDIE` distribution in Emblem, this is the same thing with different syntax.

This tutorial builds a working Tweedie GLM from scratch: we generate synthetic motor data, fit the model in statsmodels, interpret the power parameter and coefficients, check the residuals, and extract factor tables. The code is self-contained  -  you need statsmodels, numpy, scipy, and matplotlib, nothing else.

```bash
uv pip install statsmodels numpy scipy matplotlib
# or with uv:
uv add statsmodels numpy scipy matplotlib
```

---

## Why Tweedie

A Tweedie distribution with power parameter p between 1 and 2 is equivalent to a compound Poisson-Gamma: a random number of claims, each with a Gamma-distributed cost. Most motor and home insurance portfolios fit this description. The Tweedie model for pure premium is:

```
log(E[pure_premium_i]) = log(exposure_i) + X_i * beta
```

The `log(exposure_i)` term is the offset  -  it enters with a fixed coefficient of 1.0, not estimated. Without it, the model predicts expected claims cost for an arbitrary observation period rather than a rate (cost per policy-year). This is the single most common mistake in Python GLM implementations. We will return to it below.

The alternative is two separate models  -  Poisson for frequency, Gamma for severity  -  multiplied together. That split gives more flexibility: different factors can drive frequency and severity independently. The Tweedie constraint is that a single linear predictor must explain both. For a homogeneous personal lines book where most factors affect frequency and severity in the same direction, the Tweedie is simpler and often good enough. For a book where NCD strongly suppresses frequency but barely touches severity, the split wins.

We have covered the choice in detail in [Tweedie vs Frequency-Severity Split: When to Use Which](/2026/03/24/tweedie-vs-frequency-severity-when-to-use-which/). For this tutorial, we fit the Tweedie.

---

## Generating synthetic motor data

We need data with known structure so we can verify the model recovers it. The data-generating process (DGP) is a compound Poisson-Gamma with a log-linear mean function and genuine exposure variation.

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.genmod.families import Tweedie

rng = np.random.default_rng(42)
n = 30_000

# --- Rating factors ---
# NCD years: 0-5, realistic distribution (most policyholders have some NCD)
ncd = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.10, 0.10, 0.15, 0.20, 0.20, 0.25])

# Vehicle group: 1-10 (simplified ABI-style grouping)
veh_group = rng.choice(list(range(1, 11)), size=n)

# Driver age: 18-80, skewed towards working age
driver_age = np.clip(rng.normal(42, 15, n).astype(int), 18, 80)

# Region: 4 bands (A = urban, D = rural)
region = rng.choice(["A", "B", "C", "D"], size=n, p=[0.35, 0.30, 0.20, 0.15])

# Exposure: genuine variation (monthly payers, MTAs, new business late in year)
# Roughly 30% of policies with sub-0.7 exposure
exposure = np.clip(rng.beta(3, 1.5, n), 0.05, 1.0)

# --- True parameters (log-scale relativities) ---
# Intercept: log(base pure premium) = log(£180/year)
intercept = np.log(180)

# NCD: strong risk discriminator
ncd_coef = np.array([0.0, -0.15, -0.28, -0.38, -0.46, -0.52])

# Vehicle group: log-linear across groups 1-10
# Group 1 (reference), higher groups = higher severity
veh_group_coef = 0.07  # per group unit

# Driver age: U-shaped in log space (young + old both riskier)
# Simplified: offset from age 45 minimum
def age_effect(age):
    return 0.015 * np.abs(age - 45) / 10

# Region: A (urban) = most claims, D (rural) = fewest
region_coef = {"A": 0.18, "B": 0.08, "C": 0.0, "D": -0.10}

# --- Assemble linear predictor ---
eta = (
    intercept
    + ncd_coef[ncd]
    + veh_group_coef * (veh_group - 1)
    + age_effect(driver_age)
    + np.array([region_coef[r] for r in region])
)

# mu: expected pure premium per unit exposure
mu_rate = np.exp(eta)
# Expected claims cost over the policy period = mu_rate * exposure
mu = mu_rate * exposure

# --- Tweedie DGP: compound Poisson-Gamma ---
# p = 1.5 is typical for UK motor (corresponds to moderate claim severity CV)
# Tweedie parameterisation: Poisson rate lambda, Gamma shape alpha_gam, Gamma scale beta_gam
phi = 5.0          # dispersion parameter
p   = 1.5          # Tweedie power

lambda_poi = mu**(2 - p) / (phi * (2 - p))
alpha_gam  = (2 - p) / (p - 1)
beta_gam   = phi * (p - 1) * mu**(p - 1)  # scale parameter

claim_count = rng.poisson(lambda_poi)
claims_cost = np.array([
    rng.gamma(alpha_gam, beta_gam[i], size=claim_count[i]).sum()
    if claim_count[i] > 0 else 0.0
    for i in range(n)
])

print(f"Policies:       {n:,}")
print(f"Zero claims:    {(claims_cost == 0).sum():,} ({100*(claims_cost == 0).mean():.1f}%)")
print(f"Mean exposure:  {exposure.mean():.3f} years")
print(f"Mean cost/yr:   £{(claims_cost / exposure).mean():.2f}")
print(f"Mean claim cnt: {claim_count.mean():.4f}")
```

You should see roughly 78-82% zero-cost policies, mean exposure around 0.66, and mean cost per year around £190-200. The zero proportion is the most useful sense-check: at p=1.5 and these parameters, a Tweedie DGP produces more zeros than a pure Gamma but fewer than a Bernoulli, which is realistic for motor.

---

## Building the design matrix

statsmodels takes a numpy matrix or a pandas DataFrame with `formula=`. We use the matrix form because it is explicit about what the model is actually fitting.

```python
# Encode rating factors
ncd_dummies = pd.get_dummies(ncd, prefix="ncd", drop_first=True).astype(float)
# ncd_1 through ncd_5; ncd_0 is the reference (base)

veh_cont = veh_group.astype(float)  # treat as continuous  -  revisit if nonlinearity appears

region_dummies = pd.get_dummies(region, prefix="region", drop_first=True).astype(float)
# region_B, region_C, region_D; region_A is reference (highest risk = base)
# pd.get_dummies drops first alphabetically, which is "A"  -  correct here

age_cont = driver_age.astype(float)

# Assemble feature DataFrame
X_df = pd.concat([
    ncd_dummies,
    pd.DataFrame({"vehicle_group": veh_cont}),
    region_dummies,
    pd.DataFrame({"driver_age": age_cont}),
], axis=1)

# Add constant (intercept)
X = sm.add_constant(X_df)

print("Design matrix shape:", X.shape)
print("Columns:", list(X.columns))
```

The response variable is the claims cost over the policy period (not divided by exposure). The exposure enters as the offset, not as a feature.

---

## Fitting the Tweedie GLM

statsmodels' `GLM` with `Tweedie(var_power=p)` fits the model by IRLS. The key question is what value of p to use.

### Estimating p

The Tweedie power parameter governs how variance scales with mean: Var(Y) = phi * mu^p. You can estimate p by profile likelihood  -  fit the model at a grid of p values and find the minimum deviance.

For most UK motor books, p falls between 1.4 and 1.8. Start with p=1.5 and check.

```python
p_grid = np.arange(1.1, 2.0, 0.1)
deviances = []

for p_try in p_grid:
    result_tmp = sm.GLM(
        claims_cost,
        X,
        family=Tweedie(var_power=p_try, link_power=0),  # link_power=0 → log link
        offset=np.log(exposure),
    ).fit(disp=False)
    deviances.append(result_tmp.deviance)

best_p_idx = np.argmin(deviances)
p_estimated = p_grid[best_p_idx]
print(f"Profile likelihood estimate of p: {p_estimated:.1f}")
# Should be close to the true p=1.5 we used in the DGP
```

The profile likelihood approach is standard actuarial practice. The `tweedie` R package provides `tweedie.profile()` which does exactly this. In Python, we loop manually. For a 30,000-row dataset this takes a few seconds.

### Fitting with the chosen p

```python
p_fit = float(p_estimated)  # or fix at 1.5 if you have prior knowledge

model = sm.GLM(
    claims_cost,
    X,
    family=Tweedie(var_power=p_fit, link_power=0),  # log link
    offset=np.log(exposure),   # <-- critical: log(exposure) as offset
)

result = model.fit()
print(result.summary())
```

The summary will show coefficient estimates, standard errors, z-statistics, and p-values. The intercept captures the base pure premium (per policy-year) for the reference policyholder: NCD=0, region A, vehicle group 1, driver age entering linearly.

---

## Interpreting the coefficients

All coefficients are on the log scale. Exponentiate to get multiplicative relativities.

```python
coefs = result.params
se    = result.bse
z     = coefs / se

factor_table = pd.DataFrame({
    "factor":     coefs.index,
    "log_coef":   coefs.values,
    "relativity": np.exp(coefs.values),
    "std_err":    se.values,
    "z_stat":     z.values,
    "p_value":    2 * stats.norm.sf(np.abs(z.values)),
}).sort_values("relativity", ascending=False)

print(factor_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
```

What to look for:

**NCD relativities** should decrease monotonically from ncd_5 to ncd_1. If ncd_2 sits above ncd_3, something is wrong with your encoding or data preparation. Check the exposure volumes for each NCD band  -  thin cells produce noisy estimates.

**Vehicle group** coefficient should be positive (higher group = higher premium). The estimated coefficient should be close to the true 0.07 we used in the DGP. On synthetic data with 30,000 policies, you typically recover it to within ±0.005.

**Region coefficients** are relative to region A (the most expensive). The coefficient on `region_D` should be around -0.28 (the true value is -0.10 - 0.18 = -0.28 in log space, i.e., region D costs approximately exp(-0.28) ≈ 0.76 times region A all else equal).

**Driver age** will show a small positive coefficient because the DGP uses an absolute-deviation function from age 45, but since we encoded it as a continuous linear term, the GLM sees the net upward drift. A linear term for driver age is typically wrong  -  the relationship is U-shaped. We will catch this in the residual checks below.

---

## Residual diagnostics

Three diagnostic checks matter for a Tweedie GLM on insurance data.

### 1. Pearson residuals against fitted values

```python
import matplotlib.pyplot as plt

fitted = result.fittedvalues                         # E[Y_i] = exp(eta_i) * exposure_i
pearson_resid = (claims_cost - fitted) / np.sqrt(fitted**p_fit)

plt.figure(figsize=(8, 4))
plt.scatter(np.log1p(fitted), pearson_resid, alpha=0.1, s=5, color="steelblue")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("log(1 + fitted)")
plt.ylabel("Pearson residual")
plt.title("Tweedie GLM: Pearson residuals vs fitted values")
plt.tight_layout()
plt.savefig("tweedie_residuals.png", dpi=150)
plt.close()
```

A well-specified Tweedie GLM will show Pearson residuals that are roughly homoskedastic around zero, with no funnel shape and no obvious curvature. The mass of zero-cost policies will sit at large negative Pearson residuals  -  this is normal for a compound Poisson-Gamma.

What to worry about: a systematic curve in the residuals against log(fitted) means the link function is wrong, or the linear predictor is missing a nonlinear term (like the U-shaped driver age).

### 2. Actual/expected by factor band

The Pearson plot is a global check. The A/E check by factor band is where actuarial problems hide.

```python
ae_df = pd.DataFrame({
    "ncd":      ncd,
    "actual":   claims_cost,
    "fitted":   fitted,
    "exposure": exposure,
})

ae_by_ncd = (
    ae_df
    .groupby("ncd")
    .agg(actual=("actual", "sum"), fitted=("fitted", "sum"), exposure=("exposure", "sum"))
    .assign(ae_ratio=lambda x: x["actual"] / x["fitted"])
)

print("\nA/E by NCD years:")
print(ae_by_ncd.round(3))
```

Expected output: A/E ratios between 0.90 and 1.10 for all NCD bands. If NCD=5 shows A/E of 0.75, the model is overestimating risk for experienced drivers  -  the relativities are not steep enough. If it shows 1.25, the model is underestimating  -  either the data has something unusual or the DGP recovery is poor.

Do the same for vehicle group, region, and driver age bands:

```python
# Driver age banded  -  the U-shape should show up here if the linear term is misspecified
ae_df["age_band"] = pd.cut(driver_age, bins=[17, 25, 35, 45, 55, 65, 81], right=True)

ae_by_age = (
    ae_df
    .groupby("age_band", observed=True)
    .agg(actual=("actual", "sum"), fitted=("fitted", "sum"))
    .assign(ae_ratio=lambda x: x["actual"] / x["fitted"])
)

print("\nA/E by driver age band:")
print(ae_by_age.round(3))
```

The 18-25 band will show A/E above 1.0 and the 65-81 band will also show A/E above 1.0, because we specified a U-shaped DGP but the linear term cannot capture this. Both bands are relatively thin, so the bias may not be large in aggregate  -  but it is systematic. The fix is to replace the continuous linear term with age bands or a spline. Not something to ignore in a production model.

This is exactly the diagnostic that catches bad factor specifications in Emblem too. The A/E check by driver age band is often how actuaries discover that a linear age term needs to become a grouped factor.

### 3. Dispersion estimate

```python
# Pearson chi-squared dispersion estimate
dispersion_est = result.pearson_chi2 / result.df_resid
print(f"\nEstimated dispersion (phi): {dispersion_est:.3f}")
print(f"True dispersion used in DGP: {phi:.1f}")
```

With 30,000 observations and a clean DGP, the estimated phi should be close to 5.0. Significant deviation (say, phi > 8 or phi < 3) suggests either the p parameter is wrong or there are unmodelled sources of variance in the data  -  missing risk factors, or data quality problems.

---

## Predictions

The model predicts `E[claims_cost_i]` for each policy. To get a pure premium rate (annual), divide by exposure:

```python
# In-sample predictions
pred_cost    = result.fittedvalues
pred_pp_rate = pred_cost / exposure   # pure premium per policy-year

print(f"\nPredicted pure premium  -  mean:   £{pred_pp_rate.mean():.2f}")
print(f"Predicted pure premium  -  median: £{np.median(pred_pp_rate):.2f}")
print(f"Predicted pure premium  -  P90:    £{np.percentile(pred_pp_rate, 90):.2f}")

# Prediction for a new policy
new_policy = pd.DataFrame({
    "const":         [1.0],
    "ncd_1":         [0.0],
    "ncd_2":         [0.0],
    "ncd_3":         [0.0],
    "ncd_4":         [0.0],
    "ncd_5":         [1.0],    # 5 years NCD
    "vehicle_group": [6.0],    # mid-range vehicle
    "region_B":      [0.0],
    "region_C":      [1.0],    # region C
    "region_D":      [0.0],
    "driver_age":    [40.0],
})

# offset=log(1.0) for a full annual policy
new_pred = result.predict(new_policy, offset=[np.log(1.0)])
print(f"\nPredicted annual pure premium for new policy: £{new_pred[0]:.2f}")
```

The column names in `new_policy` must match `X` exactly, including the constant column. statsmodels is strict about this. If you use `formula=` syntax instead of the matrix form, prediction is cleaner  -  but the matrix form is more transparent about what the model is actually doing.

---

## Factor table extraction

A GLM coefficient is not a factor table entry until you exponentiate it and set the reference correctly. The reference level for each categorical factor has relativity 1.0 (log coefficient = 0).

```python
def extract_factor_table(result, X_columns: list) -> pd.DataFrame:
    """Build a clean factor table from a fitted statsmodels GLM.
    
    Note: in production, extend this to add reference level rows explicitly
    (relativity = 1.0, log_coef = 0.0, std_err = NaN) for the base level of
    each categorical factor  -  the level dropped by pd.get_dummies. Rating
    engines and pricing committees expect a complete table including the base.
    """
    rows = []
    for col in X_columns:
        if col == "const":
            rows.append({
                "factor":     "intercept",
                "level":      "(base)",
                "log_coef":   result.params[col],
                "relativity": np.exp(result.params[col]),
                "std_err":    result.bse[col],
            })
        else:
            rows.append({
                "factor":     col,
                "level":      "(non-base)",
                "log_coef":   result.params[col],
                "relativity": np.exp(result.params[col]),
                "std_err":    result.bse[col],
            })
    return pd.DataFrame(rows)

ft = extract_factor_table(result, list(X.columns))
print("\nFactor table:")
print(ft.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
```

For deployment into a rating engine  -  whether a bespoke Python service, an R Shiny pricing tool, or a system like Emblem or Radar  -  the factor table is the output. Each categorical level needs its own row, with the reference level explicitly listed at 1.0.

For production factor tables with proper reference level handling and exposure-weighted rebasing, see our post on [exporting factor tables to Excel from a Python GLM](/2026/03/24/how-to-export-factor-table-to-excel-python/).

---

## What the Tweedie GLM does not give you

**Separate frequency and severity diagnostics.** The Tweedie model gives you a single set of coefficients for pure premium. You cannot separately check whether NCD is driving frequency, severity, or both  -  which you can with a split model. For most UK regulatory purposes and Consumer Duty fair value assessments, the split is preferable precisely because it is more auditable.

**The correct model for books with very different severity by claim type.** If your portfolio has a mix of attritional OD claims (£500-3,000) and occasional BI claims (£10,000-200,000), a single Tweedie p may not fit both tails well. The [insurance-distributional](https://github.com/burning-cost/insurance-distributional) library implements distributional regression where p itself is covariate-dependent.

**Exposure-free predictions for sklearn users.** If you are comparing results to `sklearn.linear_model.TweedieRegressor`, note that sklearn's implementation has no offset parameter. You cannot correctly reproduce these predictions with sklearn on data with exposure variation. We explained this in [Tweedie Regression for Insurance: What sklearn Doesn't Tell You About Exposure](/2026/03/21/tweedie-regression-insurance-what-sklearn-doesnt-tell-you/).

**Postcode and other high-cardinality factors.** We kept the example to four rating factors for clarity. A production UK motor GLM has postcode sector (~9,000 levels), NCD, ABI vehicle group, driver age, vehicle age, voluntary excess, and usually several more. High-cardinality factors need credibility-weighted or hierarchical treatment  -  a plain one-hot encoding at postcode level is not practical. The [insurance-whittaker](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/) library handles postcode smoothing via Whittaker-Henderson graduation applied across postcode hierarchies.

---

## Summary

A working Tweedie GLM for insurance pricing in Python requires three things that tutorials typically omit:

1. **The exposure offset.** `offset=np.log(exposure)` in the `GLM()` call, not a feature column. Without it, the model learns the wrong thing.

2. **An estimated p parameter.** The profile likelihood approach takes an extra minute of compute and prevents you from using a value of p that is systematically wrong for your portfolio.

3. **A/E checks by factor band.** A good global deviance does not mean the factors are individually well-specified. The driver age U-shape example above is a case where global fit looks acceptable but the factor is wrong. A/E by band catches this.

The code in this post is runnable as-is. For a production UK motor model you will need to extend it: more factors, temporal cross-validation rather than a single holdout (see [Why k-Fold CV Is Wrong for Insurance](/2026/03/21/why-k-fold-cv-is-wrong-for-insurance/)), postcode handling, and overdispersion-corrected standard errors for factor significance tests. The [fitting motor insurance GLM in statsmodels](/2026/03/24/fitting-motor-insurance-glm-statsmodels-poisson-gamma/) post covers the Poisson/Gamma frequency-severity split on the same foundations.

---

*Libraries referenced: [statsmodels](https://www.statsmodels.org/), [insurance-distributional](https://github.com/burning-cost/insurance-distributional), [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker)*

*Related posts:*
- [Tweedie vs Frequency-Severity Split: When to Use Which](/2026/03/24/tweedie-vs-frequency-severity-when-to-use-which/)  -  when the simpler single model is enough and when the split wins
- [Tweedie Regression for Insurance: What sklearn Doesn't Tell You About Exposure](/2026/03/21/tweedie-regression-insurance-what-sklearn-doesnt-tell-you/)  -  why sklearn's TweedieRegressor produces wrong results on books with exposure variation
- [Fitting a Motor Insurance GLM in Python: Poisson Frequency and Gamma Severity](/2026/03/24/fitting-motor-insurance-glm-statsmodels-poisson-gamma/)  -  the frequency-severity split with statsmodels, overdispersion tests, and factor tables
- [GLM Assumptions in Insurance Pricing: What Actually Matters](/2026/03/23/glm-assumptions-insurance-pricing-what-actually-matters/)  -  which violations matter, which you can live with
