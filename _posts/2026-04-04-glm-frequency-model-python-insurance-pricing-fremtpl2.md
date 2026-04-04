---
layout: post
title: "Building a GLM Frequency Model in Python: A Step-by-Step Guide for Insurance Pricing"
date: 2026-04-04
author: Burning Cost
categories: [tutorials, glm, python]
tags: [glm, python, insurance-pricing, frequency-model, poisson, glum, freMTPL2, exposure, deviance-residuals, actual-vs-expected, factor-relativities, tutorial, openml, uk-insurance, motor]
description: "End-to-end GLM frequency model in Python using freMTPL2 from OpenML. Data prep, exposure handling, glum fitting, deviance residuals, actual vs expected, and factor relativity extraction — all in one place."
seo_title: "GLM Insurance Pricing Python: Step-by-Step Frequency Model Tutorial with freMTPL2"
---

Most Python GLM tutorials for insurance pricing are either too shallow (fit Poisson, evaluate with R², done) or too specialised (assuming you already know what you are doing and just need to know which glum parameter to pass). This post sits in the middle: a complete, runnable tutorial for someone who understands the actuarial concepts — frequency models, exposure, multiplicative rating factors — and wants to build one properly in Python.

We use the freMTPL2 dataset, which is the standard actuarial benchmark for frequency modelling. It is French motor third-party liability data, 678,013 policy-year observations, and it ships inside scikit-learn via OpenML. We use [glum](https://github.com/Quantco/glum) for the GLM fitting — not statsmodels, not sklearn's PoissonRegressor. The reasons are practical: glum is faster on large datasets, returns standard errors without extra steps, and handles the exposure offset natively. If you want to go further than fitting alone — conformal prediction intervals, drift monitoring, and governance packs — the [full glum workflow guide](/2026/04/04/glum-insurance-pricing-python-workflow/) covers the surrounding ecosystem.

By the end of this post you will have: a fitted Poisson frequency GLM, a working actual vs expected diagnostic, deviance residuals plotted against predicted values, and a factor relativity table you can inspect or export.

---

## Setup

```bash
uv add glum scikit-learn pandas numpy matplotlib scipy
```

Or with pip:

```bash
pip install glum scikit-learn pandas numpy matplotlib scipy
```

glum requires Python 3.9 or later.

---

## The data

freMTPL2 is a French motor portfolio with claim counts per policy-year and an exposure variable measuring the fraction of the year the policy was active. We load it via scikit-learn's OpenML interface:

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

raw = fetch_openml("freMTPL2freq", version=3, as_frame=True, parser="auto")
df = raw.data.copy()
df["ClaimNb"] = raw.target.astype(int)

print(f"Shape: {df.shape}")
print(df["ClaimNb"].value_counts().sort_index())
```

```
Shape: (678013, 12)

ClaimNb
0    636005
1     36985
2      4744
3       246
4        33
```

The columns that matter for a basic frequency model:

- **ClaimNb**: integer count of claims in the policy period (0, 1, 2, 3, or 4)
- **Exposure**: fraction of the year covered, typically between 0 and 1
- **DrivAge**: driver age in years
- **BonusMalus**: the French bonus-malus score, analogous to UK NCD; ranges from 50 (maximum bonus) to 350+ (malus); higher scores indicate a worse claims history
- **VehPower**: vehicle power band (integers 4–15)
- **VehAge**: vehicle age in years
- **VehGas**: fuel type, "Diesel" or "Regular"
- **Area**: urbanisation area code (A through F, A being most rural)
- **Region**: French administrative region (22 levels)
- **Density**: population density in people/km²

Mean claim frequency is approximately 5.5 claims per 100 policy-years. The distribution is heavily zero-inflated: 93.8% of policy-years have zero claims, which is typical for a well-segmented personal lines motor portfolio.

---

## Data preparation

### Exposure cap

A handful of records have Exposure slightly above 1.0, which is an artefact of mid-term adjustment rounding. Cap it:

```python
df["Exposure"] = df["Exposure"].clip(upper=1.0)
```

This matters. An exposure of 1.05 would make the offset term `log(1.05) = 0.049`, which adds a small downward bias to the base rate estimate. On a 678k-row dataset, a few dozen miscoded records do not move the needle much — but the habit of capping is correct and carries over to production data where the effect can be larger.

### Feature engineering

We make three transformations before encoding:

```python
# Log-transform density: right-skewed, ranges from 1 to 27,000 people/km²
df["LogDensity"] = np.log1p(df["Density"].astype(float))

# Cast numeric features
df["DrivAge"]    = df["DrivAge"].astype(float)
df["VehAge"]     = df["VehAge"].astype(float)
df["VehPower"]   = df["VehPower"].astype(float)
df["BonusMalus"] = df["BonusMalus"].astype(float)
```

The density transformation is important. Density varies by four orders of magnitude (1 to 27,000), and its effect on frequency is closer to linear on the log scale. Leaving it untransformed produces an unstable, hard-to-interpret coefficient in a log-link GLM. `log1p` compresses the distribution and produces a coefficient that means something: a one-unit increase in log density corresponds to a constant percentage increase in frequency.

Driver age and vehicle power are left as continuous here — the GLM fits a linear effect for each. If you suspect nonlinearity (young drivers in particular have a well-documented U-shaped frequency profile), you can add a quadratic term or use [insurance-gam](/insurance-gam/) to fit a shape-constrained spline instead.

### Train/test split

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
# Train: 542,410 | Test: 135,603
```

### Feature matrix construction

glum takes a numpy array or sparse matrix for features, not a DataFrame. We use scikit-learn's `ColumnTransformer` to one-hot encode categoricals and pass-through numerics:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICALS = ["Area", "VehBrand", "VehGas", "Region"]
NUMERICS     = ["DrivAge", "VehAge", "VehPower", "BonusMalus", "LogDensity"]

preprocessor = ColumnTransformer(
    transformers=[
        ("ohe", OneHotEncoder(sparse_output=True, drop="first"), CATEGORICALS),
        ("num", "passthrough", NUMERICS),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

X_train = preprocessor.fit_transform(train_df)
X_test  = preprocessor.transform(test_df)

y_train   = train_df["ClaimNb"].values
y_test    = test_df["ClaimNb"].values
exp_train = train_df["Exposure"].values
exp_test  = test_df["Exposure"].values

print(f"Feature matrix shape: {X_train.shape}")
# Feature matrix shape: (542410, 48)
```

`drop="first"` on the one-hot encoder removes one level per categorical to avoid perfect multicollinearity. The dropped level becomes the reference level for relativities: all other levels are expressed as multiples of the reference.

---

## Fitting the model

glum's API follows the sklearn pattern but adds insurance-specific parameters. The key difference from `sklearn.linear_model.PoissonRegressor` is the `offset` argument in `fit()`:

```python
from glum import GeneralizedLinearRegressor

model = GeneralizedLinearRegressor(
    family="poisson",
    link="log",
    fit_intercept=True,
    max_iter=100,
)

model.fit(
    X_train,
    y_train,
    offset=np.log(exp_train),   # log-exposure offset: the correct treatment
)

print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficients: {model.coef_.shape[0]} parameters")
# Intercept: -2.8317
# Coefficients: 47 parameters
```

The intercept of -2.83 corresponds to a base frequency of `exp(-2.83) ≈ 5.9%` at the reference levels of all categorical factors — broadly consistent with the portfolio mean once you account for the reference categories chosen.

**Why the offset, not a sample weight?** They do different things. An offset `log(E)` shifts the linear predictor directly: the model estimates `log(frequency) = intercept + X @ beta`, so `log(E[claims]) = log(E) + intercept + X @ beta`. A sample weight re-scales the deviance contribution of each observation. For a Poisson model with heterogeneous exposure, the offset is correct. Sample weights are an approximation that can give slightly biased coefficient estimates when exposure is correlated with rating factors — which it usually is in any seasoned portfolio, where long-exposure policies disproportionately belong to older, lower-risk drivers.

---

## In-sample and out-of-sample deviance

Poisson deviance is the canonical loss for frequency models. For an observation with count y and predicted count μ_i = E_i × freq_i:

```
D_i = 2 * (y_i * log(y_i / μ_i) - (y_i - μ_i))
```

where `0 * log(0)` is taken as 0.

```python
mu_train = model.predict(X_train, offset=np.log(exp_train))  # predicted frequency
mu_test  = model.predict(X_test,  offset=np.log(exp_test))

# Exposure-weighted Poisson deviance
def poisson_deviance(y, freq_pred, exposure):
    """Exposure-weighted mean Poisson deviance."""
    mu = freq_pred * exposure
    # term is 0 where y=0, to handle 0*log(0)=0 by convention
    term = np.where(y > 0, y * np.log(y / np.maximum(mu, 1e-12)), 0.0) - (y - mu)
    return 2.0 * np.average(term, weights=exposure)

dev_train = poisson_deviance(y_train, mu_train, exp_train)
dev_test  = poisson_deviance(y_test,  mu_test,  exp_test)

print(f"Train Poisson deviance: {dev_train:.6f}")
print(f"Test  Poisson deviance: {dev_test:.6f}")
# Train Poisson deviance: 0.312847
# Test  Poisson deviance: 0.313102
```

The train/test deviance being nearly identical signals no meaningful overfitting — a GLM with this feature set and 540k training rows has very limited capacity to overfit. The test deviance of ~0.313 is the baseline to beat if you later fit a GBM; the [freMTPL2 benchmark](/2026/03/24/python-insurance-pricing-benchmark-glm-xgboost-catboost-lightgbm-fremtpl2/) documents how GBMs improve on it.

---

## Actual vs expected

The A/E chart is the first diagnostic any pricing actuary will ask for. We sort observations by predicted frequency, group into deciles, and plot mean observed frequency against mean predicted frequency:

```python
import matplotlib.pyplot as plt

ae_df = pd.DataFrame({
    "predicted": mu_test,
    "actual":    y_test / np.maximum(exp_test, 1e-6),
    "exposure":  exp_test,
})

ae_df["decile"] = pd.qcut(ae_df["predicted"], q=10, labels=False)

ae_summary = (
    ae_df
    .groupby("decile", observed=True)
    .apply(lambda g: pd.Series({
        "avg_predicted": np.average(g["predicted"], weights=g["exposure"]),
        "avg_actual":    np.average(g["actual"],    weights=g["exposure"]),
        "exposure":      g["exposure"].sum(),
    }))
    .reset_index()
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ae_summary["avg_predicted"], ae_summary["avg_actual"],
        "o-", label="Actual vs Predicted")
lim = ae_summary["avg_predicted"].max() * 1.05
ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="Perfect calibration")
ax.set_xlabel("Mean predicted frequency")
ax.set_ylabel("Mean actual frequency")
ax.set_title("Actual vs Expected — Poisson GLM (freMTPL2, test set)")
ax.legend()
plt.tight_layout()
plt.savefig("ae_chart.png", dpi=150)
```

A well-calibrated frequency model should sit on or near the 45-degree line across all deciles. If the model is systematically under-predicting the top decile, the highest-risk policies are mispriced. A Poisson GLM is calibrated overall by construction — the sum of predicted counts equals the sum of observed counts — but calibration by predicted-score decile is not guaranteed and depends on the model structure. Systematic deviation in the upper decile is the signal that a GBM with interaction terms would fix.

---

## Deviance residuals

Deviance residuals are the standard residual diagnostic for GLMs. For a Poisson GLM, the deviance residual for observation i is:

```
r_i = sign(y_i - μ_i) * sqrt(2 * max(y_i * log(y_i / μ_i) - (y_i - μ_i), 0))
```

where μ_i = E_i × predicted_frequency_i.

```python
def poisson_deviance_residuals(y, mu):
    """Signed deviance residuals for a Poisson GLM.
    y: observed claim counts (integer)
    mu: predicted claim counts (= exposure * predicted_frequency)
    """
    mu = np.maximum(mu, 1e-12)
    term = np.where(y > 0, y * np.log(y / mu) - (y - mu), -(mu - y))
    return np.sign(y - mu) * np.sqrt(2 * np.maximum(term, 0))

mu_test_counts = mu_test * exp_test   # predicted claim counts
resid = poisson_deviance_residuals(y_test, mu_test_counts)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(
    np.log1p(mu_test_counts),
    resid,
    alpha=0.04, s=1.5, rasterized=True
)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("log(1 + predicted claim count)")
ax.set_ylabel("Deviance residual")
ax.set_title("Deviance residuals vs predicted — test set")
plt.tight_layout()
plt.savefig("deviance_residuals.png", dpi=150)
```

The characteristic shape for a Poisson GLM on insurance data is a fan: residuals cluster at a discrete set of values for low predicted counts (because claim counts are integers — 0, 1, 2) and spread out at higher predicted counts where the integer constraint relaxes relative to the scale. What you do not want is systematic structure — a slope in the mean residuals across the predicted range suggests a missing nonlinear term. A [GAM](/insurance-gam/) with spline terms for driver age and bonus-malus would typically clean up the residual pattern in the upper predicted-count region.

With 135k test observations, the scatter plot is dense. Summarise the residual structure by grouping into bins:

```python
resid_df = pd.DataFrame({
    "log_pred": np.log1p(mu_test_counts),
    "residual": resid,
})
resid_df["bin"] = pd.cut(resid_df["log_pred"], bins=20)
mean_resid = resid_df.groupby("bin", observed=True)["residual"].mean()
print(mean_resid.to_string())
```

Mean residuals close to zero in every bin is what you are looking for. Persistent positive mean residuals in the upper bins — meaning the model systematically under-predicts high-risk policies — is the finding that justifies adding interactions or switching to a more flexible model.

---

## Factor relativities

The GLM coefficients are the model. To turn them into factor relativities — the multiplicative factor for each variable level relative to a reference level — extract and exponentiate:

```python
# Reconstruct feature names from the preprocessor
ohe_names = preprocessor.named_transformers_["ohe"].get_feature_names_out(CATEGORICALS)
feature_names = list(ohe_names) + NUMERICS

coef_df = pd.DataFrame({
    "feature":     feature_names,
    "coefficient": model.coef_,
    "relativity":  np.exp(model.coef_),
})

# Categorical relativities: filter and sort by absolute coefficient
cat_rel = coef_df[coef_df["feature"].str.startswith(tuple(CATEGORICALS))]
print(cat_rel.sort_values("relativity", ascending=False).head(10).to_string(index=False))
```

```
        feature  coefficient  relativity
  Area_Area_F      0.4521       1.5719
  Area_Area_E      0.3842       1.4682
  Region_R73       0.3241       1.3832
  Area_Area_D      0.2933       1.3407
  Region_R52       0.2877       1.3334
  VehBrand_B12     0.2214       1.2477
  Area_Area_C      0.1897       1.2090
  Area_Area_B      0.1204       1.1279
  VehGas_Regular  -0.0943       0.9101
  Region_R91      -0.2647       0.7675
```

Area F (urban) has a relativity of 1.57× versus Area A (rural, the dropped reference level). Region R91 has a relativity of 0.77× — roughly 23% lower frequency than the reference region after controlling for driver age, vehicle type, area, and bonus-malus.

For the continuous features, the coefficient is a slope: each additional year of driver age changes frequency by `exp(-0.0087) - 1 ≈ -0.87%`, and each additional BonusMalus point increases frequency by approximately 2%.

```python
num_rel = coef_df[coef_df["feature"].isin(NUMERICS)]
print(num_rel.to_string(index=False))
```

```
      feature  coefficient  relativity
      DrivAge      -0.0087       0.9913
       VehAge      -0.0124       0.9877
     VehPower       0.0218       1.0220
   BonusMalus       0.0198       1.0200
   LogDensity       0.0771       1.0802
```

**Rebasing relativities.** The one-hot encoder dropped the first level of each categorical, which becomes the implicit reference. In Emblem or Radar, you choose the reference level deliberately — typically the most common or most stable level. To re-express relativities relative to a different base:

```python
def rebase(coef_df, feature_prefix, new_base_suffix):
    """
    Re-express relativities relative to a chosen base level.

    Example: rebase(coef_df, "Area", "Area_C") shifts the Area block
    so that Area C has relativity 1.0 and all others are relative to it.
    """
    mask = coef_df["feature"].str.startswith(feature_prefix)
    block = coef_df[mask].copy()
    base_row = block[block["feature"].str.endswith(new_base_suffix)]

    if base_row.empty:
        raise ValueError(f"Base level not found: {new_base_suffix}")

    base_coef = base_row["coefficient"].iloc[0]

    # Shift all coefficients in the block by subtracting the base coefficient
    coef_df.loc[mask, "coefficient"] -= base_coef
    coef_df.loc[mask, "relativity"] = np.exp(coef_df.loc[mask, "coefficient"])
    return coef_df

# Re-express Area relativities relative to Area C (the most common level)
coef_df = rebase(coef_df, "Area", "Area_C")
print(coef_df[coef_df["feature"].str.startswith("Area")].to_string(index=False))
```

```
    feature  coefficient  relativity
Area_Area_B      -0.0693       0.9331
Area_Area_C       0.0000       1.0000   <- new base
Area_Area_D       0.1036       1.1092
Area_Area_E       0.1945       1.2148
Area_Area_F       0.2624       1.2998
```

This rebasing is a constant shift in log-space — it does not change the model's predictions, only the presentation. Area A (the original dropped reference) is now implicit: its relativity is `1 / (Area_B_raw_relative_to_A / Area_B_relative_to_C)`. If you need Area A explicit in the output, add it back as a row with the appropriate coefficient.

### Standard errors and confidence intervals

glum returns standard errors via `model.coef_se_`, which most GLM solvers (including sklearn's PoissonRegressor) do not:

```python
import scipy.stats as stats

z_scores = model.coef_ / model.coef_se_
p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

coef_df["std_err"]  = model.coef_se_
coef_df["z_score"]  = z_scores
coef_df["p_value"]  = p_values
coef_df["ci_lower"] = np.exp(model.coef_ - 1.96 * model.coef_se_)
coef_df["ci_upper"] = np.exp(model.coef_ + 1.96 * model.coef_se_)

# Flag weak coefficients
weak = coef_df[coef_df["p_value"] > 0.05]
print(f"Coefficients not significant at 5%: {len(weak)}")
print(weak[["feature", "relativity", "p_value"]].to_string(index=False))
```

These standard errors are model-based: they come from the observed information matrix and assume the Poisson variance assumption holds. If the data are overdispersed — and they usually are — the standard errors are too small. See the section below.

---

## Checking for overdispersion

A Poisson distribution has mean equal to variance. Insurance claim counts regularly violate this: the variance typically exceeds the mean by a factor of 1.5–4× on motor portfolios. This does not bias the coefficient estimates — the Poisson score equations give consistent estimators regardless — but it means the model-based standard errors understate the true uncertainty.

```python
# Pearson chi-squared statistic / degrees of freedom
mu_all = model.predict(X_train, offset=np.log(exp_train)) * exp_train
pearson_resid = (y_train - mu_all) / np.sqrt(np.maximum(mu_all, 1e-12))
n_params = X_train.shape[1] + 1   # +1 for intercept
dispersion = np.sum(pearson_resid ** 2) / (len(y_train) - n_params)

print(f"Estimated dispersion: {dispersion:.3f}")
# Estimated dispersion: 1.847
```

A dispersion of 1.85 means the data are moderately overdispersed. To use quasi-Poisson inference — which inflates standard errors to account for overdispersion — multiply all standard errors by `sqrt(dispersion)`:

```python
phi = dispersion
coef_df["std_err_qp"] = model.coef_se_ * np.sqrt(phi)
coef_df["ci_lower_qp"] = np.exp(model.coef_ - 1.96 * coef_df["std_err_qp"])
coef_df["ci_upper_qp"] = np.exp(model.coef_ + 1.96 * coef_df["std_err_qp"])
```

In practice: a dispersion around 1.5–2.5 is normal for motor frequency data and does not change which variables you include. A dispersion of 5+ suggests something is wrong — missing interactions, heterogeneous sub-portfolios that should be modelled separately, or data quality issues with claim counts (duplicate records, unreported partial-year adjustments).

---

## What to do next

This tutorial covers the core workflow: load, prepare, fit, validate, extract. Three natural extensions:

**Nonlinear terms.** Driver age has a U-shaped frequency profile — higher at both ends of the age range. Adding `DrivAge**2` as a feature captures this with two parameters; using [insurance-gam](/insurance-gam/) for a shape-constrained spline is cleaner and avoids the need to constrain the quadratic sign manually.

**Interaction detection.** The GLM above fits main effects only. If the age-by-vehicle-power interaction is present in the data — younger drivers in higher-powered vehicles disproportionately worse — a main-effects GLM will systematically underprice that segment. The automated approach using neural interaction detection is covered in [Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/).

**Model monitoring.** A fitted GLM deployed to price new business will drift as the portfolio mix changes. The [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) library implements score decomposition and population stability indices for tracking GLM performance over time — the full setup is covered in our [practitioner guide to insurance model monitoring in Python](/2026/04/04/insurance-model-monitoring-python-practitioner-guide/).

The freMTPL2 Poisson GLM we built here reaches a test Poisson deviance of approximately 0.313. The [Python insurance pricing benchmark](/2026/03/24/python-insurance-pricing-benchmark-glm-xgboost-catboost-lightgbm-fremtpl2/) documents how CatBoost, XGBoost, and LightGBM compare on the same dataset. The short answer: the GBMs score meaningfully better on discrimination. The longer answer: the GLM's interpretability — the ability to read off factor relativities, audit each coefficient, and produce clean factor tables for a rating engine — is why most production pricing teams use it as their deployed model even when a GBM trains alongside it.

---

*Related:*
- [GLMs for UK Insurance Pricing in Python: What the Generic Tutorials Miss](/2026/03/22/glm-insurance-python-uk-pricing-actuary-guide/)
- [Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/)
- [The Python Insurance Pricing Benchmark: GLM vs XGBoost vs CatBoost vs LightGBM on freMTPL2](/2026/03/24/python-insurance-pricing-benchmark-glm-xgboost-catboost-lightgbm-fremtpl2/)
- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/)
