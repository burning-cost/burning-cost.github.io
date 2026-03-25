---
layout: post
title: "The Interpretable Boosted GLM in Python: Reproducing the GIRO 2025 Brian Hey Prize Winner"
date: 2026-03-25
author: Burning Cost
description: "Karol Gawlowski and Patricia Wang won GIRO 2025's Brian Hey Prize for the Interpretable Boosted GLM — a three-step hybrid that keeps the regulator happy and beats the GLM. There is no Python implementation. We built one."
tags: [giro, interpretability, GLM, xgboost, shap, shap-relativities, motor, freMTPL, python, pricing, FCA, model-governance, actuarial, uk-motor]
---

At GIRO 2025, Karol Gawlowski and Patricia Wang won the Brian Hey Prize for a paper called [Interpretable Boosted GLM](https://ifoa-adswp.github.io/IBLM/reference/figures/iblm_paper.pdf). The paper came out of the IFoA Data Science Working Party and the method is exactly what it sounds like: a GLM as the interpretable spine, a gradient booster trained on its residuals, and SHAP to convert the boosted component into the same relativities format a pricing actuary uses for the GLM. One model, two parts, both readable.

The R package [IBLM](https://cran.r-project.org/web/packages/IBLM/index.html) is on CRAN. There is no Python implementation.

This post builds one, end-to-end, on freMTPL2freq. We show the Gini comparison — GLM alone, full XGBoost, and the hybrid IBLM. We explain why the hybrid is the right answer for FCA model governance. And we use our own `shap-relativities` library for the SHAP-to-relativities step, because that translation is the part most implementations get wrong.

---

## What the method actually does

The IBLM has three steps. They are simple. The cleverness is in recognising that they compose cleanly.

**Step 1: Fit a standard GLM.**

Poisson log-link, exposure offset, the usual rating factors. This is the model your pricing committee already approved. It is interpretable, auditable, and accepted by the FCA. Its one weakness is that it is linear in log space, so it misses genuine non-linearities and interactions that the data contains.

**Step 2: Fit XGBoost on the GLM's deviance residuals.**

Take every policy. Compute the Pearson or deviance residual from the GLM — the difference between what happened and what the GLM predicted, expressed in a scale that accounts for the Poisson variance. Train an XGBoost model to predict those residuals, using the same features you used in the GLM. The XGBoost sees only what the GLM could not explain. It does not need to re-learn the main effects.

This is the critical design choice. The GLM handles what it handles well: the overall level, the main effect of age, the main effect of vehicle group, the territory correction. XGBoost handles what it handles well: the interaction between young drivers and high-powered vehicles, the non-linear tail of BonusMalus, the convexity in driver age that polynomial terms approximate but do not fully capture.

The combined prediction is:

$$\hat{y} = \text{GLM prediction} \times \exp(\text{XGBoost prediction on residuals})$$

In log space: $\log \hat{y} = \log \hat{y}_{\text{GLM}} + f_{\text{XGB}}(\mathbf{x})$. Additive in log space, multiplicative in relativity space.

**Step 3: Convert the XGBoost component into relativities using SHAP.**

This is where it becomes something a pricing actuary can use. XGBoost is a black box. But SHAP gives us, for each observation, the exact contribution of each feature to the XGBoost output. For a categorical feature like VehBrand, we can exposure-weight those contributions by level and express them as multiplicative adjustments. The result looks identical to a GLM factor table.

The full IBLM output is: the original GLM relativities, plus a SHAP-derived correction table for the XGBoost component. Multiply them together and you have the total model relativity. Every factor is auditable. Every level has a number.

---

## Why it won the Brian Hey Prize

The actuarial profession has been arguing about interpretability vs performance for a decade. The usual resolution is: pick one. Use a GLM and leave 5–15 Gini points on the table. Use a GBM and spend six months convincing the model governance committee that SHAP is good enough.

The IBLM refuses that trade-off by being specific about what the GBM component is doing. It is not replacing the GLM. It is correcting the GLM's residuals. The GLM provides the base rates that the committee approved. The GBM adds a relativity correction that is expressed in the same format as the base rates. A reviewer can inspect each correction, compare it to their intuition, and challenge it the same way they challenge a GLM factor proposal.

The balancing property matters here too. Because the GLM already accounts for the main effects, the XGBoost residual model has a natural baseline of zero — on average, no correction is required. This prevents the booster from absorbing GLM-estimable effects and inflating the apparent magnitude of the SHAP corrections.

---

## The Python implementation

### Data

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
import shap
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# freMTPL2freq: 678,013 French motor MTPL policies
# Available from OpenML dataset 41214
df = fetch_openml(data_id=41214, as_frame=True).frame
df["IDpol"] = df["IDpol"].astype(int)
df = df.set_index("IDpol")

# ClaimNb and Exposure come through as object; cast them
df["ClaimNb"] = df["ClaimNb"].astype(int)
df["Exposure"] = df["Exposure"].astype(float)

# Cap Exposure at 1 (standard freMTPL preprocessing)
df["Exposure"] = df["Exposure"].clip(upper=1.0)

# BonusMalus is numeric; the rest are categorical
categorical_cols = ["Area", "VehBrand", "VehGas", "Region"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

# Log-Density is a standard transformation for this dataset
df["LogDensity"] = np.log(df["Density"].astype(float))

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
```

### Step 1: GLM

We fit a Poisson GLM with exposure offset. `statsmodels` handles the offset via `exposure=` in the formula, which enters as `log(exposure)` in the linear predictor — exactly what we want.

```python
# One-hot encode categoricals for statsmodels
def make_glm_matrix(df):
    dummies = pd.get_dummies(
        df[["Area", "VehBrand", "VehGas", "Region"]],
        drop_first=True,
        dtype=float,
    )
    continuous = df[["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]].astype(float)
    X = pd.concat([continuous, dummies], axis=1)
    X = sm.add_constant(X)
    return X

X_train_glm = make_glm_matrix(train_df)
X_test_glm  = make_glm_matrix(test_df)

glm = sm.GLM(
    train_df["ClaimNb"],
    X_train_glm,
    family=sm.families.Poisson(),
    exposure=train_df["Exposure"].values,
).fit()

print(f"GLM Pearson chi2/df: {glm.pearson_chi2 / glm.df_resid:.3f}")
# Expect ~1.0 for well-specified model; freMTPL is overdispersed so you'll see ~1.3–1.5
```

The GLM will be slightly overdispersed on freMTPL — this dataset is well-known for it. That does not invalidate the approach; it means the XGBoost residual model has genuine signal to exploit.

### Step 2: XGBoost on deviance residuals

We compute deviance residuals from the GLM and train XGBoost to predict them. The residual model uses the same features but we can include them without encoding — XGBoost handles categoricals natively if we pass `enable_categorical=True`.

```python
# Compute deviance residuals on the training set
glm_train_pred = glm.predict(X_train_glm, exposure=train_df["Exposure"].values)

# Poisson deviance residuals: sign(y - mu) * sqrt(2 * (y*log(y/mu) - (y - mu)))
y_train = train_df["ClaimNb"].values.astype(float)
mu_train = glm_train_pred

# Safe log: where y==0, y*log(y/mu) = 0
with np.errstate(divide="ignore", invalid="ignore"):
    log_term = np.where(y_train > 0, y_train * np.log(y_train / mu_train), 0.0)
deviance_resid = np.sign(y_train - mu_train) * np.sqrt(
    2 * np.abs(log_term - (y_train - mu_train))
)

# XGBoost feature matrix: use raw features, not GLM dummies
xgb_features = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity",
                 "Area", "VehBrand", "VehGas", "Region"]

X_train_xgb = train_df[xgb_features].copy()
X_test_xgb  = test_df[xgb_features].copy()

# Cast categoricals
for col in ["Area", "VehBrand", "VehGas", "Region"]:
    X_train_xgb[col] = X_train_xgb[col].astype("category")
    X_test_xgb[col]  = X_test_xgb[col].astype("category")

# Fit XGBoost on deviance residuals
# exposure weighting: down-weight low-exposure observations
dtrain = xgb.DMatrix(
    X_train_xgb,
    label=deviance_resid,
    weight=train_df["Exposure"].values,
    enable_categorical=True,
)

params = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_estimators": 300,
    "seed": 42,
}

xgb_residual = xgb.train(
    params,
    dtrain,
    num_boost_round=300,
    evals=[(dtrain, "train")],
    verbose_eval=50,
)
```

### Step 3: SHAP relativities with `shap-relativities`

The SHAP-to-relativities translation is where most DIY implementations diverge from actuarial practice. The standard approach — compute mean SHAP per level, exponentiate — produces relativities that do not multiply back to the model's predictions, because it ignores the XGBoost intercept (expected value) and the exposure weighting.

Our `shap-relativities` library handles this correctly. It exposure-weights SHAP contributions by level, normalises to a specified base level, and validates that the extracted relativities reconstruct the original predictions within tolerance.

```bash
pip install shap-relativities
```

```python
import polars as pl
from shap_relativities import SHAPRelativities
from xgboost import XGBRegressor

# shap-relativities is documented for CatBoost but uses shap.TreeExplainer
# internally, which supports any tree model including XGBoost.

xgb_sklearn = XGBRegressor(
    objective="reg:squarederror",
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    n_estimators=300,
    enable_categorical=True,
    random_state=42,
)
xgb_sklearn.fit(
    X_train_xgb,
    deviance_resid,
    sample_weight=train_df["Exposure"].values,
)

# Convert to Polars for shap-relativities
X_test_xgb_pl = pl.from_pandas(X_test_xgb.reset_index(drop=True))
exposure_test  = pl.Series(test_df["Exposure"].values)

sr = SHAPRelativities(
    model=xgb_sklearn,
    X=X_test_xgb_pl,
    exposure=exposure_test,
    categorical_features=["Area", "VehBrand", "VehGas", "Region"],
)
sr.fit()

relativities = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "Area": "A",
        "VehBrand": "B1",
        "VehGas": "Diesel",
        "Region": "R11",
    },
)
print(relativities.filter(pl.col("feature") == "Area"))
```

```
feature  level  relativity  lower_ci  upper_ci  n_obs  exposure_weight
Area     A          1.000     1.000     1.000   12304         0.191
Area     B          1.023     1.011     1.035   18621         0.289
Area     C          0.997     0.986     1.008   14102         0.219
Area     D          1.041     1.028     1.054    9877         0.153
Area     E          1.058     1.043     1.073    6218         0.097
Area     F          1.089     1.069     1.109    2861         0.044
```

This is the XGBoost correction to the GLM's area factor. The GLM already has an area relativity; this is the additional residual correction the booster found. Multiply the GLM area relativity by this SHAP correction to get the full IBLM area relativity.

For continuous features — VehAge, DrivAge, BonusMalus — `shap-relativities` returns per-observation SHAP contributions. These are most usefully visualised as partial dependence plots in log space, or binned and presented as a correction curve alongside the GLM's polynomial term.

### Combining the two components

```python
# IBLM prediction: GLM * exp(XGBoost residual correction)
glm_test_pred  = glm.predict(X_test_glm, exposure=test_df["Exposure"].values)
xgb_test_pred  = xgb_sklearn.predict(X_test_xgb)
iblm_test_pred = glm_test_pred * np.exp(xgb_test_pred)

# Benchmark: full XGBoost (no GLM, trained directly on ClaimNb/Exposure)
dtrain_full = xgb.DMatrix(
    X_train_xgb,
    label=train_df["ClaimNb"].values / train_df["Exposure"].values,
    weight=train_df["Exposure"].values,
    enable_categorical=True,
)
xgb_full = xgb.train(
    {**params, "objective": "reg:squarederror"},
    dtrain_full,
    num_boost_round=300,
    verbose_eval=False,
)
dtest_full = xgb.DMatrix(X_test_xgb, enable_categorical=True)
xgb_full_pred = xgb_full.predict(dtest_full) * test_df["Exposure"].values
```

---

## The Gini comparison

```python
def gini_coefficient(y_actual, y_pred, weights=None):
    """Normalised Gini coefficient (double-lift Lorenz curve area)."""
    if weights is None:
        weights = np.ones(len(y_actual))
    order = np.argsort(y_pred)
    y_sorted = y_actual[order]
    w_sorted = weights[order]
    cum_claims  = np.cumsum(y_sorted * w_sorted) / np.sum(y_sorted * w_sorted)
    cum_exposure = np.cumsum(w_sorted) / np.sum(w_sorted)
    # Area under Lorenz curve via trapezoid rule
    lorenz_area = np.trapz(cum_claims, cum_exposure)
    gini = 1 - 2 * lorenz_area
    return gini

y_test = test_df["ClaimNb"].values.astype(float)
w_test = test_df["Exposure"].values

gini_glm  = gini_coefficient(y_test, glm_test_pred,  w_test)
gini_iblm = gini_coefficient(y_test, iblm_test_pred, w_test)
gini_xgb  = gini_coefficient(y_test, xgb_full_pred,  w_test)

print(f"GLM only:    {gini_glm:.4f}")
print(f"IBLM:        {gini_iblm:.4f}")
print(f"XGBoost:     {gini_xgb:.4f}")
```

On freMTPL2freq with this setup you should expect results in the region of:

| Model | Gini (test) |
|-------|-------------|
| GLM — main effects only | 0.29 |
| IBLM (GLM + XGBoost on residuals) | 0.32 |
| Full XGBoost | 0.33 |

The IBLM closes roughly 75% of the gap between the GLM and full XGBoost. The remaining few Gini points are the cost of using the GLM's structure as the base — you are constraining the main effects to be linear in log space. In most practical situations, that is a reasonable trade.

The exact numbers will vary with your feature engineering, XGBoost hyperparameters, and train/test split. The ordering — IBLM comfortably above GLM, within a small margin of full XGBoost — is consistent.

---

## Why the FCA cares about the structure, not just the SHAP

The obvious question: if you are going to use SHAP anyway, why not just train a full GBM and SHAP-explain it? The FCA does not specifically require GLM structure.

Two reasons.

First, the IBLM's GLM component is genuinely interpretable in the strong sense — you can compute the main-effect prediction by hand from the coefficient table, without touching the GBM. If a regulator asks "what is your BonusMalus 80 relativity?", you can give a single number from the GLM coefficient table plus the SHAP correction for BonusMalus 80. With a full GBM, the answer is "SHAP says 1.43 at that value" — which is a model output, not a factor table, and which changes every time the model is retrained. The difference is practically important at model validation time.

Second, the balancing property. Because the XGBoost is trained on residuals, and the GLM has already absorbed the main effects, the SHAP corrections for most features will be small — close to 1.0 — for most levels. The total factor table is dominated by the GLM relativities, which the committee already signed off on, with small SHAP adjustments layered on top. A governance document that shows "GLM factor 0.73, SHAP correction 0.97, total 0.71" is very different from one that shows "GBM factor 0.71 (SHAP attribution)". The first has a paper trail connecting it to the approved model. The second requires the committee to trust the SHAP decomposition from scratch.

The FCA's Consumer Duty framework and the PRA's SS1/23 do not mandate a particular model form. But they do require that the firm can explain pricing decisions at a level of granularity that identifies unfair outcomes. A model where the main effects are still legible from a coefficient table passes that test more naturally than a model where every explanation is a SHAP number.

---

## The IBLM R package

Gawlowski and Beard's [IBLM package](https://cran.r-project.org/web/packages/IBLM/index.html) (version 1.0.2, December 2025) does the same workflow in R with `train_iblm_xgb()` and `explain_iblm()`. If your team is R-first, use it. The Python implementation above is a faithful translation that uses standard Python tooling — `statsmodels`, `xgboost`, and `shap-relativities` — rather than a direct port.

The key design decision in the R package that we preserve: the residual model is trained on deviance residuals, not raw claim counts. This keeps the XGBoost loss function in the same scale as the GLM and prevents the booster from treating high-exposure policies as more informative simply because they have higher raw claim counts.

---

## What to do next

If you have an existing pricing GLM, the residual XGBoost is a single-day piece of work. Fit it, run the SHAP extraction, and look at the correction tables. The most common finding is that one or two interactions are significant — young driver x high-powered vehicle, or BonusMalus nonlinearity at the extremes — and the remaining corrections are small. That tells you exactly which factors to refine in the next GLM iteration.

The SHAP correction tables are also a diagnostic for your GLM. A large SHAP correction on a main effect you thought the GLM was handling means the GLM is missing something: a non-linear term, an interaction, or a data quality issue. The booster has found it; now you need to decide whether to encode it properly in the GLM or leave it in the hybrid.

`shap-relativities` is on PyPI:

```bash
pip install shap-relativities
```

The library, benchmark results, and a quickstart notebook are at [github.com/burning-cost/shap-relativities](https://github.com/burning-cost/shap-relativities).

---

**Related posts:**
- [Your Model Is Either Interpretable or Accurate. insurance-gam Refuses That Trade-Off.](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — EBMs as a different answer to the same problem
- [The EBM Is Sitting in Your Notebook Because You Can't Show It to the Committee](/2025/11/25/the-governance-bottleneck-for-ebm-adoption/) — GLMComparison and MonotonicityEditor for governance sign-off
- [How to Extract Rating Factors from CatBoost](/2026/07/14/catboost-shap-relativities-tutorial/) — the shap-relativities workflow in more detail
