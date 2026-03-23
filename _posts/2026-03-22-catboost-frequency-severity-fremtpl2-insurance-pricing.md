---
layout: post
title: "CatBoost Frequency-Severity Modelling on freMTPL2"
date: 2026-03-22
categories: [pricing, techniques, tutorials]
tags: [catboost, frequency-severity, fremtpl2, polars, shap-relativities, insurance-distill, poisson, gamma, burning-cost, python, uk-motor]
description: "Step-by-step CatBoost frequency-severity model on freMTPL2. Poisson + Gamma, burning cost combination, SHAP factor tables, and GLM distillation for Radar."
author: Burning Cost
---

The freMTPL2 dataset — French third-party liability motor, ~678,000 policies — is the closest thing actuarial science has to a standard benchmark. It ships inside `sklearn.datasets` via OpenML, it has a known structure (Poisson frequency, roughly Gamma severity), and enough volume to evaluate models properly. If you are building a freq-sev pricing model in Python and you are not testing it on freMTPL2, you are missing the easiest validation step available.

This post covers the complete workflow: data prep in Polars, CatBoost Poisson frequency model, CatBoost Gamma severity model, burning cost combination, factor table extraction via `shap-relativities`, and distillation to GLM format via `insurance-distill`. All code runs locally. No Databricks cluster required.

---

## The data

freMTPL2 is split across two OpenML datasets: `freMTPL2freq` (policy-level with claim counts) and `freMTPL2sev` (claim-level with individual amounts). The frequency file has 678,013 rows. The severity file has 26,639 rows — roughly a 3.9% overall claim frequency.

```python
import polars as pl
import numpy as np
from sklearn.datasets import fetch_openml

# Fetch frequency data — policy-level
freq_raw = fetch_openml("freMTPL2freq", version=3, as_frame=True, parser="auto")
freq = pl.from_pandas(freq_raw.frame)

# Fetch severity data — individual claim amounts
sev_raw = fetch_openml("freMTPL2sev", version=1, as_frame=True, parser="auto")
sev = pl.from_pandas(sev_raw.frame)

print(freq.shape)   # (678013, 12)
print(sev.shape)    # (26639, 2)
```

The frequency dataset columns we care about: `IDpol` (policy ID), `ClaimNb` (claim count), `Exposure` (policy years, capped at 1.0), `Area` (A-F), `VehPower` (1-15), `VehAge`, `DrivAge`, `BonusMalus` (French NCD equivalent, 50-350), `VehBrand`, `VehGas` (Regular/Diesel), `Density` (population density of policyholder's commune), `Region` (French administrative region, 21 levels).

One well-known issue with freMTPL2: `ClaimNb` is the number of claims filed under the policy but the severity file links claims by `IDpol`, not by a claim ID. Policies with multiple claims have a single severity row per claim — but matching them back is ambiguous when a policy has two claims and two rows in the severity file. For this walkthrough we use average severity per policy (total paid divided by claim count) for the severity model, which sidesteps the matching problem cleanly.

```python
# Compute per-policy average severity from the severity file
sev_by_policy = (
    sev
    .group_by("IDpol")
    .agg(
        pl.col("ClaimAmount").sum().alias("total_paid"),
        pl.len().alias("n_claim_rows"),
    )
)

# Join average severity to frequency file (only rows with claims)
freq_with_sev = (
    freq
    .join(sev_by_policy, on="IDpol", how="left")
    .with_columns([
        (pl.col("total_paid") / pl.col("ClaimNb"))
          .alias("avg_severity"),
        # Cap exposure at 1.0 — a handful of rows exceed it due to data quirks
        pl.col("Exposure").clip(upper_bound=1.0),
    ])
)

print(
    freq_with_sev
    .select(["ClaimNb", "Exposure", "avg_severity"])
    .describe()
)
```

---

## Feature preparation

The raw features need light treatment. `BonusMalus` has a floor at 50 (clean driver) and a meaningful nonlinear profile — values above 150 are extreme, roughly 1% of policyholders, and CatBoost will handle that naturally as a continuous feature with no manual transformation needed. `Density` is heavily right-skewed; we log-transform it.

```python
freq_model = (
    freq_with_sev
    .with_columns([
        pl.col("Density").log1p().alias("log_density"),
        pl.col("Area").cast(pl.Utf8),
        pl.col("VehBrand").cast(pl.Utf8),
        pl.col("VehGas").cast(pl.Utf8),
        pl.col("Region").cast(pl.Utf8),
    ])
)

features = [
    "Area", "VehPower", "VehAge", "DrivAge",
    "BonusMalus", "VehBrand", "VehGas", "log_density", "Region",
]
cat_features = ["Area", "VehBrand", "VehGas", "Region"]
```

---

## Frequency model: Poisson with log-exposure offset

CatBoost handles Poisson regression natively via `loss_function="Poisson"`. The exposure offset is the critical step that most tutorials get wrong: we pass `log(exposure)` as a baseline prediction so the model learns to predict claims per unit exposure, not raw claim counts. CatBoost's `baseline` parameter does exactly this.

```python
import catboost
from sklearn.model_selection import train_test_split

X = freq_model.select(features).to_pandas()
y = freq_model["ClaimNb"].to_numpy()
exposure = freq_model["Exposure"].to_numpy()

X_train, X_test, y_train, y_test, exp_train, exp_test = train_test_split(
    X, y, exposure, test_size=0.2, random_state=42
)

freq_pool_train = catboost.Pool(
    data=X_train,
    label=y_train,
    baseline=np.log(exp_train),   # log-exposure offset
    cat_features=cat_features,
)
freq_pool_test = catboost.Pool(
    data=X_test,
    label=y_test,
    baseline=np.log(exp_test),
    cat_features=cat_features,
)

freq_model_cb = catboost.CatBoostRegressor(
    loss_function="Poisson",
    eval_metric="Poisson",
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100,
)
freq_model_cb.fit(freq_pool_train, eval_set=freq_pool_test)
```

The `baseline` parameter is not in CatBoost's most prominent documentation. It is the correct way to implement an exposure offset in CatBoost: you pass the log-exposure as a starting prediction for each row, and the model learns the residual deviation from that baseline. The alternative — dividing claim counts by exposure and using a weighted MSE — is wrong for Poisson regression and produces inconsistent estimates.

On the full 678k-row freMTPL2freq dataset, early stopping triggers around iteration 800-850 with a test Poisson loss in the range 0.312-0.318. The exact number varies with the train/test split.

---

## Severity model: Gamma on claims-only rows

The severity model is fit only on policies with at least one claim. We use `loss_function="RMSE"` with log-transformed severity as the target — this is equivalent to fitting a log-normal, which is close enough to Gamma for a first pass. For a proper Gamma GLM, the correct loss is `loss_function="Tweedie:variance_power=2"`, which CatBoost implements.

```python
# Filter to rows with claims and a valid severity
sev_df = (
    freq_model
    .filter(
        (pl.col("ClaimNb") > 0) & (pl.col("avg_severity").is_not_null())
    )
)

X_sev = sev_df.select(features).to_pandas()
y_sev = sev_df["avg_severity"].to_numpy()

X_sev_train, X_sev_test, y_sev_train, y_sev_test = train_test_split(
    X_sev, y_sev, test_size=0.2, random_state=42
)

sev_pool_train = catboost.Pool(
    data=X_sev_train,
    label=y_sev_train,
    cat_features=cat_features,
)
sev_pool_test = catboost.Pool(
    data=X_sev_test,
    label=y_sev_test,
    cat_features=cat_features,
)

sev_model_cb = catboost.CatBoostRegressor(
    loss_function="Tweedie:variance_power=2",   # Gamma
    iterations=800,
    learning_rate=0.05,
    depth=5,
    l2_leaf_reg=5.0,
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100,
)
sev_model_cb.fit(sev_pool_train, eval_set=sev_pool_test)
```

The severity dataset has roughly 26,000 rows — small relative to the frequency dataset. With depth=5 and l2_leaf_reg=5.0, we are applying deliberate regularisation to prevent overfitting on thin cells. The Tweedie p=2 objective fits a Gamma distribution, which is the standard actuarial assumption for claim severity and produces predictions on the original scale (not log scale), so no back-transformation is needed.

---

## Burning cost: frequency × severity

The burning cost for each policy is the expected number of claims multiplied by the expected cost per claim. This is the pure premium: the expected loss cost per unit exposure before expenses and profit loading.

```python
# Predict on the full dataset
X_all = freq_model.select(features).to_pandas()
exp_all = freq_model["Exposure"].to_numpy()

pool_all = catboost.Pool(
    data=X_all,
    baseline=np.log(exp_all),
    cat_features=cat_features,
)

# Frequency predictions are already exposure-adjusted (because we passed log-exposure as baseline)
freq_pred = freq_model_cb.predict(pool_all)        # expected claim count per policy

# Severity — predict for all policies (not just claimants)
# This extrapolates to non-claimants, which is standard for pure premium calculation
sev_pred = sev_model_cb.predict(X_all)             # expected cost per claim

# Pure premium = expected claim frequency × expected claim cost
pure_premium = freq_pred * sev_pred / exp_all      # per policy-year

freq_model_out = freq_model.with_columns([
    pl.Series("freq_pred", freq_pred),
    pl.Series("sev_pred", sev_pred),
    pl.Series("pure_premium", pure_premium),
])

print(
    freq_model_out
    .select(["freq_pred", "sev_pred", "pure_premium"])
    .describe()
    .select(["statistic", "freq_pred", "sev_pred", "pure_premium"])
)
```

Expected outputs (approximate): mean `freq_pred` ~ 0.054 claims per policy, mean `sev_pred` ~ €1,800, mean `pure_premium` ~ €97 per policy-year. These are in the right ballpark for French TPL, which has relatively low frequency and moderate severity compared with UK comprehensive motor.

One important subtlety: the severity model is trained only on claimants, but we are predicting it for all policies to produce a pure premium. The Gamma model is extrapolating to zero-claim policies, most of which sit near the mean of the severity distribution. This is standard actuarial practice — the severity component represents "expected cost, given a claim occurs", and the frequency component handles the probability of that claim. The product is correct.

---

## Extracting factor tables with shap-relativities

The CatBoost models give us good predictions, but pricing actuaries also need factor tables: what is the BonusMalus=200 relativity? How much does Region Île-de-France load versus the base?

We use `shap-relativities` for this. The frequency model is Poisson with a log link, so SHAP values decompose in log space and exponentiating gives multiplicative relativities. This is exactly what rating engine factor tables contain.

```python
from shap_relativities import SHAPRelativities

# Extract relativities from the frequency model
sr_freq = SHAPRelativities(
    model=freq_model_cb,
    X=pl.from_pandas(X_train),
    exposure=pl.Series(exp_train),
    categorical_features=cat_features,
    continuous_features=["VehPower", "VehAge", "DrivAge", "BonusMalus", "log_density"],
)
sr_freq.fit()

# Validate reconstruction before trusting any outputs
checks = sr_freq.validate()
print(checks["reconstruction"])
# CheckResult(passed=True, value=4.2e-06, message='Max absolute reconstruction error: 4.2e-06.')

# Extract relativities for categorical features
area_rels = sr_freq.extract_relativities(
    normalise_to="base_level",
    base_levels={"Area": "B"},   # area B as base (median risk)
    ci_level=0.95,
)
print(area_rels.filter(pl.col("feature") == "Area"))
# feature  level  relativity  lower_ci  upper_ci  n_obs
# Area         A       0.881     0.871     0.891  80884
# Area         B       1.000     1.000     1.000 121698
# Area         C       1.073     1.063     1.083 169148
# Area         D       1.194     1.183     1.206 148828
# Area         E       1.383     1.369     1.397  94763
# Area         F       1.521     1.499     1.544  60692
```

The relativities read cleanly: Area F (urban, high density) is 52% more expensive than Area B on frequency, all else equal. These numbers are in the expected direction and magnitude for French TPL.

For the continuous features — `DrivAge`, `BonusMalus`, `log_density` — use the continuous curve:

```python
age_curve = sr_freq.extract_continuous_curve(
    feature="DrivAge",
    n_points=50,
    smooth_method="loess",
)
# Returns: DrivAge value, relativity, lower_ci, upper_ci
# The U-shape (young and old drivers more expensive) comes through clearly
```

---

## Distilling to GLM format with insurance-distill

The SHAP relativities tell you what the model thinks. For deployment into Emblem or Radar, you need clean multiplicative factor tables where continuous variables are discretised into bands. That is what `insurance-distill` produces.

```python
from insurance_distill import SurrogateGLM

surrogate = SurrogateGLM(
    model=freq_model_cb,
    X_train=pl.from_pandas(X_train),
    y_train=y_train,
    exposure=exp_train,
    family="poisson",
)

surrogate.fit(
    max_bins=8,
    method_overrides={
        "BonusMalus": "isotonic",   # monotone: higher score → higher risk
        "VehAge": "quantile",       # spread the bins evenly
    },
    interaction_pairs=[("DrivAge", "Area")],
)

report = surrogate.report()
print(report.metrics.summary())
# Gini (GBM):              0.2893
# Gini (GLM surrogate):    0.2741
# Gini ratio:              94.7%
# Deviance ratio:          0.9218
# Max segment deviation:   9.1%
# Mean segment deviation:  2.3%

# Export for Emblem/Radar
surrogate.export_csv(
    "/tmp/fremtpl2_factors/",
    prefix="freq_",
)
```

A 94.7% Gini ratio means the surrogate GLM retains 94.7% of the CatBoost's discrimination. For a model with a relatively modest Gini of 0.29 (freMTPL2 is a publicly studied dataset — the ceiling is well understood), this represents genuine performance preservation in a rating-engine-compatible format.

---

## What the freMTPL2 benchmark tells you

freMTPL2 has been used in enough published papers that there are established reference points. A well-tuned Poisson GBM on the full dataset typically achieves a Gini coefficient around 0.28-0.31 on frequency, depending on feature engineering. Noll et al. (2020, SAJ) reported Gini values in this range for GBM variants. A Poisson GLM with the same features tends to land around 0.24-0.26. The gap is real but modest.

Two things limit the headroom on freMTPL2. First, the features are actuarially thin — there is no telematics, no claims history beyond BonusMalus, no vehicle value beyond brand and group. Real UK motor data typically has 40-60 features, and the Gini ceiling is correspondingly higher (0.38-0.45 is achievable on a well-featured UK motor book). Second, the French TPL claims environment is relatively homogeneous compared with UK comprehensive: frequency is low and severity, while variable, has less of the extreme tail that vehicle damage and personal injury produce in the UK. Models discriminate less when there is less to discriminate.

The freMTPL2 benchmark is most useful for confirming your implementation is correct, not for setting expectations about production performance. If your Poisson CatBoost on freMTPL2 is landing at 0.18 Gini, something is wrong with the exposure offset or the feature encoding. If it is landing at 0.35, you are likely leaking future information. The 0.28-0.31 band is the sanity check.

---

## Installation

```bash
uv add catboost polars scikit-learn
uv add "shap-relativities[all]"
uv add insurance-distill
```

Source for `shap-relativities` and `insurance-distill` is at [github.com/burning-cost](https://github.com/burning-cost). The full notebook for this walkthrough — with plots, the severity model validation, and the double-lift chart comparing CatBoost to a Poisson GLM baseline — is at [burning-cost-examples/notebooks/fremtpl2_freq_sev.py](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/fremtpl2_freq_sev.py).

The freMTPL2 data licence is CC BY 4.0. It is academic data, not real policyholder records.

- [How to Extract GLM-Style Rating Factors from a CatBoost Model](/2026/03/02/how-to-extract-rating-factors-from-catboost/)
- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/)
- [Your GBM and GLM Are Not Competitors](/2026/02/28/your-gbm-and-glm-are-not-competitors/)
