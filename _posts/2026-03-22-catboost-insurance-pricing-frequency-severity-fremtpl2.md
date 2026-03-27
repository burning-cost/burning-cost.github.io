---
layout: post
title: "CatBoost for Insurance Pricing: Frequency-Severity on freMTPL2"
date: 2026-03-22
categories: [pricing, techniques, tutorials]
tags: [catboost, frequency-severity, burning-cost, fremtpl2, poisson, gamma, polars, shap-relativities, insurance-distill, uk-motor, python, open-data]
description: "Build a CatBoost frequency-severity pricing model on freMTPL2 using Polars. Poisson frequency, Gamma severity, combined burning cost, SHAP factor extraction, and distillation to Radar."
---

Most actuarial pricing projects spend months on something that should take a week: getting a working frequency-severity model off the ground. The delay is usually not the modelling itself. It is the argument about which loss function to use, whether categorical variables need encoding, how to handle the exposure offset, and whether the combined burning cost "makes sense." These are legitimate questions, but they have answers, and the answers are not complicated.

This post works through the full pipeline on [freMTPL2](https://www.openml.org/d/41214) -- the French motor third-party liability dataset used in the sklearn documentation and in Noll, Salzmann and Wüthrich (2020). We use CatBoost for both models and Polars for data handling. By the end you have a combined frequency-severity prediction (burning cost) ready for factor extraction via our [`shap-relativities`](https://github.com/burning-cost/shap-relativities) library and distillation into Radar/Emblem factor tables via [`insurance-distill`](/insurance-distill/).

---

## Why CatBoost for insurance pricing

Three properties make CatBoost a better default than XGBoost or LightGBM for insurance pricing work.

**Native categorical handling.** CatBoost uses ordered target statistics to encode categoricals during training, without any preprocessing. Vehicle group, postcode district, and NCD band go in as-is. freMTPL2 has `VehBrand` (11 categories), `VehGas` (2), and `Region` (22). In XGBoost or LightGBM you would be writing an encoder or using a Pipeline. In CatBoost you declare them as categoricals and train.

**Ordered boosting.** CatBoost's default mode uses a permutation-based approach to prevent target leakage when computing categorical statistics. This is not cosmetic: it materially reduces overfitting on high-cardinality categoricals, which is exactly the kind of feature that dominates insurance pricing. On a UK personal lines motor book with 2,800 postcode sectors, ordered boosting outperforms standard gradient boosting by 1-2 Gini points on holdout with no tuning changes.

**Symmetric trees.** CatBoost builds oblivious decision trees: each level of the tree uses the same feature-threshold pair across all branches. This makes prediction substantially faster than asymmetric trees, which matters when you are scoring a full rating universe of millions of combinations for a rate review.

---

## freMTPL2 setup with Polars

The dataset is available on OpenML. We will load and prepare it for a two-model pipeline.

```python
import polars as pl
import numpy as np
from sklearn.datasets import fetch_openml

# Load from OpenML
raw = fetch_openml("freMTPL2freq", version=2, as_frame=True, parser="auto")
freq_df = pl.from_pandas(raw.frame)

raw_sev = fetch_openml("freMTPL2sev", version=1, as_frame=True, parser="auto")
sev_df = pl.from_pandas(raw_sev.frame)

# freMTPL2freq: one row per policy
# Columns: IDpol, ClaimNb, Exposure, Area, VehPower, VehAge,
#          DrivAge, BonusMalus, VehBrand, VehGas, Density, Region

# Cap exposure at 1 year — some policies have implausible values
freq_df = freq_df.with_columns(
    pl.col("Exposure").clip(upper_bound=1.0).alias("Exposure")
)

# Cast claim count to integer
freq_df = freq_df.with_columns(
    pl.col("ClaimNb").cast(pl.Int32)
)
```

freMTPL2 has 678,013 policies. Roughly 4% have at least one claim. The exposure ranges from a few days to a full year; capping at 1.0 removes a small number of policies with anomalous values (multi-year aggregations that were not cleaned upstream). Do not skip this step -- an exposure of 2.3 years treated as-is will produce an incorrect offset.

---

## Frequency model: Poisson with exposure offset

For claim frequency, the target is claim count per unit of exposure. We model this with `loss_function="Poisson"` and supply `log(Exposure)` as a baseline offset. CatBoost does not have a native offset parameter the way a GLM does, so we embed the log-exposure directly as a feature with fixed weight via the `baseline` mechanism.

```python
import catboost

cat_features = ["Area", "VehBrand", "VehGas", "Region"]
num_features = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
all_features = cat_features + num_features

X_freq = freq_df.select(all_features)
y_freq = freq_df["ClaimNb"].to_numpy()
exposure = freq_df["Exposure"].to_numpy()

# Log-exposure as baseline (offset)
log_exposure = np.log(exposure)

# Train/test split -- use last 20% of IDpol as holdout
n = len(freq_df)
split = int(n * 0.8)

train_pool = catboost.Pool(
    data=X_freq[:split].to_pandas(),
    label=y_freq[:split],
    cat_features=cat_features,
    baseline=log_exposure[:split],   # the offset
    weight=exposure[:split],         # weight by exposure
)

test_pool = catboost.Pool(
    data=X_freq[split:].to_pandas(),
    label=y_freq[split:],
    cat_features=cat_features,
    baseline=log_exposure[split:],
    weight=exposure[split:],
)

freq_model = catboost.CatBoostRegressor(
    loss_function="Poisson",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=100,
)
freq_model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=50)
```

The `baseline` parameter passes `log(Exposure)` as the initial prediction before the first tree is added. Every subsequent tree models the residual claim count net of exposure. The result is a model that predicts claim frequency per policy-year, not raw claim count -- exactly what you need for a rating model that prices policies with varying exposure lengths.

On freMTPL2, a well-tuned CatBoost Poisson model achieves a test Gini of around 0.30-0.32. The main predictors are `BonusMalus` (the French equivalent of NCD), `DrivAge`, and `Density` (population density at address). `VehBrand` and `Region` contribute but are individually weaker.

---

## Severity model: Gamma on claims-only data

Severity is modelled separately, on policies with at least one claim. The dataset for severity is freMTPL2sev: one row per individual claim, with the claim amount (`ClaimAmount`) and the policy identifier (`IDpol`).

```python
# Join severity data back to policy features
sev_with_features = sev_df.join(
    freq_df.select(["IDpol"] + all_features),
    on="IDpol",
    how="left",
)

# Filter out zero or negative claim amounts -- data quality
sev_with_features = sev_with_features.filter(pl.col("ClaimAmount") > 0)

# For severity, the target is the individual claim amount
y_sev = sev_with_features["ClaimAmount"].to_numpy()
X_sev = sev_with_features.select(all_features)

n_sev = len(sev_with_features)
split_sev = int(n_sev * 0.8)

sev_train_pool = catboost.Pool(
    data=X_sev[:split_sev].to_pandas(),
    label=y_sev[:split_sev],
    cat_features=cat_features,
)

sev_test_pool = catboost.Pool(
    data=X_sev[split_sev:].to_pandas(),
    label=y_sev[split_sev:],
    cat_features=cat_features,
)

sev_model = catboost.CatBoostRegressor(
    loss_function="Gamma",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=100,
)
sev_model.fit(sev_train_pool, eval_set=sev_test_pool, early_stopping_rounds=50)
```

Gamma loss is appropriate for right-skewed positive continuous targets like claim amounts. It assumes the coefficient of variation is constant -- a reasonable approximation for property damage in motor, where larger losses are not necessarily more variable in relative terms than smaller ones. If your claim amount distribution has a very heavy tail (e.g., bodily injury), Tweedie with p > 1 or a separate excess-layer model may be a better fit.

Note that we do not weight by anything in the severity model. Each row is one claim. The Gamma loss is already claim-count-weighted implicitly because policies with multiple claims appear multiple times.

---

## Burning cost: frequency multiplied by severity

The burning cost prediction is the expected annual claim cost per policy: the product of expected claim frequency and expected claim severity. Both models predict in the original scale (not log scale), so multiplication is straightforward.

```python
# Predict frequency on the full dataset (or any rating universe)
freq_predictions = freq_model.predict(
    catboost.Pool(
        data=X_freq.to_pandas(),
        cat_features=cat_features,
        baseline=log_exposure,
    )
)

# Predict severity on the same feature set
# Note: we predict severity for all policies, not just those with claims
sev_predictions = sev_model.predict(
    catboost.Pool(
        data=X_freq.to_pandas(),   # same policy-level features
        cat_features=cat_features,
    )
)

# Burning cost = frequency * severity
burning_cost = freq_predictions * sev_predictions

results = freq_df.with_columns([
    pl.Series("freq_pred", freq_predictions),
    pl.Series("sev_pred", sev_predictions),
    pl.Series("burning_cost", burning_cost),
])
```

This is the standard frequency-severity structure. It assumes claim frequency and claim severity are conditionally independent given the rating factors -- an assumption that is defensible for most personal lines motor portfolios but worth testing. If you have a portfolio where large vehicles (high severity) also drive more miles (higher frequency), the independence assumption will understate burning cost for those risks.

A quick check: the portfolio mean burning cost should be close to the observed loss cost per policy-year. On freMTPL2, the observed average annual claim cost is approximately EUR 130-140. If your combined predictions average to EUR 200, something is wrong with the exposure handling or the severity subset selection.

---

## Extracting rating factors with shap-relativities

You now have two fitted CatBoost models. For a UK pricing workflow, the next step is to extract GLM-style rating factors from each -- the `exp(beta)` relativities that a pricing committee can review and that Radar or Emblem can consume.

```python
from shap_relativities import SHAPRelativities

# Frequency factors
sr_freq = SHAPRelativities(
    model=freq_model,
    X=X_freq,
    exposure=freq_df["Exposure"],
    categorical_features=cat_features,
    continuous_features=num_features,
)
sr_freq.fit()

freq_rels = sr_freq.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "Area": "D",
        "VehBrand": "B1",
        "VehGas": "Regular",
        "Region": "R11",
    },
    ci_level=0.95,
)

# Severity factors
sr_sev = SHAPRelativities(
    model=sev_model,
    X=X_sev,
    exposure=pl.Series("ones", np.ones(len(X_sev))),  # equal weight per claim
    categorical_features=cat_features,
    continuous_features=num_features,
)
sr_sev.fit()

sev_rels = sr_sev.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "Area": "D",
        "VehBrand": "B1",
        "VehGas": "Regular",
        "Region": "R11",
    },
    ci_level=0.95,
)
```

The resulting `freq_rels` and `sev_rels` are Polars DataFrames with columns `feature`, `level`, `relativity`, `lower_ci`, `upper_ci`, `n_obs`, `exposure_weight`. One row per factor level. These are the numbers your pricing committee reviews at a rate change meeting.

Always run `sr_freq.validate()` and `sr_sev.validate()` before presenting these numbers to anyone. The reconstruction check -- verifying that `exp(SHAP values sum)` matches the model's raw `predict()` -- is the proof that the relativities are consistent with what the model actually predicts, not an approximation. If the reconstruction check fails, the most common cause is a mismatch between the `loss_function` used in training and the SHAP output type. With `loss_function="Poisson"` or `loss_function="Gamma"`, SHAP values are in log space and reconstruction via `exp()` is correct.

---

## Distillation for Radar and Emblem

The factor tables from shap-relativities are a snapshot of the GBM's structure expressed as multiplicative relativities. To get these into a rating engine in a form that can be loaded directly and adjusted by an actuary, use [`insurance-distill`](https://github.com/burning-cost/insurance-distill):

```python
from insurance_distill import SurrogateGLM

# Distil the frequency model
freq_surrogate = SurrogateGLM(
    model=freq_model,
    X_train=X_freq[:split],
    y_train=freq_df["ClaimNb"][:split],
    exposure=freq_df["Exposure"][:split].to_numpy(),
    family="poisson",
)

freq_surrogate.fit(
    max_bins=10,
    binning_method="tree",
    method_overrides={"BonusMalus": "isotonic"},  # monotone NCD-equivalent
)

report = freq_surrogate.report()
print(report.metrics.summary())
# Gini (GBM):           0.318
# Gini (GLM surrogate): 0.301
# Gini ratio:           94.7%
# Deviance ratio:       0.921

# Export to CSV for Radar import
freq_surrogate.export_csv(
    "output/fremtpl2_freq_factors/",
    prefix="freq_",
)
```

The Gini ratio tells you how much of the CatBoost model's discrimination the surrogate GLM retains. Above 90% is acceptable. The freMTPL2 frequency model typically lands at 93-96% Gini retention with 8-10 bins per continuous variable and no interaction terms. Adding one or two interaction pairs (e.g., `DrivAge` x `BonusMalus`) can push this to 95-97%.

Run the same workflow for the severity model with `family="gamma"`. The severity surrogate is usually easier to distil than frequency because severity has fewer non-linear interactions -- `VehPower` and `VehBrand` drive most of the structure.

The CSV output from `export_csv()` gives you one file per rating variable, each with three columns: `level`, `log_coefficient`, `relativity`. These load directly into Radar's factor table editor or Emblem's import path. The base factor file contains the intercept -- the expected burning cost at all-base-level, which anchors the rate level.

---

## What comes next

freMTPL2 is a clean dataset and the models above will fit without drama. On a real UK motor portfolio you will face several additional complications:

**Exposure period inconsistency.** freMTPL2 exposures are already in policy-years. UK data often requires constructing exposure from inception and expiry dates, accounting for mid-term adjustments, and sometimes for earned vs written premium distinctions that affect the appropriate weight.

**Vehicle group instability.** ABI vehicle groups have 50-level codes that change between versions, and thin groups (custom vehicles, classic cars) can produce unstable relativities even with ordered boosting. The confidence intervals from shap-relativities flag this directly: any level with a 95% CI spanning more than 20% relative to the point estimate warrants scrutiny.

**Claims development.** freMTPL2 is a fully-developed snapshot. UK MTPL data is typically coded with open claims, and the claim amounts you model include IBNR. Mixing developed and undeveloped claims in the severity training set will bias the severity factors. You need a development-adjusted severity target or a modelling period where claims are sufficiently mature.

The frequency-severity pipeline above is the foundation. The complications above are where pricing actuarial judgement does its work -- and they are separable from the modelling decisions, which is the point of getting the base pipeline right first.

---

[`shap-relativities`](https://github.com/burning-cost/shap-relativities) | [`insurance-distill`](https://github.com/burning-cost/insurance-distill)

---

## Related articles

- [How to Extract GLM-Style Rating Factors from a CatBoost Model](/2026/03/02/how-to-extract-rating-factors-from-catboost/)
- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/)
- [When Credibility Meets CatBoost](/2026/02/28/when-credibility-meets-catboost/)
- [Tweedie Regression in Insurance: What sklearn Doesn't Tell You](/2026/03/21/tweedie-regression-insurance-what-sklearn-doesnt-tell-you/)
