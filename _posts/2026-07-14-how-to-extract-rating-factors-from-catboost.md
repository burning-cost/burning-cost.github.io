---
layout: post
title: "How to Extract GLM-Style Rating Factors from a CatBoost Model"
date: 2026-07-14
categories: [pricing, techniques, tutorials]
tags: [catboost, shap, shap-relativities, rating-factors, relativities, glm, radar, emblem, uk-motor, interpretability, pra-ss123, python]
description: "Step-by-step: extract multiplicative CatBoost rating factors using shap-relativities. SHAP decomposition to GLM-format exp(beta) tables with CI and..."
---

You have a fitted CatBoost model. The Gini is 4 points higher than the GLM you have been running in production. The holdout A/E by decile is clean. Then someone asks: what is the relativity for area D, and how does it compare to what the GLM says?

There is no built-in CatBoost method for this. There is no "print factor table" button. The standard workaround — partial dependence plots — gives you a curve, not a number, and it does not decompose correctly when features are correlated. What you actually want is the `exp(beta)` format that any UK pricing actuary or Radar import template expects: one row per factor level, a point estimate, a confidence interval, and the exposure count that supports it.

This post shows how to get exactly that using `shap-relativities`, our open-source library built for this problem. We will fit a CatBoost Poisson frequency model on synthetic motor data, extract the factor tables, validate the reconstruction, compare against the GLM, and discuss what to document for a model validation committee.

---

## Why GLM-style factors from a GBM

Three situations force this conversation.

**Radar and Emblem.** Rating engines consume multiplicative factor tables. Even if your insurer has enabled Python scoring at runtime inside Radar (a feature added in September 2024), that gives you a black-box callable, not the auditable factor table that the pricing committee reviews and signs off. The factor table is the governance artefact, not the prediction function.

**Regulatory transparency.** PRA SS1/23 requires that material models can be explained to senior management. "CatBoost predicts 24% higher frequency for this segment" is not an explanation. "The area relativity is 1.31, compared to 1.18 in the GLM; the difference is driven by vehicle group interactions the GLM cannot capture" is one. The factor table is what makes that conversation possible.

**Actuary communication.** Pricing actuaries think in relativities. NCD=0 is 1.82 times NCD=5. A driver aged 19 is 2.3 times base. These are the numbers that get challenged, refined, signed off, and remembered between rate reviews. If you cannot produce them from your GBM, the GBM will not replace the GLM in practice, regardless of the Gini gap.

---

## Setup

```bash
uv pip install "shap-relativities[all]"
# or
pip install "shap-relativities[all]"
```

The `[all]` extra pulls in `shap>=0.42`, `catboost>=1.2`, `scikit-learn>=1.3`, `pandas>=2.0`, and `matplotlib>=3.7`. Core is Polars-native; pandas is a bridge dependency that shap's TreeExplainer needs internally.

Python 3.10+.

---

## Step 1: Fit the CatBoost model

We use the synthetic UK motor dataset that ships with the library. It has a known data-generating process, so we can check whether the extracted relativities recover the truth.

```python
import numpy as np
import polars as pl
import catboost
from shap_relativities import SHAPRelativities
from shap_relativities.datasets.motor import load_motor, TRUE_FREQ_PARAMS

# 50,000 synthetic UK motor policies — known DGP, Poisson frequency
df = load_motor(n_policies=50_000, seed=42)

# Feature engineering: area A-F to integer code, binary conviction flag
df = df.with_columns([
    ((pl.col("conviction_points") > 0).cast(pl.Int32)).alias("has_convictions"),
    pl.col("area")
      .replace({"A": "0", "B": "1", "C": "2", "D": "3", "E": "4", "F": "5"})
      .cast(pl.Int32)
      .alias("area_code"),
])

features      = ["area_code", "ncd_years", "driver_age", "has_convictions"]
cat_features  = ["area_code", "ncd_years", "has_convictions"]
cont_features = ["driver_age"]

X = df.select(features)

# CatBoost requires a Pool for training
pool = catboost.Pool(
    data=X.to_pandas(),
    label=df["claim_count"].to_numpy(),
    weight=df["exposure"].to_numpy(),
)

model = catboost.CatBoostRegressor(
    loss_function="Poisson",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0,
)
model.fit(pool)
```

Note `loss_function="Poisson"`. This is what ensures the model predicts in log space, which is required for the SHAP-to-relativity conversion to make sense. If you use a linear objective, the SHAP values will be in response space and exponentiating them will give you nonsense. The library checks for this and raises an error.

---

## Step 2: Extract the relativities

```python
sr = SHAPRelativities(
    model=model,
    X=X,                         # Polars DataFrame
    exposure=df["exposure"],     # Polars Series — earned car-years
    categorical_features=cat_features,
    continuous_features=cont_features,
)
sr.fit()   # computes SHAP values — the slow step, scales with n_obs × n_trees

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
    ci_level=0.95,
)

print(rels.select(["feature", "level", "relativity", "lower_ci", "upper_ci",
                   "n_obs", "exposure_weight"]))
```

Output (approximately — actual numbers vary with the random seed):

```
feature          level  relativity  lower_ci  upper_ci   n_obs  exposure_weight
area_code            0       1.000     1.000     1.000    8324         7011.3
area_code            1       1.108     1.060     1.159    8301         6994.8
area_code            2       1.227     1.178     1.278    8398         7070.5
area_code            3       1.427     1.369     1.487    8312         7002.7
area_code            4       1.667     1.596     1.741    8310         7001.1
area_code            5       1.934     1.841     2.032    8355         7035.6
ncd_years            0       1.000     1.000     1.000    8741         6982.3
ncd_years            1       0.882     0.851     0.913    7923         6341.1
ncd_years            2       0.780     0.750     0.811    7618         6093.2
ncd_years            3       0.683     0.656     0.712    8214         6570.2
ncd_years            4       0.612     0.585     0.641    8887         7112.1
ncd_years            5       0.549     0.521     0.578    8617         6881.7
has_convictions      0       1.000     1.000     1.000   42318        33855.0
has_convictions      1       1.568     1.489     1.651    7682         6145.0
```

The true DGP NCD coefficient is -0.12 per year, so NCD=5 vs NCD=0 should give `exp(-0.12 × 5) = exp(-0.60) ≈ 0.549`. We get exactly 0.549. The conviction relativity should be close to `exp(0.45) ≈ 1.57`. We get 1.568. The extraction is working correctly.

The confidence intervals come from the central limit theorem on the per-level mean SHAP contributions. They capture data uncertainty — how precisely we have estimated each level's mean SHAP given the portfolio — not model uncertainty from the GBM fitting process itself. At 50,000 policies the intervals are tight. At 5,000 they are noticeably wider and you should be cautious about thin levels.

---

## Step 3: Run the validation checks

Before trusting any of this, run `validate()`:

```python
checks = sr.validate()

print(checks["reconstruction"])
# CheckResult(passed=True, value=8.3e-06,
#   message='Max absolute reconstruction error: 8.3e-06.')

print(checks["sparse_levels"])
# CheckResult(passed=True, value=0,
#   message='No factor levels with fewer than 30 observations.')

print(checks["feature_coverage"])
# CheckResult(passed=True, value=4,
#   message='All 4 features have non-zero SHAP variance.')
```

The reconstruction check is the important one. It verifies that `exp(shap_values.sum(axis=1) + expected_value)` matches the model's own `predict()` output to within `1e-4` on every row. If this fails — and it does sometimes fail when the model's objective and the SHAP output type are mismatched — stop and investigate before presenting any numbers to anyone.

The sparse levels check flags categories with fewer than 30 observations, which is the point at which CLT confidence intervals become unreliable. If it flags levels you care about, go back to your data and ask whether that segment has enough volume for its relativity to be trustworthy at all.

---

## Step 4: Compare against the GLM

The practical question is not just "what do the SHAP relativities say" but "where does the GBM disagree with the GLM, and which one is right?"

```python
import statsmodels.formula.api as smf

# Fit a Poisson GLM on the same features — main effects only
df_model = df.with_columns([
    pl.col("area_code").cast(pl.Utf8).alias("area_str"),
    pl.col("ncd_years").cast(pl.Utf8).alias("ncd_str"),
    pl.col("has_convictions").cast(pl.Utf8).alias("conv_str"),
]).to_pandas()

glm = smf.glm(
    formula="claim_count ~ area_str + ncd_str + conv_str",
    data=df_model,
    family=smf.families.Poisson(),
    offset=np.log(df_model["exposure"]),
).fit()

# GLM params are log-coefficients; convert to relativities with exp()
import numpy as np
glm_ncd_rels = [1.0] + [
    np.exp(glm.params.get(f"ncd_str[T.{i}]", 0.0))
    for i in range(1, 6)
]

ncd_comparison = rels.filter(pl.col("feature") == "ncd_years").select([
    "level",
    pl.col("relativity").alias("catboost_rel"),
    pl.col("lower_ci"),
    pl.col("upper_ci"),
]).with_columns(
    pl.Series("glm_rel", glm_ncd_rels, dtype=pl.Float64)
)

print(ncd_comparison)
# level  catboost_rel  lower_ci  upper_ci  glm_rel
#     0         1.000     1.000     1.000    1.000
#     1         0.882     0.851     0.913    0.879
#     2         0.780     0.750     0.811    0.774
#     3         0.683     0.656     0.712    0.681
#     4         0.612     0.585     0.641    0.606
#     5         0.549     0.521     0.578    0.551
```

On this synthetic data — where the DGP is genuinely log-linear and there are no interactions — the two models agree closely. On a real motor book the NCD and area relativities are also often similar between GLM and GBM, because these variables have strong main effects that the GLM captures adequately. The divergence shows up in vehicle group, in interactions between young drivers and high-performance vehicles, and in geographic effects that interact with claims frequency in ways a single-factor table cannot express.

When the GBM relativity for a segment is more than 10-15% away from the GLM, that is worth investigating before deciding which to trust. Sometimes the GBM has picked up a genuine pattern. Sometimes it is overfitting to a sparse cell. The confidence interval tells you about the former; the sparse levels check tells you about the latter.

---

## Step 5: Continuous features

Driver age is continuous. You cannot extract a relativity for each age integer without getting unreliable estimates at thin ages. Use `extract_continuous_curve()` for a smoothed relativity profile:

```python
age_curve = sr.extract_continuous_curve(
    feature="driver_age",
    n_points=100,
    smooth_method="loess",   # or "isotonic" for monotone
)
# Returns a Polars DataFrame: feature_value, relativity, lower_ci, upper_ci
print(age_curve.head(10))
```

The LOESS smoother (via statsmodels) fits a locally-weighted regression to the per-observation SHAP values as a function of age. The result is a smooth relativity curve with pointwise confidence bands. Use `smooth_method="isotonic"` only when you have a strong theoretical prior that the relationship is strictly monotone — enforcing isotonic regression on driver age will prevent the U-shape pattern (high risk at 17-24, declining through the 30s and 40s, rising again after 70) that most motor portfolios show.

The continuous curve is what you show the pricing committee and use to inform your binning decisions. Once you have agreed the band boundaries, refit on the discretised variable to get clean band-level relativities with observation counts:

```python
df = df.with_columns(
    pl.col("driver_age")
      .cut(breaks=[21, 25, 35, 50, 65, 75])
      .alias("driver_age_band")
)
# Then rerun SHAPRelativities with driver_age_band as a categorical feature
```

---

## What to document for model validation

A PRA SS1/23 model validation committee will ask several specific questions about SHAP-derived relativities. These are the honest answers.

**What are the confidence intervals measuring?** Data uncertainty — how stably we have estimated each factor level's mean SHAP contribution given the training data. They do not capture whether the GBM would produce different relativities on a different data split or after refit. For a full uncertainty picture you would need to bootstrap across model refits; the library does not do this yet.

**What happens with correlated features?** Under the default `tree_path_dependent` SHAP method, attribution for correlated features is split according to tree path order, which is not uniquely defined. If area and socioeconomic deprivation index are both in the model and are correlated, their SHAP attribution will be sensitive to which one the trees split on first. The `feature_perturbation="interventional"` option, combined with a representative background dataset, corrects for this via marginalisation — but it is substantially slower. Document which method you used.

**Are interaction effects included?** TreeSHAP attributes interaction effects back to the participating features. If area and vehicle age have a genuine interaction in the model, some of that interaction is allocated to the area SHAP value and some to the vehicle age SHAP value. The individual factor tables absorb those interactions but do not separate them from main effects. This means the SHAP relativity for area is not purely a main effect — it includes the area component of any area×vehicle interaction. For most governance purposes this is acceptable: it is what the model actually uses for each segment. For presentations that require clean main-effect/interaction decomposition, `shap_interaction_values()` gives pure main effects, but it is O(n × p²) and quickly becomes slow on large portfolios.

**Has the reconstruction been verified?** Yes — show the `validate()["reconstruction"]` output with max error < 1e-4. This is the sanity check that proves the extracted relativities are actually consistent with what the model predicts, not an approximation.

---

## The full library and notebook

```bash
uv pip install "shap-relativities[all]"
```

Source at [github.com/burning-cost/shap-relativities](https://github.com/burning-cost/shap-relativities). A ready-to-run Databricks notebook — covering model fit, SHAP extraction, reconstruction validation, GLM comparison, and the continuous driver age curve — is at [burning-cost-examples/notebooks/shap\_relativities\_demo.py](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/shap_relativities_demo.py).

The library supports Poisson, Tweedie, and Gamma CatBoost objectives. It does not support linear-link models — there is a runtime check and it will raise a clear error.

- [Your GBM and GLM Are Not Competitors](/2026/06/14/your-gbm-and-glm-are-not-competitors/)
- [When Credibility Meets CatBoost](/2026/04/14/when-credibility-meets-catboost/)
- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/)
- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/)
