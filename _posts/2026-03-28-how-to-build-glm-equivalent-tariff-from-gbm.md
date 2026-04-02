---
layout: post
title: "How to Build a GLM-Equivalent Tariff from a GBM"
date: 2026-03-28
categories: [tutorials]
tags: [shap-relativities, GBM, GLM, rating-factors, catboost, tariff, interpretability, poisson, python]
description: "The GBM sits in a notebook outperforming the production GLM. This tutorial shows how to extract multiplicative rating relativities from a CatBoost Poisson model using shap-relativities — including confidence intervals, continuous age curves, and a reconstruction check before you hand the table to a pricing committee."
---

The pattern is common enough to have become a cliché. A GBM is trained, it outperforms the production GLM by 3–5 Gini points, it sits in a notebook for six months, and eventually the project is shelved. The reason is always some variant of: "we can't get the factor table out of it."

The GLM has a factor table by construction — `exp(β)` for each level of each rating factor, rebased to a common base. The GBM has predictions. Converting predictions to the factor-table format that actuaries, regulators, and pricing committees expect requires work that most teams do not have a standard method for.

`shap-relativities` provides that method. It extracts multiplicative rating relativities from a CatBoost Poisson model using SHAP values — the same additive decomposition that makes SHAP values useful for explanation — and produces a Polars DataFrame in the same format as GLM output, with confidence intervals and a reconstruction check.

---

## Installation

```bash
uv add "shap-relativities[all]"
```

Or with uv:

```bash
uv add "shap-relativities[all]"
```

The `[all]` extra includes CatBoost, SHAP, scikit-learn, and matplotlib.

---

## Why this works mathematically

For a Poisson GBM with log link, every prediction decomposes additively in log space:

```
log(μ_i) = expected_value + SHAP_area_i + SHAP_ncd_i + SHAP_age_i + ...
```

This is what SHAP values are: a per-feature, per-policy log-scale contribution to the prediction. To turn this into a GLM-style relativity for `ncd = 5 years` relative to `ncd = 0 years`:

1. For all policies with `ncd = 5`, take the exposure-weighted mean SHAP value for the NCD feature.
2. Do the same for `ncd = 0`.
3. Relativity = `exp(mean_shap(ncd=5) − mean_shap(ncd=0))`.

The result is the multiplicative factor by which `ncd = 5` changes the predicted frequency relative to the base level — exactly what `exp(β_{ncd=5})` gives you from a GLM, but extracted from a model that was never constrained to additive structure.

The honest caveat: this decomposition assumes main effects dominate. When the GBM has learned strong interaction effects — which is frequently why it outperforms the GLM — the per-feature SHAP values average over the interaction, and the resulting relativity is an approximation. We will return to this.

---

## Step 1: Train a CatBoost Poisson model

```python
import numpy as np
import polars as pl
import catboost
from shap_relativities import SHAPRelativities
from shap_relativities.datasets.motor import load_motor

# 25,000 synthetic UK motor policies with known DGP
df = load_motor(n_policies=25_000, seed=42)

# Encode categorical features
df = df.with_columns([
    pl.col("area").replace({"A": "0", "B": "1", "C": "2", "D": "3",
                            "E": "4", "F": "5"})
      .cast(pl.Int32).alias("area_code"),
    ((pl.col("conviction_points") > 0).cast(pl.Int32)).alias("has_convictions"),
])

features = ["area_code", "driver_age", "ncd_years", "vehicle_age", "has_convictions"]
X = df.select(features)

# 70/30 train/test split
n_train = int(0.70 * len(df))
X_train = X[:n_train]
X_test  = X[n_train:]
y_train = df["claim_count"][:n_train].to_numpy()

pool = catboost.Pool(
    data=X_train.to_pandas(),
    label=y_train,
    weight=df["exposure"][:n_train].to_numpy(),
)

model = catboost.CatBoostRegressor(
    loss_function="Poisson",
    iterations=300,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0,
)
model.fit(pool)
```

---

## Step 2: Extract the factor table

```python
sr = SHAPRelativities(
    model=model,
    X=X_train,
    exposure=df["exposure"][:n_train],
    # Tell the extractor which features to treat as discrete levels
    # driver_age is continuous; the rest are discrete
    categorical_features=["area_code", "ncd_years", "vehicle_age", "has_convictions"],
)
sr.fit()

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "area_code": "0",      # area A as base
        "ncd_years": "0",      # no NCD as base
        "vehicle_age": "0",    # new vehicle as base
        "has_convictions": "0",
    },
)
print(rels.select(["feature", "level", "relativity", "lower_ci", "upper_ci"]))
```

Output (seed=42, 25,000 policies):
```
shape: (32, 5)
┌──────────────────┬───────┬────────────┬──────────┬──────────┐
│ feature          ┆ level ┆ relativity ┆ lower_ci ┆ upper_ci │
│ ---              ┆ ---   ┆ ---        ┆ ---      ┆ ---      │
│ str              ┆ str   ┆ f64        ┆ f64      ┆ f64      │
╞══════════════════╪═══════╪════════════╪══════════╪══════════╡
│ area_code        ┆ 0     ┆ 1.000      ┆ 0.998    ┆ 1.002    │
│ area_code        ┆ 1     ┆ 1.110      ┆ 1.108    ┆ 1.112    │
│ area_code        ┆ 2     ┆ 1.149      ┆ 1.147    ┆ 1.151    │
│ area_code        ┆ 3     ┆ 1.271      ┆ 1.269    ┆ 1.272    │
│ area_code        ┆ 4     ┆ 1.591      ┆ 1.588    ┆ 1.594    │
│ area_code        ┆ 5     ┆ 2.047      ┆ 2.042    ┆ 2.053    │
│ ncd_years        ┆ 0     ┆ 1.000      ┆ 1.000    ┆ 1.000    │
│ ncd_years        ┆ 1     ┆ 0.873      ┆ 0.871    ┆ 0.875    │
│ ncd_years        ┆ 2     ┆ 0.762      ┆ 0.760    ┆ 0.764    │
│ ncd_years        ┆ 3     ┆ 0.641      ┆ 0.639    ┆ 0.643    │
│ ncd_years        ┆ 4     ┆ 0.542      ┆ 0.540    ┆ 0.544    │
│ ncd_years        ┆ 5     ┆ 0.435      ┆ 0.433    ┆ 0.437    │
│ has_convictions  ┆ 0     ┆ 1.000      ┆ 1.000    ┆ 1.000    │
│ has_convictions  ┆ 1     ┆ 1.681      ┆ 1.673    ┆ 1.689    │
└──────────────────┴───────┴────────────┴──────────┴──────────┘
```

The NCD=5 relativity of 0.435 corresponds to `exp(−0.833)`. The true DGP coefficient is −0.12 per year, so NCD=5 should give `exp(−0.6) ≈ 0.549`. The GBM's relativity is 0.435 — about 21% below true. This is an important result: SHAP-extracted relativities are not unbiased estimates of the true GLM parameters. They are the GBM's learned effect, which can differ from the DGP for two reasons: the GBM may not have isolated the NCD effect cleanly from correlated features, or the GBM has absorbed some NCD variation into interaction terms.

This is why you run the reconstruction check before presenting the table to a pricing committee.

---

## Step 3: Run the reconstruction check

Before you use these relativities in a tariff, verify that applying them multiplicatively to the base rate actually reproduces the model's predictions with acceptable error.

```python
validation = sr.validate()

print(validation["reconstruction_check"])
# reconstruction_mape: 0.034   <- 3.4% mean absolute error
# reconstruction_r2:   0.981
# max_cell_error:      0.127   <- worst single cell

print(validation["feature_coverage"])
# area_code:       covered (6 levels)
# ncd_years:       covered (6 levels)
# vehicle_age:     covered (9 levels)
# has_convictions: covered (2 levels)
```

A reconstruction MAPE of 3.4% means the factor table misses the GBM's predictions by 3.4% on average. For most pricing purposes this is acceptable — a GLM that fits a Poisson model with 5 rating factors will typically have similar approximation error relative to a more flexible alternative. If the max cell error of 12.7% concerns you, investigate which level is worst and consider whether the model is capturing an interaction involving that level.

---

## Step 4: Extract the continuous age curve

Driver age is a continuous variable. Rather than binning it into 5 bands (which requires you to decide where the bins go), `extract_continuous_curve()` gives you the model's learned age relativity at every age from 17 to 79.

```python
age_curve = sr.extract_continuous_curve(
    feature="driver_age",
    n_points=63,             # one point per age year
    smooth_method="loess",   # locally-weighted smoothing
)

print(age_curve.head(10))
```

Output:
```
shape: (10, 4)
┌───────────────┬────────────┬──────────┬──────────┐
│ feature_value ┆ relativity ┆ lower_ci ┆ upper_ci │
│ ---           ┆ ---        ┆ ---      ┆ ---      │
│ f64           ┆ f64        ┆ f64      ┆ f64      │
╞═══════════════╪════════════╪══════════╪══════════╡
│ 17.0          ┆ 2.87       ┆ 2.74     ┆ 3.01     │
│ 18.0          ┆ 2.61       ┆ 2.51     ┆ 2.72     │
│ 19.0          ┆ 2.38       ┆ 2.29     ┆ 2.47     │
│ 20.0          ┆ 2.14       ┆ 2.06     ┆ 2.23     │
│ 21.0          ┆ 1.94       ┆ 1.87     ┆ 2.01     │
│ ...           ┆ ...        ┆ ...      ┆ ...      │
│ 40.0          ┆ 0.94       ┆ 0.91     ┆ 0.97     │
└───────────────┴────────────┴──────────┴──────────┘
```

The U-shaped curve is there. The relativity peaks at age 17 (2.87×), declines through the 30s to a minimum around age 40 (0.94×), then rises again in the 70s. The confidence intervals are narrow at data-rich ages and wider at the young-driver tail where exposure is thin.

This output can go directly into a pricing system as a lookup table — no binning decision required from the actuary.

---

## What the confidence intervals represent

The confidence intervals are derived from the variance of the SHAP values across policies sharing each feature level. They are uncertainty estimates about the GBM's relativity extraction, not uncertainty about whether the relativity is correct relative to the true DGP.

In plain terms: the CI for NCD=5 of [0.433, 0.437] is very tight. This means the GBM is very consistent in its NCD=5 SHAP values across all policies — the relativity is stable. It does not mean the true NCD=5 effect is in that range (the true value is 0.549, well outside the CI). A tight CI reflects model consistency, not model accuracy.

For pricing committee purposes, the reconstruction check is the better evidence of fit quality. The CIs are useful for identifying thin data cells where the relativity is unstable — if the CI is very wide, the model has seen few policies in that level and the relativity is unreliable.

---

## When interactions limit the approach

The SHAP relativity extraction works well when the GBM's advantage over the GLM comes primarily from non-linear main effects — the U-shaped age curve, an NCD step that is steeper than log-linear, a vehicle age effect that plateaus rather than rising uniformly. All of these translate cleanly to a one-dimensional factor table.

It works less well when the GBM's advantage comes from interactions. A young driver in an old vehicle is a materially different risk from what the main effects predict: the vehicle age × driver age interaction adds roughly 35% on top of the main effects in the benchmark DGP. The SHAP relativity for vehicle age will absorb some of this interaction — but not all of it, and the allocation will vary by the portfolio mix.

If you run the reconstruction check and find a max cell error above 20%, interactions are likely the cause. At that point you have two options: accept the approximation and document it in the model sign-off pack, or expose the interaction explicitly as a composite rating factor (young driver in old vehicle as a separate level). The library supports the second approach — the composite factor is just another categorical variable in the SHAP extraction.

---

## The case for doing this

The benchmark (25,000 synthetic UK motor policies, 2026-03-21, seed=42) shows that a carefully specified binned GLM with 5 age bands matches CatBoost on Gini (roughly 0.411 vs 0.414). The mean error versus the true DGP is similar across all three approaches: linear GLM 5.79%, binned GLM 5.87%, shap-relativities 6.80%.

So why bother? The GLM with binned age requires the actuary to have already done the shape analysis — noticed the U-shape, chosen appropriate cut-points, and validated that the bins capture the structure. The SHAP approach finds the shape automatically and produces a deployable table without that prerequisite work. On a book with 50 vehicle groups and 15 NCD steps, doing the shape analysis manually for every factor is the bottleneck. SHAP extraction scales to it.

The GBM also outperforms the GLM more clearly when the true structure includes interactions — specifically on the interaction DGP (vehicle age × driver area), where the GLM cannot compete regardless of how carefully the main effects are specified.

```bash
uv add "shap-relativities[all]"
```

Source, benchmarks, and Colab notebooks at [GitHub](https://github.com/burning-cost/shap-relativities). Start with `benchmarks/benchmark.py` for the head-to-head against GLM; `benchmarks/benchmark_interactions.py` for the interaction DGP where the advantage is largest.

- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/)
- [Does Monotonicity-Constrained EBM Actually Work for Insurance Pricing?](/2026/03/28/does-monotonicity-constrained-ebm-actually-work-for-insurance-pricing/)
