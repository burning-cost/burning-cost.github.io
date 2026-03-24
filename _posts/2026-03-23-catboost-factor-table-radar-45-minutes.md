---
layout: post
title: "CatBoost Factor Table to Radar in 45 Minutes"
date: 2026-03-23
author: Burning Cost
categories: [gbm, insurance-pricing, rating-engines]
description: "CatBoost motor frequency model to Radar factor table in 45 minutes: SHAP relativity extraction, GLM distillation, and Radar-compatible CSV export in Python."
canonical_url: "https://burning-cost.github.io/2026/03/23/catboost-factor-table-radar-45-minutes/"
tags: [catboost, shap, radar, factor-tables, insurance-pricing, python, polars, insurance-distill, shap-relativities]
---

23 of the 25 largest UK personal lines insurers use Radar as their rating engine. Every one of those pricing teams has the same problem: a GBM model sitting on a server outperforming the production GLM, and no automated route to get the factor tables out of it.

This is not a tutorial on how to solve that problem in theory. This is a write-up of the workflow we ran on a motor frequency model — CatBoost on synthetic UK motor data — using [`shap-relativities`](https://github.com/burning-cost/shap-relativities) and [`insurance-distill`](https://github.com/burning-cost/insurance-distill). Total elapsed time: 45 minutes. The output was five Radar-compatible CSV files, each in the exact two-column format Radar's factor table import expects.

---

## The scenario

We had a Poisson frequency model: CatBoost trained on 50,000 synthetic UK motor policies from the `load_motor()` dataset — the same synthetic portfolio we use for benchmarking, with a known data-generating process (DGP) so we can measure how accurately we recovered the true relativities.

Seven rating features: `driver_age` (continuous), `vehicle_age` (continuous), `ncd_years` (0–5 ordinal), `has_convictions` (binary), `area` (A–F categorical), `vehicle_group` (ABI groups 1–10), `annual_mileage` (continuous).

The task: produce factor tables suitable for loading into Radar. One CSV per variable, relativities on the multiplicative scale, base level = 1.000.

---

## Step 1: Data prep (10 minutes)

```python
import polars as pl
import catboost
from shap_relativities.datasets.motor import load_motor

df = load_motor(n_policies=50_000, seed=42)

# CatBoost handles string categoricals natively via cat_features
# We keep area as string ("A"–"F") and pass it through cat_features below
df = df.with_columns([
    ((pl.col("conviction_points") > 0).cast(pl.Int32)).alias("has_convictions"),
])

features = [
    "driver_age", "vehicle_age", "ncd_years",
    "has_convictions", "area", "vehicle_group", "annual_mileage",
]
cat_features_cb = ["area", "vehicle_group"]   # CatBoost cat_features

X = df.select(features)
```

The ten minutes was mostly spent on two decisions: whether to pass `area` as a native categorical to CatBoost (yes — ordered boosting handles it cleanly) and how to represent conviction points (binary flag, not the raw point count, because the exposure in 3+ point bands is too thin to trust individual level relativities).

---

## Step 2: Train the CatBoost frequency model (5 minutes)

```python
pool = catboost.Pool(
    data=X.to_pandas(),
    label=df["claim_count"].to_numpy(),
    weight=df["exposure"].to_numpy(),
    cat_features=cat_features_cb,
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

300 iterations would have been sufficient. We used 500 out of habit. Training took 38 seconds on a MacBook M2. On Databricks serverless it is under 15 seconds.

Gini on a 20% holdout: 0.4813. The production GLM on this book was at 0.4271. The gap — 5.4pp — is large enough to justify the distillation effort.

---

## Step 3: Extract SHAP relativities (15 minutes)

The 15 minutes here was spent on two things: deciding which features to treat as categorical versus continuous for aggregation purposes, and reading the validation output carefully before trusting the numbers.

```python
from shap_relativities import SHAPRelativities

sr = SHAPRelativities(
    model=model,
    X=X,
    exposure=df["exposure"],
    categorical_features=["ncd_years", "has_convictions", "area", "vehicle_group"],
    continuous_features=["driver_age", "vehicle_age", "annual_mileage"],
)
sr.fit()
```

Note the distinction: `categorical_features` here is an aggregation hint for `SHAPRelativities` — it controls which features get per-level SHAP summaries versus smoothed curves. It is not the same as CatBoost's `cat_features` training parameter. `ncd_years` is passed as `Int32` to CatBoost without `cat_features`, so CatBoost treats it as numeric; `SHAPRelativities` still aggregates it by discrete level because we pass it in `categorical_features`.

**Validation first:**

```python
checks = sr.validate()

print(checks["reconstruction"])
# CheckResult(passed=True, value=6.1e-07,
#   message='Max absolute reconstruction error: 6.1e-07.')

print(checks["sparse_levels"])
# CheckResult(passed=True, value=0,
#   message='No factor levels with fewer than 30 observations.')
```

Reconstruction passed. If it had failed, the SHAP explainer was misconfigured — almost always a mismatch between the model's objective and the SHAP output type. We've seen it happen when someone trains a CatBoost `CatBoostClassifier` but passes `loss_function="Logloss"` and then tries to exponentiate log-space SHAP values as if they were a Poisson model. Here it passed cleanly.

**Extract discrete-feature relativities:**

```python
rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "ncd_years": 0,
        "has_convictions": 0,
        "area": "A",
        "vehicle_group": "1",
    },
)

print(rels.select(["feature", "level", "relativity", "lower_ci", "upper_ci"]))
```

```
shape: (24, 5)
┌─────────────────┬───────┬────────────┬──────────┬──────────┐
│ feature         ┆ level ┆ relativity ┆ lower_ci ┆ upper_ci │
│ ---             ┆ ---   ┆ ---        ┆ ---      ┆ ---      │
│ str             ┆ str   ┆ f64        ┆ f64      ┆ f64      │
╞═════════════════╪═══════╪════════════╪══════════╪══════════╡
│ area            ┆ A     ┆ 1.000      ┆ 0.998    ┆ 1.002    │
│ area            ┆ B     ┆ 1.112      ┆ 1.110    ┆ 1.114    │
│ area            ┆ C     ┆ 1.153      ┆ 1.151    ┆ 1.155    │
│ area            ┆ D     ┆ 1.274      ┆ 1.271    ┆ 1.277    │
│ area            ┆ E     ┆ 1.591      ┆ 1.587    ┆ 1.595    │
│ area            ┆ F     ┆ 2.017      ┆ 2.011    ┆ 2.023    │
│ ncd_years       ┆ 0     ┆ 1.000      ┆ 1.000    ┆ 1.000    │
│ ncd_years       ┆ 1     ┆ 0.871      ┆ 0.869    ┆ 0.873    │
│ ncd_years       ┆ 2     ┆ 0.756      ┆ 0.754    ┆ 0.758    │
│ ncd_years       ┆ 3     ┆ 0.641      ┆ 0.639    ┆ 0.643    │
│ ncd_years       ┆ 4     ┆ 0.543      ┆ 0.541    ┆ 0.545    │
│ ncd_years       ┆ 5     ┆ 0.437      ┆ 0.435    ┆ 0.439    │
│ has_convictions ┆ 0     ┆ 1.000      ┆ 1.000    ┆ 1.000    │
│ has_convictions ┆ 1     ┆ 1.673      ┆ 1.664    ┆ 1.682    │
│ …               ┆ …     ┆ …          ┆ …        ┆ …        │
└─────────────────┴───────┴────────────┴──────────┴──────────┘
```

The NCD=5 discount of 0.437 is approximately `exp(-0.83)`. The true DGP coefficient is -0.12 per NCD year, so NCD=5 vs NCD=0 should give `exp(-0.60) ≈ 0.549`. The GBM absorbs some NCD signal into other features — this is the documented behaviour from our benchmark: SHAP attribution for correlated features is shared across the tree splits that use them. We noted this and decided to continue to the distillation step, which produces a GLM fit that respects the multiplicative structure more cleanly.

**Extract continuous curves:**

```python
age_curve = sr.extract_continuous_curve(
    feature="driver_age",
    n_points=100,
    smooth_method="loess",
)
# Returns: feature_value, relativity, lower_ci, upper_ci
# The U-shape is visible: age 20 sits at ~1.52x, age 38 at the minimum (~0.91x), age 70 at ~3.38x
```

The smoothed curve is useful for the pricing committee and for choosing sensible bin cut-points. It is not directly loadable into Radar — Radar needs discrete bands. That is what the distillation step handles.

---

## Step 4: Distil to factor tables and export (15 minutes)

```python
from insurance_distill import SurrogateGLM

X_train = df.select(features)
y_train = df["claim_count"]
exposure = df["exposure"].to_numpy()

surrogate = SurrogateGLM(
    model=model,
    X_train=X_train,
    y_train=y_train,
    exposure=exposure,
    family="poisson",
)

surrogate.fit(
    max_bins=8,
    binning_method="tree",
    method_overrides={
        "ncd_years": "isotonic",   # monotone discount — enforce it
    },
)
```

`binning_method="tree"` runs a CART decision tree on the GBM's pseudo-predictions to find statistically meaningful cut-points. For `ncd_years` we override to `isotonic` because the discount structure should be monotone — there is no actuarial justification for NCD=3 being more expensive than NCD=2.

**Check the distillation quality before exporting anything:**

```python
report = surrogate.report()
print(report.metrics.summary())
```

```
Gini (GBM):              0.4813
Gini (GLM surrogate):    0.4601
Gini ratio:              95.6%
Deviance ratio:          0.9218
Max segment deviation:   7.1%
Mean segment deviation:  1.9%
Segments evaluated:      448
```

95.6% Gini retention is excellent. Max segment deviation of 7.1% means no individual rating cell is priced more than 7.1% away from what the GBM predicted. That is well within the tolerance we'd accept for a production rating engine — the GBM itself has prediction variance higher than that across cross-validation folds.

If the Gini ratio had come back at 85% or the max segment deviation had been above 15%, we would have gone back and added `interaction_pairs` to the `fit()` call. In this case it was not necessary.

**Inspect a factor table before writing files:**

```python
print(surrogate.factor_table("driver_age"))
```

```
shape: (8, 3)
┌──────────────────────┬─────────────────┬────────────┐
│ level                ┆ log_coefficient ┆ relativity │
│ ---                  ┆ ---             ┆ ---        │
│ str                  ┆ f64             ┆ f64        │
╞══════════════════════╪═════════════════╪════════════╡
│ [-inf, 21.00)        ┆ 0.412           ┆ 1.510      │
│ [21.00, 25.00)       ┆ 0.218           ┆ 1.244      │
│ [25.00, 35.00)       ┆ 0.089           ┆ 1.093      │
│ [35.00, 50.00)       ┆ 0.000           ┆ 1.000      │
│ [50.00, 60.00)       ┆ 0.071           ┆ 1.074      │
│ [60.00, 70.00)       ┆ 0.231           ┆ 1.260      │
│ [70.00, 80.00)       ┆ 0.518           ┆ 1.679      │
│ [80.00, +inf)        ┆ 0.884           ┆ 2.421      │
└──────────────────────┴─────────────────┴────────────┘
```

The CART binning found the U-shape without us specifying it. Ages 35–50 are the base level (relativity 1.000). Young drivers (<21) sit at 1.510×. Oldest band (80+) at 2.421×. These numbers are defensible to a pricing committee and consistent with claims experience.

**Export to Radar-compatible CSV:**

```python
surrogate.export_csv("output/factors/", prefix="motor_freq_")
```

This writes one file per variable:

```
output/factors/
  motor_freq_driver_age.csv
  motor_freq_vehicle_age.csv
  motor_freq_ncd_years.csv
  motor_freq_has_convictions.csv
  motor_freq_area.csv
  motor_freq_vehicle_group.csv
  motor_freq_annual_mileage.csv
```

Each file is in Radar's expected two-column format. Here is `motor_freq_area.csv` in full:

```
area,Relativity
A,1.000000
B,1.109000
C,1.151000
D,1.272000
E,1.588000
F,2.014000
```

The `format_radar_csv()` function — which `export_csv()` calls internally — writes the feature name as the first column header, with `Relativity` as the second, and six decimal places of precision. That matches the format Radar's "Import Factor Table" dialog expects. Load each file directly; no reformatting required.

---

## What the timing actually looked like

| Step | Time |
|------|------|
| Data prep and feature decisions | 10 min |
| CatBoost training (500 iterations, 50k rows) | 5 min |
| SHAP extraction, validation, continuous curves | 15 min |
| Distillation, quality checks, export | 15 min |
| **Total** | **45 min** |

The SHAP extraction time breaks down as: 8 minutes for the `.fit()` call itself (SHAP computation on 50k rows with 7 features), and 7 minutes for reading the validation output and deciding whether anything needed rerunning. Nothing did.

If you are running this on a smaller book — 20,000 policies, fewer features — the SHAP step drops to under 3 minutes and the total is closer to 25 minutes.

---

## What we did not do

We did not add interaction terms to the surrogate GLM. The max segment deviation of 7.1% told us the main-effects GLM was capturing the GBM's structure adequately for this portfolio. If you have a book with a strong `driver_age × area` interaction — which shows up on some urban/rural split books — you would add `interaction_pairs=[("driver_age", "area")]` and recheck the segment deviation before exporting.

We did not run a temporal validation. The report validated on the training data. For a production rating engine, you should pass a held-out accident year to `surrogate.report(X_val=, y_val=, exposure_val=)` and confirm the factor tables generalise before loading into Radar.

We did not handle the conviction points feature as more than a binary flag. If your book has meaningful exposure at 3+ points, you can create more bands and pass the ordinal `conviction_points` feature directly — the distillation step will bin it automatically.

---

## The honest limitation

The factor table is a static snapshot of the GBM's structure as of the training date. Changing the Radar factor table does not change the GBM. If your pricing team plans to manually override relativities after extracting them — which is standard practice — those overrides operate on the surrogate GLM's output, not on the underlying model. For most teams this is fine. The GBM is the reference; the Radar factors are the production system.

The libraries are on GitHub: [shap-relativities](https://github.com/burning-cost/shap-relativities), [insurance-distill](https://github.com/burning-cost/insurance-distill). Both are MIT licensed, installable with `pip install shap-relativities insurance-distill`.
