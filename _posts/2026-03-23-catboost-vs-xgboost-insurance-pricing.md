---
layout: post
title: "CatBoost vs XGBoost for Insurance Pricing"
date: 2026-03-23
author: Burning Cost
categories: [gbm, machine-learning, insurance-pricing]
description: "A practical comparison of CatBoost and XGBoost for UK personal lines insurance pricing — categorical handling, Tweedie support, and why we default to CatBoost."
tags: [catboost, xgboost, gbm, insurance-pricing, python, machine-learning]
---

Both CatBoost and XGBoost fit gradient-boosted trees. Both support Tweedie loss. Both run on your laptop and on Databricks. The comparison most people publish treats them as near-equivalent with minor ergonomic differences. We disagree. For insurance pricing specifically, CatBoost is the right default — and the reasons are structural, not cosmetic.

All Burning Cost libraries use CatBoost as the primary GBM. This post explains why, and where XGBoost still has a genuine edge.

```bash
uv add catboost xgboost polars
```

---

## The categorical handling problem is not solved by one-hot encoding

UK motor pricing models routinely carry 30–60 categorical features: vehicle make, model group, payment frequency, occupation code, area code, broker channel. In a mid-size insurer's training set you might have 400+ vehicle makes, 2,000+ model groups, and 150+ occupation codes.

XGBoost's answer to categorical variables is encoding them before they arrive at the model. In practice this means one-hot encoding (OHE) or target encoding in a preprocessing step. OHE on 400 vehicle makes produces 400 binary columns. The feature matrix gets wide, sparse, and expensive. Worse, one-hot encoding breaks the tree-splitting logic: it is now impossible to learn a natural grouping like "prestige German makes vs. everything else" in a single split — you need a separate split on each indicator column, the interaction is fragmented across depth levels.

Target encoding works better but requires careful out-of-fold discipline to avoid leakage, adds a preprocessing step that must be serialised alongside the model, and still discards the ordered structure of nominal categories.

CatBoost handles categoricals natively. You pass a `cat_features` list; the model handles the rest internally. The critical mechanism is **ordered target statistics**: for each categorical level, the expected target value is estimated using only the observations that precede the current observation in a random permutation of the training data. This prevents the leakage that naive target encoding produces. The result is a per-category statistic that is actually valid on out-of-sample data, computed without you writing a single line of CV-aware preprocessing.

```python
import polars as pl
import catboost as cb

# Load motor data
df = pl.read_parquet("motor_training.parquet")

cat_cols = ["vehicle_make", "vehicle_model_group", "occupation_code",
            "area_code", "payment_freq", "broker_channel"]

# CatBoost — no preprocessing, pass categories directly
X = df.select(pl.all().exclude(["pure_premium", "exposure"])).to_pandas()
y = df["pure_premium"].to_numpy()
w = df["exposure"].to_numpy()

pool = cb.Pool(
    data=X,
    label=y,
    weight=w,
    cat_features=cat_cols,
)

model = cb.CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=200,
)

model.fit(pool)
```

Compare with XGBoost, where you must preprocess first:

```python
import xgboost as xgb
import polars as pl
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

df = pl.read_parquet("motor_training.parquet")

cat_cols = ["vehicle_make", "vehicle_model_group", "occupation_code",
            "area_code", "payment_freq", "broker_channel"]
num_cols = [c for c in df.columns if c not in cat_cols + ["pure_premium", "exposure"]]

# Encode categoricals — XGBoost can use enable_categorical but it's
# still experimental; OrdinalEncoder + enable_categorical is the stable path
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat = enc.fit_transform(df.select(cat_cols).to_pandas())
X_num = df.select(num_cols).to_numpy()
X = np.hstack([X_num, X_cat])

dtrain = xgb.DMatrix(
    data=X,
    label=df["pure_premium"].to_numpy(),
    weight=df["exposure"].to_numpy(),
    feature_names=num_cols + cat_cols,
    enable_categorical=True,
)

params = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.5,
    "max_depth": 6,
    "eta": 0.03,
    "seed": 42,
}

model_xgb = xgb.train(params, dtrain, num_boost_round=2000,
                       verbose_eval=200)
```

XGBoost's `enable_categorical` flag (introduced in v1.7) does support native categoricals, but as of early 2026 it uses a simpler one-versus-rest partitioning strategy rather than CatBoost's ordered statistics. For high-cardinality vehicle and occupation codes, the quality difference is real.

---

## Tweedie and Poisson support

Both frameworks support Tweedie loss correctly. XGBoost uses `objective="reg:tweedie"` with a `tweedie_variance_power` parameter; CatBoost uses `loss_function="Tweedie:variance_power=1.5"`. Both fit compound Poisson-Gamma models for `p` between 1 and 2, and Poisson for `p=1`.

The difference is in what you can tune. CatBoost exposes the variance power as a learnable parameter through `TweedieRegressor` in its sklearn wrapper, which means you can cross-validate over it directly. With XGBoost you grid-search manually.

For a frequency–severity split — which is still the dominant structure in UK personal lines production models — Poisson on frequency and Gamma on severity:

```python
# CatBoost frequency model
freq_model = cb.CatBoostRegressor(
    loss_function="Poisson",
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=0,
)
freq_model.fit(cb.Pool(X, label=claim_counts, weight=exposure, cat_features=cat_cols))

# CatBoost severity model
sev_model = cb.CatBoostRegressor(
    loss_function="Gamma",     # CatBoost supports Gamma loss natively
    iterations=1000,
    learning_rate=0.03,
    depth=5,
    random_seed=42,
    verbose=0,
)
```

Both frameworks handle exposure offsets via the `weight` parameter, which is the right approach for rate-on-line modelling. Neither exposes a true offset (as opposed to a weight), so if you need a log-offset for a proper Poisson GLM you need to handle it in the target transformation. This is a shared limitation, not specific to either.

---

## Monotonic constraints

Actuarial sign-off on a GBM almost always requires monotonic constraints: older vehicles must not produce lower risk scores than newer ones, higher NCB bands must not produce higher risk, and so on. Without these constraints a GBM will exploit noise in ways that are commercially and regulatorily indefensible.

Both CatBoost and XGBoost support monotonic constraints. The syntax differs but the capability is equivalent. CatBoost uses `monotone_constraints` as a dict; XGBoost uses the same parameter name with a tuple of -1/0/1 values matching column order.

```python
# CatBoost — dict by feature name, clean
model = cb.CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=2000,
    monotone_constraints={"vehicle_age": 1, "ncb_years": -1, "driver_age_young": -1},
    verbose=0,
)

# XGBoost — tuple matching column order, error-prone if columns shift
params["monotone_constraints"] = (0, 0, 1, -1, -1, 0, 0, 0)  # fragile
```

CatBoost's dict-based syntax is less error-prone when your feature set changes — a real concern in production pricing systems where features get added between model rebuilds.

---

## Ordered boosting and overfitting on insurance data

Insurance training sets are structurally prone to overfitting. You have exposures from 50k to 5 million vehicle-years, but rare vehicle–area–driver combinations have single-digit observations. A GBM that has not been regularised will over-index on these thin cells.

CatBoost's ordered boosting — where the gradient is computed on a separate permuted subset of the data, not the same data used to build the tree — acts as an implicit regulariser against this. The academic basis (Prokhorenkova et al., NeurIPS 2018) showed reduced overfitting particularly on smaller datasets with high cardinality. Insurance data, where your effective sample size for the long tail of VHC codes is small, fits that profile.

In our experience running out-of-sample Gini comparisons on UK motor portfolios, CatBoost with default settings outperforms XGBoost with default settings. You can close the gap by carefully tuning XGBoost regularisation parameters, but the burden of proof is reversed: CatBoost defaults transfer better to new portfolios.

---

## Symmetric trees and inference speed

CatBoost builds symmetric (oblivious) trees: the same feature and threshold is used for every split at a given depth level. This produces shallower, more regular tree structures compared to XGBoost's asymmetric trees.

Symmetric trees are faster at inference. A symmetric tree of depth `d` requires exactly `2^d` leaf evaluations per observation with simple bitwise indexing. XGBoost's asymmetric trees require conditional branching that is harder to vectorise. For a production pricing engine applying a model to 2 million vehicle records in batch, the inference speed difference is non-trivial: roughly 3–5x faster in our benchmarks on aarch64.

If you are scoring in real-time (for PCW rates), this matters. If you are re-rating a book overnight, it matters less.

---

## Feature importance and SHAP

This is where XGBoost has a genuine edge. The `shap` library (Lundberg et al.) was originally developed with XGBoost in mind, and the TreeExplainer implementation is tightly coupled to XGBoost's internal data structures. SHAP values from `xgboost` objects compute correctly and quickly via the C++ fast path.

CatBoost has built-in SHAP support via `model.get_feature_importance(type="ShapValues")`, which is accurate and consistent. But if your team is using `shap.TreeExplainer` directly — as many pricing teams do because it integrates with existing SHAP plots and dashboards — you may encounter edge cases with CatBoost's symmetric tree structure that require the slower Python path.

For interactive model explanation in `shap`, XGBoost is currently smoother. For SHAP values computed in batch and exported to a dashboard, CatBoost works fine.

CatBoost's own feature importance methods (PredictionValuesChange, LossFunctionChange) are reliable and faster than SHAP for routine feature selection. We use these as the first pass, and reserve SHAP for regulatory and explainability reports.

---

## Polars compatibility

As of early 2026, neither CatBoost nor XGBoost accepts Polars DataFrames directly — both require conversion to pandas or numpy. The conversion is one line (`df.to_pandas()` or `df.to_numpy()`), but it adds overhead on large datasets and complicates lazy evaluation pipelines.

XGBoost 2.0 added a `QuantileDMatrix` that supports Arrow-backed arrays, which means you can pass a Polars DataFrame via the Arrow IPC path with zero copy:

```python
import pyarrow as pa

arrow_table = df.to_arrow()
dtrain = xgb.QuantileDMatrix(arrow_table, label=y, weight=w)
```

CatBoost does not yet support this path. For pipelines that are otherwise end-to-end Polars, this is a genuine XGBoost advantage — no redundant copy of a 10GB training dataset into pandas.

---

## Our recommendation

Use CatBoost. The native categorical handling alone justifies it for UK personal lines pricing, where high-cardinality vehicle and occupation codes are unavoidable. The ordered boosting regularisation is a meaningful free lunch on the kinds of thin-tailed data distributions insurance generates. The symmetric trees give you faster inference in production. The dict-based monotonic constraint syntax is more maintainable.

All Burning Cost libraries — from [insurance-severity](https://github.com/Burning-Cost/insurance-severity) to [insurance-fairness](/insurance-fairness/) — default to CatBoost for GBM fits. We made this decision after benchmarking both frameworks on public motor and home datasets, and it has held up across 18 months of library development.

Use XGBoost when:
- Your team has existing SHAP pipelines built on TreeExplainer and you cannot absorb the migration cost
- You are running an end-to-end Polars pipeline and the zero-copy Arrow path matters
- You need to reference Kaggle solutions or community examples (XGBoost's community is still larger)
- You are already running an XGBoost model in production and the current performance is acceptable — do not rebuild for the sake of it

The Kaggle community default has historically been XGBoost, then LightGBM. Neither was designed with actuarial use cases as a primary constraint. CatBoost was, in effect, designed for exactly the problems UK pricing analysts face: high cardinality categoricals, exposure weighting, and a need for well-calibrated outputs from relatively small datasets.

That is why we default to it, and why you probably should too.

- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — the CatBoost-based distributional regression library that models per-risk dispersion alongside the mean
- [EBM, ANAM, or PIN: Choosing an Interpretable Architecture for UK Insurance Pricing](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — when GBM accuracy is not enough and you need a model that prints factor tables directly
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — distribution-free prediction intervals that wrap any sklearn-compatible GBM
