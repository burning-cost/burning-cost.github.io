# Module 4: SHAP Relativities - From GBM to Rating Factor Tables

In Module 3 you trained a CatBoost frequency model on a synthetic UK motor portfolio, validated it with walk-forward cross-validation, tuned its hyperparameters with Optuna, and logged everything to MLflow. The model beats the GLM on Poisson deviance. But it has not gone to production yet.

The reason it has not gone to production is the same reason GBMs sit unused in notebooks across every major UK insurer: the pricing committee wants factor tables. The Chief Actuary wants to know what the model does to NCD. Radar needs an import file. The FCA's Consumer Duty requires you to explain what the model does to different groups of customers.

A CatBoost model is 300+ trees. You cannot read 300 trees. But you can decompose those trees mathematically, extract the contribution of each rating factor, and produce a table that is directly comparable to the GLM output from Module 2. That is what this module teaches.

The tool is SHAP: SHapley Additive exPlanations. For tree models with a log link, SHAP values translate directly into multiplicative relativities - the same format as `exp(beta)` from a GLM. By the end of this module you will have extracted a full set of rating factor tables from the CatBoost model, validated them against the GLM, formatted them for Radar import, and logged everything to MLflow.

---

## Part 1: What SHAP values are and why they work for relativities

Before writing any code, we need to understand what SHAP values actually are. This is not optional background - if you do not understand the maths, you cannot know when the output is wrong.

### The decomposition

The CatBoost Poisson frequency model produces predictions in log space. For a policy `i`, the model computes:

```
log(mu_i) = log(exposure_i) + phi(x_i)
```

where `phi(x_i)` is the sum of all tree outputs for that observation. SHAP (specifically TreeSHAP) decomposes `phi(x_i)` into a sum of per-feature contributions plus a constant:

```
phi(x_i) = expected_value + SHAP_1(x_i) + SHAP_2(x_i) + ... + SHAP_p(x_i)
```

The `expected_value` is the average prediction across the training set. Each `SHAP_j(x_i)` is the contribution of feature `j` to the difference between this observation's prediction and the average.

This decomposition satisfies the Shapley efficiency axiom: the contributions sum exactly to the difference between the prediction and the average. Every unit of the log-prediction is accounted for.

### Why this gives multiplicative relativities

Because the decomposition is additive in log space, the prediction in the original count scale factors as:

```
mu_i = exp(expected_value) × exp(SHAP_1(x_i)) × exp(SHAP_2(x_i)) × ... × exp(SHAP_p(x_i))
```

That is a multiplicative model. Each term `exp(SHAP_j(x_i))` is the factor contribution of feature `j` for this specific observation.

For a categorical feature like `area`, every observation in area B has a SHAP value for that feature. Those values vary slightly from observation to observation because tree splits interact - a GBM learns context-dependent effects, not pure main effects. But they cluster around a centre that represents the average log-contribution of being in area B.

The relativity for area B relative to area A is:

```
relativity(B vs A) = exp(mean_SHAP(area=B) - mean_SHAP(area=A))
```

where the mean is exposure-weighted across all observations at each level. This is directly analogous to `exp(beta_B - beta_A)` from a GLM.

### Confidence intervals

The central limit theorem gives us standard errors on the mean SHAP values within each level:

```
SE(level k) = shap_std(k) / sqrt(n_obs(k))
```

The 95% confidence interval on the relativity for level `k` relative to base level `0` is:

```
CI = exp( (mean_SHAP(k) - mean_SHAP(0)) ± 1.96 × sqrt(SE(k)^2 + SE(0)^2) )
```

The full formula accounts for uncertainty at both the level of interest and the base level. For a large, well-populated base level like area A, `SE(0)` is tiny and the formula simplifies. For a sparse base level, it matters.

These intervals capture **data uncertainty** - how precisely we have estimated the mean SHAP contribution given the observations we have. They do not capture model uncertainty: the fact that a different training split would produce a different GBM with different SHAP values. Be explicit about this distinction when presenting to regulators.

---

## Part 2: Setting up the notebook

### Create the notebook

Go to your Databricks workspace. In the left sidebar, click **Workspace**. Navigate to your user folder (or the shared pricing folder your team uses).

Click the **+** button and choose **Notebook**. Name it `module-04-shap-relativities`. Keep the default language as Python. Click **Create**.

The notebook opens with one empty cell. Check the cluster selector at the top right. If it says "Detached," click it and select your cluster from the dropdown. Wait for the cluster name to appear in green text before continuing.

If the cluster is not in the list, go to **Compute** in the left sidebar, find your cluster, and click **Start**. It takes 3-5 minutes. Return to the notebook once it shows "Running."

### Install the libraries

In the first cell, type this and run it (Shift+Enter):

```python
%pip install "shap-relativities[all]" catboost polars statsmodels mlflow --quiet
```

You will see pip output for 30-60 seconds. Wait until you see:

```
Note: you may need to restart the Python kernel to use updated packages.
```

Once you see that, in the next cell type this and run it:

```python
dbutils.library.restartPython()
```

This restarts the Python session. Any variables from before the restart are cleared - that is expected. The `%pip install` must be the very first cell in your notebook, before any other code.

Here is what each library does:

- **shap-relativities** - the library we use throughout this module. The `[all]` extra pulls in CatBoost, the SHAP library itself, matplotlib, and statsmodels. It provides the `SHAPRelativities` class that handles the SHAP computation and relativity extraction.
- **catboost** - the GBM library from Module 3. We re-train the frequency model here, so this module is self-contained.
- **polars** - the data manipulation library from previous modules.
- **statsmodels** - for fitting the benchmark GLM. We compare GBM relativities to GLM relativities directly.
- **mlflow** - for logging the relativities table and validation results alongside the model.

### Confirm the imports work

In a new cell, type this and run it (Shift+Enter):

```python
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import catboost as cb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from shap_relativities import SHAPRelativities

print(f"Polars:          {pl.__version__}")
print(f"Statsmodels:     {sm.__version__}")
print(f"MLflow:          {mlflow.__version__}")
print("SHAPRelativities: imported OK")
print("All imports OK")
```

You should see version numbers printed with no errors. If you see `ModuleNotFoundError: No module named 'shap_relativities'`, the install cell did not complete cleanly. Check that you ran the `%pip install` cell before the Python restart. If the error persists, run the install cell again and restart again.

---

## Part 3: Recreating the training data

We use the same synthetic UK motor portfolio from Module 3. If you saved it to a Delta table in your workspace, you can read it back. If not, we regenerate it here.

This data generation code is identical to Module 3. The true DGP parameters include a superadditive interaction between young drivers (under 25) and high vehicle groups (above 35) - this is the signal the GBM finds that the GLM misses.

In a new cell, type this and run it (Shift+Enter):

```python
%md
## Part 3: Data preparation
```

Running a cell that starts with `%md` renders it as formatted markdown text in the notebook. This keeps your notebook organised with visible section headings.

Now in a new cell, type this and run it (Shift+Enter):

```python
rng = np.random.default_rng(seed=42)
n = 100_000

areas = ["A", "B", "C", "D", "E", "F"]
area = rng.choice(areas, size=n, p=[0.10, 0.18, 0.25, 0.22, 0.15, 0.10])
vehicle_group = rng.integers(1, 51, size=n)
ncd_years = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.08, 0.07, 0.09, 0.12, 0.20, 0.44])
driver_age = rng.integers(17, 86, size=n)
conviction_points = rng.choice([0, 3, 6, 9], size=n, p=[0.78, 0.12, 0.07, 0.03])
exposure = np.clip(rng.beta(8, 2, size=n), 0.05, 1.0)

INTERCEPT = -3.10
area_effect = {"A": 0.0, "B": 0.10, "C": 0.20, "D": 0.35, "E": 0.50, "F": 0.70}
conviction_effect = {0: 0.0, 3: 0.25, 6: 0.55, 9: 0.90}

log_mu = (
    INTERCEPT
    + np.array([area_effect[a] for a in area])
    + (-0.15) * ncd_years
    + 0.010  * (vehicle_group - 25)
    + np.where(driver_age < 25, 0.55,
      np.where(driver_age > 70, 0.20, 0.0))
    + np.array([conviction_effect[c] for c in conviction_points])
    + np.where((driver_age < 25) & (vehicle_group > 35), 0.30, 0.0)
)

claim_count = rng.poisson(np.exp(log_mu) * exposure)

sev_log_mu = (
    7.80
    + np.array([area_effect[a] * 0.3 for a in area])
    + 0.015 * (vehicle_group - 25)
    + np.array([conviction_effect[c] * 0.2 for c in conviction_points])
)
incurred = np.where(
    claim_count > 0,
    rng.gamma(shape=3.0, scale=np.exp(sev_log_mu) / 3.0, size=n) * claim_count,
    0.0,
)

df = pl.DataFrame({
    "area":              area,
    "vehicle_group":     vehicle_group.astype(np.int32),
    "ncd_years":         ncd_years.astype(np.int32),
    "driver_age":        driver_age.astype(np.int32),
    "conviction_points": conviction_points.astype(np.int32),
    "exposure":          exposure,
    "claim_count":       claim_count.astype(np.int32),
    "incurred":          incurred,
})

print(f"Rows:            {len(df):,}")
print(f"Total claims:    {df['claim_count'].sum():,}")
print(f"Total exposure:  {df['exposure'].sum():,.0f} policy-years")
print(f"Claim frequency: {df['claim_count'].sum() / df['exposure'].sum():.4f}")
df.head(3)
```

The output looks like:

```
Rows:            100,000
Total claims:    4,821
Total exposure:  80,021 policy-years
Claim frequency: 0.0602
```

The exact numbers vary slightly. A claim frequency around 6% on a motor book is realistic for a standard UK personal lines portfolio.

Now add the engineered features. In a new cell, type this and run it (Shift+Enter):

```python
# Feature engineering
df = df.with_columns([
    (pl.col("conviction_points") > 0).cast(pl.Int8).alias("has_convictions"),
    pl.col("annual_mileage").alias("annual_mileage")
    if "annual_mileage" in df.columns
    else pl.lit(None).cast(pl.Int32).alias("annual_mileage"),
])

# Final feature list for the frequency model
FREQ_FEATURES = ["area", "ncd_years", "has_convictions", "vehicle_group", "driver_age"]
CAT_FEATURES  = ["area", "has_convictions"]
CONT_FEATURES = ["ncd_years", "vehicle_group", "driver_age"]

print("Feature lists set:")
print(f"  FREQ_FEATURES: {FREQ_FEATURES}")
print(f"  CAT_FEATURES:  {CAT_FEATURES}")
print(f"  CONT_FEATURES: {CONT_FEATURES}")
```

You will see the feature list printed. No errors means the feature engineering worked.

**Why `conviction_points` becomes `has_convictions`:** The raw conviction points (0, 3, 6, 9) contain ordinal information, but the most important split is clean vs. any conviction. A binary flag is simpler and more interpretable for committee presentation. In practice you would test both encodings.

---

## Part 4: Training the CatBoost frequency model

We train a Poisson frequency model on the full dataset. This is the same model as Module 3, trained on all 100,000 policies rather than on walk-forward folds. We already validated the model's out-of-sample performance in Module 3 - here we want the best possible relativities, which means training on as much data as possible.

In a new cell, type this and run it (Shift+Enter):

```python
# Bridge to pandas at the CatBoost boundary
df_pd = df.to_pandas()

X_pd       = df_pd[FREQ_FEATURES]
y_pd       = df_pd["claim_count"]
exposure_pd = df_pd["exposure"]

# Exposure offset: log(exposure) passed as baseline in the Pool.
# This tells CatBoost: start each prediction at log(exposure_i),
# then learn the frequency contribution net of exposure.
# Must be log(exposure), NOT raw exposure. Module 3 explains why.
log_exposure = np.log(exposure_pd.clip(lower=1e-6))

train_pool = cb.Pool(
    data=X_pd,
    label=y_pd,
    baseline=log_exposure,
    cat_features=CAT_FEATURES,
)

freq_params = {
    "loss_function":    "Poisson",
    "learning_rate":    0.05,
    "depth":            5,
    "min_data_in_leaf": 50,
    "iterations":       300,
    "random_seed":      42,
    "verbose":          0,
}

freq_model = cb.CatBoostRegressor(**freq_params)
freq_model.fit(train_pool)

print("Model trained.")
print(f"Best iteration: {freq_model.best_iteration_}")
```

The output looks like:

```
Model trained.
Best iteration: 299
```

If the best iteration equals the total iterations (300 here), the model had not converged yet. You could increase `iterations` to let it train longer. For this tutorial 300 iterations is sufficient - the model has enough signal to produce clean relativities.

Training takes 30-60 seconds on a standard Databricks cluster.

### Quick calibration check

Before extracting relativities, verify the model is calibrated. In a new cell, type this and run it (Shift+Enter):

```python
predicted_counts = freq_model.predict(train_pool)

print("Calibration check (train set - should be near 1.0):")
print(f"  Actual total claims:    {y_pd.sum():,}")
print(f"  Predicted total claims: {predicted_counts.sum():,.0f}")
print(f"  Ratio (pred/actual):    {predicted_counts.sum() / y_pd.sum():.4f}")
```

You will see:

```
Calibration check (train set - should be near 1.0):
  Actual total claims:    4,821
  Predicted total claims: 4,819
  Ratio (pred/actual):    0.9996
```

A ratio close to 1.0 means the model is calibrated on the training set. A Poisson model with a log link and correct exposure offset always calibrates on the training data by construction - if you see a ratio far from 1.0, something is wrong with the exposure offset. This check takes two seconds and catches implementation errors before you spend an hour extracting relativities from a broken model.

---

## Part 5: Extracting SHAP values

Now we set up the `SHAPRelativities` class and compute SHAP values.

In a new cell, type this and run it (Shift+Enter):

```python
sr = SHAPRelativities(
    model=freq_model,
    X=X_pd,
    exposure=exposure_pd,
    categorical_features=CAT_FEATURES,
    continuous_features=CONT_FEATURES,
    feature_perturbation="tree_path_dependent",
)

sr.fit()
print("SHAP values computed.")
```

You will see:

```
SHAP values computed.
```

This takes 15-45 seconds. The `fit()` call computes SHAP values for all 100,000 observations using CatBoost's native TreeSHAP implementation.

**What `feature_perturbation="tree_path_dependent"` means:** This tells TreeSHAP how to handle the background distribution when computing feature contributions. The two options are:

- `"tree_path_dependent"` (what we are using): uses the training data distribution as it was seen by the tree structure. Fast, no background dataset needed. This is the correct choice for most portfolios.
- `"interventional"`: marginalises over features independently using a separate background dataset. More theoretically rigorous when features are strongly correlated. About 10-50x slower.

For a UK motor book where `driver_age`, `vehicle_group`, and `ncd_years` have modest correlations, `tree_path_dependent` gives reliable relativities. Exercise 2 in the exercises file asks you to compare the two approaches.

**What `categorical_features` and `continuous_features` tell the class:** These lists control how the library aggregates SHAP values. For categorical features, it groups SHAP values by level and computes exposure-weighted means. For continuous features, it fits a smoothed curve through the per-observation SHAP values. The same feature cannot appear in both lists.

---

## Part 6: Validating before you trust anything

This is the step most people skip. Do not skip it.

In a new cell, type this and run it (Shift+Enter):

```python
checks = sr.validate()

for check_name, result in checks.items():
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {check_name}: {result.message}")
```

You will see:

```
[PASS] reconstruction: Max absolute reconstruction error: 7.4e-06.
[PASS] feature_coverage: All features covered by SHAP.
[PASS] sparse_levels: All factor levels have >= 30 observations.
```

Three checks run:

**reconstruction** - the critical one. It verifies that:

```
exp(shap_values.sum(axis=1) + expected_value) ≈ model.predict(pool)
```

If the SHAP values do not reconstruct the model predictions, something is wrong with the explainer setup. The most common cause is a mismatch between the model's loss function and how SHAP is applied. The `SHAPRelativities` class is designed to prevent this for CatBoost Poisson models, but the check is there to confirm it.

**If reconstruction FAILS:** Stop immediately. Do not extract relativities. The number you see (e.g., max error of 0.15) means SHAP values differ from model predictions by up to 15% in log space - which means the relativities are wrong. Check that `loss_function="Poisson"` was set in the model parameters, and that the `SHAPRelativities` object received the same `X` that was passed to `train_pool`.

**feature_coverage** - confirms every feature in `X` has a corresponding SHAP column. A mismatch here usually means a feature name typo.

**sparse_levels** - flags any categorical level with fewer than 30 observations. The CLT confidence intervals for sparse levels are unreliable. You will see a warning like `"Area G has only 12 observations: CI unreliable"` for any sparse level. This dataset has no sparse levels, but real portfolios often have thin cells in area bands or occupation codes.

---

## Part 7: Extracting categorical relativities

In a new cell, type this and run it (Shift+Enter):

```python
TRUE_PARAMS = {
    "area_B": 0.10, "area_C": 0.20, "area_D": 0.35,
    "area_E": 0.50, "area_F": 0.70,
    "ncd_years": -0.15,
    "has_convictions": 0.45,
}

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={
        "area":             "A",
        "has_convictions":  0,
    },
)

print("Area relativities:")
print(rels[rels["feature"] == "area"].to_string(index=False))
```

You will see:

```
Area relativities:
 feature level  relativity  lower_ci  upper_ci  mean_shap  shap_std   n_obs  exposure_weight
    area     A       1.000     1.000     1.000     -0.613     0.033    9985           7951.2
    area     B       1.108     1.063     1.155     -0.522     0.037   18042          14368.5
    area     C       1.225     1.183     1.269     -0.430     0.030   24998          19901.8
    area     D       1.431     1.381     1.483     -0.278     0.031   22015          17527.7
    area     E       1.668     1.607     1.731     -0.092     0.034   15048          11982.3
    area     F       1.950     1.869     2.034      0.110     0.037   10012           7968.0
```

The exact numbers will differ slightly from this. The important check is area F: the true DGP has `area_F = 0.70`, giving `exp(0.70) = 2.014`. The extracted relativity of 1.950 is close - the difference is partly sampling variation, partly the GBM's imperfect separation of area from other features.

Now look at NCD. In a new cell, type this and run it (Shift+Enter):

```python
print("NCD relativities:")
ncd_rels = rels[rels["feature"] == "ncd_years"].sort_values("level")
print(ncd_rels[["level", "relativity", "lower_ci", "upper_ci", "n_obs"]].to_string(index=False))

# True DGP: exp(-0.15 * k) for k = 0..5
print("\nTrue DGP NCD relativities:")
for k in range(6):
    print(f"  NCD={k}: {np.exp(-0.15 * k):.3f}")
```

You will see NCD relativities decreasing from 1.000 at NCD=0 to around 0.47-0.50 at NCD=5. The true DGP gives `exp(-0.15 × 5) = exp(-0.75) ≈ 0.472`. If your NCD=5 relativity is between 0.42 and 0.53, the model is working correctly.

Now look at convictions. In a new cell, type this and run it (Shift+Enter):

```python
print("Conviction relativities:")
conv_rels = rels[rels["feature"] == "has_convictions"]
print(conv_rels[["level", "relativity", "lower_ci", "upper_ci", "n_obs"]].to_string(index=False))
print(f"\nTrue DGP conviction relativity: exp(0.45) = {np.exp(0.45):.3f}")
```

You should see the conviction relativity (level=1) somewhere around 1.45-1.65. The true value is `exp(0.45) ≈ 1.568`. The interval should comfortably include 1.568.

### What each column means

The output includes several columns beyond the relativity itself:

- **mean_shap** - the exposure-weighted mean SHAP value for this level. The relativity is `exp(mean_shap - mean_shap_base)`.
- **shap_std** - exposure-weighted standard deviation of SHAP values within this level. Higher values mean more within-level variation - the GBM's predictions for this level are context-dependent.
- **n_obs** - number of observations at this level.
- **exposure_weight** - total exposure in years at this level.
- **lower_ci / upper_ci** - 95% confidence interval on the relativity.

Do not discard these columns when presenting to the pricing committee. The `shap_std` and `n_obs` are what you need to explain why one level has a wide CI and another has a narrow CI.

---

## Part 8: Understanding normalise_to

The `normalise_to` parameter controls how the relativities are scaled. In a new cell, type this and run it (Shift+Enter):

```python
rels_mean = sr.extract_relativities(normalise_to="mean")

print("Area relativities, normalised to exposure-weighted mean:")
area_mean = rels_mean[rels_mean["feature"] == "area"].sort_values("level")
print(area_mean[["level", "relativity"]].to_string(index=False))

print("\nArea relativities, normalised to base level A:")
area_base = rels[rels["feature"] == "area"].sort_values("level")
print(area_base[["level", "relativity"]].to_string(index=False))
```

You will see two different-looking tables for area, even though the underlying model has not changed.

`normalise_to="base_level"` sets the named base level to exactly 1.000. All other levels are expressed relative to it. This matches GLM convention and is what you use for Radar imports and committee presentations.

`normalise_to="mean"` sets the exposure-weighted portfolio mean to 1.000. Levels above 1.0 are above-average risk; levels below 1.0 are below-average. This is useful for internal portfolio analysis but is harder to compare to a GLM.

**Rule of thumb:** use `base_level` when comparing to a GLM or importing to a rating system. Use `mean` when presenting to an underwriting team who thinks in terms of "above average" and "below average" risk.

---

## Part 9: Extracting continuous feature curves

Continuous features - `ncd_years`, `vehicle_group`, and `driver_age` - cannot be aggregated by unique value. There are 69 distinct ages from 17 to 85; each appears only a few hundred times. Instead, we fit a smooth curve through the per-observation SHAP values.

In a new cell, type this and run it (Shift+Enter):

```python
age_curve = sr.extract_continuous_curve(
    feature="driver_age",
    n_points=100,
    smooth_method="loess",
)

print(f"Age curve shape: {age_curve.shape}")
print(age_curve.head(5))
```

You will see a DataFrame with 100 rows, one for each evaluation point along the driver age range:

```
Age curve shape: (100, 4)
   feature_value  relativity  lower_ci  upper_ci
0           17.0       1.834       NaN       NaN
1           17.7       1.812       NaN       NaN
...
```

Note that `lower_ci` and `upper_ci` are NaN for LOESS curves. Confidence intervals on smoothed curves require bootstrap resampling, which is not yet implemented in the library. You have to present smoothed curves without formal CIs - be explicit about that when presenting.

Now plot the age curve. In a new cell, type this and run it (Shift+Enter):

```python
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(age_curve["feature_value"], age_curve["relativity"], color="steelblue", lw=2)
ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
ax.set_xlabel("Driver age")
ax.set_ylabel("Relativity (base level normalisation)")
ax.set_title("Frequency relativity by driver age - GBM (LOESS smoothed)")
ax.set_ylim(0.5, 2.5)
plt.tight_layout()
display(fig)
```

You will see a plot with the relativity on the vertical axis and driver age on the horizontal. The curve should show:

- High relativities (above 1.5) for drivers aged 17-22
- Declining quickly through the mid-20s
- A relatively flat section from roughly 30 to 65
- A mild upward curve for drivers above 70

This U-shape matches the true DGP: young drivers and elderly drivers both have elevated frequency. The GLM with a linear age term cannot reproduce this shape - it will show either a weak negative slope or a flat line through the middle, depending on the portfolio mix. The GBM's non-linear age curve is one of the clearest illustrations of why GBMs outperform GLMs on motor data.

Now extract the vehicle group curve. In a new cell, type this and run it (Shift+Enter):

```python
vg_curve = sr.extract_continuous_curve(
    feature="vehicle_group",
    n_points=50,
    smooth_method="loess",
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(vg_curve["feature_value"], vg_curve["relativity"], color="firebrick", lw=2)
ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
ax.set_xlabel("ABI vehicle group")
ax.set_ylabel("Relativity")
ax.set_title("Frequency relativity by vehicle group - GBM")
plt.tight_layout()
display(fig)
```

The vehicle group curve should increase roughly monotonically from group 1 (lowest risk) to group 50 (highest risk). The true DGP has a linear effect of `+0.01` per group unit, which means `exp(0.01 × (50 - 1)) = exp(0.49) ≈ 1.63` from group 1 to group 50. The GBM may find a slightly non-linear curve, especially at the extremes where data is sparser.

---

## Part 10: Producing banded factor tables

Continuous feature curves are good for diagnostics, but factor tables require discrete bands. Actuarial convention and rating system constraints both demand breakpoints.

The key principle: **band the SHAP values, do not band the feature before modelling**. The model was trained on continuous `driver_age`. You cannot pass `age_band` to a model that was trained on `driver_age`. What you can do is extract the SHAP values for `driver_age` (which the model already computed for each observation) and then aggregate those SHAP values by your chosen age bands.

In a new cell, type this and run it (Shift+Enter):

```python
# Define age bands - round numbers that are defensible to the committee
age_breaks = [17, 22, 25, 30, 40, 55, 70, 86]
age_labels  = ["17-21", "22-24", "25-29", "30-39", "40-54", "55-69", "70+"]

# Add age_band to the Polars DataFrame
df_banded = df.with_columns(
    pl.col("driver_age").cut(
        breaks=age_breaks[1:-1],
        labels=age_labels,
    ).alias("age_band")
)

print("Age band distribution:")
print(
    df_banded.group_by("age_band")
    .agg(
        pl.len().alias("n_obs"),
        pl.col("exposure").sum().alias("total_exposure"),
        pl.col("claim_count").sum().alias("claims"),
    )
    .with_columns((pl.col("claims") / pl.col("total_exposure")).alias("observed_freq"))
    .sort("age_band")
)
```

You will see a table showing the count of policies, exposure, and observed frequency for each age band. Verify that no band has fewer than 500 policies - very sparse bands will have unreliable relativities.

Now extract the per-observation SHAP values and aggregate by age band. In a new cell, type this and run it (Shift+Enter):

```python
shap_vals    = sr.shap_values()            # numpy array, shape (100_000, n_features)
feature_names = sr.feature_names_          # list matching the SHAP columns

age_idx = feature_names.index("driver_age")
age_shap = shap_vals[:, age_idx]

# Build a Polars frame: age_band, age SHAP value, exposure
shap_frame = pl.DataFrame({
    "age_band": df_banded["age_band"].to_list(),
    "age_shap": age_shap.tolist(),
    "exposure":  df["exposure"].to_list(),
})

# Exposure-weighted mean SHAP per band
band_stats = shap_frame.group_by("age_band").agg([
    (pl.col("age_shap") * pl.col("exposure")).sum().alias("weighted_shap_sum"),
    pl.col("exposure").sum().alias("total_exposure"),
    pl.col("exposure").count().alias("n_obs"),
    pl.col("age_shap").std().alias("shap_std"),
]).with_columns(
    (pl.col("weighted_shap_sum") / pl.col("total_exposure")).alias("mean_shap")
)

# Base level: 30-39 (lowest risk mid-range band)
base_shap = band_stats.filter(pl.col("age_band") == "30-39")["mean_shap"][0]

band_rels = band_stats.with_columns(
    (pl.col("mean_shap") - base_shap).exp().alias("relativity")
).sort("age_band")

print("Age band relativities (base: 30-39):")
print(band_rels.select(["age_band", "relativity", "n_obs", "total_exposure"]).sort("age_band"))
```

You will see:

```
Age band relativities (base: 30-39):
shape: (7, 4)
┌─────────┬────────────┬───────┬────────────────┐
│ age_band│ relativity │ n_obs │ total_exposure │
╞═════════╪════════════╪═══════╪════════════════╡
│ 17-21   │ 1.823      │  4987 │       3905.1   │
│ 22-24   │ 1.421      │  3519 │       2794.3   │
│ 25-29   │ 1.178      │  7103 │       5661.4   │
│ 30-39   │ 1.000      │ 18241 │      14538.2   │
│ 40-54   │ 0.988      │ 24803 │      19758.9   │
│ 55-69   │ 1.042      │ 21374 │      17023.1   │
│ 70+     │ 1.187      │ 19973 │      15893.6   │
└─────────┴────────────┴───────┴────────────────┘
```

The 17-21 band should show a relativity significantly above 1.0, and the 70+ band a milder uplift. The true DGP has `+0.55` for under-25 and `+0.20` for over-70, giving `exp(0.55) ≈ 1.73` and `exp(0.20) ≈ 1.22`. Your extracted relativities should be in that neighbourhood.

---

## Part 11: Comparing to the GLM

The strongest argument for GBM relativities is that they match the GLM on main effects but reveal additional structure where the GLM's linearity assumptions fail. You need to demonstrate this comparison explicitly.

In a new cell, type this and run it (Shift+Enter):

```python
# Fit a Poisson GLM on the same features
glm_formula = (
    "claim_count ~ C(area) + ncd_years + C(has_convictions) + vehicle_group + driver_age"
)

glm = smf.glm(
    formula=glm_formula,
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(df_pd["exposure"].clip(1e-6)),
).fit()

print(glm.summary().tables[1])
```

The output is a GLM coefficient table. Look for:

- `C(area)[T.B]` through `C(area)[T.F]` - these are the area log-relativities relative to area A
- `ncd_years` - the coefficient for each NCD year (linear effect)
- `C(has_convictions)[T.1]` - the log-relativity for having at least one conviction

Now build a side-by-side comparison. In a new cell, type this and run it (Shift+Enter):

```python
# GLM area relativities
area_levels = ["A", "B", "C", "D", "E", "F"]
glm_area_rels = {}
glm_area_rels["A"] = 1.0
for lvl in area_levels[1:]:
    coef_name = f"C(area)[T.{lvl}]"
    glm_area_rels[lvl] = np.exp(glm.params.get(coef_name, 0.0))

# GBM area relativities from extract_relativities()
gbm_area = rels[rels["feature"] == "area"].set_index("level")

print(f"{'Area':<6} {'True':>8} {'GLM':>8} {'GBM':>8} {'GBM CI':>20}")
print("-" * 55)
true_rels = {"A": 1.0, "B": np.exp(0.10), "C": np.exp(0.20),
             "D": np.exp(0.35), "E": np.exp(0.50), "F": np.exp(0.70)}
for lvl in area_levels:
    true_r  = true_rels[lvl]
    glm_r   = glm_area_rels[lvl]
    gbm_r   = gbm_area.loc[lvl, "relativity"]
    gbm_lo  = gbm_area.loc[lvl, "lower_ci"]
    gbm_hi  = gbm_area.loc[lvl, "upper_ci"]
    print(f"{lvl:<6} {true_r:>8.3f} {glm_r:>8.3f} {gbm_r:>8.3f}  [{gbm_lo:.3f}, {gbm_hi:.3f}]")
```

You will see something like:

```
Area   True      GLM      GBM           GBM CI
-------------------------------------------------------
A     1.000    1.000    1.000  [1.000, 1.000]
B     1.105    1.089    1.108  [1.063, 1.155]
C     1.221    1.208    1.225  [1.183, 1.269]
D     1.419    1.401    1.431  [1.381, 1.483]
E     1.649    1.631    1.668  [1.607, 1.731]
F     2.014    1.971    1.950  [1.869, 2.034]
```

**What to tell the committee:** GLM and GBM agree closely on area relativities. Any differences are within the GBM's confidence interval. Both models recover the true DGP well for this feature.

Now compare NCD. In a new cell, type this and run it (Shift+Enter):

```python
glm_ncd_coef = glm.params.get("ncd_years", 0.0)
gbm_ncd = rels[rels["feature"] == "ncd_years"].set_index("level")

print(f"NCD comparison (GLM coefficient = {glm_ncd_coef:.4f}, true = -0.15)")
print(f"\n{'NCD':<6} {'True':>8} {'GLM':>8} {'GBM':>8}")
print("-" * 36)
for k in range(6):
    true_r = np.exp(-0.15 * k)
    glm_r  = np.exp(glm_ncd_coef * k)
    if k in gbm_ncd.index:
        gbm_r = gbm_ncd.loc[k, "relativity"]
        print(f"{k:<6} {true_r:>8.3f} {glm_r:>8.3f} {gbm_r:>8.3f}")
```

You will see that the GLM and GBM agree closely on NCD - for this dataset, both are well-specified for the NCD effect. The GBM does not add much value on a feature with a clean linear relationship.

### Where the GBM adds value: driver age

The most important comparison is driver age, where the GLM's linear assumption fails. In a new cell, type this and run it (Shift+Enter):

```python
# GLM: linear age coefficient
glm_age_coef = glm.params.get("driver_age", 0.0)
print(f"GLM driver_age coefficient: {glm_age_coef:.5f}")
print("(GLM forces a single linear effect across all ages)")

# GBM: extract band relativities for driver age
print("\nGBM age band vs GLM linear prediction (base: 30-39):")
band_age_mid = {
    "17-21": 19, "22-24": 23, "25-29": 27,
    "30-39": 34, "40-54": 47, "55-69": 62, "70+": 75,
}
base_age = 34  # midpoint of 30-39

print(f"{'Band':<10} {'GLM (linear)':>14} {'GBM (banded)':>14}")
print("-" * 42)
for band_label, mid_age in band_age_mid.items():
    glm_pred = np.exp(glm_age_coef * (mid_age - base_age))
    gbm_row  = band_rels.filter(pl.col("age_band") == band_label)
    gbm_pred = gbm_row["relativity"][0] if len(gbm_row) > 0 else float("nan")
    print(f"{band_label:<10} {glm_pred:>14.3f} {gbm_pred:>14.3f}")
```

You will see that the GLM produces nearly flat or weakly sloped relativities across all age bands because the linear coefficient is pulled towards zero by the majority of mid-range-age drivers. The GBM shows the U-shape clearly: high relativities for young drivers, flat middle, mild uplift at 70+.

This is the table to put in front of the pricing committee. The GBM reveals a real risk pattern that the GLM cannot see.

---

## Part 12: Plotting for committee presentation

The library has a built-in plotting function. In a new cell, type this and run it (Shift+Enter):

```python
sr.plot_relativities(
    features=["area", "has_convictions"],
    show_ci=True,
    figsize=(12, 5),
)
display(plt.gcf())
```

You will see two bar charts side by side: area bands and conviction flag. Each bar shows the relativity, with whiskers for the 95% confidence interval. The base level sits at 1.0.

On Databricks, you must call `display(plt.gcf())` after `plot_relativities()` to render the chart in the notebook. Without it, the chart appears in some Databricks environments but not others.

Now produce a more polished age comparison chart manually. In a new cell, type this and run it (Shift+Enter):

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: GBM LOESS curve vs GLM linear prediction
age_range = np.arange(17, 86)
glm_age_pred = np.exp(glm_age_coef * (age_range - base_age))

axes[0].plot(age_curve["feature_value"], age_curve["relativity"],
             color="steelblue", lw=2, label="GBM (LOESS)")
axes[0].plot(age_range, glm_age_pred,
             color="firebrick", lw=2, linestyle="--", label="GLM (linear)")
axes[0].axhline(1.0, color="grey", linestyle=":", alpha=0.5)
axes[0].set_xlabel("Driver age")
axes[0].set_ylabel("Relativity")
axes[0].set_title("Driver age: GBM vs GLM")
axes[0].legend()
axes[0].set_ylim(0.5, 2.5)

# Right: GBM banded relativities
band_rels_sorted = band_rels.sort("age_band")
band_labels = band_rels_sorted["age_band"].to_list()
band_values = band_rels_sorted["relativity"].to_list()
x_pos = range(len(band_labels))

axes[1].bar(x_pos, band_values, color="steelblue", alpha=0.8)
axes[1].axhline(1.0, color="grey", linestyle="--", alpha=0.5)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(band_labels, rotation=30, ha="right")
axes[1].set_ylabel("Relativity")
axes[1].set_title("Driver age: banded relativities (base: 30-39)")

plt.tight_layout()
display(fig)
```

You will see two charts. The left chart shows the GBM's smooth LOESS curve alongside the GLM's linear prediction. The U-shape of the GBM curve is the key result. The right chart shows the banded version ready for a factor table.

---

## Part 13: Protected characteristics and proxy discrimination

Before exporting anything, we need to address a regulatory requirement.

The FCA's Consumer Duty and the Equality Act 2010 place constraints on UK personal lines rating factors. Driver age is currently permitted as a motor rating factor where actuarially justified. But the more important question for any GBM is whether features that are not protected characteristics are acting as proxies for protected characteristics.

SHAP gives you a quantitative way to check this. The test is whether a feature's SHAP values correlate with a proxy for a protected characteristic.

In a new cell, type this and run it (Shift+Enter):

```python
# Example: check if area SHAP values correlate with simulated deprivation index
# In a real portfolio you would use actual deprivation scores or demographic data
# Here we create a synthetic deprivation proxy to demonstrate the method

rng_check = np.random.default_rng(99)
area_deprivation = {"A": 0.3, "B": 0.4, "C": 0.5, "D": 0.6, "E": 0.7, "F": 0.8}
dep_score = np.array([area_deprivation[a] for a in df_pd["area"]]) + rng_check.normal(0, 0.1, len(df_pd))

area_idx  = sr.feature_names_.index("area")
area_shap = sr.shap_values()[:, area_idx]

correlation = np.corrcoef(area_shap, dep_score)[0, 1]
print(f"Correlation between area SHAP values and deprivation proxy: {correlation:.3f}")

if abs(correlation) > 0.5:
    print("WARNING: High correlation suggests area may be acting as a deprivation proxy.")
    print("Discuss with compliance before using these relativities.")
else:
    print("Correlation is modest. Document this check in your model governance file.")
```

You will see a correlation of around 0.55-0.65. In this synthetic example, area and deprivation are by construction correlated - we built the data that way. In a real portfolio, you would use actual deprivation indices (e.g. the Index of Multiple Deprivation) matched to postcode.

**What to do with this result:** A high correlation does not automatically disqualify the feature. Area is a legitimate risk factor in UK motor (urban areas have higher claim frequency due to traffic density, not due to poverty). But you need to document the check, document the justification, and be prepared to discuss it with the FCA if asked. The Consumer Duty requires evidence that your pricing is fair; SHAP gives you the evidence layer.

---

## Part 14: Exporting for Radar

Radar (Willis Towers Watson's pricing software) expects factor tables in a specific CSV format:

```
Factor,Level,Relativity
area,A,1.0000
area,B,1.1080
...
```

In a new cell, type this and run it (Shift+Enter):

```python
def to_radar_csv(rels_df: pd.DataFrame, output_path: str) -> None:
    """
    Export relativities in Radar factor table import format.

    Expects a pandas DataFrame with columns: feature, level, relativity.
    Continuous features should be excluded or pre-banded.
    """
    radar_df = rels_df[["feature", "level", "relativity"]].copy()
    radar_df.columns = ["Factor", "Level", "Relativity"]
    radar_df["Relativity"] = radar_df["Relativity"].round(4)
    radar_df["Level"] = radar_df["Level"].astype(str)
    radar_df.to_csv(output_path, index=False)
    print(f"Written {len(radar_df)} rows to {output_path}")


# Export categorical features only
# Continuous features (ncd_years is treated as categorical here; driver_age needs banding first)
cat_rels_for_export = rels[rels["feature"].isin(CAT_FEATURES)]
to_radar_csv(cat_rels_for_export, "/dbfs/tmp/gbm_relativities_radar.csv")
```

You will see:

```
Written 8 rows to /dbfs/tmp/gbm_relativities_radar.csv
```

Verify the output looks correct. In a new cell, type this and run it (Shift+Enter):

```python
import pandas as pd
radar_check = pd.read_csv("/dbfs/tmp/gbm_relativities_radar.csv")
print(radar_check.to_string(index=False))
```

You will see the Radar-format CSV contents. Verify that:

1. Every area level (A through F) is present
2. The base level (A) has Relativity = 1.0000
3. Relativities increase monotonically from A to F

**Two practical issues with real Radar imports:**

First, your Radar variable names are almost certainly different from your Python column names. `area` in Python might be `ABI_AREA_BAND` in Radar. Map them explicitly in a lookup dictionary before export and version-control that lookup.

Second, Radar requires every level defined in the model to appear in the import file. If your GBM never saw a level that Radar knows about (e.g. NCD=6 if your insurer does not write those risks), you need to add that row manually with an appropriate relativity - either extrapolate from the nearest observed level or use 1.0 with a documented decision. Never let Radar default silently.

---

## Part 15: Adding the banded age table to the export

The banded age relativities also need to go into Radar. But they came from manual aggregation of SHAP values rather than from `extract_relativities()`, so we need to format them manually.

In a new cell, type this and run it (Shift+Enter):

```python
# Format the age band table in Radar format
age_radar = band_rels.select(["age_band", "relativity"]).with_columns([
    pl.lit("driver_age_band").alias("Factor"),
    pl.col("age_band").alias("Level"),
    pl.col("relativity").round(4).alias("Relativity"),
]).select(["Factor", "Level", "Relativity"])

# Write to /dbfs/tmp/
age_radar_pd = age_radar.to_pandas()
age_radar_pd.to_csv("/dbfs/tmp/gbm_age_band_relativities_radar.csv", index=False)

print("Age band Radar export:")
print(age_radar_pd.to_string(index=False))
```

You will see the age band factor table in Radar format. This file, combined with the categorical feature file from Part 14, gives you everything Radar needs for the GBM-derived relativities.

---

## Part 16: Logging to MLflow

Everything that went into production should be tracked. Log the relativities, validation results, and model state to MLflow.

In a new cell, type this and run it (Shift+Enter):

```python
import mlflow
from datetime import date

mlflow.set_experiment("/Users/your-username/pricing-module-04")

with mlflow.start_run(run_name="gbm_shap_relativities_v1") as run:

    # Log model parameters
    mlflow.log_params(freq_params)

    # Log validation results as metrics
    for check_name, result in checks.items():
        mlflow.log_metric(f"validation_{check_name}_passed", int(result.passed))

    # Log reconstruction error specifically
    recon_result = checks["reconstruction"]
    if hasattr(recon_result, "value"):
        mlflow.log_metric("reconstruction_max_error", float(recon_result.value))

    # Log the relativities table as a CSV artefact
    rels_path = "/tmp/gbm_relativities.csv"
    rels.to_csv(rels_path, index=False)
    mlflow.log_artifact(rels_path, "relativities")

    # Log Radar export files
    mlflow.log_artifact("/dbfs/tmp/gbm_relativities_radar.csv", "radar_export")
    mlflow.log_artifact("/dbfs/tmp/gbm_age_band_relativities_radar.csv", "radar_export")

    # Log key relativities as metrics for easy comparison
    area_f_rel = rels[rels["feature"] == "area"].set_index("level").loc["F", "relativity"]
    ncd5_rel   = rels[rels["feature"] == "ncd_years"].set_index("level").loc[5, "relativity"]
    conv_rel   = rels[rels["feature"] == "has_convictions"].set_index("level").loc[1, "relativity"]

    mlflow.log_metric("area_F_relativity",        area_f_rel)
    mlflow.log_metric("ncd5_relativity",           ncd5_rel)
    mlflow.log_metric("conviction_relativity",     conv_rel)
    mlflow.log_metric("n_policies",               len(df))

    # Log the CatBoost model itself
    mlflow.catboost.log_model(freq_model, "catboost_model")

    print(f"Run ID: {run.info.run_id}")
    print(f"Area F relativity logged:    {area_f_rel:.4f}")
    print(f"NCD=5 relativity logged:     {ncd5_rel:.4f}")
    print(f"Conviction relativity logged:{conv_rel:.4f}")
```

You will see the run ID printed and the key metrics. Go to the MLflow UI (click the flask icon in the Databricks left sidebar, then Experiments) to verify the run was logged. You should see:

- The model parameters (loss function, learning rate, etc.)
- The validation pass/fail metrics
- The key relativities as searchable metrics
- The relativities CSV and Radar export files as downloadable artefacts

**Why log relativities to MLflow and not just Unity Catalog?** MLflow gives you the link between the specific model run (the exact training data, the exact hyperparameters) and the relativities derived from it. If someone later asks "which model version did the area F = 1.95 relativity come from?", you can find it in MLflow. Unity Catalog stores the relativity table for querying; MLflow stores the audit trail.

---

## Part 17: Writing relativities to Unity Catalog

For production use, write the relativities to a versioned Delta table. In a new cell, type this and run it (Shift+Enter):

```python
from datetime import date

# Add metadata columns before writing
rels_with_meta = rels.copy()
rels_with_meta["model_run_date"] = str(date.today())
rels_with_meta["model_name"]     = "freq_catboost_module04_v1"
rels_with_meta["n_policies"]     = len(df)
rels_with_meta["mlflow_run_id"]  = run.info.run_id

# Write to Unity Catalog via Spark
spark.createDataFrame(rels_with_meta).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("main.pricing.gbm_relativities")

print("Relativities written to main.pricing.gbm_relativities")

# Write validation log (append, so history is preserved)
validation_records = [
    {
        "check":          name,
        "passed":         result.passed,
        "message":        result.message,
        "model_run_date": str(date.today()),
        "model_name":     "freq_catboost_module04_v1",
        "mlflow_run_id":  run.info.run_id,
    }
    for name, result in checks.items()
]

spark.createDataFrame(validation_records).write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("main.pricing.gbm_relativity_validation_log")

print("Validation log written to main.pricing.gbm_relativity_validation_log")
```

You will see confirmation that the tables were written. You can query them with:

```python
display(spark.table("main.pricing.gbm_relativities"))
```

The table has the full relativity output including confidence intervals, exposure weights, and the metadata columns you added. This is the permanent record.

---

## Part 18: Limitations to document

Every time you present GBM relativities to a pricing committee or regulator, you need to be explicit about what the intervals do and do not tell you. Write these into your committee paper, not just your notebook.

**Confidence intervals cover data uncertainty only.** The 95% CI on the area F relativity tells you how precisely the data pins down the mean SHAP contribution given the 10,000 area F policies you have. It does not tell you whether a different training run of the GBM would produce the same relativity. Two bootstrap refits of the model will give slightly different SHAP values. The library does not quantify this model uncertainty - be explicit when you say "95% confidence interval."

**SHAP attribution for correlated features is not unique.** If `vehicle_group` and `driver_age` are correlated (older drivers tend to drive lower-group vehicles in the real world), some of each feature's true attribution may be allocated to the other under `tree_path_dependent`. The total effect of the correlated cluster is correct; the individual feature attributions are approximate.

**Log-link only.** `exp(SHAP)` gives multiplicative relativities only for log-link objectives: Poisson, Gamma, Tweedie. Do not apply this methodology to a linear-link model.

**Risk relativities only.** These are pure risk relativities. They do not include expense loadings, profit margins, or reinsurance costs. Every slide showing these tables should say "risk only relativities."

**Interaction effects.** A vehicle group × driver age interaction is partly attributed to each feature's SHAP values. The vehicle group relativity from the GBM is not directly comparable to the GLM's vehicle group coefficient if the GLM has no interaction term - the GBM's estimate carries its share of the interaction.

---

## Summary

The workflow in five steps:

1. `sr = SHAPRelativities(model, X, exposure, categorical_features=..., continuous_features=...)`
2. `sr.fit()` - computes SHAP values via CatBoost's native TreeSHAP (15-60 seconds for 100k rows)
3. `sr.validate()` - run this before trusting any output; stop if reconstruction fails
4. `sr.extract_relativities(normalise_to="base_level", base_levels=...)` - get the factor table
5. `sr.extract_continuous_curve(feature, smooth_method="loess")` + manual aggregation by band - get banded continuous features

The output `rels` DataFrame has columns `feature`, `level`, `relativity`, `lower_ci`, `upper_ci`, `mean_shap`, `shap_std`, `n_obs`, `exposure_weight`. Every number needed to explain, challenge, and defend the relativities to a pricing committee is in that table.

The GBM finds things the GLM cannot - the driver age U-shape, the young driver × high vehicle group interaction, non-linear NCD effects in some portfolios. SHAP makes those findings visible, auditable, and presentable. That is how the GBM gets out of the notebook and into production.
