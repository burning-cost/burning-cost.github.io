# Module 3: GBMs for Insurance Pricing - CatBoost

In Module 2 you fitted a Poisson GLM for claim frequency and a Gamma GLM for claim severity on a synthetic UK motor portfolio. You extracted relativities, ran diagnostics, and logged the model to MLflow. That GLM is your benchmark. This module asks whether a gradient boosted machine can beat it - and teaches you how to answer that question honestly.

The honest answer requires a proper evaluation framework. By the end of this module you will have fitted a CatBoost frequency model using walk-forward cross-validation designed for insurance data, tuned its hyperparameters with Optuna, fitted a Gamma-equivalent severity model, and produced the two diagnostics - Gini coefficient and double lift chart - that you would present to a pricing committee. Everything is logged to MLflow so the comparison is auditable.

We use the same synthetic motor portfolio as Module 2. The GLM you fitted there is the benchmark throughout this module.

---

## Part 1: Why this matters - the question about GBMs

The question most pricing teams ask about GBMs is: "does it beat the GLM?" That is the wrong question. It leads to cherry-picked test splits, optimistic Gini comparisons, and over-confident deployment decisions.

The right question is: "does the GBM find genuine risk signal that the GLM misses, and is that signal large enough to justify the additional governance overhead?"

The difference matters. A GBM always fits the training data better than a GLM - it is more flexible by design. What you need to establish is whether that additional flexibility corresponds to real out-of-sample discrimination, and whether the improvement is large enough to justify a more complex model that requires a more expensive audit trail.

This module shows you how to establish that rigorously.

### Why CatBoost specifically

There are three major GBM libraries in common use: XGBoost, LightGBM, and CatBoost. For insurance pricing, CatBoost has two properties that make it the better choice.

**Native categorical handling.** Rating factors in UK motor pricing are categorical: area band, vehicle group, NCD years. XGBoost and LightGBM require you to encode these as integers or one-hot dummy variables before training. One-hot encoding a 50-level vehicle group creates 50 binary features and destroys any ordinal structure. Integer encoding imposes an assumption that the factor has a linear effect in the encoded order. CatBoost uses ordered target statistics - a form of regularised mean encoding computed on a permuted training set - which handles categorical factors correctly without either assumption.

**Proper Poisson loss with exposure offset.** CatBoost implements a Poisson log-likelihood loss and integrates the exposure offset cleanly into its Pool object (explained in Part 3). This is the right objective for claim count modelling and the cleanest integration for insurance data.

---

## Part 2: Setting up the notebook

### Create the notebook

Go to your Databricks workspace. In the left sidebar, click **Workspace**. Navigate to your user folder (or your Git repo if you set one up in Module 1).

Click the **+** button and choose **Notebook**. Name it `module-03-gbm-catboost`. Keep the default language as Python. Click **Create**.

The notebook opens with one empty cell. Check the cluster selector at the top. If it says "Detached," click it and select your cluster from the dropdown. Once the cluster name appears in green, you are connected.

If the cluster is not running, go to **Compute** in the left sidebar, find your cluster, and click **Start**. It takes 3-5 minutes. Come back to the notebook once it shows "Running."

### Install the libraries

The libraries we need for this module are not all pre-installed on Databricks. In the first cell of your notebook, type this and run it (Shift+Enter):

```python
%pip install catboost optuna insurance-cv mlflow polars
```

You will see a long stream of output as pip downloads and installs each package. Wait for it to finish completely. At the end you will see:

```
Note: you may need to restart the Python kernel to use updated packages.
```

In the next cell, run:

```python
dbutils.library.restartPython()
```

This restarts the Python session. Any variables from before the restart are gone - this is expected. Always put your `%pip install` cell at the very top of the notebook, before any other code, so you only need to restart once per session.

Here is what each library does:

- **catboost** - the GBM library we use throughout this module. Built by Yandex, it handles categorical features natively and has a clean API for Poisson and Tweedie loss functions.
- **optuna** - a hyperparameter tuning framework. It uses Bayesian optimisation to search the parameter space efficiently rather than trying every combination in a grid.
- **insurance-cv** - a small library that generates walk-forward cross-validation splits specifically designed for insurance data, with an IBNR buffer year. We explain this fully in Part 6.
- **mlflow** - the experiment tracking library built into Databricks. It logs model parameters, metrics, and the model artefact itself so every run is reproducible and auditable.
- **polars** - the data manipulation library from Module 2. If it is already at cluster level, the install is a no-op.

### Confirm the imports work

In the next cell, type this and run it:

```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost
import optuna
import json
from datetime import date
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import roc_auc_score
from insurance_cv import WalkForwardCV
from mlflow.tracking import MlflowClient

print(f"Polars:   {pl.__version__}")
print(f"CatBoost: __version__ not exposed, but imported OK")
print(f"Optuna:   {optuna.__version__}")
print("All imports OK")
```

You should see version numbers with no errors. If you see `ModuleNotFoundError`, the install cell did not complete - check that you ran it before the restart, and if so run `%pip install` again.

**Why `import mlflow.catboost` is a separate line:** `mlflow` does not automatically load its CatBoost integration. The submodule `mlflow.catboost` must be imported explicitly before you can call `mlflow.catboost.log_model()`.

---

## Part 3: The exposure offset - the most important implementation detail

Before we build the model, we need to cover something that is easy to get wrong and hard to detect once you have got it wrong.

### What an exposure offset is

In a Poisson GLM, exposure enters as an offset: `log(mu) = Xb + log(exposure)`. The model predicts claim count `mu = exp(Xb) * exposure`. A policy with 0.5 years of exposure gets half the predicted claims of an otherwise identical policy with a full year - which is correct, because it was only exposed for half the year.

In CatBoost, the equivalent is the `baseline` parameter. The model's linear predictor is `f(X) + baseline`, and with a log link, the prediction is `exp(f(X) + baseline)`. To replicate the GLM offset, you pass `baseline=np.log(exposure)`.

### The trap

The wrong version: `baseline=exposure`. This passes the raw exposure value (typically 0.25 to 1.0 for mid-year policies) directly as the baseline. The prediction becomes `exp(f(X) + exposure)`. For a policy with 0.5 years of exposure, this multiplies the prediction by `exp(0.5) = 1.65` rather than `exp(log(0.5)) = 0.5`. The model still trains and converges - CatBoost has no way of knowing the baseline is wrong - but the predictions are systematically incorrect in a way that depends on exposure level.

Even worse: no baseline at all. This assumes every policy has exposure = 1.0. For a book where most policies have exposure below 1.0, this overestimates predicted counts uniformly and produces a calibration shift that looks like a model problem but is actually an implementation error.

The correct pattern, written once clearly so it is easy to refer back to:

```python
# exposure is in years (e.g. 0.25 to 1.0 for mid-year policies)
# CORRECT: log-transform before passing as baseline
train_pool = Pool(
    X_train,
    y_train,
    baseline=np.log(exposure_train),   # must be log(exposure), not exposure
    cat_features=CAT_FEATURES,
)
```

Exercise 1 in `exercises.md` quantifies this error empirically. We recommend doing Exercise 1 before continuing, as it builds real intuition for why this matters.

---

## Part 4: Regenerating the motor dataset

We use the same 100,000-policy synthetic motor portfolio from Module 2. If you have it saved as a Delta table in your workspace, you can read it back. If not, we regenerate it here.

Create a new cell with a markdown header to keep the notebook organised. In Databricks, a cell starting with `%md` renders as formatted text rather than running Python:

```python
%md
## Part 4: Data preparation
```

Now create the next cell and paste in the data generation code. This is the same DGP (data generating process) as Module 2, so the GLM and GBM are trained on the same underlying truth:

```python
rng = np.random.default_rng(seed=42)
n = 100_000

# Rating factors
areas = ["A", "B", "C", "D", "E", "F"]
area = rng.choice(areas, size=n, p=[0.10, 0.18, 0.25, 0.22, 0.15, 0.10])
vehicle_group = rng.integers(1, 51, size=n)
ncd_years = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.08, 0.07, 0.09, 0.12, 0.20, 0.44])
driver_age = rng.integers(17, 86, size=n)
conviction_points = rng.choice([0, 3, 6, 9], size=n, p=[0.78, 0.12, 0.07, 0.03])
exposure = np.clip(rng.beta(8, 2, size=n), 0.05, 1.0)

# True log-frequency: log link, multiplicative effects
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
    # Superadditive interaction: young driver x high vehicle group
    # This is what the GLM misses and the GBM finds
    + np.where((driver_age < 25) & (vehicle_group > 35), 0.30, 0.0)
)

claim_count = rng.poisson(np.exp(log_mu) * exposure)

# Severity: Gamma-distributed, log link
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
    "accident_year":      rng.choice([2019, 2020, 2021, 2022, 2023, 2024], size=n,
                                      p=[0.12, 0.14, 0.16, 0.18, 0.20, 0.20]),
    "area":               area,
    "vehicle_group":      vehicle_group.astype(np.int32),
    "ncd_years":          ncd_years.astype(np.int32),
    "driver_age":         driver_age.astype(np.int32),
    "conviction_points":  conviction_points.astype(np.int32),
    "exposure_years":     exposure,
    "claim_count":        claim_count.astype(np.int32),
    "incurred":           incurred,
})

df.head(5)
```

Run this cell. It takes a second or two. The output shows the first five rows of the DataFrame as a table.

What you are looking at: 100,000 motor policies across accident years 2019-2024. Each row is one policy. The `accident_year` column is important - we use it to construct our cross-validation folds later. The true DGP contains a superadditive interaction between young drivers (under 25) and high vehicle groups (above 35): these policies are worse than the multiplicative combination of the youth penalty and the vehicle group effect would suggest. This is the signal that the GLM underestimates and the GBM should find.

Now check some basic counts:

```python
print(f"Total policies: {len(df):,}")
print(f"Total claims:   {df['claim_count'].sum():,}")
print(f"Total exposed:  {df['exposure_years'].sum():,.0f} policy-years")
print(f"Claim frequency:{df['claim_count'].sum() / df['exposure_years'].sum():.4f}")
print()
print("Policies by accident year:")
print(
    df.group_by("accident_year")
    .agg(
        pl.len().alias("policies"),
        pl.col("claim_count").sum().alias("claims"),
        pl.col("exposure_years").sum().alias("exposure"),
    )
    .with_columns((pl.col("claims") / pl.col("exposure")).alias("freq"))
    .sort("accident_year")
)
```

You should see roughly 12,000-20,000 policies per accident year, with a portfolio claim frequency around 5-6%. The frequency should be fairly stable across years - the synthetic DGP has no trend. This is the starting point.

---

## Part 5: Feature definitions

Before fitting anything, we need to decide which features go into the model and which are categorical.

Create a new cell:

```python
# Feature definitions for the motor GBM
CONTINUOUS_FEATURES = ["driver_age", "vehicle_group", "ncd_years"]
CAT_FEATURES        = ["area", "conviction_points"]
FEATURES            = CONTINUOUS_FEATURES + CAT_FEATURES

FREQ_TARGET  = "claim_count"
EXPOSURE_COL = "exposure_years"
SEV_TARGET   = "incurred"
```

Run it. There is no output - you are just setting up variable names that the rest of the notebook will use.

**Why `conviction_points` is categorical:** The values are 0, 3, 6, and 9. These are penalty point totals. They are not continuous quantities: the step from 0 to 3 points (one minor offence) is qualitatively different from 6 to 9 (approaching disqualification territory). Treating them as a continuous number would impose a linear assumption on the effect. As a categorical, CatBoost learns the effect of each penalty level independently using ordered target statistics.

**Why `vehicle_group` is continuous:** ABI groups 1-50 have a roughly monotone relationship with risk - higher groups are generally more expensive vehicles. The continuous treatment allows CatBoost to find non-linear effects within that trend, which tree splits handle naturally.

**Why `driver_age` is continuous:** Age has a well-known non-linear effect - a peak of risk in the under-25 band and a secondary peak in the over-70 band, with a flat middle. Tree splits on a continuous variable capture this shape without requiring manual bucketing.

---

## Part 6: Why random cross-validation is wrong for insurance data

This is the section that most pricing teams get wrong, and getting it wrong leads to over-confident deployment decisions.

Standard machine learning practice is an 80/20 random train/test split. For insurance data, this is wrong. It produces optimistic performance estimates that overstate real out-of-sample performance.

The reason is temporal autocorrelation. Insurance claims data has structure across accident years: inflation trends, frequency cycles, mix changes. A model trained on a random 80% of 2019-2024 data sees policies from all six years in both training and test. When it makes predictions on the test set, it is not generalising to new data - it is filling in gaps in a history it has already seen. The temporal patterns in the test set match those in the training set, so the model looks better than it will be when predicting 2025 policies from 2024 training data.

Walk-forward cross-validation replicates the actual deployment scenario:

- Fold 1: train on 2019-2020, validate on 2022 (2021 excluded as IBNR buffer)
- Fold 2: train on 2019-2021, validate on 2023 (2022 excluded as IBNR buffer)
- Fold 3: train on 2019-2022, validate on 2024 (2023 excluded as IBNR buffer)

**What the IBNR buffer does:** Claims from the most recent accident year are typically 30-50% underdeveloped at the time you would be fitting the model. A claim from December 2022 accident year may not be fully settled until late 2023 or 2024. Including that year in training gives the model access to partially-developed claims data that will not be available at deployment time. Excluding the year immediately before the validation year is more conservative - and more honest about what the model can actually do in production.

### Setting up the walk-forward CV

The `insurance-cv` library generates these splits automatically. Create a new cell:

```python
# Convert to pandas for the CV split - WalkForwardCV uses pandas index arrays
df_pd = df.to_pandas()

cv = WalkForwardCV(
    year_col="accident_year",
    min_train_years=2,
    ibnr_buffer_years=1,
    n_splits=3,
)

folds = list(cv.split(df_pd))

# Print the fold structure so you can see exactly what is being trained and validated
for i, (train_idx, val_idx) in enumerate(folds):
    train_years = sorted(df_pd.iloc[train_idx]["accident_year"].unique().tolist())
    val_years   = sorted(df_pd.iloc[val_idx]["accident_year"].unique().tolist())
    print(f"Fold {i+1}: train years = {train_years}, validate years = {val_years}")
```

Run this. You should see output like:

```
Fold 1: train years = [2019, 2020], validate years = [2022]
Fold 2: train years = [2019, 2020, 2021], validate years = [2023]
Fold 3: train years = [2019, 2020, 2021, 2022], validate years = [2024]
```

**Why 2021 is missing from Fold 1:** It is the IBNR buffer year between the training data (ending 2020) and the validation year (2022). Same for 2022 in Fold 2, and 2023 in Fold 3.

The `folds` variable contains a list of (train_index, val_index) pairs. These are the row indices into `df_pd` that belong to each split. You will use them in the CV loop in Part 8.

---

## Part 7: The CatBoost Pool object

Before we get to the CV loop, we need to understand the `Pool` object. It is CatBoost's way of bundling together the features, the target, and any metadata (exposure offset, categorical feature list) so you do not have to pass them separately to every function call.

A Pool looks like this:

```python
from catboost import CatBoostRegressor, Pool

train_pool = Pool(
    data=X_train,           # feature matrix (DataFrame or numpy array)
    label=y_train,          # target column (claim counts)
    baseline=np.log(w_train),  # exposure offset: log(exposure_years)
    cat_features=CAT_FEATURES, # list of column names that are categorical
)
```

The Pool pre-processes the categorical features using CatBoost's ordered target statistics. It does this work once at Pool construction time rather than on every training iteration. This is why you should always build the Pool once and reuse it - constructing it inside a loop wastes the encoding work.

Once you have a Pool, you pass it directly to `model.fit()` and `model.predict()`:

```python
model.fit(train_pool, eval_set=val_pool)  # eval_set is the validation Pool
predictions = model.predict(val_pool)      # returns predicted counts (not frequencies)
```

**What `.predict()` returns:** The CatBoost Poisson model predicts expected claim counts, not expected claim frequencies. A policy with exposure 0.5 and a frequency of 0.06 gets a predicted count of 0.03. To convert to frequency, divide by exposure: `predicted_count / exposure`.

---

## Part 8: The loss function - Poisson deviance

We need a metric for comparing model performance across CV folds. RMSE is wrong for count data.

RMSE treats a miss of 2 claims on a policy with 0.1 expected claims the same as a miss of 2 claims on a policy with 3 expected claims. The first is a catastrophic relative error; the second is 67% error. RMSE weights them equally. This is not what we want for insurance pricing.

The Poisson deviance weights residuals appropriately for count data. Create a new cell and define the function:

```python
def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray, exposure: np.ndarray) -> float:
    """
    Scaled Poisson deviance per unit exposure.

    y_true and y_pred are on the count scale (not frequency).
    We convert to frequency internally, then compute deviance per unit exposure.
    Dividing by total exposure gives a per-policy-year figure that is
    comparable across portfolios with different exposure distributions.

    Lower is better.
    """
    freq_pred = np.clip(y_pred / exposure, 1e-10, None)  # avoid log(0)
    freq_true = y_true / exposure
    deviance = 2 * exposure * (
        np.where(freq_true > 0, freq_true * np.log(freq_true / freq_pred), 0.0)
        - (freq_true - freq_pred)
    )
    return float(deviance.sum() / exposure.sum())
```

Run this cell. No output - you are defining a function for later use.

The formula penalises large misses more when the prediction is small. On UK motor data, a well-fitted Poisson GLM achieves deviance around 0.18-0.22 on a temporal validation set. The CatBoost model typically achieves 0.16-0.20 after tuning. The absolute values matter less than the comparison on the same test set.

---

## Part 9: The cross-validation loop

Now we put it together. The CV loop trains a fresh CatBoost model on each fold and evaluates it on the held-out year. We track the deviance across folds to get a mean and variance.

Create a new cell:

```python
cv_deviances = []

for fold_idx, (train_idx, val_idx) in enumerate(folds):
    df_train = df_pd.iloc[train_idx]
    df_val   = df_pd.iloc[val_idx]

    X_train = df_train[FEATURES]
    y_train = df_train[FREQ_TARGET].values
    w_train = df_train[EXPOSURE_COL].values

    X_val = df_val[FEATURES]
    y_val = df_val[FREQ_TARGET].values
    w_val = df_val[EXPOSURE_COL].values

    train_pool = Pool(X_train, y_train, baseline=np.log(w_train), cat_features=CAT_FEATURES)
    val_pool   = Pool(X_val,   y_val,   baseline=np.log(w_val),   cat_features=CAT_FEATURES)

    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Poisson",
        eval_metric="Poisson",
        random_seed=42,
        verbose=0,     # 0 means no training progress output per fold
    )
    model.fit(train_pool, eval_set=val_pool)

    y_pred = model.predict(val_pool)
    fold_dev = poisson_deviance(y_val, y_pred, w_val)
    cv_deviances.append(fold_dev)

    val_year = sorted(df_pd.iloc[val_idx]["accident_year"].unique().tolist())
    print(f"Fold {fold_idx+1} (validate {val_year}): Poisson deviance = {fold_dev:.4f}")

print(f"\nMean CV deviance: {np.mean(cv_deviances):.4f}")
print(f"Std CV deviance:  {np.std(cv_deviances):.4f}")
```

Run this cell. It trains three CatBoost models. On Free Edition this takes 2-4 minutes. The output should look like:

```
Fold 1 (validate [2022]): Poisson deviance = 0.1923
Fold 2 (validate [2023]): Poisson deviance = 0.1887
Fold 3 (validate [2024]): Poisson deviance = 0.1841
```

```
Mean CV deviance: 0.1884
Std CV deviance:  0.0034
```

The exact numbers will vary slightly. What you are looking for is that the three folds produce similar deviances - if one fold is substantially worse than the others, it may indicate a data quality issue in that year or a genuine model instability.

**A note on what the standard deviation means:** Three folds gives a crude estimate of variance, not a proper confidence interval. Do not report a confidence interval computed from three folds as if it is statistically rigorous. What you can say is: "CV deviance across three temporal folds ranged from X to Y, with a mean of Z."

**What `verbose=0` does:** By default, CatBoost prints training progress on every iteration. On 500 iterations across three folds, that is 1,500 lines of output. Setting `verbose=0` suppresses this. If you want to see progress (useful when debugging), change it to `verbose=100` to print every 100 iterations.

---

## Part 10: Hyperparameter tuning with Optuna

The default parameters we used in Part 9 (depth=6, learning_rate=0.05, iterations=500) are reasonable starting points but not optimised. Optuna searches the parameter space to find better values.

We tune on the last fold only - train on 2019-2022, validate on 2023. Tuning on all folds is more rigorous but multiplies compute time by the number of folds. For a 100,000-policy book, 40 trials on a single fold takes 15-20 minutes on a standard cluster.

First, extract the last fold data. Create a new cell:

```python
# Use the last fold for tuning: train 2019-2022, validate 2023
# (fold index 2, since folds is 0-indexed)
train_idx_t, val_idx_t = folds[-1]

df_train_t = df_pd.iloc[train_idx_t]
df_val_t   = df_pd.iloc[val_idx_t]

X_train_t = df_train_t[FEATURES]
y_train_t = df_train_t[FREQ_TARGET].values
w_train_t = df_train_t[EXPOSURE_COL].values

X_val_t = df_val_t[FEATURES]
y_val_t = df_val_t[FREQ_TARGET].values
w_val_t = df_val_t[EXPOSURE_COL].values

# Build Pool objects ONCE outside the objective function.
# CatBoost re-encodes categoricals at Pool construction time.
# If you construct inside the objective, this encoding work happens
# on every trial - wasted effort across 40 trials.
train_pool_t = Pool(X_train_t, y_train_t, baseline=np.log(w_train_t), cat_features=CAT_FEATURES)
val_pool_t   = Pool(X_val_t,   y_val_t,   baseline=np.log(w_val_t),   cat_features=CAT_FEATURES)

print(f"Tuning on: {sorted(df_train_t['accident_year'].unique().tolist())} -> validate {sorted(df_val_t['accident_year'].unique().tolist())}")
```

Now define the Optuna objective function. Create a new cell:

```python
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress Optuna's own verbose output

def objective(trial: optuna.Trial) -> float:
    params = {
        "iterations":    trial.suggest_int("iterations", 200, 1000),
        "depth":         trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Poisson",
        "eval_metric":   "Poisson",
        "random_seed":   42,
        "verbose":       0,
    }
    model = CatBoostRegressor(**params)
    model.fit(train_pool_t, eval_set=val_pool_t)
    pred = model.predict(val_pool_t)
    return poisson_deviance(y_val_t, pred, w_val_t)
```

**What each parameter does:**

- **depth**: the maximum depth of each tree. Controls how many features can interact. Depth 4 means at most 4-way interactions. For motor data with 5-8 features, depth 4-6 is usually optimal. Depth 7+ overfits without improving validation deviance.
- **learning_rate**: how large a step each tree takes. Lower rates require more iterations but generalise better. We search on a log scale (0.02 to 0.15) because the effect is multiplicative.
- **l2_leaf_reg**: L2 regularisation on leaf values. Increase this if training deviance is much lower than validation deviance - it is the standard overfitting signal.
- **iterations**: the number of trees. Interacts with learning_rate - a low rate needs more iterations to converge. Optuna handles this by exploring the joint space.

Now run the study. Create a new cell:

```python
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=40, show_progress_bar=True)

best_params = study.best_params
print(f"\nBest Poisson deviance (40 trials): {study.best_value:.5f}")
print("\nBest parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
```

Run this. Optuna runs 40 trials. On Free Edition this takes 10-20 minutes. The progress bar shows completed trials. Let it run.

**What Optuna is doing internally:** The first 10-15 trials are essentially random exploration. After that, Optuna uses a Tree-structured Parzen Estimator (TPE) to concentrate subsequent trials on the most promising regions of the parameter space. The marginal improvement from trials 30-40 is typically small compared to trials 1-20 - we use 40 to be thorough.

After it finishes, run this in the next cell to see which parameters drove most of the variation across trials:

```python
importances = optuna.importance.get_param_importances(study)
print("Parameter importances (what drove trial-to-trial variation):")
for param, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
    print(f"  {param}: {imp:.3f}")
```

On UK motor data with 5-8 features, typical results: `depth` accounts for 40-55% of trial variance, `learning_rate` 25-35%, `l2_leaf_reg` 10-20%, `iterations` 5-15%. This tells you that depth is the parameter worth tuning most carefully. If compute time is limited, fixing depth=5 and tuning only learning_rate and iterations in 20 trials will get you within 0.001-0.002 deviance of the full search.

---

## Part 11: Training the final frequency model and logging to MLflow

After tuning, we train the final model on all available data except the held-out test year. We use 2024 as the test year - it is the most recent accident year, making this the closest approximation to "predicting next year from this year's data."

### Build the final model parameters

Create a new cell:

```python
# Add the fixed parameters to the best tuned parameters
freq_params = {
    **best_params,
    "loss_function": "Poisson",
    "eval_metric":   "Poisson",
    "random_seed":   42,
    "verbose":       100,  # print progress every 100 iterations when training final model
}

max_year = df["accident_year"].max()
print(f"Test year (held out): {max_year}")
print(f"Training on accident years: {sorted(df.filter(pl.col('accident_year') < max_year)['accident_year'].unique().to_list())}")
```

### Prepare the final train and test sets

```python
df_train_final = df.filter(pl.col("accident_year") < max_year)
df_test_final  = df.filter(pl.col("accident_year") == max_year)

X_train_f = df_train_final[FEATURES].to_pandas()
y_train_f = df_train_final[FREQ_TARGET].to_numpy()
w_train_f = df_train_final[EXPOSURE_COL].to_numpy()

X_test_f = df_test_final[FEATURES].to_pandas()
y_test_f = df_test_final[FREQ_TARGET].to_numpy()
w_test_f = df_test_final[EXPOSURE_COL].to_numpy()

final_train_pool = Pool(X_train_f, y_train_f, baseline=np.log(w_train_f), cat_features=CAT_FEATURES)
final_test_pool  = Pool(X_test_f,  y_test_f,  baseline=np.log(w_test_f),  cat_features=CAT_FEATURES)
```

### Train and log to MLflow

MLflow runs are created with a context manager (`with mlflow.start_run(...)`). Everything inside the `with` block is recorded as part of this run. Create a new cell:

```python
with mlflow.start_run(run_name="freq_catboost_tuned") as run_freq:
    # Log what we did
    mlflow.log_params(freq_params)
    mlflow.log_param("cv_strategy",  "walk_forward_ibnr1")
    mlflow.log_param("n_cv_folds",   len(folds))
    mlflow.log_param("features",     json.dumps(FEATURES))
    mlflow.log_param("cat_features", json.dumps(CAT_FEATURES))
    mlflow.log_param("train_years",  str(sorted(df_train_final["accident_year"].unique().to_list())))
    mlflow.log_param("test_year",    str(max_year))
    mlflow.log_param("run_date",     str(date.today()))

    # Train
    freq_model = CatBoostRegressor(**freq_params)
    freq_model.fit(final_train_pool, eval_set=final_test_pool)

    # Evaluate on the test year
    y_pred_freq = freq_model.predict(final_test_pool)
    test_dev    = poisson_deviance(y_test_f, y_pred_freq, w_test_f)

    # Log metrics
    mlflow.log_metric("test_poisson_deviance", test_dev)
    mlflow.log_metric("mean_cv_deviance",      np.mean(cv_deviances))
    mlflow.log_metric("cv_deviance_std",       np.std(cv_deviances))

    # Log the model artefact
    mlflow.catboost.log_model(freq_model, "freq_model")
    freq_run_id = run_freq.info.run_id

print(f"\nTest year Poisson deviance: {test_dev:.4f}")
print(f"MLflow run ID: {freq_run_id}")
```

Run this cell. The model trains on 2019-2023 data (80,000 policies) and the training progress prints every 100 iterations. At the end you see the test deviance.

**What MLflow is recording:** The `log_params` call records all the hyperparameters. The `log_metric` calls record the evaluation results. The `log_model` call saves the entire CatBoost model object as an MLflow artefact. If someone needs to reproduce this model result in 18 months, they open the MLflow experiment, find this run by its ID, and load the model artefact. The training year range tells them which Delta table version to use for the data. That is the audit trail.

To view the MLflow run in the UI: in the Databricks left sidebar, click **Experiments**. Find the experiment for this notebook. Click on the run named "freq_catboost_tuned". You should see all the parameters and metrics logged.

---

## Part 12: Feature importances

Before moving to the severity model, let us look at what the frequency model is relying on. Create a new cell:

```python
importances = freq_model.get_feature_importance(type="FeatureImportance")

imp_df = (
    pl.DataFrame({"feature": FEATURES, "importance": importances.tolist()})
    .sort("importance", descending=True)
)
print(imp_df)
```

Run this. You will see something like:

```
shape: (5, 2)
feature            importance
ncd_years          35.2
driver_age         28.7
vehicle_group      19.4
area               10.8
conviction_points   5.9
```

The exact values vary, but NCD years and driver age are usually dominant for UK motor frequency.

**What this metric measures:** CatBoost's default importance is PredictionValuesChange - the average change in the prediction when a feature's value is varied across the training data, normalised to sum to 100. It is a portfolio-level summary. It tells you which features the model is using overall, but not how any individual feature affects a specific prediction or what direction the effect is. Module 4 covers SHAP values for that purpose.

For the pricing committee presentation, feature importances answer "what is the model using?" The double lift chart (Part 14) answers "is it finding real risk differentiation?" You need both.

---

## Part 13: The severity model

The severity model predicts average cost per claim for policies that had at least one claim. Two design decisions need explaining before we fit it.

**No exposure offset.** The cost of a claim does not depend on how long the policy was in force. A repair costing £3,500 costs £3,500 whether the policy had been running for 3 months or 12 months at the time of the accident. Exposure enters the frequency model because frequency is a rate (claims per year). Severity is a conditional cost - we condition on having a claim, and that conditioning removes the exposure effect.

**Tweedie variance_power=2 as the Gamma equivalent.** CatBoost does not offer a `Gamma` loss function by name. The Tweedie distribution with variance power p=2 is mathematically the Gamma distribution with log link. Power=1 is Poisson. Power between 1 and 2 is compound Poisson-Gamma. Power=2 is Gamma. The relationship is exact.

### Prepare the severity data

Create a new cell:

```python
# Severity model: claims-only subset
# Average severity = total incurred / claim count
df_train_sev = df_train_final.filter(pl.col("claim_count") > 0).with_columns(
    (pl.col("incurred") / pl.col("claim_count")).alias("avg_severity")
)
df_test_sev = df_test_final.filter(pl.col("claim_count") > 0).with_columns(
    (pl.col("incurred") / pl.col("claim_count")).alias("avg_severity")
)

print(f"Training claims: {len(df_train_sev):,} ({100*len(df_train_sev)/len(df_train_final):.1f}% of training policies)")
print(f"Test claims:     {len(df_test_sev):,} ({100*len(df_test_sev)/len(df_test_final):.1f}% of test policies)")
print(f"\nMean severity (training): £{df_train_sev['avg_severity'].mean():,.0f}")
print(f"Mean severity (test):     £{df_test_sev['avg_severity'].mean():,.0f}")
print(f"95th percentile (test):   £{df_test_sev['avg_severity'].quantile(0.95):,.0f}")
```

### Fit the severity model

```python
X_train_s = df_train_sev[FEATURES].to_pandas()
y_train_s = df_train_sev["avg_severity"].to_numpy()

X_test_s  = df_test_sev[FEATURES].to_pandas()
y_test_s  = df_test_sev["avg_severity"].to_numpy()

sev_params = {
    **best_params,
    "loss_function": "Tweedie:variance_power=2",   # Gamma equivalent
    "eval_metric":   "RMSE",
    "random_seed":   42,
    "verbose":       100,
}

# NOTE: No baseline parameter - severity has no exposure offset.
sev_train_pool = Pool(X_train_s, y_train_s, cat_features=CAT_FEATURES)
sev_test_pool  = Pool(X_test_s,  y_test_s,  cat_features=CAT_FEATURES)

sev_model = CatBoostRegressor(**sev_params)
sev_model.fit(sev_train_pool, eval_set=sev_test_pool)
```

**A practical note on shared hyperparameters:** Here we are using the same tuned hyperparameters for the severity model as for the frequency model. This is a tutorial simplification. In production, tune severity hyperparameters separately. The optimal depth for a Poisson frequency model on 100,000 policies is not necessarily optimal for a Gamma severity model on the 7-10% of policies that had claims. The severity model is fitting a smaller, noisier dataset with a different response distribution.

### Evaluate and log the severity model

```python
y_pred_sev = sev_model.predict(sev_test_pool)
sev_rmse   = float(np.sqrt(np.mean((y_test_s - y_pred_sev) ** 2)))
sev_mae    = float(np.mean(np.abs(y_test_s - y_pred_sev)))
mean_bias  = float(np.mean(y_pred_sev) / np.mean(y_test_s) - 1)

print(f"Severity RMSE:       £{sev_rmse:,.0f}")
print(f"Severity MAE:        £{sev_mae:,.0f}")
print(f"Mean severity bias:  {mean_bias*100:+.1f}%")
```

We evaluate severity on RMSE rather than Poisson deviance. RMSE gives an error metric in the same units as the claim amounts (pounds). For the pricing committee, a severity model that is right on average is more important than one with low RMSE at the individual level - individual claim cost is inherently unpredictable, but the portfolio average must be right.

**What the mean severity bias tells you:** A bias above 5% in either direction signals a calibration problem. If the model is predicting £3,200 mean severity when the actual mean is £3,000, every pure premium computed from this model is overstated by 7%. This is not a minor error in a book rating hundreds of thousands of policies.

Now log the severity model:

```python
with mlflow.start_run(run_name="sev_catboost_tuned") as run_sev:
    mlflow.log_params({k: v for k, v in sev_params.items() if k != "verbose"})
    mlflow.log_param("train_years",  str(sorted(df_train_sev["accident_year"].unique().to_list())))
    mlflow.log_param("test_year",    str(max_year))
    mlflow.log_param("run_date",     str(date.today()))
    mlflow.log_metric("test_rmse",   sev_rmse)
    mlflow.log_metric("test_mae",    sev_mae)
    mlflow.log_metric("mean_bias",   mean_bias)
    mlflow.catboost.log_model(sev_model, "sev_model")
    sev_run_id = run_sev.info.run_id

print(f"Severity model logged. Run ID: {sev_run_id}")
```

---

## Part 14: Comparing the GBM to the GLM

Now we answer the original question: does the GBM beat the GLM, and where?

### Fit the benchmark GLM

We need a GLM fitted on the same training data for a fair comparison. If you ran Module 2, you have one already. If not, fit a quick one here:

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm

df_glm_train = df_train_final.to_pandas()
df_glm_test  = df_test_final.to_pandas()

df_glm_train["log_exposure"] = np.log(df_glm_train["exposure_years"].clip(lower=1e-6))
df_glm_test["log_exposure"]  = np.log(df_glm_test["exposure_years"].clip(lower=1e-6))

# Convert conviction_points to string so statsmodels treats it as categorical
df_glm_train["conviction_points"] = df_glm_train["conviction_points"].astype(str)
df_glm_test["conviction_points"]  = df_glm_test["conviction_points"].astype(str)

glm_formula = (
    "claim_count ~ C(area) + ncd_years + vehicle_group + "
    "driver_age + C(conviction_points)"
)

glm_model = smf.glm(
    formula=glm_formula,
    data=df_glm_train,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_glm_train["log_exposure"],
).fit()

glm_pred_counts = glm_model.predict(df_glm_test, offset=df_glm_test["log_exposure"])
print(f"GLM fitted. Deviance: {glm_model.deviance:.1f}")
```

**Why `statsmodels` is imported here rather than at the top:** We could have imported it at the start with everything else. In this tutorial we separate it to make clear that the GLM comparison is a separate step from the CatBoost modelling. In production notebooks, put all imports at the top.

### Gini coefficient

The Gini coefficient measures discrimination: how well the model separates high-risk from low-risk policies. We use the binary AUC formulation - binary because most policies have zero claims, so the discrimination problem is effectively a binary one (will this policy claim or not?):

```python
def gini_coefficient(y_counts: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Binary discrimination Gini, based on ROC-AUC of claim/no-claim.
    Formula: Gini = 2 * AUC - 1.
    A model that discriminates perfectly scores 1.0; random guessing scores 0.0.
    """
    y_binary = (y_counts > 0).astype(int)
    auc = roc_auc_score(y_binary, y_scores)
    return float(2 * auc - 1)

# Convert count predictions to frequency predictions for comparison
gbm_freq_pred = y_pred_freq / w_test_f
glm_freq_pred = glm_pred_counts.values / w_test_f

gini_gbm = gini_coefficient(y_test_f, gbm_freq_pred)
gini_glm = gini_coefficient(y_test_f, glm_freq_pred)

print(f"Gini - GBM: {gini_gbm:.3f}")
print(f"Gini - GLM: {gini_glm:.3f}")
print(f"Lift:       {gini_gbm - gini_glm:+.3f}")
```

**Interpreting the lift:** On UK motor data, a Gini lift of 0.03-0.05 is meaningful and reproducible. A lift of 0.01-0.02 is probably within the noise of the test set. A lift of 0.06+ on a well-developed GLM is unusual - if you see this, check whether the GLM formula is missing something obvious or whether the test set has a structural difference from the training data.

Now log the Gini comparison back to the frequency model's MLflow run, so the comparison sits alongside the model:

```python
with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_metric("gini_gbm",  gini_gbm)
    mlflow.log_metric("gini_glm",  gini_glm)
    mlflow.log_metric("gini_lift", gini_gbm - gini_glm)
```

### Double lift chart

The Gini tells you how much the GBM outperforms the GLM overall. The double lift chart tells you *where* the GBM disagrees with the GLM and whether those disagreements correspond to genuine risk.

We bin the test set by decile of the ratio GBM predicted frequency / GLM predicted frequency. Within each decile, we compute the actual observed frequency. If the GBM is finding real additional signal, the deciles where the GBM predicts more than the GLM (high ratio) should show higher actual frequencies than the deciles where the GBM predicts less (low ratio). The chart should slope upward from left to right.

Create a new cell:

```python
ratio       = gbm_freq_pred / (glm_freq_pred + 1e-10)
actual_freq = y_test_f / w_test_f

n_bins    = 10
bin_edges = np.quantile(ratio, np.linspace(0, 1, n_bins + 1))
bin_idx   = np.digitize(ratio, bin_edges[1:-1])

ratio_means, actual_means, n_obs = [], [], []
for b in range(n_bins):
    mask = bin_idx == b
    if mask.sum() == 0:
        continue
    ratio_means.append(ratio[mask].mean())
    actual_means.append(actual_freq[mask].mean())
    n_obs.append(int(mask.sum()))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ratio_means, actual_means, "o-", color="steelblue", linewidth=2, label="Actual frequency")
ax.axhline(
    actual_freq.mean(),
    linestyle="--",
    color="grey",
    label=f"Portfolio mean ({actual_freq.mean():.4f})",
)
ax.set_xlabel("GBM / GLM predicted frequency ratio (by decile)")
ax.set_ylabel("Actual observed frequency")
ax.set_title("Double Lift Chart: CatBoost vs GLM - Motor Frequency")
ax.legend()
plt.tight_layout()
plt.show()
```

Run this. The chart appears in the cell output.

**Reading the chart:**
- A positively sloping line (actual frequency rises from left to right) confirms the GBM is finding real additional risk signal. The deciles where the GBM predicts more than the GLM actually are higher risk.
- A flat line means the GBM's additional complexity is not translating to better risk identification. The GLM is capturing the same information.
- A negatively sloping line means the GBM is finding spurious patterns. Do not deploy a model where this chart slopes downward.

On our synthetic data, the chart should slope upward because the DGP contains a superadditive interaction between young driver and high vehicle group that the multiplicative GLM underestimates. The GBM tree structure captures this interaction directly.

**What is happening in the top decile?** Look at the policies where the GBM predicts much higher frequency than the GLM. Run this:

```python
top_mask = bin_idx == (n_bins - 1)
print(f"Top decile: {top_mask.sum():,} policies")
print(f"Actual frequency in top decile:   {actual_freq[top_mask].mean():.4f}")
print(f"Portfolio mean actual frequency:  {actual_freq.mean():.4f}")
print(f"Ratio:                            {actual_freq[top_mask].mean() / actual_freq.mean():.2f}x portfolio")
print()

df_top = df_test_final.to_pandas().loc[top_mask]
df_all = df_test_final.to_pandas()
print(f"Mean driver age  - top decile: {df_top['driver_age'].mean():.1f}, portfolio: {df_all['driver_age'].mean():.1f}")
print(f"Mean vehicle grp - top decile: {df_top['vehicle_group'].mean():.1f}, portfolio: {df_all['vehicle_group'].mean():.1f}")
print(f"Pct age < 25     - top decile: {(df_top['driver_age'] < 25).mean():.1%}, portfolio: {(df_all['driver_age'] < 25).mean():.1%}")
```

You will find that the top decile is concentrated in young drivers in high vehicle groups - the interaction the DGP contains. This is the interaction the GLM misses and the GBM finds.

Save the chart as an MLflow artefact:

```python
fig.savefig("/tmp/double_lift.png", dpi=120, bbox_inches="tight")
with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_artifact("/tmp/double_lift.png", artifact_path="charts")
print("Double lift chart saved to MLflow.")
```

---

## Part 15: Registering the model

The last step is registering the frequency model in the MLflow Model Registry. We register it as "challenger" - not "production." The challenger alias means it is available for review, but nothing in the production pipeline loads a model by the `challenger` alias. Only a human decision - explicitly changing the alias to `production` - moves it into production. That is the governance gate.

Create a new cell:

```python
client = MlflowClient()
MODEL_NAME = "motor_freq_catboost_m03"

freq_uri   = f"runs:/{freq_run_id}/freq_model"
registered = mlflow.register_model(model_uri=freq_uri, name=MODEL_NAME)

# Use aliases (not stage transitions, which are deprecated in MLflow 2.9+)
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="challenger",
    version=registered.version,
)

# Tag with metadata for the governance review
client.set_model_version_tag(
    name=MODEL_NAME,
    version=registered.version,
    key="cv_strategy",
    value="walk_forward_ibnr1",
)
client.set_model_version_tag(
    name=MODEL_NAME,
    version=registered.version,
    key="gini_lift",
    value=str(round(gini_gbm - gini_glm, 4)),
)

print(f"Registered: {MODEL_NAME} version {registered.version} as 'challenger'")
```

**Why `mlflow.register_model` not `log_model`?** `mlflow.catboost.log_model()` logs the model as an artefact attached to the run. `mlflow.register_model()` takes that logged artefact and registers it in the Model Registry - a separate catalogue that tracks versions, aliases, and tags across runs. A model can be logged to many runs but only registered once per meaningful version.

**The `@challenger` loading syntax:** Once registered, you can load the model anywhere in the organisation with:

```python
model = mlflow.catboost.load_model("models:/motor_freq_catboost_m03@challenger")
```

This is how Module 4 will load the model for SHAP extraction without needing to retrain it.

**On stage transitions vs aliases:** Do not use `client.transition_model_version_stage()`. It is deprecated in MLflow 2.9+ and generates warnings on Databricks Runtime 14.x. Use `set_registered_model_alias()` instead, as shown above.

---

## What the pricing committee needs to see

When you present this to the pricing committee, they need two things.

**A clear statement of the evaluation framework.** Something like: "We trained on accident years 2019-2023 and tested on 2024, using the same training data version for both GLM and GBM. Walk-forward CV with an IBNR buffer of one year was used for hyperparameter tuning. The test year was held out throughout and not used in any tuning decisions."

**The double lift chart with interpretation.** Something like: "The GBM outperforms the GLM by X Gini points. The double lift chart shows the additional discrimination is concentrated in the top three deciles of the GBM/GLM ratio - policies where the GBM is predicting materially more than the GLM. Those policies have actual frequencies two to three times the portfolio average, confirming genuine risk identification. The bottom seven deciles show essentially flat actual frequencies, consistent with both models agreeing on the bulk of the book."

That is a complete and honest statement. If the Gini lift is 0.03+, you have a reasonable case for moving to Module 4 (SHAP extraction). If it is 0.01 or less, the GBM is not adding enough to justify the governance overhead.

---

## What comes next

Module 4 - SHAP Relativities. The GBM is currently a black box to the pricing committee. The challenger model is registered, the Gini lift is logged, and the double lift chart is saved to MLflow. None of that is enough for the committee to approve deployment.

Module 4 uses `shap-relativities` to extract multiplicative factor tables from the GBM. These are tables the committee can read, debate, and sign off on - the same format as the GLM relativities from Module 2. If the SHAP relativities look actuarially sensible (young drivers are expensive, high vehicle groups are expensive, high NCD is cheap), the committee can approve. If any factor table looks wrong, the committee can reject without the GBM touching the production system.

That governance gate is the point. Building an accurate GBM is straightforward. Getting it through a rigorous governance process and into production pricing is what takes expertise.
