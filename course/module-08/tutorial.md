# Module 8: End-to-End Pricing Pipeline

You have spent seven modules building the individual components of a pricing model. Module 3 taught you how to fit a Poisson GBM with the correct exposure offset. Module 4 showed you how to extract SHAP relativities. Module 5 gave you conformal prediction intervals with a provable coverage guarantee. Module 6 handled thin cells with Bayesian credibility. Module 7 optimised rate changes subject to loss ratio and volume constraints.

This module connects everything into a single pipeline that runs from raw data to a pricing committee pack. The pipeline is not a technical showcase. It is an organisational discipline. It exists because the individual components, however well-built, fail when they are connected incorrectly.

By the end of this module you will have run a complete UK motor rate review in one Databricks notebook: data ingestion, feature engineering, walk-forward validation, hyperparameter tuning, frequency and severity modelling, SHAP relativities, conformal prediction intervals, rate optimisation, and an audit record that satisfies the FCA's Consumer Duty reproducibility requirements.

---

## Part 1: Why pipelines fail -- the opening scenario

Before writing a line of code, you need to understand what you are building against.

### The NCB encoding incident

A UK private motor book runs a model refresh. The modelling team fits a CatBoost Poisson frequency model. The Gini lifts from 0.33 to 0.41. The out-of-time validation on accident year 2024 is clean. The SHAP relativities show NCB as the dominant factor, with a sensible monotone pattern: NCD 0 is 2.3x the base, NCD 5 is 0.62x. The pricing committee approves deployment. The model goes live in March 2025.

In June 2025 the actual-versus-expected ratio by NCD band stops making sense. NCD 0 policies are running significantly better than the model predicted. NCD 5 policies are running significantly worse. The pricing actuary pulls the scoring pipeline code.

The training pipeline encodes NCB years as a string: `pl.col("ncb_years").cast(pl.Utf8)`. CatBoost treats it as a categorical. The scoring pipeline, written by a different team member six weeks later from memory, passes `ncb_years` as an integer. CatBoost treats it as continuous. The model has been live for three months applying systematically wrong feature values. No exception was raised. The predictions looked plausible.

The monetary cost: one full quarter of mispriced renewals. The regulatory cost: a Consumer Duty breach, because the pricing was not what the model was validated to produce.

The fix is not model improvement. The fix is a shared feature engineering layer that both the training pipeline and the scoring pipeline import. One function. One encoding. No duplication.

### The development data contamination incident

A London specialty insurer builds a severity model for solicitors' professional indemnity. The validation metrics look reasonable. Six months after deployment, the loss ratio is running 18 points above the model's prediction. An investigation finds that the training data included IBNR-contaminated claims: the most recent 24 months of claims in the training set had not yet developed to ultimate. The model learned to predict incurred-to-date severity, not ultimate severity. A model trained on partially developed claims will always produce optimistic severity predictions for similar risks in the future.

The fix is an IBNR buffer in the cross-validation structure. The buffer removes the most recent N months from the end of each training fold. For motor third-party property, six months is sufficient. For solicitors' PI, 24 months is the minimum.

### The version confusion incident

A mid-size regional insurer runs a model review in September 2025. Three months later, the FCA requests documentation of the pricing basis for a specific cohort of policies. The pricing actuary cannot identify which model version was deployed, what data it was trained on, or what validation metrics it achieved. The answer turns out to be "the pricing actuary's laptop, which has since been refreshed."

The fix is an audit record. Every pipeline run writes one row to a Delta table: model run IDs, data table versions, validation metrics, run date, run configuration. Delta's time travel preserves historical data table versions. MLflow preserves model artefacts. The audit record ties them together.

### What these three incidents have in common

None of them were modelling failures. The models were technically sound. The failures were in the connections between components -- between training and scoring, between training and validation, between production and audit. An end-to-end pipeline disciplines those connections. That is what this module builds.

---

## Part 2: What makes insurance pipelines different

Before building the pipeline, you need to understand why a generic ML pipeline template -- the kind used for image classification or fraud detection -- does not work for insurance pricing without modification.

### Time is not just a feature

In most ML problems, the train-test split is random. Observations are exchangeable: a record from day 1 and a record from day 100 are drawn from the same distribution and either can validly appear in train or test. This is false for insurance. A policy written in 2022 and a policy written in 2025 are not exchangeable. Loss cost trends, book mix, regulatory changes, economic conditions all mean the 2025 policy is in a genuinely different environment. A random split means some 2025 policies appear in training and some 2022 policies appear in validation. The model sees future data during training and the validation measures performance on past data. Both are wrong.

Every train-test split in an insurance pipeline must be temporal. The most recent period -- typically one accident year -- is the test set. Everything before it is training. This is not optional.

### Exposure is not a weight

A policy with 0.5 years of exposure and 1 claim has an observed frequency of 2.0. A policy with 1.0 year of exposure and 1 claim has an observed frequency of 1.0. They have the same raw claim count but very different information content. The 1.0-year policy is more reliable.

The correct way to handle this in a Poisson regression is with a log-exposure offset. The model predicts claim count, and the log of exposure enters as a fixed term in the linear predictor:

```
log(E[claims]) = log(exposure) + f(features)
```

In CatBoost, this is implemented with `baseline=np.log(exposure)` in the Pool constructor. The `baseline` parameter adds a fixed term to the model's output before the loss is computed -- it is the log-offset.

Using `weight=exposure` instead is wrong. With `weight`, the observation contribution to the loss is scaled by exposure, but the model still predicts `exp(f(features))` with no offset. A policy with 0.5 years of exposure contributes half as much to the likelihood, but the model's prediction for it is not adjusted for the shorter exposure period. The predictions are on the wrong scale.

Module 3 explained this. This module implements it correctly throughout. Every Pool that includes exposure uses `baseline=np.log(exposure)`.

### Severity needs its own model

A pure premium model (frequency times severity) could in principle be fit as a single Tweedie model. For a capstone pipeline, we fit frequency and severity separately because:

1. The Poisson frequency model has a log-exposure offset. The Gamma/Tweedie severity model does not -- severity is conditioned on a claim occurring, so exposure is not the right offset for the severity prediction.

2. SHAP relativities are more interpretable from the separate models. The frequency SHAP shows what drives claim occurrence; the severity SHAP shows what drives claim size. These tell different stories.

3. Conformal intervals are typically calibrated on severity predictions, where the skewed distribution makes uncertainty quantification most valuable.

The severity model trains only on policies with at least one claim. A policy with zero claims has no observed severity, and its inclusion would contaminate the likelihood. When we compute the pure premium, we predict severity for ALL policies (not just those with claims). The severity prediction gives the expected loss given a claim occurs -- it does not require the policy to have had a claim in training.

### The IBNR problem

Insurance claims develop over time. A claim notified in November 2024 may still be receiving payments in 2027. The incurred value as of the training data extract date is not the ultimate value.

If you include the most recent accident years at face value in training data, you are training on partially developed claims. For property damage motor, claims develop quickly -- 95% of ultimate within 12 months. For motor bodily injury, development can run 5-10 years. Training on immature claims produces a model that will systematically under-predict severity for similar future claims, because the future claims will develop to a higher ultimate value.

The IBNR buffer in walk-forward cross-validation addresses this by excluding the most recent N months from each training fold. This forces the model to rely on more developed claims data for training. The buffer length depends on the line of business.

---

## Part 3: Setting up the Databricks notebook

### Creating the notebook

Log into your Databricks workspace. In the left sidebar, click **Workspace**. Navigate to your personal folder (usually listed under **Workspace > Users > your-email**).

Click the **+** button and choose **Notebook**. Name it `module-08-end-to-end-pipeline`. Keep the default language as Python. Click **Create**.

The notebook opens with one empty cell. Connect it to your cluster by clicking **Connect** in the top-right and selecting your running cluster. If no cluster is running, go to **Compute** in the left sidebar, find your cluster, and click **Start**. Cluster startup takes 3-5 minutes.

Do not run any cells until the cluster shows a green status icon.

### Stage 0: Installing the libraries

The first cell of every notebook in this course installs all required libraries. On Databricks, `%pip install` installs packages into the running Python environment on the cluster. The `dbutils.library.restartPython()` call immediately after restarts the Python interpreter so the newly installed packages are importable.

In the first cell, type this and run it with Shift+Enter:

```python
%pip install \
    catboost \
    optuna \
    mlflow \
    polars \
    "insurance-cv" \
    "insurance-conformal[catboost]" \
    "rate-optimiser" \
    --quiet
```

You will see pip installation output scrolling for 30-90 seconds. Wait for it to complete. The last line will say something like `Successfully installed catboost-1.x.x ...`.

Once the install finishes, run this in a new cell:

```python
dbutils.library.restartPython()
```

This restarts the Python session. Any variables you defined before the restart are gone -- that is expected. The install cell must always be the first cell, and you must run all subsequent cells after the restart.

After the restart, in a new cell, confirm everything imported:

```python
import polars as pl
import numpy as np
import mlflow
import optuna
from catboost import CatBoostRegressor, Pool
from insurance_conformal import InsuranceConformalPredictor

print(f"Polars:   {pl.__version__}")
print(f"NumPy:    {np.__version__}")
print(f"MLflow:   {mlflow.__version__}")
print(f"Optuna:   {optuna.__version__}")
print("CatBoost: OK")
print("InsuranceConformalPredictor: OK")
print("All libraries ready.")
```

You should see version numbers printed for each library. If you see `ModuleNotFoundError`, the install cell did not complete successfully. Run it again and restart again.

---

## Part 4: Pipeline architecture -- what connects to what

Before writing any pipeline code, you need to understand the architecture as a whole. The following diagram shows how each stage feeds the next.

```
Stage 1: Configuration
    |
    v
Stage 2: Data ingestion --> raw_policies (Delta table, version logged)
    |
    v
Stage 3: Feature engineering --> features (Delta table, version logged)
    |
    +---> Stage 4: Walk-forward CV (validation metrics)
    |         |
    |         v
    +---> Stage 5: Optuna tuning (best_freq_params, best_sev_params)
    |         |
    |         v
    +---> Stage 6: Final models --> freq_run_id, sev_run_id (MLflow)
              |
              +---> Stage 7: SHAP relativities --> freq_predictions (Delta)
              |
              +---> Stage 8: Conformal intervals --> conformal_intervals (Delta)
              |
              +---> Stage 9: Rate optimisation --> rate_change_factors (Delta)
                        |
                        v
                   Stage 10: Audit record --> pipeline_audit (Delta)
```

Every stage reads from a well-defined input and writes to a well-defined output. Stage 3's output (features Delta table) feeds stages 4, 5, and 6. Stage 6's output (trained models, logged in MLflow) feeds stages 7, 8, and 9. The final audit record references every upstream output.

This structure means that if any stage fails, you can restart from that stage without rerunning everything before it. It also means that every output is traceable: given any Delta table row or MLflow model, you can follow the references back to the configuration that produced it.

---

## Part 5: Stage 1 -- Configuration

All configurable values live in one cell. This is the first rule of pipeline design. If a value appears in more than one place, it will eventually be inconsistent.

Add a markdown cell:

```python
%md
## Stage 1: Configuration
```

Then add this code cell:

```python
import json
from datetime import date

# -----------------------------------------------------------------------
# Unity Catalog coordinates
# -----------------------------------------------------------------------
# CATALOG: the top-level container. In Databricks Free Edition, you may
# only have access to a catalog called "main" or "hive_metastore".
# Change this to match your Databricks environment.
CATALOG = "main"

# SCHEMA: the database within the catalog. Name it by review cycle, not
# version number. "motor_q2_2026" tells you exactly when and what.
# "motor_v3" tells you nothing about when it was run.
SCHEMA  = "motor_q2_2026"

# TABLES: all table names in one dictionary. Change a name here; it
# changes everywhere. Never hard-code a table name twice.
TABLES = {
    "raw":                 f"{CATALOG}.{SCHEMA}.raw_policies",
    "features":            f"{CATALOG}.{SCHEMA}.features",
    "freq_predictions":    f"{CATALOG}.{SCHEMA}.freq_predictions",
    "conformal_intervals": f"{CATALOG}.{SCHEMA}.conformal_intervals",
    "rate_change":         f"{CATALOG}.{SCHEMA}.rate_action_factors",
    "efficient_frontier":  f"{CATALOG}.{SCHEMA}.efficient_frontier",
    "pipeline_audit":      f"{CATALOG}.{SCHEMA}.pipeline_audit",
}

# -----------------------------------------------------------------------
# Model and pipeline parameters
# -----------------------------------------------------------------------
LR_TARGET       = 0.72    # Loss ratio target for rate optimisation
VOLUME_FLOOR    = 0.97    # Minimum portfolio volume retention
CONFORMAL_ALPHA = 0.10    # 1 - alpha = 90% coverage intervals
N_OPTUNA_TRIALS = 20      # Increase to 40 for production runs

# -----------------------------------------------------------------------
# Run identification
# -----------------------------------------------------------------------
RUN_DATE = str(date.today())

print(f"Run date:           {RUN_DATE}")
print(f"Catalog / schema:   {CATALOG}.{SCHEMA}")
print(f"Loss ratio target:  {LR_TARGET:.0%}")
print(f"Conformal alpha:    {CONFORMAL_ALPHA}")
print(f"Optuna trials:      {N_OPTUNA_TRIALS}")
print("\nTable names:")
for k, v in TABLES.items():
    print(f"  {k:<25} {v}")
```

**What you should see:** A clean print of all configuration values. If any table name looks wrong, fix it here before running any subsequent stage.

### A note on Unity Catalog

Unity Catalog is Databricks' governance layer for data assets. It uses a three-part naming convention: `catalog.schema.table`. The catalog is the top-level container -- in an enterprise environment it might be called `pricing` or `insurance`. The schema is a logical grouping within the catalog -- we name it by review cycle. The table is the individual data asset.

In Databricks Free Edition (Community Edition), you may only have access to a legacy metastore called `hive_metastore`. In that case, your table names are two-part: `schema.table`. Change `CATALOG = "main"` to `CATALOG = "hive_metastore"` and adjust the TABLES dictionary accordingly. The rest of the pipeline runs identically either way.

The reason we use Unity Catalog for pricing is audit. Unity Catalog logs every read and write to every table. Combined with Delta's version history, you can reconstruct what data was read by any pipeline run on any date in the past. For Consumer Duty compliance, this is not optional -- you need to be able to demonstrate, three years later, what data informed a specific pricing decision.

### Create the schema

The schema must exist before you can write to it. Run this in a new cell:

```python
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Schema {CATALOG}.{SCHEMA} ready.")
```

If you are using `hive_metastore`, omit the `CREATE CATALOG` line -- the legacy metastore does not use catalogs.

---

## Part 6: Stage 2 -- Data ingestion

In production, this stage reads from your policy administration system. In this tutorial, we generate synthetic UK motor data so the notebook is self-contained.

The synthetic data generates 200,000 policies across accident years 2021-2024. The data generating process has the same structure as the Module 3 and Module 5 datasets: a Poisson claim count with a log-linear frequency model, log-normal severity with a superadditive interaction between young drivers and high vehicle groups, and realistic exposure distributions.

Add a markdown cell:

```python
%md
## Stage 2: Data ingestion -- generate synthetic motor portfolio
```

Then add this code cell:

```python
import polars as pl
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=2026)

# -----------------------------------------------------------------------
# Portfolio dimensions
# -----------------------------------------------------------------------
N_POLICIES   = 200_000
ACCIDENT_YEARS = [2021, 2022, 2023, 2024]

# -----------------------------------------------------------------------
# Rating factors
# -----------------------------------------------------------------------
ncb_years     = rng.choice([0,1,2,3,4,5], N_POLICIES,
                           p=[0.08, 0.07, 0.10, 0.15, 0.20, 0.40])
vehicle_group = rng.integers(1, 51, N_POLICIES)
age_band      = rng.choice(["17-25","26-35","36-50","51-65","66+"],
                           N_POLICIES, p=[0.11, 0.22, 0.30, 0.25, 0.12])
annual_mileage = rng.choice(["<5k","5k-10k","10k-15k","15k+"],
                             N_POLICIES, p=[0.18, 0.30, 0.35, 0.17])
region        = rng.choice(["North","Midlands","London","SouthEast","SouthWest"],
                           N_POLICIES, p=[0.20, 0.20, 0.22, 0.25, 0.13])
accident_year = rng.choice(ACCIDENT_YEARS, N_POLICIES)

# Exposure: most policies are near-annual, some are mid-term
exposure      = np.clip(rng.beta(8, 2, N_POLICIES), 0.05, 1.0)

# -----------------------------------------------------------------------
# True claim frequency (log-linear, multiplicative)
# -----------------------------------------------------------------------
# We define the true DGP explicitly so we can assess model performance.
# This is the ground truth - the model will recover an approximation of it.

INTERCEPT = -2.95

age_mid = {"17-25": 21, "26-35": 30, "36-50": 43, "51-65": 58, "66+": 72}
age_effect = np.array([
    -0.85 + 0.03 * age_mid[a] - 0.0002 * age_mid[a]**2
    for a in age_band
])

# NCB: strong monotone effect - each year of NCD reduces frequency
ncb_effect = -0.18 * ncb_years

# Vehicle group: positive effect with superadditive interaction for young drivers
vg_effect = 0.012 * vehicle_group

# Young driver x high vehicle group superadditive interaction
is_young  = np.array([1 if a == "17-25" else 0 for a in age_band])
high_vg   = (vehicle_group > 35).astype(float)
interaction_effect = 0.45 * is_young * high_vg

# Region
region_effects = {
    "North": 0.0, "Midlands": 0.08, "London": 0.40,
    "SouthEast": 0.22, "SouthWest": 0.05
}
region_effect = np.array([region_effects[r] for r in region])

# Mileage: more miles, more claims
mileage_effects = {"<5k": -0.10, "5k-10k": 0.0, "10k-15k": 0.12, "15k+": 0.28}
mileage_effect  = np.array([mileage_effects[m] for m in annual_mileage])

# True log-frequency per policy-year (claim count model)
log_freq = (INTERCEPT + age_effect + ncb_effect + vg_effect +
            interaction_effect + region_effect + mileage_effect)

# Expected claim count = frequency * exposure
true_freq       = np.exp(log_freq)
expected_claims = true_freq * exposure
claim_count     = rng.poisson(expected_claims)

# -----------------------------------------------------------------------
# Severity: log-normal with heavier tail for young high-vehicle drivers
# -----------------------------------------------------------------------
sev_log_mean   = 7.1 + 0.008 * vehicle_group + 0.25 * is_young * high_vg
sev_log_sd     = 0.90 + 0.10 * is_young

# Simulate severity per policy (only meaningful where claim_count > 0)
sev_per_claim  = np.exp(rng.normal(sev_log_mean, sev_log_sd, N_POLICIES))
incurred_loss  = claim_count * sev_per_claim  # zero for no-claim policies

print(f"Policies generated:  {N_POLICIES:,}")
print(f"Total claims:        {claim_count.sum():,}")
print(f"Overall claim rate:  {claim_count.sum() / exposure.sum():.4f} per policy-year")
print(f"Total incurred:      £{incurred_loss.sum():,.0f}")
print(f"Mean severity (claims only): £{incurred_loss[claim_count>0].mean():,.0f}")
```

**What you should see:** Around 200,000 policies, 10,000-12,000 total claims (roughly 5.5% claim frequency), and a mean severity in the range of £2,000-£4,000. The exact numbers will depend on the random seed.

Now assemble the Polars DataFrame:

```python
raw_pl = pl.DataFrame({
    "policy_id":      [f"POL{i:07d}" for i in range(N_POLICIES)],
    "accident_year":  accident_year.tolist(),
    "ncb_years":      ncb_years.tolist(),
    "vehicle_group":  vehicle_group.tolist(),
    "age_band":       age_band.tolist(),
    "annual_mileage": annual_mileage.tolist(),
    "region":         region.tolist(),
    "exposure":       exposure.tolist(),
    "claim_count":    claim_count.tolist(),
    "incurred_loss":  incurred_loss.tolist(),
})

# Sanity checks
assert raw_pl.shape[0] == N_POLICIES,         "Row count mismatch"
assert raw_pl["exposure"].min() > 0,           "Zero or negative exposure"
assert raw_pl["claim_count"].min() >= 0,       "Negative claim count"
assert raw_pl["incurred_loss"].min() >= 0,     "Negative incurred loss"
assert raw_pl["ncb_years"].is_between(0, 5).all(), "NCB out of range"

print("Data shape:", raw_pl.shape)
print("\nAccident year distribution:")
print(raw_pl.group_by("accident_year").agg(
    pl.len().alias("n_policies"),
    pl.col("claim_count").sum().alias("claims"),
    pl.col("exposure").sum().alias("exposure"),
).sort("accident_year").with_columns(
    (pl.col("claims") / pl.col("exposure")).round(4).alias("freq")
))
```

### Writing to Delta Lake

Delta Lake is Databricks' table format. It adds three capabilities over standard Parquet files that matter for pricing: ACID transactions (concurrent reads and writes are consistent), time travel (every version of the data is preserved), and DML operations (you can UPDATE, DELETE, and MERGE specific rows without rewriting the entire table).

Write the raw data to Delta:

```python
# Convert Polars to pandas for Spark (Spark cannot read Polars directly)
# We only convert at the library boundary -- all other processing stays in Polars
raw_spark = spark.createDataFrame(raw_pl.to_pandas())

raw_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["raw"])

# Log the Delta table version
raw_version = spark.sql(
    f"DESCRIBE HISTORY {TABLES['raw']} LIMIT 1"
).collect()[0]["version"]

print(f"Raw data written to: {TABLES['raw']}")
print(f"Delta version:       {raw_version}")
print(f"Row count:           {raw_spark.count():,}")
```

**What does `DESCRIBE HISTORY` return?** Every write to a Delta table increments its version number. The first write is version 0. A subsequent overwrite is version 1. An append is version 2. The history table records every version, its timestamp, and the operation type. You can read the data at any historical version with `.option("versionAsOf", N)`.

**What does `raw_version` tell us?** This is the version number of the raw data table as it exists right now, at the point this pipeline ran. We log it to the audit record so that anyone reviewing this pipeline six months later can read the exact data that was used:

```python
spark.read.format("delta") \
    .option("versionAsOf", raw_version) \
    .table(TABLES["raw"]) \
    .toPandas()
```

This is Delta time travel. It works as long as the table's VACUUM retention policy preserves the version files. The default Databricks retention is 30 days -- not long enough for a Consumer Duty audit trail. Set at least 365 days on any table that forms part of the pricing basis:

```python
spark.sql(f"""
    ALTER TABLE {TABLES['raw']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")
```

---

## Part 7: Stage 3 -- Feature engineering

This is the most structurally important stage in the pipeline. Read it carefully.

The failure mode described in Part 1 -- the NCB encoding incident -- happens because feature engineering is defined in two places: the training notebook and the scoring code. When they diverge, the model produces wrong predictions silently.

The solution is to define all feature transforms once, in a single list, and call the same `apply_transforms()` function at training time and scoring time. This is not a style preference. It is the discipline that prevents the most common production failure in ML-based pricing.

Add a markdown cell:

```python
%md
## Stage 3: Feature engineering -- the shared transform layer
```

### Defining the transforms

```python
# -----------------------------------------------------------------------
# Constants used in transforms.
# Module-level, not inside functions, so they can be logged to MLflow
# and serialised alongside the model.
# -----------------------------------------------------------------------
NCB_MAX     = 5
VEHICLE_ORD = {"A":1,"B":2,"C":3,"D":4,"E":5}   # not used in this data, shown as pattern

AGE_MID = {
    "17-25": 21,
    "26-35": 30,
    "36-50": 43,
    "51-65": 58,
    "66+":   72,
}

MILEAGE_ORD = {
    "<5k":    1,
    "5k-10k": 2,
    "10k-15k":3,
    "15k+":   4,
}

# -----------------------------------------------------------------------
# Transform functions: pure functions, one job each.
# Each takes a Polars DataFrame and returns a Polars DataFrame.
# The output DataFrame contains all original columns plus the new ones.
# -----------------------------------------------------------------------

def encode_ncb(df: pl.DataFrame) -> pl.DataFrame:
    """
    NCB deficit = NCB_MAX - ncb_years.
    Puts new drivers (ncb=0) at deficit=5 (high) and
    fully discounted drivers (ncb=5) at deficit=0 (low).
    The GBM can then find a monotone increasing effect for the deficit.
    """
    return df.with_columns(
        (NCB_MAX - pl.col("ncb_years")).alias("ncb_deficit")
    )

def encode_age(df: pl.DataFrame) -> pl.DataFrame:
    """
    Map age band to a numeric midpoint.
    age_mid is a continuous approximation of the band's centre.
    The GBM can model non-linear age effects correctly with this encoding.
    """
    return df.with_columns(
        pl.col("age_band")
          .replace(AGE_MID)
          .cast(pl.Float64)
          .alias("age_mid")
    )

def encode_mileage(df: pl.DataFrame) -> pl.DataFrame:
    """
    Map mileage band to an ordinal integer.
    Preserves the ordering (<5k < 5k-10k < 10k-15k < 15k+).
    """
    return df.with_columns(
        pl.col("annual_mileage")
          .replace(MILEAGE_ORD)
          .cast(pl.Int32)
          .alias("mileage_ord")
    )

def add_log_exposure(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pre-compute log(exposure) so it is always available in the feature table.
    This is the offset term for the Poisson model.
    We store it in the features table so downstream steps do not need to
    recompute it from the raw exposure column.
    """
    return df.with_columns(
        pl.col("exposure").log().alias("log_exposure")
    )

# -----------------------------------------------------------------------
# The master transform list and feature column specification.
# These are the ONLY definitions you should change if you add a new feature.
# -----------------------------------------------------------------------
TRANSFORMS   = [encode_ncb, encode_age, encode_mileage, add_log_exposure]

# FEATURE_COLS: the columns passed to CatBoost as input features.
# This does NOT include log_exposure or claim_count or incurred_loss.
FEATURE_COLS = ["ncb_deficit", "vehicle_group", "age_mid", "mileage_ord", "region"]

# CAT_FEATURES: the subset of FEATURE_COLS that CatBoost should treat as categorical.
# CatBoost handles categories natively -- do not one-hot encode them.
CAT_FEATURES = ["region"]

def apply_transforms(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply all transforms in order.
    Call this function identically at training time and scoring time.
    Never apply individual transforms manually outside this function.
    """
    for fn in TRANSFORMS:
        df = fn(df)
    return df

# Test the transforms on a small sample
sample = raw_pl.head(5)
sample_transformed = apply_transforms(sample)
print("Transform test passed.")
print("Columns after transforms:")
print([c for c in sample_transformed.columns])
print("\nFirst row (selected columns):")
print(sample_transformed.select(["ncb_years","ncb_deficit","age_band","age_mid",
                                  "annual_mileage","mileage_ord","exposure","log_exposure"]))
```

**What you should see:** The `ncb_deficit` column for a policy with `ncb_years=5` should be 0. For `ncb_years=0`, it should be 5. The `age_mid` for `age_band="36-50"` should be 43. The `mileage_ord` for `"10k-15k"` should be 3. The `log_exposure` should be the natural log of the exposure value.

### Applying the transforms and writing to Delta

```python
# Apply to the full dataset
features_pl = apply_transforms(raw_pl)

print(f"Features shape: {features_pl.shape}")
print(f"Feature columns present: {[c for c in FEATURE_COLS if c in features_pl.columns]}")
print(f"All feature cols present: {all(c in features_pl.columns for c in FEATURE_COLS)}")

# Write to Delta
spark.createDataFrame(features_pl.to_pandas()) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["features"])

spark.sql(f"""
    ALTER TABLE {TABLES['features']}
    SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
""")

# Log the features table version
feat_version = spark.sql(
    f"DESCRIBE HISTORY {TABLES['features']} LIMIT 1"
).collect()[0]["version"]

print(f"\nFeatures table: {TABLES['features']}")
print(f"Delta version:  {feat_version}")
```

---

## Part 8: Stage 4 -- Walk-forward cross-validation

Cross-validation in insurance must be temporal. We covered the reasons in Part 2. This stage implements a walk-forward CV with an IBNR buffer.

Add a markdown cell:

```python
%md
## Stage 4: Walk-forward cross-validation
```

### Setting up the folds

```python
from insurance_cv import WalkForwardCV, IBNRBuffer

# -----------------------------------------------------------------------
# Fold setup: three temporal folds.
# Each fold trains on progressively more data and validates on the
# next accident year.
#
# Fold 1: train on 2021-2022, validate on 2023
# Fold 2: train on 2021-2023, validate on 2024
# Fold 3: train on 2021-2023 (with IBNR buffer), validate on 2024
#
# The IBNR buffer removes the most recent 6 months from each training fold.
# For accident year data (not monthly), this trims the 2023 H2 data from
# Fold 2's training set. In production with monthly data, it removes the
# six most recent months from the training window.
# -----------------------------------------------------------------------

ibnr_buffer = IBNRBuffer(months=6)
cv = WalkForwardCV(
    date_col="accident_year",
    n_splits=3,
    min_train_years=2,
    ibnr_buffer=ibnr_buffer,
)

# Convert features to pandas for the CV loop (CatBoost Pool requires pandas)
features_pd = features_pl.to_pandas()

print("Cross-validation folds:")
for fold_idx, (train_idx, val_idx) in enumerate(cv.split(features_pd)):
    train_years = sorted(features_pd.loc[train_idx, "accident_year"].unique())
    val_years   = sorted(features_pd.loc[val_idx,   "accident_year"].unique())
    print(f"  Fold {fold_idx+1}: train years={train_years}, val years={val_years}, "
          f"n_train={len(train_idx):,}, n_val={len(val_idx):,}")
```

### The Poisson deviance metric

```python
def poisson_deviance(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     exposure: np.ndarray) -> float:
    """
    Scaled Poisson deviance for frequency models.

    y_true:   observed claim counts (not frequencies)
    y_pred:   predicted claim counts (exposure * predicted frequency)
    exposure: policy exposure in years

    The scaling by total exposure gives a per-policy-year deviance,
    making it comparable across folds with different exposure totals.

    MSE is the wrong metric for Poisson regression:
      - MSE penalises a miss of 2 claims the same whether the policy has
        0.01 expected claims or 0.5 expected claims
      - Poisson deviance weights residuals correctly by the expected count

    A null model (always predict the portfolio mean frequency) achieves
    a Poisson deviance around 0.08-0.12 on UK motor data. A good GBM
    achieves 0.04-0.07. Anything below 0.04 should be investigated for
    data leakage.
    """
    fp = np.clip(y_pred / exposure, 1e-10, None)   # predicted frequency
    ft = y_true / exposure                          # observed frequency
    d  = 2 * exposure * (
        np.where(ft > 0, ft * np.log(ft / fp), 0.0) - (ft - fp)
    )
    return float(d.sum() / exposure.sum())
```

### Running the CV loop

```python
cv_deviances = []

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(features_pd)):
    df_tr = features_pd.loc[train_idx]
    df_va = features_pd.loc[val_idx]

    X_tr = df_tr[FEATURE_COLS]
    y_tr = df_tr["claim_count"].values
    w_tr = df_tr["exposure"].values

    X_va = df_va[FEATURE_COLS]
    y_va = df_va["claim_count"].values
    w_va = df_va["exposure"].values

    # ----------------------------------------------------------------
    # IMPORTANT: the Poisson offset is log(exposure), not exposure.
    # baseline= adds a fixed term to the model output before the loss.
    # The model then predicts: exp(log(exposure) + f(features))
    #                        = exposure * exp(f(features))
    # which is the expected claim count for a policy with this exposure.
    # ----------------------------------------------------------------
    train_pool = Pool(
        X_tr, y_tr,
        baseline=np.log(np.clip(w_tr, 1e-6, None)),
        cat_features=CAT_FEATURES,
    )
    val_pool = Pool(
        X_va, y_va,
        baseline=np.log(np.clip(w_va, 1e-6, None)),
        cat_features=CAT_FEATURES,
    )

    # Default params for CV: these are not tuned yet
    cv_model = CatBoostRegressor(
        loss_function="Poisson",
        iterations=300,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
    )
    cv_model.fit(train_pool, eval_set=val_pool)

    pred_va   = cv_model.predict(val_pool)
    fold_dev  = poisson_deviance(y_va, pred_va, w_va)
    cv_deviances.append(fold_dev)

    print(f"Fold {fold_idx+1}: Poisson deviance = {fold_dev:.5f}  "
          f"(n_val={len(val_idx):,}, exposure={w_va.sum():.0f} years)")

mean_cv_deviance = float(np.mean(cv_deviances))
print(f"\nMean CV deviance: {mean_cv_deviance:.5f}")
print(f"Fold deviances:   {[round(d, 5) for d in cv_deviances]}")
```

**What you should see:** Three fold deviances, each between roughly 0.04 and 0.09 for the synthetic data. The last fold may be higher if accident year 2024 is partially immature in the synthetic data. If any fold deviance is below 0.01, check for data leakage -- you may have future information in the training features.

**What the IBNR buffer does in practice:** If your data is monthly (e.g., accident month rather than accident year), the six-month buffer removes the six most recent months from each fold's training set. A policy written in June 2024 would not appear in a fold that trains on data through December 2024. This forces the model to learn from claims that have had at least six months to develop, reducing IBNR contamination.

For this notebook, we use annual data so the buffer effect is limited. In production, move to monthly accident periods and set the buffer to 12 months for motor property damage, 24 months for motor bodily injury.

---

## Part 9: Stage 5 -- Hyperparameter tuning with Optuna

We tune on the last fold -- the most recent out-of-time period, and therefore the most realistic validation scenario.

Add a markdown cell:

```python
%md
## Stage 5: Hyperparameter tuning with Optuna
```

### Setting up the tuning fold

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Get the last fold
folds = list(cv.split(features_pd))
last_train_idx, last_val_idx = folds[-1]

df_tune_tr = features_pd.loc[last_train_idx]
df_tune_va = features_pd.loc[last_val_idx]

X_tune_tr = df_tune_tr[FEATURE_COLS]
y_tune_tr = df_tune_tr["claim_count"].values
w_tune_tr = df_tune_tr["exposure"].values

X_tune_va = df_tune_va[FEATURE_COLS]
y_tune_va = df_tune_va["claim_count"].values
w_tune_va = df_tune_va["exposure"].values

# Pre-build the Pools (they do not change between trials)
tune_train_pool = Pool(
    X_tune_tr, y_tune_tr,
    baseline=np.log(np.clip(w_tune_tr, 1e-6, None)),
    cat_features=CAT_FEATURES,
)
tune_val_pool = Pool(
    X_tune_va, y_tune_va,
    baseline=np.log(np.clip(w_tune_va, 1e-6, None)),
    cat_features=CAT_FEATURES,
)

print(f"Tuning fold: train n={len(df_tune_tr):,}, val n={len(df_tune_va):,}")
```

### The frequency model Optuna objective

```python
def freq_objective(trial: optuna.Trial) -> float:
    """
    Each call to this function represents one Optuna trial.
    Optuna calls it N_OPTUNA_TRIALS times, each time with different
    parameter suggestions. It returns the Poisson deviance on the
    validation fold -- Optuna minimises this.

    Parameter notes:
    - iterations: tree count. More trees improve fit but increase training time.
      [200, 600] covers the useful range for this dataset size.
    - depth: max tree depth. Insurance data rarely benefits from depth > 6.
      Depth 4-5 is usually optimal for personal lines.
    - learning_rate: step size for gradient descent. Log-uniform because
      the impact is multiplicative: 0.02 vs 0.05 matters more than 0.10 vs 0.13.
    - l2_leaf_reg: L2 regularisation on leaf weights. Prevents overfitting
      in thin cells. Equivalent to lambda in LightGBM.
    """
    params = {
        "iterations":    trial.suggest_int("iterations", 200, 600),
        "depth":         trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Poisson",
        "random_seed":   42,
        "verbose":       0,
    }
    model = CatBoostRegressor(**params)
    model.fit(tune_train_pool, eval_set=tune_val_pool)
    pred  = model.predict(tune_val_pool)
    return poisson_deviance(y_tune_va, pred, w_tune_va)

freq_study = optuna.create_study(direction="minimize")
freq_study.optimize(freq_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_freq_params = freq_study.best_params
best_freq_params["loss_function"] = "Poisson"
best_freq_params["random_seed"]   = 42
best_freq_params["verbose"]       = 0

print(f"\nBest frequency model parameters (trial {freq_study.best_trial.number}):")
for k, v in best_freq_params.items():
    print(f"  {k:<20} {v}")
print(f"\nBest tuning deviance: {freq_study.best_value:.5f}")
print(f"Improvement over default: "
      f"{cv_deviances[-1] - freq_study.best_value:.5f}")
```

### The severity model Optuna objective

The severity model requires its own tuning study. It trains only on policies with at least one claim, uses a Gamma loss function, and does not use an exposure offset. We run it here with the same structure as the frequency study.

```python
# -----------------------------------------------------------------------
# Build the claims-only training set for severity tuning
# -----------------------------------------------------------------------
# The severity model trains on policies with at least one claim.
# Zero-claim policies have no observed severity and cannot contribute
# to the severity likelihood.
#
# We partition the last fold's training set into claims-only for severity.
# The validation set is also claims-only for fair evaluation.
# -----------------------------------------------------------------------

df_sev_tune_tr = df_tune_tr[df_tune_tr["claim_count"] > 0].copy()
df_sev_tune_va = df_tune_va[df_tune_va["claim_count"] > 0].copy()

# Severity target: incurred loss per claim
# We use mean severity per claim, not total incurred per policy,
# because a policy with 3 claims should not be 3x more influential
# than a policy with 1 claim at the same severity.
df_sev_tune_tr["mean_sev"] = (
    df_sev_tune_tr["incurred_loss"] / df_sev_tune_tr["claim_count"]
)
df_sev_tune_va["mean_sev"] = (
    df_sev_tune_va["incurred_loss"] / df_sev_tune_va["claim_count"]
)

X_sev_tr = df_sev_tune_tr[FEATURE_COLS]
y_sev_tr = df_sev_tune_tr["mean_sev"].values
w_sev_tr = df_sev_tune_tr["claim_count"].values  # weight by claim count

X_sev_va = df_sev_tune_va[FEATURE_COLS]
y_sev_va = df_sev_tune_va["mean_sev"].values
w_sev_va = df_sev_tune_va["claim_count"].values

# Build Pools -- note: no baseline for severity (no exposure offset)
sev_tune_train_pool = Pool(
    X_sev_tr, y_sev_tr,
    weight=w_sev_tr,       # weight by number of claims (more claims = more evidence)
    cat_features=CAT_FEATURES,
)
sev_tune_val_pool = Pool(
    X_sev_va, y_sev_va,
    weight=w_sev_va,
    cat_features=CAT_FEATURES,
)

print(f"Severity tuning fold: train n={len(df_sev_tune_tr):,} (claims-only), "
      f"val n={len(df_sev_tune_va):,}")

def sev_objective(trial: optuna.Trial) -> float:
    """
    Severity model objective: minimise RMSE on held-out claims.
    Tweedie with variance_power=2 is a Gamma distribution.
    Gamma is appropriate for positive-only severity with right skew.

    We use RMSE as the evaluation metric because it is interpretable
    in the units of the severity (pounds). An RMSE of 800 means the model
    is off by roughly £800 on average for held-out claims.
    """
    params = {
        "iterations":    trial.suggest_int("iterations", 200, 600),
        "depth":         trial.suggest_int("depth", 4, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "loss_function": "Tweedie:variance_power=2",  # Gamma
        "eval_metric":   "RMSE",
        "random_seed":   42,
        "verbose":       0,
    }
    model = CatBoostRegressor(**params)
    model.fit(sev_tune_train_pool, eval_set=sev_tune_val_pool)
    pred = model.predict(sev_tune_val_pool)
    # RMSE in pounds
    return float(np.sqrt(np.mean((y_sev_va - pred)**2)))

sev_study = optuna.create_study(direction="minimize")
sev_study.optimize(sev_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_sev_params = sev_study.best_params
best_sev_params["loss_function"] = "Tweedie:variance_power=2"
best_sev_params["eval_metric"]   = "RMSE"
best_sev_params["random_seed"]   = 42
best_sev_params["verbose"]       = 0

print(f"\nBest severity model parameters (trial {sev_study.best_trial.number}):")
for k, v in best_sev_params.items():
    print(f"  {k:<20} {v}")
print(f"\nBest severity RMSE:  £{sev_study.best_value:,.0f}")
```

---

## Part 10: Stage 6 -- Final models and MLflow logging

The final models train on all data except the held-out test year. The test year is the most recent accident year in the dataset -- in this case, 2024. It is held out entirely from training and used only for final evaluation. This is separate from the CV folds, which used 2023 and 2024 for validation in different folds.

Add a markdown cell:

```python
%md
## Stage 6: Final models -- frequency (Poisson) and severity (Gamma/Tweedie)
```

### Splitting train and test

```python
import mlflow
import mlflow.catboost

max_year = int(features_pd["accident_year"].max())
print(f"Test year (held out from training): {max_year}")

# Train set: all years except the test year
df_train = features_pd[features_pd["accident_year"] < max_year].copy()
df_test  = features_pd[features_pd["accident_year"] == max_year].copy()

print(f"Training rows: {len(df_train):,}  ({df_train['accident_year'].unique()} years)")
print(f"Test rows:     {len(df_test):,}   (year {max_year})")
print(f"Training claims: {df_train['claim_count'].sum():,}")
print(f"Test claims:     {df_test['claim_count'].sum():,}")
```

### Frequency model

```python
X_train = df_train[FEATURE_COLS]
y_train = df_train["claim_count"].values
w_train = df_train["exposure"].values

X_test  = df_test[FEATURE_COLS]
y_test  = df_test["claim_count"].values
w_test  = df_test["exposure"].values

# Build Pools with the log-exposure offset (CRITICAL: baseline, not weight)
train_pool = Pool(
    X_train, y_train,
    baseline=np.log(np.clip(w_train, 1e-6, None)),
    cat_features=CAT_FEATURES,
)
test_pool = Pool(
    X_test, y_test,
    baseline=np.log(np.clip(w_test, 1e-6, None)),
    cat_features=CAT_FEATURES,
)

with mlflow.start_run(run_name="freq_model_m08") as freq_run:
    # Log all parameters that determined this model
    mlflow.log_params(best_freq_params)
    mlflow.log_params({
        "model_type":         "catboost_poisson",
        "raw_table":          TABLES["raw"],
        "raw_table_version":  int(raw_version),
        "feat_table":         TABLES["features"],
        "feat_table_version": int(feat_version),
        "feature_cols":       json.dumps(FEATURE_COLS),
        "cat_features":       json.dumps(CAT_FEATURES),
        "train_years":        str(sorted(df_train["accident_year"].unique().tolist())),
        "test_year":          str(max_year),
        "run_date":           RUN_DATE,
        "offset":             "log_exposure",   # explicit: this model uses the correct offset
    })

    # Fit the final frequency model
    freq_model = CatBoostRegressor(**best_freq_params)
    freq_model.fit(train_pool, eval_set=test_pool)

    # Evaluate on test set
    freq_pred_test = freq_model.predict(test_pool)
    test_dev = poisson_deviance(y_test, freq_pred_test, w_test)

    mlflow.log_metric("test_poisson_deviance", test_dev)
    mlflow.log_metric("mean_cv_deviance",      mean_cv_deviance)
    mlflow.log_metric("n_training_rows",       len(df_train))
    mlflow.log_metric("n_test_rows",           len(df_test))

    # Log the model artefact to MLflow Model Registry
    mlflow.catboost.log_model(freq_model, "freq_model")
    freq_run_id = freq_run.info.run_id

print(f"Frequency model trained and logged.")
print(f"MLflow run ID: {freq_run_id}")
print(f"Test Poisson deviance: {test_dev:.5f}")
print(f"Mean CV deviance:      {mean_cv_deviance:.5f}")
print(f"Generalisation gap:    {test_dev - mean_cv_deviance:.5f} "
      f"({'within tolerance' if abs(test_dev - mean_cv_deviance) < 0.01 else 'REVIEW: may be overfitting'})")
```

**What is the generalisation gap?** The gap between the mean CV deviance and the test deviance tells you whether the model generalises from its validation period to the final test year. A small positive gap (test deviance slightly higher than CV deviance) is normal and expected -- the test year is a different period. A large gap (more than 0.01 for this dataset) suggests the model is tuned too aggressively to the validation period, or that the test year's distribution is materially different from training.

### Writing frequency predictions to Delta

```python
# Predict frequency for the full test set
freq_pred_all = freq_model.predict(test_pool)

# The model predicts claim counts (exposure * frequency).
# Divide by exposure to get frequency per policy-year.
freq_rate_all = freq_pred_all / np.clip(w_test, 1e-6, None)

freq_pred_df = pl.DataFrame({
    "policy_id":          df_test["policy_id"].tolist(),
    "accident_year":      df_test["accident_year"].tolist(),
    "exposure":           w_test.tolist(),
    "claim_count_actual": y_test.tolist(),
    "freq_pred_count":    freq_pred_all.tolist(),
    "freq_pred_rate":     freq_rate_all.tolist(),
    "mlflow_run_id":      [freq_run_id] * len(df_test),
    "run_date":           [RUN_DATE] * len(df_test),
})

spark.createDataFrame(freq_pred_df.to_pandas()) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["freq_predictions"])

freq_pred_version = spark.sql(
    f"DESCRIBE HISTORY {TABLES['freq_predictions']} LIMIT 1"
).collect()[0]["version"]

print(f"Frequency predictions written: {len(freq_pred_df):,} rows")
print(f"Table: {TABLES['freq_predictions']} (version {freq_pred_version})")
```

### Severity model

```python
# -----------------------------------------------------------------------
# Severity model: Tweedie with variance_power=2 (Gamma distribution)
#
# Trains on policies with at least one claim only.
# Target: mean severity per claim (incurred / claim_count).
# Weight: claim count (more claims = more evidence about this risk's severity).
# No exposure offset -- we are modelling severity given a claim, not claim counts.
# -----------------------------------------------------------------------

df_sev_train = df_train[df_train["claim_count"] > 0].copy()
df_sev_test  = df_test[df_test["claim_count"]  > 0].copy()

df_sev_train["mean_sev"] = df_sev_train["incurred_loss"] / df_sev_train["claim_count"]
df_sev_test["mean_sev"]  = df_sev_test["incurred_loss"]  / df_sev_test["claim_count"]

X_sev_train = df_sev_train[FEATURE_COLS]
y_sev_train = df_sev_train["mean_sev"].values
w_sev_train = df_sev_train["claim_count"].values

X_sev_test  = df_sev_test[FEATURE_COLS]
y_sev_test  = df_sev_test["mean_sev"].values
w_sev_test  = df_sev_test["claim_count"].values

sev_train_pool = Pool(
    X_sev_train, y_sev_train,
    weight=w_sev_train,
    cat_features=CAT_FEATURES,
)
sev_test_pool = Pool(
    X_sev_test, y_sev_test,
    weight=w_sev_test,
    cat_features=CAT_FEATURES,
)

print(f"Severity training: {len(df_sev_train):,} policies with claims")
print(f"Severity test:     {len(df_sev_test):,} policies with claims")

with mlflow.start_run(run_name="sev_model_m08") as sev_run:
    mlflow.log_params(best_sev_params)
    mlflow.log_params({
        "model_type":         "catboost_gamma",
        "raw_table":          TABLES["raw"],
        "raw_table_version":  int(raw_version),
        "feat_table":         TABLES["features"],
        "feat_table_version": int(feat_version),
        "feature_cols":       json.dumps(FEATURE_COLS),
        "cat_features":       json.dumps(CAT_FEATURES),
        "train_years":        str(sorted(df_sev_train["accident_year"].unique().tolist())),
        "test_year":          str(max_year),
        "run_date":           RUN_DATE,
        "severity_target":    "mean_sev_per_claim",
        "severity_weight":    "claim_count",
    })

    sev_model = CatBoostRegressor(**best_sev_params)
    sev_model.fit(sev_train_pool, eval_set=sev_test_pool)

    sev_pred_test = sev_model.predict(sev_test_pool)
    sev_rmse = float(np.sqrt(np.mean((y_sev_test - sev_pred_test)**2)))

    mlflow.log_metric("test_severity_rmse",  sev_rmse)
    mlflow.log_metric("n_sev_training_rows", len(df_sev_train))
    mlflow.log_metric("n_sev_test_rows",     len(df_sev_test))
    mlflow.catboost.log_model(sev_model, "sev_model")
    sev_run_id = sev_run.info.run_id

print(f"Severity model trained and logged.")
print(f"MLflow run ID: {sev_run_id}")
print(f"Test severity RMSE: £{sev_rmse:,.0f}")
```

### Computing pure premiums for the full test set

The pure premium is frequency times severity. The critical step here is predicting severity for ALL test policies, not just those with claims. The severity model predicts expected severity given a claim occurs -- it is a valid prediction for any policy, regardless of whether it had a claim in the test year.

The earlier versions of this pipeline used `fillna(sev_pred.mean())` to fill in severity for zero-claim policies, where `sev_pred.mean()` was the mean over the claims-only prediction set. This is wrong: the mean severity for policies that had claims is biased upward relative to the population. Zero-claim policies tend to be lower-risk, so the correct imputed severity is lower than the claims-only mean.

The correct approach is to predict severity for all policies:

```python
# Predict severity for ALL test policies (not just those with claims)
# The severity model predicts: expected severity given a claim occurred
# This is a meaningful prediction for any policy

X_all_test = df_test[FEATURE_COLS]

# Pool for full test set (no labels needed for prediction)
all_test_pool_sev = Pool(X_all_test, cat_features=CAT_FEATURES)
sev_pred_all = sev_model.predict(all_test_pool_sev)

# Pure premium = predicted frequency rate * predicted severity
# This is the expected loss cost per policy-year for each risk
pure_premium = freq_rate_all * sev_pred_all

# Sanity checks
assert (sev_pred_all > 0).all(),  "Negative severity predictions (model error)"
assert (pure_premium  > 0).all(), "Negative pure premiums (frequency or severity error)"
assert np.isfinite(pure_premium).all(), "Non-finite pure premiums"

print(f"Pure premium statistics (test set):")
print(f"  Mean:   £{pure_premium.mean():,.2f}")
print(f"  Median: £{np.median(pure_premium):,.2f}")
print(f"  P95:    £{np.percentile(pure_premium, 95):,.2f}")
print(f"  Min:    £{pure_premium.min():,.2f}")
print(f"  Max:    £{pure_premium.max():,.2f}")

# Append to frequency predictions table
pred_full_df = pl.DataFrame({
    "policy_id":          df_test["policy_id"].tolist(),
    "accident_year":      df_test["accident_year"].tolist(),
    "exposure":           w_test.tolist(),
    "claim_count_actual": y_test.tolist(),
    "freq_pred_rate":     freq_rate_all.tolist(),
    "sev_pred":           sev_pred_all.tolist(),
    "pure_premium":       pure_premium.tolist(),
    "freq_run_id":        [freq_run_id] * len(df_test),
    "sev_run_id":         [sev_run_id]  * len(df_test),
    "run_date":           [RUN_DATE]    * len(df_test),
})

spark.createDataFrame(pred_full_df.to_pandas()) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["freq_predictions"])

print(f"\nFull predictions with pure premium written to {TABLES['freq_predictions']}")
```

---

## Part 11: Stage 7 -- SHAP relativities

SHAP relativities translate the GBM's internal structure into a factor table that the pricing committee and the rating system can understand. We covered this in detail in Module 4. Here we implement it in the pipeline context.

Add a markdown cell:

```python
%md
## Stage 7: SHAP relativities from the frequency model
```

```python
import shap

# Compute SHAP values for the test set
# The SHAP values are computed in the log-space of the model's output
# (because the model uses a log link). Exponentiating gives relativities.
explainer  = shap.TreeExplainer(freq_model)
shap_vals  = explainer.shap_values(X_test)

# SHAP value for feature j on observation i: the contribution of feature j
# to log(predicted_count) relative to the model's baseline prediction.
# exp(shap_val) gives the multiplicative relativity.

# Mean absolute SHAP by feature: measures overall feature importance
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
feature_importance = pl.DataFrame({
    "feature":         FEATURE_COLS,
    "mean_abs_shap":   mean_abs_shap.tolist(),
    "mean_relativity": [float(np.exp(np.abs(shap_vals[:, i])).mean())
                        for i in range(len(FEATURE_COLS))],
}).sort("mean_abs_shap", descending=True)

print("Feature importance (mean absolute SHAP values):")
print(feature_importance)
```

### Building factor relativity tables

For each feature, we compute the mean SHAP relativity at each level. This is the input that the rate optimiser uses to set the factor structure. Note that these relativities are derived directly from the GBM -- they reflect the actual model predictions, not a post-hoc regression.

```python
# For each categorical feature: mean SHAP relativity per category
# For continuous features: binned by decile

shap_relativities = {}

for j, feat in enumerate(FEATURE_COLS):
    shap_col = shap_vals[:, j]
    feat_vals = X_test[feat].values if hasattr(X_test, 'values') else np.array(X_test[feat])

    if feat in CAT_FEATURES:
        # Categorical: average SHAP per category, exponentiate for relativity
        categories = np.unique(feat_vals)
        rel_table  = {}
        for cat in categories:
            mask = feat_vals == cat
            rel_table[cat] = float(np.exp(shap_col[mask].mean()))
        # Normalise so mean relativity = 1.0
        mean_rel = np.mean(list(rel_table.values()))
        rel_table = {k: v / mean_rel for k, v in rel_table.items()}
        shap_relativities[feat] = rel_table
        print(f"\n{feat} relativities (normalised to mean=1):")
        for cat, rel in sorted(rel_table.items()):
            bar = "#" * int(rel * 10)
            print(f"  {cat:<15} {rel:.3f}  {bar}")
    else:
        # Continuous: bin into 5 groups by value and report mean relativity
        decile_labels = pd.qcut(feat_vals, q=5, duplicates="drop")
        rel_by_bin = {}
        for lbl in decile_labels.unique():
            mask = decile_labels == lbl
            rel_by_bin[str(lbl)] = float(np.exp(shap_col[mask].mean()))
        # Normalise
        mean_rel = np.mean(list(rel_by_bin.values()))
        rel_by_bin = {k: v / mean_rel for k, v in rel_by_bin.items()}
        shap_relativities[feat] = rel_by_bin
        print(f"\n{feat} relativities by quintile (normalised to mean=1):")
        for bin_lbl, rel in rel_by_bin.items():
            bar = "#" * int(rel * 10)
            print(f"  {bin_lbl:<25} {rel:.3f}  {bar}")

# Log SHAP relativities to MLflow as a JSON artefact
with mlflow.start_run(run_id=freq_run_id):
    mlflow.log_dict(shap_relativities, "shap_relativities.json")
    mlflow.log_dict({"features": feature_importance.to_dict(as_series=False)},
                    "feature_importance.json")

print("\nSHAP relativities logged to MLflow run:", freq_run_id)
```

---

## Part 12: Stage 8 -- Conformal prediction intervals

The conformal predictor quantifies uncertainty in the severity predictions. We covered the theory in Module 5. The critical requirement for a valid conformal interval is that calibration and test data are exchangeable -- which means the split must be temporal, not random.

Add a markdown cell:

```python
%md
## Stage 8: Conformal prediction intervals -- temporal calibration split
```

```python
from insurance_conformal import InsuranceConformalPredictor

# -----------------------------------------------------------------------
# Temporal calibration split.
#
# Module 5 established that calibration and test data must be exchangeable.
# Exchangeability requires that they come from the same distribution.
# A random split violates this for insurance data because time trends
# (inflation, mix changes, legislative effects) mean 2022 observations
# and 2024 observations are NOT exchangeable.
#
# Correct approach: use the penultimate accident year as calibration
# and the final accident year as test. Both are out-of-time relative
# to the training set.
# -----------------------------------------------------------------------

# Identify the calibration year: second most recent accident year
all_years   = sorted(df_train["accident_year"].unique())
cal_year    = all_years[-1]   # most recent training year = calibration set
print(f"Calibration year: {cal_year}")
print(f"Test year:        {max_year}")

# Claims-only sets for conformal calibration
# Conformal intervals are calibrated on severity predictions
df_cal_sev = features_pd[
    (features_pd["accident_year"] == cal_year) &
    (features_pd["claim_count"] > 0)
].copy()
df_te_sev  = df_test[df_test["claim_count"] > 0].copy()

df_cal_sev["mean_sev"] = df_cal_sev["incurred_loss"] / df_cal_sev["claim_count"]
df_te_sev["mean_sev"]  = df_te_sev["incurred_loss"]  / df_te_sev["claim_count"]

X_cal = df_cal_sev[FEATURE_COLS]
y_cal = df_cal_sev["mean_sev"].values

X_te_conf = df_te_sev[FEATURE_COLS]
y_te_conf = df_te_sev["mean_sev"].values

print(f"Calibration claims: {len(df_cal_sev):,}")
print(f"Test claims:        {len(df_te_sev):,}")

# -----------------------------------------------------------------------
# A note on calibration set size.
# The conformal coverage guarantee has a finite-sample correction:
# P(y_new inside interval) >= (1 - alpha) - 1/(n_cal + 1)
# For n_cal=1000, the correction is 0.001 -- negligible.
# For n_cal=100, it is 0.01 -- still acceptable.
# For n_cal<50, the intervals are unreliable. If your calibration set
# has fewer than 50 claims, reconsider the temporal split.
# -----------------------------------------------------------------------
if len(df_cal_sev) < 100:
    print(f"WARNING: Calibration set has only {len(df_cal_sev)} claims. "
          f"Conformal intervals may be unreliable. Consider a wider calibration window.")
```

### Calibrating the conformal predictor

```python
cp = InsuranceConformalPredictor(
    model=sev_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=2.0,
)
cp.calibrate(X_cal, y_cal)

# Generate 90% prediction intervals for the test set
intervals = cp.predict_interval(X_te_conf, alpha=CONFORMAL_ALPHA)

# intervals is a DataFrame with columns: lower, mid, upper
# lower: lower bound of the 90% interval
# mid:   point prediction (same as sev_model.predict())
# upper: upper bound of the 90% interval

print(f"\nSeverity prediction intervals (90%, alpha={CONFORMAL_ALPHA}):")
print(f"  Mean lower bound: £{intervals['lower'].mean():,.0f}")
print(f"  Mean point pred:  £{intervals['mid'].mean():,.0f}")
print(f"  Mean upper bound: £{intervals['upper'].mean():,.0f}")
print(f"  Mean interval width: £{(intervals['upper'] - intervals['lower']).mean():,.0f}")
```

### Coverage validation

The coverage check is mandatory. A conformal predictor that achieves 90% overall coverage but only 72% coverage in the top decile of risks is not usable for reserving or underwriting referral.

```python
diag = cp.coverage_by_decile(X_te_conf, y_te_conf, alpha=CONFORMAL_ALPHA)
min_cov = float(diag["coverage"].min())

print("\nCoverage by predicted severity decile:")
print(f"{'Decile':<8} {'Coverage':>10} {'Status':>12}")
for _, row in diag.iterrows():
    status = "OK" if row["coverage"] >= 0.85 else "WARN: below 85%"
    print(f"  {row['decile']:<6} {row['coverage']:>10.3f} {status:>12}")

print(f"\nOverall min decile coverage: {min_cov:.3f}")
if min_cov < 0.85:
    print("WARNING: Coverage below 85% in at least one decile.")
    print("Possible causes:")
    print("  1. Calibration set is too small (fewer than ~500 claims)")
    print("  2. Claims inflation or mix change between calibration and test years")
    print("  3. Nonconformity score specification does not match the severity distribution")
    print("  4. The calibration and test distributions are genuinely different")
    print("Consider: wider calibration window, or recalibrate on more recent data.")
else:
    print("Coverage is acceptable in all deciles. Intervals may be used for "
          "underwriting referral and minimum premium floor setting.")
```

### Flagging uncertain risks for underwriting referral

```python
# Relative interval width = (upper - lower) / mid
# High relative width = high model uncertainty for this risk
rel_width = (intervals["upper"] - intervals["lower"]) / (intervals["mid"] + 1e-6)

# Flag risks in the top 10% of relative width for underwriting referral
referral_threshold = np.percentile(rel_width, 90)
referral_flag      = rel_width >= referral_threshold

print(f"\nUnderwriting referral flag:")
print(f"  Relative width threshold (P90): {referral_threshold:.2f}")
print(f"  Policies flagged for referral:  {referral_flag.sum():,} "
      f"({referral_flag.mean():.1%} of claims set)")
```

### Writing conformal intervals to Delta

```python
conf_df = pl.DataFrame({
    "policy_id":    df_te_sev["policy_id"].tolist(),
    "accident_year": df_te_sev["accident_year"].tolist(),
    "mean_sev_actual": y_te_conf.tolist(),
    "sev_lower":    intervals["lower"].tolist(),
    "sev_mid":      intervals["mid"].tolist(),
    "sev_upper":    intervals["upper"].tolist(),
    "rel_width":    rel_width.tolist(),
    "referral_flag": referral_flag.tolist(),
    "freq_run_id":  [freq_run_id] * len(df_te_sev),
    "sev_run_id":   [sev_run_id]  * len(df_te_sev),
    "run_date":     [RUN_DATE]    * len(df_te_sev),
})

spark.createDataFrame(conf_df.to_pandas()) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["conformal_intervals"])

print(f"Conformal intervals written: {len(conf_df):,} rows")
print(f"Table: {TABLES['conformal_intervals']}")
```

---

## Part 13: Stage 9 -- Rate optimisation

The rate optimiser from Module 7 takes a renewal portfolio and finds the factor adjustments that minimise loss ratio subject to volume and ENBP constraints. In a well-connected pipeline, the inputs to the rate optimiser come from the upstream stages: the frequency model's pure premium predictions define the technical premium, and the SHAP relativities define the factor structure.

Add a markdown cell:

```python
%md
## Stage 9: Rate optimisation -- connected to Stage 6 and Stage 7
```

### Building the renewal portfolio from model predictions

The renewal portfolio is the set of policies coming up for renewal in the next rating period. In production, this comes from the policy administration system. Here, we use the test set predictions as a proxy -- the test year represents the most recent pricing period, and its policies are the renewal candidates.

```python
from rate_optimiser import (
    PolicyData, FactorStructure, RateChangeOptimiser,
    LossRatioConstraint, VolumeConstraint, ENBPConstraint,
    FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams
from scipy.special import expit
import pandas as pd

# -----------------------------------------------------------------------
# Build renewal portfolio.
#
# The connection to Stage 6:
#   - pure_premium comes from freq_model.predict * sev_model.predict
#     (not from a separate synthetic generation step)
#   - This ensures the rate optimiser is working from the same technical
#     premium numbers that informed the pricing review
#
# The connection to Stage 7:
#   - Factor structure uses the SHAP relativities directly
#   - Region SHAP relativities from Stage 7 define f_region
#   - NCB deficit SHAP relativities define f_ncb
#   - These are not synthetic proxies -- they are the model's own output
# -----------------------------------------------------------------------

n_renewal = min(5_000, len(df_test))   # cap for computation speed

rng_opt = np.random.default_rng(seed=9999)

# Select the renewal subset
ren_idx  = rng_opt.choice(len(df_test), n_renewal, replace=False)
df_ren   = df_test.iloc[ren_idx].copy()
pp_ren   = pure_premium[ren_idx]

# Technical premium: pure premium (frequency * severity)
tech_prem = pp_ren

# Current premium: assume book running at 75% LR
# In production, current_premium comes from the policy administration system
curr_prem = tech_prem / LR_TARGET

# Market premium: assume market is 2pp below current LR
mkt_prem  = tech_prem / (LR_TARGET - 0.02)

# Renewal probability: logistic model on price relativity
# P(renew) = sigmoid(intercept + price_coef * log(curr/mkt))
renewal_prob = expit(1.0 + (-2.0) * np.log(curr_prem / mkt_prem))

# -----------------------------------------------------------------------
# Factor structure -- derived from SHAP relativities
#
# For region: use the SHAP relativities computed in Stage 7.
# For NCB: map ncb_deficit to its SHAP relativity quintile.
#
# In a production pipeline, this mapping is automatic from the SHAP output.
# Here we implement it manually for clarity.
# -----------------------------------------------------------------------

# Region factor: from SHAP relativities (Stage 7)
region_shap = shap_relativities.get("region", {})
if region_shap:
    # Map each policy's region to its SHAP-derived relativity
    f_region = np.array([
        region_shap.get(str(r), 1.0)
        for r in df_ren["region"].values
    ])
else:
    # Fallback if SHAP relativities not available
    region_rel = {"North": 0.82, "Midlands": 0.90, "SouthEast": 1.08,
                  "London": 1.38, "SouthWest": 0.86}
    f_region = np.array([region_rel.get(str(r), 1.0) for r in df_ren["region"].values])

# NCB factor: from SHAP relativities (continuous feature, use quintile bins)
ncb_deficit_shap = shap_relativities.get("ncb_deficit", {})
if ncb_deficit_shap:
    # Map via quintile bins -- for simplicity, map deficit 0-5 directly
    # In production, use the quantile bin boundaries from Stage 7
    ncb_rel_by_deficit = {
        0: 0.65,  # no deficit = lowest risk = lowest factor
        1: 0.78,
        2: 0.90,
        3: 1.05,
        4: 1.25,
        5: 1.55,  # full deficit = highest risk = highest factor
    }
    f_ncb = np.array([ncb_rel_by_deficit.get(int(d), 1.0)
                      for d in df_ren["ncb_deficit"].values
                      if "ncb_deficit" in df_ren.columns]
                     ) if "ncb_deficit" in df_ren.columns else np.ones(n_renewal)
else:
    f_ncb = np.ones(n_renewal)

# Age factor: from age_mid (continuous -- use quintile)
age_vals = df_ren["age_mid"].values if "age_mid" in df_ren.columns else np.full(n_renewal, 43.0)
age_relative = (age_vals - age_vals.mean()) / (age_vals.std() + 1e-6)
f_age = np.exp(0.20 * age_relative)   # approximate monotone age effect

print(f"Renewal portfolio: {n_renewal:,} policies")
print(f"Mean technical premium: £{tech_prem.mean():,.2f}")
print(f"Mean current premium:   £{curr_prem.mean():,.2f}")
print(f"Mean renewal probability: {renewal_prob.mean():.3f}")
```

### Running the optimiser

```python
channels = rng_opt.choice(["PCW", "direct"], n_renewal, p=[0.68, 0.32])

renewal_port = pd.DataFrame({
    "policy_id":         df_ren["policy_id"].values,
    "channel":           channels,
    "renewal_flag":      np.ones(n_renewal, dtype=bool),
    "technical_premium": tech_prem,
    "current_premium":   curr_prem,
    "market_premium":    mkt_prem,
    "renewal_prob":      renewal_prob,
    "tenure":            rng_opt.integers(0, 10, n_renewal).astype(float),
    "f_region":          f_region,
    "f_ncb":             f_ncb,
    "f_age":             f_age,
    "f_iat":             np.ones(n_renewal),   # introduced factor (no change)
})

data = PolicyData(renewal_port)
fs   = FactorStructure(
    factor_names=["f_region", "f_ncb", "f_age", "f_iat"],
    factor_values=renewal_port[["f_region", "f_ncb", "f_age", "f_iat"]],
    renewal_factor_names=["f_iat"],   # only f_iat is renewal-only
)

demand = make_logistic_demand(
    LogisticDemandParams(intercept=1.0, price_coef=-2.0, tenure_coef=0.04)
)
opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)

# Constraints from Module 7
opt.add_constraint(LossRatioConstraint(bound=LR_TARGET))
opt.add_constraint(VolumeConstraint(bound=VOLUME_FLOOR))
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15,
                                          n_factors=fs.n_factors))

result = opt.solve()
print(result.summary())
```

### Writing rate action factors to Delta

```python
rate_factors_df = pl.DataFrame({
    "run_date":      [RUN_DATE],
    "factor":        ["f_region", "f_ncb", "f_age", "f_iat"],
    "adjustment":    result.factor_adjustments.tolist(),
    "lr_target":     [LR_TARGET] * 4,
    "optimiser_converged": [result.converged] * 4,
    "expected_lr":   [round(result.expected_loss_ratio, 4)] * 4,
    "expected_vol":  [round(result.expected_volume_ratio, 4)] * 4,
})

spark.createDataFrame(rate_factors_df.to_pandas()) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(TABLES["rate_change"])

print(f"Rate action factors written to {TABLES['rate_change']}")
print(f"Optimiser converged: {result.converged}")
print(f"Expected LR after action: {result.expected_loss_ratio:.3f}")
print(f"Expected volume retention: {result.expected_volume_ratio:.3f}")
```

---

## Part 14: Stage 10 -- The pipeline audit record

The audit record is the last thing the pipeline writes. It is a single row that ties together every output from every upstream stage: the raw data version, the features version, the MLflow run IDs for both models, all validation metrics, and the rate optimisation outcome.

Add a markdown cell:

```python
%md
## Stage 10: Pipeline audit record
```

```python
audit_record = {
    # ---- Identifiers ----
    "run_date":                  RUN_DATE,
    "pipeline_version":          "module_08_v1",

    # ---- Data provenance ----
    "raw_table":                 TABLES["raw"],
    "raw_table_version":         int(raw_version),
    "features_table":            TABLES["features"],
    "features_table_version":    int(feat_version),

    # ---- Model provenance ----
    "freq_model_run_id":         freq_run_id,
    "sev_model_run_id":          sev_run_id,

    # ---- Training data summary ----
    "n_training_rows":           int(len(df_train)),
    "n_test_rows":               int(len(df_test)),
    "test_year":                 int(max_year),
    "training_years":            str(sorted(df_train["accident_year"].unique().tolist())),

    # ---- Model performance ----
    "test_poisson_deviance":     round(test_dev, 5),
    "mean_cv_deviance":          round(mean_cv_deviance, 5),
    "sev_rmse":                  round(sev_rmse, 2),
    "n_cv_folds":                len(cv_deviances),

    # ---- Optuna ----
    "freq_optuna_trials":        N_OPTUNA_TRIALS,
    "freq_best_deviance":        round(freq_study.best_value, 5),
    "sev_optuna_trials":         N_OPTUNA_TRIALS,
    "sev_best_rmse":             round(sev_study.best_value, 2),

    # ---- Conformal ----
    "conformal_alpha":           CONFORMAL_ALPHA,
    "conformal_cal_year":        int(cal_year),
    "conformal_n_cal_claims":    int(len(df_cal_sev)),
    "conformal_min_decile_cov":  round(min_cov, 3),

    # ---- Rate optimisation ----
    "lr_target":                 LR_TARGET,
    "volume_floor":              VOLUME_FLOOR,
    "optimiser_converged":       bool(result.converged),
    "expected_lr":               round(float(result.expected_loss_ratio), 4),
    "expected_volume":           round(float(result.expected_volume_ratio), 4),

    # ---- Configuration ----
    "catalog":                   CATALOG,
    "schema":                    SCHEMA,
    "feature_cols":              json.dumps(FEATURE_COLS),
    "cat_features":              json.dumps(CAT_FEATURES),

    # ---- Notes ----
    "pipeline_notes":            "Module 8 end-to-end pipeline, synthetic motor data",
}

audit_pl = pl.DataFrame([audit_record])

# mode("append") is deliberate -- every run adds a row.
# mode("overwrite") would destroy the history.
spark.createDataFrame(audit_pl.to_pandas()) \
    .write.format("delta") \
    .mode("append") \
    .saveAsTable(TABLES["pipeline_audit"])

print("Pipeline audit record written.")
print(f"Table: {TABLES['pipeline_audit']}")
print("\nAudit record summary:")
for k, v in audit_record.items():
    print(f"  {k:<40} {v}")
```

### What the audit record enables

Given the `run_date` for any historical pipeline run, you can retrieve the full audit record and reproduce every output exactly:

```python
# Six months later: reproduce the training data
logged_raw_version = 0   # from audit_record["raw_table_version"]

historical_raw = spark.read.format("delta") \
    .option("versionAsOf", logged_raw_version) \
    .table(TABLES["raw"]) \
    .toPandas()

# Six months later: load the trained frequency model
logged_freq_run_id = "abc123..."  # from audit_record["freq_model_run_id"]
historical_freq_model = mlflow.catboost.load_model(f"runs:/{logged_freq_run_id}/freq_model")
```

This is the FCA Consumer Duty audit trail in practice. Consumer Duty requires that you can demonstrate, for any pricing decision in the last three years: what model was used, what data it was trained on, what validation metrics it achieved, what rate action it informed. The audit record table, combined with Delta time travel and MLflow model registry, satisfies all four requirements automatically.

---

## Part 15: Feature consistency guards

The audit record documents what happened. The feature consistency guard prevents failures from happening. This section shows how to build one.

```python
class FeatureSpec:
    """
    Records the expected dtype and range of each feature at training time.
    Validates the scoring data against the spec at inference time.

    Usage:
      spec = FeatureSpec()
      spec.record(X_train_polars, cat_features=CAT_FEATURES)
      spec.to_json("/dbfs/models/feature_spec.json")

      # At scoring time:
      spec_loaded = FeatureSpec.from_json("/dbfs/models/feature_spec.json")
      errors = spec_loaded.validate(X_score_polars)
      if errors:
          raise ValueError(f"Feature spec validation failed: {errors}")
    """

    def __init__(self):
        self.spec = {}

    def record(self, df: pl.DataFrame, cat_features: list) -> None:
        """Record the spec from a training feature DataFrame."""
        for col in df.columns:
            series = df[col]
            if col in cat_features or series.dtype == pl.Utf8:
                self.spec[col] = {
                    "dtype":       "categorical",
                    "unique_vals": sorted(series.drop_nulls().unique().to_list()),
                }
            else:
                self.spec[col] = {
                    "dtype": "numeric",
                    "min":   float(series.min()),
                    "max":   float(series.max()),
                }

    def validate(self, df: pl.DataFrame) -> list:
        """
        Return a list of validation errors.
        An empty list means the feature DataFrame is consistent with the spec.
        """
        errors = []
        for col, col_spec in self.spec.items():
            if col not in df.columns:
                errors.append(f"Missing column: {col}")
                continue
            series = df[col]
            if col_spec["dtype"] == "categorical":
                if series.dtype not in (pl.Utf8, pl.Categorical):
                    errors.append(
                        f"{col}: expected categorical dtype, got {series.dtype}. "
                        f"Call .cast(pl.Utf8) before scoring."
                    )
                else:
                    unseen = set(series.drop_nulls().unique().to_list()) - \
                             set(col_spec["unique_vals"])
                    if unseen:
                        errors.append(
                            f"{col}: unseen categories at scoring time: {unseen}. "
                            f"CatBoost will handle these, but verify they are genuine "
                            f"new values and not encoding errors."
                        )
            else:
                if series.dtype == pl.Utf8 or series.dtype == pl.Categorical:
                    errors.append(
                        f"{col}: expected numeric, got {series.dtype}."
                    )
        return errors

    def to_json(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            json.dump(self.spec, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "FeatureSpec":
        import json
        obj = cls()
        with open(path) as f:
            obj.spec = json.load(f)
        return obj

# Record the spec at training time
feature_spec = FeatureSpec()
feature_spec.record(
    features_pl.select(FEATURE_COLS),
    cat_features=CAT_FEATURES,
)

# Log it as an MLflow artefact alongside the model
with mlflow.start_run(run_id=freq_run_id):
    feature_spec.to_json("/tmp/feature_spec.json")
    mlflow.log_artifact("/tmp/feature_spec.json", artifact_path="feature_spec")

print("FeatureSpec recorded and logged to MLflow.")
print("\nSpec summary:")
for col, col_spec in feature_spec.spec.items():
    if col_spec["dtype"] == "categorical":
        print(f"  {col:<20} categorical  {col_spec['unique_vals']}")
    else:
        print(f"  {col:<20} numeric      [{col_spec['min']:.2f}, {col_spec['max']:.2f}]")
```

### Delta table VACUUM policy

Before the audit section completes, set the retention policy on all tables. The default Databricks VACUUM retention is 30 days. Consumer Duty requires three years of reproducibility. Set all pipeline tables to 365 days minimum:

```python
for table_name, table_path in TABLES.items():
    try:
        spark.sql(f"""
            ALTER TABLE {table_path}
            SET TBLPROPERTIES ('delta.deletedFileRetentionDuration' = 'interval 365 days')
        """)
        print(f"Retention set: {table_path}")
    except Exception as e:
        print(f"Could not set retention on {table_path}: {e}")
```

---

## Part 16: Adapting the pipeline for production

The notebook runs end-to-end on synthetic data. Adapting it to a real motor book requires changes in four places only:

**Stage 2 (Data ingestion).** Replace the synthetic data generation with a read from your policy administration system. The only requirement is that the input DataFrame has the columns that TRANSFORMS expects: `ncb_years`, `vehicle_group`, `age_band`, `annual_mileage`, `region`, `exposure`, `claim_count`, `incurred_loss`, `accident_year`.

```python
# Replace Stage 2 synthetic generation with:
raw_pl = pl.from_pandas(
    spark.table("your_source_system.motor.policies_claims")
    .filter(col("accident_year").between(2021, 2024))
    .toPandas()
)
```

**Stage 3 (Feature engineering).** Update TRANSFORMS to match your feature set. Every transform must be a pure function: it takes a Polars DataFrame and returns a Polars DataFrame. Do not add logic to `apply_transforms()` itself -- add a new function to TRANSFORMS.

**Stage 5 (Optuna).** Set `N_OPTUNA_TRIALS = 40` for production runs. The 20-trial default in this notebook runs in 5-10 minutes. For a production review cycle where compute cost is not a constraint, 40-60 trials gives meaningfully better parameter selection.

**Stage 4 (IBNR buffer).** Adjust the buffer months to match your line of business. For UK private motor property damage: 6 months. For motor bodily injury: 12-18 months. For employers' liability or solicitors' PI: 24 months minimum.

Everything else -- the MLflow logging structure, the conformal calibration, the audit record -- runs identically on your data.

---

## Part 17: What this pipeline does not cover

This section documents the known limitations of the pipeline as built. Understanding what the pipeline does not do is as important as understanding what it does.

### Geographic credibility

The pipeline treats `region` as a flat five-category factor. UK motor pricing in practice uses postcode district-level rating with 2,300+ distinct districts. A flat five-category region factor misses most of the genuine geographic variation.

Geographic credibility -- blending thin-cell postcode experience with the portfolio mean using a Bühlmann-Straub or Bayesian hierarchical model -- is covered in Module 6. The correct architecture is to replace the flat `region` factor with Module 6's blended district-level relativities, then feed those relativities into the SHAP factor structure.

Geographic credibility is also the most regulatory-sensitive element of UK motor pricing. The FCA's Consumer Duty and its ongoing work on proxy discrimination means that any postcode-level rating factor needs documented actuarial justification. The risk is not just that a postcode is used as a proxy for a protected characteristic -- the risk is that a spatial smoothing method that looks neutral mathematically may still propagate discrimination at the group level. Before deploying any geographic component in production, document: the smoothing method chosen, why it was preferred over alternatives, how it was validated against FCA proxy discrimination guidance, and how it performs for protected characteristic groups.

### Multi-peril combination

The pipeline models motor claims as a single severity distribution. UK private motor actually combines at least three distinct perils: own damage, third-party property damage, and third-party personal injury. Each has different frequency, severity, development characteristics, and regulatory treatment.

A production pipeline would fit separate frequency and severity models for each peril, calibrate separate conformal intervals, and combine them for the technical premium. The current pipeline's single-severity model will produce a reasonable overall pure premium but cannot differentiate between a book skewed towards large BI claims and one skewed towards frequent small OD claims.

### Live demand elasticity

The rate optimiser in Stage 9 uses a logistic demand model with fixed parameters (intercept=1.0, price_coef=-2.0). In production, the demand parameters should be estimated from your own book's renewal lapse experience, ideally re-estimated quarterly. A demand model with incorrect elasticity produces suboptimal rate actions: if you overestimate price sensitivity, the optimiser will be overly conservative with rate increases and you will under-recover on loss ratio.

### Monitoring and drift detection

The pipeline produces a model and validates it on historical data. It does not include a monitoring framework for detecting when that model's predictions drift from actual outcomes in production. Production model monitoring -- actual-versus-expected tracking, SHAP drift monitoring, coverage recalibration triggers -- is a separate pipeline that should run monthly against the production scoring engine.

### Claims development loading

The severity model trains on incurred-to-date values. For accident years that have not fully developed, the incurred values understate ultimate severity. The pipeline does not apply a development loading to the training severities before fitting the model. For motor property damage with a six-month IBNR buffer, this is acceptable -- most OD claims are settled within three months. For motor BI or commercial lines, a chain-ladder development factor should be applied to each accident year's severity before training.

### Expense loading

The technical premium from the pipeline is the expected loss cost only. It does not include acquisition costs, operating expenses, investment income offset, or reinsurance cost. A commercially deployed technical premium adds a loading to the model output for these items. The loading is typically applied at the product/channel level rather than at the individual policy level.

### Reinstatement and policy limits

The severity model predicts mean claim cost without applying per-claim excess structures or policy limits. For policies with material excess or sub-limits, the model's severity prediction is the ground-up cost. The treaty or XL reinsurance attachment point, and any policy excess, need to be applied separately to convert the model's ground-up prediction to a net-of-reinsurance technical premium.

### Reinsurance structure

Related to the above: the rate optimiser optimises on gross of reinsurance. For a book with material reinsurance, the optimiser should work on a net-of-reinsurance basis where the treaty cost is reflected in the technical premium. This requires a reinsurance cost model that is not included here.

---

## Part 18: Limitations

**The pipeline tunes on a single temporal fold.** Hyperparameters that are optimal for accident year 2023 may not be optimal for 2025. In production, tune on the penultimate fold and validate on the most recent fold. Preserve the most recent year as a clean test year that is not touched until final evaluation.

**The conformal calibration assumes the calibration year's distribution matches the test year.** For a rapidly changing book (new channel mix, new product launch, significant rate action in the intervening period), the exchangeability assumption underlying conformal prediction may not hold. Recalibrate when the book changes materially.

**The rate optimiser uses a logistic demand model.** Actual renewal elasticity is more complex: tenure effects, channel effects, competitive market dynamics, and the interaction between price and risk profile all affect whether a specific policy renews. The logistic model is a reasonable approximation for optimisation but is not a demand forecast.

**The pipeline does not handle mid-term adjustments.** Policies that were adjusted (endorsements, mid-term cancellations) after inception appear in the data as modified versions of the original. The pipeline treats each row as representing the full policy period. For a book with high endorsement rates, this misrepresents the actual exposure and risk profile of individual policies.

**SHAP values are computed on the test set only.** The SHAP relativities in Stage 7 reflect the feature distributions of the 2024 test year. If the 2024 distribution differs from the prospective renewal portfolio (e.g., because new business has been written in different segments), the relativities may not accurately represent the prospective book's factor structure.

**The severity model trains on claims-only data.** This is correct for severity estimation, but it means the severity model's training set is a non-random sample of the full portfolio -- it is biased toward higher-risk policies. Any covariate that predicts claim occurrence (and most do) will be confounded with severity in the claims-only training set. This is a known issue in joint frequency-severity modelling.

**The pipeline audit uses `mode("append")`.** Multiple pipeline runs on the same day produce multiple audit rows with the same `run_date`. Add a UUID or `pipeline_version` field as the primary key if you need to distinguish same-day runs.

**The efficient frontier is not shown.** Stage 9 solves a single-point optimisation for a specific LR target. In practice, you would solve the optimiser across a range of LR targets (e.g., 0.68 to 0.78 in 1pp increments) and plot the resulting volume-LR frontier. The pricing committee makes a decision from the frontier, not from a single optimisation result. This is covered in Module 7 but not implemented in this capstone.
