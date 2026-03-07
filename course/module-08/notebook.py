# Databricks notebook source
# MAGIC %md
# MAGIC # Module 8: End-to-End Pricing Pipeline
# MAGIC ## Modern Insurance Pricing with Python and Databricks
# MAGIC
# MAGIC This notebook builds the complete pricing pipeline in a single, auditable
# MAGIC notebook. Every stage runs here. Data flows from generated synthetic policies
# MAGIC to a rate change pack in Unity Catalog Delta tables.
# MAGIC
# MAGIC **Stages:**
# MAGIC
# MAGIC 1. Library installation
# MAGIC 2. Unity Catalog schema setup
# MAGIC 3. Synthetic data generation (200,000 policies, 4 accident years)
# MAGIC 4. Feature engineering - the transform layer
# MAGIC 5. Walk-forward cross-validation with insurance-cv
# MAGIC 6. Hyperparameter tuning with Optuna
# MAGIC 7. Full model training with MLflow tracking
# MAGIC 8. SHAP relativities with shap-relativities
# MAGIC 9. Conformal prediction intervals with insurance-conformal
# MAGIC 10. Credibility blending with credibility
# MAGIC 11. Rate optimisation with rate-optimiser
# MAGIC 12. Efficient frontier
# MAGIC 13. Output pack to Unity Catalog
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Databricks Runtime 14.3 LTS or later
# MAGIC - Unity Catalog enabled
# MAGIC
# MAGIC **Note:** This notebook takes 45-90 minutes end-to-end on a standard single-node
# MAGIC cluster. Run stages 1-4 first to confirm the environment, then let the rest run.
# MAGIC Reduce `N_TRIALS = 10` if cluster time is short.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 0: Library installation
# MAGIC
# MAGIC All five Burning Cost libraries plus CatBoost, Polars, Optuna, and MLflow.

# COMMAND ----------

# MAGIC %sh uv pip install \
# MAGIC   insurance-cv \
# MAGIC   "shap-relativities[catboost]" \
# MAGIC   "insurance-conformal[catboost]" \
# MAGIC   credibility \
# MAGIC   rate-optimiser \
# MAGIC   catboost polars optuna mlflow scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: Configuration

# COMMAND ----------

# Catalog and schema - name with the review cycle, not the model version
CATALOG = "pricing"
SCHEMA  = "motor_q1_2026"

# MLflow experiment path - change to your Databricks user path
MLFLOW_EXPERIMENT_PATH = f"/Shared/pricing/{SCHEMA}"

# Hyperparameter tuning trials per model (reduce to 10 for faster runs)
N_TRIALS = 30

# Rate optimisation
LR_TARGET    = 0.70
VOLUME_FLOOR = 0.97

# Table names - written throughout the pipeline
TABLES = {
    "raw":                f"{CATALOG}.{SCHEMA}.raw_policies",
    "features":           f"{CATALOG}.{SCHEMA}.features",
    "freq_predictions":   f"{CATALOG}.{SCHEMA}.freq_predictions",
    "relativities":       f"{CATALOG}.{SCHEMA}.relativities",
    "conformal_intervals": f"{CATALOG}.{SCHEMA}.conformal_intervals",
    "blended_relativities": f"{CATALOG}.{SCHEMA}.blended_relativities",
    "rate_change":        f"{CATALOG}.{SCHEMA}.rate_change",
    "efficient_frontier": f"{CATALOG}.{SCHEMA}.efficient_frontier",
}

print(f"Schema:  {CATALOG}.{SCHEMA}")
print(f"Tables:  {list(TABLES.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: Unity Catalog schema setup
# MAGIC
# MAGIC Every artefact lives in Unity Catalog. This is what makes the pipeline
# MAGIC auditable: anyone with access to the schema can reproduce any number in
# MAGIC the rate change pack from the Delta tables.

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

print(f"Schema {CATALOG}.{SCHEMA} is ready.")
print(f"Tables will be written as the pipeline progresses.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3: Imports

# COMMAND ----------

import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import mlflow
import mlflow.catboost
from mlflow import MlflowClient
from catboost import CatBoostRegressor, Pool
import catboost as cb
from scipy.special import expit

optuna.logging.set_verbosity(optuna.logging.WARNING)
mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

print(f"CatBoost:             {cb.__version__}")
print(f"Polars:               {pl.__version__}")
print(f"Optuna:               {optuna.__version__}")
print(f"MLflow:               {mlflow.__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4: Data generation
# MAGIC
# MAGIC 200,000 UK motor policies across four accident years with realistic cohort
# MAGIC structure and premium inflation. 2025 is the most recent year and has a
# MAGIC natural IBNR under-reporting pattern.
# MAGIC
# MAGIC In production, replace this cell with:
# MAGIC `raw_pd = spark.table("source_system.motor.policies").toPandas()`

# COMMAND ----------

rng     = np.random.default_rng(2026)
N_TOTAL = 200_000
YEARS   = [2022, 2023, 2024, 2025]
N_PER_YEAR = N_TOTAL // len(YEARS)

cohorts = []
for year in YEARS:
    # Premium inflation: ~7% per year
    inflation = 1.07 ** (year - 2022)
    n = N_PER_YEAR

    age_band = rng.choice(
        ["17-25", "26-35", "36-50", "51-65", "66+"],
        n, p=[0.10, 0.20, 0.35, 0.25, 0.10],
    )
    ncb          = rng.choice([0, 1, 2, 3, 4, 5], n, p=[0.10, 0.10, 0.15, 0.20, 0.20, 0.25])
    vehicle_group = rng.choice(["A", "B", "C", "D", "E"], n, p=[0.20, 0.25, 0.25, 0.20, 0.10])
    region = rng.choice(
        ["London", "SouthEast", "Midlands", "North", "Scotland", "Wales"],
        n, p=[0.18, 0.20, 0.22, 0.25, 0.10, 0.05],
    )
    annual_mileage = rng.choice(
        ["<5k", "5k-10k", "10k-15k", "15k+"], n, p=[0.15, 0.35, 0.35, 0.15]
    )
    exposure = rng.uniform(0.3, 1.0, n)

    # True frequency
    age_freq   = {"17-25": 0.12, "26-35": 0.07, "36-50": 0.05, "51-65": 0.04, "66+": 0.06}
    freq_base  = np.array([age_freq[a] for a in age_band])
    veh_freq   = {"A": 0.85, "B": 0.95, "C": 1.00, "D": 1.10, "E": 1.25}
    freq_base *= np.array([veh_freq[v] for v in vehicle_group])
    reg_freq   = {"London": 1.15, "SouthEast": 1.05, "Midlands": 1.00,
                  "North": 0.95, "Scotland": 0.90, "Wales": 0.92}
    freq_base *= np.array([reg_freq[r] for r in region])
    mil_freq   = {"<5k": 0.75, "5k-10k": 0.90, "10k-15k": 1.05, "15k+": 1.30}
    freq_base *= np.array([mil_freq[m] for m in annual_mileage])
    claim_count = rng.poisson(freq_base * exposure)

    # Severity
    sev_base    = 2_800 * inflation
    veh_sev     = {"A": 0.75, "B": 0.90, "C": 1.00, "D": 1.15, "E": 1.40}
    sev_vehicle = np.array([veh_sev[v] for v in vehicle_group])
    reg_sev     = {"London": 1.20, "SouthEast": 1.10, "Midlands": 1.00,
                   "North": 0.95, "Scotland": 0.88, "Wales": 0.92}
    sev_region  = np.array([reg_sev[r] for r in region])
    mean_sev    = sev_base * sev_vehicle * sev_region
    incurred_loss = np.where(
        claim_count > 0,
        rng.gamma(shape=2.0, scale=mean_sev / 2.0) * claim_count,
        0.0,
    )
    earned_premium = (
        sev_base * freq_base / 0.72 * inflation
        * rng.uniform(0.94, 1.06, n)
    )

    cohorts.append(pl.DataFrame({
        "policy_id":      [f"{year}-{i:06d}" for i in range(n)],
        "accident_year":  [year] * n,
        "age_band":       age_band,
        "ncb":            ncb.tolist(),
        "vehicle_group":  vehicle_group,
        "region":         region,
        "annual_mileage": annual_mileage,
        "exposure":       exposure.tolist(),
        "earned_premium": earned_premium.tolist(),
        "claim_count":    claim_count.tolist(),
        "incurred_loss":  incurred_loss.tolist(),
    }))

raw = pl.concat(cohorts)

print(f"Total policies:    {len(raw):,}")
print(f"Overall frequency: {raw['claim_count'].sum() / raw['exposure'].sum():.4f}")
print(f"Average severity:  £{raw.filter(pl.col('incurred_loss') > 0)['incurred_loss'].mean():,.0f}")
print(f"Loss ratio:        {raw['incurred_loss'].sum() / raw['earned_premium'].sum():.3f}")

# Write raw data to Unity Catalog
(
    spark.createDataFrame(raw.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["raw"])
)
print(f"\nWritten to {TABLES['raw']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 5: Feature engineering - the transform layer
# MAGIC
# MAGIC All feature transforms are defined as pure functions in a single dictionary.
# MAGIC The same `apply_transforms` function is called at training, validation, and
# MAGIC scoring time. This prevents the training/serving skew that is the most common
# MAGIC source of model-to-production mismatch in pricing teams.

# COMMAND ----------

from __future__ import annotations

# NCB: encode as deficit from maximum
NCB_MAX = 5
def encode_ncb(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns((NCB_MAX - pl.col("ncb")).alias("ncb_deficit"))

# Vehicle group: ordinal A=1 through E=5
VEHICLE_ORD = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
def encode_vehicle(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("vehicle_group").replace(VEHICLE_ORD).cast(pl.Int32).alias("vehicle_ord")
    )

# Age band: midpoint as continuous feature
AGE_MIDPOINTS = {"17-25": 21, "26-35": 30, "36-50": 43, "51-65": 58, "66+": 72}
def encode_age(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("age_band").replace(AGE_MIDPOINTS).cast(pl.Float64).alias("age_mid")
    )

# Annual mileage: ordinal
MILEAGE_ORD = {"<5k": 1, "5k-10k": 2, "10k-15k": 3, "15k+": 4}
def encode_mileage(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("annual_mileage").replace(MILEAGE_ORD).cast(pl.Int32).alias("mileage_ord")
    )

# Region: kept as string - CatBoost handles it natively
# Log exposure
def log_exposure(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("exposure").log().alias("log_exposure"))

TRANSFORMS    = [encode_ncb, encode_vehicle, encode_age, encode_mileage, log_exposure]
FEATURE_COLS  = ["ncb_deficit", "vehicle_ord", "age_mid", "mileage_ord", "region"]
CAT_FEATURES  = ["region"]

def apply_transforms(df: pl.DataFrame) -> pl.DataFrame:
    for transform in TRANSFORMS:
        df = transform(df)
    return df

# Apply to raw data
raw_pl = pl.from_pandas(spark.table(TABLES["raw"]).toPandas())
features_pl = apply_transforms(raw_pl)

print(f"Feature columns: {FEATURE_COLS}")
print(f"Categorical:     {CAT_FEATURES}")
print(features_pl.select(FEATURE_COLS + ["claim_count", "incurred_loss", "exposure"]).head(3))

(
    spark.createDataFrame(features_pl.to_pandas())
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLES["features"])
)
print(f"\nFeatures written to {TABLES['features']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 6: Walk-forward cross-validation
# MAGIC
# MAGIC `insurance-cv` provides walk-forward folds that respect policy year ordering
# MAGIC and the IBNR development lag. We validate the model structure before committing
# MAGIC to full training.

# COMMAND ----------

from insurance_cv import WalkForwardCV

features_pd = spark.table(TABLES["features"]).toPandas()

cv = WalkForwardCV(
    year_col="accident_year",
    min_train_years=2,
    ibnr_buffer_years=1,
    n_splits=3,
)

print("Walk-forward folds:")
for fold_num, (train_idx, val_idx) in enumerate(cv.split(features_pd), 1):
    if len(val_idx) == 0:
        print(f"  Fold {fold_num}: final training fold (no validation year)")
        continue
    train_years = sorted(features_pd.iloc[train_idx]["accident_year"].unique())
    val_years   = sorted(features_pd.iloc[val_idx]["accident_year"].unique())
    print(f"  Fold {fold_num}: train={train_years}  val={val_years}")

# COMMAND ----------

# Frequency model CV
freq_cv_scores = []

for fold_num, (train_idx, val_idx) in enumerate(cv.split(features_pd), 1):
    if len(val_idx) == 0:
        continue

    train = features_pd.iloc[train_idx]
    val   = features_pd.iloc[val_idx]

    freq_train_pool = cb.Pool(
        train[FEATURE_COLS], label=train["claim_count"],
        weight=train["exposure"], cat_features=CAT_FEATURES,
    )
    freq_val_pool = cb.Pool(
        val[FEATURE_COLS], label=val["claim_count"],
        weight=val["exposure"], cat_features=CAT_FEATURES,
    )

    cv_model = cb.CatBoostRegressor(
        loss_function="Poisson", iterations=300,
        learning_rate=0.05, depth=4, random_seed=42, verbose=0,
    )
    cv_model.fit(freq_train_pool, eval_set=freq_val_pool)

    val_pred = cv_model.predict(freq_val_pool)
    y  = val["claim_count"].values / val["exposure"].values
    mu = val_pred
    mask = (y > 0) & (mu > 0)
    deviance = 2 * np.sum(
        np.where(mask, y[mask] * np.log(y[mask] / mu[mask]) - (y[mask] - mu[mask]), mu)
    )
    score = deviance / len(val)
    freq_cv_scores.append(score)
    print(f"Fold {fold_num} frequency deviance: {score:.5f}")

print(f"Mean CV deviance: {np.mean(freq_cv_scores):.5f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 7: Hyperparameter tuning with Optuna
# MAGIC
# MAGIC We tune frequency and severity separately. The most recent available
# MAGIC accident years are used as the validation set - the most realistic
# MAGIC out-of-time validation.

# COMMAND ----------

TUNE_TRAIN_YEARS = [2022, 2023, 2024]
TUNE_VAL_YEARS   = [2025]

tune_train = features_pd[features_pd["accident_year"].isin(TUNE_TRAIN_YEARS)]
tune_val   = features_pd[features_pd["accident_year"].isin(TUNE_VAL_YEARS)]

# --- Frequency tuning ---
def freq_objective(trial: optuna.Trial) -> float:
    params = {
        "loss_function":  "Poisson",
        "iterations":     trial.suggest_int("iterations", 200, 600),
        "learning_rate":  trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth":          trial.suggest_int("depth", 3, 6),
        "l2_leaf_reg":    trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "subsample":      trial.suggest_float("subsample", 0.6, 1.0),
        "random_seed":    42,
        "verbose":        0,
    }
    tp = cb.Pool(tune_train[FEATURE_COLS], label=tune_train["claim_count"],
                 weight=tune_train["exposure"], cat_features=CAT_FEATURES)
    vp = cb.Pool(tune_val[FEATURE_COLS], label=tune_val["claim_count"],
                 weight=tune_val["exposure"], cat_features=CAT_FEATURES)
    m = cb.CatBoostRegressor(**params)
    m.fit(tp, eval_set=vp)
    mu = m.predict(vp)
    y  = tune_val["claim_count"].values / tune_val["exposure"].values
    mask = (y > 0) & (mu > 0)
    dev = 2 * np.sum(np.where(mask, y[mask] * np.log(y[mask] / mu[mask]) - (y[mask] - mu[mask]), mu))
    return dev / len(tune_val)

freq_study = optuna.create_study(direction="minimize")
freq_study.optimize(freq_objective, n_trials=N_TRIALS, show_progress_bar=True)

best_freq_params = {**freq_study.best_params, "loss_function": "Poisson", "random_seed": 42, "verbose": 0}
print(f"Best frequency deviance: {freq_study.best_value:.5f}")
print(f"Best params: {best_freq_params}")

# COMMAND ----------

# --- Severity tuning ---
claims_only = features_pd[features_pd["claim_count"] > 0].copy()
mean_sev_all = claims_only["incurred_loss"] / claims_only["claim_count"]

tune_train_sev = claims_only[claims_only["accident_year"].isin(TUNE_TRAIN_YEARS)]
tune_val_sev   = claims_only[claims_only["accident_year"].isin(TUNE_VAL_YEARS)]

def sev_objective(trial: optuna.Trial) -> float:
    params = {
        "loss_function":  "Gamma",
        "iterations":     trial.suggest_int("iterations", 200, 600),
        "learning_rate":  trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth":          trial.suggest_int("depth", 3, 6),
        "l2_leaf_reg":    trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_seed":    42,
        "verbose":        0,
    }
    y_train_s = (tune_train_sev["incurred_loss"] / tune_train_sev["claim_count"]).values
    y_val_s   = (tune_val_sev["incurred_loss"] / tune_val_sev["claim_count"]).values
    tp = cb.Pool(tune_train_sev[FEATURE_COLS], label=y_train_s,
                 weight=tune_train_sev["claim_count"], cat_features=CAT_FEATURES)
    vp = cb.Pool(tune_val_sev[FEATURE_COLS], label=y_val_s,
                 weight=tune_val_sev["claim_count"], cat_features=CAT_FEATURES)
    m = cb.CatBoostRegressor(**params)
    m.fit(tp, eval_set=vp)
    y_pred = m.predict(vp)
    rmse   = float(np.sqrt(np.mean((y_val_s - y_pred) ** 2)))
    return rmse

sev_study = optuna.create_study(direction="minimize")
sev_study.optimize(sev_objective, n_trials=N_TRIALS, show_progress_bar=True)

best_sev_params = {**sev_study.best_params, "loss_function": "Gamma", "random_seed": 42, "verbose": 0}
print(f"Best severity RMSE: {sev_study.best_value:.2f}")
print(f"Best params: {best_sev_params}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 8: Full model training with MLflow tracking
# MAGIC
# MAGIC Train on all four accident years. Log everything: parameters, metrics,
# MAGIC feature importance, model artefacts. The MLflow run IDs become the audit
# MAGIC trail for every downstream output.

# COMMAND ----------

mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

# --- Frequency model ---
with mlflow.start_run(run_name="frequency_catboost_poisson") as freq_run:
    mlflow.log_params(best_freq_params)
    mlflow.log_param("feature_cols",  FEATURE_COLS)
    mlflow.log_param("cat_features",  CAT_FEATURES)
    mlflow.log_param("training_years", list(YEARS))
    mlflow.log_param("n_policies",    len(features_pd))

    full_freq_pool = cb.Pool(
        features_pd[FEATURE_COLS],
        label=features_pd["claim_count"],
        weight=features_pd["exposure"],
        cat_features=CAT_FEATURES,
    )

    freq_model = cb.CatBoostRegressor(**best_freq_params)
    freq_model.fit(full_freq_pool)

    freq_pred = freq_model.predict(full_freq_pool)
    features_pd["freq_pred"] = freq_pred

    mlflow.log_metric("train_mean_freq",   float(freq_pred.mean()))
    mlflow.log_metric("actual_mean_freq",  float((features_pd["claim_count"] / features_pd["exposure"]).mean()))

    importances = freq_model.get_feature_importance(type="FeatureImportance")
    imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances}).sort_values("importance", ascending=False)
    fig_imp, ax_imp = plt.subplots(figsize=(7, 4))
    ax_imp.barh(imp_df["feature"].tolist(), imp_df["importance"].tolist(), color="steelblue")
    ax_imp.set_xlabel("Feature importance")
    ax_imp.set_title("Frequency model feature importance")
    plt.tight_layout()
    mlflow.log_figure(fig_imp, "freq_feature_importance.png")
    plt.show()

    mlflow.catboost.log_model(freq_model, artifact_path="frequency_model")
    FREQ_RUN_ID = freq_run.info.run_id

print(f"Frequency model run: {FREQ_RUN_ID}")
print(f"Mean predicted freq: {freq_pred.mean():.5f}")

# COMMAND ----------

# --- Severity model ---
claims_pd = features_pd[features_pd["claim_count"] > 0].copy()
y_sev     = (claims_pd["incurred_loss"] / claims_pd["claim_count"]).values

with mlflow.start_run(run_name="severity_catboost_gamma") as sev_run:
    mlflow.log_params(best_sev_params)
    mlflow.log_param("n_claim_records", len(claims_pd))

    full_sev_pool = cb.Pool(
        claims_pd[FEATURE_COLS],
        label=y_sev,
        weight=claims_pd["claim_count"],
        cat_features=CAT_FEATURES,
    )

    sev_model = cb.CatBoostRegressor(**best_sev_params)
    sev_model.fit(full_sev_pool)

    sev_pred = sev_model.predict(full_sev_pool)
    claims_pd["sev_pred"] = sev_pred

    mlflow.log_metric("train_mean_severity", float(sev_pred.mean()))
    mlflow.log_metric("actual_mean_severity", float(y_sev.mean()))
    mlflow.catboost.log_model(sev_model, artifact_path="severity_model")
    SEV_RUN_ID = sev_run.info.run_id

print(f"Severity model run:  {SEV_RUN_ID}")
print(f"Mean predicted sev:  £{sev_pred.mean():,.0f}")

# Merge severity back and compute pure premium
features_pd = features_pd.merge(
    claims_pd[["policy_id", "sev_pred"]], on="policy_id", how="left"
)
features_pd["sev_pred"]     = features_pd["sev_pred"].fillna(float(sev_pred.mean()))
features_pd["pure_premium"] = features_pd["freq_pred"] * features_pd["sev_pred"]

print(f"\nPure premium summary:")
print(f"  Mean:   £{features_pd['pure_premium'].mean():,.0f}")
print(f"  Median: £{features_pd['pure_premium'].median():,.0f}")
print(f"  Model LR: {features_pd['pure_premium'].sum() / features_pd['earned_premium'].sum():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 9: SHAP relativities
# MAGIC
# MAGIC SHAP values decompose each prediction into per-feature contributions.
# MAGIC `shap-relativities` converts those contributions into multiplicative factor
# MAGIC tables: the format a rating engine expects.
# MAGIC
# MAGIC The `link="log"` parameter is required for Poisson and Gamma models, which
# MAGIC both use a log link. SHAP values are additive on the log scale; the library
# MAGIC exponentiates them to recover multiplicative relativities.

# COMMAND ----------

from shap_relativities import SHAPRelativities

# Frequency relativities
freq_rel = SHAPRelativities(
    model=freq_model,
    data=features_pd[FEATURE_COLS],
    cat_features=CAT_FEATURES,
    link="log",
)
freq_rel.fit()

# Severity relativities
sev_rel = SHAPRelativities(
    model=sev_model,
    data=claims_pd[FEATURE_COLS],
    cat_features=CAT_FEATURES,
    link="log",
)
sev_rel.fit()

# Build combined pure premium relativities
pp_relativities = {}
for feat in FEATURE_COLS:
    freq_table = freq_rel.relativity_table(feat).rename(columns={"relativity": "freq_rel"})
    sev_table  = sev_rel.relativity_table(feat).rename(columns={"relativity": "sev_rel"})
    combined   = freq_table.merge(sev_table, on=feat, how="outer")
    combined["pp_relativity"] = combined["freq_rel"] * combined["sev_rel"]
    pp_relativities[feat] = combined
    print(f"\n{feat} - pure premium relativities:")
    print(combined[[feat, "freq_rel", "sev_rel", "pp_relativity"]].to_string(index=False))

# COMMAND ----------

# Write relativities to Delta
rel_records = []
for feat, table in pp_relativities.items():
    for _, row in table.iterrows():
        rel_records.append({
            "factor":          feat,
            "level":           str(row[feat]),
            "freq_relativity": float(row.get("freq_rel", 1.0)),
            "sev_relativity":  float(row.get("sev_rel", 1.0)),
            "pp_relativity":   float(row.get("pp_relativity", 1.0)),
            "freq_run_id":     FREQ_RUN_ID,
            "sev_run_id":      SEV_RUN_ID,
        })

rel_df = pd.DataFrame(rel_records)
spark.createDataFrame(rel_df).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(TABLES["relativities"])
print(f"\nRelativities written to {TABLES['relativities']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 10: Conformal prediction intervals
# MAGIC
# MAGIC We use `insurance-conformal` to generate calibrated 90% prediction intervals
# MAGIC on pure premium. The `pearson_weighted` non-conformity score accounts for
# MAGIC the variance scaling of insurance data: large risks get wide intervals,
# MAGIC small risks get narrow ones.
# MAGIC
# MAGIC The calibration set is the most recent accident year (2025). In production,
# MAGIC use a genuinely held-out period - ideally the most recent quarter that is
# MAGIC nearly fully developed.

# COMMAND ----------

from insurance_conformal import InsuranceConformalPredictor

# Temporal split: calibrate on 2025, score on everything else
cal_mask   = features_pd["accident_year"] == 2025
calibration = features_pd[cal_mask].copy()
scoring     = features_pd[~cal_mask].copy()

print(f"Calibration set:  {len(calibration):,} policies (accident year 2025)")
print(f"Scoring set:      {len(scoring):,} policies (accident years 2022-2024)")

# We use a Tweedie conformal predictor on pure premium
# The model here is the frequency model; we compose with severity predictions
# The library wraps any CatBoost model that produces non-negative predictions
cp = InsuranceConformalPredictor(
    model=freq_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)

# Calibrate on held-out 2025 data
# The target here is pure premium rate (per unit exposure)
y_cal_pp = (calibration["incurred_loss"] / calibration["exposure"]).values
cp.calibrate(calibration[FEATURE_COLS], y_cal_pp)

print(f"\nCalibration quantile (alpha=0.10): {cp.calibration_quantile(alpha=0.10):.4f}")

# Generate intervals for the scoring set
intervals = cp.predict_interval(scoring[FEATURE_COLS], alpha=0.10)
scoring["pp_lower_90"] = intervals["lower"].values
scoring["pp_upper_90"] = intervals["upper"].values
scoring["pp_width_90"] = (intervals["upper"] - intervals["lower"]).values

print(f"\nInterval summary (scoring set):")
print(f"  Mean interval width: {scoring['pp_width_90'].mean():.4f}")
print(f"\nMean width by vehicle_group:")
print(scoring.groupby("vehicle_group")["pp_width_90"].mean().round(4).sort_values(ascending=False))

# Coverage diagnostic
y_score_pp  = (scoring["incurred_loss"] / scoring["exposure"]).values
diag        = cp.coverage_by_decile(scoring[FEATURE_COLS], y_score_pp, alpha=0.10)
marginal_cov = ((intervals["lower"].values <= y_score_pp) & (y_score_pp <= intervals["upper"].values)).mean()
print(f"\nMarginal coverage on scoring set: {marginal_cov:.4f}  (target: 0.90)")
print(diag.to_string(index=False))

# COMMAND ----------

# Write conformal intervals to Delta
spark.createDataFrame(
    scoring[["policy_id", "pure_premium", "pp_lower_90", "pp_upper_90"]].assign(
        freq_run_id=FREQ_RUN_ID,
        sev_run_id=SEV_RUN_ID,
        marginal_coverage=float(marginal_cov),
    )
).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(TABLES["conformal_intervals"])
print(f"Conformal intervals written to {TABLES['conformal_intervals']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 11: Credibility blending
# MAGIC
# MAGIC Model relativities are what the data says. The incumbent tariff relativities
# MAGIC reflect years of actuarial judgement and older data. Buhlmann-Straub credibility
# MAGIC provides a principled weighting between the two, with the credibility weight `z`
# MAGIC scaling with observed exposure.
# MAGIC
# MAGIC A `z` near 1.0 means the model has enough data to stand alone. A `z` near 0.0
# MAGIC means the cell is thin and the incumbent should dominate.

# COMMAND ----------

from credibility import BuhlmannStraub

# Incumbent relativities: what the current tariff says
# In production: read these from your rating engine or tariff management system
INCUMBENT = {
    "ncb_deficit": {0: 0.72, 1: 0.82, 2: 0.92, 3: 1.00, 4: 1.14, 5: 1.32},
    "vehicle_ord": {1: 0.82, 2: 0.93, 3: 1.00, 4: 1.12, 5: 1.38},
    "age_mid":     {21: 1.90, 30: 1.28, 43: 1.00, 58: 0.88, 72: 1.14},
    "mileage_ord": {1: 0.75, 2: 0.92, 3: 1.05, 4: 1.28},
}

blended_records = []

# Blend all features except region (categorical with insufficient ordered levels for BS)
for feat in [f for f in FEATURE_COLS if f != "region"]:
    model_table = pp_relativities[feat].copy()

    # Exposure weight per level
    exp_by_level = features_pd.groupby(feat)["exposure"].sum().reset_index()
    model_table  = model_table.merge(exp_by_level, on=feat, how="left")

    incumbent = INCUMBENT.get(feat, {})
    bs        = BuhlmannStraub()

    for _, row in model_table.iterrows():
        level     = row[feat]
        model_rel = float(row.get("pp_relativity", 1.0))
        inc_rel   = float(incumbent.get(level, 1.0))
        exp       = float(row.get("exposure", 0.0))

        bs.add_observation(
            group=feat,
            level=str(level),
            model_value=model_rel,
            incumbent_value=inc_rel,
            weight=exp,
        )

    blended = bs.blend()

    for level_str, blend_result in blended.items():
        blended_records.append({
            "factor":              feat,
            "level":               level_str,
            "model_relativity":    float(blend_result["model"]),
            "incumbent_relativity": float(blend_result["incumbent"]),
            "credibility_weight":  float(blend_result["z"]),
            "blended_relativity":  float(blend_result["blended"]),
        })
        print(
            f"{feat}={level_str}: model={blend_result['model']:.4f}  "
            f"incumbent={blend_result['incumbent']:.4f}  "
            f"z={blend_result['z']:.3f}  "
            f"blended={blend_result['blended']:.4f}"
        )

blended_df = pd.DataFrame(blended_records)
spark.createDataFrame(blended_df).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(TABLES["blended_relativities"])
print(f"\nBlended relativities written to {TABLES['blended_relativities']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 12: Rate optimisation
# MAGIC
# MAGIC The blended relativities tell us where each factor level should sit.
# MAGIC The rate optimiser decides how much to move each factor's relativities
# MAGIC as a whole to hit the portfolio loss ratio target, subject to volume,
# MAGIC ENBP, and movement constraints.

# COMMAND ----------

from rate_optimiser import (
    PolicyData, FactorStructure, RateChangeOptimiser, EfficientFrontier,
    LossRatioConstraint, VolumeConstraint, ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

# Build optimisation dataset from the full features frame
rng2        = np.random.default_rng(2026)
opt_df      = features_pd[["policy_id", "earned_premium", "pure_premium", "exposure"]].copy()
opt_df["technical_premium"] = features_pd["pure_premium"]
opt_df["current_premium"]   = features_pd["earned_premium"]

market_premium = opt_df["current_premium"] / 0.96 * rng2.uniform(0.92, 1.04, len(opt_df))
opt_df["market_premium"]  = market_premium
price_ratio               = opt_df["current_premium"] / market_premium
opt_df["renewal_prob"]    = expit(0.8 + (-1.8) * np.log(price_ratio.clip(lower=0.5)))
opt_df["renewal_flag"]    = True
opt_df["channel"]         = "PCW"

# Factor columns: normalised relativities per policy
RATE_FACTORS = ["f_ncb_deficit", "f_vehicle_ord", "f_age_mid", "f_mileage_ord"]
for fact in RATE_FACTORS:
    base = fact.replace("f_", "")
    if base in features_pd.columns:
        col = features_pd[base].values.astype(float)
        opt_df[fact] = col / (col.mean() + 1e-9)
    else:
        opt_df[fact] = 1.0

current_lr = float(opt_df["technical_premium"].sum() / opt_df["current_premium"].sum())
print(f"Current portfolio LR: {current_lr:.3f}  (target: {LR_TARGET})")

policy_data = PolicyData(opt_df)
fs = FactorStructure(
    factor_names=RATE_FACTORS,
    factor_values=opt_df[RATE_FACTORS],
    renewal_factor_names=[],
)

demand_params = LogisticDemandParams(intercept=0.8, price_coef=-1.8, tenure_coef=0.0)
demand_model  = make_logistic_demand(demand_params)

# COMMAND ----------

# Solve
opt_solver = RateChangeOptimiser(
    data=policy_data, demand=demand_model, factor_structure=fs,
)
opt_solver.add_constraint(LossRatioConstraint(bound=LR_TARGET))
opt_solver.add_constraint(VolumeConstraint(bound=VOLUME_FLOOR))
opt_solver.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.12, n_factors=fs.n_factors))
opt_solver.add_constraint(ENBPConstraint())

result = opt_solver.solve()
print(result.summary())

print(f"\nFactor adjustments:")
for factor_name, adj in result.factor_adjustments.items():
    direction = "up" if adj > 1.0 else ("down" if adj < 1.0 else "flat")
    print(f"  {factor_name:25s}: {adj:.4f}  ({(adj - 1) * 100:+.1f}%)  [{direction}]")

print(f"\nShadow prices:")
for cname, price in result.shadow_prices.items():
    print(f"  {cname:20s}: {price:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 13: Efficient frontier

# COMMAND ----------

lr_targets     = np.linspace(current_lr, 0.65, 20)
frontier_obj   = EfficientFrontier(
    policy_data=policy_data,
    factor_structure=fs,
    demand_model=demand_model,
    base_constraints=[
        VolumeConstraint(minimum=0.95),
        FactorBoundsConstraint(lower=0.88, upper=1.15),
        ENBPConstraint(),
    ],
)
frontier_results = frontier_obj.trace(lr_targets)

frontier_df = pd.DataFrame([
    {
        "lr_target":     r.lr_target,
        "achieved_lr":   r.projected_loss_ratio,
        "volume_ratio":  r.projected_volume_ratio,
        "dislocation":   r.objective_value,
        "feasible":      r.status == "optimal",
        "shadow_price_lr": r.shadow_prices.get("loss_ratio", None),
    }
    for r in frontier_results
])

feasible = frontier_df[frontier_df["feasible"]].copy()
print("Efficient frontier (feasible points):")
print(feasible[["lr_target", "achieved_lr", "volume_ratio", "shadow_price_lr"]].to_string(index=False))

# COMMAND ----------

# Frontier plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(feasible["achieved_lr"] * 100, feasible["volume_ratio"] * 100,
         marker="o", color="steelblue", linewidth=2, markersize=6)
ax1.scatter([current_lr * 100], [100], color="green", zorder=5, s=80, label="Current")
ax1.scatter([result.expected_lr * 100], [result.expected_volume_ratio * 100],
            color="red", zorder=5, s=80, label="Optimal solve")
ax1.axvline(LR_TARGET * 100, color="red", linestyle="--", alpha=0.5, label=f"Target {LR_TARGET:.0%}")
ax1.set_xlabel("Expected loss ratio (%)")
ax1.set_ylabel("Volume retention (%)")
ax1.set_title("Efficient frontier: loss ratio vs. volume")
ax1.invert_xaxis()
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

ax2 = axes[1]
shadow = feasible["shadow_price_lr"].dropna()
lr_tgt = feasible.loc[shadow.index, "lr_target"]
ax2.plot(lr_tgt * 100, shadow, marker="o", color="darkorange", linewidth=2, markersize=6)
ax2.axhline(0.30, color="red", linestyle="--", alpha=0.5, label="Threshold 0.30")
ax2.set_xlabel("LR target (%)")
ax2.set_ylabel("Shadow price on LR constraint")
ax2.set_title("Marginal cost of LR improvement")
ax2.invert_xaxis()
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)

plt.suptitle("End-to-end pipeline: rate optimisation frontier", fontsize=12)
plt.tight_layout()
plt.show()

# Write frontier
spark.createDataFrame(frontier_df).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(TABLES["efficient_frontier"])
print(f"Frontier written to {TABLES['efficient_frontier']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 14: Output pack
# MAGIC
# MAGIC Write a summary table with the pipeline's key results and the MLflow run IDs
# MAGIC as the audit trail. Every number in the rate change pack can be traced back
# MAGIC to a specific model version and training run.

# COMMAND ----------

from datetime import datetime

summary = {
    "pipeline_run_date":      datetime.now().isoformat(),
    "schema":                 f"{CATALOG}.{SCHEMA}",
    "n_policies":             int(len(features_pd)),
    "accident_years":         str(list(YEARS)),
    "freq_mlflow_run_id":     FREQ_RUN_ID,
    "sev_mlflow_run_id":      SEV_RUN_ID,
    "current_lr":             float(current_lr),
    "target_lr":              float(LR_TARGET),
    "projected_lr":           float(result.projected_loss_ratio),
    "projected_volume_ratio": float(result.projected_volume_ratio),
    "optimisation_status":    result.status,
    "factor_adjustments":     str({k: round(float(v), 4) for k, v in result.factor_adjustments.items()}),
    "shadow_price_lr":        float(result.shadow_prices.get("loss_ratio", 0.0)),
    "shadow_price_volume":    float(result.shadow_prices.get("volume", 0.0)),
    "conformal_coverage":     float(marginal_cov),
    "tables":                 str(list(TABLES.keys())),
}

spark.createDataFrame(pd.DataFrame([summary])).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable(TABLES["rate_change"])
print(f"Output pack written to {TABLES['rate_change']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What Unity Catalog contains after this pipeline
# MAGIC
# MAGIC | Table | Contents |
# MAGIC |-------|---------|
# MAGIC | `raw_policies` | 200,000 synthetic UK motor policies, 4 accident years |
# MAGIC | `features` | Engineered features, one row per policy |
# MAGIC | `relativities` | SHAP-derived frequency, severity, and pure premium relativities |
# MAGIC | `conformal_intervals` | 90% conformal prediction intervals on pure premium |
# MAGIC | `blended_relativities` | Credibility-blended relativities with factor-level weights |
# MAGIC | `rate_change` | Summary: factor adjustments, projected LR, shadow prices, run IDs |
# MAGIC | `efficient_frontier` | Full frontier: 20 (LR target, volume, dislocation) points |
# MAGIC
# MAGIC Every table carries the MLflow run IDs. Re-run with the same seed to reproduce
# MAGIC any number.

# COMMAND ----------

print("=" * 60)
print("MODULE 8 COMPLETE - END-TO-END PIPELINE")
print("=" * 60)
print(f"Schema:              {CATALOG}.{SCHEMA}")
print(f"Policies:            {len(features_pd):,}")
print(f"Accident years:      {list(YEARS)}")
print(f"")
print(f"Models:")
print(f"  Frequency (Poisson):  {FREQ_RUN_ID}")
print(f"  Severity (Gamma):     {SEV_RUN_ID}")
print(f"")
print(f"Relativities:        {len(rel_df)} factor-level entries")
print(f"Credibility blended: {len(blended_df)} factor-level entries")
print(f"")
print(f"Conformal intervals:")
print(f"  Coverage (scoring set): {marginal_cov:.4f}  (target: 0.90)")
print(f"")
print(f"Rate optimisation:")
print(f"  Current LR:   {current_lr:.4f}")
print(f"  Target LR:    {LR_TARGET:.4f}")
print(f"  Achieved LR:  {result.projected_loss_ratio:.4f}")
print(f"  Volume ratio: {result.projected_volume_ratio:.4f}")
print(f"")
print(f"Factor adjustments:")
for factor_name, adj in result.factor_adjustments.items():
    print(f"  {factor_name:25s}: {(adj - 1) * 100:+.1f}%")
print(f"")
print(f"Frontier points:     {len(feasible)} feasible")
print(f"")
print(f"All outputs in schema: {CATALOG}.{SCHEMA}")
print("=" * 60)
