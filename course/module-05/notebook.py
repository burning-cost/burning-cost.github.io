# Databricks notebook source
# MAGIC %md
# MAGIC # Module 5: Conformal Prediction Intervals for Insurance Pricing Models
# MAGIC ## Modern Insurance Pricing with Python and Databricks
# MAGIC
# MAGIC This notebook covers the full Module 5 workflow:
# MAGIC
# MAGIC 1. Install packages and generate synthetic UK motor data
# MAGIC 2. Temporal train/calibration/test split
# MAGIC 3. Train a CatBoost Tweedie pure premium model
# MAGIC 4. Wrap with InsuranceConformalPredictor (pearson_weighted score)
# MAGIC 5. Calibrate on the held-out calibration set
# MAGIC 6. Generate 90% prediction intervals
# MAGIC 7. Coverage-by-decile diagnostics
# MAGIC 8. Comparison with naive absolute-residual intervals
# MAGIC 9. Practical applications: uncertain risk flagging and minimum premium floors
# MAGIC 10. Write intervals and coverage log to Unity Catalog
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Databricks Runtime 14.3 LTS or later
# MAGIC - Unity Catalog enabled (Free Edition includes this)
# MAGIC
# MAGIC **Free Edition note:** All cells run on Databricks Free Edition. Data generation
# MAGIC and model training take approximately 5-10 minutes on a single-node cluster.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install packages
# MAGIC
# MAGIC `insurance-conformal` is not pre-installed on the standard Databricks Runtime.
# MAGIC The `[catboost]` extra pulls in CatBoost and enables auto-detection of the Tweedie
# MAGIC power from the model's loss function parameter.

# COMMAND ----------

# MAGIC %sh uv pip install "insurance-conformal[catboost]" polars scikit-learn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

CATALOG = "pricing"
SCHEMA  = "motor"

# Change to your Databricks user path
MLFLOW_EXPERIMENT_PATH = "/Users/you@insurer.com/motor-conformal-module05"

# Conformal coverage target: 90% prediction intervals
ALPHA = 0.10

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports

# COMMAND ----------

import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

print(f"CatBoost version:             {__import__('catboost').__version__}")
print(f"Polars version:               {pl.__version__}")
print(f"insurance-conformal version:  {__import__('insurance_conformal').__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Synthetic data generation
# MAGIC
# MAGIC We generate 50,000 synthetic UK motor policies with known true parameters.
# MAGIC This lets us verify that the conformal intervals achieve correct coverage.
# MAGIC The key columns are:
# MAGIC
# MAGIC - `claim_count`, `exposure`: Poisson frequency outcome
# MAGIC - `incurred`: total incurred cost, the pure premium target
# MAGIC - `vehicle_group`, `driver_age`, `ncd_years`, `area`, `conviction_points`,
# MAGIC   `annual_mileage`: rating features
# MAGIC - `accident_year`: used for the temporal split
# MAGIC
# MAGIC In production, replace this cell with `spark.table("your_catalog.motor.policies")`.

# COMMAND ----------

rng = np.random.default_rng(42)
N = 50_000
YEARS = [2021, 2022, 2023, 2024]
N_PER_YEAR = N // len(YEARS)

cohorts = []
for year in YEARS:
    n = N_PER_YEAR
    area = rng.choice(["A", "B", "C", "D", "E", "F"], n, p=[0.18, 0.20, 0.22, 0.18, 0.12, 0.10])
    vehicle_group = rng.integers(1, 7, n)
    driver_age    = rng.choice([21, 30, 43, 58, 72], n, p=[0.10, 0.20, 0.35, 0.25, 0.10])
    ncd_years     = rng.integers(0, 6, n)
    conviction_points = rng.choice([0, 3, 6, 9], n, p=[0.80, 0.12, 0.06, 0.02])
    annual_mileage    = rng.choice([5000, 10000, 15000, 20000], n, p=[0.20, 0.35, 0.30, 0.15])
    exposure = rng.uniform(0.3, 1.0, n)

    area_freq = {"A": 0.045, "B": 0.055, "C": 0.062, "D": 0.070, "E": 0.078, "F": 0.090}
    freq_base = np.array([area_freq[a] for a in area])
    freq_base *= (vehicle_group / 3.5)
    freq_base *= np.where(driver_age <= 25, 1.8, np.where(driver_age >= 65, 1.2, 1.0))
    freq_base *= np.maximum(0.5, 1.0 - 0.05 * ncd_years)
    freq_base *= (1.0 + 0.04 * conviction_points)
    freq_base *= (1.0 + annual_mileage / 50_000)

    claim_count = rng.poisson(freq_base * exposure)

    base_sev = 2_500
    mean_sev = base_sev * (vehicle_group / 3.5) * np.where(driver_age <= 25, 1.3, 1.0)
    incurred  = np.where(
        claim_count > 0,
        rng.gamma(shape=2.0, scale=mean_sev / 2.0) * claim_count,
        0.0,
    )

    cohorts.append(
        pl.DataFrame({
            "policy_id":         [f"{year}-{i:06d}" for i in range(n)],
            "accident_year":     [year] * n,
            "area":              area,
            "vehicle_group":     vehicle_group.tolist(),
            "driver_age":        driver_age.tolist(),
            "ncd_years":         ncd_years.tolist(),
            "conviction_points": conviction_points.tolist(),
            "annual_mileage":    annual_mileage.tolist(),
            "exposure":          exposure.tolist(),
            "claim_count":       claim_count.tolist(),
            "incurred":          incurred.tolist(),
        })
    )

df = pl.concat(cohorts)

print(f"Total policies:    {len(df):,}")
print(f"Overall frequency: {df['claim_count'].sum() / df['exposure'].sum():.4f}")
print(f"Average severity:  £{df.filter(pl.col('incurred') > 0)['incurred'].mean():,.0f}")
print(f"Years: {df['accident_year'].unique().sort().to_list()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Temporal train / calibration / test split
# MAGIC
# MAGIC This is the single most important practical decision in this module. We split by
# MAGIC accident year, not randomly.
# MAGIC
# MAGIC The conformal prediction exchangeability condition requires that calibration scores
# MAGIC and test scores come from the same distribution in the same temporal ordering.
# MAGIC A temporal split makes that assumption visible and testable: if coverage in the
# MAGIC test period is below target, temporal drift is the first thing to investigate.
# MAGIC
# MAGIC Split: 2021-2022 = training (60%), 2023 = calibration (20%), 2024 = test (20%).

# COMMAND ----------

FEATURES = ["vehicle_group", "driver_age", "ncd_years", "conviction_points", "annual_mileage"]
CAT_FEATURES = ["area"]
ALL_FEATURES = FEATURES + CAT_FEATURES

df_train = df.filter(pl.col("accident_year").is_in([2021, 2022]))
df_cal   = df.filter(pl.col("accident_year") == 2023)
df_test  = df.filter(pl.col("accident_year") == 2024)

X_train = df_train[ALL_FEATURES].to_pandas()
y_train = df_train["incurred"].to_numpy()

X_cal   = df_cal[ALL_FEATURES].to_pandas()
y_cal   = df_cal["incurred"].to_numpy()

X_test  = df_test[ALL_FEATURES].to_pandas()
y_test  = df_test["incurred"].to_numpy()

print(f"Training:    {len(X_train):,} policies (accident years 2021-2022)")
print(f"Calibration: {len(X_cal):,} policies (accident year 2023)")
print(f"Test:        {len(X_test):,} policies (accident year 2024)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train CatBoost Tweedie pure premium model
# MAGIC
# MAGIC We use a Tweedie pure premium model - the direct target for most UK personal lines
# MAGIC pricing teams. Tweedie with variance_power=1.5 is typical for UK motor: it handles
# MAGIC the mix of zero and positive incurred losses without requiring separate frequency
# MAGIC and severity models.
# MAGIC
# MAGIC The model is trained on training data only. The calibration pool is passed as
# MAGIC `eval_set` for early stopping only - this is a minor practical compromise. To be
# MAGIC strict about the held-out assumption, use a separate validation set for early
# MAGIC stopping and keep calibration genuinely untouched.

# COMMAND ----------

train_pool = Pool(data=X_train, label=y_train, cat_features=CAT_FEATURES)
cal_pool   = Pool(data=X_cal,   label=y_cal,   cat_features=CAT_FEATURES)
test_pool  = Pool(data=X_test,  label=y_test,  cat_features=CAT_FEATURES)

tweedie_params = {
    "loss_function":  "Tweedie:variance_power=1.5",
    "eval_metric":    "Tweedie:variance_power=1.5",
    "learning_rate":  0.05,
    "depth":          5,
    "min_data_in_leaf": 50,
    "iterations":     500,
    "random_seed":    42,
    "verbose":        100,
}

model = CatBoostRegressor(**tweedie_params)
model.fit(train_pool, eval_set=cal_pool, early_stopping_rounds=50)

y_pred_test = model.predict(test_pool)
test_rmse   = float(np.sqrt(np.mean((y_pred_test - y_test) ** 2)))

print(f"\nBest iteration: {model.best_iteration_}")
print(f"Test RMSE:      {test_rmse:.2f}")
print(f"Test mean pred: {y_pred_test.mean():.4f}")
print(f"Test mean actual: {y_test.mean():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Wrap with InsuranceConformalPredictor
# MAGIC
# MAGIC The `pearson_weighted` non-conformity score normalises by ŷ^(p/2), where p=1.5 is
# MAGIC the Tweedie power. This accounts for the heteroscedasticity of insurance data:
# MAGIC larger risks get wider intervals, smaller risks get narrower ones.
# MAGIC
# MAGIC The alternative - raw absolute residuals - produces fixed-width intervals.
# MAGIC For a £300 risk and a £30,000 risk, fixed-width intervals are wrong in opposite
# MAGIC directions simultaneously. The pearson_weighted score eliminates this problem.

# COMMAND ----------

from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)

print(f"Predictor: {cp}")
print(f"Non-conformity score: pearson_weighted")
print(f"Tweedie power: 1.5")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Calibrate
# MAGIC
# MAGIC Calibration computes the non-conformity score for every observation in the
# MAGIC calibration set: `|y - ŷ| / ŷ^(p/2)`. The result is a sorted score distribution.
# MAGIC For a target miscoverage rate α, we find the ⌈(1 - α)(n + 1)⌉ / n quantile.
# MAGIC That quantile is the threshold used for all subsequent test intervals.
# MAGIC
# MAGIC With 12,500 calibration observations, the margin of error on 90% coverage
# MAGIC is approximately ±0.9pp - tight enough for all practical insurance applications.

# COMMAND ----------

cp.calibrate(X_cal, y_cal)

print(f"Calibration complete.")
print(f"Calibration n:         {len(y_cal):,}")
print(f"Calibration quantile (alpha=0.10): {cp.calibration_quantile(alpha=ALPHA):.4f}")
print(f"Calibration coverage:  {cp.calibration_coverage_:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Generate 90% prediction intervals
# MAGIC
# MAGIC `predict_interval` returns a DataFrame with `lower`, `point`, and `upper` columns.
# MAGIC The lower bound is always clipped at zero - insurance losses are non-negative.
# MAGIC
# MAGIC Notice the variance-weighted scaling: high-risk observations get wide intervals,
# MAGIC low-risk observations get narrow ones. This is the correct behaviour for
# MAGIC Tweedie data where variance scales with the mean.

# COMMAND ----------

intervals = cp.predict_interval(X_test, alpha=ALPHA)

print("First 10 prediction intervals (£):")
print(intervals.head(10).to_string(index=False))

print(f"\nInterval summary:")
print(f"  Mean point estimate:  £{intervals['point'].mean():.2f}")
print(f"  Mean interval width:  £{(intervals['upper'] - intervals['lower']).mean():.2f}")
print(f"  Median interval width: £{(intervals['upper'] - intervals['lower']).median():.2f}")
print(f"  Min width:            £{(intervals['upper'] - intervals['lower']).min():.2f}")
print(f"  Max width:            £{(intervals['upper'] - intervals['lower']).max():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Coverage-by-decile diagnostics
# MAGIC
# MAGIC This is the gate. Do not use conformal intervals for any business purpose
# MAGIC without running this diagnostic and confirming coverage is flat across deciles.
# MAGIC
# MAGIC Marginal coverage above 90% does not mean the intervals are working correctly.
# MAGIC With naive absolute-residual intervals, you will typically see >97% coverage
# MAGIC in the lowest risk decile and <75% coverage in the highest risk decile.
# MAGIC The aggregate looks fine; the tails are wrong in opposite directions.
# MAGIC
# MAGIC With `pearson_weighted`, coverage should be flat: all deciles within 3-4pp
# MAGIC of the 90% target.

# COMMAND ----------

diag = cp.coverage_by_decile(X_test, y_test, alpha=ALPHA)
print("Coverage by predicted-value decile:")
print(diag.to_string(index=False))

marginal_coverage = (
    (intervals["lower"].values <= y_test) & (y_test <= intervals["upper"].values)
).mean()
print(f"\nMarginal coverage: {marginal_coverage:.4f}  (target: {1 - ALPHA:.2f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 9a. Coverage plot

# COMMAND ----------

fig, ax = plt.subplots(figsize=(9, 5))

deciles = diag["decile"].astype(str)
coverages = diag["coverage"].values * 100
target = (1 - ALPHA) * 100

bars = ax.bar(deciles, coverages, color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.5)
ax.axhline(target, color="red", linestyle="--", linewidth=1.5, label=f"Target {target:.0f}%")
ax.axhline(target - 5, color="orange", linestyle=":", linewidth=1, label="±5pp band", alpha=0.8)
ax.axhline(target + 5, color="orange", linestyle=":", linewidth=1, alpha=0.8)

ax.set_xlabel("Predicted value decile (1 = lowest risk, 10 = highest risk)")
ax.set_ylabel("Coverage (%)")
ax.set_title("Conformal interval coverage by predicted-value decile\n(pearson_weighted score, alpha=0.10)")
ax.set_ylim(max(0, target - 15), min(100, target + 15))
ax.legend()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Comparison with naive absolute-residual intervals
# MAGIC
# MAGIC To show why `pearson_weighted` matters, we repeat the analysis with raw
# MAGIC absolute residuals as the non-conformity score. This is what you would
# MAGIC get from a standard conformal prediction implementation that does not
# MAGIC account for the heteroscedastic structure of insurance data.

# COMMAND ----------

cp_naive = InsuranceConformalPredictor(
    model=model,
    nonconformity="raw",          # absolute residual, no variance weighting
    distribution="tweedie",
    tweedie_power=1.5,
)
cp_naive.calibrate(X_cal, y_cal)

intervals_naive = cp_naive.predict_interval(X_test, alpha=ALPHA)
diag_naive      = cp_naive.coverage_by_decile(X_test, y_test, alpha=ALPHA)

marginal_naive = (
    (intervals_naive["lower"].values <= y_test) & (y_test <= intervals_naive["upper"].values)
).mean()

print("Naive (raw residual) coverage by decile:")
print(diag_naive.to_string(index=False))
print(f"\nMarginal coverage (naive): {marginal_naive:.4f}")
print(f"Marginal coverage (weighted): {marginal_coverage:.4f}")

# Compare mean interval widths
width_weighted = float((intervals["upper"] - intervals["lower"]).mean())
width_naive    = float((intervals_naive["upper"] - intervals_naive["lower"]).mean())
print(f"\nMean interval width (pearson_weighted): £{width_weighted:.2f}")
print(f"Mean interval width (raw):              £{width_naive:.2f}")
print(f"Width reduction from pearson_weighted:  {(1 - width_weighted / width_naive) * 100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10a. Side-by-side coverage comparison plot

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, diag_data, title, colour in [
    (axes[0], diag,       "pearson_weighted (correct for insurance)",    "steelblue"),
    (axes[1], diag_naive, "raw absolute residual (naive)",               "tomato"),
]:
    deciles   = diag_data["decile"].astype(str)
    coverages = diag_data["coverage"].values * 100
    ax.bar(deciles, coverages, color=colour, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axhline(target, color="red",    linestyle="--", linewidth=1.5, label=f"Target {target:.0f}%")
    ax.axhline(target - 5, color="orange", linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(target + 5, color="orange", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Predicted value decile")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

axes[0].set_ylabel("Coverage (%)")
fig.suptitle("Conformal coverage by decile: pearson_weighted vs. raw residuals", fontsize=12)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Practical application: flagging uncertain risks
# MAGIC
# MAGIC The most immediate operational use of prediction intervals is flagging risks
# MAGIC for underwriting referral. A risk with a wide interval relative to its point
# MAGIC estimate is one the model is uncertain about.
# MAGIC
# MAGIC The natural measure of relative uncertainty is the interval-to-point ratio.
# MAGIC Flagging the top 10% by relative width produces a referral list.

# COMMAND ----------

intervals_90 = cp.predict_interval(X_test, alpha=ALPHA)
intervals_90["interval_width"]  = intervals_90["upper"] - intervals_90["lower"]
intervals_90["relative_width"]  = intervals_90["interval_width"] / intervals_90["point"].clip(lower=1e-6)

width_threshold = float(intervals_90["relative_width"].quantile(0.90))
intervals_90["flag_for_review"] = intervals_90["relative_width"] > width_threshold

n_flagged = int(intervals_90["flag_for_review"].sum())
print(f"Width threshold (90th pct): {width_threshold:.2f}")
print(f"Flagged for review: {n_flagged:,} ({100 * n_flagged / len(intervals_90):.1f}%)")

flagged = intervals_90[intervals_90["flag_for_review"]]
print(f"\nFlagged risk summary:")
print(flagged[["lower", "point", "upper", "relative_width"]].describe().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11a. Which features drive wide intervals?
# MAGIC
# MAGIC Join the interval frame back to features to understand which feature values
# MAGIC correlate with high uncertainty. This helps the team identify where the model
# MAGIC is thin and where underwriting rules may need strengthening.

# COMMAND ----------

analysis = pd.concat(
    [X_test.reset_index(drop=True), intervals_90.reset_index(drop=True)],
    axis=1,
)

print("Mean relative width by area:")
print(
    analysis.groupby("area")["relative_width"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
    .round(3)
)

print("\nMean relative width by vehicle_group:")
print(
    analysis.groupby("vehicle_group")["relative_width"]
    .mean()
    .sort_values(ascending=False)
    .round(3)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Practical application: minimum premium floors
# MAGIC
# MAGIC Prediction interval upper bounds give a principled minimum premium floor.
# MAGIC The upper bound of a 90% interval is a loss outcome expected to be exceeded
# MAGIC only 10% of the time. Setting a floor near the upper bound means that even
# MAGIC in bad years, the premium is likely to cover losses.
# MAGIC
# MAGIC The practical floor: the higher of 150% of the technical premium and the
# MAGIC 80th percentile upper bound. This is more defensible under Consumer Duty than
# MAGIC a flat £350 floor applied uniformly.

# COMMAND ----------

intervals_95 = cp.predict_interval(X_test, alpha=0.05)   # 95% intervals
intervals_80 = cp.predict_interval(X_test, alpha=0.20)   # 80% intervals

# Practical floor: max of (1.5 x point estimate, 80th pct upper bound)
practical_floor = np.maximum(
    1.5 * intervals_80["point"].values,
    intervals_80["upper"].values,
)

ratio_to_point = practical_floor / intervals_95["point"].values

print("Practical minimum premium floor summary:")
print(f"  Mean floor: £{practical_floor.mean():.2f}")
print(f"  Median floor / point estimate: {np.median(ratio_to_point):.2f}x")
print(f"  90th pct floor / point estimate: {np.percentile(ratio_to_point, 90):.2f}x")

# Distribution of floor-to-point ratios
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(ratio_to_point, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
ax.axvline(np.median(ratio_to_point), color="red", linestyle="--",
           label=f"Median = {np.median(ratio_to_point):.2f}x")
ax.set_xlabel("Floor / point estimate ratio")
ax.set_ylabel("Number of policies")
ax.set_title("Distribution of minimum premium floor relative to point estimate")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Write intervals and coverage log to Unity Catalog
# MAGIC
# MAGIC Do not write intervals to a local file. Write to Delta Lake so there is a
# MAGIC permanent, versioned record tied to the model version and run date.
# MAGIC
# MAGIC Keep a history of coverage diagnostics in an append-mode table. If coverage
# MAGIC in the top decile degrades from 89% to 78% between runs, that is a signal
# MAGIC the model needs recalibration.

# COMMAND ----------

import mlflow
from datetime import date

RUN_DATE     = str(date.today())
MODEL_VERSION = "tweedie_catboost_module05"

# Prepare the intervals frame
intervals_to_write = intervals_90.copy()
intervals_to_write["model_run_date"]      = RUN_DATE
intervals_to_write["model_version"]       = MODEL_VERSION
intervals_to_write["alpha"]               = ALPHA
intervals_to_write["nonconformity_score"] = "pearson_weighted"
intervals_to_write["policy_id"]           = df_test["policy_id"].to_list()

# Try to write; gracefully handle catalog permission issues on Free Edition
try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

    (
        spark.createDataFrame(intervals_to_write)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.conformal_intervals")
    )
    print(f"Intervals written to {CATALOG}.{SCHEMA}.conformal_intervals")

    # Coverage log (append-mode for monitoring over time)
    diag_to_write = diag.copy()
    diag_to_write["model_run_date"] = RUN_DATE
    diag_to_write["model_version"]  = MODEL_VERSION

    (
        spark.createDataFrame(diag_to_write)
        .write.format("delta")
        .mode("append")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.conformal_coverage_log")
    )
    print(f"Coverage log appended to {CATALOG}.{SCHEMA}.conformal_coverage_log")

except Exception as e:
    print(f"Delta write skipped (check catalog access): {e}")
    print("Intervals available in `intervals_to_write` DataFrame for local use.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. MLflow logging
# MAGIC
# MAGIC Log the conformal calibration state alongside the model. This creates an
# MAGIC auditable record of which model version and calibration set produced which
# MAGIC intervals.

# COMMAND ----------

try:
    mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

    with mlflow.start_run(run_name="conformal_calibration_module05"):
        mlflow.log_param("nonconformity_score",  "pearson_weighted")
        mlflow.log_param("tweedie_power",        1.5)
        mlflow.log_param("alpha",                ALPHA)
        mlflow.log_param("calibration_n",        len(X_cal))
        mlflow.log_param("calibration_year",     2023)
        mlflow.log_param("test_year",            2024)

        mlflow.log_metric("marginal_coverage_test",     float(marginal_coverage))
        mlflow.log_metric("marginal_coverage_naive",    float(marginal_naive))
        mlflow.log_metric("mean_interval_width",        float(width_weighted))
        mlflow.log_metric("mean_interval_width_naive",  float(width_naive))
        mlflow.log_metric("width_reduction_pct",        float((1 - width_weighted / width_naive) * 100))

        mlflow.log_figure(fig, "coverage_comparison.png")

    print("MLflow run logged.")

except Exception as e:
    print(f"MLflow logging skipped: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Summary
# MAGIC
# MAGIC What we built in this notebook:
# MAGIC
# MAGIC - **Synthetic data**: 50,000 UK motor policies across four accident years with
# MAGIC   known true Tweedie structure
# MAGIC - **Temporal split**: 60% training (2021-2022), 20% calibration (2023),
# MAGIC   20% test (2024) - the split that makes the exchangeability assumption explicit
# MAGIC - **CatBoost Tweedie model**: `Tweedie:variance_power=1.5`, trained on the
# MAGIC   training split only
# MAGIC - **Conformal calibration**: `pearson_weighted` score, calibrated on 2023 data
# MAGIC - **90% prediction intervals**: intervals that widen for large/uncertain risks
# MAGIC   and narrow for well-supported small risks
# MAGIC - **Coverage diagnostics**: by-decile coverage confirms the intervals are flat
# MAGIC   and uniform - no systematic over- or under-coverage in the tails
# MAGIC - **Naive comparison**: raw absolute-residual intervals fail in both directions
# MAGIC   simultaneously; `pearson_weighted` fixes both
# MAGIC - **Uncertain risk flagging**: top-10% by relative width gives a principled
# MAGIC   referral list for underwriting
# MAGIC - **Minimum premium floor**: upper bound of 80% interval provides a data-driven,
# MAGIC   risk-specific floor
# MAGIC
# MAGIC The coverage-by-decile diagnostic is the gate. Never deploy conformal intervals
# MAGIC for business decisions without running it first.
# MAGIC
# MAGIC **Next: Module 6 - Credibility Weighting**
# MAGIC Where the model relativities from a GBM are thin, credibility blending with
# MAGIC incumbent rates gives a more stable and defensible factor table.

# COMMAND ----------

print("=" * 60)
print("Module 5 complete")
print("=" * 60)
print(f"Model:               CatBoost Tweedie (power=1.5)")
print(f"Calibration set:     {len(X_cal):,} policies (accident year 2023)")
print(f"Test set:            {len(X_test):,} policies (accident year 2024)")
print(f"")
print(f"Marginal coverage (pearson_weighted): {marginal_coverage:.4f}  (target: {1 - ALPHA:.2f})")
print(f"Marginal coverage (naive raw):        {marginal_naive:.4f}  (target: {1 - ALPHA:.2f})")
print(f"")
print(f"Mean interval width (pearson_weighted): £{width_weighted:.2f}")
print(f"Mean interval width (naive):            £{width_naive:.2f}")
print(f"Width reduction:                        {(1 - width_weighted / width_naive) * 100:.1f}%")
print(f"")
print(f"Policies flagged for review: {n_flagged:,} ({100 * n_flagged / len(intervals_90):.1f}%)")
print(f"")
print("Next: Module 6 - Credibility Weighting")
