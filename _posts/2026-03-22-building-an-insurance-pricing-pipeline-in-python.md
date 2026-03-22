---
layout: post
title: "Building an Insurance Pricing Pipeline in Python: From Raw Claims to Production Tariff"
date: 2026-03-22
categories: [guides, tutorials]
tags: [python, pipeline, tutorial, insurance-datasets, shap-relativities, insurance-fairness, insurance-conformal, insurance-monitoring, catboost, fca]
description: "End-to-end UK motor pricing pipeline in Python: synthetic data generation, CatBoost frequency/severity models, GLM factor extraction, FCA fairness audit, conformal prediction intervals, and drift monitoring."
---
Most pricing tutorials stop at the model. You get a GLM fitted to the French MTPL dataset, a reasonable Gini, and a pat on the back. Then you try to apply the same approach to a live book and discover that the model is the easy part.

What a production pricing pipeline actually needs: a reproducible data ingestion layer, a frequency/severity split with proper exposure handling, a way to extract GLM-style factor tables from your GBM so the underwriting system can import them, a fairness audit that satisfies the FCA's current expectations, calibrated prediction intervals so you know how much to trust the point estimates, and monitoring that tells you when the model has drifted.

This tutorial builds that pipeline end to end using a set of open-source Python libraries maintained under the Burning Cost project. All libraries are on PyPI. We use synthetic UK motor data throughout — the kind of book a mid-sized personal lines insurer might run. We will be honest about where synthetic data diverges from reality.

---

## 1. Getting the Data

Install the dependencies:

```bash
uv add insurance-datasets shap-relativities insurance-fairness \
    insurance-conformal insurance-monitoring
uv add catboost polars scikit-learn
```

`insurance-datasets` generates synthetic UK motor policies from a known data generating process. The DGP is a Poisson frequency model and a Gamma severity model with log-linear predictors. True parameters are exported as `TRUE_FREQ_PARAMS` and `TRUE_SEV_PARAMS`, so you can validate whether your fitted GLM recovers them.

```python
from insurance_datasets import load_motor

df = load_motor(n_policies=50_000, seed=42)
print(df.shape)         # (50000, 18)
print(df.dtypes)
```

The dataset has 18 columns: `policy_id`, `inception_date`, `expiry_date`, `inception_year`, `vehicle_age`, `vehicle_group` (ABI group 1–50), `driver_age`, `driver_experience`, `ncd_years` (0–5 on the UK scale), `ncd_protected`, `conviction_points`, `annual_mileage`, `area` (ABI area band A–F), `occupation_class`, `policy_type`, plus the outcomes `claim_count`, `incurred`, and `exposure` in earned policy years.

```python
# Basic portfolio statistics
freq = df["claim_count"].sum() / df["exposure"].sum()
print(f"Portfolio claim frequency: {freq:.3f}")   # ~0.075
print(f"Cancellation rate: {(df['exposure'] < 0.99).mean():.1%}")  # ~8%
```

**What synthetic data does not give you.** Real portfolios have missing values — the DGP generates complete data. Real portfolios have data quality issues: duplicate policies, inconsistent area codes, exposure miscalculations from system migrations. Real books also have tail correlation in claims that a Poisson/Gamma independence assumption misses. Use this dataset to test algorithms and pipeline structure, not to benchmark against real portfolio performance.

The area band distribution (12% A, 18% B, 25% C, 22% D, 14% E, 9% F) and age distribution (12% under-25, 30% 25–40, 35% 40–60, 18% 60–75, 5% over-75) are calibrated to a realistic UK mid-market motor book. NCD penetration is correlated with driver experience, as you'd expect — not random.

---

## 2. Feature Engineering

We use Polars throughout. It is materially faster than pandas for the filtering and groupby operations that dominate pricing pipelines, and the lazy evaluation API makes query optimisation straightforward.

```python
import polars as pl

# Load as a Polars DataFrame directly
df = load_motor(n_policies=50_000, seed=42, polars=True)

# Banded age for use as a categorical rating factor
df = df.with_columns([
    pl.when(pl.col("driver_age") < 25).then(pl.lit("Under 25"))
      .when(pl.col("driver_age") < 40).then(pl.lit("25-39"))
      .when(pl.col("driver_age") < 60).then(pl.lit("40-59"))
      .when(pl.col("driver_age") < 75).then(pl.lit("60-74"))
      .otherwise(pl.lit("75+"))
      .alias("age_band"),

    # Binary conviction flag — conviction_points > 0
    (pl.col("conviction_points") > 0).cast(pl.Int32).alias("has_convictions"),

    # Claim frequency rate (claims per earned year) — target for frequency model
    (pl.col("claim_count") / pl.col("exposure").clip(lower_bound=1e-6))
      .alias("claim_frequency"),
])

# Severity: average cost per claim, for policies with at least one claim
claims_df = df.filter(pl.col("claim_count") > 0).with_columns([
    (pl.col("incurred") / pl.col("claim_count")).alias("avg_severity"),
])

print(df.select(["claim_frequency", "exposure"]).describe())
```

The frequency/severity split matters. A single Tweedie model is tempting because it avoids the two-model awkwardness, but it conflates two distinct processes. A bad driver in area F has high frequency. An expensive car in area A has high severity. The factors that drive each are different, and fitting them jointly loses that signal. For motor, we model frequency as Poisson and severity as Gamma, then multiply at the end.

Exposure is already calculated in `insurance-datasets` as earned years from inception to expiry (or cancellation). For a real portfolio you will calculate this yourself, typically from the system's policy transaction table. Make sure you are using *earned* exposure, not written exposure — policies written in December that run through to November next year are not fully earned at year end.

```python
# Define the feature sets for each model
FREQ_FEATURES = [
    "vehicle_group", "age_band", "ncd_years",
    "has_convictions", "area", "annual_mileage",
]

SEV_FEATURES = [
    "vehicle_group", "age_band", "vehicle_age",
]

# Train/test split on inception year — do NOT do a random split
# for time-series pricing data. Models trained on 2019-2022, tested on 2023.
train = df.filter(pl.col("inception_year") <= 2022)
test = df.filter(pl.col("inception_year") == 2023)

print(f"Train: {len(train):,} | Test: {len(test):,}")
```

A random 80/20 split is wrong here. Insurance data has calendar year effects — claims inflation, coding changes, market shifts. Your test set should be a later time period than your training set. We use inception year 2023 as the holdout.

---

## 3. Modelling

We fit CatBoost frequency and severity models. CatBoost handles categorical features natively, which removes the encoding decisions that pollute GLM workflows. It also uses the exact same log-link objectives (Poisson, Gamma) as a GLM, which means the SHAP values are in log-space and the multiplicative interpretation holds.

```python
from catboost import CatBoostRegressor
import numpy as np

# --- Frequency model ---
X_train_freq = train.select(FREQ_FEATURES).to_pandas()
y_train_freq = train["claim_count"].to_numpy()
w_train_freq = train["exposure"].to_numpy()

cat_features = ["age_band", "area"]

freq_model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    cat_features=cat_features,
    random_seed=42,
    verbose=0,
)
freq_model.fit(
    X_train_freq, y_train_freq,
    sample_weight=w_train_freq,
)

# --- Severity model (claims only) ---
claims_train = train.filter(pl.col("claim_count") > 0)
X_train_sev = claims_train.select(SEV_FEATURES).to_pandas()
y_train_sev = (
    claims_train["incurred"] / claims_train["claim_count"]
).to_numpy()

sev_model = CatBoostRegressor(
    loss_function="Gamma",
    iterations=300,
    learning_rate=0.05,
    depth=4,
    cat_features=["age_band"],
    random_seed=42,
    verbose=0,
)
sev_model.fit(X_train_sev, y_train_sev)

# --- Pure premium prediction on test set ---
X_test_freq = test.select(FREQ_FEATURES).to_pandas()
X_test_sev = test.select(SEV_FEATURES).to_pandas()

pred_freq = freq_model.predict(X_test_freq)
pred_sev = sev_model.predict(X_test_sev)
pred_pure_premium = pred_freq * pred_sev

# Quick A/E check
actual_lr = test["incurred"].sum() / (
    (pred_pure_premium * test["exposure"].to_numpy()).sum()
)
print(f"A/E ratio on 2023 holdout: {actual_lr:.3f}")  # should be near 1.0
```

We use `loss_function="Poisson"` for frequency, which gives the log-link Poisson GLM objective. The sample weights are earned exposure years — this ensures that a policy with 0.3 car-years gets one-third the influence of a full-year policy.

For severity, `"Gamma"` fits a Gamma with log-link. We model average claim cost per claim (incurred divided by claim count) on the claims-only subset. This is standard actuarial practice: including zero-claim policies in the severity model would bias the estimate downward.

One honest note: a Gradient Boosting Machine will generally outperform a GLM in Gini terms. Whether that is worth the explainability cost is a business decision. Our view is that for UK personal lines motor, the GBM wins on price accuracy and the SHAP relativity extraction (next section) recovers most of the explainability.

---

## 4. Extracting Rating Factors with `shap-relativities`

This is where most GBM pricing tutorials end with a gesture at SHAP waterfall charts and move on. That is not sufficient for production. Your underwriting system — whether that is Radar, Emblem, or an in-house factor engine — expects a table of multiplicative relativities: a base rate, then a factor per rating variable per level. A GLM produces these directly via `exp(beta)`. A GBM does not, until now.

`shap-relativities` converts TreeSHAP values from a log-link GBM into exactly that table. Each factor level gets an exposure-weighted mean SHAP value, converted to a multiplier with CLT confidence intervals.

```python
from shap_relativities import SHAPRelativities

X_train_pl = train.select(FREQ_FEATURES)

sr = SHAPRelativities(
    model=freq_model,
    X=X_train_pl,
    exposure=train["exposure"],
    categorical_features=["age_band", "area"],
)
sr.fit()

 rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area": "A", "age_band": "40-59"},
)

print(rels.filter(pl.col("feature") == "area"))
```

Output looks like:

```
shape: (6, 9)
┌─────────┬───────┬────────────┬──────────┬──────────┬───────────┬──────────┬───────┬────────────────┐
│ feature ┆ level ┆ relativity ┆ lower_ci ┆ upper_ci ┆ mean_shap ┆ shap_std ┆ n_obs ┆ exposure_weight│
│ ---     ┆ ---   ┆ ---        ┆ ---      ┆ ---      ┆ ---       ┆ ---      ┆ ---   ┆ ---            │
│ str     ┆ str   ┆ f64        ┆ f64      ┆ f64      ┆ f64       ┆ f64      ┆ i64   ┆ f64            │
╞═════════╪═══════╪════════════╪══════════╪══════════╪═══════════╪══════════╪═══════╪════════════════╡
│ area    ┆ A     ┆ 1.000      ┆ 1.000    ┆ 1.000    ┆ 0.000     ┆ …        ┆ …     ┆ …              │
│ area    ┆ B     ┆ 1.104      ┆ 1.088    ┆ 1.121    ┆ 0.099     ┆ …        ┆ …     ┆ …              │
│ area    ┆ C     ┆ 1.218      ┆ 1.203    ┆ 1.233    ┆ 0.197     ┆ …        ┆ …     ┆ …              │
│ area    ┆ D     ┆ 1.401      ┆ 1.384    ┆ 1.419    ┆ 0.337     ┆ …        ┆ …     ┆ …              │
│ area    ┆ E     ┆ 1.633      ┆ 1.610    ┆ 1.657    ┆ 0.490     ┆ …        ┆ …     ┆ …              │
│ area    ┆ F     ┆ 1.908      ┆ 1.876    ┆ 1.941    ┆ 0.646     ┆ …        ┆ …     ┆ …              │
```

Compare that to the true DGP parameters (`area_B=0.10`, `area_C=0.20`, ..., `area_F=0.65` in log-space). The GBM recovers these with reasonable fidelity on 50,000 policies. On a real portfolio the recovery will be noisier due to correlated features and class imbalance.

The `base_levels` argument sets which level in each categorical feature gets relativity 1.0 — your base rate vehicle. For area, we use Band A (rural, lowest risk). For age, the 40–59 bracket (lowest frequency band). These choices must match your actuarial methodology documentation; they are not cosmetic.

For a Radar import, you would export `rels.write_csv("area_factors.csv")` and follow the standard Radar factor table format. For Emblem, the structure is identical. The key property that makes this work: because the model uses a log-link objective, the SHAP values are additive in log-space, so `exp(shap_value)` gives a genuine multiplicative factor.

One limitation to document: when features are correlated, SHAP splits attribution between them in a way that depends on tree split order. The `vehicle_group` and `area` relativities will absorb some of each other's effect. This is a property of all tree-based attribution methods, not a bug in the library.

---

## 5. Fairness Audit with `insurance-fairness`

The FCA's Evaluation Paper EP25/2 (February 2025) makes the supervisory expectation explicit: firms must demonstrate that their pricing models do not produce discriminatory outcomes by reference to protected characteristics. Consumer Duty Outcome 4 (Price and Value) extends this to require evidence that different customer groups receive equivalent value, not just equivalent prices.

`insurance-fairness` runs a full proxy discrimination audit. We check whether `occupation_class` — a legitimate rating factor — is acting as a proxy for a protected characteristic. In a UK motor context, the live question is usually postcode or occupation discriminating on ethnicity or disability; we use occupation class here as a worked example.

First, we need a prediction column on our audit dataset:

```python
import polars as pl
from insurance_fairness import FairnessAudit

# Add predictions to the test DataFrame
X_test_freq_all = test.select(FREQ_FEATURES).to_pandas()
X_test_sev_all = test.select(SEV_FEATURES).to_pandas()

test = test.with_columns([
    pl.Series("predicted_pure_premium",
              freq_model.predict(X_test_freq_all) * sev_model.predict(X_test_sev_all))
])

# For this example we treat occupation_class >= 4 as a hypothetical
# protected group (e.g. a segment with higher disability prevalence)
test = test.with_columns([
    (pl.col("occupation_class") >= 4).cast(pl.Int32).alias("protected_group")
])

audit = FairnessAudit(
    model=freq_model,
    data=test,
    protected_cols=["protected_group"],
    prediction_col="predicted_pure_premium",
    outcome_col="incurred",
    exposure_col="exposure",
    factor_cols=["vehicle_group", "age_band", "ncd_years",
                 "has_convictions", "area", "annual_mileage"],
    model_name="Motor Frequency v1 — 2023 holdout",
    run_proxy_detection=True,
)

report = audit.run()
report.summary()
```

The report prints a RAG (Red/Amber/Green) status across three checks:

- **Demographic parity**: is the exposure-weighted mean predicted premium materially different between groups? Threshold: log-ratio > 0.1 is Amber, > 0.2 is Red.
- **Disparate impact ratio**: ratio of group mean premiums. Below 0.8 triggers Amber (the 80% rule from US employment law, adopted as a reference point in FCA guidance).
- **Calibration by group**: are A/E ratios consistent across groups? A model that is well-calibrated overall but systematically over-predicts for one group fails this check.

The proxy detection component runs mutual information scores and proxy R-squared tests between each rating factor and the protected characteristic. Factors that are highly predictive of the protected group get flagged. `flagged_factors` in the report lists them.

```python
# Export for the governance file
report.to_markdown("fairness_audit_motor_v1_2023.md")
```

The `to_markdown()` output is structured for a pricing committee pack: model metadata, dataset size and exposure, per-characteristic metric tables, and flagged factors with scores. Run this quarterly as part of your model review cycle, not as a one-off box-tick.

A practical note: the DGP in `insurance-datasets` does not include a protected characteristic in the true model — there is no genuine proxy discrimination baked in. On this synthetic data the audit will likely return Green. On real data you will find things. The occupation/postcode question in UK motor is live and contested; the FCA has not yet prescribed remediation methods, but EP25/2 signals that `opt-in` exclusions of protected attributes are not sufficient.

---

## 6. Prediction Intervals with `insurance-conformal`

A point estimate is an incomplete answer. If you quote a pure premium of £342, the relevant question for reserving and pricing is: what is the 90% prediction interval around that? Standard Poisson GLM theory gives you this via the mean-variance relationship, but GBMs do not have a parametric distribution attached. Conformal prediction provides distribution-free prediction intervals with a finite-sample coverage guarantee.

The coverage guarantee is exactly: if your calibration data is exchangeable with your test data, then 90% of test observations will fall within the predicted interval — no distributional assumptions required. The interval width adapts to local uncertainty, not just the global noise level.

```python
from insurance_conformal import InsuranceConformalPredictor

# Split calibration from training — use 2022 as calibration year
cal = train.filter(pl.col("inception_year") == 2022)
train_proper = train.filter(pl.col("inception_year") <= 2021)

# Refit on train_proper only (abbreviated here — in practice full refit)
X_cal_freq = cal.select(FREQ_FEATURES).to_pandas()
y_cal = cal["incurred"].to_numpy()
exposure_cal = cal["exposure"].to_numpy()

cp = InsuranceConformalPredictor(
    model=freq_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
)
cp.calibrate(X_cal_freq, y_cal, exposure=exposure_cal)

# 90% prediction intervals on 2023 test set
X_test_cp = test.select(FREQ_FEATURES).to_pandas()
intervals = cp.predict_interval(X_test_cp, alpha=0.10)

# Coverage check — should be close to 90%
y_test = test["incurred"].to_numpy()
lower = intervals["lower"].to_numpy()
upper = intervals["upper"].to_numpy()
covered = ((y_test >= lower) & (y_test <= upper)).mean()
print(f"Empirical coverage: {covered:.1%}")   # ~90%

# Median interval width
width = float((intervals["upper"] - intervals["lower"]).median())
print(f"Median interval width: £{width:.0f}")
```

We use `nonconformity="pearson_weighted"` — the Pearson residual `(y - ŷ) / sqrt(ŷ^p)` normalised by the predicted value. For Poisson/Tweedie data this gives roughly 30% narrower intervals than raw residuals, with the same coverage guarantee. The non-conformity score accounts for the mean-variance relationship: a policy expected to cost £500 has wider natural variation than one expected to cost £150.

The `calibrate()` call stores the empirical quantile of the non-conformity scores. `predict_interval()` then inverts this quantile to produce intervals. No model refitting required — conformal calibration is a post-hoc step.

For reserving applications, `insurance_conformal.risk.PremiumSufficiencyController` provides risk-controlled sufficiency bounds: instead of asking "does the observation fall in the interval", it asks "what loading factor do we need to ensure E[loss / premium] <= alpha". That is the right framing for capital adequacy work, though it is beyond the scope of this tutorial.

---

## 7. Deployment: Champion/Challenger and Drift Monitoring

Getting a model into production is a handover, not an endpoint.

**Champion/challenger with `insurance-deploy`.**

```python
from insurance_deploy import ModelRegistry, Experiment, QuoteLogger

registry = ModelRegistry("./model_registry")
mv = registry.register(
    freq_model,
    name="motor_freq",
    version="2.0",
    metadata={"training_date": "2024-01-01", "training_rows": 40_000},
)

# Shadow mode: challenger runs in parallel, does not affect quoted price
exp = Experiment("freq_v2_vs_v1", champion=old_mv, challenger=mv)
logger = QuoteLogger("./quotes.db")
```

`QuoteLogger` maintains an auditable record of every quote: which model version was used, the inputs, and the output price. This is the technical component of ICOBS 6B.2.51R record-keeping for ENBP compliance. Shadow mode is the default — and the right default. Live A/B pricing, where different customers are charged different prices for the same risk, raises Consumer Duty fair value concerns. Take legal advice before enabling live routing.

**Drift detection with `insurance-monitoring`.**

Once the model is live, you need to know when it stops working. Three layers:

```python
from insurance_monitoring import psi, ae_ratio_ci, MonitoringReport

# Population Stability Index on the vehicle_group distribution
# Reference = training period; current = last quarter of live data
psi_score = psi(
    reference=train["vehicle_group"].to_numpy(),
    current=live_df["vehicle_group"].to_numpy(),
    n_bins=10,
)
print(f"PSI vehicle_group: {psi_score:.4f}")
# < 0.10 = stable, 0.10-0.25 = investigate, > 0.25 = significant shift

# A/E ratio with confidence intervals
ae = ae_ratio_ci(
    actual=live_df["incurred"].to_numpy(),
    predicted=live_df["predicted_pure_premium"].to_numpy(),
    exposure=live_df["exposure"].to_numpy(),
)
print(f"A/E: {ae['ae']:.3f}  95% CI: [{ae['lower']:.3f}, {ae['upper']:.3f}]")

# Full monitoring report: PSI, Gini drift, A/E
# MonitoringReport runs on instantiation — results available immediately.
# Reference predictions: get in-sample predictions on training data first.
pred_train_freq = freq_model.predict(train.select(FREQ_FEATURES).to_pandas())
pred_train_sev  = sev_model.predict(train.select(SEV_FEATURES).to_pandas())
pred_train_pp   = pred_train_freq * pred_train_sev

report = MonitoringReport(
    reference_actual=train["incurred"].to_numpy(),
    reference_predicted=pred_train_pp,
    current_actual=live_df["incurred"].to_numpy(),
    current_predicted=live_df["predicted_pure_premium"].to_numpy(),
    exposure=live_df["exposure"].to_numpy(),
    murphy_distribution="tweedie",
)
print(report.recommendation)  # "NO_ACTION", "RECALIBRATE", or "REFIT"
print(report.to_dict())
```

PSI for feature distribution shift. The Gini drift test (`GiniDriftBootstrapTest`, included in `MonitoringReport`) for model discrimination power. A/E ratio with confidence intervals for calibration. These three together cover the IFoA model risk management framework's monitoring requirements and satisfy the typical PRA/FCA model governance expectations.

`DriftAttributor` goes a step further: it tells you *which* features explain a Gini decline, using the TRIPODD method (Panda et al. 2025). When your A/E deteriorates, you need to know whether it is the vehicle_group distribution shifting (action: retrain), the area factor drifting (action: recalibrate), or a claims inflation trend (action: manual load). `DriftAttributor` diagnoses this automatically.

---

## 8. Conclusion

We have built a complete pricing pipeline: synthetic UK motor data generated with `insurance-datasets`, Polars feature engineering with a proper time-based train/test split, CatBoost frequency and severity models with Poisson and Gamma objectives, GLM-style factor tables extracted via `shap-relativities`, a fairness audit aligned to FCA EP25/2 using `insurance-fairness`, distribution-free prediction intervals with guaranteed coverage using `insurance-conformal`, and production infrastructure via `insurance-deploy` and `insurance-monitoring`.

None of this requires a bespoke actuarial platform. The full pipeline runs in standard Python.

Individual library documentation:

- [`insurance-datasets`](https://burning-cost.github.io/insurance-datasets) — synthetic data generation
- [`shap-relativities`](https://burning-cost.github.io/shap-relativities) — GBM to GLM factor extraction
- [`insurance-fairness`](https://burning-cost.github.io/insurance-fairness) — proxy discrimination audit
- [`insurance-conformal`](https://burning-cost.github.io/insurance-conformal) — prediction intervals
- [`insurance-monitoring`](https://burning-cost.github.io/insurance-monitoring) — drift detection and calibration monitoring
- [`insurance-deploy`](https://burning-cost.github.io/insurance-deploy) — champion/challenger and ENBP logging
