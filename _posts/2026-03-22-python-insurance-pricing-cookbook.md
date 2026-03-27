---
layout: post
title: "Python Insurance Pricing Cookbook: 20 Recipes for Common Tasks"
date: 2026-03-22
categories: [blog]
author: Burning Cost
tags: [python, pricing, catboost, polars, shap-relativities, insurance-distill, insurance-fairness, insurance-monitoring, insurance-causal, insurance-conformal, insurance-credibility, insurance-whittaker, insurance-trend, insurance-survival, insurance-datasets, insurance-cv, insurance-governance, reference]
description: "20 short code recipes for common insurance pricing tasks. Each recipe uses a real API from one of our open-source libraries. Copy and adapt."
---

A reference post. Each recipe is a standalone snippet - minimal setup, real API calls, expected output in comments. Install whichever libraries you need with `uv add`.

No GLM appendices. No theoretical preambles. If you want the background, the linked posts have it.

---

## Data Preparation

### Recipe 1: How do I load a synthetic UK motor dataset with a known data generating process?

Use `insurance-datasets` when you want a portfolio with verifiable ground truth - useful for checking whether your model recovers the correct relativities.

```python
from insurance_datasets import load_motor, MOTOR_TRUE_FREQ_PARAMS

df = load_motor(n_policies=50_000, seed=42)
# Returns a Polars DataFrame with columns:
# inception_date, expiry_date, exposure, area, ncd_years,
# driver_age, vehicle_group, conviction_points, claim_count, claim_amount

print(df.shape)         # (50000, 12)
print(MOTOR_TRUE_FREQ_PARAMS["ncd_years"])  # -0.12 (per year, log-scale)
# NCD=5 vs NCD=0 true relativity: exp(-0.12 * 5) = 0.549
```

The DGP runs a Poisson frequency model with known coefficients. After fitting your model, compare extracted relativities against `MOTOR_TRUE_FREQ_PARAMS` to verify the pipeline is working.

---

### Recipe 2: How do I set up temporal cross-validation with an IBNR buffer?

Standard k-fold is wrong for insurance data. Use `insurance-cv` to generate walk-forward splits that respect the temporal structure and exclude partially-developed claim periods.

```python
import polars as pl
from insurance_cv.splits import walk_forward_split, InsuranceCV

df = load_motor(n_policies=100_000, seed=42)

splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=24,   # require at least 2 years of training data
    test_months=6,         # 6-month test windows
    step_months=6,         # non-overlapping test windows
    ibnr_buffer_months=3,  # 3-month buffer — safe for motor
)

print(len(splits))        # 2 folds for a 5-year dataset with these settings
print(splits[0].label)
# fold-01 train:2019-01 to 2021-09 | test:2021-12 to 2022-05

# Use directly with sklearn
cv = InsuranceCV(splits, df)
```

Rows with inception dates in the 3-month IBNR buffer are excluded from both train and test. For liability or professional indemnity, increase `ibnr_buffer_months` to 12 or more - motor and home are typically safe at 3-6 months.

---

### Recipe 3: How do I generate policy-year aligned splits for rate-change analysis?

When your book has annual rate changes, mixing pre- and post-change data in the same fold distorts the validation. Use `policy_year_split` to enforce clean calendar year boundaries.

```python
from insurance_cv.splits import policy_year_split

splits = policy_year_split(
    df,
    date_col="inception_date",
    n_years_train=3,
    n_years_test=1,
    step_years=1,
)

for s in splits:
    print(s.label)
# fold-01 PY2019-2021 -> PY2022
# fold-02 PY2020-2022 -> PY2023
```

---

## Model Building

### Recipe 4: How do I extract GLM-style rating factors from a CatBoost model?

`shap-relativities` converts CatBoost SHAP values into multiplicative factor tables - the `exp(beta)` format that pricing committees and Radar import templates expect.

```python
from shap_relativities import SHAPRelativities
import catboost, polars as pl

# Assume: model = fitted CatBoostRegressor(loss_function="Poisson")
# Assume: X = Polars DataFrame of features, exposure = Polars Series

sr = SHAPRelativities(
    model=model,
    X=X,
    exposure=exposure,
    categorical_features=["area_code", "ncd_years", "has_convictions"],
    continuous_features=["driver_age"],
)
sr.fit()

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
    ci_level=0.95,
)
print(rels.filter(pl.col("feature") == "ncd_years"))
# feature     level  relativity  lower_ci  upper_ci   n_obs
# ncd_years       0       1.000     1.000     1.000    8741
# ncd_years       1       0.882     0.851     0.913    7923
# ncd_years       5       0.549     0.521     0.578    8617

# Always run the reconstruction check before presenting results
checks = sr.validate()
print(checks["reconstruction"])
# CheckResult(passed=True, value=8.3e-06, ...)
```

The reconstruction check verifies that `exp(shap_values.sum(axis=1) + expected_value)` matches `model.predict()` to within `1e-4` on every row. If it fails, stop and investigate the objective/output-type mismatch before showing numbers to anyone.

---

### Recipe 5: How do I distil a CatBoost model into a Radar-ready factor table?

[`insurance-distill`](/insurance-distill/) fits a surrogate GLM on GBM pseudo-predictions and exports one CSV per feature - ready to load into Radar or any multiplicative rating engine.

```python
from insurance_distill import SurrogateGLM

surrogate = SurrogateGLM(
    model=fitted_catboost,
    X_train=X_train,          # Polars DataFrame
    y_train=claim_counts,
    exposure=exposure_arr,
    family="poisson",
)
surrogate.fit(
    features=["driver_age", "vehicle_group"],
    categorical_features=["area_code", "ncd_years"],
    max_bins=10,
)

report = surrogate.report()
print(report.metrics.summary())
# GBM Gini: 0.412 | GLM Gini: 0.389 | Gini ratio: 0.944
# Deviance ratio: 0.961 | Max segment deviation: 3.2%

paths = surrogate.export_csv("output/factors/", prefix="motor_freq_")
# ['output/factors/motor_freq_driver_age.csv', ...]
```

The Gini ratio tells you how much discrimination the GLM retains relative to the GBM. Anything above 0.90 is generally acceptable for a main-effects surrogate. Below 0.85, consider adding interaction pairs via `interaction_pairs=[("driver_age", "vehicle_group")]`.

---

### Recipe 6: How do I model the full conditional distribution of claims, not just the mean?

`insurance-distributional` fits a Tweedie GBM that returns both the mean and variance of the conditional loss distribution - useful for safety loading and reinsurance pricing.

```python
from insurance_distributional import TweedieGBM

model = TweedieGBM(power=1.5)   # power=1 is Poisson, power=2 is Gamma
model.fit(X_train, y_train, exposure=exposure_train)
pred = model.predict(X_test, exposure=exposure_test)

print(pred.mean[:5])         # E[Y|X] — pure premium per risk
# [142.3, 89.1, 312.7, 201.4, 67.8]
print(pred.variance[:5])     # Var[Y|X] — conditional variance
# [2341.2, 1203.5, 8932.1, 4512.3, 891.2]

cov = pred.volatility_score()  # CoV = sqrt(Var) / mean per risk
# High CoV risks need a proportionally larger safety loading

# Proper scoring for model comparison
crps = model.crps(X_test, y_test)
print(f"CRPS: {crps:.4f}")   # Lower is better
```

Two risks with identical means but different variances require different safety loadings under Solvency II. The standard Tweedie GLM gives you the mean only. `TweedieGBM` gives you both.

---

### Recipe 7: How do I price an excess of loss reinsurance layer per risk?

`FlexCodeDensity` fits a nonparametric conditional density estimator via CatBoost and cosine basis functions. From that density you can compute the expected loss in any XL layer without assuming a parametric severity distribution.

```python
from insurance_distributional import FlexCodeDensity

fc = FlexCodeDensity(max_basis=30)
fc.fit(X_severity, y_severity)    # y_severity = claim amounts > 0

# Expected loss in £500k xs £1m layer
ev = fc.price_layer(X_test, attachment=500_000.0, limit=1_000_000.0)
print(ev[:3])
# [2341.2, 891.5, 5102.8]  — expected layer loss per risk
```

`max_basis=30` is a good starting point for 10,000–100,000 severity observations. Go higher if CRPS on a holdout set is still improving; go lower for thin portfolios.

---

## Fairness and Governance

### Recipe 8: How do I check my pricing model for proxy discrimination?

[`insurance-fairness`](/insurance-fairness/) provides FCA Consumer Duty-aligned bias metrics. Start with `detect_proxies` to find which rating factors are correlated with protected characteristics, then run calibration-by-group to check whether any group is systematically over- or under-priced.

```python
from insurance_fairness import detect_proxies, calibration_by_group
import polars as pl

# Step 1: proxy detection — which rating factors correlate with gender?
result = detect_proxies(
    df=df,
    protected_col="gender",
    factor_cols=["area_code", "vehicle_group", "occupation", "ncd_years"],
)
result.summary()
# feature         mutual_info   proxy_r2   rag
# occupation          0.142      0.183    AMBER
# area_code           0.031      0.029    GREEN
# vehicle_group       0.018      0.021    GREEN

# Step 2: calibration by group — is the model equally calibrated for each group?
cal = calibration_by_group(
    df=df_with_predictions,
    protected_col="gender",
    prediction_col="predicted_freq",
    outcome_col="claim_count",
    exposure_col="exposure",
    n_deciles=10,
)
print(f"Max A/E disparity: {cal.max_disparity:.3f}")  # 0.034
print(f"RAG: {cal.rag}")                               # GREEN
```

A model that is equally calibrated across groups (A/E ≈ 1.0 in all deciles for all groups) satisfies the sufficiency criterion - the most defensible standard under Equality Act 2010 Section 19.

---

### Recipe 9: How do I compute a demographic parity ratio for my pricing model?

```python
from insurance_fairness import demographic_parity_ratio

result = demographic_parity_ratio(
    df=df_with_predictions,
    protected_col="gender",
    prediction_col="predicted_premium",
    exposure_col="exposure",
    log_space=True,      # correct for multiplicative models
    n_bootstrap=500,     # bootstrap CIs
    ci_level=0.95,
)
print(f"Log-ratio: {result.log_ratio:.3f}")   # log(mean_M / mean_F)
print(f"Ratio: {result.ratio:.3f}")           # exp(log_ratio)
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"RAG: {result.rag}")                   # GREEN / AMBER / RED
# Log-ratio: 0.038  Ratio: 1.039  95% CI: [0.031, 0.046]  RAG: GREEN
```

A ratio of 1.039 means male policyholders pay 3.9% more on average. This alone tells you nothing - it may reflect genuine risk differences. Use `calibration_by_group` alongside this to determine whether the difference is actuarially justified.

---

### Recipe 10: How do I generate a model card for a PRA model validation committee?

`insurance-governance` provides a structured `MRMModelCard` that captures the fields a Model Risk Committee expects - intended use, assumptions, limitations, approval conditions - and serialises to JSON for audit trail.

```python
from insurance_governance import MRMModelCard, Assumption, Limitation

card = MRMModelCard(
    model_id="motor-freq-tppd-v3",
    model_name="Motor TPPD Frequency Model",
    version="3.0.1",
    model_class="pricing",
    intended_use="Set pure premium components for UK private motor TPPD. "
                 "Permitted use: underwriting, rate filing, Solvency II capital.",
    not_intended_for=["Commercial vehicle pricing", "Claims reserving"],
    target_variable="claim_count",
    distribution_family="Poisson",
    model_type="CatBoost GBM with SHAP-derived factor tables",
    rating_factors=["area_code", "ncd_years", "driver_age", "vehicle_group"],
    gwp_impacted=85_000_000,     # £85m GWP
    customer_facing=True,
    assumptions=[
        Assumption(
            description="Loss emergence pattern stable over the training window (2021-2025).",
            risk="MEDIUM",
            mitigation="Quarterly A/E monitoring by inception cohort.",
        ),
    ],
    limitations=[
        Limitation(
            description="Model trained on direct channel only. Aggregator channel behaviour not represented.",
            impact="May underprice aggregator-sourced business by up to 8%.",
            population_at_risk="Aggregator channel policyholders",
            monitoring_flag=True,
        ),
    ],
)

# Serialise to JSON for governance system upload
import json
print(json.dumps(card.to_dict(), indent=2, default=str)[:300])
```

---

## Monitoring

### Recipe 11: How do I run an A/E ratio check with Poisson confidence intervals?

```python
from insurance_monitoring import ae_ratio
import numpy as np

ae = ae_ratio(
    actual=claim_counts,           # np.ndarray or pl.Series
    predicted=predicted_freq,
    exposure=earned_exposure,
    segments=area_codes,           # optional: per-segment breakdown
)
# Returns float if no segments:
print(f"Overall A/E: {ae:.3f}")   # 0.974

# With segments, returns a Polars DataFrame:
# segment  actual  predicted  ae_ratio  n_policies
# A          234      241.2    0.970        4821
# B          312      301.8    1.034        5103
# ...
```

A/E outside [0.90, 1.10] on a sufficiently large segment is a meaningful signal. On thin segments - fewer than ~200 claims - the sampling variance dominates and point estimates are unreliable.

---

### Recipe 12: How do I detect which features are driving model performance drift?

`DriftAttributor` implements TRIPODD (Panda et al. 2025, arXiv:2503.06606): it identifies which features - including interactions - explain why model performance has degraded between train and monitor periods.

```python
from insurance_monitoring import DriftAttributor

attributor = DriftAttributor(
    model=fitted_catboost,
    features=["driver_age", "vehicle_group", "area_code", "ncd_years"],
    loss="poisson_deviance",
    alpha=0.05,              # Bonferroni-corrected type I error
    n_bootstrap=200,
)
attributor.fit_reference(X_train, y_train, exposure_train)

result = attributor.test(X_monitor, y_monitor, exposure_monitor)
print(result.significant_features)
# ['driver_age', ('driver_age', 'vehicle_group')]  — these two explain the drift

print(result.drift_contributions)
# {'driver_age': 0.034, ('driver_age', 'vehicle_group'): 0.021, ...}
```

Single-feature drift (driver_age has shifted) is addressable by recalibration. An interaction effect drifting without the main effects drifting usually means genuine model degradation - refit rather than recalibrate.

---

### Recipe 13: How do I run an anytime-valid champion/challenger frequency test?

`SequentialTest` uses mixture SPRT (Johari et al. 2022). Unlike a standard t-test or chi-squared test, you can check the result weekly or monthly without inflating type I error. No pre-specified sample size required.

```python
from insurance_monitoring import SequentialTest
import datetime

test = SequentialTest(
    metric="frequency",
    alternative="two_sided",
    alpha=0.05,
    tau=0.03,                    # prior std dev on log-rate-ratio; 3% expected effects
    min_exposure_per_arm=500.0,  # car-years before any stopping decision
)

# Feed monthly updates
for month_data in monthly_batches:
    result = test.update(
        champion_claims=month_data["champ_claims"],
        champion_exposure=month_data["champ_exposure"],
        challenger_claims=month_data["chal_claims"],
        challenger_exposure=month_data["chal_exposure"],
        calendar_date=datetime.date(2026, month_data["month"], 1),
    )
    print(result.summary)
    # Month 4: "Challenger freq 6.2% lower (95% CS: 0.891–0.981). Evidence: 24.3 (threshold 20.0). Reject_H0."
    if result.should_stop:
        print(f"Decision: {result.decision}")
        break
```

The `lambda_value >= threshold` (i.e., `>= 1/alpha`) is the stopping criterion. Once you exceed it, the test has found evidence at the claimed type I error rate, regardless of how many interim checks you ran.

---

## Causal Inference

### Recipe 14: How do I estimate the causal price elasticity for renewal pricing?

`RenewalElasticityEstimator` uses Double Machine Learning (Chernozhukov et al. 2018) to remove confounding from observed renewal data. Naive logistic regression on price gives biased estimates because the assignment of price changes was not random.

```python
from insurance_causal.elasticity import RenewalElasticityEstimator

est = RenewalElasticityEstimator(
    cate_model="causal_forest",     # CausalForestDML with CatBoost nuisance models
    binary_outcome=True,            # renewal = 0/1
    n_folds=5,
    n_estimators=200,
)
est.fit(
    df=renewal_df,
    confounders=["driver_age", "vehicle_group", "ncd_years", "region"],
)

ate, lb, ub = est.ate()
print(f"ATE: {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")
# ATE: -0.312  95% CI: [-0.378, -0.246]
# A 100% price increase reduces renewal probability by 31.2pp.
# For a 10% increase: approximately -3.1pp.

# Per-policyholder CATE (heterogeneous treatment effects)
cates = est.cate(df=renewal_df)
```

The ATE is the semi-elasticity in log-log specification: treatment is `log(offer / last_year)`. For a 10% price increase, divide by 10.

---

## Uncertainty Quantification

### Recipe 15: How do I generate prediction intervals for a Tweedie claims model?

`InsuranceConformalPredictor` wraps any fitted model and produces distribution-free prediction intervals with exact finite-sample coverage guarantees. The Pearson-weighted nonconformity score is correct for Tweedie/compound-Poisson data and gives ~30% narrower intervals than raw residuals.

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=fitted_catboost_tweedie,
    nonconformity="pearson_weighted",   # (y - yhat) / sqrt(yhat^p)
    distribution="tweedie",
)
cp.calibrate(X_cal, y_cal, exposure=exposure_cal)

intervals = cp.predict_interval(X_test, alpha=0.10)
# Returns Polars DataFrame with columns: lower, upper, predicted
print(intervals.head(3))
# lower   upper   predicted
#  89.2   421.3      187.4
#  12.1   198.7       67.3
# 234.1  1102.4      502.8
```

`alpha=0.10` gives 90% coverage. The guarantee is marginal (on average across the test set), not conditional on each subgroup. For conditional coverage diagnostics, use `CoverageDiagnostics` or `subgroup_coverage`.

---

### Recipe 16: How do I monitor Poisson model calibration in production without repeated-testing inflation?

`PITMonitor` uses probability integral transforms and mixture e-processes (Henzi, Murph, Ziegel 2025, arXiv:2603.13156). You compute the PIT - the model's predictive CDF evaluated at the observed claim count - and feed them sequentially. The monitor alarms if and when the PIT distribution deviates from Uniform(0,1), which is what a correctly calibrated model produces.

```python
from insurance_monitoring import PITMonitor
from scipy.stats import poisson

monitor = PITMonitor(alpha=0.05)

# Each observation: compute PIT from predictive CDF, then update
for row in production_policies:
    mu = row["lambda_hat"] * row["exposure"]   # expected claims
    pit = float(poisson.cdf(row["claims"], mu))  # F(y | x)
    alarm = monitor.update(pit, exposure=row["exposure"])
    if alarm:
        print(f"Alarm at step {alarm.time} (changepoint ~{alarm.changepoint})")
        # Evidence {alarm.evidence:.1f} >= threshold {alarm.threshold:.1f}
        break

summary = monitor.summary()
print(f"Steps processed: {summary.t}, Evidence: {summary.evidence:.2f}")
```

The type I error guarantee is P(ever alarm | model calibrated) ≤ 0.05 for all t, forever. Standard monthly Hosmer-Lemeshow tests inflate type I error rapidly when run repeatedly. Use `update_many(pits)` to process a batch in one call.

---

## Credibility

### Recipe 17: How do I apply Bühlmann-Straub credibility to a fleet or scheme book?

`BuhlmannStraub` estimates the structural parameters (within-group variance v and between-group variance a) from the data, then computes credibility-weighted loss rates that blend each scheme's own experience with the portfolio mean.

```python
from insurance_credibility import BuhlmannStraub
import polars as pl

# df has one row per (scheme, year) with loss_rate and exposure
bs = BuhlmannStraub()
bs.fit(
    data=panel_df,
    group_col="scheme_id",
    period_col="policy_year",
    loss_col="loss_rate",      # claims / exposure
    weight_col="exposure",
)

results = bs.summary()
print(results)
# Collective mean  mu = 0.087
# Process variance  v = 0.000214   (EPV, within-group)
# Between-group var a = 0.000052   (VHM, between-group)
# Credibility param k = 4.12       (v / a)
#
# scheme_id  Exposure  Obs. Mean    Z   Cred. Premium  Complement
# SCH001         3.4      0.103  0.823          0.100       0.087
# SCH002         1.4      0.062  0.341          0.079       0.087
# SCH003        14.1      0.091  0.967          0.091       0.087
```

A credibility factor of 0.341 means a scheme has only 1.3 years of effective exposure - trust the portfolio mean heavily. At 0.967, the scheme is large enough to stand on its own experience.

---

## Specialist Methods

### Recipe 18: How do I smooth a noisy age-frequency curve with automatic lambda selection?

`WhittakerHenderson1D` minimises a penalised sum of squares with the smoothing parameter lambda selected automatically via REML. The result is a smoothed curve with Bayesian credible intervals - useful for informing driver age banding decisions.

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

ages = np.arange(17, 81)
# loss_ratios is the observed A/E ratio at each age, exposures is car-years
wh = WhittakerHenderson1D(order=2)   # order=2: penalise second differences
result = wh.fit(ages, loss_ratios, weights=exposures)

print(f"Selected lambda: {result.lambda_:.1f}")   # e.g. 847.3
print(f"EDF: {result.edf:.1f}")                   # effective degrees of freedom

smooth_df = result.to_polars()
# x, y, weight, fitted, ci_lower, ci_upper, std_fitted
print(smooth_df.filter(pl.col("x").is_between(17, 25)))
# x     y       weight  fitted  ci_lower  ci_upper
# 17  1.832     234.1   1.741     1.648     1.834
# 18  1.701     412.3   1.680     1.611     1.749
# ...
```

`order=2` penalises curvature (second differences) and produces smooth U-shaped curves, which is the right shape for driver age frequency. `order=1` penalises slope changes - appropriate for step functions like NCD.

---

### Recipe 19: How do I fit a frequency and severity trend to quarterly data?

`LossCostTrendFitter` fits log-linear trends to both frequency and severity simultaneously, then decomposes the combined loss cost trend and produces a prospective projection.

```python
from insurance_trend import LossCostTrendFitter

fitter = LossCostTrendFitter(
    periods=["2022Q1", "2022Q2", "2022Q3", "2022Q4",
             "2023Q1", "2023Q2", "2023Q3", "2023Q4",
             "2024Q1", "2024Q2", "2024Q3", "2024Q4"],
    claim_counts=[110, 108, 112, 107, 103, 105, 101, 98, 95, 97, 93, 91],
    earned_exposure=[1000, 998, 1005, 1002, 1010, 1008, 1012, 1009, 1015, 1013, 1018, 1016],
    total_paid=[572000, 581000, 589000, 595000, 578000, 603000, 614000, 622000, 601000, 629000, 641000, 653000],
    periods_per_year=4,
)

result = fitter.fit()
print(result.summary())
# Frequency trend:   -3.8% per annum  (95% CI: -5.1% to -2.5%)
# Severity trend:    +6.2% per annum  (95% CI: +4.8% to +7.7%)
# Loss cost trend:   +2.2% per annum  (compound)

print(result.decompose())
# Projected loss cost at Q4 2025: £657,200 (+2.2% from Q4 2024)
```

When frequency is falling (fewer claims) but severity is rising faster (larger individual claims), the loss cost trend can still be positive. Decomposing the two components is essential for understanding whether trend is structural (more expensive vehicles, higher repair costs) or behavioural.

---

### Recipe 20: How do I fit a shared frailty model to identify high-risk repeat claimants?

`AndersenGillFrailty` fits a Cox intensity model with shared gamma frailty via EM. The posterior frailty score for each policyholder is a Bühlmann-Straub credibility estimate of their individual risk relative to their observables - useful for NCD-style experience rating.

```python
from insurance_survival.recurrent.models import AndersenGillFrailty
from insurance_survival.recurrent.data import RecurrentEventData
import pandas as pd

# claims_df: one row per risk interval per policy
# columns: policy_id, start_days, stop_days, event, driver_age, vehicle_group
data = RecurrentEventData(
    df=claims_df,
    id_col="policy_id",
    start_col="start_days",
    stop_col="stop_days",
    event_col="event",
    covariate_cols=["driver_age", "vehicle_group"],
)

model = AndersenGillFrailty(frailty="gamma", max_iter=50, verbose=False)
model.fit(data)

summary = model.result.summary()
print(summary)
#                     coef    se      HR  HR_lower_95  HR_upper_95   p_value
# driver_age         0.022  0.004   1.022        1.014        1.030  < 0.001
# vehicle_group      0.031  0.007   1.032        1.018        1.045  < 0.001

print(f"Frailty theta: {model.result.theta:.3f}")
# theta=0.31 — moderate frailty; Var[z_i] = 1/theta = 3.2
# High theta (> 2) means frailty explains little; low theta means frailty dominates

# Posterior frailty scores — individual risk multipliers
credibility_scores = model.credibility_scores()
# Values > 1.0 identify policyholders whose actual claim rate exceeds
# what their observables predict
```

The frailty dispersion parameter theta is the key diagnostic. `theta ≈ 0.3` means substantial unexplained heterogeneity - the risk factors you have do not fully explain individual claim propensity, and the frailty scores contain useful information for experience rating.

---

## Installation

Each library is a separate package. Install only what you need:

```bash
uv add insurance-datasets insurance-cv
uv add "shap-relativities[all]"
uv add insurance-distill
uv add insurance-distributional
uv add insurance-fairness
uv add insurance-governance
uv add insurance-monitoring
uv add insurance-causal
uv add insurance-conformal
uv add insurance-credibility
uv add insurance-whittaker
uv add insurance-trend
uv add insurance-survival
```

All libraries are on PyPI and require Python 3.10+. CatBoost and Polars are core dependencies across most of them - no pandas except as a bridge where third-party tools require it (e.g., SHAP's TreeExplainer, lifelines).

---

## Related posts

- [How to Extract GLM-Style Rating Factors from a CatBoost Model](/2026/03/02/how-to-extract-rating-factors-from-catboost/)
- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/)
- [Why k-Fold CV Is Wrong for Insurance](/2026/03/21/why-k-fold-cv-is-wrong-for-insurance/)
- [Your Pricing Model Might Be Discriminating](/2026/03/03/your-pricing-model-might-be-discriminating/)
- [Your Pricing Model Is Drifting](/2026/03/03/your-pricing-model-is-drifting/)
- [Conformal Prediction Intervals for Insurance Pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)
- [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/)
