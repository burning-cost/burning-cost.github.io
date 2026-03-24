---
layout: post
title: "The Complete Python Motor Pricing Stack: From Raw Data to Production"
date: 2026-03-24
categories: [guides, tutorials]
tags: [python, motor-pricing, pipeline, insurance-datasets, insurance-whittaker, insurance-gam, insurance-distill, shap-relativities, insurance-fairness, insurance-monitoring, insurance-conformal, insurance-optimise, catboost, fca, ebm, glm, gbm, uk-insurance]
description: "How eight Burning Cost libraries fit together in a real UK motor pricing workflow: data generation, Whittaker smoothing, EBM tariff, GBM-to-GLM distillation, FCA fairness audit, drift monitoring, conformal intervals, and constrained rate optimisation."
---

Most posts about pricing tools treat each library as a standalone thing. You install it, run the quickstart, close the tab. But a UK motor pricing workflow is not a collection of isolated steps: the smoothed age curve feeds the EBM model, the EBM factors go into the distillation step, the distilled factor table informs the rate optimiser, and monitoring tells you when the whole thing needs to be revisited.

This post shows how eight Burning Cost libraries connect in practice. It is an overview — each section has its own deep-dive post linked at the end. What we will show is which class to call, what output to expect, and why it feeds the next step.

The full workflow:

1. Synthetic data with a known DGP — `insurance-datasets`
2. Age and area curve smoothing — `insurance-whittaker`
3. EBM tariff model — `insurance-gam`
4. Factor table extraction — `insurance-distill` (or `shap-relativities` for CatBoost)
5. FCA fairness audit — `insurance-fairness`
6. Drift monitoring setup — `insurance-monitoring`
7. Conformal prediction intervals — `insurance-conformal`
8. Constrained rate optimisation — `insurance-optimise`

---

## Setup

```bash
pip install insurance-datasets insurance-whittaker "insurance-gam[ebm]" \
    insurance-distill "shap-relativities[ml]" insurance-fairness \
    insurance-monitoring "insurance-conformal[catboost]" insurance-optimise \
    catboost polars
```

---

## 1. Data

`insurance-datasets` generates synthetic UK motor policies from a published Poisson-Gamma DGP. The true coefficients are exported alongside the data, which means you can check whether your fitted model recovers them — something impossible with the standard freMTPL2 benchmark, which has unknown true parameters.

```python
from insurance_datasets import load_motor, TRUE_FREQ_PARAMS

df = load_motor(n_policies=50_000, seed=42)
# (50000, 18): policy_id, vehicle_age, vehicle_group (ABI 1-50),
# driver_age, ncd_years (0-5), conviction_points, annual_mileage,
# area (A-F), claim_count, incurred, exposure, ...

print(TRUE_FREQ_PARAMS)
# {'driver_age_young': -0.55, 'ncd_years': -0.12, 'area_F': 0.31, ...}
```

Split on `inception_year` rather than randomly. A random split leaks future policy experience into the training window, inflating Gini by 3–7 points on a typical motor book. Use `inception_year <= 2022` for training and `inception_year == 2023` for validation.

---

## 2. Smoothing age and area curves

Before fitting any model, smooth the one-way experience tables. Raw age curves have noise from thin cells — age 47 appearing cheaper than age 46 for no underwriting reason. `insurance-whittaker` applies Whittaker-Henderson penalised smoothing with automatic REML lambda selection, following Biessy (2026, ASTIN Bulletin).

```python
import polars as pl
from insurance_whittaker import WhittakerHenderson1D

one_way = (
    df.group_by("driver_age")
    .agg([pl.col("claim_count").sum(), pl.col("exposure").sum()])
    .sort("driver_age")
)

ages      = one_way["driver_age"].to_numpy()
freq_raw  = (one_way["claim_count"] / one_way["exposure"]).to_numpy()
exposures = one_way["exposure"].to_numpy()

wh = WhittakerHenderson1D(order=2, lambda_method="reml")
result = wh.fit(ages, freq_raw, weights=exposures)

smoothed_df = result.to_polars()
# columns: x, y, weight, fitted, ci_lower, ci_upper
# result.lambda_  → e.g. 847.3  (selected automatically)
# result.edf      → e.g. 5.2    (effective degrees of freedom)
```

The Bayesian credible intervals on `ci_lower` and `ci_upper` are what distinguish this from a manual Excel smooth. Age 17–21 bands on a typical UK motor book have intervals wide enough that you should not be optimising pricing in those cells without additional data. For 2-D smoothing — an age × NCD cross-table — `WhittakerHenderson2D` takes the same arguments with a 2-D input array.

---

## 3. EBM tariff model

A Poisson GLM is interpretable and regulators understand it. An EBM (`insurance-gam`) is also interpretable — the shape functions are the model, equivalent to GLM factors, but the non-linearity is discovered rather than assumed. On our synthetic motor DGP, the EBM recovers a U-shaped driver age hazard and a convex NCD discount curve that a standard polynomial GLM misses.

```python
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

features = ["driver_age", "vehicle_age", "ncd_years",
            "annual_mileage", "area_code", "vehicle_group"]

X_train   = df_train.select(features)
y_train   = df_train["claim_count"].to_numpy()
exp_train = df_train["exposure"].to_numpy()

model = InsuranceEBM(loss="poisson", interactions="3x")
model.fit(X_train, y_train, exposure=exp_train)

rt = RelativitiesTable(model)
print(rt.table("ncd_years"))
# shape_value  relativity
# 0            1.000
# 1            0.861
# 3            0.694
# 5            0.521

print(rt.summary())
# feature          edf    contribution
# driver_age       6.1    0.31
# ncd_years        2.4    0.18
# vehicle_group    4.8    0.14
```

The `RelativitiesTable` output is in `exp(β)` format — identical to a GLM. A pricing committee can challenge it factor by factor. The `interactions="3x"` setting caps pairwise interaction terms at three, which is enough to capture most structure on a standard motor book without overfitting.

---

## 4. Factor table extraction: from GBM to GLM

Not every team will want an EBM in production. Many prefer a CatBoost GBM for its predictive performance and then need to translate the model into a multiplicative factor table for Radar or Emblem. Two libraries handle this, depending on your workflow.

**GLM surrogate** — one Poisson GLM trained to replicate the GBM's predictions, with automatic binning:

```python
from insurance_distill import SurrogateGLM

surrogate = SurrogateGLM(
    model=catboost_model,
    X_train=X_train,
    y_train=y_train,
    exposure=exp_train,
    family="poisson",
)
surrogate.fit(max_bins=10, interaction_pairs=[("driver_age", "area_code")])

report = surrogate.report()
print(report.metrics.summary())
# Gini (GBM):          0.3241
# Gini (surrogate):    0.3087   ← 95.2% retention
# Max segment error:   8.3%
```

The surrogate retains 90–97% of the GBM's Gini in our benchmarks. That gap is the price of interpretability; most pricing teams consider it acceptable.

**Continuous factor curves** — the full age relativity curve, no binning required:

```python
from shap_relativities import SHAPRelativities

sr = SHAPRelativities(model=catboost_model, X=X_train, exposure=exp_train)
sr.fit()

rels = sr.extract_relativities()
# pl.DataFrame: feature, level, relativity, lower_ci, upper_ci, n_obs, exposure_weight

age_table = rels.filter(pl.col("feature") == "driver_age")
age_table.write_csv("age_relativities.csv")
# feature     level  relativity  lower_ci  upper_ci  exposure_weight
# driver_age  17.0   1.84        1.61      2.10      142
# driver_age  18.0   1.72        1.54      1.92      287
```

We use `shap-relativities` when the age curve shape is genuinely uncertain and we want the model to find the banding rather than imposing it. We use `insurance-distill` when we need a clean GLM that a Radar import template can consume directly.

---

## 5. Fairness audit

Before anything goes near production, run an FCA proxy discrimination check. `insurance-fairness` identifies which rating factors act as proxies for protected characteristics — area code is the canonical UK example, with Citizens Advice (2022) estimating a £280/year ethnicity penalty in UK motor insurance driven entirely through postcode-correlated factors. The library computes exposure-weighted calibration by group and produces a Markdown report with explicit PRIN 2A and TR24/2 references.

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=catboost_model,
    data=df_test,
    protected_cols=["gender"],
    prediction_col="model_prediction",
    outcome_col="claim_count",
    exposure_col="exposure",
    factor_cols=["area_code", "vehicle_group", "ncd_years"],
)

report = audit.run()

print(report.overall_rag)
# 'amber'

print(report.flagged_factors)
# ['area_code']  ← proxy_r2 > 0.10 threshold

report.to_markdown("fairness_audit_2026Q1.md")
```

A `proxy_r2 > 0.10` means the factor explains more than 10% of variance in the protected characteristic — the threshold at which the FCA TR24/2 (August 2024) review found most insurers' assessments were inadequate. The `to_markdown()` output maps every finding to the specific regulatory reference and is structured for a pricing committee pack or FCA file review.

---

## 6. Monitoring setup

Production models drift. The aggregate A/E ratio is a poor early-warning signal because it cancels errors across segments. A model that is 15% cheap on under-25s and 15% expensive on mature drivers reads 1.00 at portfolio level and raises no alarm. `insurance-monitoring` monitors the features, not just the headline number.

Set this up against your training reference period before deployment:

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",
)

print(report.recommendation)
# 'RECALIBRATE' | 'REFIT' | 'NO_ACTION' | 'INVESTIGATE' | 'MONITOR_CLOSELY'

print(report.to_polars().filter(pl.col("band") != "green"))
# metric            value   band
# ae_ratio          1.08    amber
# gini_p_value      0.054   amber
# csi_driver_age    0.14    amber  ← book skewing older
```

The three failure modes are separated: covariate shift (CSI per feature), calibration drift (A/E with Poisson confidence interval), and discrimination decay (Gini drift z-test from arXiv 2510.04556). RECALIBRATE means a multiplicative scalar fix — a few hours of work. REFIT means the model needs rebuilding. An undifferentiated "drift detected" alert cannot tell you which one you are looking at.

---

## 7. Conformal prediction intervals

Point estimates from a Tweedie GBM carry implicit uncertainty that standard parametric intervals handle poorly on heterogeneous motor books. `insurance-conformal` wraps any fitted model with a distribution-free coverage guarantee using the Pearson-weighted non-conformity score: `|y − ŷ| / ŷ^(p/2)`. This accounts for the Tweedie heteroscedasticity that the raw absolute residual ignores.

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=catboost_model,
    score_type="pearson_weighted",
    coverage=0.90,
)
cp.calibrate(X_cal, y_cal)

intervals = cp.predict_interval(X_test)
# pl.DataFrame: prediction, lower, upper, width

print(cp.empirical_coverage(y_test, intervals))
# 0.903   ← guaranteed >= 0.90 by construction
```

On a heteroskedastic UK motor DGP (50,000 policies, seed=42), Pearson-weighted intervals are 13–14% narrower than parametric Tweedie intervals at the same 90% coverage level. The guarantee is finite-sample valid — it holds regardless of the claim distribution. Over-wide intervals on low-risk policies are wasted capital. Under-wide intervals on high-risk policies are the ones that matter most when they are wrong.

---

## 8. Rate optimisation

The final step turns the technical price into a quote. `insurance-optimise` solves for the set of price multipliers that maximises expected profit subject to FCA PS21/5 ENBP constraints (renewal price ≤ new business equivalent), a target loss ratio, and a retention floor.

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

config = ConstraintConfig(
    lr_max=0.72,
    retention_min=0.80,
    max_rate_change=0.20,
    enbp_buffer=0.01,
    technical_floor=True,
)

opt = PortfolioOptimiser(
    technical_price=technical_price,      # from EBM or surrogate GLM
    expected_loss_cost=expected_loss_cost,
    p_renewal=p_renewal,
    price_elasticity=price_elasticity,    # from insurance-causal or manual estimate
    is_renewal=is_renewal,
    enbp=enbp_ceiling,
    constraints=config,
)

result = opt.optimise()
print(result.summary())
# Expected profit:   £2.41M  (+18.3% vs flat loading)
# Portfolio LR:      0.719
# Retention:         82.4%
# ENBP violations:   0

frontier = opt.efficient_frontier(n_points=20)
# Profit-retention Pareto curve for the pricing committee
```

The efficient frontier changes the conversation with underwriting. Instead of "what multiplier do we use?", the question becomes "where on this curve do we want to operate?" Every point is feasible, ENBP-compliant, and annotated with the implied combined ratio. The optimiser runs in under a minute on a 10,000-policy renewal book using analytical gradients throughout.

---

## How the pieces fit

The workflow is linear but the feedback loops are what matter:

- **Whittaker** smoothed curves → validate the EBM's one-way output and inform feature engineering
- **EBM** or **CatBoost** → source model for the distillation or SHAP extraction step
- **Distill / shap-relativities** → factor table that feeds the rating engine and the optimiser's `technical_price`
- **Fairness audit** → runs before production sign-off; proxy flags may require re-engineering features upstream
- **Monitoring** → runs monthly post-deployment; a REFIT recommendation restarts the cycle from step 3
- **Conformal** → uncertainty bounds that feed the optimiser's scenario mode (run under the upper CI for conservative pricing)
- **Optimise** → final quote prices, ENBP-enforced, with a JSON audit trail for every run

None of these are optional on a production UK motor book. The fairness audit is live FCA enforcement risk — TR24/2 opened six Consumer Duty investigations. Monitoring is how you avoid a 12-month lag between model failure and loss ratio deterioration. The conformal intervals are the only way to make defensible statements about pricing uncertainty.

---

## Going deeper

Each library has its own post:

- [Whittaker-Henderson Smoothing for Insurance Pricing](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/)
- [Your Model Is Either Interpretable or Accurate. insurance-gam Refuses That Trade-Off.](/2026/03/14/insurance-gam-interpretable-nonlinearity/)
- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/)
- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/)
- [Your Pricing Model Might Be Discriminating](/2026/03/07/your-pricing-model-might-be-discriminating/)
- [Your Pricing Model Is Drifting](/2026/03/03/your-pricing-model-is-drifting/)
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/03/06/conformal-prediction-intervals-for-insurance-pricing/)
- [Constrained Rate Optimisation: The Efficient Frontier](/2026/02/21/constrained-rate-optimisation-efficient-frontier/)

All libraries are on PyPI. All have Databricks notebooks in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples).
