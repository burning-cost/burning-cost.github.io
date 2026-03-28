---
layout: post
title: "Causal AI for Pricing Actuaries: A Practical Guide"
date: 2026-03-25
categories: [pricing, techniques, tutorials]
tags: [causal-inference, double-machine-learning, dml, causal-forest, hte, did, its, insurance-causal, catboost, econml, doubleml, uk-personal-lines, price-elasticity, python]
description: "Why GLM coefficients aren't causal effects, and how to fix that using insurance-causal: DML with CatBoost nuisances, causal forests for heterogeneous treatment effects, and DiD/ITS rate change evaluation."
---

The Actuary Magazine ran a piece on causal AI this month. It covered the conceptual terrain well -- potential outcomes, the fundamental problem of causal inference, the difference between correlation and effect. What it did not cover is how to actually do this on insurance data, in Python, with code that works.

This post fills that gap. We will go from the specific problem with GLM price coefficients, through Double Machine Learning with CatBoost nuisance models, causal forests for heterogeneous treatment effects, and the `RateChangeEvaluator` for measuring what actually happened after your last rate change. Every code example uses the real `insurance-causal` API -- verified against the source.

```bash
uv pip install insurance-causal
```

---

## The problem with GLM price coefficients

Here is the scenario most UK pricing teams are in. You run a renewal conversion GLM. Age, vehicle group, NCD band, postcode sector, channel -- the standard set. You include a `pct_price_change` variable so you can read off the price elasticity. The GLM converges. You report a coefficient of, say, -0.45 on `log(1 + pct_price_change)`. The team interprets this as: for every 1% price increase, conversion drops 0.45 percentage points.

That number is wrong. Not imprecise -- structurally wrong.

The reason is confounding. Price changes are not random. You applied larger increases to high-risk customers, to segments experiencing adverse development, to customers who came in on introductory rates. Those same customers have different baseline conversion rates that are driven by the same risk factors. The regression picks up both the causal price effect and the correlation between "received a large increase" and "was already likely to lapse." You cannot separate them by adding more controls to the GLM, because the controls are themselves part of the causal pathway.

The formal name for this is **omitted variable bias** through confounding, and it is present in every observational pricing model that includes a price term alongside risk factors.

The solution is **Double Machine Learning (DML)**, introduced by Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, and Robins in the *Econometrics Journal* (2018, Vol 21(1): C1-C68). The idea is elegant: partial out the influence of the confounders from both the outcome and the treatment separately, using flexible machine learning models, and then regress the residual outcome on the residual treatment. The residual treatment is, by construction, the part of the price change not explained by the risk factors -- the exogenous variation. The coefficient on it is the causal effect.

---

## DML in practice: `CausalPricingModel`

The `insurance-causal` library wraps DoubleML's `DoubleMLPLR` (partially linear regression) with CatBoost nuisance models and an interface designed for pricing actuaries.

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(
    outcome="renewed",
    outcome_type="binary",
    treatment=PriceChangeTreatment(column="pct_price_change", scale="log"),
    confounders=["driver_age", "vehicle_group", "ncd_years", "postcode_band", "channel"],
    cv_folds=5,
    random_state=42,
)
model.fit(df)
ate = model.average_treatment_effect()
print(ate)
```

The output is an `AverageTreatmentEffect` dataclass:

```
Average Treatment Effect
  Treatment: pct_price_change
  Outcome:   renewed
  Estimate:  -0.3821
  Std Error:  0.0143
  95% CI:    (-0.4101, -0.3541)
  p-value:   0.0000
  N:         87,432
```

The `PriceChangeTreatment` with `scale="log"` applies `log(1 + D)` to the raw percentage change. The resulting coefficient is a semi-elasticity: a 1-unit increase in `log(1 + pct_price_change)` is associated with a `-0.38` change in the renewal probability. For a 10% price increase, `log(1.10) ≈ 0.095`, so the implied conversion impact is about -3.6 percentage points.

Under the hood, this is running two CatBoost models: one for `E[renewed | X]` (the outcome nuisance) and one for `E[pct_price_change | X]` (the treatment nuisance / propensity model), with 5-fold cross-fitting so the nuisance estimation errors are asymptotically independent of the causal score. The adaptive regularisation schedule in the library automatically reduces CatBoost capacity for datasets under 10,000 observations to avoid a known over-partialling problem where high-capacity nuisance models absorb treatment signal and leave nothing to identify from.

### How biased was the GLM?

```python
bias_report = model.confounding_bias_report(naive_coefficient=-0.61)
```

This returns a DataFrame with columns `naive_estimate`, `causal_estimate`, `bias`, `bias_pct`, and an `interpretation` field. A result like `bias_pct = +37.4` means the GLM overstated the true price sensitivity by 37% -- a large enough error to produce systematically incorrect demand models and, consequently, suboptimal renewal pricing.

The direction of the bias is not always the same. In some portfolios the GLM understates sensitivity because high-risk customers who received large increases are less price-sensitive than average. The library will tell you.

---

## More nuance: nuisance model quality

DML is only as good as its nuisance models. If `E[D|X]` is a poor fit -- meaning the CatBoost model cannot predict the price change from the confounders -- the treatment residuals carry a lot of unexplained variation, which is good for identification. But if the model fits too well, with R² above 0.95, the price changes are nearly deterministic given the rating factors and there is almost no exogenous variation to identify from.

```python
from insurance_causal import diagnostics

report = diagnostics.nuisance_model_summary(model)
print(report)
# {'outcome_r2': 0.341, 'treatment_r2': 0.427, 'treatment_residual_variance': 0.0031}
```

A `treatment_r2` above 0.95 triggers an explicit warning. This is the signal that you need to think harder about the source of identification -- for instance, whether there is an instrument (a tariff revision that affected some segments but not others) that would give you cleaner variation to work with.

---

## Heterogeneous treatment effects: causal forests

An ATE of -0.38 is a population average. In most renewal books, the price sensitivity of a 25-year-old with NCD=0 coming through a price comparison website is not the same as a 55-year-old with NCD=5 on direct renewal. The average obscures the variation that actually matters for pricing decisions -- and for FCA fairness analysis (more on that below).

The `causal_forest` subpackage implements `CausalForestDML` from EconML (Athey, Tibshirani & Wager, *Annals of Statistics* 2019) with CatBoost nuisances and the formal heterogeneity inference framework from Chernozhukov, Demirer, Duflo and Fernandez-Val (2020/2025, published in *Econometrica*).

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
    make_hte_renewal_data,
)

# Synthetic renewal dataset with known heterogeneity by NCD
df = make_hte_renewal_data(n=10_000, seed=42)

confounders = ["age", "ncd_years", "vehicle_group", "channel"]

est = HeterogeneousElasticityEstimator(
    n_estimators=200,
    catboost_iterations=200,
    min_samples_leaf=20,
    binary_outcome=True,
    random_state=42,
)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)
```

The `min_samples_leaf=20` default is larger than EconML's default of 5. Insurance rating segments are often sparse -- there may be fewer than 200 policies with NCD=0 in the PCW channel under age 25 -- and leaves with fewer than 20 observations produce unreliable CATE estimates. Increase it further for sparser books.

### Per-customer CATEs

```python
cates = est.cate(df)           # numpy array, shape (n,)
lb, ub = est.cate_interval(df, alpha=0.05)   # 95% intervals per row
ate, ate_lb, ate_ub = est.ate()
```

The `cate()` method calls `CausalForestDML.effect()` on the feature matrix. Each element is the estimated semi-elasticity for that policyholder.

### Group average treatment effects by NCD

```python
gates_by_ncd = est.gate(df, by="ncd_years")
```

This returns a Polars DataFrame with columns `ncd_years`, `cate`, `ci_lower`, `ci_upper`, `n`. If any group has fewer than 500 observations the library emits a warning. In the synthetic dataset with known heterogeneity, the GATE estimates should increase monotonically with NCD -- customers who have been with you longest are the least price-sensitive.

---

## Formal heterogeneity testing: BLP, GATES, CLAN

Getting per-customer CATEs is the easy part. The harder question is: does the heterogeneity we observe reflect true variation in price sensitivity, or is it noise from the forest? The library implements the three-part inference framework from Chernozhukov et al. (2020/2025):

```python
inf = HeterogeneousInference(n_splits=100, k_groups=5)
result = inf.run(df, estimator=est, cate_proxy=cates)
print(result.blp)
```

**BLP (Best Linear Predictor).** Regresses the realised treatment effect on the CATE proxy, using independent data splits to avoid circularity. The key number is `beta_2`: the coefficient on `(W - e_hat) * (S - S_bar)` where `S` is the CATE estimate. If `beta_2 > 0` and statistically significant, heterogeneity is real.

```
BLPResult(beta_1=-0.3812, beta_2=0.2341***, p=0.0003, heterogeneity=True)
```

**GATES.** Sorts customers into 5 quintiles by estimated CATE and estimates the average treatment effect within each. The GATE table should be monotone increasing if the forest has correctly ordered the heterogeneity.

```python
print(result.gates.table)
```

**CLAN (Classification Analysis).** Compares covariate means between the top and bottom GATE groups -- i.e., the most and least price-sensitive customers. This tells you which risk factors are actually driving the heterogeneity, which is the commercially useful output.

```python
print(result.clan.table)
```

The CLAN table shows feature means for the top and bottom quintiles, with t-statistics. If `ncd_years` has a large mean difference between groups, NCD is a driver of price sensitivity -- which is sensible but worth verifying in the data rather than assuming.

---

## Continuous treatment AME: `PremiumElasticity`

The `CausalPricingModel` uses the partially linear DML estimator, which treats the treatment as entering the outcome equation linearly (after partialling out confounders). For most applications that is fine. Where it can struggle is with continuous premium treatments over a wide range -- say, actual premium charged across a portfolio where the highest premiums are 10x the lowest.

The `autodml` subpackage implements the Automatic Debiased ML estimator (Chernozhukov et al., *Econometrica* 2022, Vol. 90(3): 967-1027). The key difference: it avoids estimating the generalised propensity score entirely, using a Riesz representer approach that is more numerically stable for continuous treatments.

```python
from insurance_causal.autodml import PremiumElasticity

model = PremiumElasticity(
    outcome_family="poisson",
    n_folds=5,
    nuisance_backend="catboost",
    riesz_type="forest",
    random_state=42,
)
model.fit(X, D, Y, exposure=exposure)
result = model.estimate()
print(result.summary())
```

Here `X` is a feature matrix (numpy array or pandas DataFrame), `D` is the continuous treatment (actual premium or log-premium), `Y` is the outcome (claim count for Poisson), and `exposure` is earned years at risk. The `estimate()` method returns an `EstimationResult` with `estimate`, `se`, `ci_low`, `ci_high`.

For segment-level effects, the EIF (efficient influence function) scores decompose additively, so no refitting is needed:

```python
segment_results = model.effect_by_segment(df["channel"])
for seg in segment_results:
    print(seg.segment_name, seg.result.estimate, seg.result.ci_low, seg.result.ci_high)
```

---

## Measuring what your last rate change actually did: `RateChangeEvaluator`

You changed rates on young drivers in January 2026. Three months later, someone asks: what actually happened to conversion? The naive comparison -- January-March conversion vs. the same period last year -- is confounded by everything that changed in the market between January 2025 and January 2026: Motor claims inflation, the FCA motor market review activity in early 2024, book mix shifts. You cannot attribute the conversion change to your rate action without controlling for those factors.

The library implements two approaches. **Difference-in-Differences (DiD)** when you have a control group (segments that did not receive the rate change). **Interrupted Time Series (ITS)** when the entire book was treated. DiD has stronger identification because the control group absorbs concurrent market effects; ITS identifies from the pre-intervention trend and is much more sensitive to confounding.

```python
from insurance_causal.rate_change import RateChangeEvaluator, make_rate_change_data

# Synthetic panel: 40 segments, 20 post periods, true ATT = -5%
df = make_rate_change_data(n_segments=40, true_att=-0.05, seed=0)

evaluator = RateChangeEvaluator(
    outcome_col="outcome",
    treatment_period=9,         # period in which rate change took effect
    unit_col="segment",           # for cluster-robust SEs
    weight_col="earned_exposure",  # strongly recommended
)
result = evaluator.fit(df)
print(result.summary())
```

The evaluator automatically selects DiD or ITS based on whether the `treated` column contains both 0 and 1 values. It runs a parallel trends test and warns if the pre-trend F-statistic is significant at 10% -- the key diagnostic for whether DiD identification is credible.

The `UK_INSURANCE_SHOCKS` dictionary in the library lists ten UK market events between 2017 and 2024 that can confound rate change estimates: the Ogden rate cuts in 2017-Q1, GIPP implementation in 2022-Q1, the motor claims inflation peak in 2023-Q1, and others. If your `treatment_period` falls within two quarters of any of these, the library emits a warning automatically.

```python
from insurance_causal.rate_change import UK_INSURANCE_SHOCKS
from insurance_causal.rate_change._shocks import check_shock_proximity

# Check if our treatment period is near any known shock
nearby = check_shock_proximity("2022-Q2", proximity_quarters=2)
```

### ITS for whole-book changes

```python
# make_rate_change_data with mode="its" used below
df_its = make_rate_change_data(true_level_shift=-0.02, seed=0, mode="its")

evaluator = RateChangeEvaluator(
    outcome_col="outcome",
    treatment_period=9,
    weight_col="earned_exposure",
)
result = evaluator.fit(df_its)
```

ITS fits a segmented regression with a level shift and slope change at the treatment period, using Newey-West HAC standard errors to account for autocorrelation. The `add_seasonality=True` parameter adds quarterly dummies -- important for loss ratio time series that carry seasonal renewal patterns.

---

## Fairness implications

Price sensitivity heterogeneity has a direct connection to FCA pricing fairness requirements. If your causal forest shows that customers in protected groups (by age, by geography acting as a proxy for ethnicity) have systematically lower price sensitivity, and you have been using population-average elasticities to set renewal prices, you may be giving smaller discounts to those groups than their actual sensitivity warrants.

The `insurance-fairness` library's `FairnessAuditor` works naturally with CATE outputs: compare the CATE distribution across protected groups and test whether the means are statistically different. The CLAN output from `HeterogeneousInference` is directly useful here -- if protected-class proxies appear at the top of the CLAN covariate table, that is a flag worth taking seriously before the next renewal pricing cycle.

---

## Monitoring the causal estimates over time

Causal estimates are not static. If the book mix shifts substantially -- a new aggregator distribution, a change in the customer base that finds you -- the population-level ATE will change even if the underlying structural relationship between price and conversion does not. The [`insurance-monitoring`](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) library's drift detection framework can be applied to the DML treatment residual distribution: if `E[D|X]` has changed, the nuisance model is out of date and the causal estimate will be biased.

Track the treatment nuisance R² from `diagnostics.nuisance_model_summary()` as a monthly KPI. A sudden increase in treatment R² means pricing is becoming more deterministic -- the tariff is increasingly determining who gets what rate without residual variation, and the causal estimate will have larger standard errors. A sudden decrease means there is new unexplained variation in the treatment, which could indicate data quality issues or an unmodelled pricing source.

---

## What DML cannot do

Two things worth being clear about.

**DML cannot adjust for unobserved confounders.** The identification assumption is conditional ignorability: all variables that jointly affect the treatment and outcome are in the confounder set. If actual annual mileage affects both the premium offered (through telematics or stated mileage) and conversion (high-mileage customers are more price-sensitive), and you only have stated mileage, your estimate is still biased. The `diagnostics.sensitivity_analysis()` method in the library is currently being redesigned (the previous formula was incorrect and has been removed); for now, reason carefully about what confounders you may have missed.

**DML requires exogenous variation in the treatment.** If the pricing system applies exactly the technical premium to every customer with no commercial loading, no underwriter discretion, and no band rounding, the price change is entirely determined by the rating factors in the model. There is no exogenous variation to identify from. The treatment nuisance R² will approach 1. The standard error of the ATE will be very large. This is not a failure of the library; it is a fundamental identification problem. If you do not have natural variation, you need an instrument -- a tariff revision that was rolled out to some segments but not others, for example.

---

## Summary

GLM price coefficients are confounded. DML removes that confounding by partialling out the risk factor influence from both outcome and treatment before regressing residual on residual. The `insurance-causal` library makes this straightforward: `CausalPricingModel` handles the plumbing, the `causal_forest` subpackage gives you per-customer CATEs with formal heterogeneity tests, and `RateChangeEvaluator` gives you credible estimates of what your last rate action actually caused.

The library is at `uv pip install insurance-causal`. The source is on GitHub at [insurance-causal](https://github.com/burningcost/insurance-causal). If you are reading this having come from the Actuary Magazine piece and want to talk through an application to your book, [get in touch](/work-with-us/).

- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/)  -  the full library post with the complete DML pipeline: conversion model, retention model, ENBP compliance checker, and demand curves
- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/)  -  post-deployment monitoring that tells you whether the causal elasticity estimate has remained stable as the portfolio shifts

---

*References: Chernozhukov et al. (2018) "Double/Debiased Machine Learning," Econometrics Journal 21(1): C1-C68. Chernozhukov et al. (2022) "Automatic Debiased Machine Learning," Econometrica 90(3): 967-1027. Athey, Tibshirani & Wager (2019) "Generalized Random Forests," Annals of Statistics 47(2): 1148-1178. Chernozhukov, Demirer, Duflo & Fernandez-Val (2020/2025) "Generic Machine Learning Inference on Heterogeneous Treatment Effects," Econometrica (forthcoming). Wager & Athey (2018) "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests," JASA 113(523): 1228-1242.*
