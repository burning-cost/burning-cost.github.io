---
layout: post
title: "Every Python Library for Insurance Pricing in 2026 - A Practitioner's Guide"
date: 2026-03-23
categories: [tools, guides]
description: "A comprehensive guide to Python libraries for insurance pricing - from GLMs to causal inference, monitoring, and fairness auditing."
---

If you work in insurance pricing and you want to use Python seriously, the first problem you hit is that there is no single reference for what tools exist. The generic data science ecosystem — scikit-learn, XGBoost, pandas — was not designed for actuarial work. The actuarial ecosystem in Python is scattered, poorly documented, and mostly either life insurance or reserving. A pricing actuary looking for a credible guide to the Python landscape gets nothing useful.

This is that guide. We cover every category of work a pricing actuary does, what standard tools exist, where they fall short, and where specialised libraries help. We are direct about when the standard tool is sufficient — there is no reason to add a dependency if scikit-learn solves your problem.

This is written for UK general insurance pricing actuaries. It assumes you have used Emblem or Radar, you know what a Poisson GLM is, and your question is "what Python tools should I actually be using?"

---

## GLM and tariff building

### The standard option: glum

[glum](https://github.com/Quantco/glum) from Quantco is the right Python GLM library for pricing work. It has a scikit-learn-compatible API, supports Poisson, Gamma, Tweedie, and Binomial families with the correct log/logit links, handles offset terms natively in `fit()`, and returns coefficient standard errors. It runs 3-10x faster than sklearn's IRLS implementation on portfolios of 100k+ policies.

```python
from glum import GeneralizedLinearRegressor
import numpy as np

model = GeneralizedLinearRegressor(family="poisson", l2_regularization=0.01)
model.fit(X, y, sample_weight=None, offset=np.log(exposure))
```

Use glum. Do not use `sklearn.linear_model.PoissonRegressor` for production pricing unless the dataset is small — it does not give you standard errors and it is slower.

`statsmodels.genmod.GLM` is the other option. It has better diagnostics output (AIC, BIC, deviance tables) but does not support the per-coefficient L2 penalty matrices that glum provides. For exploratory work and formal statistical tests, statsmodels is useful. For fitting the production model, glum.

### Where Emblem sits in this

Emblem remains the dominant GLM tool in UK personal lines. It has a commercial support contract, integrates with Radar, and handles rate change audits in a way that Python currently does not. We are not arguing you should replace Emblem with Python. The realistic mode of operation is Python for development and exploration, Emblem or Radar for production deployment and governance sign-off.

The GBM-to-GLM path — fitting a gradient boosting model, extracting SHAP-based factor relativities, and distilling those into a GLM that Emblem can consume — is where Python adds the most value upstream of Emblem.

### Specialised: insurance-glm-tools

Our [insurance-glm-tools](https://github.com/burningcost/insurance-glm-tools) library sits on top of glum and handles the actuarial-specific bookkeeping: reference level rebasing, factor table extraction with standard errors, iterative proportional fitting, and interaction diagnostics using normalised deviance. These are things you would otherwise write manually every time.

---

## Gradient boosting models

### CatBoost

We use [CatBoost](https://catboost.ai/) for gradient boosting in pricing work. The specific reasons:

- Native categorical handling via ordered target statistics (no manual encoding required for factors like vehicle group or region)
- Tweedie loss function built in, with correct deviance implementation for frequency-severity combined models
- Built-in SHAP support via `get_feature_importance(type='ShapValues')` using exact TreeSHAP, not approximate
- Pool object for specifying cat_features, which matters when your feature matrix contains a mix of continuous and categorical rating factors

```python
from catboost import CatBoostRegressor, Pool

pool = Pool(X_train, y_train, cat_features=cat_cols, weight=exposure)
model = CatBoostRegressor(loss_function="Tweedie:variance_power=1.5", iterations=500)
model.fit(pool)
```

LightGBM and XGBoost are credible alternatives. CatBoost handles unencoded categoricals without preprocessing, which in practice saves significant pipeline complexity on a UK motor book with 9,000 postcode sectors.

### shap-relativities

Raw SHAP values from a CatBoost model are additive contributions in log-premium space. To turn them into multiplicative relativities that an actuary can read as a factor table, you need to rebase, exponentiate, and handle interactions correctly.

[shap-relativities](https://github.com/burningcost/shap-relativities) does this. It takes a fitted CatBoost model and a feature matrix, returns a factor table per feature with relativities, credibility-weighted smoothed versions, and a base premium. The output format is designed to import into Emblem or Radar.

```python
from shap_relativities import SHAPRelativities

rel = SHAPRelativities(model, X, feature_names=feature_names, exposure=exposure)
factor_table = rel.get_factor_table("VehicleGroup")
```

---

## Frequency-severity and distributional models

### The standard split

The standard approach is a two-model pipeline: Poisson for claim frequency, Gamma for average claim severity. Both via glum. This is correct for UK regulatory purposes — you need to show separate frequency and severity factors for Consumer Duty fair value analysis and for reserve review credibility.

Tweedie compound Poisson-Gamma in a single model is faster to fit but loses the auditability. For submission to Lloyd's or for PRA SS1/23 model risk documentation, the two-model path is cleaner.

### Specialised models

[insurance-severity](https://github.com/burningcost/insurance-severity) handles the distributional modelling problem: fitting Gamma, log-normal, Burr, and Pareto distributions to claim severity, with large loss identification, ILF derivation, and credibility blending of fitted and empirical distributions at the layer boundary.

For spliced distributions — a body model below a retention point and a GPD tail above it — `scipy.stats` contains the distributions but does not handle the splicing and fitting as a joint optimisation. [insurance-distributional](https://github.com/burningcost/insurance-distributional) wraps this into a single `SplicedDistribution` class.

If you want full quantile regression for pricing (predicting the conditional distribution, not just the mean), [insurance-quantile](https://github.com/burningcost/insurance-quantile) extends CatBoost's quantile loss with actuarial-specific features: exposure weighting, coherent quantile crossing correction, and calibration diagnostics.

---

## Cross-validation and model selection

### Why k-fold is wrong

Standard k-fold cross-validation does not work correctly for insurance pricing data. Insurance policies have temporal structure: policies from 2022 inform rate changes in 2023. A fold that includes future policies in training is leaking development patterns forward in time. On a 5-year motor book, k-fold can overstate model performance by 10-15 Gini points.

The correct approach is walk-forward temporal cross-validation with an IBNR development buffer between training and validation folds.

### insurance-cv

[insurance-cv](https://github.com/burningcost/insurance-cv) implements this. It takes an accident date column, a development buffer (typically 18-24 months for UK motor), and a number of folds, and returns train/validation splits that respect the temporal ordering. It integrates directly with glum and CatBoost.

```python
from insurance_cv import TemporalCVSplitter

splitter = TemporalCVSplitter(date_col="accident_date", buffer_months=18, n_splits=5)
for train_idx, val_idx in splitter.split(df):
    # fit on train, evaluate on val
```

---

## Experience rating and credibility

### The standard: no Python tool exists

There is no widely-used Python library for Bühlmann-Straub credibility weighting or experience rating in the actuarial sense. This is one of the genuine gaps in the Python insurance ecosystem.

`sklearn.linear_model.Ridge` can implement a credibility-style shrinkage estimator if you construct the penalty matrix correctly, but it does not expose the between-group and within-group variance component estimation that Bühlmann-Straub requires.

### insurance-credibility

[insurance-credibility](https://github.com/burningcost/insurance-credibility) implements Bühlmann-Straub credibility weighting with exposure-proportional variance, iterative variance component estimation via the Bühlmann-Gisler method, and blending of segment-level experience with portfolio prior. It handles the typical UK pricing use case: adjusting segment-level A/E ratios from experience data for use as offsets in the next modelling cycle.

---

## Rating factor smoothing

### Whittaker-Henderson

Rating factor graduation — smoothing raw A/E ratios or SHAP-derived relativities across an ordered rating variable like vehicle age or driver age — is a standard actuarial technique that has no standard Python implementation.

The Whittaker-Henderson smoother (Henderson 1924) minimises a penalised least-squares criterion balancing fit fidelity against smoothness. It is the basis of most actuarial graduation software.

[insurance-whittaker](https://github.com/burningcost/insurance-whittaker) implements it with exposure weighting, customisable penalty order (first or second difference), and credibility blending for thin bands. The standard use case is smoothing postcode-sector relativities before converting to a geodemographic rating factor.

```python
from insurance_whittaker import WhittakerSmoother

smoother = WhittakerSmoother(lambda_=10, order=2)
smoothed = smoother.fit_transform(raw_relativities, weights=exposure_by_band)
```

---

## Model monitoring

### The generic tools

[Evidently](https://github.com/evidentlyai/evidently) (~5,000 GitHub stars, March 2026) is the most complete generic model monitoring library in Python. It covers data drift, target drift, and model performance degradation with a clean HTML report output. If you need a general-purpose drift dashboard, Evidently is solid.

[NannyML](https://github.com/NannyML/nannyml) implements CBPE (Confidence-Based Performance Estimation) for estimating model performance without labels, which is relevant when claims are still developing. Both are generic ML tools that require actuarial adaptation.

### What generic tools miss

A Gini coefficient for a Poisson model weighted by exposure is not something Evidently computes. An A/E ratio by risk segment with credibility intervals is not in NannyML. A double-lift curve comparing two models head-to-head on out-of-time data is not a standard ML monitoring metric.

[insurance-monitoring](https://github.com/burningcost/insurance-monitoring) implements the actuarial monitoring metrics: exposure-weighted Gini, A/E ratio by segment with Poisson confidence intervals, double-lift curves, predicted vs actual by development period, and a lift chart that matches the output format used in Emblem validation reports. These are the metrics a Chief Actuary expects to see in a quarterly model review.

---

## Fairness and proxy discrimination

### The FCA context

The FCA's Consumer Duty and pricing fairness rules prohibit price outcomes that unfairly discriminate against consumers with protected characteristics. The practical regulatory problem is that insurers do not hold sensitive attributes like ethnicity, but their rating factors (postcode, occupation, vehicle type) can act as proxies. Proxy discrimination testing — determining whether a rating structure has a disparate impact attributable to a protected characteristic — is now part of the actuarial model governance process.

### Generic fairness libraries

[Fairlearn](https://fairlearn.org/) is the Microsoft-backed Python fairness library, covering demographic parity, equal opportunity, and similar group fairness metrics. [AIF360](https://aif360.readthedocs.io/) from IBM is broader in scope. Both assume you hold sensitive attributes — they compute disparity metrics by comparing outcomes across protected groups directly.

Neither library addresses the proxy discrimination problem where sensitive attributes are unavailable (the common insurance case) or implements the optimal transport testing method described in the academic literature on insurance fairness (Lindholm et al., 2022; Actuarial Journal, 2024).

### insurance-fairness

[insurance-fairness](https://github.com/burningcost/insurance-fairness) implements the optimal transport proxy discrimination test, Wasserstein distance between rate distributions across demographic groups estimated from census data, conditional independence testing via kernel methods, and a factor contribution analysis showing which rating variables drive the disparity. The output maps directly to FCA Consumer Duty fair value documentation requirements.

```python
from insurance_fairness import ProxyDiscriminationAudit

audit = ProxyDiscriminationAudit(sensitive_attr="derived_ethnicity_index", method="optimal_transport")
result = audit.fit(df, predicted_premium_col="premium", factor_cols=rating_factors)
print(result.wasserstein_distance, result.p_value)
```

---

## Conformal prediction

### The problem with prediction intervals from GLMs

A GLM or GBM gives you a point prediction and, if you work for it, an asymptotic confidence interval from the Fisher information matrix. Neither is a distribution-free coverage guarantee. In practice, GLM prediction intervals for severity models are systematically too narrow in the tail because the distributional assumption is violated.

[MAPIE](https://github.com/scikit-learn-contrib/MAPIE) implements conformal prediction with a scikit-learn API and is the standard starting point. It gives a coverage-guaranteed interval with no distributional assumptions on the residuals.

[insurance-conformal](https://github.com/burningcost/insurance-conformal) extends MAPIE for actuarial use: exchangeability-aware calibration that accounts for temporal non-stationarity in claims data, split conformal sets for frequency-severity paired predictions, and integration with CatBoost models via the Pool interface.

---

## Causal inference and price elasticity

### When you need this

Measuring the causal effect of a price change on lapse or conversion — not the correlation, but the causal effect — requires methods that advertising data science uses routinely but insurance pricing has adopted slowly. The standard problem: you raised rates on one segment and you want to know whether the lapse increase was caused by the rate change or by a coincident book mix shift.

### Standard tools

[DoWhy](https://github.com/py-why/DoWhy) (Microsoft, PyPI) provides a graph-based causal inference framework. [EconML](https://github.com/microsoft/EconML) (also Microsoft) implements double machine learning (DML), causal forests, and instrumental variable estimators. Both are production-grade tools with active development.

For price elasticity in insurance, DML via EconML is the right approach: partial out the confounders with a first-stage model (e.g., CatBoost predicting lapse from non-price factors), then estimate the price effect on the residuals.

[insurance-causal](https://github.com/burningcost/insurance-causal) wraps EconML's DML with insurance-specific defaults: CatBoost first-stage nuisance models, exposure-weighted treatment effects, and a format for outputting price elasticity estimates as rating factors.

---

## Optimisation and portfolio management

Standard convex optimisation in Python is well-served by [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) and [cvxpy](https://www.cvxpy.org/). Both are general tools that pricing actuaries can use directly. For constrained rate change optimisation — maximise profit subject to average rate change, competitive position, and portfolio volume constraints — the problem is generally small enough that scipy's SLSQP solver handles it.

[insurance-optimise](https://github.com/burningcost/insurance-optimise) adds the insurance-specific objective functions: combined ratio minimisation, technical price tracking, Gini-constrained rate moves, and the budget neutrality constraint that requires aggregate premium to remain within a tolerance band. These are standard pricing constraints that require custom objective function code in scipy.

---

## Model governance

### Standard tools

For model risk management documentation, there is no dominant Python tool — most of the regulatory compliance workflow (PRA SS1/23, FCA Consumer Duty model risk) is handled in Word and Excel at UK insurers. Python contributes validation outputs, not governance documentation.

[MLflow](https://mlflow.org/) is the standard tool for experiment tracking, model registration, and artifact storage. If your organisation runs MLflow (common in data platform teams), it gives you an audit trail of model versions, hyperparameters, and metric history. This is the closest thing to automated model governance tooling that is widely deployed.

[insurance-governance](https://github.com/burningcost/insurance-governance) generates PRA SS1/23-formatted validation reports from glum and CatBoost model objects: sensitivity analysis, stress testing, out-of-time performance, factor significance tests, and a model assumptions register. It does not replace human judgement in the model risk process but it automates the evidence assembly that otherwise takes an actuary two weeks.

---

## Data handling

One note on data tooling: use [Polars](https://pola.rs/), not pandas, for insurance data work. The performance gap is material on a UK motor book — Polars processes 500k-policy datasets roughly 5-10x faster than pandas for the groupby and join operations that dominate actuarial data pipelines. Polars is lazy by default, which means you can build complex transformation pipelines and execute them once without intermediate materialisation. The API is cleaner for type-safe operations on mixed-type insurance data.

```python
import polars as pl

df = (
    pl.scan_parquet("motor_claims.parquet")
    .filter(pl.col("accident_year") >= 2020)
    .with_columns([
        (pl.col("paid_amount") / pl.col("exposure")).alias("pure_premium"),
    ])
    .group_by(["vehicle_group", "accident_year"])
    .agg([pl.col("pure_premium").mean(), pl.col("exposure").sum()])
    .collect()
)
```

---

## The realistic picture

The Python insurance pricing ecosystem in 2026 is thin outside of GLMs and gradient boosting. For core modelling — GLM fitting, GBM with SHAP, temporal cross-validation — the tools are mature. For specialised actuarial work — credibility weighting, Whittaker graduation, actuarial monitoring metrics, proxy discrimination testing — the standard data science ecosystem does not cover it and you either write it yourself or use a library purpose-built for the problem.

The honest comparison with Emblem and Radar: Python does not replace them. For rate change governance, collaborative pricing review, and Radar Live deployment, there is no Python equivalent of the Emblem workflow. Python adds value upstream (exploratory modelling, GBM development, causal analysis) and downstream (monitoring, fairness auditing, conformal intervals) — not in the middle, where Emblem is well-established and the switching cost is high.

A sensible stack for a UK personal lines pricing team in 2026: glum and CatBoost for modelling, insurance-cv for validation, shap-relativities for factor extraction, insurance-monitoring for performance tracking, insurance-fairness for Consumer Duty compliance, Emblem/Radar for production rate governance.

---

## Quick reference

| Use case | Standard tool | When to add specialised library |
|---|---|---|
| GLM fitting | glum | insurance-glm-tools for factor tables, IFPs |
| GBM | CatBoost | shap-relativities for factor extraction |
| Severity distribution | scipy.stats | insurance-severity for large loss and ILFs |
| Spliced distributions | — | insurance-distributional |
| Quantile regression | CatBoost (quantile loss) | insurance-quantile for coherent intervals |
| Cross-validation | sklearn | insurance-cv for temporal walk-forward |
| Credibility | sklearn Ridge (approximate) | insurance-credibility for Bühlmann-Straub |
| Rating factor smoothing | — | insurance-whittaker |
| Drift monitoring | Evidently, NannyML | insurance-monitoring for Gini, A/E, double-lift |
| Fairness testing | Fairlearn, AIF360 | insurance-fairness for proxy discrimination |
| Conformal intervals | MAPIE | insurance-conformal for temporal exchangeability |
| Causal / elasticity | DoWhy, EconML | insurance-causal for insurance defaults |
| Optimisation | scipy.optimize, cvxpy | insurance-optimise for pricing constraints |
| Model governance | MLflow | insurance-governance for PRA SS1/23 reports |

All Burning Cost libraries are MIT-licensed, on PyPI, and have Databricks notebooks for validation. Install any of them with `uv pip install <library-name>`.
