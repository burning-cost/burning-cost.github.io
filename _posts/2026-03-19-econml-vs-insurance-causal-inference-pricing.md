---
layout: post
title: "EconML vs insurance-causal: Causal Inference for Insurance Pricing"
date: 2026-03-19
author: Burning Cost
categories: [causal-inference, libraries, comparisons, pricing]
description: "EconML is the standard Python library for causal ML. It was not built for insurance pricing, Poisson/Gamma exposure models, or the dual-selection bias problems specific to renewal portfolios. Here is what the difference means when you actually try to estimate a causal price effect."
canonical_url: "/2026/03/19/econml-vs-insurance-causal-inference-pricing/"
tags: [EconML, insurance-causal, DML, double-machine-learning, causal-forests, CATE, price-elasticity, Poisson, exposure, renewal, causal-inference, python, uk-insurance]
---

If you search "causal machine learning Python", EconML is the answer. It is maintained by Microsoft Research, has roughly 3,800 GitHub stars, and covers double/debiased machine learning (DML), causal forests, instrumental variables, policy learning, and CATE estimation across a comprehensive API. It is cited in dozens of academic papers and used in production in tech, healthcare, and economics.

For insurance pricing - specifically, estimating causal effects in renewal portfolios with exposure-weighted claim outcomes - it leaves you to solve the hard parts yourself.

This post is a direct comparison. Not to dismiss EconML: it is a remarkable piece of engineering and `insurance-causal` wraps some of its components directly. But there are specific problems in insurance pricing that EconML does not handle, and knowing which problems those are saves a pricing team several weeks of debugging.

```bash
uv add insurance-causal
```

---

## What EconML does

EconML provides estimators for the following causal questions:

- **Average treatment effects (ATE)** via partially linear regression and double/debiased ML
- **Heterogeneous treatment effects (CATE)** via causal forests (`CausalForestDML`), meta-learners (T-, S-, X-, DR-learner), and the linear DML family
- **Policy learning**: finding treatment assignment policies that maximise welfare
- **Instrumental variables**: two-stage least squares and deep IV variants

The API is clean. Estimators follow scikit-learn conventions with `fit`, `effect`, `ate`, and `ate_interval` methods. Documentation is thorough with Jupyter notebooks for each estimator family.

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

est = CausalForestDML(
    model_y=GradientBoostingRegressor(),
    model_t=GradientBoostingClassifier(),
    n_estimators=200,
    honest=True,
    random_state=42,
)
est.fit(Y=y_train, T=t_train, X=X_train, W=W_train)
ate, ate_lower, ate_upper = est.ate_interval(X_test)
cate = est.effect(X_test)
```

For the question "what is the causal effect of a binary treatment on an outcome, controlling for these features?", this works. The academic backing is solid: Chernozhukov et al. (2018) for DML, Wager & Athey (2018) and Athey et al. (2019) for causal forests.

---

## Where EconML stops and the insurance-specific problems begin

### Problem 1: no exposure handling

Every insurance outcome of interest - claim frequency, claim count, loss ratio - is a rate, not a level. A policy in-force for three months contributes a quarter of the exposure of an annual policy. When you estimate the effect of a rating factor or price change on claim frequency, you need to weight observations by earned exposure and use a Poisson (or Tweedie) likelihood, not squared error.

EconML does not have an exposure parameter. Its outcome model defaults to squared error. You can pass a custom `model_y` that accounts for exposure internally, but this requires you to:

1. Know to do this
2. Understand how exposure interacts with the DML residual computation
3. Implement the offset correctly in your nuisance model
4. Verify that the cross-fitting procedure preserves the offset structure

None of this is documented for insurance use cases. It is solvable, but it is a significant amount of non-trivial work - and it is the kind of work that pricing actuaries should not have to do from scratch.

`insurance-causal` handles it directly:

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import BinaryTreatment

model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    treatment=BinaryTreatment(column="new_business_flag"),
    confounders=feature_cols,
    exposure_col="earned_exposure",       # Poisson offset applied automatically
)
model.fit(df)
ate = model.average_treatment_effect()
print(f"ATE: {ate.estimate:.4f} (95% CI: [{ate.ci_lower:.4f}, {ate.ci_upper:.4f}])")
```

The exposure column becomes an offset in the nuisance model for `E[Y|X]`. The DML score is computed in the Poisson log-likelihood space. The coefficient is interpretable as a multiplicative rate effect, which is what a pricing GLM coefficient means. EconML will give you a number if you hand it raw claim counts; that number will be confounded by exposure.

### Problem 2: categorical features without manual encoding

UK motor and home insurance data contains dozens of categorical features: vehicle group (ABI group, often 50+ levels), occupation code, payment method, region, cover type, broker channel. EconML's default nuisance models - gradient boosting from scikit-learn - require these to be pre-encoded. One-hot encoding a 50-level categorical variable in a 250,000-row training set inflates dimensionality and degrades nuisance model accuracy, which directly degrades causal estimate quality in DML.

`insurance-causal` uses CatBoost for all nuisance models. CatBoost handles high-cardinality categoricals natively using ordered target statistics, without manual encoding. This matters for nuisance model fit, and nuisance model fit matters for DML: poor `E[Y|X]` or `E[D|X]` estimation produces residuals that carry confounding, contaminating the causal coefficient.

```python
# EconML: you supply the nuisance models, encoding is your problem
from econml.dml import LinearDML
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

model_y = Pipeline([("enc", OneHotEncoder(handle_unknown="ignore")),
                    ("reg", GradientBoostingRegressor())])
est = LinearDML(model_y=model_y, model_t=model_t)
# 50-level ABI group → 50 dummies → noise in the nuisance model
```

```python
# insurance-causal: CatBoost handles categoricals natively
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    treatment=PriceChangeTreatment(column="price_change_pct", scale="log"),
    confounders=feature_cols,
    # No encoding step. CatBoost uses ordered statistics on the raw levels.
)
```

### Problem 3: the confounding bias report

The single most practically useful output in causal pricing work is not the ATE estimate - it is the comparison between the naive GLM coefficient and the causal estimate. A pricing actuary who has been using a GLM for five years will ask: "how much does confounding actually matter here?" EconML will give you a causal estimate. It will not tell you how different it is from a naive correlation, why, or how sensitive it is to unobserved confounders.

`insurance-causal` produces a confounding bias report after every fit:

```python
report = model.confounding_bias_report(glm_model=fitted_glm)
print(report)
```

```
                        Naive (GLM)   Causal (DML)   Bias
treatment_col
price_change_pct          -0.423        -0.681       -0.258
95% CI (causal)                         [-0.712, -0.650]
Bias as % of naive                       61.0%
Sensitivity bound (Gamma=1.5)            [-0.830, -0.532]
```

The sensitivity bound implements Rosenbaum-style sensitivity analysis: how strong would an unobserved confounder need to be - in terms of association with both treatment and outcome - to overturn the conclusion? A pricing team presenting to a reserving committee or model risk function needs this number. EconML does not compute it.

### Problem 4: dual selection bias

This is the most technically distinctive problem in insurance causal inference and the one most likely to produce badly wrong estimates if ignored.

UK renewal portfolios have two selection layers. First: only policies that renew are observed in the post-renewal period. If your causal estimate uses only renewing policies, it is conditioned on selection - and renewal is correlated with the treatment (premium change) in ways that bias the estimate. Second: claim severity is only observed for policies that both renew and submit a claim. Estimating the causal effect of a risk factor on claim severity requires accounting for both selection stages simultaneously.

EconML has a selection-corrected DML variant, but it handles a single binary selection variable. Treating the two stages as independent problems produces inconsistent estimates.

`insurance-causal` implements `DualSelectionDML`, which follows Dolgikh & Potanin (2025, arXiv:2511.12640). It uses control functions rather than inverse probability weighting for each selection stage:

```python
import numpy as np
from insurance_causal.autodml import DualSelectionDML

# Build arrays from df: Y=claim_severity (NaN for non-selected), D=risk_factor,
# Z=selection indicators (shape n x 2), X=confounders, W_Z=exclusion restrictions
Y = df["claim_severity"].to_numpy()              # NaN where not (renewed & claimed)
D = df["risk_factor"].to_numpy()
Z = df[["renewed", "claimed"]].to_numpy()        # both selection stages
X = df[feature_cols].to_numpy()
W_Z = df[["competitor_quote"]].to_numpy()        # instrument for renewal selection

estimator = DualSelectionDML(estimand="ATES", n_folds=5, random_state=42)
estimator.fit(Y, D, Z, X, W_Z=W_Z)
result = estimator.estimate()
print(result.summary())
# ATES: ate=... se=... 95% CI=[..., ...]  n_selected=.../...
```

This is a problem that does not exist outside insurance in the same form. EconML was not built to solve it.

### Problem 5: the elasticity module

The `insurance-causal.elasticity` subpackage solves a specific deliverable that pricing teams need and EconML does not provide: a causal estimate of price sensitivity that accounts for renewal selection bias, outputs an elasticity surface by rating segment, and feeds directly into a pricing optimiser.

EconML's `CausalForestDML` will estimate heterogeneous treatment effects. It will not:

- Apply the log-log transformation standard in price elasticity estimation
- Correct for renewal selection via the Heckman-style control function approach
- Produce a segment-level elasticity surface grouped by rating factors
- Return the elasticity estimate in the format a pricing optimiser expects

```python
from insurance_causal.elasticity import RenewalElasticityEstimator, ElasticitySurface

est = RenewalElasticityEstimator(
    cate_model="causal_forest",    # CausalForestDML under the hood (via EconML)
    binary_outcome=True,
)
est.fit(df, outcome="renewed", treatment="log_price_ratio", confounders=feature_cols)

# Elasticity surface by segment
surface = ElasticitySurface(est)
summary = surface.segment_summary(df, by=["abi_group", "ncd_years"])
# polars DataFrame: one row per segment, columns: elasticity, ci_lower, ci_upper, n

# Or: portfolio-level ATE with proper log-log interpretation
ate, lb, ub = est.ate()
print(f"ATE: {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")
# ATE: -1.34 → a 10% price increase reduces renewal probability by ~13.4 ppts
```

Note what is happening internally: `RenewalElasticityEstimator` wraps EconML's `CausalForestDML`. The insurance-specific work is in the data transformations, exposure handling, selection correction, and output formatting that happens before and after the EconML call.

---

## The GATES/CLAN/RATE inference layer

Causal forests produce per-observation CATE estimates, but the interesting inference questions are at the group level: which rating segments have significantly different treatment responses? Which segments should be prioritised for intervention? This is what GATES (Group Average Treatment Effects), CLAN (Characteristics of the Affected), and RATE (Rank Average Treatment Effect) tests address.

EconML computes CATE estimates. It does not implement GATES, CLAN, or RATE inference natively. These are documented in Chernozhukov et al. (2018) and Athey & Wager (2021) but are not in the EconML API as of March 2026.

`insurance-causal`'s causal forest module implements all three through the `HeterogeneousInference` class:

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
)

est = HeterogeneousElasticityEstimator(n_estimators=200, catboost_iterations=300)
est.fit(df_train, outcome="renewed", treatment="price_change_pct",
        confounders=feature_cols)

# Compute per-row CATE estimates — the proxy for heterogeneity
cates = est.cate(df_train)

# HeterogeneousInference takes hyperparams, not the estimator
inference = HeterogeneousInference(n_splits=100, k_groups=5)

# Run BLP, GATES, and CLAN in one call
result = inference.run(df_train, estimator=est, cate_proxy=cates)

# GATES: group average treatment effects — are segment-level effects real?
print(result.gates.table)

# CLAN: which covariates distinguish high-effect from low-effect customers?
print(result.clan.table)

# BLP: the formal heterogeneity test — beta_2 > 0 means heterogeneity exists
print(f"BLP beta_2: {result.blp.beta_2:.4f} (p={result.blp.beta_2_pvalue:.4f})")
print(result.summary())
```

RATE (Rank Average Treatment Effect) is the practical heterogeneity test: it tells you whether your CATE estimates are capturing real variation or just noise. A RATE AUTOC close to zero means your causal forest is not identifying meaningful heterogeneity; it is telling you the ATE is sufficient. This is an important negative result - it tells the pricing team not to bother with segment-specific elasticities. EconML will give you a CATE for every observation without helping you determine whether the variation is real.

---

## The Riesz representer approach for continuous treatments

EconML requires you to estimate the generalised propensity score (GPS) when the treatment is continuous - an offer price, a discount percentage, a price index. In renewal portfolios where the treatment is the actual premium charged, the GPS is ill-posed: the price distribution is continuous, concentrated, and shaped by a rating algorithm that is highly correlated with the features. Estimating a density for this process produces numerically unstable residuals.

`insurance-causal`'s `autodml` subpackage uses the Riesz representer approach (Chernozhukov et al. 2022, Econometrica 90(3)) to avoid estimating the GPS entirely:

```python
from insurance_causal.autodml import PremiumElasticity

estimator = PremiumElasticity(
    outcome_family="poisson",   # Poisson outcome with exposure offset
    n_folds=5,
    riesz_type="forest",        # recommended for insurance data
)
# fit() takes arrays: X=confounders, D=treatment, Y=outcome, exposure=exposure
estimator.fit(
    X=df[feature_cols].to_numpy(),
    D=df["offer_price"].to_numpy(),
    Y=df["claim_count"].to_numpy(),
    exposure=df["earned_exposure"].to_numpy(),
)
result = estimator.estimate()
print(result.summary())
# AME: -0.89  se=0.077  95% CI=[-1.04, -0.74]  (Average Marginal Effect)
```

EconML does not implement the Riesz representer approach for DML as of version 0.15. It is in the academic literature; it is not in the package.

---

## Side-by-side comparison

| | EconML | insurance-causal |
|---|---|---|
| Primary use case | General causal ML, any domain | Insurance pricing causal questions |
| Stars / contributors | ~3,800 / 80+ | Small, focused |
| DML | Yes - Linear, NonParametric, Forest variants | Yes - wraps DoubleML with CatBoost nuisance |
| Causal forests | Yes - `CausalForestDML` | Yes - wraps EconML's `CausalForestDML` |
| Exposure/offset handling | No | Yes - Poisson/Gamma throughout |
| Categorical features | Manual encoding required | CatBoost native (no encoding step) |
| Confounding bias report | No | Yes - naive vs causal, sensitivity bounds |
| Dual selection bias | Single-stage only | `DualSelectionDML` for both selection stages |
| GATES / CLAN / RATE | No | Yes - `HeterogeneousInference` |
| Price elasticity surface | No | Yes - `RenewalElasticityEstimator` |
| Riesz representer (continuous treatment) | No | Yes - `PremiumElasticity` in `autodml` subpackage |
| Regulatory framing | None | Confounding bias, sensitivity, audit-ready output |
| Documentation | Excellent - notebooks, papers, API reference | Focused on pricing use cases |
| Academic validation | Extensive | Newer; builds on EconML and DoubleML foundations |

---

## Our view

EconML is a general-purpose causal ML toolkit and it is excellent at what it is designed to do. If your problem is "estimate a causal effect in any domain using DML or causal forests", EconML is the right starting point. Its documentation, community, and academic grounding are vastly ahead of any insurance-specific alternative.

The gap is in the last mile. Insurance pricing causal questions have specific structure: outcomes are rates with exposure offsets, data has high-cardinality categoricals, portfolios have double selection bias, and the practical deliverable is an elasticity surface rather than a generic CATE. EconML does not provide these - not because of a design failure, but because it was not built for this problem.

`insurance-causal` is essentially a translation layer: it takes the econometrics from EconML (and DoubleML), wraps it with insurance-specific data handling, and adds the diagnostics a pricing actuary actually needs. The two can be used together - and in several modules, they are.

If you are building a general causal inference capability on your data science team, start with EconML. If you are a UK pricing team trying to answer "what is the causal price elasticity of our motor book, corrected for renewal selection, broken out by ABI group?" - start with `insurance-causal`.

---

`insurance-causal` is at [github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal). Python 3.10+. Polars-native.

---

**Related posts:**
- [Fairlearn vs insurance-fairness](/2026/03/20/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) - the same pattern in fairness: a general-purpose library and where it stops
- [MAPIE vs insurance-conformal](/2026/03/20/mapie-vs-insurance-conformal-prediction-intervals/) - conformal prediction in insurance and where the standard library breaks

---

**More library comparisons:** How our insurance-specific libraries compare to popular open-source alternatives.

- [Fairlearn vs insurance-fairness](/2026/03/20/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) - proxy discrimination auditing
- [EquiPy vs insurance-fairness](/2026/03/19/equipy-vs-insurance-fairness/) - optimal transport fairness
- [MAPIE vs insurance-conformal](/2026/03/20/mapie-vs-insurance-conformal-prediction-intervals/) - conformal prediction intervals
- [DoWhy vs insurance-causal](/2026/03/18/dowhy-vs-insurance-causal-inference-insurance-pricing/) - causal graphs and refutation
- [Evidently vs insurance-monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) - model monitoring
- [NannyML vs insurance-monitoring](/2026/03/21/nannyml-vs-insurance-monitoring-drift-detection-insurance/) - drift detection
- [Alibi Detect vs insurance-monitoring](/2026/03/18/alibi-detect-vs-insurance-monitoring-drift-detection/) - statistical drift tests
- [sklearn TweedieRegressor vs insurance-distributional](/2026/03/22/sklearn-tweedie-vs-insurance-distributional-regression/) - distributional regression
