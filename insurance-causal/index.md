---
layout: page
title: "insurance-causal"
description: "Causal inference for insurance pricing. Double machine learning removes confounding bias from rating factors. Causal forest for heterogeneous treatment effects."
permalink: /insurance-causal/
schema: SoftwareApplication
github_repo: "https://github.com/burning-cost/insurance-causal"
pypi_package: "insurance-causal"
---

Causal inference for insurance pricing. Double Machine Learning removes confounding bias from observational data. Causal forest estimates segment-level heterogeneous treatment effects for renewal pricing and price elasticity.

**[View on GitHub](https://github.com/burning-cost/insurance-causal)** &middot; **[PyPI](https://pypi.org/project/insurance-causal/)**

---

## The problem

Your GLM coefficients are biased wherever rating variables correlate with distribution channel or policyholder selection. A standard regression tells you the correlation between a price change and renewal rate; it does not tell you whether the price change *caused* the renewal outcome, or whether both are explained by risk level, channel, or broker relationships.

Double Machine Learning removes this confounding bias by partialling out the influence of observed confounders from both the outcome and the treatment before regressing one residual on the other. It does not require a structural model of the selection process — just enough data and flexible ML models for the nuisance functions.

The causal forest extension (v0.4.0) adds heterogeneous treatment effect estimation: instead of an average treatment effect for the whole portfolio, you get a Conditional Average Treatment Effect (CATE) per policy, with formal BLP/GATES/CLAN inference for segment-level aggregates.

---

## Installation

```bash
pip install insurance-causal
```

---

## Quick start

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",
    treatment=PriceChangeTreatment(column="pct_price_change"),
    confounders=["age", "vehicle_age", "postcode_band", "ncb"],
)
model.fit(df)
ate = model.average_treatment_effect()
print(ate)  # AverageTreatmentEffect(coef=-0.42, ci=(-0.51, -0.33))
```

Causal forest for segment-level price response:

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
    make_hte_renewal_data,
)

df = make_hte_renewal_data(n=10_000)
est = HeterogeneousElasticityEstimator(n_estimators=200)
est.fit(df, confounders=["age", "ncd_years", "channel"])
cates = est.cate(df)   # per-policy treatment effects

inference = HeterogeneousInference(est)
gates = inference.gates(df, n_groups=5)   # GATES aggregates by CATE quantile
clan = inference.clan(df, covariate="age")  # CLAN: who is most elastic?
```

---

## Key API reference

### `CausalPricingModel`

Portfolio-level average treatment effect via Double Machine Learning (DoubleML).

```python
CausalPricingModel(
    outcome,        # str, outcome column name
    outcome_type,   # "binary" | "continuous"
    treatment,      # Treatment object
    confounders,    # list of confounder column names
    n_folds=5,      # cross-fitting folds
)
```

- `.fit(df)` — fits nuisance models and estimates treatment effect
- `.average_treatment_effect()` — returns `AverageTreatmentEffect` with `.coef`, `.ci`, `.p_value`
- `.confounding_report()` — DataFrame showing confounding bias before/after DML

### `insurance_causal.causal_forest` subpackage

**`HeterogeneousElasticityEstimator`** — CausalForestDML with formal BLP/GATES/CLAN inference.

- `.fit(df, confounders)` — fits the causal forest
- `.cate(df)` — Polars Series of per-policy CATEs
- `.ate()` — average treatment effect with confidence interval

**`HeterogeneousInference`** — formal inference on the causal forest output.

- `.gates(df, n_groups)` — GATES: group average treatment effects by CATE quintile
- `.clan(df, covariate)` — CLAN: most/least affected group characteristics
- `.blp(df)` — Best Linear Predictor test for treatment effect heterogeneity
- `.rate(df)` — RATE/AUTOC/QINI targeting evaluation

**`TargetingEvaluator`** — evaluates whether the estimated CATEs are useful for targeting.

### `insurance_causal.elasticity` subpackage

```python
from insurance_causal.elasticity import RenewalElasticityEstimator

est = RenewalElasticityEstimator()
est.fit(df, confounders=confounders)
ate, lb, ub = est.ate()
```

### `insurance_causal.autodml` subpackage

Automatic Debiased ML via Riesz Representers — avoids GPS estimation for continuous treatments.

```python
from insurance_causal.autodml import PremiumElasticity

model = PremiumElasticity(outcome_family="poisson", n_folds=5)
model.fit(X, D, Y)
result = model.estimate()
# result.elasticity, result.ci, result.dose_response_curve
```

---

## Benchmark

Tested on synthetic insurance data with nonlinear confounding (n=50,000 policies). DML removes nonlinear confounding bias at portfolio scale. Naively estimated OLS treatment effects showed 15–25% bias; DML estimates were within confidence interval of the true effect in all simulations. At small n (< 2,000 per group), honest causal forest can over-partial — use DML for portfolio-level effects, causal forest for segment-level CATEs with n ≥ 2,000 per group.

---

## Related blog posts

- [Causal Price Elasticity for UK Renewal Pricing](https://burning-cost.github.io/2026/03/14/causal-price-elasticity-for-uk-renewal-pricing/)
- [insurance-causal vs EconML](https://burning-cost.github.io/2026/03/19/econml-vs-insurance-causal-inference-pricing/)
- [insurance-causal vs DoWhy](https://burning-cost.github.io/2026/03/18/dowhy-vs-insurance-causal-inference-insurance-pricing/)
- [Rate Change Lapse Evaluation via Causal Inference](https://burning-cost.github.io/2026/03/10/rate-change-lapse-evaluation-causal-inference/)
- [Causal Price Elasticity Tutorial](https://burning-cost.github.io/2026/03/15/causal-price-elasticity-tutorial/)
