---
layout: post
title: "OLS Elasticity in a Formula-Rated Book Measures the Wrong Thing"
date: 2026-03-14
categories: [libraries, pricing, causal-inference]
tags: [elasticity, causal-forest, DML, double-machine-learning, FCA, PS21-5, ENBP, renewal, CausalForestDML, insurance-elasticity, python, motor, econml, catboost]
description: "CausalForestDML separates causal price effect from risk-lapse correlation in UK motor renewal. insurance-elasticity - per-customer CATE and ENBP optimiser."
---
> **Update (March 2026):** `insurance-elasticity` has been merged into [`insurance-causal`](https://github.com/burning-cost/insurance-causal) ([PyPI](https://pypi.org/project/insurance-causal/)). Install `insurance-causal` and use `insurance_causal.elasticity` for the same functionality.

> **Update (March 2026):** `insurance-demand` has been merged into [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise). Install `insurance-optimise` and use `insurance_optimise.demand` for the same functionality.

A UK motor renewal book has a structural problem that corrupts every naive price elasticity estimate. Your pricing model re-rates every customer using risk factors. Those same risk factors drive the renewal decision independently of price - higher-risk customers are harder to retain for reasons that have nothing to do with what you charge them. When you regress renewal indicator on log price change, you are picking up both effects simultaneously and cannot separate them from within the regression.

The result is a biased elasticity. Not slightly biased - in our benchmarks on synthetic UK motor data with a realistic data-generating process, OLS relative bias against the true elasticity is 20–80%. A renewal optimiser built on that number is optimising on a false premise.

[`insurance-elasticity`](/insurance-causal/) is our causal price elasticity library: CausalForestDML and LinearDML for heterogeneous semi-elasticity estimation, a diagnostic that catches the near-deterministic price problem before you fit, a full elasticity surface across two portfolio dimensions, and a profit-maximising ENBP-constrained optimiser for FCA PS21/5 compliance.

```bash
uv add "insurance-causal[all]"
```

---

## How the confounding works in a formula-rated book

When your pricing model generates a renewal offer, the price change is largely a deterministic function of the risk factors in X. Any customer whose risk factors have moved - new claim, birthday, moved postcode - receives a price change driven by that movement. The renewal decision is also driven by those same risk factors: a customer who has had a claim in the last 12 months may lapse for many reasons beyond the premium increase.

The formal structure: D (log price change) is not randomly assigned. It is approximately m₀(X) - a deterministic function of rating factors. When D ≈ m₀(X), the residualised treatment D̃ = D - E[D|X] has near-zero variance. OLS on the raw data sees the correlation between price changes and lapse but cannot identify whether it is the price causing the lapse or the risk movement that caused both.

Double Machine Learning residualises both D and Y on X separately:

1. Fit E[Y|X] on renewal outcomes using CatBoost, compute Ỹ.
2. Fit E[D|X] on log price changes using CatBoost, compute D̃.
3. Regress Ỹ on D̃. The coefficient is the causal semi-elasticity.

The Neyman-orthogonal score means errors in steps 1 and 2 produce second-order bias in the elasticity estimate, not first-order. You get a valid confidence interval. The residualised D̃ is the genuinely exogenous variation in price - changes that were not mechanically driven by the pricing formula.

The benchmark result on synthetic data: OLS relative bias of 20–80% against the true elasticity. LinearDML and CausalForestDML both reduce this to 1–10%.

---

## The near-deterministic price problem: check before you fit

The diagnostic that distinguishes `insurance-elasticity` from a generic DML wrapper is `ElasticityDiagnostics`. When Var(D̃) / Var(D) < 10% - less than 10% of price variation is exogenous after conditioning on X - you have near-deterministic treatment. The confidence intervals blow up and the point estimate is noise. You need to know this before fitting, not after.

```python
from insurance_causal.elasticity.data import make_renewal_data
from insurance_causal.elasticity.diagnostics import ElasticityDiagnostics

df = make_renewal_data(n=50_000)

diag = ElasticityDiagnostics()
report = diag.treatment_variation_report(
    df,
    treatment="log_price_change",
    confounders=["age", "ncd_years", "vehicle_group", "region", "channel"],
)
print(report.summary())
# treatment_r2: 0.91  (91% of price variation explained by rating factors)
# residual_var_ratio: 0.09  (9% is exogenous)
# weak_treatment: True
# Suggestion: Consider A/B price test data or panel variation.
```

If `weak_treatment` is True, do not proceed to fitting without addressing it. The suggestions cover the main remedies: A/B price test data, within-customer panel variation from policy anniversaries, quasi-experiments from bulk re-rates, and the PS21/5 regression discontinuity at the ENBP boundary.

This diagnostic is not optional on UK motor data. Formula-rated books routinely have treatment R² above 0.85.

---

## Fitting the elasticity model

```python
from insurance_causal.elasticity.fit import RenewalElasticityEstimator

confounders = ["age", "ncd_years", "vehicle_group", "region", "channel"]

est = RenewalElasticityEstimator(
    cate_model="causal_forest",   # non-parametric CATE surface
    n_estimators=200,
    catboost_iterations=500,
    n_folds=5,
)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)

# Average treatment effect
ate, lb, ub = est.ate()
print(f"ATE: {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")
```

The ATE is the average semi-elasticity: a 1-unit increase in log price change (approximately a 100% price increase) changes renewal probability by `ATE` percentage points. For the typical 5–20% renewal re-rates in UK personal lines, the practical interpretation is: a 10% price increase changes renewal probability by approximately ATE × log(1.1) ≈ ATE × 0.095.

**Why CatBoost for the nuisance models.** UK insurance data is full of high-cardinality categoricals - region, vehicle group, occupation, payment method. CatBoost handles them natively via ordered target encoding, without the one-hot explosion that penalised regression requires. A 2024 systematic evaluation (preprint, arXiv:2403.14385) found gradient boosted trees outperform LASSO in the DML nuisance step when confounding is nonlinear, which it always is with postcode and age-vehicle interactions.

**LinearDML vs CausalForestDML.** LinearDML assumes constant elasticity (or heterogeneity only through explicitly interacted features). It is 30–60× faster than CausalForestDML and the right choice for quick portfolio-level ATE estimation or benchmarking. CausalForestDML is non-parametric and provides valid pointwise confidence intervals via honest splitting - the right choice when you want the heterogeneous elasticity surface and segment-level GATE estimates.

---

## Heterogeneous elasticity: CATE and GATE

The average elasticity is rarely the right number to put into a renewal optimiser. Customers differ. A PCW customer in their first year with maximum renewal competition has a different elasticity than a 15-year loyal customer on a direct channel. Treating them identically means you are over-discounting the loyal customer and under-discounting the switcher.

`CausalForestDML` provides per-customer CATE. `gate()` aggregates these into segment-level estimates with confidence intervals:

```python
gate = est.gate(df, by="ncd_years")
print(gate)
#   ncd_years  gate   ci_lower  ci_upper
#   0          -3.41  -3.82     -3.00
#   1          -2.88  -3.21     -2.55
#   2          -2.31  -2.59     -2.03
#   3          -1.89  -2.12     -1.66
#   5+         -1.02  -1.19     -0.85
```

The pattern here - elasticity declining in magnitude with NCD - is structurally present in UK motor data. Customers with more years’ NCD face greater hassle cost when switching: they must obtain an NCD proof letter from their current insurer and present it to the new insurer, with risk of delays in NCD verification that can affect the quoted price. There is also loyalty inertia — customers who have accumulated significant NCD history are often longer-tenured and less habitually price-sensitive. These friction effects mean high-NCD customers are correspondingly less price-elastic, not because NCD is lost on switching (it is not: UK motor insurers accept NCD proof letters), but because the switching process involves more steps. A renewal optimiser that uses the pooled ATE is systematically over-discounting the high-NCD segment and under-retaining the low-NCD segment.

The elasticity surface shows two dimensions simultaneously:

```python
from insurance_causal.elasticity.surface import ElasticitySurface

surface = ElasticitySurface(est)
fig = surface.plot_surface(df, dims=["ncd_years", "age_band"])
fig.savefig("elasticity_surface.png", dpi=150, bbox_inches="tight")
```

The synthetic benchmark data uses a DGP with elasticities ranging from -3.5 (no-NCD customers) to -1.0 (maximum NCD), and -3.0 (ages 17-24) to -1.2 (ages 65+), with PCW customers 30% more elastic. CausalForestDML recovers this structure. OLS cannot.

---

## FCA PS21/5: the ENBP constraint

Since January 2022, UK GI firms cannot quote a renewing customer a price above the equivalent new business price through the same channel (FCA PS21/5, ICOBS 6B.2). The constraint is per-policy and per-channel: you cannot average across the book.

`RenewalPricingOptimiser` builds the ENBP constraint directly into the profit-maximising optimisation:

```python
from insurance_causal.elasticity.optimise import RenewalPricingOptimiser

opt = RenewalPricingOptimiser(
    est,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,    # minimum: technical premium (no subsidy)
)
priced_df = opt.optimise(df, objective="profit")

# Compliance audit
audit = opt.enbp_audit(priced_df)
print(f"Breaches: {(audit['compliant'] == False).sum()} / {len(audit)}")
```

The optimiser maximises expected profit subject to three hard constraints per policy: offer price ≤ ENBP (FCA PS21/5), offer price ≥ floor loading × technical premium (no subsidy), and the renewal probability response implied by the fitted CATE. The `enbp_audit()` method returns a per-row compliance flag for reporting to the compliance function. That flag is what you attach to a regulatory query — and the [insurance-governance model inventory](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) is the right place to record which version of the elasticity model produced the compliance evidence.

---

## Portfolio demand curve

Once the elasticity model is in production, [insurance-monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) lets you track whether the estimated renewal response is holding — the mSPRT sequential test handles exactly the monthly-checking problem that renewal champion/challenger experiments create.


The demand curve shows the trade-off between renewal rate and expected profit across a range of uniform price changes. It is the first thing a pricing committee asks for when evaluating an elasticity model:

```python
from insurance_causal.elasticity.demand import demand_curve

demand_df = demand_curve(est, df, price_range=(-0.25, 0.25, 50))
# DataFrame: price_change, predicted_renewal_rate, expected_profit
```

The efficient frontier - renewal rate vs expected profit across the ENBP-constrained optimised pricing grid - is in the worked example at `price_elasticity_optimisation.py` in the examples repository. Running that on your own data before interpreting results is strongly recommended: it shows how each component behaves and where the ENBP constraint bites.

---

## How this fits with insurance-causal and insurance-demand

Three libraries touch elasticity in this stack. They are not redundant.

[`insurance-causal`](/insurance-causal/) is the general causal inference library: DML for any treatment and outcome in insurance, with the confounding bias report, DAG validation, sensitivity analysis, and CATE by arbitrary segment. The right tool when your question is "how much of this GLM coefficient is actually causal?" across a broad range of factors.

[`insurance-demand`](https://github.com/burning-cost/insurance-optimise) is the demand modelling library: conversion, retention, demand curve construction, and portfolio-level price sensitivity. The right tool when you want the full conversion + renewal demand model and demand curve.

[`insurance-elasticity`](https://github.com/burning-cost/insurance-causal) is the causal price elasticity specialist: CausalForestDML for heterogeneous semi-elasticity, the near-deterministic price diagnostic, the elasticity surface, and the ENBP-constrained optimiser. The right tool when you specifically need a defensible, per-customer elasticity estimate for renewal pricing decisions and FCA compliance documentation.

If your question is renewal pricing for a UK personal lines book under PS21/5, start here.

---

## Performance

Benchmarked on synthetic UK motor renewal data (50,000 policies, known DGP, 70/15/15 train/cal/test split):

| Metric | OLS Naive | LinearDML | CausalForestDML |
|---|---|---|---|
| ATE relative bias vs truth | 20%–80% | 1%–10% | 1%–10% |
| NCD GATE RMSE | Baseline | N/A | 30%–60% better |
| 95% CI covers true ATE | N/A | Yes | Yes |
| Fit time relative to OLS | 1× | 30×–60× | 100×–300× |

The GATE RMSE improvement is largest on books where the renewal population has strong segment-level price sensitivity heterogeneity. The fit-time cost of CausalForestDML is real: on 50,000 policies with 5-fold cross-fitting, plan for 20-40 minutes on a standard CPU. For exploratory work, use `cate_model="linear"` and drop to `n_folds=3`.

---

## References

- Chernozhukov et al. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1).
- Athey & Wager (2019). Estimating treatment effects with causal forests. *Annals of Statistics*, 47(2).
- Guelman & Guillén (2014). A causal inference approach to measure price elasticity in automobile insurance. *Expert Systems with Applications*, 41(2).
- FCA PS21/5 (2021). General Insurance Pricing Practices Policy Statement.

---

**Related reading:**
- [Continuous Treatment Causal Inference for Insurance Pricing](/2026/03/12/insurance-autodml/) - the `insurance-causal` library for automatic nuisance model selection and confounding bias reporting
- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) - the insurance-optimise demand model with FCA GIPP compliance
- [Constrained Rate Optimisation and the Efficient Frontier](/2026/02/21/constrained-rate-optimisation-efficient-frontier/) - the rate change optimiser that consumes elasticity estimates
