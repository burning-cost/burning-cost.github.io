---
layout: post
title: "Causal Price Elasticity for UK Motor Renewal: CausalForestDML with an ENBP-Constrained Optimiser"
date: 2026-03-14
categories: [libraries, pricing, elasticity]
tags: [price-elasticity, causal-inference, DML, causal-forest, CATE, ENBP, FCA, PS21-5, renewal, motor, python, catboost, econml, insurance-elasticity]
description: "OLS renewal elasticity is biased 20-80% on formula-rated UK motor books. insurance-elasticity wraps CausalForestDML with an ENBP-constrained profit optimiser."
---

*This post covers `insurance-elasticity`, which is specifically about causal renewal price elasticity and FCA PS21/5-compliant pricing optimisation. For a more general treatment of confounding in GLM rating factors (age, telematics, vehicle group), see [When exp(beta) Lies](/2026/03/05/your-rating-factor-might-be-confounded/). For DML applied to conversion and demand curves more broadly, see [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/).*

---

The naive elasticity estimate for a UK motor book is wrong. Not directionally wrong, usually, but quantitatively wrong in ways that distort renewal pricing strategy. On synthetic benchmarks with a known data-generating process, OLS elasticity is biased by 20–80% relative to the true causal effect, depending on how formula-rated the book is. CausalForestDML reduces that to 1–10%.

The reason OLS fails is structural. In a formula-rated book, the renewal price is a near-deterministic function of the observable risk factors: age, vehicle, postcode, NCD, claims history. High-risk customers get higher increases. High-risk customers also lapse at higher rates for reasons that have nothing to do with price — their risk profile has deteriorated, their alternatives are limited, their prior insurer relationship is weaker. A logistic regression of renewal flag on price change picks up both the causal price effect and the risk-correlated lapse propensity. It cannot separate them.

Double Machine Learning separates them. Partial out the confounders from both the outcome (renewal flag) and the treatment (log price change). What remains in the residualised treatment is the exogenous variation — price changes that were not mechanically driven by the risk profile. Regress the residualised renewal indicator on residualised price change. The coefficient is the causal semi-elasticity.

[`insurance-elasticity`](https://github.com/burning-cost/insurance-elasticity) wraps EconML's `CausalForestDML` and `LinearDML` to do exactly this, with UK insurance defaults and an ENBP-constrained pricing optimiser built in.

```bash
uv add "insurance-elasticity[all]"
```

---

## The near-deterministic price problem

Before fitting anything, run the treatment variation diagnostic:

```python
from insurance_elasticity.diagnostics import ElasticityDiagnostics

diag = ElasticityDiagnostics()
report = diag.treatment_variation_report(
    df,
    treatment="log_price_change",
    confounders=["age", "ncd_years", "vehicle_group", "region", "channel"],
)
print(report.summary())
```

The diagnostic checks `Var(D̃) / Var(D)` — the fraction of price variation that survives after conditioning on the observed confounders. When that ratio falls below 10%, less than a tenth of price variation is exogenous. DML has almost nothing to work with. The confidence intervals blow up and the point estimate is noise.

In a heavily formula-rated book this is a real constraint. If `report.weak_treatment` is True, the report's suggestions cover the main remedies: A/B price test data, panel data with within-customer variation over multiple renewal cycles, quasi-experiments from bulk re-rate events, and the FCA PS21/5 regression discontinuity (customers at the ENBP boundary receive a mechanically constrained price regardless of their risk profile, which provides genuinely exogenous variation).

Most teams skip this check, fit the elasticity model, get a coefficient with a confidence interval that spans zero and two standard errors, and conclude that price has no effect. The conclusion is wrong. The data does not contain enough exogenous variation to estimate the effect — which is a different finding, and an important one.

---

## Fitting the elasticity model

```python
from insurance_elasticity.fit import RenewalElasticityEstimator

est = RenewalElasticityEstimator(
    cate_model="causal_forest",   # non-parametric CATE surface
    n_estimators=200,
    catboost_iterations=500,
    n_folds=5,
)
est.fit(
    df,
    outcome="renewed",
    treatment="log_price_change",
    confounders=["age", "ncd_years", "vehicle_group", "region", "channel"],
)

ate, lb, ub = est.ate()
print(f"ATE: {ate:.3f}  95% CI: [{lb:.3f}, {ub:.3f}]")
```

The treatment variable is `log(offer_price / last_year_price)`. This gives a semi-elasticity directly: a 1-unit change in treatment (100% price increase) changes renewal probability by `ATE` percentage points. For the typical 5–20% re-rates in UK personal lines, interpret as: a 10% increase changes renewal probability by approximately `ATE × log(1.1) ≈ ATE × 0.095`.

CatBoost handles the nuisance models for both `E[Y|X]` and `E[D|X]` because UK motor data is full of high-cardinality categoricals — region, vehicle group, occupation class, payment method — that CatBoost handles natively without label encoding. The alternative is one-hot encoding and gradient boosting, which works but requires more care.

`CausalForestDML` is the right choice for the elasticity surface because it makes no assumptions about how heterogeneity enters the treatment effect. The causal forest estimates τ(x) — the individual-level treatment effect — by splitting on features that predict treatment effect heterogeneity rather than outcome level. It provides valid pointwise confidence intervals via honest splitting (separate data used for splitting and for leaf estimation). `LinearDML` is faster but requires pre-specifying the heterogeneity structure; use it for quick portfolio-level ATE estimates where the CATE surface is not the goal.

---

## Heterogeneous elasticity: GATE by segment

The average treatment effect is the wrong thing to target for renewal pricing. Different customers have materially different price sensitivity.

On the synthetic benchmark DGP — calibrated to mimic a UK motor book — the true elasticities range from −3.5 for customers with no NCD to −1.0 for customers with maximum NCD. PCW (price comparison website) customers are 30% more elastic than direct customers at the same NCD level. A renewal pricing strategy that applies the ATE uniformly to all segments is leaving money on the table on the low-elasticity segments and overpricing on the high-elasticity segments.

```python
gate = est.gate(df, by="ncd_years")
print(gate)
```

The GATE (group average treatment effect) gives segment-level causal elasticity estimates with confidence intervals. `CausalForestDML` achieves 30–60% lower RMSE on NCD GATE estimates relative to OLS on the benchmark, because OLS conflates the risk-correlated lapse component with the price effect and averages it uniformly across segments where the correlation is strongest.

The elasticity surface visualises CATE across two dimensions simultaneously:

```python
from insurance_elasticity.surface import ElasticitySurface

surface = ElasticitySurface(est)
fig = surface.plot_surface(df, dims=["ncd_years", "age_band"])
fig.savefig("elasticity_surface.png", dpi=150)
```

---

## FCA PS21/5 and the ENBP-constrained optimiser

Since January 2022, UK GI firms must not quote a renewing customer a price above the equivalent new business price (ENBP). FCA PS21/5 (GIPP) imposes this as a hard per-policy constraint. A renewal pricing optimiser that ignores it is not a compliant pricing strategy.

`RenewalPricingOptimiser` maximises expected profit subject to the ENBP constraint as a hard per-policy bound, not a soft average:

```python
from insurance_elasticity.optimise import RenewalPricingOptimiser

opt = RenewalPricingOptimiser(
    est,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,   # minimum price = technical premium
)
priced_df = opt.optimise(df, objective="profit")

# Per-policy FCA ICOBS 6B.2 compliance flag
audit = opt.enbp_audit(priced_df)
print(f"Breaches: {(audit['compliant'] == False).sum()} / {len(audit)}")
```

The optimisation rule is: for each policy, offer the price that maximises `expected_margin × τ̂(x)` subject to `offer_price ≤ enbp`. Where τ̂(x) is small (low-elasticity customers: maximum NCD, long tenure), the optimiser prices closer to the technical premium floor. Where τ̂(x) is large (high-elasticity customers: no NCD, PCW, first renewal), the optimiser offers a larger discount because the marginal cost of retention is lower than the foregone margin.

The `enbp_audit()` output is a per-row DataFrame with `compliant` flag, `offer_price`, and `enbp_limit`. This is the output you hand to compliance. The `to_markdown()` method formats it for committee packs.

---

## Portfolio demand curve

The demand curve sweeps a range of portfolio-wide price changes and estimates renewal rate and expected profit at each point:

```python
from insurance_elasticity.demand import demand_curve

demand_df = demand_curve(est, df, price_range=(-0.25, 0.25, 50))
```

This is the efficient frontier: how retention rate and expected profit trade off across price scenarios. The curve is derived from the causal elasticity estimates, not from naive OLS. Where the OLS elasticity is biased upward (overstating price sensitivity), the OLS demand curve will show a steeper drop-off in retention as prices rise — making the commercial case for larger discounts look stronger than it is. The causal demand curve is more accurate.

---

## What this is not

`insurance-elasticity` is specifically about renewal price elasticity and FCA PS21/5-compliant pricing optimisation. It is not a general causal inference framework. For causal analysis of other rating factors — telematics scores, vehicle group relativities, age effects — see [`insurance-causal`](https://github.com/burning-cost/insurance-causal), which handles arbitrary treatment types and outcome distributions without the renewal-specific defaults.

The OLS bias benchmark (20–80% relative to true ATE) comes from the synthetic UK motor DGP in `notebooks/benchmark.py`, where the true elasticities are known. On your data the bias could be smaller or larger. Run the treatment variation diagnostic first. If exogenous variation is genuinely present in the data, the DML estimate will be credible. If it is not, no method can recover the true elasticity from that data.

```bash
uv add "insurance-elasticity[all]"
```

Source and documentation at [github.com/burning-cost/insurance-elasticity](https://github.com/burning-cost/insurance-elasticity).

---

**Related articles from Burning Cost:**
- [When exp(beta) Lies: Confounding in GLM Rating Factors](/2026/03/05/your-rating-factor-might-be-confounded/) — DML for general rating factor confounding, not renewal-specific
- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) — demand curve and conversion modelling with `insurance-optimise`
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) — when confounded factors load onto protected characteristics
