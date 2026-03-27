---
layout: post
title: "Renewal Classification — When Risk-Based Pricing Conflicts With Retention Targets"
date: 2026-03-25
categories: [pricing, survival-analysis, causal-inference]
tags: [insurance-survival, insurance-causal, insurance-optimise, PS21/5, retention, python]
description: "How to combine lapse hazard models with causal price elasticity under PS21/5 constraints for UK motor and home renewal pricing."
---

Your risk model says +15%. Your retention model says they'll lapse at +8%. The board wants 92% retention. PS21/5 says you cannot charge the customer more than equivalent new business. These four constraints do not resolve themselves through intuition or spreadsheet negotiation. This post shows how to model the problem properly.

---

## The conflict

At renewal, a pricing actuary faces four simultaneous pressures. The risk model says the technical price should rise — perhaps because frequency deteriorated, or because this customer has had a claim, or because postcode-level loss trends have shifted. The retention model says that above a certain price threshold, a meaningful fraction of customers will leave. The board has set a retention target — 92% is common for UK motor — because below that level the book shrinks faster than it reprices. And PS21/5 (ICOBS 6B.2, effective January 2022) imposes a hard legal ceiling: the renewal price cannot exceed the equivalent new business price (ENBP) through the same channel.

The naive resolution is to apply the technical uplift, check it against ENBP, then discount back to whatever retention target the commercial team demands. This is wrong for at least three reasons.

First, the retention model most teams use is a logistic regression on last year's renewal outcomes. That model has a severe selection bias problem: the customers you observed renewing were those to whom you offered a competitive price. The customers who lapsed were disproportionately those who shopped around. The price sensitivity you estimated is a function of your pricing decisions, not a structural property of customer behaviour. It will underestimate true elasticity for segments where you have historically been cheap, and overestimate it for segments where you have been expensive.

Second, treating the ENBP cap as a post-hoc clip wastes information. If the technical price is already at ENBP, the decision space collapses to a single point. You should know that before running an optimisation, not after.

Third, the retention target is a portfolio-level constraint, not a per-policy one. You do not need to retain every customer — you need to retain the right customers. The value of an inelastic, profitable risk differs from the value of an elastic, unprofitable one. A flat retention target applied to individual policies destroys this distinction.

---

## Step 1: Model lapse propensity with a cure model

The standard approach treats renewal probability as a single-component logistic regression. This misspecifies the population. In any UK motor or home portfolio, a meaningful fraction of customers — typically 35–50% based on our analysis of publicly available cohort data — are structurally loyal. They will renew regardless of small price movements because they have accumulated NCD, because they are not engaged shoppers, or because the friction of switching exceeds their price sensitivity. Modelling these customers as price-sensitive overstates retention risk and leads to unnecessary discounting.

The correct structure is a mixture cure model: a two-component latent class model where one group is "immune" to lapsing (at any price movement in the normal range) and the other is susceptible. `insurance-survival`'s `WeibullMixtureCure` implements this.

```python
from insurance_survival.cure import WeibullMixtureCure

model = WeibullMixtureCure(
    incidence_formula="ncd_years + age + vehicle_age + annual_premium_change_pct",
    latency_formula="ncd_years + age + annual_premium_change_pct",
    n_em_starts=5,
    random_state=42,
)
model.fit(
    df,
    duration_col="tenure_months",
    event_col="lapsed",
)

# P(immune) — the structurally loyal fraction per policy
cure_scores = model.predict_cure_fraction(df)

# P(lapse by month 12) for the susceptible segment
lapse_by_12 = 1.0 - model.predict_population_survival(df, times=[12])[12].values
```

The `result_` attribute gives you the full summary: incidence sub-model coefficients (which factors drive the loyal/susceptible classification) and latency sub-model parameters (which factors govern how quickly susceptibles lapse). In our experience on UK motor, the incidence model is dominated by NCD years and channel (direct versus aggregator), while the latency model is more sensitive to the price change itself.

The key output is `cure_scores`: per-policy P(immune). For a customer in the immune segment, a +10% price increase has a close-to-zero effect on lapse probability. For a customer in the susceptible segment, the latency model gives you the survival curve — lapse probability as a function of time and price.

---

## Step 2: Estimate true price elasticity via Double ML

Even with a cure model, you still need a causal estimate of price elasticity for the susceptible segment. This is where the standard approach falls apart.

The naive approach — regress renewal_flag on log_price_change, controlling for risk factors — is biased. Pricing decisions are correlated with risk factors. Customers who got larger increases last year had worse risk profiles. Customers in price-competitive segments got smaller increases. The coefficient on price_change in an OLS regression conflates the structural price effect with these selection effects.

Double Machine Learning (DML) removes this bias. The mechanism: fit a model of E[renewal | risk_factors] (the outcome nuisance) and a model of E[price_change | risk_factors] (the treatment nuisance). Regress the residuals against each other. The remaining variation in price_change, after removing the component explained by risk factors, is the part that is — conditional on observables — as-good-as-random. The residual regression identifies the structural elasticity.

`insurance-causal`'s `CausalPricingModel` wraps DoubleML's partially linear regression estimator for this:

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(
    outcome="renewed",
    outcome_type="binary",
    treatment=PriceChangeTreatment(column="log_price_change"),
    confounders=["age", "vehicle_age", "postcode_band", "ncd_years", "channel"],
)
model.fit(df_susceptible)  # fit on susceptible segment only
ate = model.average_treatment_effect()
# ate.estimate: e.g. -0.42 (a 1% log price increase reduces renewal prob by 0.42pp)
```

The ATE gives you portfolio-level elasticity. But elasticity is not uniform across the book — it is highest for customers who are on aggregators and have lower NCD, and lowest for direct-channel customers with five or more years NCD. To segment this, use `HeterogeneousElasticityEstimator` from the causal forest subpackage:

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
)

est = HeterogeneousElasticityEstimator(
    binary_outcome=True,
    n_estimators=200,
    min_samples_leaf=20,
    random_state=42,
)
est.fit(
    df_susceptible,
    outcome="renewed",
    treatment="log_price_change",
    confounders=["age", "ncd_years", "vehicle_age", "channel", "postcode_band"],
)

cates = est.cate(df_susceptible)  # per-policy elasticity estimates

# GATES: group average treatment effects by NCD band
gates_by_ncd = est.gate(df_susceptible, by="ncd_years")
```

The GATES output gives you group-level elasticity sorted by NCD band — or by channel, or by vehicle group, depending on what you pass to `by`. This is where the strategic insight lives: the customers with 5+ years NCD on the direct channel have GATE estimates close to zero (they are inelastic). The customers with 0–1 years NCD on an aggregator have the largest magnitude GATE estimates (they are highly elastic). These are distinct segments requiring different pricing strategies.

The `HeterogeneousInference` class adds BLP/GATES/CLAN formal inference — testing whether the heterogeneity you are seeing is statistically real, versus sampling noise:

```python
inf = HeterogeneousInference(n_splits=100, k_groups=5)
result = inf.run(df_susceptible, estimator=est, cate_proxy=cates)
print(result.summary())
# BLP beta_2 > 0 and p < 0.01 confirms heterogeneity is real
```

---

## Step 3: Understand what PS21/5 actually constrains

ICOBS 6B.2 says the renewal price must not exceed the ENBP through the same channel. ENBP is the price you would offer the same customer if they were shopping as a new customer today — factoring in current market pricing, your current NB rating factors, and any promotional adjustments.

Most teams implement this as a clip: after running the renewal rating engine, cap the output at ENBP. This is legally sufficient but strategically suboptimal.

The correct framing: ENBP is the upper bound of the feasible pricing set. For a customer where the technical price is £480 and ENBP is £510, you have a £30 range of feasible prices. For a customer where the technical price is £520 and ENBP is £510, the feasible set collapses to a single price — you must offer £510 regardless of the technical position. In the second case, you are accepting a below-technical price because the regulation forces it. Knowing this upfront — before the optimisation — lets you identify where you are structurally constrained and where you have room to optimise.

Compute the ENBP headroom per policy as part of the data preparation:

```python
df = df.with_columns([
    (pl.col("enbp") - pl.col("technical_price")).alias("enbp_headroom"),
    (pl.col("enbp_headroom") <= 0).alias("enbp_binding"),
])
```

Policies where `enbp_binding` is True have no pricing discretion. The optimisation should exclude them from the decision space — they are priced at ENBP by default.

---

## Step 4: Joint optimisation

With cure scores, CATE estimates, and ENBP headroom, you can build the joint optimisation. The problem is:

- Maximise expected portfolio profit
- Subject to: retention rate >= 92%, ENBP constraint per renewal policy, loss ratio bound

`insurance-optimise`'s `PortfolioOptimiser` handles this directly. The ENBP array is passed as an input; the optimiser treats it as a hard upper bound on the renewal price through the `build_bounds` mechanism.

```python
import numpy as np
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

config = ConstraintConfig(
    lr_max=0.72,
    retention_min=0.92,
    max_rate_change=0.25,
    enbp_buffer=0.00,   # no buffer — price right up to ENBP if warranted
)

opt = PortfolioOptimiser(
    technical_price=df["technical_price"].to_numpy(),
    expected_loss_cost=df["expected_loss_cost"].to_numpy(),
    p_demand=df["renewal_prob_at_current"].to_numpy(),  # from cure model
    elasticity=df["cate_elasticity"].to_numpy(),        # from causal forest
    renewal_flag=df["is_renewal"].to_numpy(),
    enbp=df["enbp"].to_numpy(),
    constraints=config,
)

result = opt.optimise()
print(result)
```

The `p_demand` input takes the renewal probability at the current price from the cure model output: `1 - lapse_by_12`. The `elasticity` input takes the per-policy CATE from the causal forest. The optimiser then solves for the price multiplier that maximises expected profit — `(price - cost) * renewal_prob(price)` — across the portfolio, respecting the 92% retention constraint, the ENBP ceiling, and the loss ratio bound.

The optimiser operates in multiplier space (price / technical_price), with SLSQP and analytical gradients. For a portfolio of 50,000 renewals, a single solve takes a few seconds.

---

## Step 5: Which risks to fight for, which to let go

The optimisation result tells you the optimal price per policy. But the more useful output is the strategic segmentation it implies.

From the GATES analysis, you have estimated elasticity by segment. From the cure model, you have the loyal/susceptible classification. From the ENBP headroom, you know where you are constrained. These three dimensions define four distinct strategic positions:

**Fight for and price aggressively** — susceptible, elastic, ENBP headroom available. These customers will leave if you push the technical price through. The optimiser will find a price below ENBP and below technical where the expected profit is maximised given retention constraint. In practice, these are typically aggregator-channel customers with low NCD who received a large technical uplift.

**Accept at technical price** — immune (high cure score), inelastic, ENBP headroom available. These customers will renew regardless. Price at or near technical — but not above ENBP. The expected profit is maximised at technical price; discounting them is wasteful.

**Constrained by ENBP** — technical price exceeds ENBP. Price at ENBP. No decision to make. The regulatory constraint binds and you take the below-technical margin. Monitor these risks: if the technical price consistently exceeds ENBP, it is a signal that your NB pricing is too aggressive relative to renewal cost.

**Let go profitably** — susceptible, elastic, and unprofitable even at ENBP. These customers will lapse at any price above ENBP, and the margin at ENBP is negative. Do not discount to retain them. The retention constraint should be set to exclude these risks from the denominator.

The CLAN output from `HeterogeneousInference` tells you which features most strongly differentiate the high-elasticity from the low-elasticity group. That is where you should look for the underwriting and market segmentation insight — not in a post-hoc review of the retention model output, but in a statistically validated comparison of covariate distributions across the estimated treatment effect distribution.

---

## Where teams go wrong

The most common failure is treating these three problems — risk pricing, retention modelling, and ENBP compliance — as sequential steps rather than as a joint optimisation. The risk team produces a technical price. The commercial team applies a retention discount. The compliance team clips to ENBP. Each step discards information from the previous one.

The second most common failure is using an observational logistic regression as the retention model. The selection bias in that model is not small — it can easily be large enough to invert the optimal pricing decision for a segment. A customer who looks inelastic in the logistic regression may have always been offered a competitive price; the true structural elasticity, measured causally, may be much higher.

The third failure is implementing the ENBP constraint as a post-hoc clip rather than as a bound in the optimisation. This matters because the shadow price of the ENBP constraint — the marginal value of relaxing it by £1 — tells you how much you are giving up due to the regulatory constraint versus due to the retention constraint. Knowing which constraint is binding is commercially important and is straightforward to extract from the `OptimisationResult` audit trail.

The libraries referenced in this post — [`insurance-survival`](/insurance-monitoring/), [`insurance-causal`](/insurance-causal/), [`insurance-optimise`](/insurance-optimise/) — are pip-installable and designed to be used together. The workflow is not trivial, but neither is the problem it solves.

```bash
uv add insurance-survival insurance-causal insurance-optimise
```

- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) — the `insurance-optimise` demand model with DML elasticity estimation and FCA ENBP compliance checker
- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/) — how to monitor whether the retention model's Gini coefficient has remained stable as the renewal book evolves
