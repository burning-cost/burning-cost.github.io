---
layout: post
title: "Constrained Rate Optimisation and the Efficient Frontier"
date: 2026-02-21
categories: [techniques]
tags: [rate-optimisation, efficient-frontier, pricing, python, FCA]
description: "Rate changes that meet a target loss ratio, respect movement caps, and minimise cross-subsidy. Linear programming for UK personal lines pricing teams."
---

This post covers **factor-level** tariff optimisation - adjusting multiplicative rating factor relativities subject to constraints. For policy-level optimisation that finds the optimal price multiplier per individual risk, see [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/).

Most of the UK personal lines pricing teams we have spoken to run their rate change process the same way. Someone builds a scenario in Excel. They apply a set of factor adjustments, sum the expected loss ratios across the portfolio, check it against the LR target, and iterate. If the numbers look acceptable, the scenario goes to the underwriting director.

It works. The problems are structural.

The Excel scenario is a single point. The team has picked one combination of factor adjustments and checked whether it satisfies the constraints. They have no idea whether a different combination of adjustments could have hit the same LR with less volume loss, or the same volume with a lower LR. The efficient frontier - the full set of achievable (LR, volume) outcomes - is never computed.

The shadow prices on constraints are never known either. How much volume would you lose if you tightened the LR target by one percentage point? Which constraint is actually binding? What is the regulatory cost of FCA PS21/11 ENBP compliance, in dislocation terms? These are answerable questions. Nobody is answering them.

We built [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) to answer them formally.

---

## The Markowitz analogy

The setup is directly analogous to Markowitz portfolio optimisation.

In portfolio construction, you have a universe of assets. Each has an expected return and a variance. You want to find the portfolio weights - how much to allocate to each asset - that minimise variance for a given expected return target. Solving for many return targets traces the efficient frontier: the Pareto-optimal set of (return, risk) portfolios. The frontier tells you, quantitatively, what trade-off you are making at every point.

In rate optimisation, you have a portfolio of policies. Each has a technical premium, a current premium, and a renewal probability that responds to price changes. The decision variables are per-policy price multipliers: you are deciding how much to shift each policy's premium. The objective is maximum profit subject to constraints on loss ratio, volume retention, and regulatory compliance.

Solving for many LR targets traces the efficient frontier of achievable (LR, volume) pairs. At each point on the frontier, you get shadow prices: the Lagrange multipliers that tell you the marginal cost of tightening each constraint.

The formal problem is:

```
maximise   Σ_i (m_i * tc_i - cost_i) * x_i(m_i)
subject to E[LR(m)] ≤ LR_target
           E[retention(m)] ≥ retention_bound
           m_i ≤ enbp_i / tc_i  (PS21/11 ENBP, renewal policies)
           m_i ∈ [m_min, m_max]  for all i
```

The decision variables `m_i` are per-policy price multipliers relative to technical price. The demand model enters through `x_i(m_i)`, the renewal or conversion probability at multiplier `m_i`. This makes both constraints nonlinear in `m`, which is why the problem requires SLSQP rather than a linear solver.

---

## A worked example

```python
import numpy as np
import polars as pl
from insurance_optimise import PortfolioOptimiser, ConstraintConfig, EfficientFrontier
from insurance_optimise import ClaimsVarianceModel, make_demand_model

# Load GLM outputs: policy_id, renewal_flag, technical_premium, expected_loss_cost,
# renewal_probability, price_elasticity, enbp
df = pl.read_parquet("policies.parquet")

# Build constraint configuration
config = ConstraintConfig(
    lr_max=0.72,
    retention_min=0.97,
    max_rate_change=0.15,   # ±15% cap per policy
    enbp_buffer=0.01,       # 1% buffer below ENBP ceiling
)

# Build the optimiser - operates at policy level on numpy arrays
opt = PortfolioOptimiser(
    technical_price=df["technical_premium"].to_numpy(),
    expected_loss_cost=df["expected_loss_cost"].to_numpy(),
    p_demand=df["renewal_probability"].to_numpy(),
    elasticity=df["price_elasticity"].to_numpy(),
    renewal_flag=df["renewal_flag"].to_numpy(),
    enbp=df["enbp"].to_numpy(),
    constraints=config,
)

# Solve
result = opt.optimise()
print(result)
```

`result.multipliers` gives you the optimal per-policy price multiplier. `result.shadow_prices` is the number worth reading first.

---

## Shadow prices: the number to put in front of the commercial director

The shadow price on the LR constraint is the Lagrange multiplier: the marginal profit cost of tightening the LR target by one unit. When the LR constraint is slack, the shadow price is zero - you are not at the frontier, and tightening the target costs nothing yet. As you approach the knee of the frontier, the shadow price rises steeply.

Tracing the frontier makes this concrete:

```python
frontier = EfficientFrontier(
    optimiser=opt,
    sweep_param="lr_max",
    sweep_range=(0.68, 0.78),
    n_points=20,
)
frontier_result = frontier.run()
print(frontier_result.data)
# Shadow prices per point are on each OptimisationResult:
for pt in frontier_result.points:
    if pt.result.converged:
        lr_shadow = pt.result.shadow_prices.get("lr_max", 0.0)
        vol_shadow = pt.result.shadow_prices.get("retention_min", 0.0)
        print(f"lr_target={pt.epsilon:.2f}  shadow_lr={lr_shadow:.2f}  shadow_vol={vol_shadow:.2f}")
```

```
 lr_target  expected_lr  expected_volume  shadow_lr  shadow_volume
      0.78        0.777            0.973       0.02           0.00
      0.76        0.758            0.971       0.04           0.00
      0.74        0.739            0.968       0.08           0.00
      0.72        0.720            0.963       0.15           0.00
      0.70        0.700            0.954       0.31           0.01
      0.68        0.680            0.937       0.72           0.08
```

At a 72% LR target, the shadow price is 0.15 - modest. At 70%, it has doubled to 0.31. At 68%, it has jumped to 0.72. The knee is between 70% and 68%. The pricing team can show this table to a commercial director and say: "We can push to 70%, but below that the volume cost per LR point gained accelerates sharply." That is a quantified, defensible position. It is not available from a scenario in Excel.

The volume shadow price tells a similar story. At a 97% volume retention bound it is zero throughout the upper LR range - the volume constraint is not binding. It only starts binding below 70% LR, where the required rate increases are large enough to trigger meaningful lapse.

---

## The FCA ENBP constraint

PS21/11 prohibits renewal premiums above the new business equivalent through the same channel. For most UK motor and home portfolios, this is a genuinely binding constraint: renewal customers typically received loyalty discounts that are now prohibited. When you take rate on renewals, you have to be careful which policies are constrained by ENBP.

The library enforces this per-policy via the `enbp` array and the `enbp_buffer` in `ConstraintConfig`:

```python
config = ConstraintConfig(
    lr_max=0.72,
    retention_min=0.97,
    enbp_buffer=0.01,   # stay 1% below the ENBP ceiling
)

opt = PortfolioOptimiser(
    technical_price=technical_premiums,
    expected_loss_cost=loss_costs,
    p_demand=renewal_probs,
    elasticity=elasticities,
    renewal_flag=is_renewal,
    enbp=enbp_values,     # NB-equivalent price per renewal policy
    constraints=config,
)
```

The shadow price on the ENBP constraint tells you the dislocation cost of regulatory compliance: how much profit the optimiser forgoes because of the ENBP restriction. This number belongs in your PS21/11 impact analysis.

No other open-source tool we are aware of implements this constraint at all. Commercial tools such as Radar Optimiser and Earnix have it in some form, but with opaque solver implementations and no programmatic access to shadow prices.

---

## The stochastic formulation

The deterministic problem uses E[LR] ≤ target. The stochastic formulation, following Branda (2014), is stricter: P(LR ≤ target) ≥ α. The LR must stay below the target with confidence level α, not just in expectation.

Under a normal approximation - appropriate for large books where the CLT applies - this reformulates to:

```
E[LR] + z_α × σ[LR] ≤ target
```

where σ[LR] comes from the variance estimates in your GLM. For a Tweedie claims model you pass the dispersion parameter and power; the library derives the policy-level variance and aggregates.

```python
from insurance_optimise import ClaimsVarianceModel, PortfolioOptimiser, ConstraintConfig

var_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=loss_costs,
    dispersion=1.2,
    power=1.5,
)

config = ConstraintConfig(
    lr_max=0.72,
    retention_min=0.97,
    stochastic_lr=True,
    stochastic_alpha=0.95,
)

opt = PortfolioOptimiser(
    technical_price=technical_premiums,
    expected_loss_cost=loss_costs,
    p_demand=renewal_probs,
    elasticity=elasticities,
    renewal_flag=is_renewal,
    enbp=enbp_values,
    claims_variance=var_model.variance_claims,
    constraints=config,
)
result = opt.optimise()
```

The stochastic solver will always recommend higher rate than the deterministic one. The difference is the "uncertainty premium" - the additional rate required to maintain the LR target with 95% confidence rather than just in expectation. If your book has high variance (large Tweedie dispersion, concentrated exposure), this can be substantial. If you have a large, diversified portfolio, it will be small. That is not a surprise result, but it is a result the solver quantifies rather than leaves to intuition.

---

## What this is not

The library consumes GLM outputs; it does not fit them. Use statsmodels, CatBoost, or Emblem for the models, then feed the technical premiums and factor values here. It is also an offline rate strategy tool, not a real-time quote engine - Radar Live and Earnix handle individual-level pricing at point of quote.

---

## Getting started

```bash
uv add insurance-optimise

# With stochastic module (requires cvxpy):
uv add "insurance-optimise[stochastic]"
```

Source and issue tracker on [GitHub](https://github.com/burning-cost/insurance-optimise). The priority backlog includes a competitive equilibrium module (Lerner index pricing as baseline), Bayesian demand model integration to propagate posterior uncertainty over price elasticity through the optimiser, and a Consumer Duty fair value checker.

The `portfolio_summary()` method is the first thing to run before any solve - it gives you baseline metrics at current prices so you can see what the optimiser is starting from.

- [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/)
- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/)
