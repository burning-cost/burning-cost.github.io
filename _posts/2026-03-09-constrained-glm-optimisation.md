---
layout: post
title: "Constrained GLM Optimisation for Rate Adequacy"
date: 2026-03-09
categories: [pricing, techniques, libraries]
tags: [glm, rate-optimisation, slsqp, enbp, fca-ps215, shadow-prices, efficient-frontier, scipy, python, insurance-optimise, constrained-optimisation, loss-ratio]
description: "A GLM gives you technically adequate prices. Constrained optimisation finds the factor adjustments that hit your LR target, satisfy FCA PS21/5 ENBP, respect movement bounds, and minimise cross-subsidy simultaneously. This is the implementation guide for insurance-optimise v0.2.0."
---

A GLM gives you technically adequate prices at the risk level. The business problem is different: take the GLM relativities you have, apply a set of multiplicative factor adjustments across your tariff, and find the combination that satisfies four simultaneous requirements - portfolio loss ratio target, FCA PS21/5 ENBP constraint, volume floor, and factor movement guardrails - while minimising customer dislocation.

That problem cannot be solved by scenario analysis in Excel. It requires constrained nonlinear optimisation. This post covers how to formulate it correctly, why SLSQP with analytical Jacobians is the right solver, what shadow prices tell you about your constraint structure, and how to trace the efficient frontier to make the LR/volume tradeoff explicit.

The code uses [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) v0.2.0, MIT licence. Install with:

```bash
uv add insurance-optimise
```

---

## The problem, stated precisely

You have a multiplicative tariff with K rating factors. Each factor has a set of relativities fitted from a GLM. After the GLM is fitted, the pricing team wants to take rate: adjust the relativities upward or downward by factor-level multipliers.

The decision variable is a vector `m` of length K:

```
m = [m_1, m_2, ..., m_K]
```

where `m_k` is the multiplicative adjustment applied to all relativities in factor k. A value of 1.05 means every relativity in that factor band is scaled up by 5% - a parallel shift on the log scale, preserving the factor's shape. A value of 1.0 is no change.

The adjusted premium for policy i is:

```
pi_i(m) = pi_i_current * prod_k(m_k)
```

Note that every policy receives the same adjustment vector. This is the factor-level problem. It is distinct from the policy-level problem where each policy gets its own multiplier - see [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/) for that formulation.

Given a demand model that produces renewal probabilities `p_i(pi_i(m))` as a function of price, the expected loss ratio at adjustment vector m is:

```
E[LR(m)] = sum_i(p_i(m) * c_i) / sum_i(p_i(m) * pi_i(m))
```

where `c_i` is the technical premium from the GLM (the expected loss cost per exposure unit). Expected volume relative to the current pricing is:

```
E[vol(m)] = sum_i(p_i(m)) / sum_i(p_i(current))
```

The optimisation problem:

```
minimise   sum_k (m_k - 1)^2
subject to E[LR(m)] <= LR_target
           E[vol(m)] >= vol_bound
           pi_i^renewal(m) <= pi_i^NB_equiv(m)   for ENBP-relevant policies
           m_k in [m_k_min, m_k_max]              for all k
```

The objective - minimum sum of squared deviations from 1.0 - is the minimum dislocation criterion: find the smallest rate change, in L2 terms, that achieves the required outcomes. It is not a profit objective. If you want to maximise profit at the factor level, you need the policy-level optimiser.

---

## Constraint anatomy

### Loss ratio ceiling

The LR constraint is an upper bound on the demand-weighted ratio of expected claims to expected earned premium. The `LossRatioConstraint` in scipy SLSQP convention returns `LR_target - E[LR(m)]`, which is non-negative (satisfied) when the expected LR is at or below target.

The LR constraint is nonlinear in m because both the numerator and denominator depend on m through the demand model. When the demand model is a logistic function with log price ratio as the linear predictor:

```
p_i(m) = expit(alpha_i + beta * log(pi_i(m) / pi_market_i))
```

the gradient of E[LR] with respect to m_k involves a product-rule expansion over the demand probabilities. The gradient is:

```
d(E[LR])/d(m_k) = [sum_i(dp_i/dm_k * c_i) * D - sum_i(p_i * c_i) * dD/dm_k] / D^2
```

where D = sum_i(p_i * pi_i(m)) is the expected premium, and dp_i/dm_k = p_i * (1 - p_i) * beta / m_k (from the chain rule through the logistic). Computing this analytically at each iteration is what makes the solver fast enough for production use.

The insurance-optimise library computes these gradients numerically when you use the default `DemandModel` interface. For books of up to 10,000 policy rows, the numerical gradients via scipy's finite differencing are acceptable - SLSQP handles them well. For larger books or frequent re-solving (e.g., frontier tracing), the logistic demand form supports closed-form gradients.

### Volume floor

The volume constraint prevents excessive book shrinkage. The `VolumeConstraint` enforces:

```
sum_i(p_i(m)) / sum_i(p_i(current)) >= vol_bound
```

The denominator is the sum of renewal probabilities at current pricing - stored in the `renewal_prob` column of `PolicyData`. This makes the constraint relative to what you would expect at the current rate level, not to total policy count. A `vol_bound` of 0.97 means: the rate change may not reduce expected renewals by more than 3% of the current expected renewals. It says nothing about absolute renewal rate.

This formulation matters. If your current renewal rate is 72%, a volume floor of 0.97 allows it to fall to 72% * 0.97 = 69.8%. If you accidentally set the floor relative to policy count rather than expected count, you will find the constraint is much harder to satisfy than intended.

### ENBP constraint (FCA PS21/5)

FCA PS21/5 (effective January 2022, continuing under Consumer Duty) prohibits renewal pricing above the equivalent new business price through the same channel. The constraint is:

```
pi_i^renewal(m) <= pi_i^NB_equiv(m)   for all renewal policies in scope
```

In a multiplicative tariff, the NB-equivalent premium is the premium the customer would receive if they approached as a new customer through the same channel - which means excluding renewal-only factors (tenure discounts, NCB-at-renewal) from the adjustment. Factors that apply to both new and renewal business get the full m_k adjustment on both sides; renewal-only factors get m_k on the renewal side and 1.0 on the NB side.

The `ENBPConstraint` handles this through the `renewal_factor_names` field of `FactorStructure`. Declare which factors are renewal-only:

```python
from insurance_optimise import FactorStructure

fs = FactorStructure(
    factor_names=["f_age_band", "f_ncb", "f_vehicle_group", "f_region", "f_tenure_discount"],
    factor_values=df.select(["f_age_band", "f_ncb", "f_vehicle_group", "f_region", "f_tenure_discount"]),
    renewal_factor_names=["f_tenure_discount"],  # renewal-only; excluded from NB equivalent
)
```

When the factor adjustment vector applies uniformly to both renewal and NB factors, ENBP is automatically satisfied - the constraint does not bind. It binds when renewal-specific adjustments diverge from NB adjustments, which happens when you try to take more rate on renewal-only factors than on the shared factors.

The regulatory implementation detail: PS21/5 is channel-specific. A PCW renewal premium must not exceed the PCW new business price; a direct renewal must not exceed the direct new business price. Cross-channel comparison is not required.

```python
from insurance_optimise import ENBPConstraint

opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
```

### Factor bounds

Factor bounds prevent the solver from making unreasonably large adjustments that would destabilise the tariff or shock customers beyond acceptable limits. They are implemented as box constraints - the `bounds` argument to `scipy.optimize.minimize` - rather than as SLSQP inequality constraints. The distinction matters: SLSQP handles box constraints via projection, which is more numerically stable than adding them as additional rows to the constraint Jacobian.

```python
from insurance_optimise import FactorBoundsConstraint

# Uniform bounds: all factors can move -10% to +15%
opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors))

# Per-factor bounds: tighter on tenure discount (ENBP-sensitive)
import numpy as np
lower = np.array([0.90, 0.90, 0.90, 0.90, 0.98])  # tenure discount: max -2%
upper = np.array([1.15, 1.15, 1.15, 1.15, 1.05])  # tenure discount: max +5%
opt.add_constraint(FactorBoundsConstraint(lower=lower, upper=upper, n_factors=5))
```

---

## Full implementation

The complete setup from GLM outputs to optimised factor adjustments:

```python
import polars as pl
import numpy as np

from insurance_optimise import (
    PolicyData, FactorStructure, DemandModel,
    RateChangeOptimiser, EfficientFrontier,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from insurance_optimise.demand import make_logistic_demand, LogisticDemandParams

# 1. Policy data - one row per in-force policy
#    technical_premium: GLM-fitted expected loss cost (frequency * severity)
#    current_premium: what you are currently charging
#    renewal_flag: 1 for renewals, 0 for new business
#    renewal_prob: predicted renewal probability at current_premium
df = pl.read_parquet("motor_book.parquet")
data = PolicyData(df)

# 2. Factor structure
#    factor_values: the current relativity value for each policy x factor cell
#    These come from your GLM factor tables, not from the policy data directly
factor_names = ["f_age_band", "f_ncb", "f_vehicle_group", "f_region", "f_tenure_discount"]
fs = FactorStructure(
    factor_names=factor_names,
    factor_values=df.select(factor_names),
    renewal_factor_names=["f_tenure_discount"],
)

# 3. Demand model
#    Intercept and price_coef from your retention/conversion model
#    PCW UK motor: price_coef typically -2.0 to -3.5 (semi-elasticity on log scale)
params = LogisticDemandParams(intercept=1.2, price_coef=-2.5, tenure_coef=0.06)
demand = make_logistic_demand(params)

# 4. Optimiser setup
opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)
opt.add_constraint(LossRatioConstraint(bound=0.72))
opt.add_constraint(VolumeConstraint(bound=0.97))
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors))

# 5. Feasibility check before solving
print(opt.feasibility_report())
#  constraint        value  satisfied
#  loss_ratio_ub     0.042  True      (current LR is 4.2pp below target)
#  volume_lb         0.000  True      (at current rates, vol floor is exactly met)
#  enbp             -0.003  False     (ENBP violated at current rates for some PCW renewals)
#  factor_bounds     0.100  True

# 6. Solve
result = opt.solve()
print(result.summary())
```

The feasibility report is the step most implementations skip. If your constraints are infeasible at current rates (as the ENBP violation above shows), the solver will correctly report that no feasible solution exists. Diagnosing infeasibility before the solve - rather than staring at a failed OptimizeResult - saves substantial debugging time.

---

## Why SLSQP, and what it requires from you

SLSQP (Sequential Least Squares Programming) is the right solver for this problem for three reasons.

First, it handles nonlinear inequality constraints natively. Interior point methods (IPOPT) are stronger for large-N but require interfacing with separate binaries. Trust-region methods in scipy do not support arbitrary inequality constraints. SLSQP is in scipy, handles multiple inequality constraints, and converges in tens to low hundreds of iterations for well-posed problems of K = 5 to 20 factors.

Second, it handles box constraints via projection rather than as additional inequality rows. This is both faster and more numerically stable - the projection step is exact and cheap, whereas soft-penalising bounds via inequality constraints adds rows to the Jacobian and creates conditioning issues.

Third, it extracts Lagrange multipliers from the solution. The `OptimizeResult.v` attribute in scipy's SLSQP implementation carries the KKT multipliers for inequality constraints. The library extracts these automatically:

```python
result.shadow_prices
# {
#   'loss_ratio_ub': 0.031,
#   'volume_lb': 0.000,
#   'enbp': 0.004
# }
```

The shadow price (Lagrange multiplier) on a binding constraint tells you the marginal improvement in the objective per unit relaxation of the constraint bound. A shadow price of 0.031 on the LR constraint means: relax the LR target by one percentage point (from 72% to 73%), and the minimum achievable dislocation falls by 0.031 (a sum-of-squares reduction across the K factor adjustments). Zero on the volume constraint means it is not binding - you can hit the LR target without approaching the volume floor. A non-zero ENBP shadow price means compliance is costing you: it is forcing factor adjustments away from the unconstrained optimum.

One genuine SLSQP trap: the solver can return `success=True` with active constraint violations when tolerance settings are loose (scipy issue #8986). The library runs a post-solve feasibility check regardless of the success flag. With `ftol=1e-8` (tighter than scipy's default `1e-6`), this is rare in practice but the check runs unconditionally.

---

## The efficient frontier

The efficient frontier is the most important output that scenario analysis cannot produce. It shows the complete set of achievable (LR, volume) pairs - every Pareto-optimal rate strategy.

```python
frontier = EfficientFrontier(opt)
frontier_df = frontier.trace(lr_range=(0.68, 0.78), n_points=20)
```

The trace sweeps from the most relaxed LR target (0.78) to the tightest (0.68), solving the constrained optimisation at each point with warm starting - using the solution from the previous (slightly more relaxed) point as the initial guess for the next. Warm starting typically halves the number of iterations per solve compared to cold starting from all-ones.

The output DataFrame has one row per LR target:

```
lr_target  expected_lr  expected_volume  feasible  shadow_lr  shadow_volume
     0.78        0.775            0.974     True      0.02           0.00
     0.76        0.758            0.972     True      0.04           0.00
     0.74        0.739            0.968     True      0.08           0.00
     0.72        0.720            0.963     True      0.15           0.00
     0.70        0.700            0.954     True      0.31           0.01
     0.68        0.680            0.937     True      0.72           0.08
     0.66           --               --    False        --             --
```

The rising shadow price on the LR constraint signals the frontier's knee. Moving from a 0.78 LR target to 0.76 costs 0.02 units of dislocation per percentage point. Moving from 0.70 to 0.68 costs 0.72 units. The marginal cost of LR improvement is accelerating sharply as you approach 0.68 - you are near the feasibility boundary, and further improvement requires either infeasible factor adjustments or breaching the volume floor.

The knee of the frontier - where shadow prices start rising steeply - is where the conversation with commercial directors should happen. The question "should we target 70% or 68% LR?" is not a modelling question. It is a business question about how much volume you are willing to sacrifice and how much factor dislocation your customers will absorb. The frontier gives you the terms of that trade. The shadow prices give you the rate of exchange at each point.

```python
# Inspect shadow price summary across the frontier
print(frontier.shadow_price_summary())

# Plot the frontier (requires matplotlib)
frontier.plot(annotate_shadow=True)
```

---

## Stochastic formulation: managing parameter uncertainty

The deterministic formulation uses point estimates of expected claims. This ignores two sources of uncertainty that matter in practice.

The first is claims variance within the GLM. A Tweedie GLM with power p and dispersion phi gives variance `phi * mu^p` per policy. At the portfolio level, under independence, the loss ratio variance is approximately:

```
Var[LR] ≈ sum_i(p_i * phi * c_i^p) / (sum_i(p_i * pi_i))^2
```

The second is parameter uncertainty in the elasticity estimate. If your demand model gives a 90% confidence interval of [-3.1, -1.9] on the price coefficient, the optimal rate changes span a range that the deterministic formulation ignores.

The stochastic extension (Branda 2013) reformulates the LR constraint as a chance constraint:

```
P(LR(m) <= LR_target) >= alpha
```

Under a normal approximation for the portfolio LR (reasonable for books of 10,000+ policies by the central limit theorem), this reformulates as a deterministic constraint with a variance penalty:

```
E[LR(m)] + z_alpha * sqrt(Var[LR(m)]) <= LR_target
```

For alpha = 0.95, z_alpha = 1.645. The variance term is a function of the GLM dispersion parameter:

```python
from insurance_optimise.stochastic import ClaimsVarianceModel, StochasticRateOptimiser

# Dispersion from your Tweedie GLM summary
# In statsmodels: result.scale (the dispersion parameter phi)
variance_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=data.df["technical_premium"].to_numpy(),
    dispersion=1.3,   # phi from Tweedie GLM
    power=1.5,        # p from Tweedie GLM
)

stochastic_opt = StochasticRateOptimiser(
    data=data, demand=demand, factor_structure=fs,
    variance_model=variance_model,
    lr_bound=0.72,
    alpha=0.95,
)
result = stochastic_opt.solve()
```

The stochastic solver recommends a higher rate than the deterministic one. The difference - the amount of additional rate required to maintain the LR constraint with 95% confidence rather than in expectation - is the uncertainty premium. For a UK motor book with typical Tweedie dispersion around 1.2 to 1.5, this is usually 1 to 2 percentage points of LR headroom, which translates to an average rate uplift of 1 to 3 percentage points depending on your book's composition.

Whether to use the stochastic formulation is a governance decision, not a statistical one. If your CFO treats the LR target as a hard floor that must be met with high probability, use it. If the LR target is a planning assumption and mild misses are tolerated, the deterministic formulation is appropriate.

---

## Audit trail and regulatory evidence

Every optimisation run produces a reproducible audit trail. For FCA Consumer Duty and PS21/5, you need to demonstrate that the ENBP constraint was enforced and that the methodology is documented. For the underwriting director's sign-off, you need to show that the rate change was derived from a formal optimisation, not a spreadsheet scenario.

```python
import json

# After result = opt.solve()
audit = {
    "timestamp": "2026-03-09T11:30:00Z",
    "n_policies": data.n_policies,
    "factor_names": fs.factor_names,
    "constraints": {
        "loss_ratio_bound": 0.72,
        "volume_bound": 0.97,
        "factor_bounds": {"lower": 0.90, "upper": 1.15},
        "enbp_channels": ["PCW", "direct"],
    },
    "solver": {
        "method": "SLSQP",
        "tol": 1e-8,
        "maxiter": 500,
    },
    "result": {
        "success": result.success,
        "n_iterations": result.n_iterations,
        "factor_adjustments": result.factor_adjustments,
        "expected_lr": result.expected_lr,
        "expected_volume": result.expected_volume,
        "shadow_prices": result.shadow_prices,
        "objective": result.objective_value,
    },
}

with open("rate_change_audit_2026Q2.json", "w") as f:
    json.dump(audit, f, indent=2)
```

The JSON is the evidence trail. The factor adjustments are the rate change recommendation. The shadow prices are the commercial analysis. The solver metadata is the methodology documentation.

For ENBP specifically: the shadow price on the ENBP constraint is positive when compliance is forcing the solver away from the unconstrained optimum. This is directly relevant to PS21/5 impact analyses - the regulator's question "what is the cost of ENBP to the firm?" has a precise answer in the shadow price.

---

## Practical scale and performance

The factor-level optimiser has K decision variables (K = number of rating factors), not N policy-level variables. For a typical UK personal lines tariff with 5 to 20 factors, K is small. The objective and constraint functions require O(N) computation per evaluation - iterating over all N policies - but K is small enough that SLSQP converges quickly.

Expected runtimes:

- N = 5,000 policies, K = 8 factors: under 2 seconds per solve, 20 frontier points in under 30 seconds
- N = 50,000 policies, K = 12 factors: 5-15 seconds per solve, 20 frontier points in 2-4 minutes
- N = 500,000 policies, K = 12 factors: 30-90 seconds per solve; consider segmenting first

For production UK personal lines books of 100,000 to 500,000 policies, the practical approach is to aggregate to a segment table (group by all factor cells) before passing to the optimiser, then apply the resulting factor adjustments to the full book. A segment table for a 5-factor tariff with typical cardinality has 2,000 to 20,000 rows - well within the fast-converging range.

The segment aggregation step:

```python
# Aggregate to unique factor cells
segment_df = (
    df.group_by(factor_names)
    .agg([
        pl.col("technical_premium").sum().alias("technical_premium"),
        pl.col("current_premium").sum().alias("current_premium"),
        pl.col("renewal_flag").mean().alias("renewal_flag"),
        pl.col("renewal_prob").mean().alias("renewal_prob"),
    ])
)
# Pass segment_df to PolicyData, not the full policy df
data = PolicyData(segment_df)
```

The optimiser sees the segment-level LR and volume statistics. The resulting factor adjustments apply identically to every policy in each factor cell, which is exactly what a multiplicative tariff requires.

---

## Connecting to the broader pipeline

The insurance-optimise takes two upstream inputs: GLM-fitted technical premiums, and demand model renewal probabilities.

For the technical premium, any GLM or GBM fitted to your claims experience feeds in directly. The `technical_premium` column is simply the model's predicted expected loss cost per policy - frequency * severity for a frequency-severity model, or the aggregate cost prediction for a combined model. If you are using GBM outputs, [`shap-relativities`](https://github.com/burning-cost/shap-relativities) can extract factor-structured relativities from the GBM that the optimiser can then adjust.

For the demand model, the renewal probability at current pricing goes in the `renewal_prob` column, and the demand model callable is passed to the optimiser so it can re-evaluate probabilities at each candidate rate change. The [`insurance-causal`](https://github.com/burning-cost/insurance-causal) library uses Double Machine Learning to estimate causal price elasticity from historic quote data; the resulting logistic model feeds directly into the `DemandModel` wrapper.

After the optimisation, the downstream step is ratebook update and deployment. The `factor_adjustments` dict maps factor name to multiplicative adjustment. Applying these to your existing GLM factor tables produces the revised relativities that go into your rating system. That step - from adjusted relativities to rating engine update - is outside the scope of this library but is described in the insurance-optimise repository's examples directory.

For monitoring after the rate change is live: the question "did the rate change actually deliver the targeted LR improvement?" is a causal inference problem, not a simple comparison. [`insurance-causal-policy`](https://github.com/burning-cost/insurance-causal-policy) implements the SDID estimator for exactly this use case.

---

## Limitations worth naming

The objective function is convex in m (sum of squares), but the constraints are nonlinear in m through the demand model. The full problem is not convex. SLSQP finds a local optimum, not necessarily the global one. For K <= 10 factors, we recommend running with three to five random initialisations and taking the best feasible solution - the solver is fast enough that this adds minimal overhead.

The demand model must be evaluated hundreds of times during the solve. A CatBoost model with a 1ms prediction time per call adds up to seconds of overhead per iteration. If your demand model is slow, wrap it in a cache keyed on the price ratio vector, or use the parametric logistic form for the optimisation and validate the result against the full CatBoost model afterwards.

The ENBP constraint enforces a maximum per-policy condition: the solver checks that no individual renewal premium exceeds its NB equivalent. Formally, it evaluates the maximum excess across all renewal policies in scope and requires that maximum to be non-positive. This is the correct regulatory formulation. The practical caveat is that the optimiser operates at factor level - the same adjustment multiplier applies to all policies in a factor cell - so a constraint binding at the optimum may have marginal per-policy violations in adjacent cells. The per-policy post-optimisation check must always run before the rate change is submitted for approval.

---

**Related articles from Burning Cost:**
- [Constrained Rate Optimisation and the Efficient Frontier](/2026/02/21/constrained-rate-optimisation-efficient-frontier/)
- [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/)
- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/)
