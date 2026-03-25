---
layout: post
title: "Does constrained rate optimisation actually work for insurance pricing?"
date: 2026-03-24
categories: [libraries, validation]
tags: [optimisation, pricing, demand, elasticity, slsqp, enbp, fca, retention, loss-ratio, motor, pareto]
description: "Benchmark results on synthetic UK motor renewal books. The constrained optimiser outperforms flat rate changes on profit and retention simultaneously. What it does not do: fix a bad elasticity estimate. Numbers and honest caveats."
---

Every UK motor insurer runs an annual renewal rate cycle. The standard approach is a spreadsheet: apply a flat uplift to all renewals, eyeball the retention impact, check you are not breaching FCA PS21/5 (ENBP), and call it done. The optimiser pitch is that this is provably suboptimal. You have elastic PCW customers who will lapse if you hit them with the same increase you are applying to loyal direct customers. Flat loading trades away profitable retention to apply margin equally to people with very different price sensitivities.

The question is whether an SLSQP optimiser with demand modelling actually does better, or whether the improvement is theoretical - marginal gains drowned by elasticity estimation error.

We ran the benchmark. Here is what we found.

---

## The setup

2,000 synthetic UK motor renewal policies. Portfolio characteristics:

- Technical premiums log-normally distributed, mean £545 (SD ~35%)
- Expected loss ratio 65% with cross-policy variation
- 55% PCW-acquired, 45% direct - realistic UK motor mix
- Mean price elasticity −1.65 (PCW mean −2.0, direct mean −1.2), with within-segment variation
- Base retention ~80% at current prices
- ENBP ceiling (FCA PS21/5) set per policy as a multiple of technical price

The comparison is between two approaches, both using the same portfolio:

- **Uniform +7%:** apply a flat 1.07 multiplier to all renewals, then clip to ENBP. This is what most pricing teams do.
- **Constrained optimiser:** `PortfolioOptimiser` from [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise), SLSQP with analytical Jacobians. Constraints: LR cap 68%, retention floor 78%, ±25% maximum rate change per policy, ENBP compliance, technical floor (price ≥ cost).

Elasticities are treated as known for this benchmark - we are testing whether the optimiser finds a better solution given the demand model, not whether elasticity estimation works (that is a separate, harder problem). SLSQP with analytical gradients converges in under one second at N=2,000.

---

## The results

### Portfolio-level comparison: uniform vs optimised

| Metric | Uniform +7% | Constrained optimiser | Change |
|--------|-------------|----------------------|--------|
| GWP | baseline | similar | within ±2% |
| Expected profit | baseline | **+£4,000–8,000** (+5–8%) | +5–8% |
| Retention rate | ~74–76% | **~78–80%** | +2–4pp |
| Loss ratio | ~67–69% | **~65–67%** | −2pp |
| ENBP compliance | post-hoc capping | built into constraints | structural |
| FCA audit trail | No | Yes (per-policy JSON) | - |
| Solve time (N=2,000) | N/A | **<1 second** | - |

The optimiser achieves higher profit and higher retention simultaneously. That sounds too good - and the mechanism explains why it is not. Flat loading applies a 7% increase to every customer uniformly. The elastic PCW customers (mean elasticity −2.0) lapse at higher rates under 7% than they would under a smaller increase. The inelastic direct customers (mean elasticity −1.2) would accept a larger increase than 7%. The optimiser does exactly this reallocation: smaller increases to PCW, larger to direct, subject to all the constraints. Net result: fewer lapses on the elastic side, more margin captured on the inelastic side.

The profit uplift of £4,000–8,000 on a 2,000-policy book corresponds to roughly £2–4 per policy. At 50,000 policies: £100,000–200,000 per renewal cycle, before accounting for retention carry-over effects and reduced acquisition cost from not having to replace lapsed customers.

### The ENBP effect

PS21/5 (ENBP) bans renewal premiums from exceeding what a new customer would be quoted for the same risk. In the flat-loading baseline, ENBP is typically handled by post-hoc capping: apply the flat uplift, then clip renewals that breach the ceiling. This is compliant but wasteful - it removes margin from policies where you could legitimately charge more.

The constrained optimiser treats ENBP as a hard upper bound on each policy's multiplier, which it is. The per-policy JSON audit trail records the constraint configuration, the optimal multiplier, and whether ENBP was binding for each renewal. You can show this to the FCA. The flat-loading approach produces no equivalent evidence of systematic ENBP governance.

### The Pareto surface: profit, retention, fairness

On a 1,000-policy synthetic UK motor book with deprivation quintile as the fairness dimension:

| Optimisation approach | Profit (£) | Retention | Fairness disparity ratio |
|-----------------------|-----------|-----------|--------------------------|
| Single-objective SLSQP (profit only) | 31,650 | 0.871 | **1.168** |
| Pareto balanced point (TOPSIS 0.5/0.3/0.2) | 28,940 | **0.912** | **1.043** |
| Pareto min-disparity point | 22,180 | 0.951 | **1.011** |

The single-objective optimiser is blind to the fairness dimension. It achieves a disparity ratio of 1.168 - the most-deprived quintile pays 16.8% more on average than the least-deprived - without that appearing anywhere in the optimisation. The pricing committee does not know this is happening.

The Pareto surface makes it visible. A 9% reduction in profit (£31,650 to £28,940) buys a 1.25pp improvement in retention and a 0.125 reduction in fairness disparity. Whether that trade is worth making is a governance decision, not a technical one. But Consumer Duty (PS22/9) requires that decision to be made explicitly and documented. The Pareto surface is the structured input to that conversation.

---

## Using the library

The minimum viable optimisation run:

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

config = ConstraintConfig(
    lr_max=0.68,
    retention_min=0.78,
    max_rate_change=0.25,
    enbp_buffer=0.01,   # 1% safety margin below ENBP ceiling
    technical_floor=True,
)

opt = PortfolioOptimiser(
    technical_price=technical_price,      # from your GLM
    expected_loss_cost=expected_loss_cost, # from your GLM
    p_demand=p_renewal,                    # current renewal probability
    elasticity=price_elasticity,           # from insurance-elasticity
    renewal_flag=is_renewal,
    enbp=enbp,                             # new business quote per policy
    constraints=config,
)

result = opt.optimise()
# OptimisationResult(CONVERGED, N=2000, profit=..., lr=0.664, retention=0.793)

result.save_audit("renewal_q1_2026_audit.json")
```

The efficient frontier, which is the conversation your pricing committee needs to have, is a single additional call:

```python
from insurance_optimise import EfficientFrontier

frontier = EfficientFrontier(
    opt,
    sweep_param="volume_retention",
    sweep_range=(0.78, 0.95),
    n_points=15,
)
result = frontier.run()
frontier.plot()  # profit-retention curve
```

If elasticity estimates carry material uncertainty - which they always do - use scenario mode:

```python
result_scenarios = opt.optimise_scenarios(
    elasticity_scenarios=[
        elasticity * 0.75,   # customers more price-sensitive
        elasticity,           # central estimate
        elasticity * 1.25,   # customers less price-sensitive
    ],
    scenario_names=["pessimistic", "central", "optimistic"],
)
```

Report the spread across scenarios to the pricing committee, not just the central case.

---

## What the optimiser does not fix

### Elasticity inputs drive everything

The optimiser finds the mathematically optimal multipliers given the demand model. If the elasticity inputs are wrong - and in UK motor, elasticity estimates from observational PCW data typically carry ±50% relative uncertainty - the "optimal" strategy is optimal for the wrong demand curve.

The benchmark on 50,000 PCW quotes is honest about this. DML elasticity estimation on data with narrow price variation (standard deviation of log price ratio = 0.045, typical of quarterly loading cycles) produced an estimate of −4.03 against a true effect of −2.0. Larger absolute bias than naive full-controls logistic regression. The 95% CI of [−5.65, −2.40] is wide and has a robustness value of 2.1% - small residual confounding can move the estimate materially.

The pricing lift remains positive (+143.8% profit versus flat loading) because even a directionally correct but biased elasticity estimate allocates prices in the right direction across segments. However, the magnitude of the per-policy gains is not reliable when elasticity uncertainty is this large. Use scenario mode and report the spread. Do not present a single optimal solution as a point estimate.

### Log-linear demand breaks down above ±20% rate change

The constant-elasticity log-linear demand model `x(m) = x0 * m^ε` is calibrated for UK personal lines annual renewals - rate changes in the ±10–15% range. Above ±20%, the model implies demand responses that do not hold empirically: extreme price increases do not simply produce proportionally more lapse. Retention floors. Solutions recommending rate changes beyond ±20% should be scrutinised regardless of what SLSQP returns.

### SLSQP finds local, not global, optima

SLSQP is a local solver. Starting from the initial point (technical premiums by default) it finds a local optimum that may not be globally optimal. For N up to 1,000 with well-behaved constraint sets, this is rarely a practical issue - the feasible region is usually convex. At N > 1,000 with binding, interacting constraints (ENBP + retention floor + LR cap simultaneously active), non-convexity can appear. For production use on large books with multiple binding constraints, run several restarts from different initial points and compare objectives.

### N > 5,000 requires segment aggregation

SLSQP with analytical Jacobians handles N=5,000 in seconds. At N=50,000, the Jacobian becomes large and solve time extends to 5–15 minutes. The standard fix is to aggregate to pricing cells (vehicle group × age band × region × channel), optimise at cell level, and disaggregate multipliers back to policies. This is standard actuarial practice and does not reduce pricing quality provided the cells are homogeneous on elasticity.

### The ENBP ceiling requires accurate NB quotes

The optimiser enforces ENBP as a hard per-policy upper bound using the `enbp` array you provide. If those values are stale - the NB quote was generated six months ago, or by a different rating engine than the renewal model - the ENBP ceiling may be inaccurate. The library cannot detect this. Data pipeline governance is required to ensure ENBP inputs are contemporaneous and consistent with the renewal rating engine. This is an operations problem, not an optimisation problem.

---

## Our read

The benchmark result is clear. On a UK motor renewal book with heterogeneous price elasticities across PCW and direct channels, constrained optimisation outperforms flat loading on profit and retention simultaneously, with structural ENBP compliance built in. The per-policy gains are £2–4 at N=2,000; at production scale, the annual impact is meaningful.

The Pareto surface is the piece we find most compelling from a regulatory standpoint. Single-objective optimisation is actively blind to fairness - it achieves a 16.8% premium disparity across deprivation quintiles without registering the fact. Consumer Duty requires that trade-off to be made explicitly. The Pareto surface is the structured format for doing so.

The limitations are real and we have not softened them. Elasticity uncertainty is the dominant source of risk in any demand-curve-based pricing strategy. The library handles this honestly - scenario mode, wide confidence intervals, documented robustness values. The answer to elasticity uncertainty is not to abandon demand-curve pricing, which would mean returning to flat loading with its demonstrably suboptimal properties. The answer is to represent and communicate the uncertainty, which is what the tools support.

We think every pricing team running a UK motor renewal cycle should have constrained optimisation in the loop, with scenario outputs as the standard deliverable to the pricing committee rather than a single point solution. The library makes that practical.

The library is at [github.com/burning-cost/insurance-optimise](https://github.com/burning-cost/insurance-optimise). The full benchmark is at `benchmarks/benchmark.py` and `benchmarks/benchmark_pareto.py`, runnable locally or on Databricks.
