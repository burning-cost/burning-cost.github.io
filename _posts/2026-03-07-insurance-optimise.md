---
layout: post
title: "Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement"
date: 2026-03-07
categories: [pricing, libraries]
tags: [optimisation, portfolio-pricing, enbp, fca, slsqp, elasticity, renewal-pricing, motor, home, python]
description: "Applying rate changes without solving the demand-constraint system simultaneously is a guaranteed route to suboptimal profit. insurance-optimise is an open-source constrained portfolio rate optimiser for UK personal lines — SLSQP, analytical Jacobians, FCA ENBP enforcement, shadow prices."
---

This post covers **policy-level** profit optimisation: finding the optimal price multiplier for each individual risk given elasticity estimates. For factor-level tariff optimisation that adjusts multiplicative rating factor relativities, see [Constrained Rate Optimisation and the Efficient Frontier](/2026/02/21/constrained-rate-optimisation-efficient-frontier/).

The typical UK personal lines pricing cycle works like this. The technical model produces a set of risk-adequate prices. Separately, a demand model estimates how customers will respond to price changes. Someone — often the Chief Pricing Actuary, in a spreadsheet — then tries to reconcile the two: applying rate changes that are technically justified, commercially viable, and ENBP-compliant, while watching the portfolio loss ratio and not shocking customers with 40% increases.

This manual reconciliation process produces rates that are approximately right and definitively not optimal. It cannot be optimal because it is not solving the joint problem. It is solving a sequence of sub-problems that interact with each other in ways that manual iteration cannot fully untangle.

The interaction is direct: raising a segment's rate reduces demand, which changes the portfolio GWP, which changes the loss ratio calculation, which may tighten the ENBP constraint for some policies, which changes how much room you have to take rate elsewhere. The only way to find the profit-maximising rate change set — subject to all constraints simultaneously — is to solve it as a constrained optimisation problem.

That is what [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) does.

---

## What the joint problem looks like

There are N policies in your book. For each policy i, you are choosing a price multiplier m_i (the ratio of your offered price to the technical price). Your profit is:

```
f(m) = Σ_i (m_i × tc_i − cost_i) × x_i(m_i)
```

where `tc_i` is the technical price, `cost_i` is the expected loss cost, and `x_i(m_i)` is the demand model (the probability that this customer accepts the price `m_i × tc_i`).

You are maximising this over m, subject to:

- **FCA PS21/11 (ENBP):** For every renewal policy, the offered price must not exceed the price that customer would be quoted as a new customer through the same channel. One-sided: you can price below ENBP, you cannot price above.
- **Loss ratio ceiling:** aggregate expected claims / aggregate expected GWP ≤ target (e.g. 70%).
- **GWP floor:** your finance team needs a minimum written premium number.
- **Retention floor:** you cannot lose more than X% of your renewal book without triggering underwriting and capital concerns.
- **Rate change guardrails:** no individual policy moves more than ±25% in a single year.
- **Technical floor:** no policy priced below cost.

These constraints are coupled. Relaxing your retention floor gives you room on loss ratio. Tightening ENBP (by maintaining a buffer below the constraint) may force you to take less rate on renewals than the technical model supports, which pushes GWP-driven profit onto new business. The optimal multiplier for any one policy depends on the constraint values for all other policies.

Manual iteration does not solve this. It approximates it badly.

---

## Using the library

Install with uv:

```bash
uv add insurance-optimise
```

The main input is segment-level data: technical price, expected loss cost, baseline demand probability (renewal rate or conversion rate at current price), and price elasticity from your demand model. The elasticity output from [`insurance-causal`](https://github.com/burning-cost/insurance-causal) feeds directly in.

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

config = ConstraintConfig(
    lr_max=0.70,            # aggregate LR ceiling
    gwp_min=50_000_000,     # £50m minimum GWP
    retention_min=0.82,     # 82% renewal retention floor
    max_rate_change=0.25,   # ±25% per policy
    enbp_buffer=0.01,       # 1% safety margin below ENBP
    technical_floor=True,   # no pricing below cost
)

opt = PortfolioOptimiser(
    technical_price=df['tc'].to_numpy(),
    expected_loss_cost=df['cost'].to_numpy(),
    p_demand=df['p_renew'].to_numpy(),
    elasticity=df['elasticity'].to_numpy(),
    renewal_flag=df['is_renewal'].to_numpy(),
    enbp=df['enbp'].to_numpy(),
    constraints=config,
)

result = opt.optimise()
```

The result object carries everything you need:

```python
result.expected_profit       # £ profit at optimal prices
result.expected_loss_ratio   # aggregate LR
result.expected_retention    # renewal retention rate
result.shadow_prices         # Lagrange multipliers per constraint
result.summary_df            # per-policy: multiplier, new premium, ENBP binding?
```

The `summary_df` is a Polars DataFrame with one row per policy. The `enbp_binding` column tells you which renewals are priced tight to the ENBP ceiling — this is your ICOBS 6B.2 compliance evidence.

---

## The shadow prices are the output that matters

The shadow price (Lagrange multiplier) on a binding constraint tells you the marginal cost of that constraint in profit terms. This is the number pricing actuaries need and almost never have.

```python
result.shadow_prices
# {'lr_max': 1247.3, 'gwp_min': 0.0, 'retention_min': 8423.1}
```

In this example: the loss ratio ceiling is binding (shadow price £1,247, meaning each 1-percentage-point relaxation of the LR target buys you approximately £1,247 in profit). The GWP floor is not binding (shadow price zero — you are meeting it with room to spare). The retention floor is the hard constraint, costing you £8,423 per percentage-point.

This tells you immediately where to direct commercial conversations. The actuarial team trying to argue for a tighter loss ratio target has a £1,247/pp number to put in front of the CFO. The sales team pushing for an 85% retention floor has an £8,423/pp number that quantifies exactly what they are asking the business to give up.

Without constrained optimisation, you have approximate intuitions. With it, you have prices.

---

## SLSQP with analytical Jacobians

The solver is scipy's SLSQP (Sequential Least Squares Programming). We run it in multiplier space (optimising m_i = p_i / tc_i rather than p_i directly), which keeps the decision variables O(1) in magnitude and reduces SLSQP's scale sensitivity.

The critical performance decision is analytical gradients. Without them, SLSQP uses finite differences: at each iteration, it perturbs each variable by a small amount and measures the objective change. For N policies that is 2N extra function evaluations per iteration — at N = 10,000 that is slow enough to be impractical. With analytical gradients, each iteration requires one forward pass and one gradient evaluation: fast regardless of N.

The objective gradient has a clean form:

```
d(profit)/d(m_i) = tc_i × x_i(m_i) + (m_i × tc_i − cost_i) × x_i'(m_i)
```

The first term is the marginal revenue from increasing this policy's price. The second term is the profit margin times the demand loss from increasing price. The optimum trades these off at the margin. We compute all constraint Jacobians analytically, including the loss ratio gradient, which involves a ratio of sums and requires the quotient rule but is exact.

One genuine trap with SLSQP: it can return `success=True` even when constraint violations are material (scipy issue #8986). The library does not trust the success flag. After every solve, we check:

```python
# In _check_feasibility — always run after scipy returns
for con in self._scipy_constraints:
    val = con["fun"](m)
    if val < -tol:   # constraint violated
        return False
```

With the default `ftol=1e-9` (tighter than scipy's default `1e-6`), premature convergence is rare. But the post-solve check runs regardless.

---

## Scenarios: handling elasticity uncertainty

Elasticity estimates carry uncertainty. A 90% confidence interval of [-0.55, -0.29] on the average treatment effect from DML means the optimal rate changes span a range. The scenario mode quantifies that range:

```python
result = opt.optimise_scenarios(
    elasticity_scenarios=[
        df['elasticity_lower'].to_numpy(),   # pessimistic demand (more elastic)
        df['elasticity'].to_numpy(),          # central estimate
        df['elasticity_upper'].to_numpy(),    # optimistic demand (less elastic)
    ],
    scenario_names=['pessimistic', 'central', 'optimistic'],
)

result.profit_mean   # £ expected profit
result.profit_p10    # lower bound across scenarios
result.profit_p90    # upper bound across scenarios

result.multiplier_mean    # mean optimal multiplier per policy
result.multiplier_p10     # lower bound on multiplier recommendation
result.multiplier_p90     # upper bound
```

This is the same scenario analysis that actuaries already run manually for reserving and pricing. Running it properly, with the optimiser solving the full constrained problem under each scenario, produces a defensible range rather than a manual sensitivity on a single spreadsheet tab.

---

## The efficient frontier

The tension between profit and retention is not a single decision point. It is a curve. The efficient frontier shows you every point on that curve — the profit-maximising solution for every possible retention target.

```python
from insurance_optimise import EfficientFrontier

frontier = EfficientFrontier(
    optimiser=opt,
    sweep_param='volume_retention',
    sweep_range=(0.80, 0.95),
    n_points=15,
)

frontier_result = frontier.run()
frontier_result.pareto_data()
# Polars DataFrame: retention, profit, gwp, loss_ratio (15 rows)
```

Each of the 15 points is an independent optimisation at a different retention floor. At 80% retention, the optimiser has more pricing freedom and extracts more profit. At 95%, it is heavily constrained and profit is lower — but you are holding more of your book. The frontier shows you the rate of exchange: how much profit you give up per percentage point of retention. That is the number that should drive the commercial conversation, not a gut feeling about what "reasonable" retention looks like.

---

## The regulatory audit trail

Every optimisation run produces a JSON audit trail:

```python
import json

with open('optimisation_audit.json', 'w') as f:
    json.dump(result.audit_trail, f, indent=2)
```

The trail captures inputs (policy counts, premium totals, constraint configuration), solver settings (method, tolerances, restarts), solution statistics (multiplier distribution, portfolio metrics), constraint evaluation at solution (binding/slack status), shadow prices, and convergence metadata. Under FCA Consumer Duty and ICOBS 6B.2, pricing teams need to demonstrate that the ENBP constraint was enforced and that the methodology is documented. This is the documentation.

Commercial optimisers (Akur8 Optim, WTW Radar Optimiser, Earnix Price-It) are closed algorithms. You can tell the FCA that the vendor enforces ENBP. You cannot show them the code. With `insurance-optimise` you can show the code, the constraint implementation, the Jacobians, and the solution trace. For a regulator under Consumer Duty asking "how do you know you are not overcharging existing customers?", that is a better answer than "our vendor does it."

---

## Practical scale

Runtime at SLSQP with analytical gradients:

- N = 1,000 policies: under 1 second
- N = 10,000 policies: 5–30 seconds depending on constraint binding
- N = 100,000 policies: 60–300 seconds (consider segment aggregation first)

For production UK personal lines books (typically 100,000 to several million policies) the practical approach is to run the optimiser at segment level (group similar risks), take the optimal multipliers per segment, and interpolate to individual policy level before rating engine deployment. The library produces the segment-level output. The ratebook update step is outside its scope but documented in the examples.

For the efficient frontier across 15 retention targets, `n_jobs=-1` parallelises via joblib if installed. Fifteen independent SLSQP solves at N = 5,000 run in under two minutes on a standard laptop.

---

## Where it sits in the stack

`insurance-optimise` is the 19th library in our suite and completes what we think of as the renewal pricing stack:

1. **Technical model**: GLM or GBM producing expected loss cost per risk
2. **[`insurance-causal`](https://github.com/burning-cost/insurance-causal)**: Double Machine Learning to estimate causal price elasticity, removing the confound between risk deterioration and demand response
3. **`insurance-optimise`**: constrained portfolio optimisation taking both outputs and finding the profit-maximising rate changes subject to regulatory and business constraints

Each step is individually useful. The three together replace the Excel-based manual reconciliation process that, based on the pricing teams we have spoken to, is still the dominant approach in UK personal lines. The Excel approach cannot enforce FCA constraints algebraically, cannot compute shadow prices, and cannot produce an efficient frontier. It produces a reasonable answer with no guarantee of optimality and no audit trail.

---

## Limitations

SLSQP finds local optima. The profit objective is non-convex (product of a price multiplier and a sigmoid-shaped demand function), so the global optimum is not guaranteed. For N ≤ 100 segments, we recommend running differential evolution as a global initialisation pass and feeding the result to SLSQP for refinement. For larger N, multiple random restarts (`n_restarts=5`) with the best feasible solution selected is practical.

The demand model choices (`log_linear` constant elasticity and `logistic`) are standard but not universal. If your demand model has a different functional form, the gradient calculations would need extending. We think most UK personal lines demand models are adequately approximated by one of the two, but we have not tested every market segment.

ENBP is taken as an input. Computing it correctly (applying the right new business model, the right channel, the right cash-incentive adjustment) is the firm's responsibility. Wrong ENBP inputs produce constraint violations that the library cannot detect, because it has no reference for what the "correct" ENBP is. Garbage in, garbage out; the library enforces the constraint you give it.

---

The `insurance-optimise` library is open source under the MIT licence at [github.com/burning-cost/insurance-optimise](https://github.com/burning-cost/insurance-optimise).

---

**Related articles from Burning Cost:**
- [Constrained Rate Optimisation and the Efficient Frontier](/2026/02/21/constrained-rate-optimisation-efficient-frontier/)
- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/)
