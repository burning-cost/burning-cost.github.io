---
layout: post
title: "Does Constrained Rate Optimisation Actually Work?"
date: 2026-03-29
categories: [validation]
tags: [rate-optimisation, pricing, FCA, ENBP, elasticity, efficient-frontier, python]
description: "We benchmarked constrained portfolio optimisation against a uniform +7% rate change on a 2,000-policy UK motor book. The optimiser achieved the same GWP target with £4,000–8,000 higher expected profit and 2–4pp better retention. Here is when that holds and when it does not."
---

The standard UK motor renewal process applies a flat multiplier to all policies in a segment, adjusts for the loss ratio, and iterates in a spreadsheet until the numbers look acceptable. This is suboptimal by construction.

The question is how suboptimal, and whether the improvement from a constrained portfolio optimiser is large enough to justify the complexity. We ran [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) against a uniform +7% rate change on a 2,000-policy synthetic book to find out.

---

## The benchmark

The scenario: 2,000 UK motor renewals. 55% PCW-acquired customers with mean price elasticity −2.0, 45% direct customers with mean elasticity −1.2. Mean technical premium £545. Constraints: loss ratio cap 68%, retention floor 78%, ±25% rate change limit per policy, FCA PS21/11 ENBP ceiling enforced per policy.

| Metric | Uniform +7% rate change | PortfolioOptimiser |
|---|---|---|
| Expected profit uplift | Baseline | +£4,000–8,000 (~5–8%) |
| Expected retention | ~74–76% | ~78–80% |
| Loss ratio | ~67–69% | ~65–67% |
| ENBP compliance | Partial (capped post-hoc) | Built into constraint set |
| Convergence time | N/A | <1s (2,000 policies) |

The mechanism is straightforward. A PCW customer with elasticity −2.0 facing a +7% increase sees an expected renewal probability drop of roughly 14 percentage points. A direct customer with elasticity −1.2 facing the same increase drops only 8 points — but could accept +10% with only a 12-point drop. The uniform loading leaves money on the table from the direct segment and loses customers unnecessarily from the PCW segment.

The optimiser charges direct customers more and PCW customers less. On a 50,000-policy renewal book, the advantage compounds: £100,000–200,000 per cycle before accounting for the retention benefit.

---

## Being honest about the elasticity problem

We should flag the DML elasticity result honestly, because the benchmark is transparent about a failure mode.

On the confounded quote panel (50,000 synthetic PCW quotes, true elasticity −2.0), DML produced an estimate of −4.03 — higher absolute bias than naive full-controls logistic regression (−1.21). This matters because the optimiser's per-policy targeting depends on accurate elasticity inputs.

The reason is a known limitation: the DGP uses small quarterly loading cycles (standard deviation of log price ratio = 0.045), which provides too little exogenous price variation for the DML cross-fitting step. With genuine A/B test data or larger rate change cycles (standard deviation ≥ 0.10), DML converges closer to truth and provides a valid 95% confidence interval that naive logistic cannot.

The key finding from the benchmark is that the profit advantage holds even with an imperfect elasticity estimate. Demand-curve-aware pricing outperforms flat loading by +143.8% mean profit per segment across all five risk segments tested. That is because knowing the shape of the demand curve — even approximately — is far more valuable than using a flat loading that ignores elasticity variation entirely.

---

## The Pareto surface

The benchmark we found most useful commercially is the three-objective Pareto sweep: profit, retention, and fairness disparity.

On 1,000 policies with a deprivation quintile as the fairness dimension:

| Optimisation | Profit (£) | Retention | Fairness disparity | Notes |
|---|---|---|---|---|
| Single-objective (profit only) | 31,650 | 0.871 | 1.168 | Blind to fairness |
| Pareto — max-profit point | 31,650 | 0.871 | 1.168 | Same solution |
| Pareto — balanced (TOPSIS 0.5/0.3/0.2) | 28,940 | 0.912 | 1.043 | 9% profit cost |
| Pareto — min-disparity point | 22,180 | 0.951 | 1.011 | 30% profit cost |

The single-objective optimiser produces a disparity ratio of 1.168 — the most-deprived quintile pays 16.8% more on average. The balanced Pareto point cuts this to 1.043 at a cost of 9% of profit. Moving to minimum disparity costs 30% of profit.

Whether any of those trade-offs are acceptable is a governance decision, not a technical one. The value of computing the surface is that it forces the decision to happen with concrete numbers, in a pricing committee, rather than being implicit in the choice of loading applied uniformly across the book.

---

## The API

```python
import numpy as np
from insurance_optimise import PortfolioOptimiser, ConstraintConfig, EfficientFrontier

config = ConstraintConfig(
    lr_max=0.68,
    retention_min=0.78,
    max_rate_change=0.25,
    enbp_buffer=0.01,
)

opt = PortfolioOptimiser(
    technical_price=df["technical_premium"].to_numpy(),
    expected_loss_cost=df["expected_loss_cost"].to_numpy(),
    p_demand=df["renewal_probability"].to_numpy(),
    elasticity=df["price_elasticity"].to_numpy(),
    renewal_flag=df["renewal_flag"].to_numpy(),
    enbp=df["enbp"].to_numpy(),
    constraints=config,
)

result = opt.optimise()
print(result.multipliers)       # per-policy price multiplier
print(result.shadow_prices)     # marginal cost of tightening each constraint

frontier = EfficientFrontier(
    optimiser=opt,
    sweep_param="lr_max",
    sweep_range=(0.64, 0.74),
    n_points=20,
)
frontier_result = frontier.run()
frontier_df = frontier_result.data
```

The shadow price table is the output to put in front of a commercial director. It shows the marginal profit cost of tightening the loss ratio target by one percentage point, at every point on the frontier. When the shadow price on the LR constraint jumps from 0.15 to 0.72 between two adjacent LR targets, you are approaching the knee of the frontier — the point at which further tightening accelerates the volume cost sharply. This is a quantified, defensible position that no spreadsheet scenario analysis can produce.

---

## What the library does not do

It consumes model outputs; it does not fit the models. Your technical premiums and price elasticity estimates need to come from a GLM, GBM, or DML estimation step. The optimiser is an offline rate strategy tool — not a real-time quote engine. For individual-level pricing at point of quote, Radar Live and Earnix serve that function.

The library also does not validate that your elasticity estimates are correct. The ENBP constraint is enforced per-policy using the `enbp` array you supply, and the solver produces a JSON audit trail per run for FCA scrutiny. But if the elasticity inputs are systematically biased, the optimiser will faithfully find the optimal solution to the wrong problem.

One practical note: for firms implementing ENBP-constrained optimisation for the first time on a legacy book, reconstructing per-policy ENBP from policies priced under previous systems is typically a 2-3 month data engineering exercise before the optimiser can run. The library assumes ENBP is a clean input array — it does not help you build it.

---

## Verdict

Use constrained rate optimisation if:
- You have a renewal book with meaningful elasticity variation between PCW-acquired and direct customers — the profit advantage scales with elasticity dispersion across the portfolio
- You have FCA PS21/11 ENBP obligations and are currently enforcing the ceiling post-hoc on the optimised output rather than inside the constraint set — you are likely violating the spirit of the regulation and your solver is not finding the true optimum
- You want to take the efficient frontier to a commercial director as a quantified trade-off rather than a single scenario

Do not expect it to:
- Improve on a flat loading if your elasticity estimates are all the same — the optimiser cannot create value from uniform inputs
- Replace the demand model — accurate elasticity estimates are the most important input and the hardest to obtain
- Handle new business pricing at scale in real time — this is a batch tool for renewal strategy, not a quote engine

The technique works. The profit advantage on the benchmark (£4,000–8,000 on 2,000 policies, <1 second to solve) is large enough to be worth the engineering effort of building the data pipeline. The honest caveat is that the quality of the elasticity inputs determines the quality of the output — garbage in, optimally priced garbage out.

```bash
uv add insurance-optimise
```

Source and benchmarks at [GitHub](https://github.com/burning-cost/insurance-optimise). Start with `benchmarks/benchmark.py` for the head-to-head against uniform rate change, then `benchmarks/benchmark_pareto.py` for the Consumer Duty Pareto surface.

- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) — the full library post covering the DML elasticity pipeline that feeds the optimiser in this benchmark
- [Constrained Rate Optimisation and the Efficient Frontier](/2026/02/21/constrained-rate-optimisation-efficient-frontier/)
- [Does DML Causal Inference Actually Work for Insurance Pricing?](/2026/03/25/does-dml-causal-inference-actually-work/)
