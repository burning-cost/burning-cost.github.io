---
layout: post
title: "ConvexRiskReinsuranceOptimiser: Optimal Multi-Line Treaty Design Without the Guesswork"
date: 2026-04-02
categories: [libraries]
tags: [reinsurance, optimisation, convex-duality, solvency-ii, cvar, de-finetti, insurance-optimise, lloyds, arXiv-2603.00813]
description: "insurance-optimise v0.7.0 adds ConvexRiskReinsuranceOptimiser — analytically optimal multi-line reinsurance contracts under CVaR or variance constraints, based on Shyamalkumar & Wang (2026)."
author: burning-cost
---

You have five lines of business. Each has a different reinsurance loading — motor at 15%, property at 22%, liability at 30%. You have a Solvency II SCR target to hit after ceding. Which risks do you cede, how much, and in what structure?

The standard answer is: optimise numerically, hope the solver converges, sense-check against intuition, iterate. The problem is that numerical optimisation over treaty structures is slow, opaque, and gives you no intuition about why it chose what it chose.

[insurance-optimise v0.7.0](https://pypi.org/project/insurance-optimise/) adds `ConvexRiskReinsuranceOptimiser`, implementing the results from Shyamalkumar & Wang (2026) ([arXiv:2603.00813](https://arxiv.org/abs/2603.00813)). The paper solves this problem analytically. No black-box optimisation. Just bisection on a single scalar, and closed-form contracts for each line.

---

## What the paper shows

The De Finetti problem asks: find admissible contracts R_i (one per risk line) that minimise total ceded premium sum_i beta_i * E[R_i] subject to rho(retained aggregate) <= c, where beta_i is the reinsurer's loading and rho is a convex risk measure.

Classical treatments assume independent risks, identical loadings, and a variance constraint. The Shyamalkumar & Wang result handles all three generalisations at once: dependent risks, heterogeneous loadings, and CVaR or variance as the risk measure.

The theoretical engine is a convex duality theorem. The constrained optimisation reduces to a one-parameter family indexed by a Lagrange multiplier lambda*. Find lambda* and you have the full contract structure. For CVaR, the optimal contract for each line is:

```
R_i* = (S - cumulative_capacity_i - q)_+ wedge X_i   [if beta_i < lambda/(1 - alpha)]
R_i* = 0                                               [if beta_i >= lambda/(1 - alpha)]
```

where S is the aggregate loss, q is a threshold, and the risks are processed in ascending order of beta_i. The variance case has an analogous closed form involving a fixed-point sigma.

Three things are immediately actionable from this structure:

**Cheapest risk cedes first.** Lines are ordered by loading. The cheapest reinsurance covers the first excess layer; the next cheapest covers the next layer. Lines above the threshold K = lambda/(1-alpha) do not cede at all — they are too expensive relative to the constraint tightness.

**The threshold K encodes the constraint.** When the budget is loose (large c), K is low, more lines cede, and the premium cost is low. When the budget is tight, K rises, lines drop out, and you pay more for the remaining cession. The entire efficient frontier follows from sweeping lambda*.

**Contract shape is neither quota-share nor stop-loss.** Each R_i* is a stop-loss on the residual aggregate after cheaper lines have absorbed their layer. This is the provably optimal structure under the risk measure constraint — not a convenience assumption.

---

## The API

```python
import numpy as np
from insurance_optimise import ConvexRiskReinsuranceOptimiser, RiskLine

risks = [
    RiskLine(name="motor",     expected_loss=5_000, variance=8_000_000, safety_loading=0.15),
    RiskLine(name="property",  expected_loss=3_000, variance=4_000_000, safety_loading=0.22),
    RiskLine(name="liability", expected_loss=1_500, variance=2_500_000, safety_loading=0.30),
]

# Pass pre-simulated samples from your collective risk model (recommended)
# or let the library simulate via multivariate lognormal approximation
rng = np.random.default_rng(42)
samples = rng.lognormal(
    mean=np.log([5000, 3000, 1500]),
    sigma=0.4,
    size=(50_000, 3)
)

opt = ConvexRiskReinsuranceOptimiser(
    risks=risks,
    risk_measure='cvar',   # 'cvar' or 'variance'
    alpha=0.995,           # 1-in-200 year; aligns with Solvency II standard formula
    budget=12_000,         # CVaR_99.5 constraint on retained aggregate
    aggregate_loss_samples=samples,
)

result = opt.optimise()
print(result.summary())
```

The `summary()` method returns a Polars DataFrame with per-line contract terms:

| name | safety_loading | expected_ceded_loss | ceded_premium | cession_rate | ceded |
|------|---------------|--------------------|--------------:|-------------:|-------|
| motor | 0.15 | 1,847 | 277 | 37% | True |
| property | 0.22 | 892 | 196 | 30% | True |
| liability | 0.30 | 0 | 0 | 0% | False |

Motor cedes the most at 37% — cheapest loading, first in line. Liability cedes nothing — at 30% loading, it is above the K threshold given the budget constraint. This is exactly what the theory predicts, and it is verifiable from the `lambda_star` in the result:

```python
K = result.lambda_star / (1 - 0.995)   # = lambda* / (1 - alpha)
# Lines with safety_loading >= K get R_i* = 0
```

---

## The efficient frontier

The budget c is the lever. Tighten the SCR constraint and you pay more ceded premium for fewer retained losses. The `.frontier()` method sweeps this automatically:

```python
df = opt.frontier(n_points=40)
# Columns: budget, total_ceded_premium, retained_risk, n_lines_ceded,
#          ceded_premium_motor, ceded_premium_property, ceded_premium_liability
```

The output is a Pareto front of (ceded_premium, retained_CVaR). Plot it and you get the reinsurance efficiency curve — how much SCR reduction each additional pound of premium buys. The curve is strictly convex: the first £100 of premium buys a large SCR reduction; the last £100 to reach a very tight constraint buys very little.

Equally useful: the `n_lines_ceded` column shows where lines drop in or out as the budget tightens. At loose budgets, only motor cedes. At tighter budgets, property joins. At very tight budgets, all three cede. This is the loading-order prediction of the theory made concrete.

---

## Sensitivity analysis

The `sensitivity()` method varies one parameter and returns a sweep:

```python
# How does total ceded premium change if motor's loading rises from 10% to 25%?
df = opt.sensitivity(
    param='loading_motor',
    values=[0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
)
# Columns: param_value, total_ceded_premium, retained_risk, n_lines_ceded, lambda_star
```

As motor's loading rises, the K threshold crosses it and motor drops out — at which point property takes on more of the cession burden, and total cost rises while retained risk also rises slightly (the portfolio was relying on cheap motor capacity). The sensitivity output makes the crossover loading explicit rather than leaving it as a number to guess.

The budget and alpha parameters work the same way:

```python
# Cost of tightening the SCR target from 1-in-100 to 1-in-500
df = opt.sensitivity(param='alpha', values=[0.99, 0.995, 0.998, 0.999])
```

---

## What this replaces

The common alternative approaches, and why they are worse:

**Per-line stop-loss treaties, sized independently.** Optimising each line separately ignores dependence. If motor and property are correlated, a joint cession structure that treats the aggregate can be cheaper for the same CVaR reduction. The cross-line structure in R_i* exploits this automatically.

**Uniform quota-share.** Simple but provably suboptimal under heterogeneous loadings. The theory shows the optimal cession is stop-loss-shaped and loading-ordered, not a flat fraction. Quota-share is easy to explain but you pay for the simplicity in excess ceded premium.

**Numerical portfolio optimisation.** SciPy's SLSQP or similar can search over treaty parameters, but the search space is high-dimensional, convergence depends on starting points, and the result is a number with no theoretical interpretation. The duality approach here gives you the answer analytically, plus lambda* as an interpretable shadow price.

---

## Using your own loss samples

The most important parameter is `aggregate_loss_samples`. The internal fallback (multivariate lognormal via Gaussian copula) is a useful sanity check but is not a substitute for samples from your actual collective risk model.

If you have a fitted Tweedie GLM per line, simulate the aggregate:

```python
# Example: 3-line collective risk simulation
# Assume your model produces (50000, 3) array of aggregate annual losses
samples = your_collective_risk_model.simulate(n=50_000, seed=0)

opt = ConvexRiskReinsuranceOptimiser(
    risks=risks,
    risk_measure='cvar',
    alpha=0.995,
    budget=your_scr_target,
    aggregate_loss_samples=samples,
)
```

The CVaR quantile estimation is empirical — no distributional assumption. The variance solver does not need samples at all (it uses E[X_i], Var(X_i), Cov(X_i, X_j) analytically), but for practical use CVaR is almost always the right choice for Solvency II work.

---

## Relationship to RobustReinsuranceOptimiser

The library now has two reinsurance optimisers. They solve different problems.

`RobustReinsuranceOptimiser` (v0.5.0, [arXiv:2603.25350](https://arxiv.org/abs/2603.25350)) solves a dynamic control problem: given surplus evolving over time, what proportional cession fraction maximises firm value under ambiguity about loss parameters? The result is a cession schedule indexed by surplus level — more reinsurance near ruin, less when you are safe. Use this when the intertemporal problem matters and parameter uncertainty is the main concern.

`ConvexRiskReinsuranceOptimiser` (v0.7.0, this release) solves a static portfolio problem: given a mixed portfolio and a capital constraint, what is the cheapest treaty structure? The result is a per-line contract, not a dynamic cession fraction. Use this for treaty design — specifically for answering "how should we structure our outwards programme this year given our SCR target and negotiated loadings?"

The two are complementary. A Lloyd's syndicate might use `ConvexRiskReinsuranceOptimiser` to design the treaty structure and `RobustReinsuranceOptimiser` to size the proportional element within each line dynamically.

---

## What this does not do

`ConvexRiskReinsuranceOptimiser` does not handle non-proportional excess-of-loss treaties with per-occurrence limits and reinstatement provisions. The contract structure R_i*(X_1,...,X_n) is more general than quota-share but is still applied to aggregate losses per line, not to individual large losses. Cat XL sizing is a different problem.

It also does not price the reinsurance dynamically — beta_i is taken as an input. The optimiser tells you the cost-optimal structure given negotiated loadings, not what loadings you should accept.

The library is on [PyPI](https://pypi.org/project/insurance-optimise/) and the source is on [GitHub](https://github.com/burning-cost/insurance-optimise). The paper is at [arXiv:2603.00813](https://arxiv.org/abs/2603.00813).
