---
layout: post
title: "Robust Discrete Pricing Optimisation via Knapsack"
date: 2026-03-26
categories: [techniques]
tags: [optimisation, discrete-pricing, knapsack, robustness, GIPP, PS21-5, FCA, portfolio-pricing, insurance-optimise, python]
description: "A new paper from ETH Zürich (arXiv:2603.18653) frames the conversion from technical price to commercial premium as a Multiple-Choice Knapsack Problem. Under 1% revenue cost for robust margin protection."
---

Most insurance pricing optimisation literature assumes you can set a continuous price. You cannot. Your rating engine has discrete bands. Your PCW has pricing increments. Your underwriting system rounds to whole pounds. The neat gradient-descent story breaks down when the decision space is a menu rather than a manifold.

A paper by Zi Yuan Eric Shao from ETH Zürich (arXiv:2603.18653, March 2026) takes this seriously. It frames the problem of converting a technical price to a commercial offering  -  selecting one price point from a finite menu, for each product, across a portfolio  -  as a Multiple-Choice Knapsack Problem (MCKP). The result handles discrete pricing, margin constraints, regulatory closeness requirements, and demand uncertainty, in under one second for portfolios of 500 products.

---

## Why discrete pricing is harder than it looks

The continuous pricing problem is well understood. Maximise expected revenue subject to a margin constraint: the first-order conditions give you Dorfman–Steiner equations, and if the demand functions are well-behaved you have a gradient to follow.

Discretise it and everything changes. Even with only three allowed prices per product and 100 products, the combinatorial search space is 3^100. A greedy approach  -  pick the best price for each product independently  -  ignores the coupling between products through the margin constraint. The overall margin depends on all your choices simultaneously, so you cannot optimise product-by-product.

The standard workaround is to pretend the problem is continuous, solve it, and round. The rounding step is where things go wrong. For portfolios near the margin constraint boundary, rounding in the wrong direction on a handful of products can breach the constraint. For heterogeneous portfolios  -  different weights, different menus, different demand sensitivities  -  the rounding error is unpredictable.

The MCKP approach avoids the round-and-pray pattern. It converts the problem exactly and solves the LP relaxation with a structural guarantee on the integrality gap.

---

## The MCKP formulation

The setup is a portfolio of n products. For each product i, you have:

- A finite price menu: x_{i,1}, ..., x_{i,m_i} (e.g. five £10 increments around the technical price)
- A demand forecast ĝ_i(x_{i,j}): expected volume at each price
- A weight ω_i: policy count, exposure, or notional
- A reference price a_i: the technical premium
- A fairness band σ_i: maximum permitted deviation from the reference

Objective: choose exactly one price per product to maximise total revenue:

```
N(x) = Σ_i ω_i × x_{i,j(i)} × ĝ_i(x_{i,j(i)})
```

subject to a margin constraint  -  revenue divided by expected loss cost must meet a target Δ:

```
N(x) / D(x) ≥ Δ
```

and a closeness constraint for each product:

```
|x_i - a_i| / a_i ≤ σ_i
```

The closeness constraint is where regulatory requirements on pricing proximity enter the model. In the UK context, the GIPP rules (General Insurance Pricing Practices, FCA PS21/5) require that renewal prices do not exceed the equivalent new business price. A pricing team using this framework would set the reference price a_i to the ENBP ceiling and σ_i to enforce the regulatory band. The paper does not reference GIPP explicitly  -  it is general discrete pricing theory  -  but the constraint maps directly onto it.

The margin constraint couples all n products. That coupling is what makes greedy approaches fail.

**The MCKP reduction.** For each product i, define a baseline choice j*(i) as the option with the highest margin slack. For each alternative option j, compute:

```
v_{ij} = ω_i × x_{i,j} × ĝ_i(x_{i,j})    # revenue contribution
s_{ij} = v_{ij} - Δ × d_{ij}               # margin slack (d_{ij} is loss cost at price j)
c_{ij} = s_{i,j*(i)} - s_{ij}              # cost of deviating from baseline
```

The margin constraint becomes:

```
Σ_i c_{i,j(i)} ≤ S_max
```

where S_max is the total available margin slack. This is exactly the MCKP capacity constraint. You choose one option per item, cannot exceed capacity, maximise total value. Standard MCKP.

---

## Solving the LP relaxation exactly

The LP relaxation is tractable via upper-hull geometry. For each product i, the feasible (cost, value) pairs form a set of points in two dimensions. The upper convex hull gives the options that are ever LP-optimal  -  points strictly below the hull are dominated. Upper-hull filtering discards them before solving.

The remaining problem  -  a convex combination of adjacent hull points per product, subject to total cost ≤ S_max  -  is solved by a greedy algorithm filling capacity along hull segments. Exact for the LP relaxation. Runtime: O(nm log m).

The integrality gap  -  the revenue loss from rounding back to one actual price per product  -  is bounded by a single hull jump: the maximum value difference between adjacent hull points across all products. This bound declines as O(1/n). On synthetic portfolios with n ≥ 100 and m = 50 price options per product, the relative gap stays below 0.15% of optimal revenue. Runtime for the full solve: under one second at n = 500.

---

## Robustness to demand uncertainty

Demand forecasts are estimates. They carry elasticity uncertainty that a DML model can quantify but not eliminate. The paper's second contribution makes the margin constraint robust to that uncertainty.

The framework is Bertsimas–Sim budget-of-uncertainty (2004). Each product's demand can deviate from its point estimate by a worst-case amount t_{ij}. The Γ-budget says: at most Γ products can simultaneously realise their worst-case deviation. This is conservative without being maximally conservative  -  you are not assuming every product hits its worst-case simultaneously.

The robust margin constraint:

```
Σ_i s_i(x_i) - β(x, Γ) ≥ 0
```

where β(x, Γ) sums the Γ largest uncertainty penalties across chosen options. This term couples the constraint non-separably.

The key structural result is a parametric dual decomposition: for any fixed threshold θ ≥ 0, the robust constraint separates into independent per-product subproblems:

```
Σ_i [s_{ij} - max(0, |t_{ij}| - θ)] ≥ Γθ
```

The algorithm enumerates over at most nm candidate θ values, solving a separable MCKP at each. This reduces a coupled, non-separable robust problem to a one-dimensional search over a finite breakpoint set.

**The cost of robustness.** Across synthetic portfolios with n ∈ {30, ..., 500}, robust margin protection costs less than 1% of nominal revenue. That is the number worth holding on to. You are paying under 1% of revenue to guarantee your margin constraint holds even when up to Γ products simultaneously hit worst-case demand. For any pricing team carrying elasticity uncertainty  -  which is every pricing team  -  that is a reasonable price.

---

## What this means for `insurance-optimise`

The [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) library solves the continuous version: policy-level SLSQP with analytical Jacobians, FCA ENBP enforcement, loss ratio and retention constraints. Its limitation is the continuous price space assumption.

UK pricing teams do not output continuous prices. They output rated premiums from rating engines with banded factors. The commercial adjustment that `insurance-optimise` optimises is typically applied at segment level, where the segment premium is already a discrete choice from a finite band.

The MCKP paper provides the theoretical grounding for a discrete variant: instead of optimising a continuous multiplier, you select from a per-segment price menu. The reduction is cleaner at segment level (n = 30–500 rating cells) than at individual policy level. We think the natural integration point is a `DiscretePortfolioOptimiser` that accepts a per-segment price menu alongside the existing constraint configuration, with the robustness parameter Γ linked to elasticity confidence intervals from [`insurance-causal`](https://github.com/burning-cost/insurance-causal).

This is not yet in the library.

---

## Limitations

**Demand model quality dominates.** The MCKP machinery is exact given the demand forecasts. The forecasts are not exact. A 10% elasticity estimation error produces a 10% error in optimal prices regardless of how clean the optimisation is. The robustness framework addresses this, but Γ itself requires calibration.

**Menu construction is implicit.** The paper takes the price menu as given. Deciding menu spacing, number of points, and bounds relative to technical is a user problem, not an algorithm problem. A poorly designed menu produces technically optimal solutions that are commercially useless.

**n ≤ 500 in the empirical tests.** UK personal lines portfolios have far more rating cells than 500 if you enumerate the full factor interaction space. The O(1/n) gap result suggests performance improves with n, but the paper has not demonstrated n = 5,000. Extrapolating the theoretical guarantees is fine. Assuming the runtime scales is not.

**No individual-level guarantees.** MCKP operates at segment level. Per-policy ENBP compliance  -  which FCA PS21/5 requires  -  needs propagating separately, with per-policy feasibility checks. The segment-level optimum is not necessarily feasible at individual policy level once ENBP is enforced policy-by-policy.

---

## The reference

Zi Yuan Eric Shao, "Robust Discrete Pricing Optimization via Multiple-Choice Knapsack Reductions," arXiv:2603.18653, March 2026. Department of Mathematics, ETH Zürich.

The paper is a clean piece of operations research applied to a real pricing problem. It does not claim to solve UK insurance pricing  -  it solves a cleanly stated optimisation problem. The gap between that and production is the domain engineering: connecting the formulation to rating engines, demand models, regulatory constraints, and segment definitions. That gap is the interesting part.

---

Related:

- [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/)
- [Constrained Rate Optimisation and the Efficient Frontier](/2026/02/21/constrained-rate-optimisation-efficient-frontier/)
- [Does Constrained Rate Optimisation Actually Work?](/2026/03/29/does-constrained-rate-optimisation-actually-work/)
- [What FCA EP25/2 Actually Shows](/2026/03/25/fca-ep25-2-gipp-market-outcomes-cost-push-not-regulation/)
