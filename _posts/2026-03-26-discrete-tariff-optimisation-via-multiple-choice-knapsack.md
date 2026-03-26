---
layout: post
title: "Discrete Tariff Optimisation via Multiple-Choice Knapsack"
date: 2026-03-26
categories: [pricing, optimisation]
tags: [tariff, discrete-pricing, knapsack, MCKP, bertsimas-sim, robustness, GIPP, PS21-5, ENBP, insurance-optimise, python, combinatorial-optimisation]
description: "Continuous optimisers produce real-valued outputs. Commercial tariffs are discrete grids. Shao (2026) gives a polynomial-time exact algorithm for the gap — no MIP solver needed, integrality gap under 0.15% at n=100, and a Bertsimas-Sim robustness certificate at under 1% revenue cost."
---

Every pricing team runs into the same wall. You build a constrained portfolio optimiser, it tells you the optimal price for each rating cell, and then someone in the tariff team asks: "Which band does that map to?" The continuous optimum is £347.83. Your tariff grid goes £340, £350, £360. You round. You recheck the loss ratio constraint. It no longer holds. You round differently. You iterate.

This is not a minor inconvenience. Naïve rounding on a 100-cell tariff with a tight margin constraint is genuinely unreliable — and as the number of rating cells grows, the number of ways rounding can break your constraints grows faster.

Shao (2026), arXiv:2603.18653, frames this as a combinatorial optimisation problem and shows it reduces to the Multiple-Choice Knapsack Problem (MCKP): a classical structure with a known polynomial-time LP relaxation and a provable O(1/n) integrality gap. The paper is from ETH Zürich. We think it is correct, well-structured, and practically useful. This post is our explanation of it.

---

## The problem stated precisely

You have $$n$$ rating cells (or product segments), indexed $$i = 1, \ldots, n$$. Each cell has a finite price menu $$\mathcal{X}_i = \{x_{i,1}, \ldots, x_{i,m}\}$$ — the $$m$$ tariff grid points available for that cell. You must select exactly one price per cell. Let $$g_i(x_i)$$ be the demand function — expected volume at price $$x_i$$. The objective is to maximise nominal revenue:

$$N(x) = \sum_i w_i \, x_i \, g_i(x_i)$$

subject to a portfolio margin constraint:

$$S(x) = \sum_i w_i \left[ x_i \, g_i(x_i) - c_i \, g_i(x_i) \right] \geq M_{\text{target}}$$

and a fairness band per cell:

$$\frac{|x_i - a_i|}{a_i} \leq \sigma_i \quad \text{for all } i$$

where $$a_i$$ is a reference price (technical price, or prior-year price for renewals) and $$\sigma_i$$ is the permitted proportional deviation.

Binary decision variables $$z_{ij} \in \{0,1\}$$ with $$\sum_j z_{ij} = 1$$ select one price per cell. The search space is $$m^n$$. For $$n = 100$$ cells and $$m = 50$$ grid points: $$50^{100}$$. Exhaustive search is not an option.

---

## The MCKP reduction

The key step is a baseline-slack transformation that converts the margin constraint into a standard knapsack capacity constraint — which is necessary because MCKP requires non-negative item weights.

Define the baseline revenue contribution for item $$i$$:

$$B_i = w_i \, a_i \, g_i(a_i)$$

This is what cell $$i$$ contributes to revenue when priced at its reference. Now define, for each item $$i$$ and choice $$j$$:

$$v_{ij} = w_i \, x_{ij} \, g_i(x_{ij}) - B_i$$

$$c_{ij} = \max\!\left(0,\; B_i - w_i \, x_{ij} \, g_i(x_{ij})\right)$$

The value $$v_{ij}$$ is how much revenue this choice contributes relative to the baseline — it can be negative if pricing below reference. The cost $$c_{ij}$$ measures how far a choice falls below baseline, and is zero for any choice that beats the baseline. Non-negativity of $$c_{ij}$$ is guaranteed by construction.

Set $$S_{\max} = \sum_i B_i - M_{\text{target}}$$. The margin constraint becomes:

$$\sum_{i,j} c_{ij} \, z_{ij} \leq S_{\max}$$

Combined with $$\sum_j z_{ij} = 1$$ and $$z_{ij} \in \{0,1\}$$, the problem is now:

$$\max \sum_{i,j} v_{ij} \, z_{ij} \quad \text{s.t.} \quad \sum_{i,j} c_{ij} \, z_{ij} \leq S_{\max}, \quad \sum_j z_{ij} = 1 \; \forall i, \quad z_{ij} \in \{0,1\}$$

This is the canonical Multiple-Choice Knapsack Problem. No approximation has been made: the transformation is exact. The reason it works is that the margin constraint is ratio-type — comparing aggregate performance to a baseline. Subtracting the baseline makes the costs non-negative and converts the constraint to a standard budget form.

---

## LP relaxation: O(nm log m) exact algorithm

Relax the integrality constraint: allow $$z_{ij} \in [0,1]$$. The LP relaxation is solvable exactly in $$O(nm \log m)$$ time, exploiting the MCKP structure.

**Theorem 5 (Shao, 2026):** The optimal LP solution always lies on the upper convex hull of each item's $$(c_{ij}, v_{ij})$$ point set in $$\mathbb{R}^2$$. At most one item is fractionally mixed in any optimal LP solution.

The algorithm follows directly:

1. For each item $$i$$, construct the upper convex hull $$H_i$$ of the points $$\{(c_{ij}, v_{ij}) : j = 1, \ldots, m\}$$. Discard dominated points — empirically, this filters 80–90% of menu choices.
2. Collect all hull *segments* across all items. Each segment connects two adjacent hull vertices and has a slope $$dv/dc$$ (value per unit of capacity consumed).
3. Sort segments globally by slope in decreasing order.
4. Greedily fill capacity $$S_{\max}$$: allocate full items ($$z_{ij} = 1$$) in slope order until the next item would exceed capacity.
5. Fractionally allocate the straddling item. This is the unique fractional component.

The hull construction dominates: $$O(m \log m)$$ per item, $$O(nm \log m)$$ total. The greedy fill is linear. You do not need a general LP solver.

---

## The integrality gap

How much do you lose by rounding the single fractional item to an integer?

**Proposition 9 (Shao, 2026):** The additive gap is bounded by the maximum adjacent-hull value jump across all items:

$$\text{OPT}_{\text{LP}} - \text{OPT}_{\text{IP}} \leq \Delta V_{\max}$$

where $$\Delta V_{\max}$$ is the largest revenue difference between two adjacent hull vertices across all $$n$$ items. The gap is bounded by one rounding step on one item.

**Corollary 10:** Under uniform item value boundedness (all $$|v_{ij}| \leq V$$), the relative gap satisfies:

$$\frac{\text{OPT}_{\text{LP}} - \text{OPT}_{\text{IP}}}{\text{OPT}_{\text{LP}}} = O(1/n)$$

The intuition: total LP value grows as $$O(nV)$$, while the single-item rounding loss is bounded by a constant. On synthetic tests with $$n = 100$$ cells and $$m = 50$$ price steps, the relative gap is below 0.15%. For $$n = 50$$ cells, below 0.30%. These are well below any business threshold.

This matters because it replaces the current situation — naïve rounding with no gap certificate — with a formally bounded answer. You know how far from optimal you are.

---

## Robustness: Bertsimas-Sim Gamma-budget

The demand function $$g_i(x_i)$$ is estimated from data and is uncertain. Shao (2026) handles this using the Bertsimas-Sim (2004) Gamma-budget model: demand at each cell can deviate from its estimate, but at most $$\Gamma$$ cells deviate simultaneously. This is less conservative than worst-case over all $$2^n$$ combinations, and more tractable.

The worst-case margin under $$\Gamma$$ deviations is:

$$S_{\text{robust}}(x) = \sum_i s_i(x_i) - \beta(x, \Gamma) \geq 0$$

where $$\beta(x, \Gamma) = \sum_{k=1}^{\Gamma} |t_i(x_i)|_{(k)}$$ sums the $$\Gamma$$ largest absolute demand deviations. The key result (**Theorem 6**) is the parametric dual decomposition:

$$\beta(x, \Gamma) = \min_{\theta \geq 0} \left\{ \Gamma\theta + \sum_i \max\!\left(0, |t_i(x_i)| - \theta\right) \right\}$$

For any fixed $$\theta$$, the penalty is *separable across items*. This converts the $$n$$-coupled robust constraint into a one-dimensional enumeration. The algorithm:

1. Enumerate $$\theta$$ over the finite set $$\mathcal{B}$$ of candidate breakpoints — the sorted unique values of $$|t_i(x_{ij})|$$ across all $$i, j$$. Empirically, fewer than 50 breakpoints matter.
2. For each $$\theta$$: compute modified costs $$c_{ij}^{\theta} = c_{ij} + \max(0, \delta_{ij} - \theta)$$, then solve the MCKP via upper-hull greedy as before.
3. Verify robust feasibility using the dual certificate.
4. Return the best robustly feasible solution.

The paper recommends $$\Gamma \approx \sqrt{n}$$ as the default — so $$\Gamma = 7$$ for $$n = 50$$ cells, $$\Gamma = 22$$ for $$n = 500$$. This mirrors the practical guidance for Bertsimas-Sim in other domains. The empirical robustness cost across all synthetic test instances is under 1% of nominal revenue. You get formal uncertainty protection essentially for free.

---

## UK GIPP mapping

Under the FCA's General Insurance Pricing Practices rules (PS21/5, effective January 2022), renewal prices must not exceed the equivalent new business price (ENBP). The Shao fairness band maps to this directly:

- $$a_i = \text{ENBP}_i$$ — the PS21/5 reference for renewal cells
- $$\sigma_i^{\text{upper}} = 0$$ — the GIPP constraint is one-sided: renewals may not exceed ENBP
- $$\sigma_i^{\text{lower}}$$ — set by actuarial policy (technical price floor)

So for renewal cells, the price menu is pre-filtered to $$[\text{tc}_i, \text{ENBP}_i] \cap \text{grid}$$. MCKP then selects from this feasible set. The filtering is computationally free — it happens before the optimisation step. For new business cells where GIPP does not apply, the band can be symmetric: $$\pm\sigma_i$$ around the technical price.

One caveat the paper does not fully address: if MCKP operates at cell level (say $$n = 100$$ rating cells), it selects one price per cell. Individual policy ENBP compliance still needs to be checked post-hoc, because the ENBP varies between policies within a cell. The MCKP band is a cell-average approximation. For tightly defined cells with small within-cell ENBP variation this is fine; for broad cells it is not.

---

## How this fits with insurance-optimise

Our existing [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) library uses SLSQP in continuous multiplier space. The two approaches operate at different problem scales and are complementary, not competing.

| | insurance-optimise (SLSQP) | Shao (2026) MCKP |
|---|---|---|
| Granularity | Policy level ($$N \sim 10\text{k}$$–$$100\text{k}$$) | Rating cell level ($$n \sim 30$$–$$500$$) |
| Price space | Continuous $$\mathbb{R}$$ | Discrete finite menu |
| Optimality | Local SLSQP convergence | LP bound + $$O(1/n)$$ gap |
| Robustness | Scenario mode (empirical) | Bertsimas-Sim certificate |
| Runtime | $$O(N^2)$$ per iteration | $$O(nm \log m)$$ per $$\theta$$ |

The natural pipeline is two-stage. Stage 1: run `PortfolioOptimiser` at policy level to find the continuous optimal multiplier surface. Stage 2: aggregate to rating cells, construct the discrete price menu from the tariff grid, and run MCKP to select one commercial price per cell — the nearest feasible integer solution with a provable gap certificate.

Stage 2 is what is currently missing. The gap it fills: SLSQP gives you a real-valued optimum that must be rounded before implementation. Naïve rounding can violate the margin constraint. MCKP gives you the best tariff-grid selection with guaranteed margin satisfaction and a known distance from the LP bound.

---

## Implementation

There is no dedicated Python MCKP library. The specialised algorithm is approximately 700 lines of NumPy — no MIP solver required.

The core components: menu filtering and the baseline-slack transformation (~60 lines), upper-hull construction via `scipy.spatial.ConvexHull` or a direct $$O(m \log m)$$ scan (~80 lines), the greedy LP solver (~70 lines), integer rounding with feasibility repair (~60 lines), and the theta-enumeration loop for robustness (~100 lines). Using a general MIP solver (OR-Tools, HiGHS via `scipy.optimize.milp`) is an alternative but is 10–100x slower for this structure because it does not exploit MCKP's upper-hull property.

The proposed `DiscretePortfolioPricer` class takes an $$n \times m$$ price menu array, an $$n \times m$$ demand array (evaluated at each menu point), reference prices, the fairness band, a margin target, and the Gamma budget. It returns the selected price index per cell, expected revenue, the LP gap certificate, and a feasibility flag. A `from_continuous_result()` classmethod would accept an `OptimisationResult` from `PortfolioOptimiser` and construct the discrete problem from rating-cell aggregates.

---

## Honest caveats

Three limitations worth stating plainly.

**The menu is exogenous.** The paper treats the price grid as a given input. Tariff design — how many bands to use, where to place them, how graduation should work — is itself an optimisation problem that MCKP does not address. MCKP tells you the best selection *from the menu you give it*.

**The O(1/n) gap requires well-bounded item values.** The formal guarantee assumes uniform boundedness across items. If one rating cell has disproportionately large revenue weight, the gap bound degrades. In practice, for balanced cell structures this is not a concern; for heavily skewed books (say, one cell represents 40% of GWP), check the gap certificate empirically.

**Demand model quality dominates.** The integrality gap is $$O(1/n)$$ relative to the LP optimum for *your demand model*. If the demand model is poorly calibrated — a real risk in UK personal lines for thin segments or volatile perils — the nominal optimum is already wrong. MCKP gives you the best grid selection given your inputs; it cannot compensate for bad inputs. Invest in the demand model first.

---

## What is genuinely new

The Bertsimas-Sim Gamma-budget model (2004) and the upper-hull LP algorithm for MCKP (Pisinger, 1995) are established operations research. What Shao (2026) adds is the baseline-slack transformation that converts an insurance margin constraint — which is ratio-type and would otherwise produce negative costs — into the non-negative-cost MCKP form. This connection has not appeared in the actuarial literature before. The $$O(1/n)$$ relative gap with explicit additive bound gives a formal quality certificate that heuristic rounding lacks. The Gamma-budget robust variant adapted to the pricing context with a bounded robustness cost rounds out a clean, usable result.

The paper does not compare against industry benchmarks. We do not know how current UK pricing teams handle the rounding problem in practice — likely a mix of iterative adjustment and local search, with no formal guarantee. If that description fits your process, the MCKP approach is straightforwardly better.

```bash
uv add insurance-optimise
```

Source: [github.com/burning-cost/insurance-optimise](https://github.com/burning-cost/insurance-optimise)

---

- [Does Constrained Rate Optimisation Actually Work?](/2026/03/29/does-constrained-rate-optimisation-actually-work/) — the continuous SLSQP stage that feeds into the discrete step
- [Does DML Causal Inference Actually Work for Insurance Pricing?](/2026/03/25/does-dml-causal-inference-actually-work/) — demand model quality determines both stages
- [Model Value in Pounds: Translating Gini Improvement to Loss Ratio](/2027/01/14/model-value-in-pounds-translating-gini-improvement-to-loss-ratio/) — what an improved demand model is worth before you optimise over it
