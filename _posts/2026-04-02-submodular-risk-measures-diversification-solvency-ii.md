---
layout: post
title: "Submodular Risk Measures: The Mathematics of When Diversification Stops Working"
date: 2026-04-02
author: burning-cost
categories: [research]
tags: [solvency-ii, capital-modelling, risk-measures, expected-shortfall, diversification, insurance-optimise, var, internal-model, arXiv-2603.01232, actuarial]
description: "Wang & Yu (arXiv:2603.01232) characterise which risk measures are submodular — mathematically encoding that diversification benefits are real but bounded. ES is submodular; VaR is not. Capital modellers and internal model teams should understand why this matters."
permalink: /2026/04/02/submodular-risk-measures-diversification-solvency-ii/
---

Every internal model team puts a diversification benefit on their Solvency II SCR. The number typically comes from a correlation matrix applied to capital modules, producing a combined capital requirement that is less than the sum of the parts. The mathematical justification for this is almost never stated precisely. It is assumed, implicitly, that risk aggregation follows some vaguely coherent set of rules.

Wang & Yu's paper on submodular risk measures (arXiv:2603.01232, March 2026) makes those rules precise. Their finding is stark: most risk measures used in Solvency II internal models are either submodular or they are not, and whether they are determines whether your diversification assumptions are mathematically defensible. Expected Shortfall passes. Value at Risk fails, predictably and in ways that show up in observed data.

---

## What submodularity means

Submodularity is a property from combinatorial optimisation. A set function f is submodular if the marginal gain from adding an element decreases as the set grows. In the risk measure context: a risk measure ρ is submodular if

    ρ(X ∨ Y) + ρ(X ∧ Y) ≤ ρ(X) + ρ(Y)

where X ∨ Y and X ∧ Y are the pointwise maximum and minimum of two random loss variables. This is the formal expression of a natural intuition: the gain from combining two separate diversifying risks decreases as your portfolio grows. You cannot indefinitely add new business lines and continue achieving the same marginal capital reduction.

This is distinct from — and stronger than — subadditivity, which says only that ρ(X + Y) ≤ ρ(X) + ρ(Y). Subadditivity is a property of how risks aggregate when added. Submodularity is a property of how capital requirements respond to the *structural composition* of the portfolio — which risks you put together, not just their sum.

The practical difference matters. A risk measure can be subadditive without being submodular. Submodularity implies that diversification benefits are real, bounded, and diminish at the margin.

---

## The main results

Wang & Yu characterise submodularity across four major classes of convex risk measures. The results are as follows.

**Expected loss (mean).** Modular — not submodular, but not supermodular either. Adding any risk to any portfolio increases expected loss by exactly its own expected loss, independent of what else is in the portfolio. This is the null result.

**Certainty equivalents.** Submodular precisely when the underlying utility function's loss function is convex. For the exponential utility (entropic risk measure), this requires the penalty function to be convex in the loss — which it is for standard parameterisations.

**Coherent distortion risk measures.** This is the important class. Wang & Yu show that a law-invariant coherent risk measure is submodular if and only if it is a distortion risk measure — and the class of law-invariant coherent distortion risk measures includes Expected Shortfall at every confidence level. ES at 99.5% (the Solvency II SCR standard) is submodular.

**Value at Risk.** Not submodular. The paper documents this empirically on US equity returns: rolling historical estimation shows persistent VaR violations of submodularity during market stress periods. This is not a surprising result — VaR's failure of subadditivity under non-elliptical distributions is well known — but the systematic empirical documentation of submodularity violations during stress is new and important.

**Shortfall risk measures.** Complete characterisation via Arrow-Pratt risk aversion. The risk measure is submodular if and only if the Arrow-Pratt measure of the underlying loss function is monotone non-decreasing. Translated: the penalty for losses must become increasingly severe at the margin, not increasingly forgiving.

---

## Why VaR failing this test matters for internal models

Most Solvency II standard formula users and many internal model firms anchor to VaR 99.5% as their capital metric, with diversification credits applied through a prescribed correlation matrix. The PRA has historically accepted this structure. The Wang-Yu result adds a new layer to the mathematical critique of this approach.

VaR's submodularity violations cluster in stress periods — the paper shows this on equity data. In calm markets, VaR approximately satisfies submodularity, which is why diversification credits computed under normal conditions appear reasonable. In stress, VaR fails to be submodular: the observed joint tail behaviour is not consistent with the marginal VaRs and the assumed correlation structure. Capital computed from VaR-plus-correlation during calm markets can understate the actual capital requirement under stress precisely because the diversification assumption breaks down when you most need it.

The internal model implication: if your SCR is built on VaR 99.5% with correlation-matrix diversification, you have a model that is not submodular. Your diversification credit is not a mathematical property of the risk measure — it is an assumption that may fail under stress. You should be stress-testing that assumption explicitly, not treating the diversification credit as a structural result.

ES does not have this problem. ES 99.5% is submodular, meaning its diversification credits are mathematically valid properties of the measure, not assumptions that require separate testing.

---

## Capital allocation implications

The submodularity property has a direct consequence for the Euler allocation — the method used to allocate aggregate capital to business units in most internal models.

Euler allocation works by taking the gradient of the aggregate risk measure with respect to each component's contribution. Under a submodular risk measure, this allocation has a natural interpretation: the marginal capital contribution of each business unit decreases as you add more diversifying units. The allocation is stable: adding a new uncorrelated line of business reduces the marginal allocation to all existing lines.

Under VaR, this stability fails. The marginal capital contribution of existing lines under stress can *increase* when you add new business, because VaR's violation of submodularity means the gradient calculation is internally inconsistent in stress scenarios. This manifests in practice as capital allocations that behave erratically when the portfolio composition changes — a symptom that internal model teams will recognise even if they have not attributed it to submodularity failure.

The practical recommendation from Wang-Yu's framework: if you are doing Euler allocation from a VaR-based aggregate capital, test the stability of the allocation across different portfolio compositions. If it is unstable, that is not a numerical artifact — it is the expected consequence of allocating capital from a risk measure that is not submodular.

---

## Optimised certainty equivalents and the AES result

Wang & Yu prove that optimised certainty equivalents (OCEs) are consistently submodular — a useful result because OCEs are a natural framework for robust capital optimisation. They also show something striking about adjusted Expected Shortfall (AES) with nonconvex penalties: under the constraint that AES must be submodular, the nonconvex penalty collapses to standard ES. You cannot add nonconvex penalties to ES and retain the submodularity property. The penalty structure must be convex.

This has a direct implication for any firm using a modified ES formulation in their internal model — adding an ad-hoc convex adjustment to capture left-tail structure, for example. If the adjustment is nonconvex, you have likely broken the submodularity of your risk measure. Run the test.

---

## What this means for pricing teams

Capital modellers building or validating internal models have three concrete takeaways.

**Test your risk measure.** The paper provides explicit conditions for submodularity across all standard risk measure families. For ES-based models, you are structurally fine. For VaR-based or mixed formulations, the empirical test is straightforward: compute ρ(X ∨ Y) + ρ(X ∧ Y) and compare to ρ(X) + ρ(Y) for pairs of business units under stress scenarios. Violations indicate your diversification credit is not a mathematical property of your model.

**Stress test your diversification credits separately from your tail calibration.** Wang & Yu's equity data shows VaR violations clustering in 2008–09, 2020, and similar stress periods. If your internal model's diversification credits were estimated on pre-stress data, they may be overstated precisely when capital matters. This is a separate test from your tail calibration: you can have a well-calibrated tail and still have structurally incorrect diversification assumptions.

**Use ES for risk optimisation.** When using [insurance-optimise](https://pypi.org/project/insurance-optimise/) to trace efficient frontiers across retention and profit objectives, the underlying risk metric matters. The EfficientFrontier sweep is solving a constrained optimisation problem — if the risk measure used to set the loss ratio or capital constraint is not submodular, the frontier's shape may not be globally valid. An ES-based constraint gives you a frontier where the diversification benefit embedded in the objective is a mathematical property of the measure, not a parameter assumption. A VaR-based constraint does not.

---

## Connection to insurance-optimise

The EfficientFrontier class in [insurance-optimise](https://pypi.org/project/insurance-optimise/) sweeps a single constraint (retention, GWP, or loss ratio) and traces the Pareto-optimal trade-off curve. The implicit question underneath the frontier is: how does capital change as we move along the frontier?

Wang-Yu's result connects here in a specific way: if you are using a loss ratio constraint that is derived from a submodular risk measure (ES-based), the constraint is mathematically consistent with the diversification structure of your portfolio. As the frontier moves from high-retention/low-profit to low-retention/high-profit, the capital allocation to each segment follows a well-behaved marginal structure.

If you are using a VaR-derived loss ratio constraint, the frontier is still mathematically valid as an optimisation result — the solver does not care about submodularity — but the capital interpretation requires additional stress testing. The diversification credits embedded in the loss ratio target may not hold in stress, and the frontier's apparently smooth trade-off curve may mask stress-period instabilities.

---

## The verdict

Wang & Yu's paper is more mathematical than most actuarial teams will need to engage with directly. The characterisation results are complete and the empirical validation on equity returns is convincing. The core message for practitioners is not difficult: ES is submodular and VaR is not, and this matters for diversification credits in any stress scenario.

The honest limitation of the paper is that it does not engage directly with the Solvency II capital structure — it treats the academic risk measure literature, not the specific regulatory implementation. Translating the theoretical results into the standard formula or internal model context requires judgment. The translation we have offered above is our reading; a PRA validator may require firmer mapping.

What the paper does provide is the mathematical language to have the diversification conversation precisely. When a board asks whether the diversification credit is justified, the answer can now include a formal criterion rather than a reference to the correlation matrix. That is worth something.
