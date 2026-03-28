---
layout: post
title: "Robust Rate Optimisation: Pricing Against Demand Model Misspecification"
date: 2026-03-13
categories: [libraries, pricing, optimisation]
tags: [dro, distributionally-robust-optimisation, wasserstein, ambiguity-set, price-of-robustness, rate-optimisation, demand-model, FCA, PS21/11, ENBP, cvxpy, insurance-dro, python, motor, home, renewal-pricing]
description: "Distributionally robust rate optimisation: worst-case demand within a Wasserstein ball. Price-of-robustness curve for UK pricing committee papers - Python."
---

<div class="notice--warning" markdown="1">
**Package update:** `insurance-dro` has been consolidated into [`insurance-optimise`](/insurance-optimise/). Install with `pip install insurance-optimise` — distributionally robust rate optimisation is available as a submodule. [View on GitHub →](https://github.com/burning-cost/insurance-optimise)
</div>


Every rate optimiser in UK personal lines works the same way. You estimate a demand model. You optimise rates against that model subject to your ENBP constraints and commercial guardrails. You get a rate recommendation. You present it to committee.

The implicit assumption, never stated, is that your demand model is correct.

It is not correct. It is calibrated on renewal data from a period when price walking was legal. FCA PS21/11 banned price walking in January 2022. Your model was trained on consumer behaviour under a pricing regime that no longer exists. The elasticities are estimates with standard errors. The model architecture is wrong in ways you have not yet discovered. And when your optimiser is wrong about demand, it is wrong in the worst possible way: it finds rates that are locally optimal under the wrong model, which means they are systematically suboptimal under reality.

This is the demand model misspecification problem. It is not exotic. Every pricing actuary who has ever submitted a rate change and watched the actual conversion deviate from forecast knows exactly what this feels like.

[`insurance-dro`](https://github.com/burning-cost/insurance-dro) adds a safety margin. Instead of optimising against a single demand model, it optimises against the worst-case demand distribution within a calibrated ball of plausible distributions around your empirical samples. The output is not a single rate vector but a curve: profit at each level of robustness, with the profit you sacrifice to get there. That curve - the price of robustness - is what goes in the committee paper.

```bash
uv add insurance-dro
```

---

## What distributionally robust optimisation actually does

The standard rate optimisation problem is:

    max_m  E_{P_N}[profit(m, xi)]    subject to ENBP, commercial constraints

where `m` is the vector of rate multipliers, `P_N` is your empirical demand distribution (point estimates from your model), and `xi` is the demand shock that realises.

DRO replaces this with:

    max_m  min_{Q in B_eps(P_N)}  E_Q[profit(m, xi)]    subject to same constraints

The inner `min` finds the worst-case distribution `Q` within an ambiguity set `B_eps(P_N)` - a ball of radius `eps` around `P_N`. The outer `max` finds the rate vector that maximises profit even against that worst case.

`B_eps(P_N)` is the Wasserstein ball: all distributions `Q` within Wasserstein distance `eps` of `P_N`. Wasserstein distance measures how much probability mass you have to move, and how far, to go from one distribution to another. A radius of `eps = 0.1` means you are protecting against demand distributions that require moving 10% of the probability mass by one unit of demand - a concrete, interpretable bound on how wrong your model can be.

The price of robustness at radius `eps` is:

    PoR(eps) = Z_nominal - Z_robust(eps)

where `Z_nominal` is the expected profit under your point-estimate model and `Z_robust(eps)` is the worst-case expected profit at robustness level `eps`. `PoR(eps=0)` is zero by definition. As `eps` increases, you sacrifice more expected profit but gain protection against larger demand shifts.

The PoR curve always has an elbow. Before the elbow, each unit of radius buys a lot of robustness cheaply. After the elbow, you are paying steeply for marginal additional protection. The elbow is the actuarially sensible robustness level.

---

## Why not KL divergence

The ACM ICAIF 2024 paper by Bhatt et al. (DOI: 10.1145/3768292.3770404) is the first to apply DRO specifically to insurance pricing. Its main finding is cautionary: robust policies using phi-divergence ambiguity sets (KL divergence, chi-squared) were overly conservative and yielded limited performance gains under distributional shifts.

The mechanism is straightforward. KL divergence constrains `E_Q[log(dQ/dP)]`. This includes distributional shifts that no pricing decision can mitigate - shifts in the baseline level of demand, structural changes in the population, macroeconomic shocks. You are protecting against perturbations that your rates cannot hedge. The result is a very conservative rate recommendation for a modest actual gain in robustness.

Wasserstein DRO does not have this problem. The Wasserstein distance measures perturbations in terms of the transport cost of probability mass. These are perturbations that look like demand shifts your model failed to anticipate - competitor price moves, seasonal effects, the behavioural changes following PS21/11. They are perturbations that different rate vectors can hedge against differently.

`insurance-dro` ships all three ambiguity set types. We recommend Wasserstein as the primary. KL and chi-squared are available as comparison points - running them lets you see empirically how much more conservative they are on your data.

---

## The ENBP constraint

FCA PS21/11, and specifically the ENBP requirement, means there is a hard floor on the rates you can offer at renewal. Any rate that would result in a customer paying materially more than they would as a new customer for equivalent cover is prohibited.

In the DRO formulation, ENBP is a hard constraint on the rate multiplier:

    m_i <= enbp_i / technical_price_i    for all policies i

This constraint binds before the optimiser touches the objective. It does not interact with the robustness level - it is not relaxed as `eps` increases. The practical effect is that some policies have very little room to move regardless of what the demand model says, and this shows up in the rate vectors across the PoR curve.

PS21/11 also tightened the link between ENBP and demand model quality. The demand models used to set renewal pricing were calibrated on pre-PS21/11 behaviour when price walking was legal. The post-PS21/11 renewal demand function is different - more elastic at the ENBP bound, less elastic below it - and you should not be confident your pre-2022 demand model captures this. This is precisely the kind of structural misspecification that Wasserstein DRO is designed to handle.

---

## Using insurance-dro

The library connects directly to `insurance-causal` and `insurance-optimise` outputs. The key input is `demand_samples`: an `(N, S)` matrix of `S` bootstrap or posterior samples from your demand model across `N` policies. This matrix is the empirical distribution `P_N`. The Wasserstein ball is constructed around it.

```python
import numpy as np
from insurance_dro import AmbiguitySet, RobustOptimiser

# Inputs from insurance-causal / your demand model
rates          = np.array([...])           # current premiums, shape (N,)
technical_price = np.array([...])          # technical cost, shape (N,)
demand_samples  = np.array([...])          # bootstrap demand samples, shape (N, S)
elasticity_se   = np.array([...])          # elasticity standard errors, shape (N,)
enbp            = np.array([...])          # ENBP per policy, shape (N,)

# Auto-calibrate Wasserstein radius from DML elasticity standard errors
ambiguity = AmbiguitySet.from_elasticity_se(elasticity_se, coverage=0.95)

# Solve the robust optimisation problem
opt = RobustOptimiser(
    rates=rates,
    technical_price=technical_price,
    demand_samples=demand_samples,
    enbp=enbp,
    ambiguity_set=ambiguity,
)
result = opt.optimise()

print(f"Nominal profit:  £{result.nominal_objective:,.0f}")
print(f"Robust profit:   £{result.robust_objective:,.0f}")
print(f"Price of robustness: £{result.price_of_robustness:,.0f} "
      f"({result.price_of_robustness_pct:.1f}%)")
```

The `from_elasticity_se()` factory calibrates the radius as:

    eps = z * base_demand * max(SE) / 2

where `z` is the coverage quantile (1.96 for 95%) and `SE` is the vector of elasticity standard errors from your DML model. This maps directly from "how uncertain is my elasticity estimate" to "how large should my Wasserstein ball be". You do not need to choose `eps` by hand.

---

## The price of robustness curve

The committee paper deliverable is the frontier. Run `robustness_frontier()` to sweep across radii and compute the full PoR curve:

```python
frontier = opt.robustness_frontier(n_points=20)

# Tabular output for committee paper
from insurance_dro.diagnostics import por_table
print(por_table(frontier))
```

A typical output looks like this:

```
Wasserstein Radius (eps) | Robust Profit | PoR (£000) | PoR (%)
-----------------------------------------------------------------
0.000 (nominal)          |    £4,820,000 |          - |       -
0.010                    |    £4,803,000 |        £17 |    0.4%
0.025                    |    £4,771,000 |        £49 |    1.0%
0.050  *elbow*           |    £4,731,000 |        £89 |    1.8%
0.075                    |    £4,682,000 |       £138 |    2.9%
0.100                    |    £4,619,000 |       £201 |    4.2%
0.150                    |    £4,471,000 |       £349 |    7.2%
0.200                    |    £4,283,000 |       £537 |   11.1%
```

The elbow at `eps = 0.050` is detected automatically via maximum curvature (Kneedle-style diagonal distance). In this example: for £89,000 of foregone profit you are protected against demand shifts up to Wasserstein radius 0.05 from your empirical distribution. Beyond the elbow, each additional unit of robustness costs roughly £250,000 per 0.05 of radius. The committee decides whether that is worth paying.

The `worst_case_demand_plot()` diagnostic shows, for each policy, what the worst-case distribution assigns as demand under the robust solution. This is useful for identifying which segments are driving the robustness cost - typically the segments where the demand model is least reliable.

---

## Calibrating the radius

`from_elasticity_se()` is the recommended starting point. If you are using `insurance-causal` for DML elasticity estimation, the standard errors from that library flow directly into this factory with no manual tuning.

If you are not using DML standard errors, three alternative approaches:

First, calibrate from holdout data. Split your demand data into train and holdout. Fit the demand model on train, compute the Wasserstein distance between the train and holdout demand distributions. Use this as your `eps`. This is the empirical out-of-sample radius.

Second, calibrate from ENBP headroom. If 80% of your renewal book is within 10% of the ENBP bound, the effective demand uncertainty you care about is bounded by the ENBP constraint anyway. Set `eps` to reflect the magnitude of demand shift that would push a material fraction of policies against the bound.

Third, sensitivity analysis. Run the frontier. Present it to committee. Ask where the elbow is. The decision about how much robustness to buy is a business decision, not a statistical one.

---

## What the library does not do

It does not make demand model estimation easier. You still need a good empirical demand model to construct `P_N`. If your demand samples are unreliable, the Wasserstein ball is centred on an unreliable estimate, and the robust solution is robust around the wrong point. Garbage in, robust garbage out.

It does not replace scenario analysis. A deterministic scenario - "what if renewal conversion falls 5% across the portfolio due to a competitor price cut" - answers a different question than a Wasserstein ball of radius `eps`. Both are useful for different parts of a committee paper.

It does not solve the identification problem introduced by PS21/11. The structural break in consumer behaviour is a one-time shift, not a distributional uncertainty that DRO can hedge. DRO protects against ongoing demand model misspecification, not against not having data from the new regime.

---

## Technical notes

`insurance-dro` uses CVXPY with the CLARABEL solver (open source, ships with CVXPY 1.4+, no commercial licence required). The Wasserstein DRO problem reduces to a finite convex program via the Esfahani-Kuhn dual (Math Programming 171:115-166, 2018): for piecewise-affine profit the minimax reduces to an LP. With linearised demand (first-order Taylor around the current rate), the profit objective is affine in the rate multipliers, and the full DRO problem is a tractable convex program.

KL divergence reformulation uses CVXPY's ExpCone constraints. Chi-squared uses the standard result that `max_{Q: D_chi2 <= rho} E_Q[L]` equals `mean_L - sqrt(rho/(1+rho)) * std_L`, which is a concave objective and therefore DCP-compliant.

137 tests, MIT-licensed, Python 3.10+.

```bash
uv add insurance-dro
```

Source at [github.com/burning-cost/insurance-dro](https://github.com/burning-cost/insurance-dro).

---

**Related reading:**
- [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/) - the base optimiser that DRO extends with distributional robustness; use insurance-optimise first, then wrap with DRO when demand model uncertainty is material
- [Coverage Is the Wrong Guarantee for Pricing Actuaries](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) - conformal risk control for the claims side, now part of [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) (conformal-risk was absorbed into v0.4.0); DRO protects against demand model misspecification, CRC protects against claims model error
- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) - the demand model whose uncertainty DRO is designed to handle; better elasticity estimates shrink the ambiguity set
