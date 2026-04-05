---
layout: post
title: "Your NCD Table Has a Frequency Bias"
date: 2026-04-02
categories: [insurance-credibility, pricing-fundamentals]
tags: [bonus-malus, ncd, credibility, frequency-bias, bms-equilibrium, insurance-credibility, actuarial, arXiv-2601.12655, lemaire, underreporting]
description: "insurance-credibility v0.1.9 adds BMSEquilibriumSimulator — Lemaire NPV reporting thresholds, Liang 2-class equilibrium, and a frequency correction for the selection bias in NCD portfolios."
author: burning-cost
---

There is a number in every UK motor pricing model that is wrong by construction. It is the claim frequency you fitted your GLM on.

Not wrong because of data quality, not wrong because of sparse cells, but wrong because your portfolio is in bonus-malus system (BMS) equilibrium. The drivers who have been in your book for five years are not a random sample from the population — they are the drivers who have reached NCD level 5 because they either had low claim rates or because they chose not to report small claims. The claim frequency you observe in that cohort is lower than their true accident frequency. By a meaningful amount.

[insurance-credibility v0.1.9](https://pypi.org/project/insurance-credibility/) adds `BMSEquilibriumSimulator` to quantify this. The theoretical basis is Liang, Zhang, Zhou & Zou (arXiv:2601.12655) for the equilibrium mechanics, and Lemaire (1977) for the NPV reporting threshold framework. Both are now implemented in `insurance_credibility.classical.bms`.

---

## The hunger-for-bonus effect

Lemaire's original insight is straightforward: a rational policyholder will only report a claim if the claim payout exceeds the lifetime NPV of the premium increases that follow. If repairing a bumper costs £300, but losing two NCD steps means paying an extra £180 per year for three years to recover — discounted NPV roughly £480 — the rational response is to pay out of pocket.

The threshold at each NCD level n is:

```
b*_n = B × (d_n - d_{n-k}) × Σ_{t=1}^{T_n} δ^t
```

Where B is the base premium, d_n is the discount at level n, k is the step-back per claim (2 in the UK standard), T_n is the horizon to rebuild, and δ is the annual discount factor. At a base premium of £1,000 with a 60% NCD discount and a 2-step setback, the reporting threshold at level 5 is approximately £370–400. Claims below that amount are systematically suppressed.

This matters for frequency modelling because the suppression rate is not uniform across the portfolio. Policyholders at high NCD levels have larger step-backs relative to where they want to be, so they suppress more. Policyholders at low NCD levels — having little to lose — report almost everything. If you pool the observed frequencies without accounting for this, you underestimate true frequency at high NCD levels and your model's NCD factor is carrying both a genuine risk signal and a reporting behaviour signal that you cannot separate.

---

## BMSEquilibriumSimulator

The simulator takes a standard UK NCD table and returns reporting thresholds, stationary distributions, and corrected frequencies:

```python
from scipy.stats import gamma as gamma_dist
from insurance_credibility.classical.bms import BMSEquilibriumSimulator
import numpy as np

# Standard 5-step UK NCD (0%→20%→40%→50%→60%→65%)
discounts = [0.0, 0.20, 0.40, 0.50, 0.60, 0.65]

sim = BMSEquilibriumSimulator(
    discounts=discounts,
    base_premium=1_000.0,
    step_back=2,
    discount_factor=0.97,
    claim_freq=0.05,
    severity_dist=gamma_dist(a=2.0, scale=350.0),  # mean £700
)
sim.fit()
```

After fitting, the key outputs:

```python
print(sim.thresholds_)
# [  0.   189.   248.   327.   371.   406.]
# Reporting threshold at each NCD class in £

print(sim.reporting_probs_)
# [1.000  0.897  0.841  0.760  0.714  0.677]
# Fraction of claims that will be reported at each class

print(sim.stationary_dist_)
# [0.018  0.051  0.089  0.142  0.287  0.413]
# Long-run portfolio distribution across NCD classes
```

The frequency correction uses the reporting probabilities directly:

```python
observed_frequencies = np.array([0.12, 0.09, 0.07, 0.055, 0.045, 0.038])
corrected = sim.corrected_freq_(observed_frequencies)
# corrected[4] ≈ 0.045 / 0.714 ≈ 0.063
# True frequency is ~40% higher than observed at NCD level 4
```

The `frequency_bias_()` method returns the relative bias `(observed - true) / true` per class — a vector of negative numbers. At NCD level 4–5, the bias is typically in the range −29% to −33% once you account for the full severity distribution. The observed frequency for your longest-tenure, most loyal customers is a meaningful underestimate of their true accident rate.

---

## The Liang equilibrium: what happens when insurers compete

The reporting threshold above assumes a single-insurer market. Liang et al. add a second insurer and ask what happens in Nash equilibrium: if a policyholder knows they can switch insurer and have their NCD history partially reset (or treated differently), does that change the suppression threshold?

The paper shows it does, via a two-class closed-form equilibrium. The `liang_equilibrium()` method implements this:

```python
result = sim.liang_equilibrium(
    theta1=0.05,   # claim rate, good driver class
    theta2=0.15,   # claim rate, bad driver class
    kappa=1.8,     # insurer's premium multiplier for bad class
    k1=2.0,        # logistic steepness parameter
    k2=0.3,        # prior probability of bad class
)

print(result['threshold'])    # equilibrium reporting threshold £
print(result['eta'])          # fraction of good drivers at equilibrium
print(result['premium_diff']) # premium spread between classes at equilibrium
```

The closed-form solution uses the choice function η(Δ) = 1 / (1 + exp(k₁Δ + log((1-k₂)/k₂))). The equilibrium threshold is b* = δ(κ-1) × [θ₁·η + θ₂·(1-η)]. In practical terms: competitive pressure from a second insurer who will quote new business without NCD penalty lowers the suppression threshold — policyholders suppress less, because the value of maintaining NCD with this insurer is reduced when they can switch without penalty. Your observed portfolio frequency is *higher* in a competitive market than under monopoly.

---

## The stationary distribution and portfolio selection bias

The stationary distribution π gives you the long-run NCD class mix for a portfolio with these parameters. This has a second application beyond frequency correction: it is the benchmark against which to test whether your current portfolio is in equilibrium.

```python
print(sim.stationary_dist_)
# [0.018  0.051  0.089  0.142  0.287  0.413]
```

A portfolio with 60% of policyholders at NCD level 5 and a claim_freq of 0.05 should, in steady state, have this distribution. If your actual portfolio has 75% at level 5, your portfolio is either younger in BMS terms (still converging) or your claim frequency is lower than the 0.05 input. Both are useful diagnostic flags when you are questioning why your renewing portfolio's frequency has drifted from your new business frequency.

The `summary()` method prints a formatted table with all of this in one view — thresholds, reporting probabilities, stationary weights, and corrected frequencies — suitable for a pricing assumptions document or a peer review pack.

---

## What this means for GLM factor selection

The frequency bias is NCD-class-specific, not constant. A GLM that includes NCD as a rating factor will partially absorb the suppression effect in its NCD relativity. The level 5 relativity will be calibrated on observed frequency, which is a blend of lower accident rate (the genuine risk signal you want) and suppression behaviour (which is a function of your premium level and step-back table, not of accident risk).

The practical implication: if you are assessing whether to compress NCD relativities — for example, in a Consumer Duty value review — you should correct observed frequencies for the suppression bias before fitting. Without correction, your level 5 relativity looks lower than it truly is on a risk basis. Compressing it based on uncorrected data underprices genuine high-NCD risk.

The corrected frequencies from `BMSEquilibriumSimulator` can be passed directly as the response variable in a GLM (using the corrected array as the target and policy count as the exposure), or used to construct observation weights that down-weight high-NCD policies where the observed data is most biased.

---

## What the simulator does not do

The Lemaire NPV formula is the deterministic version — it computes the expected NPV of premium increases assuming a constant claim rate and discount factor. The full stochastic dynamic programme, where claim frequency during the rebuilding window is itself uncertain, gives slightly different thresholds. For a first-pass frequency correction the deterministic version is directionally correct and analytically transparent. We flagged this limitation in the implementation notes and will assess whether the full DP adds material precision in a future version.

The Liang Nash equilibrium is a two-class, two-insurer model. Real markets have continuous type distributions and more than two insurers. The model captures the direction of the competitive effect; the magnitude requires calibration to your market. `liang_equilibrium()` is a structured starting point for a pricing assumptions debate, not a production pricing input.

---

**Paper (Liang et al.):** [arXiv:2601.12655](https://arxiv.org/abs/2601.12655) | **Library:** [insurance-credibility on PyPI](https://pypi.org/project/insurance-credibility/) | **GitHub:** [insurance-credibility](https://github.com/insurance-credibility/insurance-credibility)
