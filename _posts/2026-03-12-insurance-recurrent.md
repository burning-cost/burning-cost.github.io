---
layout: post
title: "Your Frequency Model Treats Every Policy-Year as Independent"
date: 2026-03-12
categories: [libraries, pricing, survival-analysis]
tags: [frailty, recurrent-events, gamma-frailty, lognormal-frailty, shared-frailty, joint-frailty, Bühlmann-credibility, fleet, pet-insurance, insurance-recurrent, python]
description: "A standard GLM models annual claim frequency as a Poisson count, treating each policy-year as independent. Policyholders with two claims last year are more likely to claim again — not because of observed covariates, but because of unobserved propensity. Shared frailty models separate population heterogeneity from true recurrence risk. insurance-recurrent is the first Python implementation."
post_number: 82
---

Motor insurance personal lines is relatively forgiving about the independence assumption. Most policyholders claim zero or one times per year. The recurrence structure is sparse.

Fleet insurance is not. A fleet driver with three claims in two years is not experiencing bad luck — they have a different underlying risk than the fleet average. Pet insurance is not: some animals are chronic condition animals and will claim repeatedly. Home insurance is not: the same property may suffer repeated escape of water events due to infrastructure age.

A standard Poisson GLM treats each policy-year as an independent draw from a Poisson distribution with the same mean (given covariates). It ignores within-subject correlation. The result: regression dilution of the repeat-claimer effect into the covariates that correlate with it.

Shared frailty models are the correct tool. They introduce a latent random effect that acts multiplicatively on each policyholder's claim rate. Policyholders with high frailty have elevated rates across all their observed years; the model learns to identify them from the pattern of recurrences.

[`insurance-recurrent`](https://github.com/burning-cost/insurance-recurrent) is the first Python implementation of shared and joint frailty models designed for insurance portfolios.

```bash
uv add insurance-recurrent
```

## Shared frailty model

The shared frailty model augments the standard proportional hazards model with a latent random effect $U_i$ that acts multiplicatively on subject $i$'s hazard:

$$\lambda_{ij}(t) = U_i \cdot \lambda_0(t) \cdot \exp(\mathbf{x}_{ij}^T \boldsymbol{\beta})$$

where $U_i \sim \text{Gamma}(\theta^{-1}, \theta^{-1})$ with variance $\theta$. Marginalising over $U_i$ gives the frailty variance as a measure of unobserved heterogeneity.

```python
from insurance_recurrent import SharedFrailtyModel

model = SharedFrailtyModel(
    frailty_dist="gamma",  # or "lognormal"
    baseline="breslow"     # Nelson-Aalen estimator
)
model.fit(
    X=features,
    times=claim_times,
    events=claim_indicators,
    subject_id=policy_ids
)

# Frailty variance estimate
print(f"Theta (frailty variance): {model.theta_:.4f}")

# Per-policyholder frailty scores
frailty_scores = model.predict_frailty(policy_ids)

# Predicted claim intensity with frailty adjustment
intensity = model.predict_intensity(X_test, exposure=exposure_test)
```

## Joint frailty model for informative censoring

The joint frailty model (Rondeau et al. 2007 SiM) adds a terminal event — typically policy lapse — to the recurrent event model. High-frailty customers are more likely to both claim frequently and lapse. Treating lapse as independent censoring produces biased frailty estimates.

```python
from insurance_recurrent import JointFrailtyModel

jfm = JointFrailtyModel(
    frailty_dist="gamma",
    recurrent_baseline="breslow",
    terminal_baseline="breslow",
    association_param="estimate"  # or fix to a value
)
jfm.fit(
    X_recurrent=X,
    times_recurrent=claim_times,
    events_recurrent=claim_indicators,
    subject_id=policy_ids,
    X_terminal=X_terminal,
    times_terminal=policy_end_times,
    events_terminal=lapse_indicators
)

print(f"Association alpha: {jfm.alpha_:.4f}")
# Positive alpha: high frailty -> high lapse propensity
```

## Bühlmann-Straub credibility connection

The connection between shared frailty and Bühlmann-Straub credibility is exact under gamma frailty. The posterior expectation of the frailty given the observed claim history is a credibility-weighted average of the prior mean and the observed rate:

$$E[U_i | \mathbf{y}_i] = Z_i \cdot \bar{y}_i / \mu_0 + (1 - Z_i) \cdot 1$$

where $Z_i = n_i / (n_i + \kappa)$ is the Bühlmann credibility weight and $\kappa = 1/\theta$ is the precision of the frailty distribution.

```python
# Retrieve credibility weights and their Bühlmann-Straub interpretation
credibility = model.credibility_weights()
print(credibility[["policy_id", "n_claims", "frailty_posterior", "credibility_weight"]])
```

This makes shared frailty directly comparable to the classical actuarial experience rating framework. The frailty model is Bühlmann-Straub with a non-parametric baseline hazard instead of a parametric one.

## Synthetic validation with RecurrentEventSimulator

```python
from insurance_recurrent import RecurrentEventSimulator

sim = RecurrentEventSimulator(
    n_subjects=1000,
    theta=0.5,           # frailty variance
    beta=np.array([0.3, -0.2, 0.1]),  # covariate effects
    baseline_rate=0.15,  # annual claim rate
    max_duration=3.0,    # years of observation
    lapse_rate=0.2       # annual lapse hazard
)
X, times, events, subject_ids, lapses = sim.simulate()
```

The simulator generates correlated recurrent event data with known frailty variance, making it straightforward to validate that the model recovers the true $\theta$.

## When to use frailty models vs GLM

A standard GLM is adequate if:
- Most policyholders appear in one policy-year only (new business dominated book)
- Claim frequency is very low (< 5% annual frequency) and zero-claims dominate
- You have no interest in within-subject effects

Use frailty models when:
- You have multi-year observations for the same policyholders
- Claim frequency is moderate to high (fleet, pet, home maintenance claims)
- You need to identify chronic claimers for underwriting action
- Experience rating requires credibility adjustment beyond a simple prior rate

---

`insurance-recurrent` is available on [PyPI](https://pypi.org/project/insurance-recurrent/) and [GitHub](https://github.com/burning-cost/insurance-recurrent). 128 tests. The Databricks notebook demonstrates fleet insurance frailty scoring on synthetic data.

Related: [insurance-survival](https://burning-cost.github.io/2026/02/27/experience-rating-ncd-bonus-malus/) for standard survival analysis, [insurance-competing-risks](https://burning-cost.github.io/2026/03/12/insurance-competing-risks/) for competing perils, [credibility](https://burning-cost.github.io/2026/02/19/buhlmann-straub-credibility-in-python/) for Bühlmann-Straub.
