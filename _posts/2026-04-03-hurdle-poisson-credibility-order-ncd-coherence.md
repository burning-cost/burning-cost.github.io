---
layout: post
title: "Your Hurdle-Poisson Model Might Say Accidents Make Drivers Safer"
date: 2026-04-03
categories: [techniques, credibility]
tags: [credibility, ncd, bonus-malus, hurdle-model, zip, random-effects, stochastic-monotonicity, experience-rating, motor, python, bayesian]
description: "Lee et al. (arXiv:2602.02398) prove that standard hurdle-Poisson models with bivariate normal random effects can violate credibility order — your frequency estimate goes down after a claim. Here is when it happens, why, and how to fix it."
author: Burning Cost
---

There is a coherence condition that every experience rating system should satisfy. If you observe a policyholder making a claim in year one, your predicted claim frequency for year two should be higher — or at least no lower — than if they had been claim-free. This is called stochastic monotonicity, or credibility order.

It sounds obvious. It is not guaranteed.

Lee, Kim, So, and Ahn (arXiv:2602.02398, February 2026) prove that standard hurdle-Poisson models with bivariate normal random effects can violate this condition for empirically realistic parameter values. On Wisconsin government property insurance data, 9.29% of policyholders in the bivariate normal model received a lower predicted claim frequency after filing a claim than after having none. They then prove when the violation occurs, and propose two alternative models that eliminate it.

---

## What credibility order actually means

The formal statement is straightforward. Let Y_t denote a policyholder's claim count in period t. A model satisfies **base credibility order** if:

```
E[Y_{t+1} | Y_1=0, ..., Y_t=0]  ≤  E[Y_{t+1} | Y_1=0, ..., Y_t=1]
```

More claims in the history implies a higher predicted frequency. The **general credibility order** is stronger: for any nondecreasing function h, the expectation E[h(Y_{t+1}) | history] respects the same ordering. This is stochastic dominance of the entire predictive distribution, not just its mean.

In an NCD system, you enforce credibility order mechanically: file a claim, move down two levels, pay a higher premium next year. The coherence is baked into the transition rules. But if you are running a Bayesian hierarchical frequency model — true posterior credibility, not a discrete scale — the coherence has to come from the model structure. And it does not always.

---

## Where the bivariate normal model breaks

The standard hurdle-Poisson random effects model looks like this:

```
Z_t | (b1, b2) ~ Bernoulli(Φ(b1))      # claim occurs or not
N_t | (b1, b2) ~ Poisson(exp(b2))      # claim count given occurrence
(b1, b2)        ~ BivNormal(μ, Σ)      # correlated normal random effects
Y_t             = Z_t · (1 + N_t)
```

The random effects (b1, b2) are correlated with covariance matrix Σ. This is the natural structure when you believe that policyholders who are more likely to have an incident (high b1) also tend to have higher counts when they do (high b2).

**Theorem 1 of Lee et al.** proves that in the limit as σ₁² → 0⁺ (the hurdle random effect variance shrinks toward zero), with σ₂² > 0 and ρ ∈ (-1, 1) fixed:

```
lim E[Y₂ | Y₁=0] − E[Y₂ | Y₁=1]  >  0
```

The expected claim count next year is *higher* after a zero than after a claim. The model has become incoherent.

The numerical confirmation is in Tables 1–3 of the paper. At σ₁² = 0.1, σ₂² = 1, ρ = 0.5: E[Y₂ | Y₁=0] = 1.218, E[Y₂ | Y₁=1] = 1.084. The violation is not marginal — it is a 12% gap in the wrong direction. At σ₁² = 5 the violation disappears.

The intuition: when the hurdle random effect has low variance, observing Y=0 carries almost no information about the shared latent structure. But observing Y>0 tells you the Poisson rate is elevated (via the count component random effect). If the two components are positively correlated and the hurdle variance is small, seeing a claim *reduces* the weight on the low-risk tail of the posterior — counterintuitively pulling the expected count down. Remark 1 in the paper extends this to ZIP models: the same pattern holds.

This is not a pathological corner case. σ₁² = 0.1 is a perfectly sensible fitted value in a fleet or commercial property book where the zero-inflation component has limited individual variation.

---

## Two fixes

### Model 3: Independent conjugate random effects

Replace the bivariate normal with independent, conjugate priors:

```
Z_t | Θ₁ ~ Bernoulli(Θ₁),  Θ₁ ~ Beta(a, b)
N_t | Θ₂ ~ Poisson(Θ₂),    Θ₂ ~ Gamma(α, β)
Y_t        = Z_t · (1 + N_t)
```

No correlation between the hurdle and count components. The conjugacy gives closed-form posterior updates after observing y₁, ..., yₜ:

- Let k = number of periods with claims, s = Σ Yᵢ (total claims)
- Posterior for Θ₁: Beta(a + k, b + t − k)
- Posterior for Θ₂: Gamma(α + s − k, β + k)

**Proposition 1** of the paper gives the exact monotonicity condition: base credibility order holds if and only if αₜ* · aₜ* ≤ βₜ* · (αₜ* + βₜ* + 1) for all t ≥ 1, where starred symbols denote posterior parameters. **Corollary 2** gives the sufficient condition: if a < β at the prior (before any data), base credibility order is guaranteed throughout the posterior sequence.

In Python this is thirty lines:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class ConjugateCountPrior:
    a: float    # Beta prior for hurdle probability
    b: float    # Beta prior for hurdle probability
    alpha: float  # Gamma prior for Poisson rate
    beta: float   # Gamma prior for Poisson rate

@dataclass
class ConjugateCountPosterior:
    a: float; b: float; alpha: float; beta: float

    def update(self, y: np.ndarray) -> "ConjugateCountPosterior":
        k = int(np.sum(y > 0))          # periods with claims
        t = len(y)
        s = int(np.sum(y))              # total claims
        return ConjugateCountPosterior(
            a     = self.a + k,
            b     = self.b + t - k,
            alpha = self.alpha + s - k,  # sum of (y-1) for y>0
            beta  = self.beta + k,
        )

    def credibility_premium(self) -> float:
        """E[Y_{t+1}] under current posterior."""
        p_claim = self.a / (self.a + self.b)
        mean_count_given_claim = self.alpha / self.beta
        return p_claim * (1 + mean_count_given_claim)

    def is_monotone(self) -> bool:
        """Proposition 1: alpha * a <= beta * (alpha + beta + 1)."""
        return self.alpha * self.a <= self.beta * (self.alpha + self.beta + 1)

def from_prior(prior: ConjugateCountPrior) -> ConjugateCountPosterior:
    return ConjugateCountPosterior(**vars(prior))

# Example: a=1, b=4, alpha=2, beta=5  (a < beta => guaranteed monotone)
prior = ConjugateCountPrior(a=1.0, b=4.0, alpha=2.0, beta=5.0)
state = from_prior(prior)
print(state.is_monotone())  # True — guaranteed by Corollary 2

# Observe 5 periods: two with claims (1 claim each), three clean
history = np.array([1, 0, 1, 0, 0])
state = state.update(history)
print(f"Credibility premium: {state.credibility_premium():.4f}")
print(f"Still monotone: {state.is_monotone()}")
```

The limitation of Model 3 is that it only achieves base credibility order — ordering of the posterior mean. It does not guarantee stochastic ordering of the full predictive distribution.

### Model 4: Comonotonic random effects (stronger guarantee, harder to fit)

Replace both random effects with a single shared latent variable Θ:

```
Z_t | Θ ~ Bernoulli(σ(c + Θ))          # logistic link
N_t | Θ ~ Poisson(log(1 + exp(d + Θ))) # softplus link
Y_t       = Z_t · (1 + N_t)
```

A single Θ drives both components. **Theorem 2** proves that general credibility order holds if and only if the derivative of the softplus link lies in [0, 1] — which it does by construction: d/dx log(1+eˣ) = eˣ/(1+eˣ) ∈ (0, 1).

The price: no conjugacy. Estimating Θ requires MCMC (the paper uses NIMBLE in R). There is no Python implementation. The posterior for Θ given observations is one-dimensional, so a simple Metropolis-Hastings sampler would work, but it is not a ten-minute build.

On the Wisconsin LGPIF data (497 government entities, 2006–2011), Model 4 outperforms everything: out-of-sample MSE of 0.9382 versus 0.9466 for Model 3 and 1.0489 for the original bivariate normal. The bivariate normal model is actually worse than a hurdle model with no random effects at all (MSE 0.9943 for ZIP with no RE). Being incoherent turns out to be expensive as well as embarrassing.

---

## Does our Bühlmann-Straub implementation satisfy credibility order?

We built [`insurance-credibility`](/insurance-credibility/) around Bühlmann-Straub estimation. The natural question is whether the scalar random effects in that model satisfy credibility order.

They do. The Bühlmann-Straub model has a single scalar random effect θᵢ per group, with E[Xᵢⱼ | θᵢ] = μ(θᵢ). Because the random effect is scalar and the mean function is monotone, observing more claims in period t shifts the posterior for θᵢ upward, which shifts the predictive mean for period t+1 upward. The violation in Lee et al. requires *two* random effects with mismatched variances — a hurdle component and a count component with different variance scales. Single-RE models cannot produce the paradox.

The Bühlmann-Straub formula Z_i = w_i / (w_i + K) grows with exposure w_i and blends toward the portfolio mean as K increases. More claims means higher estimated θᵢ means higher credibility premium. Credibility order is structurally guaranteed.

The gap in the credibility library is not a monotonicity problem but a different one: Bühlmann-Straub treats all historical periods equally, regardless of how recent they are. A claim five years ago gets the same weight as a claim last year. That is a calibration question, not a coherence question — and it is addressed by the discount weighting in our [temporal credibility post](/2026/03/17/buhlmann-straub-treats-last-year-the-same-as-five-years-ago/).

---

## Who should care

The violation Lee et al. identify is specific to **Bayesian longitudinal credibility models with two-component claim structures**. It is not a problem for:

- Standard GLM with NCD as a rating factor — NCD is a static covariate, not a posterior update
- Bühlmann-Straub — scalar RE, guaranteed coherent as above
- GBM/XGBoost frequency models — no random effects

It is a real problem for:

- Any Bayesian hierarchical frequency model that separates the zero/non-zero decision from the count given non-zero (hurdle or ZIP structure) with correlated random effects
- Commercial fleet pricing where multi-year panels support genuine posterior updating
- Solvency II internal models that include longitudinal Bayesian frequency projections

If you are running such a model, the diagnostic is a single check. With σ₁² (hurdle RE variance) << σ₂² (count RE variance) and positive ρ, you are in the violation region. Model 3 is the practical fix: drop the bivariate normal, replace with independent Beta-Bernoulli and Gamma-Poisson conjugate priors, set a < β to guarantee monotonicity, and gain closed-form updates as a bonus.

---

The paper is at [arXiv:2602.02398](https://arxiv.org/abs/2602.02398). The Python implementation of Model 3 above is a starting point; a fuller version including violation diagnostics and predictive sampling is in the [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility) roadmap.

- [Experience Rating: NCD and Bonus-Malus](/2026/02/27/experience-rating-ncd-bonus-malus/) — NCD as a Markov chain, claiming thresholds, and the non-monotone result in the ABI scale
- [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/) — scalar random effects, exposure weighting, and why single-RE models avoid the Lee et al. violation
- [Does Bühlmann-Straub Credibility Work for Insurance Pricing?](/2026/03/23/does-buhlmann-straub-credibility-work-insurance-pricing/) — benchmark results on freMTPL2 data
