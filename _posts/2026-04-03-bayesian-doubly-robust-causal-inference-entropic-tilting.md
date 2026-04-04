---
layout: post
title: "Bayesian Doubly Robust Causal Inference: What Entropic Tilting Adds and Why We Are Watching, Not Building"
date: 2026-04-03
author: Burning Cost
categories: [causal-inference, research, pricing]
tags: [bayesian, doubly-robust, AIPW, causal-inference, ATE, credible-intervals, entropic-tilting, SMC, BART, insurance-causal, Consumer-Duty, arXiv-2506-04868, research-review]
description: "Orihara, Momozaki & Sugasawa (arXiv:2506.04868) produce a Bayesian posterior over the ATE by tilting the product of independent posteriors to satisfy the DR moment condition. We explain the mechanism, map it against our existing stack, and give our honest verdict on adoption."
---

The frequentist causal inference toolkit in insurance pricing is now reasonably mature. Double Machine Learning gives you debiased ATE estimates via Neyman-orthogonal cross-fitting. TMLE gives you semiparametric efficiency and a substitution estimator that respects parameter space. Synthetic DiD gives you a panel-based ATT with a doubly-robust backstop. All three are in production use at some UK firms; all three are covered in the [Burning Cost library stack](/insurance-causal/).

The one thing none of them give you is a posterior distribution over the treatment effect. They give you point estimates and confidence intervals — asymptotically valid, but derived from large-sample Gaussian approximations. If someone asks "what is the probability that this telematics device reduced claim frequency by more than 5 per hundred policy-years?", a frequentist 95% CI cannot answer that directly. It was not designed to.

Orihara, Momozaki & Sugasawa ([arXiv:2506.04868](https://arxiv.org/abs/2506.04868)) tackle this. Their paper delivers what the title promises: a full Bayesian posterior over the ATE that retains the double robustness properties of AIPW, without the feedback problem that afflicts naive joint Bayesian models. The mechanism is elegant. We are not building it yet. But we think the underlying idea is worth understanding properly, because the conditions under which we would build it are not far away.

---

## The feedback problem in Bayesian causal models

The natural Bayesian approach to causal inference is to put a joint prior over all model parameters and run the joint posterior. You define p(alpha, beta | D) — where alpha parameterises the propensity score and beta parameterises the outcome model — and you draw from the posterior via MCMC.

This has a structural problem. In a joint Bayesian model, the outcome data Y flows into the posterior over alpha. The propensity score model is *contaminated* by outcome information. The propensity score is supposed to capture the treatment assignment mechanism, P(A=1 | X), independently of potential outcomes. If Y informs alpha, the separation breaks down. The posterior over alpha is no longer a clean model of who gets treated — it is a model that has partially learned to predict Y via alpha.

Frequentist DR estimators (AIPW, DML, TMLE) solve this by design: they fit the propensity and outcome models completely separately, then combine them in a moment function or targeting step. You lose any claim to a coherent posterior, but you gain clean separation. The outcome model cannot contaminate the propensity model, because they were never fit together.

The paper's question is whether you can have both: separate estimation and a posterior distribution.

---

## Entropic tilting: the mechanism

The solution is to start with two completely independent posteriors and then move from that product distribution to the nearest distribution that satisfies the doubly-robust moment condition.

**Step 1: fit independent posteriors.**

Run your propensity model — logistic regression with Bayesian priors, or BART — on the treatment assignment alone. You get p_n(alpha | D). Run your outcome model — a Bayesian GLM, or BART — on the outcome. You get p_n(beta | D). These are fit independently. No feedback, by construction.

**Step 2: define the DR residual.**

The standard AIPW augmentation term is:

```
B_n(alpha, beta) = (1/n) * sum_i  [(A_i - e(X_i; alpha)) / (e(X_i; alpha) * (1 - e(X_i; alpha)))]
                                   * (Y_i - m_{A_i}(X_i; beta))
```

where e(X; alpha) is the propensity score, m_a(X; beta) is the conditional mean outcome under treatment A=a, A_i is the binary treatment indicator, and Y_i is the observed outcome. When this is zero in expectation, you are sitting on the semiparametrically efficient estimating equation for the ATE.

**Step 3: tilt the product posterior.**

Define a tilted distribution:

```
pi_{n,lambda}(alpha, beta) = exp{lambda * B_n(alpha, beta)} * p_n(alpha|D) * p_n(beta|D) / Z
```

The parameter lambda is chosen so that the DR residual is exactly zero in expectation under this new distribution:

```
E_{pi_{n,lambda}}[B_n(alpha, beta)] = 0
```

This is equivalent to minimising the KL divergence from pi_{n,lambda} to the product posterior p_n(alpha|D) * p_n(beta|D), subject to the constraint that the DR moment condition holds. The tilted posterior is the *closest* distribution to the independent product that is also doubly robust.

**Step 4: extract the ATE posterior.**

The ATE under the tilted posterior is:

```
E_{pi_{n,lambda}(beta)}[ (1/n) * sum_i { m_1(X_i; beta) - m_0(X_i; beta) } ]
```

This is a full posterior distribution. You get credible intervals from the quantiles of the ATE samples, and you can compute P(ATE > x) directly.

---

## Computing it: Sequential Monte Carlo

The practical challenge is computing the tilted posterior. You cannot sample from it directly, because it involves an exponential tilt of a joint parameter posterior that has no closed form.

The paper uses Sequential Monte Carlo. Starting from particles drawn from the product posterior p_n(alpha|D) * p_n(beta|D), the algorithm increments lambda from 0 to lambda* in steps, at each step reweighting particles, resampling, and applying a kernel-smoothing move:

```python
# Sketch of the SMC algorithm — this is not library code
# Algorithm 2 from Orihara, Momozaki & Sugasawa (arXiv:2506.04868)

particles = draw_from_product_posterior(alpha_posterior, beta_posterior, n_particles)
lambda_t = 0.0

for step in range(max_steps):
    # Compute DR residual for each particle
    B_vals = [B_n(p["alpha"], p["beta"]) for p in particles]

    # Increment lambda
    lambda_new = find_lambda(lambda_t, B_vals, target_mean=0.0)
    delta_lambda = lambda_new - lambda_t

    # Reweight
    log_weights = delta_lambda * np.array(B_vals)
    weights = softmax(log_weights)

    # Resample
    idx = multinomial_resample(weights, n_particles)
    particles = [particles[i] for i in idx]

    # Smooth toward mean to prevent particle degeneracy (kernel step)
    # Each key (alpha, beta) is shrunk 0.99 toward the particle mean + noise
    mean_alpha = np.mean([p["alpha"] for p in particles])
    mean_beta  = np.mean([p["beta"]  for p in particles], axis=0)
    particles = [
        {"alpha": 0.99 * p["alpha"] + 0.01 * mean_alpha + jitter_scalar(),
         "beta":  0.99 * p["beta"]  + 0.01 * mean_beta  + jitter_vector()}
        for p in particles
    ]

    lambda_t = lambda_new
    if abs(np.mean(B_vals)) < tolerance:
        break

# ATE posterior from final particles
ate_samples = [(1/n) * sum(m1(X, p["beta"]) - m0(X, p["beta"])) for p in particles]
ate_posterior_mean = np.mean(ate_samples)
ate_credible_interval = (np.percentile(ate_samples, 2.5), np.percentile(ate_samples, 97.5))
```

There is no published Python package. Anyone implementing this needs bespoke SMC code, a working Bayesian BART backend (PyMC-BART, stochtree, or numpyro), and careful engineering around the reweighting step to avoid particle degeneracy.

---

## What the theory actually guarantees

The paper proves three formal results.

**If the outcome model is correctly specified:** the tilted posterior matches the original outcome posterior. No efficiency cost from tilting. You could have just used the Bayesian outcome model directly — the DR structure is redundant, but it is not harmful.

**If only the propensity model is correctly specified:** the posterior mean ATE is approximately consistent despite outcome misspecification. The paper explicitly acknowledges that "some bias still remains" — this is a *weaker* guarantee than classical AIPW, which achieves exact consistency when the propensity alone is correct. The Bayesian DR estimator depends more on outcome model correctness than its frequentist counterpart.

**Under nonparametric (BART) outcome models with correctly specified propensity:** the tilted posterior achieves improved convergence rates compared to the G-computation formula alone, with better finite-sample coverage. The simulation evidence in Table 2 of the paper supports this.

The asymptotic point estimate is equivalent to Bang & Robins (2005) under standard conditions. The novelty is the posterior distribution in finite samples, not the point estimate. At large n, you would get essentially the same ATE from TMLE or DML — what you would not get is P(ATE > 0.05) or the posterior variance.

---

## Where this would genuinely matter for UK pricing

The case for Bayesian DR credible intervals is strongest in three specific insurance contexts.

**Small specialist segments.** For commercial lines, HNW, or niche personal lines where n < 500 per segment, the asymptotic normal approximation in DML confidence intervals is unreliable. A Bayesian posterior is exact in finite samples, given the prior. The frequentist 95% CI for DML at n = 200 has 82–88% actual coverage in the Kang-Schafer misspecification scenario (from the paper's simulations) — the Bayesian posterior under correctly specified outcome model has coverage close to the nominal 95%.

**Prior incorporation.** If your pricing team has a strong prior from actuarial tables or previous evaluations — "rate changes on direct motor have historically produced -2% to -8% effects on new business volume" — a Bayesian model can use that. DML cannot. The prior regularises the estimate in exactly the situations where the data are thin and the estimate is otherwise noisy.

**Regulatory interpretability under Consumer Duty.** P(treatment effect > threshold) is a more actionable statement for a non-technical FCA reviewer than "p = 0.04, reject null at 5%". Consumer Duty (PS22/9, PRIN 2A) requires firms to explain how pricing decisions affect customer outcomes. A posterior-based statement is easier to explain than a confidence interval without misusing the word "probability".

---

## Why it does not fit our current stack

We are not building a `BayesianCausalPricingModel` class from this paper. Here is the actual reasoning.

**No CATE.** The paper estimates a scalar ATE. For pricing decisions, we need segment-level heterogeneous treatment effects. Bayesian Causal Forests ([`insurance-causal`](/insurance-causal/), wrapping stochtree) already provide Bayesian posteriors over *policy-level* treatment effects. The BCF posterior is strictly more useful than the Orihara et al. ATE posterior for most pricing decisions, and BCF is implemented, tested, and in production use.

**No panel extension.** Our most important causal inference use case is evaluating rate changes that happened at a specific point in time across multiple segments over multiple periods — a panel DiD problem handled by [`insurance-causal-policy`](/insurance-causal-policy/)'s SDID and DRSC estimators. The Orihara et al. DR moment condition is cross-sectional. The DR augmentation term B_n does not extend to the panel ATT without substantial theoretical work that the paper does not do.

**Implementation cost is high.** The SMC approach requires a production-quality BART backend (PyMC-BART or numpyro), bespoke particle filtering, and careful tuning of resampling parameters. PyMC and numpyro are not Databricks serverless-friendly as of mid-2025. Adding a hard dependency on either — plus the JAX stack for numpyro — is not a trade we would make for an ATE-only estimator without CATE.

**TMLE already fills most of the gap.** For single cross-section causal inference with doubly-robust properties, [`insurance-causal`](/insurance-causal/) already includes TMLE. TMLE provides semiparametric efficiency, double robustness, and EIF-based standard errors. What it does not provide is a full posterior distribution. In practice, the EIF standard error at n > 500 is so close to the posterior credible interval (under well-specified models) that the operational benefit of the full posterior is marginal.

The honest assessment: the paper scores 12 out of 25 on our BUILD criteria. The threshold is 17. It is WATCH, not BUILD.

---

## What would change our assessment

We would reconsider if any of the following happen.

**A Python package ships.** The authors have not released one. If they do — or if someone forks stochtree to implement the SMC tilting — the implementation cost drops substantially. The paper's method is well-specified enough that a competent implementation should be straightforward given the right backend.

**A panel extension is published.** Extending the entropic tilting approach to the panel DiD setting — where the DR moment condition needs to accommodate the ATT over multiple treated periods and a synthetic control weight structure — would directly address our most acute need. This is a non-trivial theoretical contribution. If it appears, this goes straight to reconsideration.

**FCA guidance on uncertainty quantification.** If the FCA or PRA issues guidance that specifically requires probabilistic uncertainty quantification in pricing causal analyses — not just confidence intervals, but posterior-style probability statements — the demand case for Bayesian DR strengthens materially. Nothing in PS22/9, EP25/2, or PRA CP6/24 currently requires this, but the direction of travel on model governance is towards stronger quantification requirements.

---

## The relationship to our existing Bayesian causal tools

If you want Bayesian posterior uncertainty on causal effects in insurance pricing, the current recommendation is BCF via [`insurance-causal`](https://github.com/burning-cost/insurance-causal):

```python
from insurance_causal import BayesianCausalForest

model = BayesianCausalForest(outcome='binary', num_mcmc=500, random_seed=42)
model.fit(X=rating_factors, treatment=rate_increase_applied, outcome=renewed)

# Policy-level CATE with posterior credible intervals
cate_df = model.cate(rating_factors)
# cate_mean  cate_lower  cate_upper  cate_std
#    -0.061      -0.074      -0.048     0.007

# P(treatment effect < -0.05) for a specific segment
import numpy as np
samples = model.cate_samples(rating_factors.iloc[[0]])   # shape: (n_mcmc,)
print(f"P(effect < -5pp): {np.mean(samples < -0.05):.2f}")
```

BCF gives you the posterior at the *policy level*, handles regularisation-induced confounding correctly via the RIC prior separation, and is already implemented on top of stochtree 0.4.0 which ships C++ compiled wheels. It is the right tool for "what is the Bayesian posterior on the causal effect of this rate change for this type of customer?"

The Orihara et al. method would, if implemented, give you a posterior on the *population-average* effect with formal DR guarantees. The two tools would be complementary — BCF for segment-level heterogeneity with Bayesian uncertainty, Bayesian DR for ATE-level formal robustness guarantees. But BCF comes first, and BCF is already built.

---

## References

- Orihara, S., Momozaki, T. & Sugasawa, S. (2025). Bayesian Doubly Robust Causal Inference via Posterior Coupling. arXiv:2506.04868.
- Hahn, P.R., Murray, J.S. & Carvalho, C.M. (2020). Bayesian Regression Tree Models for Causal Inference: Regularization, Confounding, and Heterogeneous Treatment Effects. *Bayesian Analysis* 15(3): 965–1056.
- Bang, H. & Robins, J.M. (2005). Doubly Robust Estimation in Missing Data and Causal Inference Models. *Biometrics* 61(4): 962–973.
- van der Laan, M.J. & Rubin, D.B. (2006). Targeted Maximum Likelihood Learning. *International Journal of Biostatistics* 2(1).
- Kang, J.D.Y. & Schafer, J.L. (2007). Demystifying Double Robustness: A Comparison of Alternative Strategies for Estimating a Population Mean from Incomplete Data. *Statistical Science* 22(4): 523–539.

---

**Related reading:**

- [Doubly Robust Causal Inference for Insurance: TMLE With Poisson Outcomes](/2026/03/12/insurance-tmle/) — our existing DR estimator for cross-sectional causal inference; covers propensity misspecification robustness, the Poisson targeting step, and the SuperLearner nuisance ensemble
- [Heterogeneous Lapse Effects with Bayesian Causal Forests: Beyond the Average Elasticity](/2026/03/12/insurance-bcf/) — policy-level Bayesian posteriors over treatment effects; the right tool when you need segment heterogeneity, not just an ATE with a posterior
- [Why your rate change evaluation should be doubly robust](/2026/03/31/doubly-robust-synthetic-control-rate-change-evaluation/) — DRSC for panel DiD; the context in which a Bayesian extension of the DR panel estimator would be most valuable
