---
layout: post
title: "What Fraction of Your Book Has Infinite-Mean Losses?"
date: 2026-04-03
author: burning-cost
categories: [research, severity-modelling]
tags: [bayesian-nonparametric, heavy-tail, severity, evt, gpd, dirichlet-process, normalised-stable, shifted-gamma-gamma, subsidence, escape-of-water, reinsurance, insurance-severity, arXiv-2602.07228, nieto-barajas, MCMC, python, actuarial]
description: "Nieto-Barajas (arXiv:2602.07228) proposes a Bayesian nonparametric mixture of Shifted Gamma-Gamma distributions that eliminates the EVT threshold selection problem. The posterior of a tail index parameter answers a question no other method in our toolkit can: what fraction of this book has genuinely infinite-mean loss potential? Useful for reinsurance triaging and subsidence/flood portfolios. Not yet in insurance-severity — here is why."
permalink: /2026/04/03/bnp-heavy-tail-severity-arXiv-2602-07228/
---

One of the permanent embarrassments of extreme value theory in insurance is the threshold problem. To fit a Generalised Pareto Distribution to a claims portfolio you must first choose a threshold u above which GPD applies. The usual toolkit — mean excess plots, Hill plots, automated profile likelihood — all require you to eyeball something or pick a parameter. The threshold drives the fitted tail index, the tail index drives the TVaR, and the TVaR drives the reinsurance pricing. The entire chain hangs on a choice that has no principled answer.

Luis Nieto-Barajas at ITAM in Mexico City submitted a paper to arXiv in February 2026 (arXiv:2602.07228) that proposes a way to sidestep this. It does not solve every problem in severe-loss modelling — it introduces new ones — but it answers one question no other tool in our kit currently addresses: *what fraction of this book has genuinely infinite-mean loss potential?*

That is not an academic question. It determines whether you should be buying excess-of-loss cover or aggregate stop-loss cover. It determines whether your reinsurer's price is expensive or correct. For certain UK peril classes — subsidence, escape of water, bodily injury — we think it is a question worth being able to answer precisely.

---

## The Shifted Gamma-Gamma distribution

The paper's load-bearing object is the Shifted Gamma-Gamma (SGG) distribution. It is built from a two-stage hierarchy:

```
X - mu | Y  ~  Gamma(gamma, y)       [shape gamma, rate y]
Y | alpha, beta  ~  Gamma(alpha, beta)
```

Marginalising over Y gives a four-parameter family with support on [mu, infinity). The four parameters do distinct jobs:

| Parameter | Role |
|-----------|------|
| mu        | Location shift — keeps support on [mu, ∞) |
| gamma     | Body shape — controls the peak |
| alpha     | **Tail index** — this is the one that matters |
| beta      | Scale |

The tail index alpha is the key diagnostic. The SGG nests GPD as a special case when gamma = 1, with shape parameter xi = 1/alpha. Under the GPD correspondence, the k-th moment is finite iff xi < 1/k, i.e. alpha > k. The classification therefore follows directly:

- **alpha < 1**: infinite mean (and therefore infinite variance) — the reinsurance concern zone
- **1 ≤ alpha < 2**: finite mean, infinite variance — intermediate
- **alpha ≥ 2**: all moments finite — light tail

This structure is not arbitrary. When gamma = 1, the SGG reduces exactly to a GPD with shape parameter xi = 1/alpha and scale sigma = beta/alpha. The SGG nests GPD as a special case while allowing the body shape to depart from the exponential — something GPD cannot do. For claims distributions that are neither pure Pareto nor pure exponential in the body, this matters.

---

## The BNP mixing step

Fitting a single SGG to a claims portfolio assumes all risks share the same tail character. That assumption is obviously wrong for any heterogeneous book. Subsidence claims include both minor cosmetic cracking (alpha large — finite moments, no reinsurance concern) and structural failures requiring full demolition (alpha small — infinite mean, serious reinsurance concern). A single-component SGG will average these two populations and tell you nothing useful about either.

Nieto-Barajas addresses this by placing a Bayesian nonparametric prior on the mixing distribution. Each observation gets its own parameter vector (mu_i, gamma_i, alpha_i, beta_i), drawn from an unknown mixing distribution G. Rather than assuming G is, say, a Normal or a Dirichlet, the prior on G is a Normalised Stable (N-stable) process:

```
G ~ NS(nu, G_0)
```

where nu ∈ (0, 1) is a concentration parameter and G_0 is the centering measure — the "average" distribution over SGG parameters. The N-stable process is almost surely discrete, which means it naturally induces clustering of observations into a finite (but data-determined) number of groups. At nu = 0.5, the N-stable produces heavier-tailed weight distributions than the Dirichlet process (which is the nu → 0 limit). The stick-breaking weights are:

```
V_j ~ iid Beta(1 - nu, j * nu)
W_j = V_j * prod_{k < j} (1 - V_k)
```

The full model hierarchy, concisely:

```
X_i | theta_i  ~  SGG(mu_i, gamma_i, alpha_i, beta_i)
theta_i | G    ~  iid G
G              ~  NS(nu, G_0)
```

The posterior of alpha_i for each observation answers, directly, whether that observation belongs to a heavy-tail cluster or a light-tail cluster. Aggregate over the posterior and you get the tail decomposition:

- P(alpha < 1): fraction of observations in the infinite-mean regime
- P(1 ≤ alpha < 2): fraction in the intermediate regime
- P(alpha ≥ 2): fraction in the light-tail regime

On the paper's accident insurance dataset (n = 1,383 claims, median $6,000, mean $17,158), the fitted model finds: **14.4% of observations in infinite-mean territory, 58.8% intermediate, 26.8% light-tail**. This is not a histogram curiosity. It means that 14% of claims, by count, are drawn from a distributional family where the standard actuarial assumption of finite mean is wrong.

---

## UK personal lines where this matters

**Subsidence.** UK household subsidence is structurally bimodal. The majority of claims are cosmetic — hairline cracks, minor movement, monitoring and underpinning costs running to £5,000–£15,000. A small fraction are structural failures requiring demolition, rebuild, or extensive civil engineering, with costs from £60,000 to over £500,000. These two populations have different tail indices. A single GPD fitted to all subsidence claims will be wrong about both. BNP SGG will identify the structural failure cluster and report its alpha separately. The tail decomposition tells you directly what fraction of your subsidence book is in a regime where excess-of-loss protection is structurally necessary rather than optional.

**Escape of water.** The same logic applies, with three natural modes: contained drying (£1,500–£3,500), full strip-out and reinstatement (£8,000–£20,000), and structural damage requiring decant and major reinstatement (£60,000+). As we noted in the [two-hump distributions post](/2026/04/02/two-hump-distributions-severity-model-95th-percentile/), a unimodal model fitted to EoW data is wrong about the structure from the start. BNP SGG's clustering identifies these sub-populations without requiring you to pre-specify how many there are.

**Bodily injury.** Whiplash and soft tissue claims settle under £5,000. Serious and catastrophic injury claims — brain injury, spinal cord injury, PPO cases — start at £15,000 and have no ceiling. The infinite-mean question is directly relevant to PPO cases, where the settlement is an annuity rather than a lump sum, and the expected long-run liability may be theoretically unbounded. Knowing what fraction of a BI book sits in the alpha < 1 regime is a reinsurance structuring input.

---

## What this gives you that nothing else does

We maintain a growing collection of extreme value methods in [insurance-severity](https://pypi.org/project/insurance-severity/). To be direct about where BNP SGG stands relative to what already exists:

| Method | Threshold needed | Covariates | Truncation/limits | Speed | Tail UQ |
|--------|-----------------|------------|-------------------|-------|---------|
| TruncatedGPD | Required | No | Yes | Fast | Profile lik |
| Spliced composite | Required | Partial | No | Fast | Bootstrap |
| BMA threshold (pipeline) | Averaged | Yes | No | Moderate | BMA weights |
| BayesianGPD (pipeline) | Required | No | No | MCMC | Honest CrI |
| EQRN | Required | Yes | No | GPU | Bootstrap |
| **BNP SGG** | **None** | **No** | **No** | **Slow** | **Posterior** |

The two columns where BNP SGG is uniquely better are the first and the last. No threshold, and the tail uncertainty quantification is a proper Bayesian posterior rather than a profile likelihood or bootstrap approximation. No other method in the toolkit answers: "What fraction of this book has infinite-mean losses?" with a posterior distribution over that fraction.

The columns where it loses are real problems for production use. No covariates means it cannot do per-risk tail estimation — EQRN already does that. No truncation support means it cannot handle the per-policy limits that are standard in UK personal lines, which would cause it to systematically overestimate how heavy the tail is (censored observations look heavier than they are). The speed is genuinely prohibitive: the paper's Fortran implementation takes 40 minutes for n = 1,383 on a 3.00 GHz Intel Xeon. A Python/NumPyro reimplementation would be slower.

---

## The MCMC machinery

The fitting algorithm is a Gibbs sampler extending Neal's Algorithm 8 to the N-stable process setting. Five steps per sweep:

1. **Latent Y_i update**: conjugate — the conditional posterior is Gamma(gamma_i + alpha_i, x_i - mu_i + beta_i). No Metropolis step; this is a direct draw.

2. **Auxiliary values**: draw r = 3 new parameter vectors from the centering measure G_0. This is Neal's augmentation trick for handling non-conjugate cluster assignment.

3. **Cluster assignment**: assign each observation to an existing cluster or a new one. The probability of joining cluster j is proportional to (n*_j - nu) × f(x_i | theta*_j), where n*_j is the cluster size excluding observation i. The (n*_j - nu) term is the N-stable analogue of the Dirichlet n_j term; nu acts as a discount factor penalising small clusters.

4. **Cluster parameter update**: Metropolis-Hastings on (mu, gamma, alpha, beta) within each cluster, using a uniform random walk proposal with adaptive step sizes tuned in batches of 50 iterations.

5. **nu update**: the concentration parameter nu is sampled via adaptive MH, with prior Be(0.5, 0.5) — the arcsine prior, which allows the data to determine whether clustering is tight or diffuse.

The paper reports that on both real datasets (accident claims and England population data), the model over-clusters slightly relative to the truth — 19 posterior clusters on 1,383 observations vs a true structural count of perhaps 3–4 — but correctly identifies the heavy and light tail sub-populations despite this over-clustering. For tail decomposition purposes, over-clustering is harmless; what matters is that the alpha posteriors correctly separate into the right regimes.

---

## What a Python implementation would look like

We have not built this yet. The scoring puts it at 12/20 against our 14/20 build threshold — primarily because the 40-minute MCMC runtime is prohibitive for production use, and because the missing truncation support is a systematic bias risk on real UK insurance data with heterogeneous policy limits. BayesianGPD (arXiv:2510.14637) scores 16/20 and addresses a different problem (honest credible intervals under serial dependence); it is the higher priority Bayesian EVT addition.

If we were building it, the API would follow the existing insurance-severity sklearn-compatible pattern:

```python
from insurance_severity import BNPHeavyTailSeverity

model = BNPHeavyTailSeverity(
    n_components_max=30,   # truncation level for N-stable stick-breaking
    nu_prior=(0.5, 0.5),   # arcsine prior — paper's recommended default
    n_warmup=1000,
    n_samples=2000,
    seed=42,
)
model.fit(large_losses)    # positive claims above a reporting threshold

# The novel output: posterior tail decomposition
decomp = model.tail_decomposition()
# {'heavy': 0.144, 'intermediate': 0.588, 'light': 0.268}

# Posterior mean number of clusters
print(model.n_clusters_)   # e.g. 19

# Density and quantile estimates from posterior predictive
model.pdf(x)
model.ppf(0.99)
model.tvar(0.99)
```

The `tail_decomposition()` method is the uniquely novel output. Everything else — density, quantile, TVaR — can be approximated from existing tools. The posterior fraction in each tail regime is what no other method delivers.

The dependency recommendation is NumPyro (≥0.15.0) with JAX (≥0.4.25). The N-stable stick-breaking construction is expressible in NumPyro's distribution language; JAX handles vectorised cluster parameter updates. The SGG log-likelihood must be implemented from scratch — it is not in NumPyro's distribution library. The critical constraint on x - mu > 0 (the SGG has support [mu, ∞)) requires careful handling in the log-likelihood to avoid NaN gradients.

For truncation — when policy limits cap individual claims — the corrected log-likelihood requires the SGG survival function:

```python
# Per-observation truncation correction (policy limit T_i)
log_lik_corrected = log_f_SGG(x, theta) - log_F_SGG(T, theta)
```

The SGG CDF has no closed form; it requires numerical integration. Without this correction, fitting BNP SGG to UK personal lines data with heterogeneous limits will underestimate alpha — making tails appear heavier than they are. We would not ship a version without truncation support.

---

## What the paper does not tell you

**No covariate extension.** The model fits a portfolio-level tail distribution. It cannot produce per-risk alpha estimates. If you want to know whether older properties have heavier subsidence tail distributions than newer ones — a straightforward and actuarially relevant question — BNP SGG has no answer. You could, in principle, parameterise the centering measure G_0 as a function of covariates, but this extension is not in the paper and is non-trivial to implement correctly.

**No formal equivalence to the spliced approach.** The connection between BNP SGG and a spliced body-plus-GPD model is intuitive — both are trying to describe a multi-regime loss distribution — but there is no formal result here. The BNP approach makes the regime boundaries soft and data-determined; the splice makes them hard and user-specified. The BNP approach is more principled; the splice is more interpretable and directly produces ILF schedules. These are not the same model with different parameterisations.

**Computational cost is a real barrier.** The 40-minute Fortran runtime for n = 1,383 claims scales badly. For a UK personal lines motor book with 50,000 large losses, the model is computationally infeasible without subsampling. The paper does not address subsampling. Variational inference exists as an alternative but VI with discrete latents is unreliable for BNP models — the posterior tail fraction estimates under VI are likely to be biased in ways that are hard to detect.

---

## How to use the ideas now

Without a production implementation, the paper's contribution is still immediately useful for two things.

**Reinsurance triage.** Before buying or renewing an excess-of-loss treaty, run a visual tail analysis using insurance-severity's existing EVT toolkit. If the mean excess plot shows strong non-linearity (the mean excess function is not approximately linear above a threshold), a single GPD is likely misspecified. This is the signature of a multi-regime tail — the case where BNP SGG would tell you something a single GPD cannot. Flag it for qualitative discussion with your reinsurer.

**Subsidence and EoW reserve review.** For a book with 500–2,000 large subsidence or EoW claims above a reporting threshold, a rough two-component analysis — split the data at an obvious trough in the loss distribution, fit GPD separately above and below — gives you a first approximation to the tail decomposition that BNP SGG would produce automatically. It is manual and fragile, but it answers the same qualitative question: is the heavy-loss sub-population pulling the tail into infinite-mean territory?

```python
from insurance_severity.evt import TruncatedGPD
import numpy as np

# Rough manual two-component approximation
# Claims above reporting threshold, with per-claim policy limits
large_losses = np.array([...])  # claims > £25,000
policy_limits = np.array([...])  # per-claim

# Split at a natural trough — inspect histogram first
attritional_large = large_losses[large_losses < 100_000]
structural = large_losses[large_losses >= 100_000]

for component, name in [(attritional_large, "attritional-large"),
                        (structural, "structural")]:
    limits_component = policy_limits[large_losses >= (100_000 if name == "structural" else 0)]
    limits_component = limits_component[limits_component >= component.min()]

    gpd = TruncatedGPD()
    gpd.fit(component, upper_truncation=limits_component)

    xi = gpd.xi_  # tail index; 1/xi maps to alpha in SGG notation
    print(f"{name}: xi={xi:.3f}, alpha_approx={1/xi:.3f}")
    print(f"  Infinite mean:     {xi > 1.0}")   # xi > 1 <=> alpha < 1
    print(f"  Infinite variance: {xi > 0.5}")   # xi > 0.5 <=> alpha < 2
```

This is a workaround, not a solution. It requires a user-specified split point (the trough), assumes both sub-populations are Pareto-tailed, and produces no posterior uncertainty on the tail decomposition fractions. What BNP SGG would give you is the same decomposition without the arbitrary split, with a posterior distribution over the fraction in each regime, and with automatic detection of how many distinct tail sub-populations exist.

---

## The verdict

Nieto-Barajas (arXiv:2602.07228) is a genuine methodological contribution. The SGG+N-stable combination is new in Python — there is no existing implementation. The posterior tail decomposition is a diagnostic that has no equivalent in the current EVT toolkit. For reinsurance structuring decisions on UK specialty books — subsidence, EoW, bodily injury — knowing what fraction of the portfolio has infinite-mean loss potential is a question worth answering rigorously.

The barriers are real: no truncation support (a systematic bias risk for real insurance data), no covariate extension, and computational cost that is prohibitive for large books without subsampling. We are not building it in the current phase. When the author releases code, or when a follow-up paper addresses truncation, the calculus changes.

For now, the paper earns a careful read for anyone doing reinsurance analytics or tail risk diagnostics on heterogeneous portfolios. The SGG structure is worth understanding even if you are not running the full Bayesian machinery — the insight that gamma ≠ 1 allows body shapes that GPD cannot capture, while nesting GPD exactly when gamma = 1, is directly useful for understanding when GPD is misspecified.

---

*The paper is Nieto-Barajas, L.E. (2026), "Modelling heavy tail data with bayesian nonparametric mixtures", arXiv:2602.07228, stat.ME. Submitted February 2026.*

*Related:*
- *[Two-Hump Distributions: Why Your Severity Model Gets the 95th Percentile Wrong](/2026/04/02/two-hump-distributions-severity-model-95th-percentile/) — bimodal severity in UK motor BI and EoW; NeuralGaussianMixture*
- *[Stochastic Ordering for Extreme Claims: What It Actually Tells Cat XL Pricers](/2026/04/02/stochastic-ordering-extreme-claims-cat-xl-reinsurance/) — structural portfolio comparison for cat XL*
- *[Spliced Severity Distributions: When One Distribution Isn't Enough](/2025/03/15/spliced-severity-distributions-when-one-distribution-isnt-enough/) — the parametric alternative with explicit threshold*
