---
layout: post
title: "PyINLA: INLA Finally Comes to Python. Here's What Pricing Teams Should Know."
seo_title: "PyINLA Bayesian GLMMs for Insurance Pricing: INLA Without R or MCMC"
date: 2026-04-03
categories: [techniques]
tags: [bayesian, glmm, inla, pyinla, random-effects, credibility, poisson, gamma, frequency-severity, python, hierarchical-models, spatial, pricing]
description: "INLA — the method that made Bayesian spatial GLMMs tractable in R — now has a proper Python package. 92–278x faster than PyMC, no R dependency. We explain what it does, why pricing teams should care, and why it is not ready for production yet."
math: true
author: burning-cost
---

R-INLA has been the tool of choice for Bayesian spatial statisticians for fifteen years. Epidemiologists, ecologists, and geographers have used it to fit random effects models that would take MCMC hours in under a second. Insurance pricing teams, who live almost entirely in Python and have no particular interest in learning R, have watched from the other side of a language barrier.

That barrier has mostly gone away. On 28 March 2026, the same KAUST group that built R-INLA — Esmail Abdul Fattah, Elias Krainski, and Håvard Rue — published a paper and a package: [PyINLA](https://pyinla.org) (arXiv:2603.27276). It is pip-installable, has no R dependency, and supports the model classes that insurance pricing actually uses: Poisson frequency, Gamma severity, negative binomial for overdispersed counts, and IID random effects for groups like region, scheme, and broker. It runs 92–278x faster than equivalent PyMC models in the paper's benchmarks.

Our verdict is **WATCH, not BUILD**. The package is v0.2.0 with an explicitly unstable API, insurance-scale performance is unverified, and the licence for commercial use needs clarifying. We expect to revisit this at v1.0. But the underlying method is sound and the fit for pricing is real, so it is worth understanding now.

---

## What INLA actually does

INLA stands for Integrated Nested Laplace Approximations. Rue, Martino, and Chopin published the method in 2009, and R-INLA has been the reference implementation since then. The core problem INLA solves is Bayesian inference for **latent Gaussian models** (LGMs).

An LGM has three levels. At the top, a vector of hyperparameters θ governs the prior variances of the random effects. In the middle, a latent Gaussian field x — the combined fixed effects, random effects, and any structured components — is drawn from N(0, Q(θ)⁻¹). At the bottom, the observations y are drawn from a likelihood that depends on a linear predictor η, which is a function of x.

$$y_i \mid x, \theta \sim \text{likelihood}(\eta_i, \theta)$$
$$x \mid \theta \sim \mathcal{N}(0, Q(\theta)^{-1})$$
$$\theta \sim \pi(\theta)$$

MCMC approaches this by sampling from the joint posterior p(x, θ | y). INLA sidesteps the sampling entirely. It approximates three quantities analytically:

1. A Gaussian approximation to p(x | y, θ) — used as a base for Laplace corrections
2. A Laplace approximation to the marginal posterior p(θ | y) — integrating out the latent field
3. The marginal posteriors p(θⱼ | y) and p(xᵢ | y) via numerical quadrature over the (low-dimensional) hyperparameter space

The key enabling condition is that the latent precision matrix Q(θ) must be **sparse**. This holds whenever the random effects have local structure: IID group intercepts, random walks over time, AR(1) processes, and spatial Markov random fields all produce sparse Q. For insurance GLMMs with region, scheme, and broker effects, this is always satisfied.

What this means operationally: INLA delivers marginal posteriors — means, standard deviations, full credible intervals — for every parameter in the model, without a Markov chain, without convergence diagnostics, and without warm-up periods. For a standard pricing GLMM, the typical runtime is under a second.

---

## Why pricing teams have wanted this for years

The standard pricing GLMM — Poisson claim frequency with fixed rating factors and IID random effects for region or scheme — is a natural fit for Bayesian estimation. Credibility theory is essentially the method-of-moments solution to the same problem. But full Bayesian inference has been blocked by two practical obstacles.

The first is MCMC friction. Stan and PyMC produce correct answers, but getting there requires tuning warm-up, checking R-hat convergence, inspecting chain mixing, and waiting minutes to hours for non-trivial models. A pricing actuary with a quarterly cycle cannot iterate on model structure if each fit takes twenty minutes.

The second is the R barrier. R-INLA is mature and well-validated, but asking a Python-based pricing team to maintain an R environment and pass data between two runtimes is a legitimate friction. Tools like rpy2 exist but are brittle in production.

INLA in Python, without MCMC, removes both obstacles simultaneously. A Poisson GLMM with region and scheme random effects fits in under a second. The full model specification lives in Python with the rest of the pricing stack.

---

## What the benchmarks actually show

The paper reports two directly relevant comparisons against PyMC with NUTS.

**Football GLMM (Poisson, IID random effects, n=626, 40 groups):**
- PyINLA: 0.24 seconds
- PyMC NUTS (4 chains × 25,000 draws): 21.74 seconds
- Speedup: 92×

The posterior means agreed to three decimal places. Correlation of random effect estimates was r = 1.0000.

**Scottish lip cancer (BYM spatial Poisson, n=56 districts):**
- PyINLA: 0.42 seconds
- PyMC NUTS (4 chains × 5,000 draws + 3,000 tuning): 117.2 seconds
- Speedup: 278×

The main covariate coefficient differed by 0.04 posterior standard deviations. District-level risk correlation was r = 0.9999.

We should be honest about what these benchmarks do and do not show. The comparison is against PyMC only — Stan with well-tuned HMC is faster than the PyMC numbers above, and the absolute speedup against Stan is probably closer to 10–50× based on prior literature (Modrák's 2018 comparison, which remains the most careful public analysis). A 10× speedup over Stan is still very useful operationally. The 278× headline should not be taken as the universal case.

More importantly: the largest dataset in the paper has n = 2,964. Insurance pricing datasets commonly run to 500K–1M rows. INLA's computational complexity is dominated by sparse Cholesky factorisation of Q, not by n directly — the latent dimension for a Poisson GLMM with IID effects is (fixed effects + groups), not observations — so there is good theoretical reason to expect it to scale. R-INLA is documented to handle 100K+ spatial observations. But whether the Python implementation introduces overhead at insurance scale is an open question the paper does not answer.

---

## The model classes that matter for pricing

PyINLA's supported families include Poisson, negative binomial, Gamma, Gaussian, binomial, beta, and survival models. The latent components include IID random intercepts, random walks (RW1, RW2), AR(1), BYM/BYM2 spatial effects, and SPDE continuous spatial fields.

For a UK personal lines pricing team, the useful model classes are:

**Poisson frequency with IID random effects.** The primary use case. Region IID + scheme IID + broker IID, all simultaneously estimated with shrinkage controlled by PC priors. The model accounts for exposure via the E parameter:

```python
from pyinla import pyinla

model = {
    "response": "claim_count",
    "fixed": ["1", "age_band", "vehicle_age", "ncd_years", "vehicle_class"],
    "random": [
        {
            "id": "region_id",
            "model": "iid",
            "hyper": {"prec": {"prior": "pc.prec", "param": [1.0, 0.05]}}
        },
        {
            "id": "scheme_id",
            "model": "iid",
            "hyper": {"prec": {"prior": "pc.prec", "param": [1.0, 0.05]}}
        }
    ]
}

result = pyinla(
    model=model,
    family="poisson",
    data=df,
    E=df["earned_exposure"].to_numpy()
)

result.summary_fixed      # mean, sd, 2.5%, median, 97.5% for each fixed effect
result.summary_random     # group-level posteriors
result.summary_hyperpar   # posterior for between-group variance τ²
```

**Gamma severity with region effects.** Same structure, different family. Claim-weighted to account for the number of claims contributing to each severity observation:

```python
model = {
    "response": "avg_severity",
    "fixed": ["1", "cover_type", "vehicle_age", "ncd_years"],
    "random": [
        {"id": "region_id", "model": "iid",
         "hyper": {"prec": {"prior": "pc.prec", "param": [0.5, 0.05]}}}
    ]
}
result = pyinla(model=model, family="gamma", data=df,
                weights=df["claim_count"].to_numpy())
```

**Temporal structure.** An AR(1) component for underwriting year effects sits alongside IID group effects with no extra effort. For reserve-influenced pricing or multi-year portfolios, this matters.

**Spatial models.** BYM2 for postcode sector spatial risk, or SPDE for continuous geographical risk surfaces. These are available immediately — not as a future extension.

---

## The relationship to Bühlmann-Straub credibility

Bühlmann-Straub is the method-of-moments estimator for the linear Bayes credibility problem. For a Normal-Normal model, it produces the same point estimate as a full Bayesian hierarchical model. But it stops there: you get a credibility factor Z and a credibility-weighted estimate, with no uncertainty on Z itself.

PyINLA extends this in three directions. First, it delivers p(τ² | y) — the full posterior for between-group variance — rather than a point estimate of the structural parameter a. Second, it propagates uncertainty in τ² through to the credibility weights, so you can say something like "Scheme A has credibility factor Z = 0.63, but the 90% credible interval is [0.38, 0.81]." Third, it handles non-Normal likelihoods directly: a Poisson frequency GLMM with IID group effects is not a Normal approximation but the actual model.

For thin portfolios — three schemes, four years of data — the Bühlmann-Straub estimate of the structural parameter is highly unstable. The posterior p(τ² | y) from INLA makes that instability visible rather than hiding it behind a single number. That visibility is useful for governance conversations: "We have genuine uncertainty about how much to trust this scheme's experience" is a defensible position at a reserving or pricing committee. "We applied a credibility factor of Z = 0.63" is less so.

We are not suggesting credibility is wrong. Bühlmann-Straub is still the right default for most pricing teams: it is closed-form, requires no prior specification, and produces interpretable output. PyINLA becomes valuable when a team needs posterior uncertainty on the credibility weights, non-Normal likelihoods, or layered structure (spatial + temporal + group effects simultaneously).

---

## PC priors: why the default prior choice is good

PyINLA uses penalised complexity (PC) priors by default for all hyperparameters. PC priors (Simpson et al. 2017) are constructed to shrink toward the base model — in this case, the model with no between-group variation (τ → ∞). This is the right default for insurance random effects.

The prior `{"prior": "pc.prec", "param": [1.0, 0.05]}` specifies P(σ > 1.0) = 0.05 — a 5% prior probability that the between-group standard deviation exceeds 1.0 on the log scale, which corresponds to a factor of e ≈ 2.7 between the mean risk and a 1σ group. For scheme or broker effects that is generous. For postcode district effects on flood risk it might be too conservative. But the prior is explicit and interpretable, which is more than can be said for many Bayesian implementations.

---

## What we are watching for

PyINLA is v0.2.0 with an explicitly pre-stable API. The authors say to expect rapid changes. We would not build production code on top of this now.

The specific triggers that would prompt us to move from watching to building:

**v1.0 release with stable API.** This is the primary gate. An unstable API means rewrites, not features.

**Licence confirmed as permissive.** The paper is published under CC BY-NC-ND 4.0 (a standard preprint licence, not a code licence). The pyinla.org site references Apache-2.0/MIT, but that attribution appears to describe a separate and dormant project. Before any commercial use or open-source embedding, we need the code licence confirmed in writing.

**Benchmark on a 500K-row Poisson GLMM.** The paper's largest example has under 3,000 rows. The theoretical scalability argument is sound — INLA operates on the latent dimension, not n — but we want empirical confirmation at insurance scale before trusting it in production.

**Documentation with worked examples.** pyinla.org exists but is thin. The combination of a stable API and complete documentation is what will make this useful to pricing teams who are not already INLA experts.

---

## When it matures

When PyINLA reaches v1.0 with these conditions met, the right build would be an `insurance-bayesian-glmm` package: a thin wrapper providing insurance-idiomatic API over PyINLA for Poisson frequency and Gamma severity models with IID group effects. Policy-level data frames in, posterior summaries and credibility intervals out, with automatic handling of the exposure offset and comparison output against a standard GLM benchmark. That package would sit naturally in the Burning Cost stack between our existing `insurance-credibility` implementation (Bühlmann-Straub, fast, frequentist-flavoured) and any future MCMC-based work.

The method is the right method. The implementation is not ready. We will check back in six months.

---

**Paper:** Esmail Abdul Fattah, Elias Krainski, Håvard Rue. "PyINLA: Fast Bayesian Inference for Latent Gaussian Models in Python." arXiv:2603.27276, March 2026.
**Package:** `pip install pyinla` · [pyinla.org](https://pyinla.org)
