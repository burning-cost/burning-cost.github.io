---
layout: post
title: "Bayesian Nonparametric Severity: Fitting Heavy Tails Without Choosing a Threshold"
date: 2026-04-03
author: Burning Cost
categories: [pricing, severity, techniques]
tags: [BNP, bayesian-nonparametric, heavy-tail, EVT, severity, Dirichlet-process, N-stable, SGG, shifted-gamma-gamma, threshold-selection, MCMC, reinsurance, subsidence, motor-BI, UK-insurance, arXiv-2602.07228, tail-index]
description: "The hardest part of fitting a GPD is picking the threshold. A new Bayesian nonparametric approach eliminates the choice entirely — and tells you what fraction of your book has infinite-variance loss potential. Here is what it does, where it adds value, and where to use something simpler."
---

Every EVT workflow begins with the same uncomfortable question: where does the tail start?

You plot the mean excess function. You look for a region where it rises roughly linearly. You pick a threshold. Then a colleague picks a different threshold and gets a different tail index. Neither of you is wrong — the choice is genuinely arbitrary above a certain point, and the downstream numbers (ILF schedules, XL layer costs, large loss loadings) are sensitive to it. The standard fix is profile likelihood: scan a grid of candidate thresholds and pick the one that maximises the likelihood. This is better than eyeballing, but it is still fitting the wrong model at every candidate and picking the least-wrong one.

Luis Nieto-Barajas at ITAM published a paper in February 2026 (arXiv:2602.07228) proposing a different approach: eliminate the threshold entirely. Instead of splitting the distribution at some cutoff and fitting a GPD to exceedances, he fits the whole distribution simultaneously using a Bayesian nonparametric mixture. The model figures out, from the data, which observations are heavy-tailed and which are not — without you specifying a cutoff in advance.

This is a genuine methodological advance. It is also, as we will explain, a tool with serious practical limitations for day-to-day insurance pricing. But for portfolio-level tail diagnostics — particularly in specialty lines, casualty reserving, and subsidence — it offers something no other method in the toolkit provides.

---

## The threshold problem in UK insurance

To see why threshold selection matters, consider three UK scenarios where the tail is commercially important.

**Motor bodily injury.** UK motor BI claims above £100k have a GPD tail index (ξ) in the range 0.15–0.45, depending on vehicle group and claim type. The ILF from a basic limit of £500k to £1m changes materially if ξ shifts from 0.20 to 0.35. That shift is plausibly explained by threshold selection alone — move the threshold from the 90th to the 95th percentile and you are fitting GPD to a different slice of data with different empirical properties. We showed in a previous post that [truncation bias from policy limits](/2026/03/20/your-gpd-is-lying-because-your-claims-are-truncated/) makes this worse: you may be fitting the wrong distribution to the wrong data simultaneously.

**Property subsidence.** Subsidence claims in England are bimodal in a way that most severity models cannot capture. Minor claims (cracking, cosmetic movement, soil compaction) cluster around £8k–£20k. Structural claims (underpinning, rebuild, NHBC involvement) cluster around £45k–£90k with a heavy tail above that. A single GPD fitted above a threshold will either miss the bimodality entirely (threshold too low) or fit only the structural sub-population with inadequate data (threshold too high). Neither is satisfactory.

**Liability long-tail.** Employers liability and public liability claims exhibit what reinsurers call "clash risk" — a small fraction of claims that are genuinely catastrophic (catastrophic injury, multi-claimant events, long-tail mesothelioma) sit in the same dataset as routine settlement-range claims. These sub-populations have fundamentally different tail behaviour. A single-threshold GPD imposes one tail index on what is effectively a mixture.

The common thread is that the underlying loss-generating process is not homogeneous above the threshold. There are multiple sub-populations with different tail behaviour. The standard approach — pick a threshold, fit GPD — aggregates them into a single tail index and pretends it is structural.

---

## How BNP mixture models work

The core idea is to model the entire severity distribution as a mixture of simpler distributions, where the number of mixture components is not fixed in advance. The data determines how many components are needed.

### The building block: Shifted Gamma-Gamma

The paper's building block is the Shifted Gamma-Gamma (SGG) distribution, a four-parameter family built from a hierarchical Gamma construction:

```
X - mu | Y  ~  Gamma(gamma, Y)      [shape gamma, rate Y]
Y | alpha, beta  ~  Gamma(alpha, beta)
```

Integrating out Y gives a distribution with location mu, shape gamma, tail index alpha, and scale beta. The critical parameter is **alpha**:

- alpha < 1: infinite variance — the reinsurance concern zone. Claims in this regime can be unboundedly expensive in the aggregate sense; the variance of total losses diverges.
- 1 ≤ alpha < 2: finite mean but infinite variance — still dangerous for excess-of-loss pricing.
- alpha ≥ 2: light tail, all moments finite — routine claims.

Two things make SGG particularly useful. First, when the shape parameter gamma equals 1, SGG reduces exactly to a Generalised Pareto Distribution with tail index ξ = 1/alpha. So GPD is a special case — SGG is strictly more flexible. Second, the tail classification above maps directly onto reinsurance decisions: infinite-variance claims behave differently in aggregate, and knowing what fraction of a portfolio sits in that regime is directly actionable.

### The Dirichlet Process, and why we need something heavier

In standard Bayesian nonparametrics, the mixing distribution — the distribution over cluster identities — typically follows a Dirichlet process (DP). The DP is almost surely discrete, which means draws from it naturally cluster observations. You do not specify how many clusters there are; the model infers the number from the data.

The paper uses a Normalised Stable (N-stable) process instead of a DP. The difference is subtle but relevant for heavy-tailed data: the N-stable process generates weight distributions with heavier tails than the DP, making it more natural for situations where one cluster accounts for a disproportionate fraction of extreme observations. The concentration parameter nu, between 0 and 1, controls the clustering tendency. At nu = 0 you recover the Dirichlet process. The paper recommends nu ~ Beta(0.5, 0.5) as a prior, letting the data decide.

The full model is:

```
X_i | theta_i  ~  SGG(mu_i, gamma_i, alpha_i, beta_i)
theta_i | G  ~  iid G
G  ~  N-stable(nu, G_0)
```

Each observation is assigned to a cluster with its own SGG parameters. The mixture is infinite in principle, but the posterior concentrates on a finite number of clusters. In the paper's accident insurance dataset (n = 1,383 claims, median $6,000, mean $17,158 — substantial right skew), the posterior settles on around 19 clusters, not a pre-specified number.

### What the posterior gives you

After running MCMC, you have a posterior distribution over the tail index alpha for every observation. This lets you compute:

```
P(alpha < 1)        = fraction of claims with infinite-variance tail potential
P(1 ≤ alpha < 2)    = fraction with finite mean, infinite variance
P(alpha ≥ 2)        = fraction with light tail
```

On the paper's accident insurance dataset, the decomposition comes out as:

| Tail regime | Posterior probability |
|---|---|
| Heavy tail (alpha < 1, infinite variance) | 14.4% |
| Intermediate (1 ≤ alpha < 2) | 58.8% |
| Light tail (alpha ≥ 2) | 26.8% |

This is a number no other severity fitting method produces. The Hill estimator gives you a single ξ for the whole dataset. The spliced composite gives you a single GPD shape above the threshold. Neither tells you "14% of claims are coming from a regime where the variance is infinite." The BNP mixture does.

For a subsidence portfolio, an analogous decomposition might reveal that the minor-claim sub-population sits clearly in the light-tail regime (alpha ≥ 2) while the structural-claim sub-population sits firmly in the heavy-tail regime (alpha < 1), validating the intuition that the distribution is genuinely bimodal in tail behaviour — not just in location.

---

## Against the alternatives

The standard UK pricing toolkit for heavy-tail severity has several options. Here is where BNP mixture models sit relative to each.

**Lognormal, Gamma, Burr.** These are single-family distributions fitted to the whole severity range. Their tail behaviour is fixed by the family — lognormal has a fixed tail index of zero (light tail in the GPD sense), which makes it structurally wrong for BI above £200k or subsidence structural claims. These distributions are appropriate for attritional claims, not for tail analysis.

**Spliced composites (Lognormal-GPD, Gamma-GPD).** A body distribution below a threshold, a GPD tail above it. The [composite models in insurance-severity](/2026/03/13/insurance-composite/) implement this properly, with mode-matching for covariate-dependent thresholds and direct ILF output. For most pricing applications, this is the right tool. The limitation is the threshold: it must be chosen, and a single GPD above it cannot represent a mixture of tail regimes.

**TruncatedGPD.** The [truncation-corrected EVT approach](/2026/03/20/your-gpd-is-lying-because-your-claims-are-truncated/) handles policy limits properly and eliminates the truncation bias. Still requires a threshold, still single-regime above it.

**BNP SGG mixture.** No threshold required. Multiple tail regimes identified automatically. Posterior alpha decomposition. But: no covariate support (every risk in the portfolio gets the same model), no truncation correction (a serious problem for real insurance data with policy limits), slow MCMC (40 minutes on n = 1,383 claims in Fortran; expect 2–5 hours in Python), no direct ILF output.

| Method | Threshold needed? | Covariates | Policy limit correction | ILF output | Runtime |
|---|---|---|---|---|---|
| Spliced composite | Yes | Yes (mode-matching) | No | Direct | Seconds |
| TruncatedGPD | Yes | No | Yes | Via EVT | Seconds |
| BMA threshold selection | Averaged | Yes | No | Yes | Minutes |
| BNP SGG mixture | **No** | **No** | **No** | Indirect | **Hours** |

The BNP SGG mixture is uniquely threshold-free and uniquely gives a posterior tail decomposition. It gives up everything else to get there.

---

## Concrete example: UK subsidence claims

Consider a hypothetical subsidence portfolio with 800 large claims above a £10k deductible. The empirical distribution shows the bimodal pattern: a cluster around £12k–£25k (cosmetic/minor structural) and a cluster around £60k–£100k (underpinning, significant structural intervention), with a heavy tail above £200k for complete rebuilds.

A standard GPD fitted above a £50k threshold might give ξ ≈ 0.35. A standard GPD fitted above a £75k threshold might give ξ ≈ 0.50. The difference matters: at a £1m reinsurance attachment point, these two tail indices imply materially different layer costs.

A BNP SGG mixture, fitted to the whole distribution without a threshold, would likely identify two or three clusters with distinct alpha values:
- Cluster 1 (minor claims): alpha ≈ 2.5–3.0 — light tail, all moments finite
- Cluster 2 (structural claims): alpha ≈ 1.2–1.5 — intermediate, finite mean but heavy variance
- Cluster 3 (catastrophic claims): alpha ≈ 0.5–0.8 — heavy tail, infinite variance

The posterior tail decomposition then gives you: roughly 8–12% of this portfolio is generating infinite-variance losses. That is the number you want when deciding whether to buy aggregate stop-loss cover versus per-risk excess of loss, or when stress-testing the XL layer under extreme scenarios. No other method gives you that fraction directly.

---

## The limitations are real

We want to be specific about where BNP SGG breaks down for standard UK insurance use.

**No truncation correction.** Most UK insurance claims data is truncated. Policy limits cap what you observe above the retention. Reinsurance data is available only above the attachment point (left-truncated). IBNR claims introduce right-censoring. The paper provides no treatment for any of this.

This is not a minor omission. When you fit BNP SGG to data that is right-truncated at policy limits, the model sees a pile-up of claims at the limit and interprets it as genuine observations, not censored ones. This pulls alpha estimates upward — the model thinks the tails are lighter than they are. The bias is systematic and does not average out with more data. For real UK motor BI data with heterogeneous policy limits, fitting BNP SGG without a truncation correction will give you systematically wrong tail decompositions. No truncation correction currently exists for this model.

**No covariates.** Every risk in the portfolio gets the same posterior tail decomposition. You cannot ask "what fraction of young driver BI claims have infinite-variance loss potential?" because there is no mechanism to condition on covariates. For personal lines pricing, where per-risk differentiation is the whole point, this is a fundamental limitation. The composite regression models and extreme quantile regression networks both handle covariates; BNP SGG does not.

**Computational cost.** The paper's Fortran implementation takes 40 minutes on n = 1,383 claims. A Python implementation would likely take 2–5 hours for the same dataset. For a UK motor BI book with 50,000+ claims, this is infeasible without significant subsampling or approximation. Variational inference can cut the runtime, but VI with discrete latent variables (cluster assignments) is known to underestimate posterior uncertainty — potentially biasing the tail decomposition estimates you care about most.

**No direct ILF or TVaR output.** Pricing outputs require additional post-processing via posterior predictive simulation. For a team that needs a number for a treaty negotiation on Tuesday morning, this is friction.

---

## When to use it

Given these limitations, where does BNP SGG genuinely add value?

**Portfolio-level tail triaging.** For a specialty lines book with 500–2,000 large losses above a reporting threshold — a mid-tier Lloyd's syndicate's large-loss run, a reinsurer's ground-up analysis of a casualty portfolio — the posterior tail decomposition is a legitimate diagnostic. The question "what fraction of this portfolio is generating infinite-variance losses?" has a defensible answer via BNP SGG that no other method provides. This is valuable input to treaty structure decisions.

**Heterogeneous populations without a known split.** Subsidence (minor vs structural vs catastrophic), bodily injury (soft tissue vs catastrophic disability vs fatality), D&O (settlement-range vs litigation-range vs class action) — these are genuinely multi-regime distributions where the sub-populations are not known in advance. BNP SGG's clustering identifies them without requiring you to pre-specify the number of regimes or their locations.

**Actuarial reserve reviews.** For a closed portfolio of large claims, the BNP SGG posterior gives a more rigorous characterisation of the tail than a mean excess plot or a parametric GPD fitted to the top 200 claims. The posterior credible intervals around the tail fraction estimates are honest — they reflect the genuine uncertainty in how many observations are in the heavy-tail regime.

**Where not to use it:** personal lines pricing with covariates, any dataset with material policy limit truncation, any analysis where you need ILF tables or TVaR numbers on a normal timeline.

---

## Implementation status

The paper's Fortran code is not released. A Python implementation requires around 800 lines: SGG log-likelihood from scratch, N-stable stick-breaking weights, Neal's Algorithm 8 extended to the N-stable process, adaptive Metropolis-Hastings for cluster parameter updates, and post-MCMC tail decomposition. The recommended dependencies are NumPyro and JAX.

We have not built this and are not planning to build it in the near term. Our standard research evaluation gives it 12/20 — below the 14/20 threshold for a full implementation. The novelty is genuine (the SGG + N-stable combination does not exist in Python, and the tail decomposition is a new actuarial diagnostic concept), but the feasibility is low and the value is constrained by the lack of truncation handling and covariates.

The conditions that would change this: the author releases code (cuts implementation effort by roughly 60%); a truncation extension appears in a follow-up paper; or a specific use case emerges where the portfolio-level tail decomposition is the direct deliverable and the data limitations are manageable.

---

## What to use instead, now

For most UK personal lines pricing problems involving heavy tails:

If your problem is **body and tail, with covariates**: use `LognormalBurrComposite` or `LognormalGPDComposite` from `insurance-severity`. These handle covariate-dependent thresholds, produce ILF schedules, and run in seconds. See [Composite Severity Regression](/2026/03/13/insurance-composite/).

If your problem is **tail only, with policy limit truncation**: use `TruncatedGPD` from `insurance-severity`. The truncation correction is free if you have the censoring flag. See [Truncation-Corrected GPD](/2026/03/20/your-gpd-is-lying-because-your-claims-are-truncated/).

If your problem is **per-risk large loss loadings without parametric assumptions**: use quantile GBMs via `insurance-quantile`. See [Large Loss Loading for Home Insurance](/2026/03/04/large-loss-loading-for-home-insurance/).

If your problem is **portfolio-level tail characterisation for specialty lines or casualty, with clean (non-truncated) data, and you have the compute time**: BNP SGG mixture models are the right conceptual tool. Wait for a Python implementation before committing to it — or expect to invest several weeks in the implementation yourself.

The paper is worth reading regardless of whether you implement it. The connection between SGG and GPD (gamma = 1 reduces exactly to GPD, so the whole EVT toolkit nests inside the BNP framework), the N-stable process as a heavier-weighted alternative to the Dirichlet process, and the posterior tail decomposition framework are ideas that should inform how you think about severity modelling even if you never run the MCMC.

---

**Reference:** Nieto-Barajas, L.E. (2026). Modelling heavy tail data with bayesian nonparametric mixtures. arXiv:2602.07228.

---

**Related posts:**
- [Truncation-Corrected GPD: Unbiased Tail Index Under Policy Limits](/2026/03/20/your-gpd-is-lying-because-your-claims-are-truncated/) — the right tool when you have censored claims data and need a tail index
- [Composite Severity Regression: Getting the Tail Right Without Throwing Away the Body](/2026/03/13/insurance-composite/) — spliced models with covariate-dependent thresholds for pricing
- [Large Loss Loading for Home Insurance](/2026/03/04/large-loss-loading-for-home-insurance/) — per-risk tail loadings using quantile GBMs when parametric EVT is too restrictive
