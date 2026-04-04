---
layout: post
title: "Threshold-Free Heavy-Tail Severity: A Genuinely Novel Idea That Isn't Ready for Production"
date: 2026-04-03
categories: [techniques, research]
tags: [severity, heavy-tail, bayesian-nonparametric, evt, gpd, shifted-gamma-gamma, mcmc, reinsurance, insurance-severity, spliced-composites, EQRN, tail-index, arXiv-2602.07228, Nieto-Barajas, watch-not-build]
description: "Nieto-Barajas (arXiv:2602.07228) proposes a Bayesian nonparametric mixture of Shifted Gamma-Gamma distributions that eliminates EVT threshold selection entirely and produces a posterior decomposition of your book into heavy, intermediate, and light tail regimes. It scores 12/20 — elegant theory, impractical for production pricing today."
author: burning-cost
math: true
---

Threshold selection is the open wound of extreme value modelling. You have 1,400 bodily injury claims, you want to know which ones are genuinely heavy-tailed, and you reach for a Generalised Pareto Distribution. The GPD is the right tool — the Pickands-Balkema-de Haan theorem guarantees it for sufficiently high thresholds — but the question "sufficiently high" turns out to be doing a lot of work. The mean excess plot gives you a range. The Hill estimator gives you a different range. Profile likelihood gives you a third. You pick something, you know it is arbitrary, and you know that your tail index estimate is entangled with that arbitrary choice in ways that are very difficult to quantify.

Luis Nieto-Barajas at ITAM Mexico has published a paper (arXiv:2602.07228, February 2026) that proposes eliminating the threshold problem entirely. No threshold. You fit the whole distribution in one model, and the posterior distribution of a tail-control parameter tells you, for each observation, how likely it is to belong to the infinite-variance regime. We think this is a genuine methodological contribution. We also think pricing teams should not build it yet.

---

## What the paper actually does

The core idea is a Bayesian nonparametric mixture of Shifted Gamma-Gamma (SGG) distributions. The SGG is a four-parameter family defined by marginalising a conditional Gamma over a Gamma mixing distribution:

$$X - \mu \mid Y \sim \text{Gamma}(\gamma, Y), \quad Y \mid \alpha, \beta \sim \text{Gamma}(\alpha, \beta)$$

Marginalising over $Y$ gives the SGG density:

$$f(x \mid \mu, \gamma, \alpha, \beta) = \frac{\beta^\alpha}{\mathrm{B}(\alpha,\gamma)} \cdot \frac{(x-\mu)^{\gamma-1}}{(\beta + x - \mu)^{\alpha+\gamma}}, \quad x \geq \mu$$

The parameter $\alpha$ is the tail index. When $\alpha < 1$, the SGG has infinite variance. When $\alpha < 0.5$, it has an infinite mean. When $\gamma = 1$, the SGG reduces exactly to the GPD with $\xi = 1/\alpha$ and $\sigma = \beta/\alpha$ — so the BNP mixture nests GPD as a special case and is strictly more flexible for body shapes where $\gamma \neq 1$.

The mixing distribution over SGG parameter vectors is a Normalised Stable (N-stable) process rather than the more familiar Dirichlet process. The N-stable process produces heavier-tailed cluster weight distributions, which is appropriate for extreme value contexts where a small number of clusters may account for most of the large observations. The concentration parameter $\nu \in (0,1)$ controls how many clusters the model generates; at $\nu = 0.5$ (the paper's recommended default) you get somewhat more diffuse weight distributions than the Dirichlet process, which tends to under-cluster for heavy-tailed data.

MCMC proceeds via a generalised version of Neal's Algorithm 8: a five-step Gibbs sweep that alternates between updating latent variables, cluster assignments, cluster parameters, and the concentration parameter. The latent $Y_i$ update is fully conjugate (no Metropolis-Hastings step needed). The cluster parameter update uses adaptive Metropolis-Hastings with batched tuning, targeting an acceptance rate of 30–40%.

---

## The posterior alpha decomposition

This is the part of the paper we think is genuinely novel in an insurance context. After running MCMC, for each observation $i$ you have a posterior distribution over its tail index $\alpha_i$. You can compute:

- $P(\alpha_i < 1)$: probability the observation belongs to the **infinite-variance** regime — the reinsurance concern zone
- $P(1 \leq \alpha_i < 2)$: probability of the **intermediate** regime — finite mean, infinite variance — still dangerous for excess-of-loss treaties
- $P(\alpha_i \geq 2)$: probability of the **light tail** regime — all moments finite

The paper applies this to 1,383 accident insurance claims (median USD 6,000, mean USD 17,158 — a distribution with the kind of right skew you would recognise from a UK motor bodily injury book). The posterior tail decomposition comes out at: P(alpha < 1) = 14.4%, P(1 <= alpha < 2) = 58.8%, P(alpha >= 2) = 26.8%. The model fits 19 posterior clusters to this data. Runtime: approximately 40 minutes on an Intel Xeon 3.00 GHz in Fortran.

No other method in the insurance-severity toolkit answers the question: "What fraction of this book has infinite-variance loss potential?" Not the spliced composites. Not TruncatedGPD. Not EQRN. The posterior alpha decomposition is a novel actuarial diagnostic.

What it tells you is directly actionable for one specific decision: whether to buy excess-of-loss versus aggregate stop-loss cover. If 14% of your observations are in the infinite-variance regime, the risk profile of your excess-of-loss programme looks very different from a portfolio where the answer is 2%. It is also directly relevant to reserve review discussions — this is a more rigorous basis for assessing tail regime than staring at a mean excess plot.

---

## How it compares to the existing toolkit

We run the [insurance-severity library](https://github.com/burning-cost/insurance-severity) with spliced composites, TruncatedGPD, and EQRN for the standard heavy-tail toolkit. The BNP SGG approach has a distinct profile.

**Against TruncatedGPD.** The existing `evt.py` fits a profile-MLE GPD on exceedances above a fixed threshold, with per-observation truncation corrections for policy limits. TruncatedGPD wins on practical data handling: it handles right truncation at policy limits, which is standard for UK insurance data with heterogeneous limits. The BNP SGG paper provides no treatment of truncation or censoring at all. Fitting it naively to truncated insurance data will produce biased alpha estimates — specifically, it will see truncated large claims as evidence of lighter tails, systematically underestimating the heavy-tail fraction. This is a systematic bias, not noise, and it rules out direct application to most UK motor or home claims portfolios without substantial additional work.

**Against spliced composites.** `LognormalBurrComposite` and `GammaGPDComposite` fit a body distribution below a threshold and a tail distribution above it, with profile likelihood threshold selection. They produce ILFs and TVaR directly. The BNP SGG model eliminates the threshold but gives up speed (MLE in seconds vs MCMC in tens of minutes), ILF/TVaR interpretability (possible from posterior predictive samples, but not a standard output), and the ability to do per-risk estimation. For a pricing team that needs an ILF schedule by end of week, the composite models are the right tool.

**Against BMA threshold selection.** The paper from Jessup, Mailhot, and Pigeon (arXiv:2504.20216) proposes Bayesian model averaging over candidate thresholds with covariate-dependent threshold selection. Still unbuilt in the pipeline (it scores higher at 16/20, same as BayesianGPD), but when it is built it will handle covariates and covariate-dependent threshold uncertainty. BNP SGG is more principled — it eliminates the threshold entirely rather than averaging over it — but the BMA approach retains interpretable GPD parameterisation and covariates. For covariate-rich insurance pricing, BMA is more practically useful even if BNP SGG is theoretically cleaner.

**Against EQRN.** Extreme quantile regression neural networks (the `spqrx` module) do per-risk tail estimation with covariates. BNP SGG cannot. For personal lines pricing, there is no contest: EQRN wins because it can produce a tail estimate for each individual risk given their rating factors. BNP SGG is a portfolio-level tool.

**Against BayesianGPD.** The BayesianGPD paper (arXiv:2510.14637, KB 3017) targets honest credible intervals under serial dependence — ARMA, GARCH, and copula data-generating processes — while still requiring a threshold. Different problem spaces. BayesianGPD scores 16/20 and is higher priority in the build queue.

| Method | Threshold | Covariates | Truncation | Speed | ILF/TVaR | Tail UQ |
|--------|-----------|------------|------------|-------|-----------|---------|
| TruncatedGPD | Required | No | Yes | Fast | Yes | Profile lik |
| Spliced MLE | Required | Partial | No | Fast | Yes | Bootstrap |
| BMA threshold | Averaged | Yes | No | Moderate | Yes | BMA weights |
| BayesianGPD | Required | No | No | MCMC | No | Honest CrI |
| EQRN | Required | Yes | No | GPU | No | Bootstrap |
| **BNP SGG** | **None** | **No** | **No** | **Slow** | **Indirect** | **Posterior** |

---

## Why we are not building it yet

The score is 12/20: Novelty 4/5, Value 3/5, Feasibility 2/5, Strategic fit 3/5. Below the 14/20 threshold we use to decide whether something enters the build queue.

The feasibility problem is straightforward. The paper does not release code — the Fortran implementation is described but not provided, so any Python port requires full reimplementation from the mathematical description. Our estimate is approximately 800 lines: the SGG distribution class, the N-stable stick-breaking weights, the five-step MCMC kernel with adaptive Metropolis-Hastings, posterior diagnostics, and plot methods. That is not unusual for a research implementation, but the computational cost makes it hard to justify. Python or NumPyro with JAX is expected to be 3–8x slower than Fortran for non-conjugate BNP (standard literature benchmark for Neal-8 style algorithms). That means roughly 2–5 hours for 1,383 observations and infeasible at n > 5,000 without subsampling. A small Lloyd's syndicate's large-loss run might comfortably reach that scale.

The truncation problem is more fundamental. UK insurance data routinely has policy limits — right truncation at the limit for each claim — and reinsurance data has attachment point left truncation. Without truncation corrections, alpha estimates from the BNP SGG model are systematically biased for UK data. Adding truncation corrections requires the SGG CDF in closed form (which we do not know exists; the integral structure suggests a hypergeometric connection but this is unverified) or numerical CDF evaluation per observation per MCMC iteration, which would be another significant slowdown.

The no-covariate problem limits where the tool is even conceptually applicable. For portfolio-level tail diagnostics on a fixed book of large claims — a Lloyd's specialty syndicate reviewing its reserve adequacy, or a reinsurer triaging whether a cedant's book warrants infinite-variance treatment — covariates are not needed. For any pricing application, they are essential.

Build conditions: the paper's author releases code (reducing the engineering effort by roughly 60%); or a follow-up paper adds a truncation extension; or a specific client need emerges for portfolio tail triaging that cannot be served by the existing toolkit.

---

## What to do now

The posterior alpha decomposition is the one output from this paper that has genuine near-term value, and it does not require the full BNP machinery to approximate. If you need a portfolio-level tail regime assessment today, the practical approach is:

1. **Use the Hill estimator with bootstrap uncertainty** to get a point estimate and interval for the tail index across the book. This is fast and already in the toolkit.
2. **Fit a spliced composite with profile likelihood threshold selection** to get the body/tail structure explicitly modelled. The `GammaGPDComposite` gives you a GPD shape parameter $\xi = 1/\alpha$ for the tail, from which you can compute the equivalent tail index.
3. **Use BMA threshold selection** (once built) to get an averaged tail index estimate that is less sensitive to the threshold choice.

None of these give you the observation-level posterior alpha decomposition — "this specific claim has a 73% probability of belonging to the infinite-variance regime" — which is genuinely unique to the BNP approach. If that granularity matters for your analysis, the BNP SGG is the only method that provides it. But for most insurance pricing and reserving workflows, the portfolio-level aggregate fractions are sufficient and the existing toolkit can approximate them.

---

## The verdict

Nieto-Barajas has solved a real problem — the threshold dependence of EVT estimation — with a method that is mathematically coherent and produces a novel diagnostic that no other tool in the insurance-severity space provides. The posterior alpha decomposition has genuine value for specialty lines reserve review and reinsurance structure decisions.

It is not ready for production pricing. No truncation handling, no covariates, 40-minute Fortran runtimes for 1,400 claims (worse in Python), and a full reimplementation burden with no code release. For the typical UK motor or commercial lines pricing team, spliced composites and EQRN are the right tools today. For a specialty reinsurer with a small book of large losses and a need to characterise the tail regime formally, this is worth watching.

We are categorising it as a long-horizon watch item. The core idea is good enough that if truncation extensions and code release happen, it moves up the queue quickly.

---

**Paper:** Nieto-Barajas, L.E. (2026). 'Modelling heavy tail data with bayesian nonparametric mixtures.' arXiv:2602.07228. stat.ME. Submitted 6 February 2026.

**Related:**

- [Spliced Severity Distributions: When One Distribution Isn't Enough](/2025/03/15/spliced-severity-distributions-when-one-distribution-isnt-enough/) — the practical case for composite models over single-family distributions, with full `insurance-severity` worked example
- [insurance-severity](https://github.com/burning-cost/insurance-severity) — spliced composites, TruncatedGPD, EQRN, and the full production toolkit
