---
layout: post
title: "Your Thin-Data Quantile Estimates Are Overfit. Here Is the Fix."
date: 2026-04-03
categories: [techniques]
tags: [quantile-regression, distributional-robustness, wasserstein, thin-data, reserves, reinsurance, large-loss, severity, arXiv-2603.14991, pricing]
description: "Standard quantile regression overfits badly on small insurance segments. A new closed-form result from Zhang, Mao and Wang (2026) gives a distributionally robust QR estimator with O(N^-1/2) out-of-sample guarantees — no sub-Gaussian assumption, no extra dependencies."
math: true
author: burning-cost
---

Picture this. You have a reserve segment: 180 historical motor property damage claims. You need the 99th percentile for your large loss loading. The empirical 99th percentile is the second-largest observation in your data set. Your CatBoost quantile model has fitted a smooth surface that, at the 99th percentile level, is largely chasing the two or three largest claims in each subgroup. Everyone in the room knows the number is fragile. Nobody has a principled alternative.

This problem is not niche. It turns up in large loss loading for commercial lines, attachment point selection for per-risk XL reinsurance, SCR capital estimation for thin portfolios, and reserve quantile certification under Solvency II. The common thread is: small N, high quantile, real regulatory or financial consequences if the estimate is wrong in the conservative direction.

A paper from Zhang, Mao and Wang (arXiv:2603.14991, March 2026) solves the methodological problem. It derives a closed-form solution for worst-case quantile regression under Wasserstein distributional uncertainty — a result that did not previously exist. The estimator is practical, has no new dependencies beyond scipy and numpy, and carries provable finite-sample guarantees under conditions that insurance severity data actually satisfies.

---

## Why Standard Quantile Regression Breaks on Small Samples

Standard quantile regression minimises:

$$\min_{\beta, s} \frac{1}{n} \sum_i \rho_\tau(y_i - \beta^\top x_i - s)$$

where $\rho_\tau(u) = u(\tau - \mathbf{1}[u < 0])$ is the check (pinball) loss. The estimator is consistent and asymptotically efficient. The trouble is finite-sample behaviour at extreme quantiles.

At $\tau = 0.99$ with $N = 200$, the estimator is essentially anchored to the two largest observations. Any structural change in the claims distribution — an inflation spike, a new fraud type, a change in policyholder mix — that did not happen to manifest in those two observations will go undetected. The fitted quantile has no memory of how different the distribution *could* be.

L1-regularised quantile regression (LASSO-QR) helps with slope overfitting but does nothing about this core problem: it still has no out-of-sample distributional robustness certificate. Conformalized quantile regression (Romano et al. 2019) gives a coverage guarantee on prediction intervals, but that is a different thing — it tells you about the interval width, not about the robustness of the point estimate itself.

What we want is an estimator that explicitly asks: *given that my training distribution is only an approximation of the true claims distribution, what is the safest quantile estimate I can produce?*

---

## Wasserstein Distance in Plain English

The Wasserstein distance between two probability distributions measures the minimum cost of transporting one distribution into the other, where cost is proportional to how far you move each unit of probability mass. Think of it as the earth mover's distance: you have a pile of earth shaped like your historical claims distribution, and you need to reshape it into the true claims distribution. The Wasserstein distance is the minimum total work required.

A Wasserstein ball of radius $\varepsilon$ around your empirical distribution $\hat{F}_n$ contains all distributions reachable by moving mass a total distance of at most $\varepsilon$. For insurance, this is directly interpretable. The question "how large should $\varepsilon$ be?" translates directly to "how different could next year's claims distribution be from our historical data?" — a question your Chief Actuary can answer from first principles, informed by inflation experience, portfolio churn, or regulatory change.

This interpretability is the reason Wasserstein robustness is more useful for insurance than other notions of distributional robustness. You are not working with an abstract set of distributions satisfying some moment constraint; you are working with the set of distributions reachable from your data by a bounded perturbation that means something physically.

---

## The Closed-Form Result

The distributionally robust quantile regression (WDRQR) problem is:

$$\min_{\beta, s} \max_{F \in \mathcal{B}_p(\hat{F}_n, \varepsilon)} \mathbb{E}_F[\rho_\tau(Y - \beta^\top X - s)]$$

where $\mathcal{B}_p(\hat{F}_n, \varepsilon)$ is the type-$p$ Wasserstein ball of radius $\varepsilon$. The inner maximisation asks: over all distributions that are close to our empirical data, which one gives the worst expected check loss?

The theoretical obstacle is that prior Wasserstein DRO theory (Esfahani and Kuhn 2018) required the loss function to be Lipschitz continuous to power $p$. The check loss is globally Lipschitz — it has slope 1 almost everywhere — but it is not Lipschitz-to-power-$p$ for $p > 1$. So the standard machinery did not apply.

Zhang, Mao and Wang develop new duality arguments that exploit the piecewise-linear structure of the check loss specifically. The result is a clean closed form.

**Theorem 1 (Zhang et al. 2026):** For $p \geq 1$, the WDRQR problem is equivalent to:

$$\beta^* = \underset{\bar\beta}{\arg\min} \ \mathbb{E}_{\hat{F}_n}[\rho_\tau(Y - \bar\beta^\top X - \bar{s}^*)] + c_{\tau,p} \cdot \varepsilon \cdot \|\bar\beta\|_*$$

with intercept correction:

$$s^* = \bar{s}^* + \frac{\varepsilon}{q}\bigl(\tau^q - (1-\tau)^q\bigr) c_{\tau,p}^{1-q} \|\bar\beta^*\|_*$$

where $q = p/(p-1)$ is the conjugate exponent and $c_{\tau,p} = (\tau^q + (1-\tau)^q)^{1/q}$ is a closed-form constant depending only on $\tau$ and $p$.

Two things to notice. First, the worst-case inner problem reduces to standard QR with an additional penalty on the slope norm. This means the reformulation is computationally trivial — it is just regularised quantile regression with a specific penalty weight determined by $\varepsilon$. No bilevel optimisation, no CVXPY, no special solver.

Second, the intercept is adjusted upward (for $\tau > 0.5$) by a term that grows with $\varepsilon$ and with $\|\bar\beta^*\|$. This correction is doing something important: it is addressing the systematic downward bias that empirical high quantiles exhibit in small samples. When the training distribution is uncertain, the robust estimator pushes the quantile up to account for the possibility that the true distribution has heavier tails than the sample suggests.

---

## The p = 1 vs p = 2 Distinction

This is where the theory produces a genuinely surprising result.

For **p = 1** (city-block Wasserstein distance): both the slope and the intercept are invariant to $\varepsilon$. The robust estimator is identical to standard quantile regression regardless of how large you make the robustness ball. The interpretation is that $W_1$ robustness is already implicit in quantile regression — the piecewise-linear structure of the check loss exactly matches the geometry of the $W_1$ ball. You are not getting anything extra by adding $W_1$ uncertainty.

For **p = 2** (Euclidean Wasserstein, the standard choice): the slopes are penalised by $c_{\tau,2} \cdot \varepsilon \cdot \|\beta\|_2$, and the intercept is shifted up as described above. This is where the robustness benefit lives.

Our recommendation: use $p = 2$. The $W_2$ ball weights perturbations by squared distance, which is appropriate when "close distributions" should mean distributions with similar second moments — directly relevant for severity modelling where variance matters as much as mean shift. $p = 1$ is the correct distance for comparing medians but the wrong distance if you care about tail behaviour.

---

## The Finite-Sample Guarantee

The paper's Theorem 3 gives the following. Under a finite $s$-th moment condition with $s > 2$ (no sub-Gaussian assumption, no bounded support required), set the Wasserstein radius as:

$$\varepsilon_N(\eta) = c_\tau \cdot \frac{\log(2N+1)^{1/s}}{\sqrt{N}}$$

Then with probability at least $1 - \eta$ over the training draw, the WDRQR estimator has bounded out-of-sample check loss under the true distribution $F^*$.

The practically important properties:

**Dimension-free rate.** The $O(N^{-1/2})$ rate does not deteriorate with the number of features $d$. Most DRO guarantees scale as $\exp(d)$ or $d^{1/2}$; this one does not. For a pricing model with 20–50 engineered features on a segment of 200 claims, this matters.

**Mild moment condition.** Insurance severity distributions — lognormal, Pareto-tailed, inverse Gaussian — satisfy $s > 2$ for the body of the distribution (the extreme tail requires separate treatment, e.g. GPD extrapolation with EQRN, and this method is not claiming to address catastrophic tail estimation). The guarantee does not require your data to be sub-Gaussian.

**Rate optimal.** The convergence rate matches classical quantile regression. There is no asymptotic efficiency cost from the robustification.

What does this look like in practice? For $s = 4$ (reasonable for motor severity) and $\eta = 0.05$:

| Segment size N | $\varepsilon_N$ (relative to constant $c$) |
|---|---|
| 50 | 0.38 × c |
| 200 | 0.19 × c |
| 500 | 0.12 × c |
| 2,000 | 0.06 × c |
| 10,000 | 0.027 × c |

The constant $c$ requires calibration — it captures how uncertain you are about next year's distribution, not just the statistical noise in this year's sample. For most UK personal lines segments, calibrating $c$ from the observed distribution shift between pre- and post-pandemic years gives a sensible starting point. If your claims distribution shifted by 15% in COVID-adjusted terms between 2019 and 2022, that is information about the plausible magnitude of $\varepsilon$.

---

## Where This Actually Matters for Pricing

**Reserve segment quantiles.** The strongest use case. A commercial property segment with 150 historical claims needs a 99.5th percentile for IBNR reserves or capital allocation under Solvency II Articles 120–126. The empirical 99.5th is the largest observation. WDRQR gives a principled upward correction with an interpretable justification for the Appointed Actuary: "we are allowing for the possibility that the true severity distribution differs from historical by a Wasserstein distance of $\varepsilon$, which corresponds to a shift of approximately X% in average severity."

**Large loss loading.** Many UK actuaries currently use TVaR minus mean, fit from either a parametric distribution or a CatBoost quantile GBM, to derive large loss loading. On thin segments, the GBM is fitting noise at the 99th percentile. A linear WDRQR provides a robust floor estimate with a formal guarantee — useful as a sanity check or as the primary estimate when N is below 500.

**Excess-of-loss reinsurance pricing.** Pricing per-risk XL at an attachment of £500k on a segment with three historical claims above that level is a familiar pain. WDRQR formalises the uncertainty in a way that is defensible to a reinsurer: rather than ad-hoc credibility blending with industry curves, you are explicitly parameterising the distributional uncertainty and optimising the worst case within it. For segments where $N \leq 50$ above attachment, $\varepsilon = 0.3$–$0.5$ is not unreasonable.

---

## The Linear-Only Limitation and What To Do About It

The closed-form requires a linear model $\beta^\top X$. There is no direct extension to trees or GBMs. For most pricing applications with complex non-linear interactions, a purely linear model is inadequate for the mean, let alone the tail.

The practical workaround: treat your CatBoost GBM as a fixed feature transformation and apply WDRQR on GBM-derived features — leaf embeddings, partial dependences, or a set of engineered monotone features from the GBM. This gives you a robust calibration layer on top of the GBM's non-linear predictive structure, rather than a fully robust training procedure. It is not theoretically ideal, but it is considerably better than taking the raw CatBoost quantile prediction at $\tau = 0.99$ on 200 claims as gospel.

A decision guide for which method to use:

- **N > 2,000 per segment, $\tau \leq 0.95$:** Standard CatBoost MultiQuantile. The data is sufficient, use the GBM.
- **N 500–2,000, $\tau \leq 0.95$:** CatBoost with conformal calibration (CQR) for interval coverage. WDRQR as a cross-check.
- **N 50–500, $\tau \geq 0.95$:** WDRQR as primary estimate, possibly with GBM feature transforms. This is the target regime.
- **$\tau > 0.999$, any N:** GPD tail extrapolation (EQRN). WDRQR is not designed for extreme tail estimation beyond the sample range.

---

## When to Use This: The Three-Condition Rule

WDRQR is worth deploying when all three of the following hold:

1. **N < 500 in the segment.** Below this, empirical high quantiles are unreliable enough that formal robustification pays off.
2. **Quantile level $\tau \geq 0.95$.** The method's advantage over standard QR is largest at extreme quantiles; at $\tau = 0.75$ the improvement is marginal.
3. **The segment is subject to structural change risk.** Inflation, regulatory change, new fraud typology, portfolio mix shift. If you genuinely believe your historical data is an i.i.d. draw from a stable distribution, standard QR is fine. If you believe the world could look somewhat different from history — and for UK insurance in 2026, it can — then the robustness guarantee is doing real work.

When all three hold, the question is not whether to use WDRQR, it is how to calibrate $\varepsilon$. Start with the Theorem 3 schedule using $s = 4$ and $\eta = 0.05$, then sense-check the implied intercept correction against your actuarial judgement on what a plausible distribution shift looks like for this segment. The correction should be uncomfortable but not absurd.

---

The paper is Zhang, Mao and Wang, "Wasserstein Distributionally Robust Quantile Regression," arXiv:2603.14991, March 2026. The core result is a closed-form equivalent for the worst-case quantile regression problem over a Wasserstein ambiguity set — a result that did not exist before this paper, and one that fills a specific gap in the actuarial modelling toolkit for thin-data high-quantile estimation.
