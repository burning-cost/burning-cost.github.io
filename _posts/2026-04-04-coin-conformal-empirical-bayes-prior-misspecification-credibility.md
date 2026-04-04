---
layout: post
title: "When Empirical Bayes Goes Wrong: Lessons From Conformal Inference for Insurance Credibility"
date: 2026-04-04
categories: [credibility, techniques]
tags: [empirical-bayes, conformal-inference, buhlmann-straub, prior-misspecification, credibility, fdr, multiple-testing, insurance-credibility, arXiv-2604.01629, coin, hierarchical-bayes, structural-parameters, scheme-pricing]
description: "Seo and Lim (arXiv:2604.01629, April 2026) show that standard empirical Bayes methods inflate FDR to 0.25–0.35 when the prior is misspecified — even when the nominal rate is 0.1. The mechanism is directly analogous to the K = v/a instability that makes Bühlmann-Straub unreliable with fewer than ten groups. COIN is not a fix for credibility. But it makes the underlying problem unusually clear."
math: true
author: Burning Cost
---

A new paper from Seo and Lim (arXiv:2604.01629, April 2026) documents something that any actuary who has fitted Bühlmann-Straub to a small scheme book will recognise in their bones, even if they have never framed it this way: empirical Bayes goes badly wrong when you estimate the prior from the same data you are trying to draw inference about, and when that prior is structurally misspecified.

The paper is in the genomics literature — it is about false discovery rate control in differential expression testing, not about credibility premiums. COIN (Conformalized Empirical Bayes Normal Mean Inference) is not something we are building into [`insurance-credibility`](/insurance-credibility/). The objectives are structurally different and the paper does not claim otherwise.

But the simulation results are a useful provocation. The FDR inflation that Seo and Lim demonstrate under prior misspecification is the testing-framework analogue of the structural parameter instability that makes B-S credibility unreliable in the settings where pricing actuaries most want to use it. It is worth understanding why.

---

## What COIN is about

The setup is the heteroscedastic normal means problem. You observe $n$ pairs $(X_i, S_i^2)$ where:

$$X_i \mid \mu_i, \sigma_i^2 \sim N(\mu_i, \sigma_i^2)$$

$$S_i^2 \mid \sigma_i^2 \sim \frac{\sigma_i^2}{\nu} \chi^2_\nu$$

The $\sigma_i^2$ are drawn from some unknown distribution $G$, and the true means $\mu_i$ may or may not be zero. You want to test $H_{0i}: \mu_i = 0$ for all $i$ simultaneously while controlling FDR at level $\alpha$.

Standard empirical Bayes approaches — LS, MixTwice, gg-Mix — estimate $G$ from the data and use the estimated prior to compute posterior probabilities, which drive the testing procedure. The problem the paper identifies is that this works well when $G$ is correctly specified, and fails badly when it is not.

The failure is not modest. In simulations where the true distribution of $\mu_i$ and $\sigma_i^2$ is dependent (heavier-tailed than the assumed mixture prior, or asymmetric in ways the parametric family cannot capture), standard EB methods reach empirical FDR of 0.25–0.35 against a nominal target of 0.1. The excess is not random noise — it is structural, consistent, and persists as the sample grows.

COIN's fix is to conformalize the decision threshold: rather than trusting the EB prior estimate, it generates pseudo-calibration data from the estimated prior and uses conformal prediction logic to set the rejection threshold in a way that is robust to prior misspecification. The asymptotic FDR guarantee holds without requiring $G$ to be correctly specified.

---

## The B-S analogy

Bühlmann-Straub is also an empirical Bayes procedure. You observe claim counts or loss ratios across $r$ groups over $T$ periods. The structural parameters — the expected value of the hypothetical means $\mu$, the within-group variance $v$, and the between-group variance $a$ — are estimated from the same data you use to compute credibility factors.

The credibility weight for group $i$ is:

$$Z_i = \frac{n_i}{n_i + K}, \quad K = \frac{v}{a}$$

where $n_i$ is the exposure for group $i$. $K$ controls how much each group is pulled towards the portfolio mean. A high $K$ (high within-group noise relative to between-group variation) means you trust the portfolio more; a low $K$ means you trust the group's own experience more.

The problem is that $\hat{a}$ — the estimated variance of the hypothetical means — is the hardest structural parameter to pin down. The standard Bühlmann estimator is:

$$\hat{a} = \frac{1}{r-1}\left(\sum_i n_i \bar{X}_{i\cdot}^2 - \frac{\hat{v}}{n}\right) - \frac{\hat{v}}{r}$$

This is a moment estimator with a variance that depends on $r$, the number of groups. With $r < 10$, the sampling variance of $\hat{a}$ is large enough that $K = v/\hat{a}$ is effectively noise. You might get $K = 2$ or $K = 20$ from the same underlying data-generating process across different realisations of the portfolio. The credibility factors move accordingly.

This is exactly prior misspecification by estimation noise. The "prior" implicit in B-S — the mixing distribution on the hypothetical means $\theta_i$ — is estimated from the $r$ observed group means. When $r$ is small, that estimate is unstable, and the fitted prior does not reliably represent the actual between-group variation. The posterior credibility premiums are then systematically distorted in the same way that the COIN paper's simulations show FDR being inflated: not randomly, but in a direction that depends on how the noise in $\hat{a}$ resolves.

The COIN simulations make this vivid because FDR is a clean number. You set a target of 10%, you get 25–35%, and you can see the gap. In credibility, the distortion is harder to measure — you do not typically know the true hypothetical means, so you cannot compute an analogue of FDR. But the mechanism is the same. The estimated prior is wrong, and the posterior is corrupted.

---

## Why COIN is not the answer for credibility

It is tempting to ask whether conformalizing B-S would fix the problem. It would not, for three reasons.

**The objective is different.** COIN controls a testing error rate. It produces binary decisions: this group is genuinely different from the portfolio mean, or it is not. Credibility pricing needs something different: a continuous credibility factor $Z_i \in [0,1]$ that smoothly blends group experience with portfolio experience, and a credibility premium $P_i = Z_i \bar{X}_{i\cdot} + (1 - Z_i)\hat{\mu}$ that is the minimum-MSE estimate of the hypothetical mean. There is no natural conformal wrapper for that estimation objective.

**The data structure is different.** COIN's conformal calibration strategy — COIN-FS splits the feature indices into folds; COIN-SS splits individual observations — requires enough data to form a meaningful calibration set. In typical UK scheme pricing, you have $r = 5$–50 groups and $T = 3$–7 years. COIN-SS halves that already-sparse data. COIN-FS is inapplicable when the "features" are simply the time periods. The data regime is too thin for conformal calibration to be competitive.

**The normal mean assumption.** B-S in its standard form handles a more general family than normal means — Poisson-Gamma, Gamma-inverse-Gamma, and others. The conformal score function in COIN is derived for normal data. Generalising it to the skewed distributions that dominate motor and property claims is not straightforward.

---

## What the full posterior does instead

The correct response to prior misspecification in credibility is not to conformalize the empirical Bayes procedure. It is to abandon empirical Bayes in favour of a full Bayesian model.

`PoissonGammaCredibility` in the [`insurance-credibility`](/insurance-credibility/) library fits the full conjugate posterior:

$$\theta_i \sim \text{Gamma}(\alpha, \beta)$$

$$N_{it} \mid \theta_i, e_{it} \sim \text{Poisson}(\theta_i e_{it})$$

Rather than estimating $\hat{\alpha}$, $\hat{\beta}$ as point values and treating them as the true prior, the full posterior integrates over the uncertainty in the hyperparameters using the marginal likelihood:

$$p(\alpha, \beta \mid \text{data}) \propto p(\text{data} \mid \alpha, \beta) \cdot p(\alpha, \beta)$$

The credibility premium for group $i$ is then:

$$E[\theta_i \mid \text{data}] = \int E[\theta_i \mid \text{data}, \alpha, \beta] \cdot p(\alpha, \beta \mid \text{data}) \, d\alpha \, d\beta$$

This is not a small technicality. When $r < 10$ and $\hat{a}$ is noisy, the standard empirical Bayes approach treats the estimated prior as if it were the true prior. The full posterior propagates the uncertainty in the prior parameters through to the credibility premium. The resulting premium intervals are wider — correctly wider — when the structural parameters are uncertain.

The COIN simulations tell you quantitatively what the cost of ignoring this is: if you use a mis-estimated prior as though it were truth, your posterior inference is wrong in a systematic and predictable direction. The magnitude — FDR going from 10% to 25–35% in realistic simulation settings — is a useful empirical anchor for the seriousness of the problem. Our view is that the equivalent distortion in credibility premiums is of similar practical significance and is underappreciated.

---

## The small-$r$ regime

To be concrete about where this bites: the structural parameter instability in B-S is most acute when:

- $r < 10$ groups (affinity schemes, geographic regions in specialist lines, underwriting year cohorts)
- The group-level exposures are unbalanced (some groups much larger than others)
- The between-group variation $a$ is small relative to the within-group noise $v$ — that is, the true $K$ is large and the data is telling you to trust the portfolio heavily, but the estimate of $K$ is itself unreliable

UK MGAs and specialist scheme managers typically operate in this regime. A motor fleet affinity book with 6–8 fleet accounts, each with 3–5 years of loss data, is a representative case. Fitting B-S to this data and reading the credibility weights at face value is a mistake. The weights will vary substantially across reasonable re-specifications of the prior functional form and across bootstrap resamples of the data.

The `HierarchicalBuhlmannStraub` class handles two-level hierarchies but does not address the structural parameter uncertainty problem directly. For the small-$r$ regime, `PoissonGammaCredibility` with full hyperparameter posterior is the right tool. If the data are too thin even for that — $r < 5$ and fewer than three periods — then the honest answer is that the data do not support credible group-level adjustments, and the appropriate prior loading is zero.

---

## What COIN does contribute

The paper's conceptual contribution — wrapping an empirical Bayes procedure in a conformal layer to gain robustness against prior misspecification — is genuinely new. A future paper that applies this to credibility estimation rather than multiple testing would be interesting. It would need to address the continuous-output requirement (a conformal credibility factor rather than a binary decision), the sparse-data constraint, and the non-normal likelihood. None of those are trivial.

There is also a narrower application where COIN's framework is directly relevant: if the underwriting question is "which of our 200 affinity schemes have a genuinely elevated risk profile versus random fluctuation?", that is a multiple testing problem and COIN applies. Screen-and-classify, not price-continuously. For that decision structure, with enough schemes to make the conformal calibration feasible, COIN or its successors are the right framework.

---

## The paper

Seo, S. and Lim, J. (2026). COIN: Conformalized Empirical Bayes Normal Mean Inference. arXiv:2604.01629, submitted 2 April 2026. The FDR inflation simulations are in Section 4.2; the robustness theorem (Theorem 1) establishes asymptotic FDR control with finite-sample guarantees only in the oracle case (known $G$). Not yet peer reviewed.

---

## Related

- [Bühlmann-Straub Credibility in Python: The B-S Model and When It Fails](/2026/02/19/buhlmann-straub-credibility-in-python/) — the structural parameter estimation problem and when B-S breaks down
- [Does Bühlmann-Straub Credibility Actually Work for Insurance Pricing?](/2026/03/28/does-buhlmann-straub-credibility-actually-work/) — empirical evaluation on motor claims data
- [Bayesian Hierarchical vs. Bühlmann-Straub Credibility](/2026/03/31/bayesian-hierarchical-credibility-buhlmann-straub/) — the full posterior approach versus empirical Bayes
- [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility) — BuhlmannStraub, PoissonGammaCredibility, HierarchicalBuhlmannStraub, DynamicPoissonGammaModel
- [Credibility Theory in Python: A Complete Bühlmann-Straub Tutorial](/blog/2026/04/04/credibility-theory-python-buhlmann-straub-tutorial/) — practical implementation of the B-S and Poisson-Gamma models, including the structural parameter diagnostics that reveal the instability this paper formalises
