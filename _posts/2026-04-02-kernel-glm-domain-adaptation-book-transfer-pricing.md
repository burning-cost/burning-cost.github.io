---
layout: post
title: "You've Just Bought a Book. Your Model Doesn't Know These Risks."
date: 2026-04-02
categories: [techniques, pricing, research]
tags: [domain-adaptation, covariate-shift, kernel-glm, portfolio-transfer, pseudo-labeling, density-ratio, importance-weighting, kernel-mean-embedding, ridge-regression, glm, book-transfer, scheme-pricing, arXiv-2603.19422]
description: "When you acquire a portfolio or enter a scheme, your pricing model was fitted on a different risk population. Weill and Wang (2026) give a kernel GLM framework for correcting this without labelled target data. We work through the insurance problem, the theory, and where it sits relative to density ratio methods already in the toolkit."
math: true
author: burning-cost
---

The problem appears in at least three common UK insurance scenarios. You acquire a mid-market motor book from an insurer exiting the market; your existing model was fitted on young-driver heavy direct business and their book skews toward older, low-mileage, rural risks. You win a fleet scheme from a broker; your commercial vehicle model was calibrated on SME owner-operator traffic, not HGV fleets. You extend a home product into high-value property; your severity model has seen very little £500k+ stock exposure.

In all three cases, you have a model fitted on source data $X_s, Y_s$ and you want to price target risks $X_t$ where the distribution of $X_t$ differs from $X_s$ but — and this is the key assumption — the conditional loss distribution $P(Y \mid X)$ is the same. The vehicle age effect on frequency is a physical property of vehicles, not a quirk of your previous book. The occupational loading is real. The structure of the problem transfers; the mix of risks does not.

This is covariate shift. It has a classical remedy: reweight the source observations by $w(x) = p_t(x) / p_s(x)$, the ratio of target to source density, so that the reweighted source looks like the target. Methods like KMM (Gretton), KLIEP (Sugiyama), and ULSIF (Yamada) all estimate this density ratio. We have covered these in the context of insurance-covariate-shift previously.

A paper on arXiv in March 2026 — Weill and Wang (arXiv:2603.19422) — extends the framework in a direction that matters for pricing: it handles the case where you have no labelled target data at all, uses kernel GLMs rather than black-box models, and provides non-asymptotic theory quantifying exactly how much the covariate shift costs you. Here is what it adds and where it fits.

---

## The problem: model selection without target labels

Standard covariate shift correction reweights source data and refits. But this still requires that you evaluate candidate models on the target distribution to decide which model or regularisation setting to use. If you have no observed losses on the target data — which is exactly the situation in the first few months after acquiring a book — you cannot compute a reweighted validation loss. You are flying without instruments.

The standard workaround is to use source validation with importance weights as a proxy. This works reasonably well when the covariate shift is mild. When the shift is severe — as it often is in a book acquisition or scheme entry, where the whole point is that the acquired risks are different — source validation becomes unreliable.

Weill and Wang's solution is pseudo-labelling. Split the source data into two batches. Use the first batch to train candidate models. Use the second batch to build an "imputation model" — a model that predicts outcomes from features, trained on source data. Apply this imputation model to the unlabelled target features to generate synthetic labels. Use those synthetic labels to evaluate the candidate models on the target distribution.

The synthetic labels are wrong — they come from a model fitted on source data. But they capture the covariate structure of the target, and under the covariate shift assumption ($P(Y \mid X)$ equal in both domains), the imputation model's systematic errors cancel out in the model selection criterion, at least approximately. The paper makes this precise via non-asymptotic bounds.

---

## The kernel GLM framework

The model class is kernelised GLMs: linear, logistic, and Poisson regression in a reproducing kernel Hilbert space (RKHS), with ridge regularisation. In insurance terms: claim frequency (Poisson/log-link), claim occurrence (Bernoulli/logit), or severity (Gaussian/identity or Gamma/log) where the linear predictor is replaced by a kernel expansion.

The kernel expansion replaces $X^\top \beta$ with $\sum_{i} \alpha_i k(X_i, X)$ where $k$ is a kernel function — RBF, polynomial, or a product kernel over mixed continuous and categorical features. This gives you nonlinear decision boundaries while staying within a regularised, convex optimisation problem. The ridge penalty $\lambda \|\alpha\|^2$ in RKHS plays the same role as $L_2$ regularisation in standard GLM.

Why does kernel GLM matter here rather than, say, gradient boosting? Two reasons. First, the theoretical guarantees in the paper are derived for this specific class: the excess-risk bound — the gap between the model you select via pseudo-labelling and the oracle model you would select with true target labels — is controlled explicitly in terms of the MMD (maximum mean discrepancy) between source and target distributions, the kernel bandwidth, and the sample sizes. The bound has the form:

$$\mathcal{E}_{\text{target}}(\hat{f}) - \mathcal{E}_{\text{target}}(f^*) \lesssim \text{MMD}(p_s, p_t) \cdot \frac{1}{\sqrt{n_{\text{eff}}}}$$

where $n_{\text{eff}}$ is an "effective labelled sample size" that shrinks as the MMD grows. Severe covariate shift means you are effectively working with a smaller labelled dataset. This is not a surprising result conceptually, but having a quantified, non-asymptotic bound is useful: it tells you whether the shift you are observing (measurable via MMD on the feature distributions alone, without labels) is large enough to matter.

Second, the kernel GLM structure connects naturally to the density ratio reweighting step. In RKHS, the density ratio estimator and the GLM estimator share the same kernel representation, which allows the pseudo-labelling and importance-weighting steps to be computed together efficiently. The paper provides Python code for the full pipeline.

---

## What this adds beyond existing tools

The `insurance-covariate-shift` library handles the standard density ratio pipeline: MMD-based detection, ULSIF/RuLSIF estimation of importance weights, reweighting the source before refitting a GLM. This is the right first step for most covariate shift problems.

Weill and Wang add three things. First, the unsupervised model selection mechanism via pseudo-labelling — handling the case where you cannot evaluate on target outcomes. Second, the non-asymptotic theoretical framing that ties MMD magnitude to expected model selection error. Third, a kernel GLM that is jointly estimated with the domain adaptation step, rather than adapting importance weights and then handing off to a separate GLM.

The practical difference: suppose you have acquired a book and are three months post-close, with 500 observed claims on the new book versus 50,000 on your source portfolio. Standard ULSIF reweighting will give you a corrected source fit, but model selection — how much regularisation, which features to include — is still being guided by the source validation loss. With the pseudo-label approach, you can evaluate model candidates against synthetic target outcomes, which carry the covariate structure of the acquired book even before you have claims experience.

This is not magic. The pseudo-labels are noisy, and the effective sample size degrades as MMD grows. But "noisy model selection guided by the target feature distribution" is better than "model selection guided entirely by the source distribution."

---

## The UK book transfer problem in practice

A UK motor insurer acquiring a book from a distressed carrier in 2026 faces a specific version of this problem. The acquired book likely over-represents:

- Older vehicles (the carrier had a conservative acceptance strategy)
- Rural postcodes (lower acquisition cost, retained longer)
- Older drivers with longer tenure

The acquiring insurer's model was built on a direct-digital book heavy on urban, younger, shorter-tenure risks. The feature distributions are visibly different. MMD on vehicle age alone will be significant; across the joint feature space it will be larger still.

The standard response — hold off on deploying the old model, wait until you have enough claims to refit — is commercially unattractive. You are pricing renewals on the acquired book from month one. The Weill-Wang framework gives a principled approach for deploying a model with covariate-adapted regularisation from day one, with explicit uncertainty quantification via the effective sample size.

The scheme pricing version is slightly different. You have broker-provided portfolio data before quote (the cedant's declarations or MOF data), no loss experience, and you need to price the scheme at outset. Pseudo-labelling here means generating synthetic loss estimates from your existing model to evaluate candidate scheme-adapted models against. The MMD between your existing book and the scheme declaration data is your signal of how much adaptation is needed.

---

## What to be cautious about

The covariate shift assumption — $P(Y \mid X)$ identical in both domains — is the load-bearing assumption of the entire framework, and it is often wrong in practice.

When you acquire a book, the acquired portfolio has a different claims handling culture, different reserving philosophy, different broker relationships, and potentially different policy wording. Even with identical features $X$, the conditional loss distribution $P(Y \mid X)$ may not be the same as your source. A motor book with heavy dependence on a single accident-damage repair network that has since collapsed has different $P(Y \mid X)$ for that feature interaction, regardless of the feature values.

The paper assumes covariate shift and does not attempt to test it. In a real book transfer, the sensible approach is: assume covariate shift as the working hypothesis, apply the kernel GLM adaptation, monitor early loss emergence against the corrected predictions, and test formally whether the residual $P(Y \mid X)$ discrepancy is significant. The Weill-Wang framework does not do this monitoring; it handles the initial adaptation.

The other caution is computational. Kernel methods scale as $O(n^2)$ in memory for the full kernel matrix. With a 50,000-policy source book, you will need the Nyström approximation or random Fourier features to make this tractable. The paper includes an efficient implementation and mentions this limitation explicitly — they test on datasets up to $n \approx 10{,}000$. For UK personal lines scale, approximation methods are not optional.

---

## Where this fits in the toolkit

The covariate shift pipeline for a UK pricing team acquiring a book now has three stages:

**Detection**: MMD on source and target feature distributions. If $\text{MMD}^2$ is not statistically significant, covariate shift is not your problem; differences in $P(Y \mid X)$ may be. If it is significant, proceed.

**Correction**: ULSIF or RuLSIF importance weights from insurance-covariate-shift, reweighting source observations before refitting. This is adequate when you have early target loss experience for validation.

**Unsupervised adaptation**: Weill-Wang kernel GLM with pseudo-labelling, when you have no target labels for model selection. Reduces to the density ratio approach as target observations accumulate. The effective sample size bound tells you when you have accumulated enough target claims to trust standard reweighted validation.

The Python code from Weill and Wang is available alongside arXiv:2603.19422. It is not packaged as a library — it is research code — but the core pseudo-labelling and kernel GLM estimation logic is the right starting point for adding this as a method to insurance-covariate-shift, alongside the existing ULSIF implementation.

We think the theory in this paper is sound and the practical problem it addresses — pricing on an acquired book before you have claims experience — is one of the more common and less well-handled situations in UK general insurance. It is not ready to deploy off the shelf, but it is close enough that a team with a live book transfer in progress should read it.

The paper is open access: [arXiv:2603.19422](https://arxiv.org/abs/2603.19422).
