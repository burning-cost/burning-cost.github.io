---
layout: post
title: "Transfer Learning for Thin Portfolios: What Works, What Doesn't, and Why DANN Is the Wrong Tool"
date: 2026-03-31
categories: [machine-learning, pricing]
tags: [transfer-learning, thin-data, DANN, MMD, domain-adaptation, Loke-Bauer, NAAJ-2025, GLMTransfer, credibility, Tian-Feng, Bühlmann-Straub, insurance-thin-data, new-product-pricing, covariate-shift, negative-transfer, SKADA, UK-personal-lines, young-drivers, MGA, motorcycle]
description: "When you launch a new product with no claims history, you borrow from a related portfolio. Transfer learning formalises this. But the most-cited deep learning method for domain adaptation — DANN, with gradient reversal — is actively counterproductive for insurance tabular data. We explain why, and set out what the right tools are."
math: true
author: burning-cost
---

New product pricing is a credibility problem with no credibility. You have zero claims. You have a related portfolio with claims, but it is a different product with different risk dynamics. The question — how do you borrow enough from the source to build a working model without importing systematic errors — is one of the oldest problems in actuarial science. Credibility theory has answered it for sparse cells within a product hierarchy for over a century. Transfer learning offers a different answer for the harder case: borrowing across product lines.

Loke and Bauer's paper "Transfer Learning in the Actuarial Domain: Foundations and Applications" (NAAJ, ahead-of-print, April 2025, DOI: 10.1080/10920277.2025.2489637) addresses this territory directly. The paper is paywalled with no preprint available, which limits what we can say about the specific methods and results. We can confirm the topic — auto insurance transfer learning — but cannot confirm whether the specific approaches covered include DANN or MMD as training objectives. What we can do is set out the technical landscape clearly, explain why the most popular deep learning domain adaptation method is wrong for insurance tabular data, and describe what the `insurance-thin-data` library actually implements.

---

## The two problems that look the same but are not

There is a persistent confusion in the transfer learning literature between two distinct problems. Getting them mixed up leads you toward the wrong methods.

**Covariate shift** is where the feature distribution changes between source and target — $P_S(X) \neq P_T(X)$ — but the labelling function is the same: $P_S(Y \mid X) = P_T(Y \mid X)$. If you are transplanting a model trained on all-ages motor to price young drivers, you might expect covariate shift: young drivers appear in a different region of the feature space (more urban, more small cars, less NCD) but the relationship between features and claims is fundamentally the same physics.

**Domain shift** (or concept drift) is where the labelling function itself changes: $P_S(Y \mid X) \neq P_T(Y \mid X)$. If you are borrowing a personal motor model to price motorcycles, the labelling function changes. The severity distribution, the frequency-speed relationship, the weather effects — all structurally different. The features overlap partially; the risk dynamics do not.

DANN (Domain-Adversarial Neural Networks, Ganin et al., JMLR 2016) is designed for the covariate shift setting and assumes the labelling function is shared. The entire architecture is built around one objective: force the feature extractor to learn representations that are domain-invariant, so the label predictor trained on source generalises to target. When covariate shift is the problem, domain invariance is the solution.

For insurance pricing, this is nearly always the wrong objective.

---

## Why DANN fails for insurance tabular data

DANN trains via gradient reversal. The architecture has three components: a feature extractor, a label predictor, and a domain classifier. During backpropagation, the gradient from the domain classifier is reversed before reaching the feature extractor — this is the gradient reversal layer (GRL). The effect: the feature extractor is simultaneously trained to predict labels well on source data, and to produce representations that the domain classifier *cannot* distinguish as source vs target.

The result is domain-invariant features. Features where the source distribution and target distribution look the same.

For images of cats and dogs from different camera sensors, this is useful — the camera artefacts are noise, not signal. For insurance tabular data, the domain differences are frequently the entire point. Young drivers have higher frequency not because of noise in the feature representation, but because they are genuinely higher-risk. A domain-invariant representation that renders young and all-ages drivers indistinguishable to a classifier has destroyed precisely the information you need to price correctly.

The formal problem is that DANN's invariance assumption — $P(Y \mid h) = P_S(Y \mid h) = P_T(Y \mid h)$ where $h$ is the domain-invariant representation — is violated whenever source and target genuinely differ on pricing-relevant features. This is most insurance transfer learning problems. Young drivers vs all ages: violated. Motorcycle vs motor: violated. New market entrant borrowing from a competitor's book: likely violated.

There is a narrow use case where DANN makes sense for insurance: measurement shift, where the same underlying risk is measured differently in source and target — different telematics device calibration, different claims handling practices that affect how claims are coded, different fraud detection rules. The risk is the same; the measurement differs. Forcing invariance in that setting is correct. But this is a specialised scenario, not the general case.

---

## MMD in two roles: shift test versus training objective

MMD (Maximum Mean Discrepancy) appears in two completely different contexts in the domain adaptation literature, and the distinction matters for knowing what your tooling is actually doing.

**MMD as a shift test** answers the diagnostic question: are the source and target feature distributions materially different? You compute the kernel mean embedding of source and target in a reproducing kernel Hilbert space, take the squared distance, and test whether it is significantly different from zero using a permutation test. The output is a p-value. This is a diagnostic — no model training involved.

`CovariateShiftTest` in `insurance-thin-data` uses MMD in this role. Before any transfer learning, you run the shift test to confirm the transfer is warranted: is the source different enough from target that a naive source model will be miscalibrated, but similar enough that transfer adds value over fitting purely from target data? The test uses a mixed kernel (RBF for continuous features, indicator for categorical) appropriate for insurance tabular data.

**MMD as a training regulariser** is different. Deep Adaptation Networks (Long et al., 2015) add an MMD penalty on intermediate layer activations to the training objective:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(\text{source}) + \lambda \cdot \text{MMD}^2(\phi(h_{\text{source}}), \phi(h_{\text{target}}))$$

where $\phi$ is a kernel embedding and $h$ are intermediate representations. The intent is to encourage representation alignment without the hard invariance constraint of DANN. This is more flexible — you are penalising distribution distance rather than forcing zero distance — but the fundamental problem remains: if source and target should differ (because the pricing-relevant features genuinely differ), penalising that difference is harmful, not helpful.

For deep neural networks with many hidden layers processing rich input spaces, MMD regularisation has genuine value. For CANN architectures on insurance tabular data (two or three hidden layers, 30–150 input features), the representations are close to the raw features — there is limited depth for the regulariser to work with, and the benefit over simpler approaches is marginal.

Neither the DANN approach nor MMD-as-training-regulariser is currently in `insurance-thin-data`, and we assess the gap as low priority for exactly these reasons.

---

## What does work: the Tian-Feng debiasing approach

The `GLMTransfer` implementation in `insurance-thin-data` is based on Tian and Feng's JASA 2023 paper (arXiv:2105.14328): transfer learning under high-dimensional generalised linear models. The approach is conceptually different from DANN and correctly addresses the insurance transfer problem.

The core idea: you have a source GLM coefficient vector $\beta_S$ estimated from large source data, and you want to estimate the target coefficient vector $\beta_T$ from limited target data. Rather than assuming $\beta_T = \beta_S$ (direct transfer) or ignoring source entirely (no transfer), Tian and Feng estimate a debiasing vector $\delta = \beta_T - \beta_S$ from target data using an L1-penalised regression. The final estimate is $\hat{\beta}_T = \beta_S + \hat{\delta}$.

The debiasing vector $\delta$ explicitly captures *how* the source needs to change for the target domain. This is the right formulation for insurance transfer: young drivers probably have similar elasticities to the full motor book on most features (NCD matters in the same direction, vehicle age in the same direction) but differ on a few features (age interactions, vehicle power, urban driving share). The L1 penalty on $\delta$ encourages sparse debiasing — you are allowed to change only the coefficients that genuinely need to change.

This is the anti-DANN. Instead of forcing representations to be identical, you explicitly model the difference. For insurance tabular data, this is the correct instinct.

---

## Credibility vs transfer learning: when each wins

The comparison to Bühlmann-Straub credibility is worth making explicitly because both solve "I don't have enough data to fit a model from scratch."

Credibility theory is vertical borrowing. A sparse cell in your motor book — say, 17-year-olds on classic cars — borrows from its parent segment in the product hierarchy. Linear shrinkage toward the grand mean. Bühlmann-Straub is optimal in MSE sense under conjugate Bayesian assumptions. The `insurance-credibility` library handles this.

Transfer learning is horizontal borrowing. A new motorcycle product borrows from the motor book. A new entrant with no own data borrows from an external dataset. A post-scheme acquisition borrows from the acquired book.

The decision is not close in most cases:

| Scenario | Right tool |
|---|---|
| Young driver sub-segment of motor book | Credibility — same features, same product, sparse cell |
| New motorcycle product, have motor history | Transfer learning |
| MGA with 2,000 policies, similar book available | Transfer learning |
| Post-scheme book acquisition | Transfer learning (shift test first) |
| Niche commercial with sparse cells | Credibility within product, transfer if Z < 0.3 |
| No related portfolio, completely new product | TabPFN (in-context learning from priors) |

The key distinction: credibility works when the source is *more of the same product* — a richer version of what you are trying to price. Transfer learning works when the source is *structurally similar but different* — a different product line whose coefficient structure you want to borrow and then debias.

Transfer learning requires individual policy-level data from the source; credibility works on aggregate summaries. Transfer learning requires the source to have been modelled with a compatible feature set; credibility only requires the same response variable. These constraints mean the two tools have different data requirements, and sometimes you cannot use transfer learning even when you would want to.

---

## The instance reweighting gap

The one genuine gap in `insurance-thin-data` worth noting is instance reweighting — kernel mean matching (KMM) or KLIEP — for the covariate shift case.

The Tian-Feng approach handles domain shift by debiasing coefficients. For the purer covariate shift case (same labelling function, different feature distribution), the right tool is to reweight source observations so their feature distribution matches the target. KMM finds weights $w_i$ for each source observation such that the weighted source distribution is close to the target distribution in RKHS. KLIEP estimates the density ratio $p_T(x) / p_S(x)$ directly and uses it as importance weights.

Neither is in `insurance-thin-data` yet. SKADA (v0.5) implements both but without Poisson exposure weighting. Wrapping SKADA with exposure-weighted Poisson deviance would close this gap in roughly two days of work. We track it as medium priority — it is genuinely missing from the Python insurance ecosystem, but the Tian-Feng debiasing covers most practical cases.

---

## What we currently have in insurance-thin-data

The library (v0.1.6) provides, in the transfer subpackage:

- `CovariateShiftTest` — MMD permutation test to confirm shift before attempting transfer
- `GLMTransfer` — Tian-Feng two-step L1 debiasing with automatic harmful-source detection
- `GBMTransfer` — source-as-offset warm-start for gradient boosted trees
- `CANNTransfer` — pre-train/fine-tune with three fine-tuning strategies (head only, all layers, progressive)
- `NegativeTransferDiagnostic` — catches cases where transfer hurts (source too different)
- `TransferPipeline` — wires shift test, transfer method, fit, and diagnostic in sequence

For UK young-driver pricing, MGA portfolio launch, and post-acquisition integration — the three most common UK use cases — `GLMTransfer` is the right starting point. The shift test should run first; the negative transfer diagnostic should run after. This is precisely the pipeline the library provides.

---

## An honest note on Loke & Bauer

We should be direct about what we know and do not know from Loke and Bauer (NAAJ 2025). The paper is paywalled; no preprint exists. We know it covers transfer learning with auto insurance data. We do not know whether the authors specifically use DANN, MMD-as-loss, or instance reweighting — those attributions in our internal pipeline scans were not sourced from the paper itself. We also know that the author attribution in some earlier internal notes was incorrect: the lead author is Sooie-Hoe Loke, not Kim (there is no Kim co-author).

The methods we describe above — the case for Tian-Feng debiasing over DANN, the role of MMD as shift test versus training regulariser, the credibility vs transfer learning comparison — are grounded in the underlying papers (Ganin et al. JMLR 2016, Tian and Feng JASA 2023) rather than in Loke and Bauer specifically. When we can access the full paper, we will revisit and update.

---

## Related posts

- [Bayesian Hierarchical Models for Thin-Data Pricing](https://burning-cost.github.io/2026/02/17/bayesian-hierarchical-models-for-thin-data-pricing/) — the credibility theory side of the thin-data problem
- [When Credibility Meets CatBoost](https://burning-cost.github.io/2026/02/28/when-credibility-meets-catboost/) — combining credibility shrinkage with gradient boosting for sparse segments
- [Your Book Has Shifted and Your Model Doesn't Know](https://burning-cost.github.io/2026/03/02/your-book-has-shifted-and-your-model-doesnt-know/) — covariate shift in production pricing models and how to detect it
