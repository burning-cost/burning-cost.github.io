---
layout: post
title: "Tab-TRM: The January 2026 Insurance Neural Architecture"
date: 2026-03-25
categories: [techniques, pricing]
tags: [Tab-TRM, transformer, neural-network, GLM, GBM, CatBoost, EBM, tabular, Richman, Wüthrich, Padayachy, recursive, latent-tokens, MTPL, frequency, poisson, insurance-credibility-transformer, insurance-gam, python, architecture]
description: "Tab-TRM (arXiv:2601.07675) is a 14,820-parameter recursive model that beats CatBoost on French MTPL while connecting to GLM theory. We explain the architecture, the numbers, and whether you should care."
---

In January 2026, Padayachy, Richman, and Wüthrich published arXiv:2601.07675: Tab-TRM, a Tiny Recursive Model for insurance pricing on tabular data. It is the third transformer-family architecture from this group in roughly eighteen months — after the [Credibility Transformer](/2026/03/11/credibility-transformer/) (arXiv:2409.16653, September 2024) and work on CAFFT. Each one has improved on the last.

Tab-TRM achieves a test Poisson deviance of 23.589×10⁻² on the French MTPL benchmark. That is better than the Credibility Transformer (23.711×10⁻²), better than CAFFT (23.726×10⁻²), and substantially better than a Poisson GLM (24.102×10⁻²). It does this with 14,820 parameters — a model that fits in your laptop's L1 cache.

This post explains how the architecture works, why the parameter count is not the interesting number, and when you should (and should not) reach for it.

---

## What Tab-TRM is not

Before the architecture, a clarification about the name. Tab-TRM stands for Tabular Tiny Recursive Model. The "TRM" refers to Tiny Recursive Models — a recent NLP paradigm (related to the Hierarchical Reasoning Model) where a small, parameter-shared network is applied recursively rather than stacking distinct layers. It does not use self-attention in the standard transformer sense. There is no query-key-value operation, no multi-head attention matrix. The "Tabular Transformer" framing is loose.

This matters because "transformer" carries expectations — particularly from the FT-Transformer literature (Gorishniy et al., 2021) and from our own [insurance-credibility-transformer](https://github.com/burning-cost/insurance-credibility-transformer) library, which does use proper self-attention. Tab-TRM is closer to a structured state-space model applied iteratively to a fixed feature set.

---

## The architecture

Tab-TRM maintains two learned latent vectors at inference time: an answer token **a** and a reasoning state **z**. Both are d-dimensional (d=28 in the published experiments). The model processes input features into a fixed set of L=9 embeddings (one per covariate), then iteratively refines **a** and **z** to produce the final prediction.

### Feature tokenisation

**Continuous covariates** (5 in the MTPL benchmark: VehPower, VehAge, DrivAge, BonusMalus, Density) go through a piecewise-linear encoding (PLE) layer. Each continuous feature is quantile-clipped, then mapped to (K+1)=11 basis values using decile knots. These are projected to d-dimensional embeddings via a trainable matrix with GELU activation. PLE is an established technique for continuous features in tabular deep learning — it avoids the normalisation sensitivity of raw continuous inputs.

**Categorical covariates** (4 in MTPL: Area, VehBrand, VehGas, Region) are handled via standard entity embedding matrices, with light ℓ₁-ℓ₂ regularisation. Each category level maps to a d-dimensional vector, learned end-to-end.

At this point you have a sequence of L=9 embedding vectors. Nothing unusual — this is the same tokenisation used in the Credibility Transformer and most FT-Transformer variants.

### The recursive core

The interesting part is what happens next. Rather than passing the token sequence through T stacked transformer blocks (which would require T×block_parameters = many parameters), Tab-TRM applies a single recursive network repeatedly. The same two functions, f_z and f_a, are applied T×m times in total.

The update equations are:

**Inner loop** (reasoning refinement, m=3 steps):
```
z^(t,s+1) = z^(t,s) + f_z( Flatten([a^(t), z^(t,s), e_1, ..., e_L]) )
```

**Outer loop** (answer refinement, T=6 steps):
```
a^(t+1) = a^(t) + f_a( Concat(a^(t), z^(t,m)) )
```

Both f_z and f_a are tiny networks with residual connections and LayerNorm. The Optuna-selected configuration uses "zero hidden layers" — meaning each is a single affine map followed by GELU. The model's depth comes entirely from recursion (effective depth ≈ 2×m×T = 36 recursive applications), not from architectural width.

The inner loop integrates all L feature tokens and the current answer state into the reasoning token. The outer loop uses the finalised reasoning token to update the answer. Then the process repeats.

After T=6 outer steps, the final answer token **a^(T)** is passed through a small output FNN (two layers, 19 and 124 hidden units) to produce the predicted log-frequency, which is exponentiated in the usual way. Exposure enters as a standard offset.

---

## The GLM connection

The recursive structure has a genuine mathematical parallel to classical actuarial methods, not merely a suggestive analogy. The authors show it in the paper.

The outer-loop update mirrors iteratively reweighted least squares (IRLS), the algorithm underlying GLM fitting. In IRLS, each step refines parameter estimates given current working residuals. Tab-TRM's outer loop refines the answer token given the current reasoning state. The roles are structurally identical: at each step, a correction is applied to the current estimate based on what the model has "seen" so far.

The inner loop mirrors gradient boosting's stagewise additive correction. Each inner step updates the reasoning state by adding a learned correction — the same structure as fitting a new weak learner to current residuals.

This is not decoration. The authors validate it quantitatively. Least-squares surrogates of the update steps explain more than 90% of variance (R²>0.9). The dynamics are approximately linear. The paper tests a fully linearised variant — removing all nonlinear activations from f_z and f_a — and the deviance drops only 0.1×10⁻² (from 23.589 to 23.698×10⁻² in ensemble). The nonlinear depth is not doing much work. High-dimensional embeddings with linear dynamics capture almost all the structure that the full nonlinear model captures.

What this tells you is that Tab-TRM is not a black box that happens to perform well. The latent dynamics are interpretable, the approximation by linear operators is tight, and the connection to GLM/GBM theory is provable rather than asserted.

---

## The numbers

The benchmark is French MTPL (motor third-party liability frequency), the standard actuarial neural network benchmark introduced by Noll, Salzmann, and Wüthrich. All models use Poisson deviance as the objective. The comparison is on test Poisson deviance ×10⁻²:

| Model | Parameters | Single run | Ensemble (10) |
|---|---|---|---|
| Null (intercept only) | 1 | 25.445 | — |
| Poisson GLM | 50 | 24.102 | — |
| Plain FNN | 792 | 23.819 | 23.783 |
| CAFFT | 27,133 | 23.807 | 23.726 |
| Credibility Transformer | 1,746 | 23.788 | 23.711 |
| Tree-like PIN | 4,147 | 23.740 | 23.667 |
| **Tab-TRM** | **14,820** | **23.666** | **23.589** |

Tab-TRM is the best single model and the best ensemble. It uses 45% fewer parameters than CAFFT while outperforming it by 0.137×10⁻² in deviance terms. The parameter count of 14,820 puts it firmly in the "tiny" category — not as small as the Credibility Transformer (1,746) but roughly comparable to the tree-like PIN.

Two caveats worth stating. First, this is one benchmark. French MTPL is the actuarial community's go-to for this style of comparison precisely because everyone uses it, but it has nine features, all of which are well-understood. Models with clever inductive biases built around this feature set may not generalise to UK motor portfolios with 50+ rating factors, telematics data, or high sparsity in claims experience. Second, ensemble results use ten runs — the variance across seeds is non-trivial, and single-run performance matters for production workflows where you want one deployable model.

---

## What this means for the GLM-vs-GBM decision

The standard framing in UK pricing is: start with a GLM, consider a GBM if the Gini improvement is worth the interpretability cost, never touch neural networks because they are too slow to iterate and too hard to explain to the reserving team.

Tab-TRM does not resolve this trade-off cleanly, but it shifts some of the terms.

The parameter efficiency is genuinely useful. A 14,820-parameter model trains in minutes on CPU. The recursive structure means you do not need a GPU cluster to experiment. The near-linear latent dynamics mean that feature importance is tractable: the authors identify that BonusMalus and DrivAge dominate the continuous embedding shifts, and that Region and VehBrand dominate the categorical contributions. That is what any experienced UK motor pricer would expect — and seeing it emerge from the embedding dynamics is reassuring.

What Tab-TRM does not give you is the factor table. This is the real barrier. An [EBM or ANAM](/2026/03/14/insurance-gam-interpretable-nonlinearity/) produces a per-feature shape function you can put in a pricing committee presentation. Tab-TRM's answer token trajectory is interpretable in an academic sense but not in the "what is the rate relativity for postcode district SW1?" sense. The near-linearity result helps — you could project the answer-token dynamics back to feature space — but no one has yet built the tooling to make this routine.

For portfolios where you are using a [CANN](/2026/03/25/cann-combined-actuarial-neural-network-python/) — a neural network trained as a residual corrector on top of a GLM — Tab-TRM is a natural next step. Replace the plain FNN residual learner with a Tab-TRM, keep the GLM offset structure, and you get a model that inherits the GLM's interpretability while the Tab-TRM handles nonlinear residuals. The architecture is compatible with this approach; the paper does not test it explicitly but the offset mechanism would follow the same pattern used in CANN.

---

## Should you use this?

**Not yet, if your priority is production deployment in a regulated UK pricing environment.** There is no pip-installable implementation. The paper provides PyTorch pseudocode and the mathematical specification is complete enough to implement, but you would be building from scratch, and a from-scratch neural network architecture is a governance conversation before it is a pricing conversation. The FCA expects you to be able to explain your model; "it is a tiny recursive model with 14,820 parameters" is not yet an explanation the industry has settled on a standard framing for.

**Yes, if you are benchmarking candidate architectures for a next-generation pricing model.** Tab-TRM's parameter efficiency and training speed make it practical to include in a benchmarking study alongside CatBoost, EBM, and the Credibility Transformer. Its deviance numbers on MTPL are the best in the current published literature for this architecture class. If you are building a bespoke neural pricing tool on UK commercial lines data where the standard GLM is materially underperforming, this architecture is the right starting point.

**Worth watching for the linear variant.** The linearised Tab-TRM — which removes all nonlinear activations and enforces exact linear state-space dynamics — achieves 23.698×10⁻² deviance in ensemble. That is still better than the Credibility Transformer. Linear latent dynamics means the model is fully explainable as a matrix operation, which is a much easier governance conversation. If someone builds an insurance-tab-trm library that ships the linearised variant with a clean feature-importance API, it becomes a serious competitor to EBMs for portfolio-scale pricing.

**Do not reach for Tab-TRM if CatBoost is already performing well.** The deviance improvement from a tuned GLM to Tab-TRM ensemble is 0.513×10⁻² on French MTPL. Translated into Gini terms, this is modest — particularly on a dataset where the dominant risk factors (BonusMalus, DrivAge) are relatively smooth. On a portfolio with strong interactions, high-cardinality categoricals, or telematics features, the gap may be more material. But Tab-TRM is not a universal upgrade from GBM; it is a specific architectural bet that recursive latent reasoning is better suited to insurance pricing than attention or boosting. On nine clean features, the bet pays out. On 80 noisy features from a UK personal lines book, we do not yet know.

---

## Related work and libraries

The Tab-TRM paper cites the Credibility Transformer and builds on the same French MTPL benchmark. If you want to experiment with transformer-family architectures for insurance, [`insurance-credibility-transformer`](/insurance-credibility/) is the easiest entry point — it ships a full PyTorch implementation with 85 tests and a clean API, and the architecture (attention weight = credibility weight) is better established in the UK actuarial community.

For interpretable alternatives that produce factor tables, [`insurance-gam`](https://github.com/burning-cost/insurance-gam) covers EBM, ANAM, and PIN under one install. The ANAM architecture (Actuarial Neural Additive Model) is closest to Tab-TRM in spirit — both are neural architectures built for insurance tabular data — but ANAM is interpretable-by-construction where Tab-TRM trades that property for better predictive performance.

The Tab-TRM paper is here: [arXiv:2601.07675](https://arxiv.org/abs/2601.07675). It is 30 pages, cleanly written, and the mathematical derivations in the linearisation section are worth reading even if you have no immediate implementation plans. The ETH Zurich actuarial machine learning group is producing architectures with unusually tight connections to actuarial theory. That is not the norm in the broader deep learning literature, and it is worth understanding why these models work before the tooling catches up.
