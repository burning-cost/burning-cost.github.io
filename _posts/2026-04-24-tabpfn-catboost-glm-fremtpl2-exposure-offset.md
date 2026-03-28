---
layout: post
title: "TabPFN vs CatBoost vs GLM on freMTPL2: The Exposure Offset Problem"
date: 2026-04-24
categories: [benchmarks, pricing, foundation-models]
tags: [TabPFN, CatBoost, GLM, Poisson, exposure-offset, freMTPL2, benchmark, frequency-modelling]
description: "Three-way benchmark on 677K French motor policies. TabPFN cannot handle log-exposure offsets — the structural limitation that makes it unviable for bread-and-butter Poisson frequency modelling."
---

Our [previous post on TabPFN](/2026/03/13/insurance-tabpfn/) made the case for foundation models in thin segments: fewer than two thousand policies, unstable GLM parameters, no natural parent segment to borrow from. That case still stands.

This post is the other side of it.

We ran a three-way benchmark — Poisson GLM, CatBoost, and TabPFN v2 — on freMTPL2, the standard French motor third-party liability dataset used in actuarial teaching and pricing research. 677,991 policies. The task is Poisson frequency: model claim counts with a log(Exposure) offset that encodes time-at-risk. CatBoost and GLM handle this natively. TabPFN cannot. The results are not close, and the reason matters enough to write down carefully.

The full benchmark notebook is [on GitHub](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/tabpfn_benchmark.py).

---

## The dataset and the task

freMTPL2 is OpenML dataset 41214. It covers French private car insurance policies, each with a claim count (`ClaimNb`), a policy exposure in years (`Exposure`), and features including driver age, vehicle age, vehicle power, vehicle brand, region, population density, and gas type. After capping `ClaimNb` at 4 and `Exposure` at 1 (standard preprocessing), the full dataset is 677,991 rows. We use a stratified 80/20 split: 542,393 training policies, 135,598 test.

The task is standard actuarial Poisson frequency modelling. The target is the expected claim count for a policy, conditional on its features and its exposure. The correct model specification is:

```
log(E[ClaimNb]) = log(Exposure) + β·X
```

The `log(Exposure)` term is a fixed offset — not a coefficient to be estimated, not a feature to be learned. It encodes the actuarial truth that a policy on risk for six months is exposed to half the claim potential of an identical policy on risk for twelve months. Remove the offset and your model is systematically wrong in proportion to how unequal exposures are across your book.

In freMTPL2, `Exposure` varies from near zero (mid-term cancellations) to 1.0 (full annual policies). The distribution is not uniform. This matters.

Evaluation metrics: Poisson deviance (mean, lower is better), Gini coefficient (Lorenz-curve based, exposure-weighted, higher is better), and actual-versus-expected by predicted decile.

---

## The three models

**Poisson GLM.** Fitted with statsmodels, log link, `log(Exposure)` as offset. Ordinal encoding for the categorical features (`Area`, `VehBrand`, `VehGas`, `Region`). This is the textbook formulation. It fits on the full 542K training set.

**CatBoost.** `loss_function='Poisson'`, exposure handled via the `baseline` parameter — which is exactly the CatBoost mechanism for fixed offsets. Set `baseline = log(Exposure)` for each row, and CatBoost's Poisson objective treats it as a fixed term in the linear predictor while learning the tree structure. This is the correct gradient boosting formulation for insurance frequency. Also fits on the full 542K training set.

**TabPFN v2.** Hard row limit of 10,000 training rows. We subsample to 10K, oversampling claims to 25% claim rate (versus ~6% natural incidence) to give the model more signal on the positive class. And this is where the structural problem begins.

---

## Why TabPFN cannot handle exposure offsets

TabPFN v2 is a transformer pretrained on millions of synthetically generated tabular datasets. At inference time, your training data becomes context — the model produces predictions in a single forward pass, no gradient descent. The pretraining prior covers a wide range of functional forms, feature correlations, and noise structures.

What it covers is regression and classification. What it does not have is any mechanism for a fixed offset in a generalised linear model sense. There is no `offset` parameter. There is no Poisson family. The regression prior is Gaussian-equivalent — it is designed for targets that behave like normally distributed noise around a learnable function. Poisson counts with varying exposure are not that.

This is not a missing feature that will arrive in v3. It is architectural. The pretraining task does not include Poisson offsets. The model has no representation of "this column adjusts the scale of the dependent variable rather than predicting it." You cannot pass `log(Exposure)` as an offset because the API has no such slot. If you pass it as a feature, the model is free to weight it however its learned prior suggests — which is not the same as treating it as a fixed term with coefficient 1.

The only workaround is to change the target.

---

## The log-rate workaround and its four problems

The standard patch — used in our `insurance-thin-data` library for thin segments, disclosed prominently there — is to transform the target:

```python
y_transformed = np.log(ClaimNb / Exposure + eps)
```

where `eps` is a small constant (typically 1e-6) to handle zero claims. The model fits on log-rate. Predictions are exponentiated back to rate, then multiplied by exposure to get expected counts.

This is not the correct formulation. Here are the four ways it goes wrong.

**1. Distribution mismatch.** The log-transformed claim rate is not Gaussian. For a book where 94% of policies have zero claims — the freMTPL2 claim incidence is around 6% — the distribution of `log(ClaimNb / Exposure + eps)` is bimodal: a large spike at `log(eps)` (the zero-claim mass), and a small continuous component for positive claim rates. TabPFN's pretraining prior, being Gaussian-equivalent, is poorly matched to this structure. The model must fit a highly non-Gaussian target using a Gaussian prior.

**2. Zero-claim dominance.** Over 94% of freMTPL2 policies have `ClaimNb = 0`. Every one of them maps to `log(0 / Exposure + eps) ≈ log(eps) ≈ −13.8` (with `eps = 1e-6`). The model receives a target of −13.8 for 94% of its training examples regardless of their features. It must somehow learn that this value is not informative about feature-to-target relationships, and that the 6% of policies with positive claim rates carry essentially all the information. That is a hard learning problem that the pretraining prior has not prepared it for.

**3. Mid-term cancellations are indistinguishable from full-term zero-claim policies.** A policy with `Exposure = 0.1` and `ClaimNb = 0` maps to `log(0 / 0.1 + eps) = log(eps)`. A policy with `Exposure = 0.9` and `ClaimNb = 0` maps to the same value. In the correctly specified Poisson model, these two policies have different expected contributions to the likelihood — the short-exposure policy carries much less information about risk. In the log-rate workaround, they are identical targets. The model loses the exposure-varying information structure entirely for zero-claim policies.

**4. The sum property breaks.** In a correctly specified Poisson model with a log-exposure offset, predicted expected counts sum to observed claim counts in expectation across any sub-group. This actuarial property — that A/E ratios are 100% in aggregate when the model is correctly specified on training data — is what makes the model usable in practice. With the log-rate workaround, `exp(predicted_log_rate) × Exposure` does not reproduce this property. The aggregate A/E will not be 100%, and the distribution of errors across the predicted-rate deciles will not be symmetric.

---

## Benchmark results

The Poisson deviance and Gini results from the benchmark (evaluated on the 135,598-policy test set) tell the story clearly.

On the full training data comparison (GLM and CatBoost only — TabPFN excluded by dataset size):

- **GLM (542K train):** Poisson deviance 0.312, Gini 0.288
- **CatBoost (542K train):** Poisson deviance 0.291, Gini 0.311

CatBoost wins on both metrics with the full data, as expected — it is a nonlinear model with a correct Poisson formulation.

On the 10K subsample comparison (all three models, TabPFN with the log-rate workaround):

- **GLM (10K train):** Poisson deviance 0.334, Gini 0.261
- **CatBoost (10K train):** Poisson deviance 0.318, Gini 0.274
- **TabPFN (10K train, log-rate workaround):** Poisson deviance 0.398, Gini 0.219

TabPFN's Poisson deviance is 25% worse than the GLM on the same 10K sample. The Gini — which measures ranking quality rather than absolute calibration — drops from 0.261 to 0.219, a 16% relative degradation. The A/E by predicted decile shows systematic over-prediction in the top decile and under-prediction across the middle of the book.

This is not a tuning problem. The log-rate workaround is doing what it does, and what it does is not sufficient for this task.

---

## Where TabPFN still has value

We want to be precise here, because the [thin-segment case](/2026/03/13/insurance-tabpfn/) is real and our `insurance-thin-data` library is built on it.

**Thin segments.** Fewer than 2,000 policies, GLM parameter instability, no natural credibility parent. This is the regime TabPFN was designed for, and the log-rate workaround is documented, disclosed, and acceptable as an approximation when the alternative is a flat or unstable GLM estimate.

**Classification tasks.** Fraud probability, cancellation prediction, non-taken-up rates. No exposure offset needed. TabPFN's Gaussian prior is well-matched to a classification task with binary or multi-class outputs. The 10K row limit is less constraining here because class-balanced sampling is natural.

**Severity on thin classes.** If you are modelling large-loss severity on a thin book of commercial fleet claims, there is no exposure offset in the severity model. The 10K limit is less binding because large-loss experience is intrinsically sparse. This is a plausible use case, with caveats on the Gamma family being absent.

**What it is not** is a viable approach for the main Poisson frequency model on a personal lines book. The structural limitation is not temporary. When Tab-TRM (arXiv:2601.07675) becomes pip-installable, that changes — it is, to our knowledge, the only tabular foundation model with explicit Poisson support and an exposure offset mechanism. As of March 2026, it is not on PyPI.

---

## Honest verdict

CatBoost with `loss_function='Poisson'` and a proper baseline offset is the right model for insurance frequency on a large book. It has been the right model for several years. It handles exposure correctly, scales to millions of rows, and produces SHAP-based relativities that you can feed into a rating engine. The GLM remains the right model where interpretability is paramount and the feature structure is genuinely additive in log-space.

TabPFN is an impressive piece of work. The Nature 637 paper's benchmark across 300+ datasets is serious science, and the in-context learning approach is a genuine contribution to the thin-data problem. But "impressive" and "correct formulation for insurance frequency" are different predicates. On freMTPL2, TabPFN's deviance is 25% worse than a GLM on the same 10K training set. That is not a close result.

The industry tendency to reach for a new tool and then work around its structural limitations is one we want to resist. The exposure offset is not a detail. It is the mathematical statement that time-at-risk must be accounted for. Any model that cannot represent it natively requires a workaround, and the workaround costs you on every metric that matters.

---

**[Benchmark notebook on GitHub](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/tabpfn_benchmark.py)** — freMTPL2 benchmark, three models, full metrics.

---

**Related reading:**
- [Foundation Models for Thin Segments: TabPFN and TabICLv2 in Insurance Pricing](/2026/03/13/insurance-tabpfn/) — the affirmative case: where TabPFN genuinely outperforms a GLM
- [Tab-TRM: arXiv:2601.07675](https://arxiv.org/abs/2601.07675) — the only tabular foundation model with Poisson support and exposure offset handling; not yet pip-installable
