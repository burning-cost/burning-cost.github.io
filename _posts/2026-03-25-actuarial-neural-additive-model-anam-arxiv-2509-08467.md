---
layout: post
title: "Actuarial Neural Additive Model: What the Paper Actually Does (arXiv:2509.08467)"
date: 2026-03-25
categories: [pricing, machine-learning, research]
tags: [ANAM, NAM, GAM, interpretability, monotonicity, smoothness, Dykstra, Tweedie, Poisson, GLM, XGBoost, CANN, LocalGLMnet, Laub, french-mtpl, beMTPL97, insurance-gam, python]
description: "Laub, Pho and Wong's ANAM paper enforces smoothness and monotonicity architecturally, not as penalties. Here is what the mechanism actually is, why it matters more than the benchmark numbers, and what our insurance-gam implementation adds."
---

In September 2025, Patrick Laub, Tu Pho and Bernard Wong (UNSW) published arXiv:2509.08467: 'An Interpretable Deep Learning Model for General Insurance Pricing.' The paper introduces the Actuarial Neural Additive Model. We have already written about the [implementation in insurance-gam](/2026/03/13/your-interpretable-model-isnt-interpretable-enough/) and the [choice between EBM, ANAM and PIN](/2026/03/14/insurance-gam-interpretable-nonlinearity/). This post is different: it is about the paper itself — the methodology choices, the formal apparatus, and what those choices mean for pricing teams in practice.

The benchmark numbers get all the attention. The methodology is more interesting.

---

## The problem the paper is actually solving

The interpretability literature for insurance pricing has two branches that rarely intersect. The first is post-hoc: SHAP, integrated gradients, LIME — methods that explain a black-box model's predictions after the fact. The second is intrinsic: GLMs, GAMs, EBMs — models whose structure makes the contribution of each feature to each prediction mathematically exact.

The gap in the intrinsic branch has been the absence of a model that satisfies all of: (1) non-parametric shape functions without manual bucketing decisions, (2) formally guaranteed smoothness, (3) provably monotone constraints, (4) actuarially appropriate distributional losses, and (5) multi-task structure for joint frequency-severity modelling. You can get some of these from an EBM (InterpretML), some from a GAM fitted in Emblem, none of them all together.

ANAM is the paper's answer to that gap.

---

## The architecture: one subnetwork per covariate

The prediction structure is additive in log space — the same as a GLM or any GAM:

```
log(E[Y | x]) = offset + f_1(x_1) + f_2(x_2) + ... + f_p(x_p)
```

What is different from a GLM is that each `f_j` is a small neural network (the paper calls these FeatureNetworks), not a linear function of manually engineered terms. Each `f_j` receives only its own covariate `x_j` and nothing else. This is what makes the model intrinsically interpretable: `f_j(x_j)` is not an approximation of the feature's contribution — it *is* the feature's contribution, in the same log-space unit as a GLM parameter.

Pairwise interaction terms extend the structure to:

```
log(E[Y | x]) = offset + Σ_j f_j(x_j) + Σ_{j<k} f_{jk}(x_j, x_k)
```

where each `f_{jk}` is a dedicated two-input subnetwork for that feature pair. The paper treats interaction terms as optional: you start with main effects only and add pairs where the data support them.

This is structurally identical to a GA2M (the architecture behind EBMs), except the subnetworks are neural rather than boosted trees. The representation power of each `f_j` is therefore continuous and differentiable everywhere, rather than the step-function lookup tables you get from an EBM.

---

## How smoothness is enforced — and why it matters

The paper enforces smoothness via a Whittaker-Henderson penalty on the output of each FeatureNetwork. The penalty has the form:

```
λ * Σ_t (f_j(x_t) - 2f_j(x_{t-1}) + f_j(x_{t-2}))²
```

evaluated on a grid of feature values. This penalises second differences — the discrete analogue of the second derivative — which is the standard smoothing spline regularisation, with λ controlling the strength.

Why does this matter, given that neural networks are already smooth? Two reasons.

First, neural networks trained without smoothness constraints can produce shape functions that oscillate sharply in low-exposure regions of the feature space. The training data do not enforce smoothness at those points — there are simply not enough policies with, say, drivers aged 23 with 7 years NCD and a specific combination of vehicle group and area. Without explicit smoothing, the shape function can behave arbitrarily in those regions, producing pricing anomalies that only surface when a risk in that zone is actually quoted.

Second, smooth shape functions are directly analogous to the fitted curves that actuaries produce from spline GAMs or from manual smoothing of factor plots. A shape function with unexplained oscillations fails the same visual inspection that would reject a jagged GLM factor plot. Formally penalising the smoothness makes the shape function a tractable regulatory artefact: it behaves like an actuarial curve, not like a neural network output.

The Whittaker (1922) and Henderson (1924) citations in the paper are not ornamental. The penalty is directly in that tradition — the same mathematical structure actuaries have used to smooth mortality and morbidity curves for a century, applied to neural network outputs.

---

## How monotonicity is enforced — the Dykstra mechanism

This is the technically most interesting part of the paper, and the part that pricing teams should understand in detail.

Most ML frameworks handle monotonicity via a soft penalty: add a term to the loss function that is large when the shape function violates the monotone constraint. The problem with this is that a soft penalty is always a compromise between fitting the data and satisfying the constraint. If the data locally contradict the constraint — a small sample of young drivers with fewer claims than the adjacent age band, perhaps because of data artefacts — the penalty term and the likelihood compete. The constraint will be approximately honoured but not exactly.

Laub, Pho and Wong use Dykstra's iterative projection algorithm instead. The mechanism is architectural:

For a FeatureNetwork that is constrained to be monotone increasing, the key observation is that a ReLU network with all non-negative linear layer weights produces a monotone increasing function. This is provable: each ReLU activation is monotone increasing, each linear combination of non-negative weights of monotone increasing functions is monotone increasing, and composition of monotone increasing functions is monotone increasing. The constraint on the model is therefore reducible to a constraint on the weight matrices: all weights must be non-negative.

Dykstra's algorithm enforces this at every gradient step. After each weight update, any weight that has gone negative is projected back to zero. The projection is the minimum-norm correction — the closest non-negative point to the unconstrained gradient step. This is Dykstra's method: iterated projections onto constraint sets, each projection being the closest-point-in-the-set to the previous iterate.

The result is not "mostly monotone" or "monotone except in low-exposure regions" — it is monotone at every point in the input space, guaranteed by construction. Not a soft preference. A hard constraint satisfied at every gradient step throughout training.

This is why we argue in the [comparison guide](/2026/03/14/insurance-gam-interpretable-nonlinearity/) that ANAM is the correct choice when monotonicity is a hard tariff requirement. The EBM approach — isotonic regression applied post-fit to the shape function — makes the same guarantee only for the grid of training points. Between grid points and in unobserved regions of the feature space, the edited EBM shape function may not be monotone. With ANAM, the weight-level constraint propagates to the full continuous function.

---

## Where ANAM sits relative to CANN and LocalGLMnet

This distinction matters for a pricing team choosing an architecture, and the paper is clear about it.

**CANN** (Combined Actuarial Neural Network, Schelldorfer and Wüthrich, 2019) takes an existing GLM prediction and passes it — along with raw features — through a neural network that learns a correction to the GLM output. The resulting model is not additive in the NAM sense: the correction network sees all features simultaneously and can capture arbitrary interactions. CANN is more accurate than GLM in proportion to how much signal the correction network finds, but the resulting model is not decomposable into per-feature shape functions. CANN is a black-box correction on top of an interpretable model; ANAM is an interpretable model throughout.

**LocalGLMnet** (Richman and Wüthrich, 2022) is structurally closer to a GLM: it uses a neural network to produce feature-dependent weights on each input, generalising the GLM's fixed linear coefficients. The interaction between LocalGLMnet's network weights and the input features is not additively separable in the way ANAM's per-feature subnetworks are. The contribution of feature `x_j` to the prediction depends implicitly on all other features through the shared weight-generating network. Shape functions are not directly available: you would need to marginalise over other features, which is an approximation.

ANAM's additive structure is therefore a formal, architectural guarantee of decomposability that neither CANN nor LocalGLMnet provides.

The benchmark on beMTPL97 (Belgian MTPL frequency, the paper's main evaluation dataset) shows ANAM competitive with GLM and XGBoost on held-out Poisson deviance, while outperforming GLM in most configurations and matching or beating XGBoost on the datasets where the interpretability constraints are tightest. The paper's specific claim is not that ANAM dominates XGBoost — a black-box model with no constraints will generally be at least as accurate as any constrained model — but that ANAM's accuracy relative to GLM is large enough to justify adopting it where the interpretability requirement makes a GBM non-deployable.

We would phrase this more directly: if your model can be a GBM, use a GBM. ANAM is for the cases where it cannot.

---

## Multi-task learning and distributional flexibility

The paper includes two extensions that are directly relevant for actuarial pricing.

**Multi-task learning.** ANAM can be trained to simultaneously predict claim frequency and claim severity, sharing the FeatureNetwork parameters across both tasks. The shared representation is the key: features that are informative for both frequency and severity (driver age, vehicle characteristics) will produce better-fitting subnetworks than if frequency and severity are fitted separately on potentially different training sets with different effective exposures. This is the correct motivation for multi-task in insurance: the data for severity estimation is a subset of the data for frequency estimation (only claims contribute to severity), so a joint model extracts signal from the full policy set for both components.

**Flexible distributional regression.** The paper extends ANAM to produce parameters of the full predictive distribution, not just the mean. This is the GAMLSS-style extension: the model simultaneously predicts the mean and the dispersion parameter, with both quantities modelled as additive functions of the features. For UK pricing teams, this is directly relevant to the [per-risk volatility scoring](/2026/03/04/per-risk-volatility-scoring-with-distributional-gbms/) problem — the ability to price the variance of the claim distribution, not just its mean.

---

## What insurance-gam adds

The paper provides the theoretical framework and the beMTPL97 benchmarks. Our [`insurance-gam`](https://github.com/burning-cost/insurance-gam) implementation (available at `uv add insurance-gam`) adds:

- **Sklearn-compatible API.** `ANAM.fit(X, y, sample_weight=exposure)`, `ANAM.predict(X, exposure=exposure)`. Works in the same pipeline as `PoissonRegressor`.
- **Polars output for shape functions.** `model.shape_functions()` returns a dict of Polars DataFrames, one per feature. Each DataFrame has columns `x`, `f_x`, `f_x_lower`, `f_x_upper` — directly exportable to JSON for regulatory documentation.
- **GLM comparison overlay.** `GLMComparison(model, glm_params).plot_overlay("driver_age")` overlays the ANAM continuous shape function against your existing GLM relativity table. Divergences are the model's discoveries.
- **Poisson, Gamma, Tweedie losses.** The paper develops the theoretical framework; the library implements all three with the standard actuarial exposure convention (`log(exposure)` as fixed offset).

```bash
uv add insurance-gam
```

Source: [github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam)

---

## The honest limitations

Three things the paper does not resolve and the library inherits.

**Accuracy vs black-box GBM.** On the beMTPL97 benchmark, ANAM outperforms GLM and is competitive with XGBoost. On some configurations, XGBoost wins. This is the expected result: removing the additive constraint lets a black-box model fit residual structure that an additive model cannot. If the additive constraint is not required, a GBM is still the accurate choice. ANAM is not competing with GBMs for raw accuracy; it is competing for the cases where a GBM is not deployable.

**Training speed.** PyTorch SGD with per-batch Dykstra projection is substantially slower than interpretML's C++ boosting backend for EBMs. On a 200k-policy UK motor book, expect 5–20 minutes for ANAM versus 30–90 seconds for EBM. That is acceptable for a quarterly rebuild; it becomes a pipeline engineering problem if the model needs to be refitted more frequently.

**The additive constraint is real.** High-cardinality interactions that matter for accuracy — telematics trip microstructure, connected home sensor data — are not well-represented by a set of pairwise subnetworks. The model will find the main effects and the explicitly specified pairs; everything else is residual. On standard UK personal lines features (driver age, NCD, vehicle group, area, annual miles), this is not a problem. On novel, high-dimensional feature sets, it is.

---

## What to read next

The paper is at [arxiv.org/abs/2509.08467](https://arxiv.org/abs/2509.08467). Appendix C contains the Dykstra projection derivation. Section 4.3 covers the multi-task learning setup. The beMTPL97 results are in Section 5.

For the implementation:

- [EBM, ANAM, or PIN: Choosing an Interpretable Architecture](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — when to use which
- [Does insurance-gam Actually Work for Insurance Pricing?](/2026/03/24/does-insurance-gam-actually-work-pricing/) — benchmark results on synthetic UK motor DGP
- [Pairwise Interaction Networks](/2026/03/13/insurance-pin/) — the alternative if you need exact interaction surfaces
- [Tab-TRM: The January 2026 Insurance Neural Architecture](/2026/03/25/tab-trm-insurance-neural-architecture/) — the competing architecture from Padayachy, Richman and Wüthrich, for context
