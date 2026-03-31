---
layout: post
title: "Actuarial Neural Additive Models: Interpretable Deep Learning for Insurance Pricing"
date: 2026-03-31
categories: [interpretability]
tags: [neural-additive-models, anam, interpretability, deep-learning, gam, ebm, glm, insurance-pricing, poisson, monotonicity, pytorch, insurance-gam]
description: "Laub, Pho and Wong's ANAM gives each rating factor its own neural network sub-model, sums them like a GAM, and adds three things a generic NAM cannot do: hard monotonicity, exposure offsets, and Poisson deviance. Here is what it is, when to use it, and why it does not replace EBM."
---

A recurring frustration in insurance pricing is that the models best suited to handling messy, asymmetric, high-cardinality tabular data — gradient-boosted trees, XGBoost — produce outputs that are nearly impossible to audit in a UK regulatory context. The shape function for driver age is buried inside 500 trees. You cannot hand it to a pricing committee, let alone an actuary preparing a TCF justification.

The standard response has been the Explainable Boosting Machine (EBM), a cyclic-boosted GAM from Microsoft's InterpretML that produces explicit per-feature curves while roughly matching XGBoost on benchmark datasets. EBM works well. We use it. But it has one weakness that matters in pricing: monotonicity is a soft constraint, not a mathematical guarantee.

Laub, Pho, and Wong's paper "An Interpretable Deep Learning Model for General Insurance Pricing" (arXiv:2509.08467, NAAJ 2025) proposes a different approach: replace the boosted stumps in a GAM with small neural networks, one per rating factor, and add the three things a generic neural additive model lacks for insurance work — hard monotonicity, proper exposure offset handling, and Poisson/Gamma/Tweedie loss functions. They call the result the Actuarial Neural Additive Model (ANAM).

---

## What ANAM is

The prediction formula:

```
eta = bias + f_1(x_1) + f_2(x_2) + ... + f_k(x_k) + g_12(x_1, x_2) + ...
mu  = exp(eta + log(exposure))
```

Each `f_i` is a small multi-layer perceptron trained on feature `x_i` alone. The MLPs are summed — exactly the GAM structure, but with neural network shape functions instead of splines. Pairwise interaction terms `g_ij` are optional two-input MLPs.

The architecture in text form:

```
Feature: driver_age  ──► [Linear 1→64, ReLU] ──► [Linear 64→32, ReLU] ──► [Linear 32→1] ──► f_1(age)
Feature: vehicle_age ──► [Linear 1→64, ReLU] ──► [Linear 64→32, ReLU] ──► [Linear 32→1] ──► f_2(veh_age)
Feature: region      ──► [Embedding n→4]     ──► [Linear 4→32,  ReLU] ──► [Linear 32→1] ──► f_3(region)
Feature: ncd_years   ──► [Linear 1→64, ReLU] ──► [Linear 64→32, ReLU] ──► [Linear 32→1] ──► f_4(ncd)
                                                                                               ↓
                                                                                      Σ + bias + log(exposure)
                                                                                               ↓
                                                                                          exp(eta) = mu
```

Each subnetwork is independent. Feature 1's parameters are never updated by gradients flowing through feature 2. The additive structure is baked into the architecture, not imposed by regularisation. This is what makes the model interpretable in the same sense as a GAM: you can read off the shape function for driver age and show it to anyone.

---

## The three actuarial additions

### 1. Hard monotonicity

The key novelty over a generic NAM. For a ReLU network with weight matrices `W_1, W_2, W_3`, if every weight in every matrix is non-negative, the composition `W_3 * ReLU(W_2 * ReLU(W_1 * x))` is non-decreasing in `x`. ReLU is non-decreasing, and multiplying by non-negative matrices preserves that property — so the whole composition is non-decreasing. No further argument needed.

ANAM enforces this via a Dykstra projection step after every gradient update — clamping all weights in the subnetwork to `>= 0` (or `<= 0` for decreasing constraints). This is a hard constraint. The shape function for `ncd_years` set to `monotonicity="decreasing"` cannot produce an uptick at 7 years, no matter what the training data suggests. No tuning required, no penalty weight to calibrate.

This matters for UK motor pricing specifically. The Age Discrimination provisions of the Equality Act 2010 require that any age-based rating differential can be justified. An ANAM age curve with a hard monotonicity constraint (younger drivers pay at least as much as middle-aged drivers) is defensible in a way that an EBM curve with a soft `monotone_constraints` hint is not — the EBM may produce a small non-monotone bump and your documentation has to explain why you left it in.

### 2. Exposure offset matching GLM convention

A standard NAM applied to insurance frequency data will be wrong for any policy with non-annual exposure. ANAM handles this identically to a log-link GLM:

```
mu = exp(eta + log(exposure))
```

The log-exposure appears as an additive offset to the linear predictor. The Poisson deviance loss is also exposure-weighted. The target variable is claim counts, not claim rates. This is the convention every UK pricing actuary uses in a GLM and ANAM matches it exactly. You can migrate a portfolio off a GLM onto ANAM without changing your data pipeline.

### 3. Native Poisson, Gamma, and Tweedie losses

The training objective is proper actuarial deviance — Poisson for frequency, Gamma for severity, Tweedie for pure premium. These are the correct scoring rules for insurance data, not MSE or binary cross-entropy. They handle the zero-inflation in frequency data correctly (zero claims are the right prediction for most policies), and they penalise calibration errors in a way that is meaningful for pricing.

---

## Comparison with alternatives

| | ANAM | EBM (InterpretML) | LocalGLMnet | Standard GAM |
|---|---|---|---|---|
| Architecture | Per-feature MLP sum | Cyclic boosted stumps | Attention-weighted GLM | P-splines or thin-plate |
| Interpretability | Shape functions + interaction heatmaps | Shape functions + interaction heatmaps | Per-observation coefficient distributions | Shape functions |
| Hard monotonicity | Yes (weight clamping) | No (soft hint only) | No | No |
| Exposure offset | Yes (native) | Via init_score | Depends on implementation | Via offset term |
| Poisson/Gamma loss | Yes | Partial | Yes | Yes |
| Interaction detection | Manual pairs required | Automatic (FAST algorithm) | Implicit (full covariate vector) | Manual |
| GPU support | Yes (PyTorch) | No | Yes | No |
| Training speed | Slow (gradient descent) | Fast (boosting) | Moderate | Fast |
| Calibration out-of-box | Moderate | Good | Good | Good |
| UK track record | Limited | Growing | Limited | Established |

LocalGLMnet (Richman and Wüthrich, SAJ 2023) deserves a note here. It is often mentioned in the same breath as ANAM because both are neural networks with insurance-specific design. But LocalGLMnet does not produce a single shape function per feature. The effective coefficient for driver age varies observation by observation based on the full covariate vector. You cannot show a pricing committee "the age curve". ANAM preserves the strict per-feature shape function that GAMs provide. For regulatory sign-off in the UK market, that distinction matters.

---

## When to use ANAM, when to use EBM

We will be direct: EBM is the right default for most UK pricing teams, and ANAM does not change that.

EBM is faster to train, has automatic interaction detection, calibrates well without tuning, has no learning rate to choose, and has an established track record in actuarial literature (see Krupova, Rachdi and Guibert, arXiv:2503.21321, March 2025, for a recent French MTPL comparison). If you have never used a glass-box model before, start with EBM.

Use ANAM instead when:

**Hard monotonicity is non-negotiable.** If your pricing committee or legal team requires a mathematical guarantee that younger drivers do not receive a lower rate than middle-aged drivers — not a best-efforts constraint but an actual guarantee — ANAM provides it and EBM does not.

**You need a publishable 2D interaction surface.** The ANAM interaction network produces a `g_ij(x_i, x_j)` surface that you can visualise as a heatmap and show to a pricing committee. "Driver age 25 combined with vehicle age 3 years generates this interaction uplift." This is harder to extract cleanly from EBM.

**Your team is already in PyTorch.** If you are running a neural network stack (CANN models, severity distributions, neural embeddings), adding ANAM does not introduce a new framework. EBM sits outside that ecosystem.

**You are writing up methodology for a published paper.** ANAM has a citable NAAJ reference with a formal architecture description. That is useful if the work needs to stand up to peer review.

Do not switch from EBM to ANAM solely because "neural networks are more powerful." The paper claims ANAM outperforms EBM on their insurance benchmarks, but the specific numbers are not publicly extractable from the compressed PDF, and EBM with FAST interactions is approximately equivalent to XGBoost on most tabular benchmarks. The performance gap, if any, is smaller than the implementation complexity gap.

---

## The gap in the current implementation

Our [`insurance-gam`](https://github.com/burning-cost/insurance-gam) package includes a full `ActuarialNAM` implementation — per-feature subnetworks, Dykstra monotonicity projection, Poisson/Gamma/Tweedie losses, exposure offset, categorical embeddings, pairwise interactions, shape function extraction and relativity tables. The API is scikit-learn compatible:

```python
from insurance_gam.anam import ANAM
from insurance_gam.anam.model import FeatureConfig

configs = [
    FeatureConfig("driver_age", monotonicity="increasing"),
    FeatureConfig("vehicle_age", monotonicity="none"),
    FeatureConfig("region", feature_type="categorical", n_categories=12),
    FeatureConfig("ncd_years", monotonicity="decreasing"),
]

model = ANAM(
    feature_configs=configs,
    interaction_pairs=[("driver_age", "vehicle_age")],
    loss="poisson",
    n_epochs=200,
    patience=20,
)

model.fit(X_train, y_train, sample_weight=exposure_train)
shapes = model.shape_functions(n_points=200)
shapes["driver_age"].to_relativities()  # GLM-style relativity table
```

The one significant gap relative to the Laub, Pho, Wong paper is GLM warm-start initialisation. The paper describes starting each subnetwork's output layer from the corresponding GLM coefficient — so if the GLM says `beta_age = 0.034`, the ANAM subnetwork for age begins with that value rather than Xavier-uniform noise. This accelerates convergence considerably: the model starts from actuarial intuition rather than random initialisation, and avoids the dead-ReLU problem that can slow early training when many subnetworks are summed.

The current `insurance-gam` code uses Xavier-uniform initialisation instead. This works but means ANAM can take 50-100 epochs to arrive at what a GLM achieves in one step. Adding warm-start is approximately 120-150 lines of code — fit a `statsmodels` GLM, extract coefficients, map them to subnetwork output layer weights. It is the next implementation priority and would make ANAM a much more natural successor to a GLM-based pricing model.

The paper's reference implementation is R-only (keras/tensorflow). There is no Python package on PyPI. `insurance-gam` is the only Python implementation we are aware of.

---

## The honest verdict

ANAM is not a general-purpose upgrade to EBM. It is the right tool for a specific set of circumstances — hard monotonicity requirements, explicit interaction surfaces for regulatory documentation, PyTorch integration — and for those cases it is genuinely better than the alternatives.

The core technical contribution of Laub, Pho, and Wong is not the neural additive structure itself (that is Agarwal et al. 2021, arXiv:2004.13912). It is the combination of hard monotonicity via weight projection, actuarially correct loss functions, and GLM-compatible exposure handling in a single coherent package. Those three additions are what turn a research neural network into a tool you can actually deploy in a UK pricing environment.

The absence of GLM warm-start in the current implementation is the main reason we would not yet recommend ANAM as a default pricing model. Once that gap is closed, the convergence story changes: ANAM becomes a GLM that gradually learns to be more flexible, rather than a neural network that slowly rediscovers what the GLM already knew. That is a much more compelling proposition for a pricing actuary.

---

**Paper:** Laub, P.J., Pho, T., Wong, B. (2025). "An Interpretable Deep Learning Model for General Insurance Pricing." *North American Actuarial Journal*. [arXiv:2509.08467](https://arxiv.org/abs/2509.08467).

**Code:** [`insurance-gam`](https://github.com/burning-cost/insurance-gam) — Python implementation of ActuarialNAM with PyTorch, Poisson/Gamma/Tweedie losses, hard monotonicity, exposure offset, categorical embeddings, and shape function extraction.
