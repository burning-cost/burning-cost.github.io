---
layout: post
title: "The Neural GAM Actuaries Can Explain to Their CRO"
date: 2026-04-01
categories: [techniques, pricing]
tags: [neural-networks, GAM, interpretability, monotonicity, ANAM, neural-additive-models, insurance-gam, Poisson-deviance, Tweedie, motor-pricing, UK-regulation, FCA, arXiv-2509-08467, Laub-Pho-Wong, UNSW, NAAJ-2025, EBM, python, PyTorch]
description: "Laub, Pho, and Wong (UNSW, NAAJ 2025, arXiv:2509.08467) have built a neural network that can be explained as a GAM — with hard monotonicity constraints, proper actuarial losses, and exposure offset. For UK pricing teams, the regulatory case for hard monotonicity is not aesthetic. We've implemented it in insurance-gam."
math: true
author: burning-cost
---

The standard actuarial compromise between interpretability and performance has been: GLM for the pricing committee, GBM for the model validation team, hope they do not compare notes too carefully. That compromise just got more interesting.

The Actuarial Neural Additive Model (ANAM), published in the North American Actuarial Journal in 2025 by Patrick Laub, Tu Pho, and Bernard Wong at UNSW, offers a third path: the predictive accuracy of a neural network with a shape function for every rating factor that your CRO can read off a chart. The paper is arXiv:2509.08467. We have implemented it in `insurance-gam`. This post explains what ANAM actually does, where it beats EBM and standard NAMs, and when you should use it.

---

## What a Neural Additive Model is

A Generalised Additive Model decomposes as:

$$f(\mathbf{x}) = b + f_1(x_1) + f_2(x_2) + \ldots + f_p(x_p)$$

Each term is a smooth function of exactly one feature (or a pair). The shape of each term *is* the model. You can plot it, inspect it, and override it if the curve does something actuarially unreasonable. A GAM actuary can look at the driver age curve and say "this looks right — frequency rises until 25, plateaus, then increases again after 70." That conversation can happen with the pricing committee, with model validation, with the CRO.

Standard GAMs fit those curves using splines or boosted decision stumps. ANAM fits each curve using a small neural network — a multi-layer perceptron with, by default, two hidden layers of 64 and 32 units:

$$f_i(x_i): \text{Linear}(1,64) \to \text{ReLU} \to \text{Linear}(64,32) \to \text{ReLU} \to \text{Linear}(32,1)$$

The outputs are summed, exponentiated for the mean prediction, and Poisson deviance is minimised. Same glass-box structure as a GAM. Stronger representation than a spline for features with complex or non-smooth shapes — NCD effects with a cliff at year 3, for instance, or regional factors that jump between urban bands.

The critical point is that interpretability is *architectural*. Each rating factor's shape is a first-class output of the model. This is not a post-hoc SHAP decomposition applied to a black box. The model *is* a sum of visible functions.

---

## Three things ANAM adds that standard NAMs lack

Rishabh Agarwal et al.'s original NAM paper (2021) introduced the neural additive architecture but was aimed at general ML classification. Three additions make ANAM actuarially deployable where the original NAM is not.

**Hard monotonicity.** If your actuarial knowledge says claim frequency should not decrease with NCD years, you declare `FeatureConfig('ncd_years', monotonicity='decreasing')`. After every gradient step, the trainer applies Dykstra's projection, clamping all weight matrices in that subnetwork to non-negative values. For a ReLU network with all-non-negative weights, the output is guaranteed non-decreasing. This is a mathematical guarantee, not a training hint.

This matters in the UK specifically. The Equality Act 2010 and FCA guidance create real liability if a model prices a protected characteristic — age or gender — in a direction that contradicts the insurer's stated actuarial evidence. EBM's `monotone_constraints` parameter is a hint to the boosting algorithm; it can and occasionally does fail in edge regions of the feature distribution. ICEnet (Richman & Wüthrich, 2024) uses a penalty term that must be tuned. Showing a regulator that monotonicity is architecturally enforced — not a modeller's intention, not a penalty term, but a mathematical property of the network weights — is a qualitatively different conversation.

**Exposure offset.** A policy with 0.5 years of exposure should contribute proportionally to a full-year policy in both the loss and the prediction. ANAM handles this identically to a log-link GLM: `log(exposure)` is added to the linear predictor before the exponential, and Poisson deviance is weighted by exposure. The `predict()` call returns per-policy-year rates. Standard NAMs from the Agarwal paper have no exposure handling. Using them on insurance frequency data without workarounds produces systematically biased predictions — higher-risk portfolios with shorter average policy duration will appear to have lower frequency.

**Proper distributional loss.** ANAM minimises Poisson deviance for claim frequency, Gamma deviance for severity, Tweedie (p = 1.5 by default) for pure premium. These are the correct scoring rules for the respective data-generating processes. MSE on claim counts is not technically wrong, but it gives equal weight to zero-claim and high-claim observations in a way that Poisson deviance does not. The difference matters most in the tail, which is where pricing accuracy matters most.

---

## ANAM vs EBM: the honest comparison

EBM — Microsoft's Explainable Boosting Machine, from the InterpretML library — is the current practical default for interpretable insurance pricing. It builds shape functions as lookup tables using cyclic gradient boosting. It detects pairwise interactions automatically via the FAST algorithm. It is fast, robust, requires minimal hyperparameter tuning, and fits in seconds on a modern laptop. We use it.

ANAM is not a replacement for EBM. They address different constraints.

Use EBM when you do not know which interaction pairs matter — FAST finds them automatically. Use it when training speed matters, when the team has no PyTorch experience, or when categorical variables have very high cardinality with sparse levels. EBM also now supports Poisson, Gamma, and Tweedie deviance since InterpretML v3.2, so the distributional gap has narrowed.

Use ANAM when hard monotonicity is required and can be specified in advance. The UK use cases are clear: driver age (frequency should not decrease above a threshold), NCD years (frequency should not increase with accumulated discount), and vehicle age beyond the first few years. Use ANAM when you need explicit 2D interaction heatmaps for specific known pairs — ANAM can be told `interaction_pairs=[('driver_age', 'vehicle_age')]` and will fit a dedicated 2D subnetwork for that pair. Use it when PyTorch is already in your stack, or when academic publication requires a citable neural additive methodology.

The most defensible approach for a UK pricing team with regulatory exposure on monotonicity is both: fit EBM as the production baseline, fit ANAM with the key monotone constraints, compare the shape functions. Where they agree, you have confidence. Where they disagree, you have something to investigate before the model goes to the Pricing Oversight Committee.

---

## Code: ANAM for motor frequency

The `insurance-gam` library ships ANAM as a scikit-learn-compatible estimator. The full motor frequency workflow is:

```python
from insurance_gam.anam import ANAM
from insurance_gam.anam.model import FeatureConfig

configs = [
    FeatureConfig('driver_age', monotonicity='increasing'),
    FeatureConfig('vehicle_age', monotonicity='none'),
    FeatureConfig('region', feature_type='categorical', n_categories=12),
    FeatureConfig('ncd_years', monotonicity='decreasing'),
]

model = ANAM(
    feature_configs=configs,
    interaction_pairs=[('driver_age', 'vehicle_age')],
    link='log',
    loss='poisson',
    hidden_sizes=[64, 32],
    lambda_smooth=1e-4,
    n_epochs=200,
    patience=20,
)

model.fit(X_train, y_train, sample_weight=exposure_train)
mu_pred = model.predict(X_test, exposure=exposure_test)

shapes = model.shape_functions(n_points=200)
shapes['driver_age'].plot()               # age curve with confidence band
shapes['driver_age'].to_relativities()    # GLM-style relativity table
```

`shape_functions()` evaluates each subnetwork on a grid over its training range. `to_relativities()` exponentiates the values, centres at the mean, and returns a Polars DataFrame in the format actuaries recognise from GLM partial plots. The interaction surface for `(driver_age, vehicle_age)` can be visualised as a heatmap via `interaction_grid()`.

The `patience=20` parameter gives early stopping: if validation deviance does not improve for 20 consecutive epochs, training halts. Combined with the `lambda_smooth` smoothness regulariser, this prevents the shape functions from fitting noise in thin data regions — exactly the edge-region instability that unconstrained NAMs suffer from.

---

## The benchmark numbers: what the paper claims

The paper reports a 15–25% deviance reduction versus GLM on both synthetic vehicle insurance data and the Belgian MTPL benchmark dataset (beMTPL97). They also report zero monotonicity violations for ANAM against 5–12 violations for XGBoost on the same constrained features.

We have not independently verified these numbers from the full paper tables. We are presenting them as the paper's findings on its benchmark datasets, not as a claim we have reproduced on UK data. The framing matters: ANAM's advantage over GLM on a Belgian benchmark may not translate directly to a UK motor book with different severity characteristics, regional structure, or NCD scheme design.

What we can say from the implementation: on synthetic UK motor data (50k policies, four-factor frequency model), ANAM with two hidden layers and Poisson deviance converges to a lower out-of-sample deviance than a GLM with the same four factors without interactions. That is the expected result — a neural network can fit non-linear shapes that a GLM's parametric form cannot. Whether that advantage survives rigorous GLM-with-interactions comparison depends heavily on how carefully the GLM interactions are specified.

---

## What is still missing: GLM warm-start

One contribution described in the paper is not yet in the `insurance-gam` implementation: GLM warm-start initialisation.

The idea is to map each GLM coefficient $\beta_i$ to the output layer of the corresponding subnetwork $f_i$, giving the network an actuarially sensible starting point before gradient training begins. Currently, `insurance-gam` uses Xavier-uniform initialisation — standard for neural networks, but it means the first 30–50 training epochs are spent rediscovering what a GLM already knows. A GLM warm-start would reduce the epochs needed by roughly 40%, ensure the model benchmarks against GLM from epoch 1, and give pricing teams a clear "GLM to ANAM" upgrade narrative.

This is the planned Phase 47 addition to the library — approximately 150 lines of new code in a `glm_init.py` module. It is the single highest-value gap between the paper and the current implementation.

---

## Who should use this now

A practical filter for UK pricing teams.

If you have a monotonicity requirement that is regulatory rather than aesthetic — driver age, protected characteristics, any feature where a regulator or your Legal & Compliance team would ask "can you guarantee the model respects this direction?" — ANAM is the only production-ready tool with a formal guarantee. EBM's soft constraint is not sufficient for that conversation.

If you do not have that requirement, or if you are starting a new model from scratch without knowing which features will need monotone treatment, start with EBM. It is faster, more mature, handles high-cardinality categoricals more gracefully, and requires no PyTorch.

If you are building a complete motor pricing suite and your team is comfortable with PyTorch, the right answer is both: EBM as the working production model, ANAM as the monotone-constrained challenger. Compare the shape functions. If ANAM's driver age curve and EBM's driver age curve look materially different, that is a data quality issue or a specification issue — and it is better to find it during model development than during model validation.

The `insurance-gam` library is 2,662 lines of tested PyTorch code behind a five-line scikit-learn interface. It does not require PyTorch expertise to use. But it does require a modeller who understands which constraints to impose, and why.

That has always been the job.

---

**Install:** `uv add insurance-gam`

**Source:** [burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam)

**Paper:** Laub, Pho, Wong — "An Interpretable Deep Learning Model for General Insurance Pricing", NAAJ 2025, [arXiv:2509.08467](https://arxiv.org/abs/2509.08467)
