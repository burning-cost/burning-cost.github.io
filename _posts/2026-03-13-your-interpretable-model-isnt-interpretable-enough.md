---
layout: post
title: "Actuarial Neural Additive Models: Exact Interpretability with Tweedie Loss"
date: 2026-03-13
author: Burning Cost
categories: [pricing, machine-learning]
tags: [ANAM, NAM, interpretability, monotonicity, GLM, Poisson, Tweedie, FCA, Consumer-Duty, PRA, python, insurance-gam]
description: "Actuarial Neural Additive Models for UK pricing: exact interpretability, Tweedie loss, guaranteed monotonicity. insurance-gam - beyond SHAP and EBM limitations."
---

*This post is an ANAM deep-dive: the Actuarial Neural Additive Model architecture, why it gives mathematically guaranteed monotonicity where EBMs do not, and how to use the insurance-gam implementation with Tweedie/Gamma/Poisson losses. If you want the EBM workflow, read [EBMs for Insurance Pricing](/2026/03/09/explainable-boosting-machines-for-insurance-pricing/). If you want to choose between EBM, ANAM, and PIN, read the [comparison guide](/2026/03/14/insurance-gam-interpretable-nonlinearity/).*


Your GBM is not interpretable. You already know that. The SHAP values explain it approximately, after the fact, by attributing fractions of each prediction to each feature — a useful diagnostic, not a property of the model itself.

But here is the thing you may not have noticed: your "interpretable model" probably is not interpretable either.

If your interpretable model is an Explainable Boosting Machine, its shape functions are exact, genuinely intrinsic to the architecture, but they can be jagged in ways that produce artificial pricing discontinuities, it has no native Tweedie or Gamma loss for severity modelling, and its monotonicity constraints are post-hoc edits that break the model's internal consistency the moment you apply them. If your interpretable model is a GAM fitted inside a commercial pricing tool, the interpretability is real but the fitting is proprietary and the model building block is locked to the vendor's platform.

The Actuarial Neural Additive Model (ANAM), introduced by Laub, Pho and Wong (UNSW, arXiv:2509.08467, September 2025), resolves all three problems. We have built [`insurance-gam`](https://github.com/burning-cost/insurance-gam), the first pip-installable implementation, with sklearn-compatible API, Poisson/Tweedie/Gamma losses, smoothness regularisation, and monotonicity constraints that are mathematically guaranteed, not applied after the fact.

---

## Why "interpretable" needs to mean something specific

There are two ways a model can be interpretable. The first is intrinsic: the architecture is structured so that the contribution of each feature to each prediction is exactly decomposed by construction. The second is post-hoc: the model is a black box, and a separate explanation algorithm — SHAP, LIME, integrated gradients — approximates the feature contributions after the prediction is made.

Post-hoc interpretability is approximation. SHAP values for a CatBoost model satisfy the Shapley axioms and decompose the prediction exactly, but they do not tell you what the model would predict if a feature changed. The feature's SHAP value is an average marginal contribution over all feature orderings, which is not the same as a shape function. Two models with identical SHAP distributions can have radically different behaviour outside the training distribution.

Under PRA SS1/23 (effective May 2024), model validators are expected to understand the decision boundaries of their models and test behaviour at margins. "The SHAP values look reasonable" is harder to defend than "here is the shape function for driver age; it is monotone increasing and smooth, and here is the test coverage that proves it." This is not a hypothetical regulatory concern; it is the practical challenge of explaining ML model behaviour to someone trained to think in GLM relativity tables.

The ANAM architecture gives you intrinsic interpretability. Each feature gets its own subnetwork. The model's prediction is additive over those subnetworks:

```
log(E[Y]) = offset + f_1(x_1) + f_2(x_2) + ... + f_p(x_p)
```

The function `f_i(x_i)` is not an approximation or a SHAP summary. It is exactly the contribution of feature `x_i` to the log-linear predictor: a smooth curve you can plot, export as a Polars DataFrame, hand to a validator, and use directly in a rating engine.

---

## The monotonicity problem is not cosmetic

In any UK motor or home pricing model, you will have features where the direction of the relationship is known a priori. Claims frequency for drivers aged 17-25 does not decrease with age in that band. These are not empirical discoveries; they are prior knowledge that should be encoded in the model.

The problem with soft monotonicity penalties (including the approach used in most GBM implementations and some GAM packages) is that they are not guarantees. A soft penalty adds a term to the loss function that makes violations expensive. It does not make them impossible. If the data locally contradicts the constraint, the penalty term competes with the likelihood and the violation survives in proportion to how much likelihood is being sacrificed by honoring the constraint. You can end up with a model that is mostly monotone, with small violations hidden in low-exposure regions of the feature space.

This is a real problem for UK pricing, not a theoretical one. Small monotonicity violations in low-exposure regions are hard to catch in validation. They create pricing anomalies that only surface when a broker or an aggregator finds the rate inversion and exploits it.

`insurance-gam` uses Dykstra's projection algorithm to guarantee monotonicity exactly. The mechanism is architectural: for ReLU networks, clamping all linear layer weights to be non-negative (for increasing constraints) or non-positive (for decreasing constraints) is sufficient to guarantee monotone output at every point in the input space. This is applied every gradient step via `model.project_monotone_weights()`. The result is not "very few violations" but zero violations, provable by inspection of the shape function curve.

---

## The distributional loss matters

GLMs use Poisson deviance for claim frequency and Tweedie deviance for pure premium because those losses correspond to the actual data-generating process. Fitting a frequency model with MSE loss is wrong: it implicitly assumes Gaussian residuals, misweights the high-frequency cells that carry the most predictive information, and mishandles the zero-inflation that is structural in insurance data.

The EBM (InterpretML) supports Poisson and Gamma objectives but not native Tweedie loss for combined frequency-severity modelling. The original NAM paper (Agarwal et al. 2021, NeurIPS) uses MSE throughout — it was designed for regression and classification on general tabular data, not actuarial targets.

`insurance-gam` implements:
- **Poisson deviance**: the standard actuarial frequency loss, with exposure as offset
- **Tweedie deviance**: with configurable power parameter `p` (p=1 is Poisson, p=2 is Gamma, 1 < p < 2 is compound Poisson-Gamma)
- **Gamma deviance**: for claim severity models

The exposure handling is identical to a GLM: `log(exposure)` is added as a fixed offset to the linear predictor before the link function. This means an ANAM frequency model and a GLM frequency model use the same statistical framework and produce directly comparable shape functions.

---

## Using the library

```bash
uv add insurance-gam
```

The sklearn-compatible API makes the transition from GLM or GBM straightforward:

```python
from insurance_gam.anam import ANAM

model = ANAM(
    loss="poisson",          # or "tweedie", "gamma"
    tweedie_power=1.5,       # only used if loss="tweedie"
    monotone_features={
        "vehicle_age": "increasing",
        "driver_age_young": "decreasing",   # frequency decreasing with age 25-70
        "sum_insured": "increasing",
    },
    lambda_smooth=1e-4,      # smoothness regularisation
    lambda_l1=1e-5,          # feature sparsity
    hidden_sizes=[64, 32],   # subnetwork architecture
    interactions=[           # optional pairwise terms
        ("driver_age_band", "vehicle_group"),
    ],
)

model.fit(X_train, y_train, sample_weight=exposure_train)
```

The `monotone_features` dict specifies which features have constrained directions. Every feature listed there is guaranteed to be monotone after `fit()` completes. Features not listed are unconstrained.

Prediction and scoring follow standard sklearn conventions:

```python
preds = model.predict(X_test, exposure=exposure_test)  # full-period frequency
preds_rate = model.predict(X_test)                     # per-unit-time rate

score = model.score(X_test, y_test, sample_weight=exposure_test)
# Returns negative mean Poisson deviance (higher is better, 0 is perfect calibration)
```

---

## Shape functions for regulatory documentation

The output that matters for model validation and regulatory documentation is the shape function table. Every feature's contribution curve is available as a Polars DataFrame:

```python
shapes = model.shape_functions()
# Returns dict of Polars DataFrames, one per feature

# For a continuous feature:
print(shapes["vehicle_age"])
# shape: pl.DataFrame
# columns: x (feature value), f_x (log-linear contribution), f_x_lower, f_x_upper

# Export for regulatory documentation
for feature, df in shapes.items():
    df.write_json(f"shape_{feature}.json")

# For categorical features, per-category contributions
cat_tables = model.category_tables()
print(cat_tables["occupation"])
# shape: pl.DataFrame
# columns: category, contribution, n_policies, exposure
```

The shape function for a continuous feature is directly analogous to a GLM smooth term or a manual age-band relativity table, except it is continuous and can capture nonlinear effects within each feature's range without requiring manual bucketing decisions.

For model validation under PRA SS1/23, the shape function is the primary documentation artefact: it shows exactly what the model is doing for each feature, the monotonicity constraint is visually verifiable, and the smoothness regularisation means the curve does not have the artificial step-changes that come from fine-grained bucketing.

---

## Comparing ANAM to GLM shapes

One of the most practically useful features for a team migrating from a GLM is the GLM comparison overlay. The ANAM can show you where its shape functions agree with your existing GLM factors and where they diverge:

```python
from insurance_gam.anam import GLMComparison

# glm_params: dict of {feature_name: {level: log_relativity}}
# in the format you'd get from statsmodels or a GLM fit
comparison = GLMComparison(model, glm_params)
fig = comparison.plot_overlay("vehicle_age")
```

The overlay plots the ANAM continuous shape function against the GLM step function on the same axes. Divergences are the model's discoveries: places where the GLM's bucketing decisions lost information or where the ANAM found nonlinear structure within a band. This is the same diagnostic the CatBoost-to-GLM distillation workflow produces via SHAP, except here the ANAM shape function is exact and directly comparable to the GLM parameters in log space.

---

## Where ANAM sits versus the alternatives

The honest competitive picture:

**EBM (InterpretML)** is more mature, faster to train (C++ backend, no PyTorch overhead), and the lookup-table prediction format is extremely fast at inference. Its two concrete gaps versus `insurance-gam`: no native Tweedie deviance for combined frequency-severity modelling, and monotonicity constraints are post-hoc edits rather than architectural guarantees.

**SHAP on a GBM** gives you approximate post-hoc interpretability, useful for understanding which features drive specific predictions but not the same as a continuous shape function, and not equivalent to GLM parameters in log space without additional work.

**Commercial GAM/GLM platforms** (Akur8 and equivalents) provide shape-function interpretability in a validated UI with audit trails. The model building block they expose is essentially what `insurance-gam` provides as open-source Python.

---

## The regulatory argument

We should be direct about what the regulatory advantage is and what it is not.

The FCA confirmed in December 2025 that it will not introduce AI-specific rules for insurance pricing, relying instead on Consumer Duty and the existing principle-based framework. The interpretability advantage of ANAM is not about satisfying a specific rule. It is about the quality of the evidence you can produce when asked to demonstrate that your model produces fair outcomes.

Under Consumer Duty, if your model produces differential pricing across protected characteristic groups, you need to demonstrate that the differential is actuarially justified. An ANAM shape function for the feature driving that differential is a cleaner piece of evidence than a SHAP summary: it is exact, it is a continuous function with no hidden approximation, and it is structurally analogous to the GLM relativity tables that regulators already understand.

Under PRA SS1/23, model validators must assess whether model behaviour is understood at boundaries and under distributional shift. An ANAM model with monotonicity guarantees passes one specific validation test trivially: there are no monotonicity violations to find.

These are genuine advantages. ANAM does not automatically produce a fairer or more accurate model than a GBM. It produces a model whose behaviour is intrinsically more auditable.

---

## What the library does not do

Three honest limitations.

First, **ANAM training is slower than EBM.** PyTorch SGD with per-batch Dykstra projection is not as fast as cyclic gradient boosting on a C++ backend. For a standard UK personal lines book (200k–1m policies, 10–20 features), expect training times of 5–20 minutes on a modern laptop rather than 30–60 seconds for EBM. This is acceptable for monthly or quarterly model rebuilds but would need GPU hardware to run in a tight automated pipeline.

Second, **the additive structure is a constraint.** An ANAM is not a GBM. It cannot capture arbitrary high-order interactions. The pairwise interaction subnetworks extend the architecture to handle the most important two-way terms, but a portfolio with complex multiway interactions (telematics data, connected home sensor data) will lose predictive accuracy relative to a well-tuned GBM. The interpretability-accuracy trade-off is real; the paper (arXiv:2509.08467) benchmarks against GLM, GAM, XGBoost and EBM and shows ANAM competitive or superior in most settings on the beMTPL97 Belgian MTPL benchmark, but "competitive" is not "equal."

Third, **151 tests is not the same as production battle-hardening.** The library is v0.1. Edge cases in categorical handling with rare levels, behaviour under extreme class imbalance, and performance on very wide feature spaces (50+ features) have not all been characterised. Use with a holdout validation set and inspect shape functions before putting this in production.

---

## Getting started

```bash
uv add insurance-gam
```

Python 3.10 or later. Dependencies: PyTorch, scikit-learn, Polars, NumPy, Matplotlib.

The source is at [github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam). The architecture is based on Laub, Pho and Wong, 'An Interpretable Deep Learning Model for General Insurance Pricing', arXiv:2509.08467, September 2025, with the distributional loss suite, monotonicity projection, and sklearn API added for production use.

Start with the `examples/` directory: the `frequency_model.py` script runs a Poisson ANAM on synthetic motor data and exports shape function plots and JSON tables. The `glm_comparison.py` script shows the overlay between an ANAM and a statsmodels GLM fit on the same data. Both run in under ten minutes on a laptop.

---

Your SHAP values are not your model. They are a post-hoc explanation of a model that does not intrinsically know why it made any prediction. If you want a model whose interpretability is structural rather than approximate — one where the shape functions are the model, not a description of it — that is what this library provides.

- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/)
- [Automated Interaction Detection for Insurance GLMs: CANN, NID and LR Testing](/2026/02/27/finding-the-interactions-your-glm-missed/)
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/)
- [From CatBoost to Radar: GBM-to-GLM Distillation](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/)
