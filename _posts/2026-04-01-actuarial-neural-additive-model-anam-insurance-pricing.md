---
layout: post
title: "The Neural GAM Actuaries Can Explain to Their CRO"
date: 2026-04-01
categories: [modelling, techniques]
tags: [ANAM, NAM, GAM, neural-network, interpretability, monotonicity, FCA, Equality-Act, motor-pricing, Poisson-deviance, Tweedie, insurance-gam, PyTorch, sklearn, EBM, ICEnet, Laub-Pho-Wong, NAAJ-2025, python, uk-insurance, GLM, GBM, pricing-actuary, shape-functions, relativities]
description: "ANAM (Laub, Pho, and Wong, NAAJ 2025) fits each rating factor as a neural subnetwork with hard monotonicity constraints, exposure offsets, and proper actuarial losses. The insurance-gam library wraps the full implementation behind a five-line sklearn interface. Here is what it does, where it beats EBM, and when to use it."
math: true
author: burning-cost
---

The standard actuarial compromise between interpretability and performance has been: GLM for the pricing committee, GBM for the model validation team, hope they do not compare notes. That compromise just got more interesting. The Actuarial Neural Additive Model (ANAM), published in the *North American Actuarial Journal* in 2025 by Laub, Pho, and Wong at UNSW ([DOI: 10.1080/10920277.2024.2365595](https://doi.org/10.1080/10920277.2024.2365595)), offers a third path: the predictive accuracy of a neural network with a shape function for every rating factor that your CRO can read off a chart. This post explains what ANAM actually does, where it beats EBM and standard NAMs, and how the `insurance-gam` library makes it usable today.

---

## What is a Neural Additive Model?

A Generalised Additive Model decomposes as:

$$f(\mathbf{x}) = b + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p)$$

Each term is a smooth function of exactly one feature. The shape of each term is the model. You can plot it, inspect it, override it if the curve does something actuarially unreasonable. An actuary can look at the driver age curve and say "this looks right — frequency rises until 25, plateaus, then increases again after 70." They can have that conversation with the pricing committee and with Lloyd's. Trying to have the same conversation about a 3,000-tree GBM is categorically different.

Standard GAMs fit those curves using splines (P-splines in mgcv, cyclic boosted stumps in EBM). ANAM fits each curve using a small neural network — a multi-layer perceptron (MLP) with, by default, two hidden layers of 64 and 32 units. The architecture for each continuous feature subnetwork is:

$$f_i(x_i): \text{Linear}(1, 64) \to \text{ReLU} \to \text{Linear}(64, 32) \to \text{ReLU} \to \text{Linear}(32, 1)$$

The subnetwork outputs are summed to produce the linear predictor $\eta$, which then goes through the link inverse to produce the mean prediction $\mu$. Same glass-box structure as a GAM. Stronger representational capacity than a spline for features with complex, non-smooth shapes — NCD effects with a cliff at year 3, or vehicle group curves that reflect discrete ABI group reassignments.

The original Neural Additive Model paper (Agarwal et al., NeurIPS 2021) established the architecture. What Laub, Pho, and Wong added is everything that makes it actuarially deployable.

---

## What ANAM adds that standard NAMs do not

Three additions that make ANAM usable where standard NAMs are not.

**Hard monotonicity.** If your actuarial knowledge says claim frequency should not decrease with NCD years, you declare that constraint at model construction. After every gradient step, the trainer calls `project_monotone_weights()`, which clamps all weight matrices in that subnetwork to non-negative values via Dykstra projection onto the monotone constraint set. For a ReLU network with all-non-negative weights, the output is guaranteed non-decreasing — mathematically, not probabilistically. The guarantee holds at every point on the feature domain, not just on the training grid.

EBM's `monotone_constraints` parameter is a training hint that can fail in edge regions of sparse data. ICEnet (Richman & Wüthrich, 2024) uses a penalty term that must be tuned and does not provide a hard guarantee. ANAM's constraint is unconditional.

Why does this matter in the UK specifically? The Equality Act 2010 and FCA pricing guidance create liability if a model prices a protected characteristic (driver age, after the gender directive) in a direction that contradicts the insurer's stated actuarial evidence. If you tell a regulator that your model respects the documented relationship between age and risk, and a sufficiently extreme data combination causes the curve to invert in the 80–85 band, you have a problem. Showing that monotonicity is architecturally enforced — not just a modeller's intention — is a qualitatively different conversation with a Supervisory Actuary or the FCA.

**Exposure offset.** A policy with 0.5 years of exposure should contribute proportionally to a full-year policy in both loss function and prediction. ANAM handles this identically to a log-link GLM: $\log(\text{exposure})$ is added to the linear predictor before the exponential:

$$\mu = \exp(\eta + \log e) = e \cdot \exp(\eta)$$

The Poisson deviance is then evaluated against the claim count $y$, not the rate $y/e$. Standard NAMs have no exposure handling. Using them on insurance frequency data without workarounds produces systematically biased predictions on any portfolio with non-uniform policy durations — which is every real portfolio.

**Proper distributional losses.** ANAM minimises Poisson deviance for claim frequency, Gamma deviance for severity, and Tweedie deviance (default $p = 1.5$) for pure premium. These are the proper scoring rules for the data-generating process actuaries actually work with. The choice between Poisson and Tweedie is not cosmetic: it controls how the loss weighs high-frequency versus zero-claim observations, and it determines whether the fitted curve is consistent with the assumed distribution for regulatory purposes.

---

## ANAM versus EBM: when to use which

EBM (Microsoft's Explainable Boosting Machine, via `interpret`) is the current default for interpretable insurance pricing in the UK market. It uses cyclic gradient boosting — one feature updated per round — to build shape functions as lookup tables. Pairwise interactions are detected automatically via the FAST algorithm. It is fast, requires no hyperparameter tuning for most problems, and is well-documented for actuarial use (Suessmann & Parodi, GIRO 2023).

ANAM is not a replacement for EBM. It is a different tool for different constraints.

Use EBM when:
- You do not know which interaction pairs matter — FAST auto-detects them, ANAM requires explicit declaration
- Training speed matters — EBM fits a UK motor book in seconds; ANAM takes minutes on CPU
- The team has no PyTorch experience
- Categorical variables have very high cardinality (>100 levels with sparse coverage) — EBM's look-up table handles this more stably than an embedding

Use ANAM when:
- Hard monotonicity is required and specifiable in advance — NCD years, driver age, years in business
- You need explicit 2D interaction surfaces for specific known pairs (driver age × vehicle group) presented as heatmaps
- PyTorch is already in the stack and gradient-based training is familiar
- Academic publication or Lloyd's ICA submission requires a citable neural additive methodology

The most credible approach for a UK pricing team is both: fit EBM as the baseline, fit ANAM with the key monotone constraints, compare shape functions. Where they agree, you have confidence. Where they disagree, there is something to investigate — almost always a sparsity issue at the tails, which is exactly where EBM's boosted stumps and ANAM's neural smoothing will diverge.

---

## Code: fitting ANAM for motor frequency

The `insurance-gam` library ships ANAM as a sklearn-compatible estimator. Install it:

```bash
pip install insurance-gam
```

The full API lives in `insurance_gam.anam`. There are two ways to specify features. The shorthand, using constructor keyword arguments:

```python
from insurance_gam.anam import ANAM

model = ANAM(
    feature_names=["driver_age", "vehicle_age", "region", "ncd_years"],
    categorical_features=["region"],
    monotone_increasing=["driver_age"],
    monotone_decreasing=["ncd_years"],
    interaction_pairs=[("driver_age", "vehicle_age")],
    link="log",
    loss="poisson",
    hidden_sizes=[64, 32],
    lambda_smooth=1e-4,
    n_epochs=200,
    patience=20,
)

model.fit(X_train, y_train, sample_weight=exposure_train)
mu_pred = model.predict(X_test, exposure=exposure_test)
```

The explicit form, using `FeatureConfig`, gives you per-feature hidden layer sizes and is required if you want per-feature subnetwork depth control:

```python
from insurance_gam.anam import ANAM
from insurance_gam.anam.model import FeatureConfig

configs = [
    FeatureConfig("driver_age",  monotonicity="increasing"),
    FeatureConfig("vehicle_age", monotonicity="none"),
    FeatureConfig("region",      feature_type="categorical", n_categories=12),
    FeatureConfig("ncd_years",   monotonicity="decreasing"),
]

model = ANAM(
    feature_configs=configs,
    interaction_pairs=[("driver_age", "vehicle_age")],
    link="log",
    loss="poisson",
    hidden_sizes=[64, 32],
    lambda_smooth=1e-4,
    n_epochs=200,
    patience=20,
)

model.fit(X_train, y_train, sample_weight=exposure_train)
mu_pred = model.predict(X_test, exposure=exposure_test)
```

Note that `sample_weight` in `fit()` maps to exposure — not to arbitrary observation weights. Doubling exposure for a policy is not the same as duplicating the observation, and the implementation respects that distinction in the Poisson deviance calculation.

**Extracting shape functions.** After fitting, `shape_functions()` evaluates each subnetwork on a grid over its training range and returns a dict of `ShapeFunction` objects:

```python
shapes = model.shape_functions(n_points=200)

# Plot the driver age curve
shapes["driver_age"].plot()

# GLM-style multiplicative relativities, centred at the median
age_rel = shapes["driver_age"].to_relativities()
# Returns a Polars DataFrame: columns [x, relativity, log_relativity, feature]

# Tabular export for documentation
age_df = shapes["driver_age"].to_polars()
```

The `to_relativities()` method exponentiates the shape function values and normalises at the median feature value, producing multiplicative factors in the form actuaries recognise from GLM partial plots. A `log_relativity` column is included for direct comparison with GLM coefficients.

**Comparing against the trained EBM.** Because both EBM and ANAM expose their shape functions as arrays over a feature grid, direct comparison is straightforward. Plot both age curves on the same axes; the divergence at the tails is precisely the region where the monotonicity constraint is doing work.

**Cross-validation.** Because ANAM is sklearn-compatible, `cross_val_score` works directly:

```python
from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(
    model, X, y,
    cv=5,
    scoring="neg_mean_poisson_deviance",
    fit_params={"sample_weight": exposure},
)
print(f"CV Poisson deviance: {-scores.mean():.4f} ± {scores.std():.4f}")
```

---

## The GLM warm-start gap

One gap between the paper and the current implementation is worth flagging honestly: the original ANAM paper describes warm-starting the network from GLM parameters — mapping each GLM coefficient $\hat{\beta}_i$ to the output layer of subnetwork $f_i$. This gives the network a sensible starting point from actuarial first principles and avoids the first 30–50 epochs where the model is rediscovering what a GLM already knows.

The current `insurance-gam` implementation uses Xavier-uniform initialisation — standard for neural networks, but it means the training curve begins from an uninformed state. In practice, this means:

- 30–50 extra epochs before the validation deviance falls below the GLM baseline
- Early stopping can occasionally terminate before the model has meaningfully improved on the GLM if patience is set too tight
- The "GLM to ANAM" transition narrative — which pricing teams find reassuring when presenting to governance — requires an explicit comparison rather than an automatic handoff

GLM warm-start is the planned next addition to the library. It would reduce training epochs by roughly 40% and provide a principled audit trail showing ANAM as a smooth extension of the existing GLM, rather than a parallel model that happens to agree with it.

---

## What this means for UK pricing teams

ANAM is not a neural network that happens to be interpretable. It is a GAM that happens to be neural. The additive structure is the architecture, not a post-hoc explanation. Each rating factor's shape function is a first-class output with a gradient, a monotonicity guarantee, and a direct GLM-style relativity table.

For UK pricing teams operating under FCA guidance and Consumer Duty, the combination of architectural interpretability and regulatory-grade monotonicity guarantees makes ANAM the most defensible neural pricing model currently available. The alternative — running a GBM and explaining it via SHAP after the fact — produces an explanation of a black box rather than a glass box. Those are not the same thing, and technical actuaries reviewing your ORSA or Lloyd's ICA submission can tell the difference.

The `insurance-gam` library wraps the full implementation — 2,662 lines of tested PyTorch code — behind a five-line sklearn interface. The shape functions come out as Polars DataFrames in the relativity format actuaries already use. The monotonicity constraints require one line per feature.

The question for pricing teams is not "can we afford to use neural models?" It is "can we afford not to, when our EBM competitor has no monotonicity guarantee?"

---

**Install:** `pip install insurance-gam`

**Source:** [github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam)

**Paper:** Laub, Pho, and Wong, "Actuarial Neural Additive Models", *North American Actuarial Journal*, 2025. [DOI: 10.1080/10920277.2024.2365595](https://doi.org/10.1080/10920277.2024.2365595)
