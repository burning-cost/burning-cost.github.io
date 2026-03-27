---
layout: post
title: "EBM, ANAM, or PIN: Choosing an Interpretable Architecture for UK Insurance Pricing"
date: 2026-03-14
categories: [libraries, pricing, interpretability]
tags: [gam, ebm, nam, anam, pin, interpretability, GLM, GBM, tweedie, poisson, monotonicity, shapley, insurance-gam, french-mtpl, python]
description: "Three interpretable architectures for UK insurance pricing: EBM, ANAM, and PIN via insurance-gam. Refuse the GLM-vs-GBM accuracy trade-off with factor tables."
---

*This post is the comparison guide: when to use EBM, ANAM, or PIN, and what insurance-gam now contains. For EBM workflow details, read [EBMs for Insurance Pricing](/2026/03/09/explainable-boosting-machines-for-insurance-pricing/). For the ANAM architecture and neural additive model theory, read [Actuarial Neural Additive Models](/2026/03/13/your-interpretable-model-isnt-interpretable-enough/).*


The pricing committee has seen this presentation before. The GBM outperforms the production GLM by 5 Gini points on the holdout set. The SHAP plots look sensible. The double-lift chart is clean. And then someone asks: "What is the model doing with drivers aged 19 to 21 in the South East?" and the answer is a waterfall chart of approximate feature contributions, not a factor table, and the model stays in the notebook for another quarter.

The standard response to this dynamic is: build an EBM, or use a GAM, or extract SHAP relativities from the GBM. Each of these is a reasonable answer to a slightly different version of the problem. The difficulty is knowing which version of the problem you are actually solving.

[`insurance-gam`](/insurance-gam/) now contains three distinct interpretable model families, unified under one install. Each solves the GLM-vs-GBM trade-off differently. Choosing between them is a decision that should happen before you start fitting, not after.

```bash
uv add insurance-gam
```

---

## The problem, stated precisely

A GLM is interpretable because it is additive in log space and its parameters have a direct statistical interpretation: each coefficient is an offset to the linear predictor, directly comparable in log space, and the factor table is the model. The limitation is that the shape function for each feature is constrained: linear, or grouped into manually defined bands. A driver age curve that turns sharply at 21, flattens through the 30s, then rises again at 70 requires either fine-grained bucketing decisions made by a human or a polynomial term that may behave badly outside its training range.

A GBM removes that constraint. The shape function for each feature is non-parametric, fitted by gradient boosting, and can capture any pattern present in the data. The limitation is that the GBM has no additive structure by design. Feature contributions interact in a way that cannot be cleanly separated - which is why SHAP values are an approximation, not the model.

The family of models that resolves this tension is the Generalised Additive Model: the prediction is a sum of per-feature terms, each learned non-parametrically. The GLM is a restricted GAM (linear shape functions). The GBM is something else. EBMs, NAMs, and PIN are all genuinely additive, genuinely non-parametric, and genuinely readable.

The question is which architecture suits your portfolio, your governance process, and your production constraints.

---

## The three architectures

### EBM: the lookup-table model

The `InsuranceEBM` class wraps interpretML's `ExplainableBoostingRegressor` - a GA2M fitted by cyclic gradient boosting. Each feature gets a non-parametric shape function learned by repeated rounds of shallow-tree boosting. The shape functions are stored as lookup tables: for any given feature value, there is a pre-computed log-score contribution that a rating engine can look up directly. No forward pass through a network. No floating-point arithmetic at runtime.

```python
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

model = InsuranceEBM(
    loss='poisson',
    interactions='3x',
    monotone_constraints={'ncd': -1, 'vehicle_age': -1},
)
model.fit(X_train, y_train['claim_count'], exposure=y_train['exposure'])

rt = RelativitiesTable(model)
print(rt.table('driver_age'))
# bin_label    raw_score    relativity
# (17, 21]      0.682        1.978
# (21, 25]      0.441        1.554
# (25, 35]      0.000        1.000   ← modal bin
# (35, 50]     -0.134        0.875

rt.export_excel('relativities_v3.xlsx')
```

The `relativity` column is what goes into Radar or Emblem. Not derived from the model. The model.

EBMs train fast - seconds to minutes on a standard UK personal lines book - because the C++ boosting backend has been optimised heavily. On the French CTP dataset (670,000 policies, Poisson frequency), an EBM with `interactions='3x'` reduces deviance by 8-12% relative to a standard GLM with manual interactions, comparable to CatBoost on the same data.

The two limitations to know before you start: the EBM has no native Tweedie deviance for combined frequency-severity modelling (use `loss='poisson'` for frequency, `loss='gamma'` for severity, and multiply - or use ANAM if you need true Tweedie), and monotonicity constraints are enforceable post-fit via isotonic regression but that is not the same as architectural enforcement.

---

### ANAM: the smooth-curve model with guaranteed monotonicity

The Actuarial Neural Additive Model (Laub, Pho, Wong - UNSW, arXiv:2509.08467, September 2025) is a neural GAM: each feature gets its own subnetwork, predictions are additive over those subnetworks, and the model is interpretable by construction.

```python
from insurance_gam.anam import ANAM

model = ANAM(
    loss="tweedie",
    tweedie_p=1.5,
    monotone_increasing=["vehicle_age"],
    monotone_decreasing=["driver_age_young", "ncd_years"],
    lambda_smooth=1e-4,
    hidden_sizes=[64, 32],
)
model.fit(X_train, y_train, sample_weight=exposure_train)
```

Three things that distinguish ANAM from EBM:

**Tweedie loss is native.** `loss="tweedie"` with `tweedie_power=1.5` fits a compound Poisson-Gamma pure premium model directly. A single model, a single set of shape functions, one set of relativities for the rating engine. Whether this beats a separate frequency × severity approach depends on your data; the advantage is simplicity and the fact that you are not multiplying two approximations.

**Monotonicity is architectural.** For any feature in `monotone_features`, the monotonicity guarantee is enforced by clamping network weights via Dykstra projection at every gradient step. The resulting shape function has zero violations at any input point - not "very few violations in high-exposure regions" but zero, provably, by inspection. This matters for UK pricing: rate inversions in low-exposure regions of the driver age distribution are the kind of thing aggregators find and exploit.

**Shape functions are continuous curves.** The EBM shape function is a step function over bins. The ANAM shape function is a smooth curve across the full feature range - directly analogous to a GLM smooth term, without bucketing decisions.

```python
shapes = model.shape_functions()
print(shapes["driver_age_young"])
# x       f_x      f_x_lower    f_x_upper
# 17.0    0.672    0.641        0.703
# 18.0    0.611    0.584        0.638
# ...
```

The output of `shape_functions()` is a Polars DataFrame per feature. Under PRA SS1/23, this is a cleaner primary documentation artefact than a SHAP summary: it shows exactly what the model does for each feature, the monotonicity constraint is visually verifiable, and the smoothness means the curve does not have artificial jumps at bucket boundaries.

The cost: ANAM trains in 5-20 minutes on a laptop rather than EBM's 30-90 seconds. PyTorch SGD with per-batch projection is not a C++ boosting backend.

---

### PIN: the interaction surface model

Pairwise Interaction Networks (Richman, Scognamiglio, Wüthrich - arXiv:2508.15678, August 2025) extend the additive structure to exact pairwise interaction surfaces. The prediction is:

```
mu_i = exp( b + sum_j f_j(x_j) + sum_{j<k} f_{jk}(x_j, x_k) )
```

where each pairwise term `f_{jk}` is a two-dimensional surface - the BonusMalus × VehBrand interaction, as a table - not a SHAP approximation but the model's actual output.

```python
from insurance_gam import PINModel, PINEnsemble, PINDiagnostics

features = {
    "age_driver":  "continuous",
    "bonus_malus": "continuous",
    "density":     "continuous",
    "veh_age":     "continuous",
    "veh_power":   "continuous",
    "veh_brand":   11,   # categories
    "region":      22,
}

model = PINModel(
    features=features,
    embedding_dim=10,
    hidden_dim=20,
    token_dim=10,
    shared_dims=(30, 20),
    loss="poisson",
    lr=0.001,
    max_epochs=300,
    patience=20,
)
model.fit(X_train, y_train, exposure=exposure_train)

# Exact Shapley values — not an approximation
# Cost: 2*(q+1) forward passes per sample per background sample
shap_values = model.shapley_values(X_test, X_background, n_background=200)
```

PIN's architecture achieves interaction modelling without parameter explosion by training a single shared network across all feature pairs, differentiated by learned interaction tokens - a d0-dimensional vector per pair that tells the shared network which surface it is computing. For 9 features (9 main effects and 36 pairwise interactions, 45 terms total), the reference parameter count from the paper is 4,147.

The benchmark result is the honest reason to use PIN: on freMTPL2freq (610,000 French motor policies, out-of-sample Poisson deviance × 10⁻²), ensemble PIN achieves 23.667 - better than GLM (24.102), GAM (23.956), ensemble FNN (23.783), and the Credibility Transformer (23.711). The improvement over GLM is 0.435 units. At portfolio scale, this translates to meaningfully better risk discrimination without sacrificing the ability to tabulate and explain the model's decisions.

The interaction surface for any pair is exact:

```python
diag = PINDiagnostics(model)
diag.plot_surface("bonus_malus", "veh_brand", X_background)
```

What comes out of `plot_surface()` is a 2D heatmap of `f_{jk}(x_j, x_k)` - the actual model output for that pair, computed by passing a grid of input values through the fitted network. It can be printed as a rating factor table.

PIN has no native monotonicity constraints in v1. If "claims frequency does not decrease with bonus-malus level" is a hard tariff requirement, ANAM is the better choice for that feature.

---

## Which one to use

The decision depends on three questions: what loss function do you need, whether you have hard monotonicity requirements, and how much you care about interaction surfaces versus main-effect accuracy.

| | EBM | ANAM | PIN |
|---|---|---|---|
| Poisson (frequency) | Yes | Yes | Yes |
| Gamma (severity) | Yes | Yes | Yes |
| Tweedie (pure premium, native) | No | **Yes** | Yes |
| Architectural monotonicity | No (post-hoc edit) | **Yes** | No |
| Exact pairwise interaction surfaces | Yes (lookup tables) | Pairwise subnetworks | **Yes (neural surfaces)** |
| Exact Shapley values | No | No | **Yes** |
| Training time (200k policies, laptop) | ~60 seconds | 5-20 minutes | 10-30 minutes |
| Predictive accuracy (freMTPL2freq, Poisson deviance × 10⁻²) | ~23.75 | Competitive vs GLM | **23.667** |
| Output format | Lookup tables | Smooth curves + JSON | Neural surfaces + tables |
| Inference speed | Fast (table lookup) | Fast (forward pass) | Fast (forward pass) |

The practical guidance:

**Start with EBM** if you are migrating from an existing GLM and need to demonstrate that the EBM's factor tables are directly comparable. The `GLMComparison` class overlays EBM shape functions against your existing GLM factor table, feature by feature. It is the lowest-friction first step.

**Use ANAM** when Tweedie pure premium is the target, when you need architectural monotonicity guarantees for a tariff submission, or when the regulatory documentation benefit of continuous smooth shape functions outweighs the slower training time. ANAM is also the natural choice for a severity model where a combined frequency-severity approach with a single interpretable set of relativities is preferable to multiplying two separate models.

**Use PIN** when you specifically need to characterise pairwise interaction surfaces for a pricing committee, when you want the best published accuracy on the interpretability-constrained model class, and when exact Shapley values are a governance requirement. PIN's ensemble achieves results comparable to or better than a well-tuned GBM while remaining fully decomposable.

If the committee question is "what is the interaction between BonusMalus and VehBrand doing?", PIN gives you an exact answer. EBM gives you lookup-table interaction surfaces - also exact, but in a different format. ANAM with `interactions=[("bonus_malus", "veh_brand")]` adds that pair as a dedicated subnetwork. All three are better answers than SHAP on a GBM.

---

## What they share

All three models are in `insurance-gam` under the same install. They share the same exposure handling convention - `log(exposure)` as a fixed offset to the linear predictor, matching the GLM convention - and the same diagnostic suite: Gini coefficient, double-lift chart, A/E by segment, Lorenz curve, all exposure-weighted.

```python
from insurance_gam.ebm import gini, double_lift

g = gini(y_test['claim_count'], preds, exposure=y_test['exposure'])
lift = double_lift(y_test['claim_count'], preds, exposure=y_test['exposure'], n_bands=10)
```

The diagnostics work identically whether `preds` came from `InsuranceEBM`, `ANAM`, or `PINModel`. You can run all three models on the same holdout and compare the double-lift charts before deciding which to take to committee.

---

## The regulatory argument, stated directly

The FCA has not introduced AI-specific pricing rules, relying on Consumer Duty and the existing principle-based framework. The PRA's SS1/23 (effective May 2024) requires model validators to understand decision boundaries and test behaviour at margins — the [insurance-governance library](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) automates the statistical validation suite that SS1/23 mandates. Neither document mandates a specific architecture.

What they do create is a practical difference in the quality of evidence you can produce. A SHAP waterfall chart on a GBM answers "which features drove this prediction" approximately. An EBM factor table, an ANAM shape function, or a PIN interaction surface answers the same question exactly - and answers additional questions a SHAP chart cannot, such as "what does the model predict if only this one feature changes, across its full range?"

Under Consumer Duty, if your model produces differential pricing across protected characteristic groups, [demonstrating actuarial justification](/2026/03/20/fca-consumer-duty-pricing-fairness-python/) is easier when the shape function for the driving feature is a readable curve rather than an average marginal SHAP contribution. Under SS1/23, a model with guaranteed monotone constraints passes one specific validation test without needing to run it.

These are genuine advantages. None of them guarantee a fairer or more accurate model. They guarantee a model whose behaviour is intrinsically more auditable — which is a different and narrower claim. When [insurance-monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) flags Gini drift and triggers a refit, a model with stored shape functions makes the "what changed?" conversation with the pricing committee far more tractable than SHAP plots on a black-box.

---

## Installation and getting started

```bash
uv add insurance-gam

# With interpretML backend for EBM:
uv add "insurance-gam[interpret]"

# With Excel export:
uv add "insurance-gam[interpret,excel]"
```

Python 3.10+. EBM requires `interpret >= 0.7.0`. ANAM and PIN require `torch >= 2.0`. All three require `polars >= 0.20`.

The repository is at [github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam) ([PyPI](https://pypi.org/project/insurance-gam/)). The `examples/` directory has three scripts — `ebm_frequency.py`, `anam_pure_premium.py`, and `pin_benchmark.py` — each running on synthetic motor data and producing factor tables, shape function plots, and validation metrics. All three complete in under 30 minutes on a laptop.

- [EBMs for Insurance Tariff Construction: Boosting Accuracy with GLM Transparency](/2026/03/09/explainable-boosting-machines-for-insurance-pricing/)
- [Actuarial Neural Additive Models: Exact Interpretability with Tweedie Loss](/2026/03/13/your-interpretable-model-isnt-interpretable-enough/)
- [Pairwise Interaction Networks: The Model That Beats GBMs and Prints as a Relativities Table](/2026/03/13/insurance-pin/)
- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/)
- [Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/)
- [Does insurance-gam Actually Work for Insurance Pricing?](/2026/03/24/does-insurance-gam-actually-work-pricing/) — benchmark results on freMTPL2: EBM versus CatBoost versus GLM across Gini, double-lift, and A/E
