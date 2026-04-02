---
layout: post
title: "EBMs for Insurance Pricing: Better Than a GLM, Readable by a Pricing Committee"
date: 2026-03-09
author: Burning Cost
categories: [pricing, machine-learning, libraries]
tags: [EBM, interpretML, GAM, GLM, Poisson, Tweedie, Gamma, monotonicity, relativities, Gini, double-lift, explainability, FCA, PRA, python, insurance-gam]
description: "insurance-gam wraps EBM for UK pricing teams: Poisson/Tweedie loss, exposure offsets, RelativitiesTable, MonotonicityEditor, GLM comparison diagnostics."
---

<div class="notice--warning" markdown="1">
**Package update:** `insurance-ebm` has been consolidated into [`insurance-gam`](/insurance-gam/). Install with `uv add insurance-gam` — all functionality described here is available as a submodule. [View on GitHub →](https://github.com/burning-cost/insurance-gam)
</div>


*This post is an EBM deep-dive: how Explainable Boosting Machines work, how to fit them for Poisson/Gamma/Tweedie insurance targets, and the full actuarial workflow - relativity tables, monotonicity editing, GLM comparison, Gini diagnostics. If you want to understand the ANAM architecture, read [Actuarial Neural Additive Models](/2026/03/13/your-interpretable-model-isnt-interpretable-enough/). If you want to choose between EBM, ANAM, and PIN, read the [comparison guide](/2026/03/14/insurance-gam-interpretable-nonlinearity/).*


The two questions that come up in every conversation about replacing a GLM with something more powerful are: can you explain it to the pricing committee, and can you explain it to the regulator?

GLMs pass both tests. The factor table is the explanation. Each row says exactly how much a rating factor moves the rate, and every actuary in the room has been reading tables like that for their entire career. The downside is that GLMs leave predictive performance on the table. They cannot capture nonlinear within-factor effects without manual bucketing decisions, and they cannot detect interactions without being told where to look.

Gradient boosted models go the other way. On most insurance datasets with enough volume, a well-tuned GBM outperforms a GLM. The difficulty is the explanation step. SHAP values are post-hoc attributions, not the model itself. "The SHAP values look reasonable" is a meaningful statement about the training data but a weaker one about model behaviour at margins, under distributional shift, or on the specific policy you're trying to explain to a customer who has received an adverse decision.

Explainable Boosting Machines sit between these two. They are a class of Generalised Additive Model fitted using gradient boosting - the prediction is an additive sum of per-feature shape functions, each learned by a boosting process that cycles through features one at a time. The shape functions are exact and intrinsic: you are not post-hoc summarising a black box, you are reading the model directly. The boosting fitting process captures nonlinear within-feature effects that a GLM would miss. And the family of models they belong to - GA2M when you include pairwise interactions - has the theoretical underpinning of Lou et al. (2013, KDD) and the production implementation in interpretML's `ExplainableBoostingMachine`, documented in Nori et al. (2019, arXiv:1909.09223).

What interpretML already gives you is substantial: Poisson, Gamma, and Tweedie deviance objectives; exposure as an init_score offset; monotonicity constraints at fit time; automatic interaction detection via the FAST algorithm. These are not bolted-on features. They are first-class parts of the API.

What it does not give you is the insurance workflow layer. The output format a UK pricing team expects is a relativity table, not a raw score array. The validation process involves Gini coefficients, double-lift charts, and A/E by segment - not sklearn's R² scorer. The regulatory documentation requires plotted shape functions with base levels clearly identified. And the business constraint process involves editing shape functions after the fit, not before.

[`insurance-gam`](https://github.com/burning-cost/insurance-gam) adds that layer.

---

## Installation

```bash
uv add insurance-gam[interpret]
```

The `[interpret]` extra pulls in interpretML, which has a C++ backend and takes a moment to compile on first install. If you already have interpretML installed and want only the diagnostics and analysis tools:

```bash
uv add insurance-gam
```

---

## Fitting a frequency model

The `InsuranceEBM` class wraps `ExplainableBoostingRegressor` and adds exposure-aware fit and predict methods:

```python
import polars as pl
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable
from insurance_gam.ebm import gini, double_lift

# X_train: polars or pandas DataFrame of rating factors
# y_train: claim counts (for Poisson frequency)
# exposure: earned car years

model = InsuranceEBM(
    loss='poisson',
    interactions='3x',           # ~3 * n_features pairwise terms via FAST detection
    monotone_constraints={
        'ncd': -1,               # more NCD = lower frequency
        'vehicle_age': -1,       # older car = lower comprehensive frequency
    }
)

model.fit(X_train, y_train['claim_count'], exposure=y_train['exposure'])
```

Exposure enters as `log(exposure)` passed to `init_score`, which is the standard actuarial convention for a log-link model. The model learns the shape functions on the exposure-adjusted scale. When you predict:

```python
# Returns expected claim counts (rate × exposure)
preds = model.predict(X_test, exposure=y_test['exposure'])

# Returns per-unit-time claim rate (no exposure scaling)
rates = model.predict(X_test)
```

For severity or pure premium modelling, swap `loss='gamma'` or `loss='tweedie'` with `variance_power=1.5`. The exposure handling is identical.

---

## Relativity tables

The `RelativitiesTable` class extracts shape functions from the fitted EBM and presents them in the format a pricing committee expects: relativities normalised to a base level, with the base defined as the modal bin (the bin with the highest training weight).

```python
rt = RelativitiesTable(model)

# Single factor - returns a polars DataFrame
print(rt.table('driver_age'))
# shape: (15, 3)
# ┌────────────────────┬───────────┬────────────┐
# │ bin_label          ┆ raw_score ┆ relativity │
# │ ---                ┆ ---       ┆ ---        │
# │ str                ┆ f64       ┆ f64        │
# ╞════════════════════╪═══════════╪════════════╡
# │ (17.0, 21.0]       ┆  0.682    ┆  1.978     │
# │ (21.0, 25.0]       ┆  0.441    ┆  1.554     │
# │ (25.0, 35.0]       ┆  0.000    ┆  1.000     │  ← base (modal bin)
# │ (35.0, 50.0]       ┆ -0.134    ┆  0.875     │
# │ (50.0, 65.0]       ┆ -0.089    ┆  0.915     │
# │ (65.0, 99.0]       ┆  0.213    ┆  1.237     │
```

The `relativity` column is what goes into your rating engine. The `raw_score` is the additive log contribution - directly comparable to a GLM coefficient.

For a portfolio overview:

```python
# All features ranked by leverage (range = max_relativity / min_relativity)
print(rt.summary())
# ┌───────────────┬────────┬───────────────┬───────────────┬────────┐
# │ feature       ┆ n_bins ┆ min_relativity┆ max_relativity┆ range  │
# ╞═══════════════╪════════╪═══════════════╪═══════════════╪════════╡
# │ driver_age    ┆ 15     ┆ 0.741         ┆ 2.156         ┆ 2.908  │
# │ vehicle_group ┆ 12     ┆ 0.812         ┆ 1.893         ┆ 2.331  │
# │ area          ┆  8     ┆ 0.887         ┆ 1.642         ┆ 1.851  │
```

Exportable to Excel with one line:

```python
rt.export_excel('relativities_v3.xlsx')  # requires uv add insurance-gam[excel]
```

One sheet per rating factor, one row per bin. This is what goes to the pricing committee before sign-off.

---

## Post-fit monotonicity editing

Monotonicity constraints can be set at fit time (the interpretML native API). But fit-time constraints are not always sufficient: the constraint applies globally but the shape function can still be non-monotone in low-exposure tails where the signal is weak and the boosting process has picked up noise.

The `MonotonicityEditor` applies isotonic regression to the stored shape function scores after fitting. This is a soft enforcement: it modifies the term scores in the fitted model, not the boosting trees themselves. The result is monotone on the data range, which is what matters for your rating engine, but the underlying model structure is not altered.

```python
from insurance_gam.ebm import MonotonicityEditor

me = MonotonicityEditor(model)

# Inspect before editing
scores_before = me.get_scores('ncd')

# Enforce decreasing: higher NCD = lower score
me.enforce('ncd', direction='decrease')

# Visualise the edit
me.plot_before_after('ncd', scores_before=scores_before)
```

Use this to clean up tail noise, not to override systematic model signals. If the model is confidently non-monotone across a large portion of the feature range, that is a model signal worth investigating - isotonic regression will suppress it, but you should understand it first.

---

## GLM comparison

When migrating from a GLM or running models in parallel for validation, the `GLMComparison` class overlays the EBM's continuous shape function against your existing GLM factor table.

```python
from insurance_gam.ebm import GLMComparison

# Supply pre-computed GLM relativities as a polars DataFrame
glm_vehicle_group = pl.DataFrame({
    'level': ['G1', 'G2', 'G3', 'G4', 'G5', 'G6'],
    'relativity': [0.83, 0.91, 1.00, 1.14, 1.28, 1.47]
})

cmp = GLMComparison(model)
cmp.plot_comparison('vehicle_group', glm_relativities=glm_vehicle_group)
```

The chart plots both on the same axes, in relativity space. Where the lines converge, the EBM and GLM agree: the GLM bucketing captured the factor's effect adequately. Where they diverge, the EBM has found nonlinear structure within a band that the GLM's bucketing obscured, or the EBM has responded to data the GLM discounted.

To identify which factors diverge most across your full model:

```python
glm_rels_by_feature = {
    'driver_age': glm_driver_age,
    'vehicle_group': glm_vehicle_group,
    'area': glm_area,
}

print(cmp.divergence_summary(glm_relativities_by_feature=glm_rels_by_feature))
```

This is the same diagnostic the SHAP-to-relativity workflow produces - but here the EBM relativities are the model, not an approximation of one.

---

## Diagnostics

The diagnostics module covers the standard actuarial validation toolkit, all handling exposure weighting correctly.

**Gini coefficient.** The normalised Gini measures ordering power: how well the model separates high-risk from low-risk policies. Normalised by the oracle Gini (sort by actuals), so it is bounded [-1, 1] and interpretable across datasets.

```python
from insurance_gam.ebm import gini, lorenz_curve

g = gini(y_test['claim_count'], preds, exposure=y_test['exposure'])
print(f"Gini: {g:.3f}")

# Lorenz curve with optional plot
x, y = lorenz_curve(y_test['claim_count'], preds, exposure=y_test['exposure'],
                    plot=True)
```

An unweighted Gini on count data is almost meaningless when exposure varies across policies. The `exposure` argument is not optional in practice - it changes the result materially for any real motor or home dataset.

**Double-lift chart.** Policies sorted by predicted risk, grouped into equal-exposure deciles. Within each decile: mean actual, mean predicted, A/E ratio.

```python
from insurance_gam.ebm import double_lift

lift = double_lift(y_test['claim_count'], preds, exposure=y_test['exposure'],
                   n_bands=10)
print(lift)
# shape: (10, 5) — band, exposure, actual, predicted, ae_ratio
```

Monotone A/E deviation across bands indicates a calibration problem (model over- or under-dispersed). U-shaped deviation indicates missing interactions.

**Calibration table.** Actual vs expected by segment - the standard A/E table for presenting validation results to a pricing committee or a model validation function.

```python
from insurance_gam.ebm import calibration_table

# By risk band
tbl = calibration_table(
    y_test['claim_count'],
    preds,
    segment=X_test['risk_band'],
    exposure=y_test['exposure']
)
print(tbl)
# sorted by ae_ratio descending — worst-calibrated segments first
```

**Residual plots.** Deviance residuals by feature bin, useful for diagnosing systematic mis-estimation of features that are in the model.

```python
from insurance_gam.ebm import residual_plot

residual_plot(model, X_test, y_test['claim_count'],
              feature='driver_age', exposure=y_test['exposure'])
```

---

## Why EBMs rather than SHAP on a GBM?

We should be direct about what the EBM advantage actually is and what it is not.

The case is not primarily about regulatory compliance. The FCA confirmed in December 2025 that it will not introduce AI-specific rules for pricing, continuing to rely on Consumer Duty and the principle-based framework. Solvency II Article 121 and PRA SS3/18 (model risk management for Solvency II firms) require validators to understand model decision boundaries and test behaviour at margins - but neither specifies architectures.

The case is about the quality of evidence you can produce when asked to demonstrate that your model does what you think it does.

SHAP values on a GBM satisfy the Shapley axioms and are internally consistent. They tell you which features drove a particular prediction, averaged over all feature orderings. What they do not tell you is the model's shape function - what the GBM would predict if only this one feature changed, continuously, across its full range. The EBM shape function does tell you that, exactly, because the additive structure is the architecture. The shape function is not derived from the model. It is the model.

This distinction matters for three things. First, validation: an EBM model is easier to test at margins because the behaviour of any single factor in isolation is directly readable. Second, business sign-off: the pricing committee can review and challenge the shape functions in the same way they review and challenge GLM factors. Third, documentation: for model validation under Solvency II Article 121 and PRA SS3/18, the shape function table is a cleaner primary artefact than a SHAP summary.

The honest trade-off: a well-tuned GBM will outperform an EBM on most high-volume datasets with complex interactions. The EBM is not always the best-predicting model. It is the model with the best trade-off between predictive performance and intrinsic interpretability on typical UK personal lines data (10-30 features, hundreds of thousands of policies, Poisson or Tweedie target).

---

## What the library does not do

Three limitations worth stating plainly.

The EBM's shape functions are per-feature and pairwise. They cannot capture three-way or higher-order interactions without explicit specification. For telematics data or other high-dimensional signals where complex multiway interactions are the whole point, this is a real limitation.

Post-fit monotonicity editing via isotonic regression is not re-fitting. The edited model's shape function is monotone, but the model has not been re-trained with the constraint. If you need guaranteed monotonicity that is architecturally enforced rather than post-hoc corrected, set the `monotone_constraints` argument at fit time - interpretML's `ExplainableBoostingMachine` enforces monotonicity during the boosting process itself, and `InsuranceEBM` exposes this directly via its constructor. Post-hoc isotonic correction is a clean-up step, not a substitute for fit-time constraints on factors where monotonicity is a business requirement.

The library is v0.1. Edge cases in categorical handling with rare levels, and behaviour under extreme class imbalance, have not all been characterised. Inspect shape functions and calibration tables before putting any model in production.

---

## Getting started

```bash
uv add insurance-gam[interpret]
# or with all optional extras:
uv add insurance-gam[interpret,excel,glm]
```

Python 3.9 or later. The `[interpret]` extra installs interpretML >= 0.7.0, which provides the EBM engine. Polars >= 0.20, NumPy >= 1.21, Matplotlib >= 3.4, and scikit-learn >= 1.0 (for isotonic regression in `MonotonicityEditor`) are required in all variants.

The full workflow demo is in `notebooks/insurance_gam_demo.py`. It runs a Poisson frequency model on a synthetic motor dataset, produces relativity tables, plots the double-lift chart, and exports everything to Excel. On a standard laptop it completes in a few minutes.

Source: [github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam)
PyPI: [pypi.org/project/insurance-gam](https://pypi.org/project/insurance-gam)

The architecture builds on interpretML's `ExplainableBoostingMachine` (Nori et al., arXiv:1909.09223, 2019), which in turn builds on the GA2M framework of Lou, Caruana, Gehrke, and Hooker (KDD 2013). The actuarial workflow layer - relativities, diagnostics, GLM comparison - is specific to this library.

- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/)
- [Actuarial Neural Additive Models: Exact Interpretability with Tweedie Loss](/2026/03/13/your-interpretable-model-isnt-interpretable-enough/)
- [Calibration Testing That Goes Beyond the Residual Plot](/2026/03/09/insurance-calibration/)
- [Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/)
