---
layout: post
title: "EBMs for Insurance Tariff Construction: Boosting Accuracy with GLM Transparency"
date: 2026-03-13
categories: [libraries, pricing]
tags: [EBM, explainable-boosting-machine, interpretML, GLM, GAM, relativity-tables, monotonicity, gini, poisson, tweedie, gamma, insurance-gam, tariff, motor, python]
description: "Explainable Boosting Machines give you additive shape functions like a GAM but with GBM-level predictive accuracy. insurance-gam wraps interpretML's EBM in the workflow a UK pricing team actually uses: multiplicative relativity tables, monotonicity editing, and GLM comparison diagnostics."
---

The standard UK pricing workflow has a fault line running through it. The GLM goes to production because actuaries can read the factor tables, challenge the relativities, and explain them to a pricing committee or regulator. The GBM sits in a notebook because it is more accurate but produces nothing a pricing committee can review. The team loses the lift and carries on.

There is a third option that most UK pricing teams have not tried: Explainable Boosting Machines.

[`insurance-gam`](https://github.com/burning-cost/insurance-gam) wraps interpretML's `ExplainableBoostingRegressor` in the workflow a UK pricing team actually uses: exposure-aware fitting, multiplicative relativity tables normalised to the modal risk, post-fit monotonicity editing, and GLM comparison diagnostics. The model is transparent enough to review factor by factor, accurate enough to outperform a standard GLM on most datasets, and fast enough to fit on a laptop.

```bash
uv add insurance-gam
```

---

## What an EBM actually is

An EBM is a Generalised Additive Model trained by gradient boosting. The model structure is:

$$g(\mu_i) = \beta_0 + f_1(x_{i1}) + f_2(x_{i2}) + \ldots + f_p(x_{ip}) + \sum_{j<k} f_{jk}(x_{ij}, x_{ik})$$

where each $f_j$ is a learned shape function for a single feature, and the optional $f_{jk}$ terms are pairwise interactions. The key property is additivity: each feature contributes independently to the log score. That is exactly the same structure as a log-link GLM, except the shape functions are non-parametric curves rather than parametric effects fitted by IRLS.

The training algorithm is round-robin boosting. At each round, one feature is selected and a shallow tree is fitted to the current residuals for that feature alone, while all other features are held fixed. This cycling means each feature gets many opportunities to be refined, and the interactions between features are absorbed into the pairwise interaction terms rather than contaminating the main effects. The result is a set of per-feature shape functions that are more accurate than a GLM's linear or step-function terms but remain completely interpretable.

On the French CTP dataset (670,000 policies, Poisson frequency), an EBM fitted with `interactions='3x'` reduces deviance by around 8-12% relative to a standard GLM with manual interactions - comparable to a CatBoost model but with readable shape functions instead of SHAP approximations.

---

## What the wrapper adds

interpretML already handles Poisson/Tweedie/Gamma loss, exposure as an init_score offset, monotonicity constraints, and interaction detection. What it does not do is produce output in the format a UK pricing team expects from a tariff review.

`InsuranceEBM` adds the following:

**Exposure-aware fit and predict.** Exposure enters as `log(exposure)` added to the init_score, matching the GLM log-offset convention. `predict()` returns expected claim counts or amounts on the response scale - not log scores. `predict_log_score()` returns the additive representation if you need it for combining frequency and severity models.

**Relativity tables.** `RelativitiesTable` extracts per-feature shape functions and converts them to multiplicative relativities. The base level is the modal bin - the bin with the highest training weight. This matches GLM convention: the most common risk profile has relativity 1.0, and every other bin is expressed as a multiplier relative to that base.

**Summary with leverage.** `rt.summary()` returns all main-effect features ranked by range (max relativity / min relativity). This immediately shows you which factors are doing the most work in the tariff.

**Excel export.** `rt.export_excel('relativities.xlsx')` writes one sheet per feature in the format that works as a Radar or Emblem import template.

**Validation metrics.** Gini coefficient, Lorenz curve, double-lift chart, deviance, and A/E by segment - all exposure-weighted, all in the standard actuarial format.

---

## A frequency model from scratch

```python
import polars as pl
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable
from insurance_gam.ebm import gini, double_lift

# Fit a Poisson frequency model
model = InsuranceEBM(loss='poisson', interactions='3x')
model.fit(X_train, y_train['claim_count'], exposure=y_train['exposure'])

# Predict expected claim counts on holdout
preds = model.predict(X_test, exposure=y_test['exposure'])

# Standard actuarial validation
print(f"Gini: {gini(y_test['claim_count'], preds, exposure=y_test['exposure']):.3f}")
print(double_lift(y_test['claim_count'], preds, exposure=y_test['exposure']))

# Extract relativity tables
rt = RelativitiesTable(model)
print(rt.table('driver_age'))     # per-bin relativities, base = modal bin
print(rt.summary())               # all features ranked by leverage (range)
rt.export_excel('relativities.xlsx')
```

`rt.table('driver_age')` returns a Polars DataFrame with three columns: `bin_label`, `raw_score` (the EBM's additive log-score for that bin), and `relativity` (the exponentiated score relative to the base). A pricing actuary can read that table, challenge the shape against market intuition, and hand it to a rating engine without any further transformation.

---

## Monotonicity

A GLM constrains monotonicity by construction: you either include the rating factor as a linear term (forced monotone), or you use grouped levels (which you can hand-order). An EBM's non-parametric shape functions will sometimes produce non-monotone curves in the tails where data is thin. An age curve that shows slightly lower frequency at age 22 than 21 is probably noise.

There are two ways to handle this.

**At fit time**, pass `monotone_constraints` to `InsuranceEBM`:

```python
model = InsuranceEBM(
    loss='poisson',
    monotone_constraints={'ncd': -1, 'vehicle_age': -1}  # more NCD / older car = lower rate
)
```

The constraint is enforced by the EBM's internal boosting algorithm. The shape function will be monotone by construction, but the model may sacrifice some predictive accuracy to achieve this.

**Post-fit**, use `MonotonicityEditor`:

```python
from insurance_gam.ebm import MonotonicityEditor

me = MonotonicityEditor(model)
scores_before = me.get_scores('ncd')
me.enforce('ncd', direction='decrease')
me.plot_before_after('ncd', scores_before=scores_before)
```

Post-fit enforcement applies isotonic regression to the stored shape function scores. It modifies the scores in-place without re-fitting the boosting trees. This is a soft edit - use it to clean up noise in the tails, not to override a systematic model signal. If the shape function is strongly non-monotone across the full range of a feature, that is information: either the feature has a genuine non-linear effect, or there is a confound worth investigating before you smooth it away. Document any post-fit edits in your assumptions log.

---

## Comparing against an existing GLM

When running EBMs alongside an existing GLM - either as a challenger or as a migration check - `GLMComparison` plots the EBM shape functions against your GLM's factor table:

```python
from insurance_gam.ebm import GLMComparison
import polars as pl

glm_rel = pl.DataFrame({
    'level': ['G1', 'G2', 'G3', 'G4', 'G5'],
    'relativity': [1.0, 1.05, 1.12, 1.22, 1.35]
})

cmp = GLMComparison(model)
cmp.plot_comparison('vehicle_group', glm_relativities=glm_rel)

# Which features diverge most?
by_feature = {feat: glm_rel_by_feature[feat] for feat in model.feature_names}
print(cmp.divergence_summary(glm_relativities_by_feature=by_feature))
```

`divergence_summary()` returns a ranked table of features by how much the EBM shape function diverges from the GLM. Features at the top of that table are where the EBM is finding structure the GLM misses - these are the candidates for investigation. Is the divergence because the GLM's linear approximation is too crude? Because the EBM is picking up a confound? Or because the data in that region is thin and the EBM is over-fitting?

---

## When to use EBMs vs GLMs vs GBMs

**Use a GLM** when you need a deterministic, fully auditable tariff where every factor is constrained by actuarial judgment, when you are operating in a heavily regulated environment where the regulator expects a parameter table with standard errors, or when the portfolio is small enough that a non-parametric model will overfit.

**Use a GBM** (CatBoost, XGBoost) when pure predictive accuracy is the objective and interpretability is not a hard constraint. Accept that you will need a distillation step - [insurance-distill](https://github.com/burning-cost/insurance-distill) or SHAP-based relativity extraction - before the model can go into a rating engine.

**Use an EBM** when you want the accuracy gain of a boosted model and the transparency of a factor table without a distillation step. EBMs are the natural choice when you are building a new pricing model from scratch, have the time to review shape functions feature by feature, and want to take the output directly to a rating engine. They fit faster than CatBoost on the same dataset, produce directly readable factor tables, and handle monotonicity constraints natively.

The one place EBMs are weaker than GBMs is high-cardinality categoricals. A GBM with target encoding handles `vehicle_make_model` with 3,000 levels naturally. An EBM's binning strategy will struggle without pre-processing. For those features you either encode externally, use entity embeddings via [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools), or drop to SHAP extraction from a GBM.

---

## A note on Tweedie pure premium models

The `loss='tweedie'` option fits a pure premium model directly. Set `variance_power=1.5` for compound Poisson-Gamma (the standard actuarial choice for pure premium with a mass at zero). The EBM will fit a single model to the combined frequency-severity signal rather than requiring separate frequency and severity models.

```python
model = InsuranceEBM(loss='tweedie', variance_power=1.5, interactions='3x')
model.fit(X_train, y_train['pure_premium'], exposure=y_train['exposure'])
```

Whether this beats a frequency x severity approach depends on your data. The separate approach captures different patterns in claims incidence versus claims cost and is easier to explain ("this factor affects how often you claim; that factor affects how much it costs"). The Tweedie approach is simpler and can work well when the data is not rich enough to fit reliable severity models separately.

---

## Installation

```bash
uv add insurance-gam
# With Excel export:
uv add "insurance-gam[excel]"
# With statsmodels GLM integration for GLMComparison:
uv add "insurance-gam[glm]"
```

Python 3.10+. Requires `interpret >= 0.7.0`, `polars >= 0.20`, `numpy >= 1.21`. A full workflow notebook is in `notebooks/insurance_gam_demo.py`. Source at [github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam).

---

**See also:**
- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/) - if you need relativity tables from an existing CatBoost model rather than switching to EBMs
- [Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/) - detecting interaction candidates before deciding whether an EBM's automatic interaction detection is sufficient
- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/) - the distillation route if you need a GBM to sit behind a GLM-format rating engine
