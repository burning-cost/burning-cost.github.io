---
layout: post
title: "How to Convert a GBM to a Factor Table in Python"
date: 2029-07-15
author: Burning Cost
categories: [modelling, tutorials, libraries]
description: "GBMs outperform GLMs on most real motor books. But you cannot get a factor table out of them without extra work. Here is how to do it in Python using shap-relativities, including a runnable CatBoost example."
canonical_url: "https://burning-cost.github.io/2029/07/15/how-to-convert-gbm-to-factor-table-python/"
tags: [GBM, CatBoost, LightGBM, XGBoost, SHAP, factor-table, relativities, GLM, Emblem, Radar, pricing, actuarial, shap-relativities, python, uk-insurance, tutorial]
---

Every UK pricing team we have spoken to has the same problem. The GBM sits in a notebook outperforming the production GLM by 3–5 Gini points. The GLM goes to production anyway.

The blocker is not regulatory hostility to GBMs — it is the absence of a factor table. Your rating engine (Emblem, Radar, Earnix) expects a CSV with columns `factor`, `level`, `relativity`. The GBM produces a prediction function. Those two things are not the same, and the gap between them is where good models go to die.

This post shows how to close that gap in Python using [`shap-relativities`](https://github.com/burning-cost/shap-relativities). We will build a CatBoost frequency model on synthetic UK motor data, extract a multiplicative factor table using SHAP values, validate that the numbers are sensible, and export a CSV ready for rating engine import.

---

## Why factor tables still matter

The actuarial committee does not want to approve a prediction function. They want to see `Area F loads 1.59× vs Area A. NCD=5 discounts 43%. Conviction adds 68%.` That is a factor table, and it is the format in which pricing decisions get made, challenged, and documented. This is true regardless of what sits underneath.

Regulators have the same requirement. FCA market studies and PRA SS1/23 both contemplate scenarios where firms need to explain individual rating factors. "The GBM decided" is not an explanation. "The GBM, when decomposed by SHAP, attributes a 1.59× relativity to Area F versus the base" is.

The third pressure point is system integration. Emblem and Radar can import factor tables from CSV. They cannot import a pickled CatBoost model. If your book is priced via a rating engine rather than direct API scoring — and for most UK personal lines it still is — you need the factor table format.

---

## The R-only problem

The closest prior art in the actuarial world is the `insurancerating` package in R. It does this job well: exposure-weighted GLM relativities with credibility smoothing, one-way plots, significance tests. If your team lives in R, use it.

Python teams have nothing equivalent until now. The standard Python approach is to pull raw SHAP values and aggregate them by hand — sum SHAP values per level, exponentiate, divide by base level, cross fingers that you haven't made a broadcasting error. It works, but it is fragile, requires recoding for each project, and produces no confidence intervals.

`shap-relativities` standardises that workflow.

---

## Installation

```bash
uv add "shap-relativities[all]"
```

The `[all]` extra pulls in CatBoost, SHAP, pandas (as a bridge dependency for TreeExplainer), matplotlib, and polars. The library's output is Polars DataFrames throughout; pandas is only used internally.

---

## A runnable example

The library ships with a synthetic UK motor dataset so you can run this immediately without any proprietary data. 50,000 policies, known DGP, Poisson frequency with log-linear predictor. The true coefficients are exported via `TRUE_FREQ_PARAMS` so you can validate your extracted relativities against ground truth.

### Step 1: Load data and train the model

```python
import polars as pl
import catboost
from shap_relativities import SHAPRelativities
from shap_relativities.datasets.motor import load_motor, TRUE_FREQ_PARAMS

df = load_motor(n_policies=50_000, seed=42)

# Encode area as an integer for this example
df = df.with_columns([
    ((pl.col("conviction_points") > 0).cast(pl.Int32)).alias("has_convictions"),
    pl.col("area")
      .replace({"A": "0", "B": "1", "C": "2", "D": "3", "E": "4", "F": "5"})
      .cast(pl.Int32)
      .alias("area_code"),
])

features = ["area_code", "ncd_years", "has_convictions"]
X = df.select(features)

pool = catboost.Pool(
    data=X.to_pandas(),
    label=df["claim_count"].to_numpy(),
    weight=df["exposure"].to_numpy(),
)

model = catboost.CatBoostRegressor(
    loss_function="Poisson",
    iterations=300,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0,
)
model.fit(pool)
```

The `loss_function="Poisson"` is what makes the SHAP transformation work. CatBoost's Poisson objective uses a log link, so SHAP values are additive in log space and `exp(SHAP)` gives multiplicative contributions. If you use a linear objective, exponentiating SHAP values gives nonsense — the library will not stop you, so be deliberate about this.

### Step 2: Extract relativities

```python
sr = SHAPRelativities(
    model=model,
    X=X,
    exposure=df["exposure"],
    categorical_features=features,
)
sr.fit()

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
)

print(rels.select(["feature", "level", "relativity", "lower_ci", "upper_ci"]))
```

Output (50k policies, seed=42, run on Databricks serverless 2026-03-21):

```
shape: (14, 5)
┌─────────────────┬───────┬────────────┬──────────┬──────────┐
│ feature         ┆ level ┆ relativity ┆ lower_ci ┆ upper_ci │
├─────────────────┼───────┼────────────┼──────────┼──────────┤
│ area_code       ┆ 0     ┆ 1.000      ┆ 0.998    ┆ 1.002    │
│ area_code       ┆ 1     ┆ 1.110      ┆ 1.109    ┆ 1.111    │
│ area_code       ┆ 2     ┆ 1.149      ┆ 1.149    ┆ 1.150    │
│ area_code       ┆ 3     ┆ 1.269      ┆ 1.268    ┆ 1.269    │
│ area_code       ┆ 4     ┆ 1.588      ┆ 1.586    ┆ 1.589    │
│ …               ┆ …     ┆ …          ┆ …        ┆ …        │
│ ncd_years       ┆ 3     ┆ 0.641      ┆ 0.640    ┆ 0.642    │
│ ncd_years       ┆ 4     ┆ 0.542      ┆ 0.541    ┆ 0.543    │
│ ncd_years       ┆ 5     ┆ 0.435      ┆ 0.434    ┆ 0.436    │
│ has_convictions ┆ 0     ┆ 1.000      ┆ 1.000    ┆ 1.000    │
│ has_convictions ┆ 1     ┆ 1.681      ┆ 1.673    ┆ 1.689    │
└─────────────────┴───────┴────────────┴──────────┴──────────┘
```

This is a factor table. Area F loads 1.59× the base, NCD=5 discounts to 0.435, a conviction adds 68%. Those numbers can be written into an Emblem or Radar import template without further transformation.

One note on the `categorical_features` argument: this tells the library to aggregate SHAP values by discrete level for these features. It is not the same as CatBoost's `cat_features` training parameter. Here we are passing integer-encoded features to CatBoost without `cat_features=`, which means CatBoost treats them numerically. The `categorical_features` argument is purely an extraction hint — "group SHAP values by level for these features."

### Step 3: Validate before you trust the numbers

```python
checks = sr.validate()

print(checks["reconstruction"])
# CheckResult(passed=True, value=8.3e-06,
#   message='Max absolute reconstruction error: 8.3e-06.')

print(checks["sparse_levels"])
# CheckResult(passed=False, value=4.0,
#   message='4 factor level(s) have fewer than 30 observations. ...')
```

The reconstruction check is the one that matters. It verifies that `exp(SHAP values summed across features + expected value)` matches the model's own predictions to within 1e-4. If this fails, something is wrong with how the SHAP explainer was constructed — almost always a mismatch between model objective and SHAP output type. Fix it before proceeding.

The sparse levels check flags any level where the CLT confidence intervals are unreliable because there are fewer than 30 observations. The intervals are still calculated; treat them with caution.

### Step 4: Export to CSV for rating engine import

```python
rels.write_csv("relativities.csv")
```

The columns `feature`, `level`, `relativity`, `lower_ci`, `upper_ci` map directly to what a rating engine import template expects. Radar and Emblem both have CSV factor table import functionality. Check your platform's exact column naming requirements — but the structure is right.

---

## The maths in one paragraph

For a Poisson GBM with log link, TreeSHAP decomposes each prediction exactly:

```
log(mu_i) = E[log(mu)] + SHAP_area_i + SHAP_ncd_i + SHAP_convictions_i + ...
```

To get the multiplicative relativity for `area_code = 3` versus `area_code = 0`: take the exposure-weighted mean SHAP value across all policies with `area_code = 3`, do the same for `area_code = 0`, and exponentiate the difference. This is directly analogous to `exp(beta_3 - beta_0)` from a GLM. The confidence intervals use the CLT: `SE_k = shap_std_k / sqrt(n_k)`, where `n_k` is the count for that level. They quantify data uncertainty — how precisely we have estimated the mean SHAP contribution for that level — not model uncertainty from the GBM fitting process.

---

## Continuous features

Area band and conviction flag are discrete. Driver age and annual mileage are not. For continuous features, use `extract_continuous_curve()`:

```python
age_curve = sr.extract_continuous_curve(
    feature="driver_age",
    n_points=100,
    smooth_method="loess",
)
# Returns: feature_value, relativity, lower_ci, upper_ci
```

`smooth_method="isotonic"` enforces monotonicity — useful when you have a strong prior that the relationship is one-directional and you do not want a wiggly curve going to the actuarial committee.

---

## How does this compare to a GLM?

On a clean, correctly-specified log-linear DGP, the GLM has lower relativity error than SHAP. The benchmark numbers are:

| Approach | Mean error vs true DGP | Gini |
|---|---|---|
| Poisson GLM `exp(beta)` | 4.47% | 0.450 |
| shap-relativities | 9.44% | 0.479 |

The GLM's NCD=5 discount recovers 0.603 against a true value of 0.549 (+10%). SHAP gives 0.427 (−22%). On a clean multiplicative DGP, the GLM wins on relativity precision because its MLE is designed for exactly that functional form.

The GBM wins on Gini (+2.85pp) because it finds nonlinear patterns even in data generated by a linear model. On real portfolios with genuine interaction effects — high-group vehicles combined with limited NCD being materially worse than the main effects alone predict — the Gini gap widens to 3–8pp and the GLM's relativity precision advantage disappears because the model itself is misspecified.

Our view: if you can measure a Gini improvement of more than about 2pp on your hold-out and the worst-decile A/E improves materially, the GBM is the better model. SHAP-relativities makes it deployable. If the Gini improvement is marginal, the GLM's cleaner uncertainty quantification and easier auditability make it the honest choice.

---

## What this does not solve

**Correlated features.** If area band and socioeconomic index are correlated, SHAP attribution between them is not uniquely defined under the default `tree_path_dependent` perturbation. Use `feature_perturbation="interventional"` with a background dataset for more principled attribution — it is slower but correct.

**Interaction effects.** TreeSHAP allocates interaction effects back to individual features. If area and vehicle age interact in the model, the area relativity absorbs some of that interaction signal. The factor table is still deployable — it prices those segments correctly on average — but the individual main effects are not cleanly separated.

**Regulatory closed-form standard errors.** CLT confidence intervals from SHAP do not have the same statistical standing as GLM standard errors derived from the Fisher information matrix. For jurisdictions requiring closed-form SEs in rate filings, a GLM with documented MLE properties is still the right tool. SHAP CIs are useful for internal challenge; whether they satisfy a specific regulatory standard depends on your jurisdiction and your relationship with your supervisor.

**Adjustability.** A GLM factor table can be adjusted — you can load the conviction factor by 1.1×, recalculate the expected rate level change, and document the actuarial basis for doing so. Changing SHAP relativities does not change the model. If the pricing team expects to work with and adjust the relativities as a primary workflow, the GLM's structure is more tractable.

---

## A note on the convenience function

For one-off extractions, there is a shorter path that skips the class instantiation:

```python
from shap_relativities import extract_relativities

rels = extract_relativities(
    model=model,
    X=X,
    exposure=df["exposure"],
    categorical_features=["area_code", "ncd_years", "has_convictions"],
    base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
)
```

Use the class API when you need `.validate()`, `.plot_relativities()`, or `.extract_continuous_curve()`. Use the convenience function when you just want the DataFrame.

---

## Related tools

If you find that the GBM's advantage is driven by a specific interaction that the GLM is missing, [`insurance-interactions`](https://github.com/burning-cost/insurance-interactions) can detect and quantify those interactions automatically — useful for deciding whether to add them explicitly to the GLM rather than moving to a GBM.

If the Gini gap is not large enough to justify the GBM's complexity but you want to preserve some of its signal, [`insurance-distill`](https://github.com/burning-cost/insurance-distill) distils the GBM back into a deployable GLM via knowledge distillation. That is a different approach to the same deployment problem.

---

The library is at [`burning-cost/shap-relativities`](https://github.com/burning-cost/shap-relativities). A runnable Databricks notebook covering both the clean DGP and interaction scenarios is in [`burning-cost-examples`](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/shap_relativities_demo.py).
