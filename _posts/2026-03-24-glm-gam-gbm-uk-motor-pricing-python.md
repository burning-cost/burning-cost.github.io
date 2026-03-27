---
layout: post
title: "GLM, GAM, and GBM for UK Motor Pricing in Python"
date: 2026-03-24
author: Burning Cost
categories: [pricing]
tags: [glm, gam, gbm, catboost, poisson, motor-pricing, python, uk-insurance, freMTPL2, exposure, NCD, interpretability, shap-relativities, insurance-gam, insurance-distill, insurance-cv, insurance-monitoring, insurance-fairness, gini, statsmodels, EBM]
description: "The Python equivalent of the IFoA MLR Working Party's R tutorial: Poisson GLM baseline, EBM GAM, and CatBoost GBM on UK motor data, with the full pipeline from data to governance."
---

In July 2021, the IFoA Machine Learning in Reserving Working Party published a tutorial covering GLM, GAM, and XGBoost for general insurance pricing. It remains the most-cited practical ML reference in UK actuarial pricing. It is entirely R-based.

This is the Python equivalent. The conceptual framework is the same: a GLM as the interpretable baseline, a GAM to handle nonlinearity without surrendering interpretability, and a gradient boosted model to find the performance ceiling. What changes is the tooling: statsmodels, `insurance-gam`, CatBoost, and the Burning Cost libraries that handle the pipeline from model output to production factor tables.

This post is also a hub. Each section links to a deeper post or library for readers who want the full treatment. If you are working through the GLM-to-GBM progression for the first time in Python, reading this post in order will give you the map.

---

## The dataset

We use `freMTPL2freq` throughout. This is the standard French motor third-party liability benchmark: 678,013 policies with claim counts, exposure periods, and seven rating factors (driver age, vehicle age, vehicle power, vehicle brand, fuel type, region, and density). The modelling workflow is identical to a UK personal lines motor book; what differs is the regulatory context and the UK-specific conventions we discuss below.

```python
import polars as pl
import numpy as np
from sklearn.datasets import fetch_openml

raw = fetch_openml("freMTPL2freq", version=3, as_frame=True, parser="auto")
df_pd = raw.data.copy()
df_pd["ClaimNb"] = raw.target

df = (
    pl.from_pandas(df_pd)
    .with_columns([
        pl.col("Exposure").clip(0.0, 1.0).alias("exposure"),
        pl.col("ClaimNb").cast(pl.Int32).alias("claim_count"),
        pl.col("VehPower").cast(pl.Int32).alias("veh_power"),
        pl.col("DrivAge").cast(pl.Float64).alias("driver_age"),
        pl.col("VehAge").cast(pl.Float64).alias("veh_age"),
        pl.col("BonusMalus").cast(pl.Float64).alias("bonus_malus"),
        pl.col("Density").log().alias("log_density"),
    ])
)

train = df[:500_000]
test  = df[500_000:]
```

A few things to note before the modelling starts.

**Exposure** is the fraction of a year during which the policy was at risk. The freMTPL2 field is already in car-years; cap it at 1.0 (a handful of records have small overruns from mid-term adjustment artefacts). In a UK book you would compute `(min(expiry, year_end) - max(inception, year_start)).days / 365.25` and apply the same cap.

**Claim frequency versus severity** are modelled separately. `freMTPL2freq` gives you claim counts; `freMTPL2sev` gives you individual claim amounts. This post covers frequency modelling throughout. A pure premium model multiplies frequency by expected severity; we use frequency here because the GLM-GAM-GBM comparison is cleanest in that setting.

**NCD** does not appear explicitly in freMTPL2 but `BonusMalus` is its French equivalent: 50 is the base, values below 50 are discounts for claim-free years, values above 50 are maluses. We use it as a continuous feature in the GLM but pay attention to where the GAM and GBM see nonlinearity in it.

---

## GLM baseline

The Poisson GLM with log link and log-exposure offset is the industry standard for claim frequency modelling. In UK pricing it is typically fitted in Emblem; here we use statsmodels directly.

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

# Log-exposure offset: enters the linear predictor directly
log_exposure = np.log(train["exposure"].to_numpy())

glm = smf.glm(
    formula=(
        "claim_count ~ driver_age + veh_age + veh_power"
        " + bonus_malus + log_density"
        " + C(Region) + C(VehBrand) + C(VehGas)"
    ),
    data=train.to_pandas(),
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=log_exposure,
).fit()

print(glm.summary())
```

Three things that matter in UK practice and are often wrong in generic tutorials.

**Exposure as offset, not weight.** The model specification is `log(E[claims]) = log(exposure) + Xb`. Passing exposure as a `freq_weights` or `var_weights` argument changes the deviance calculation and produces wrong coefficient estimates. `offset=` is the correct parameter. See [GLMs for UK Insurance Pricing in Python](/2026/03/22/glm-insurance-python-uk-pricing-actuary-guide/) for a full treatment of this distinction.

**NCD as a factor, not a linear term.** `BonusMalus` runs from 50 down to below 50 for good drivers and up to 350 for repeat claimers. Treating it as a linear continuous variable forces a single slope across the entire range and will miss the convex discount structure. In UK motor the standard treatment is to band NCD years (0, 1, 2, 3, 4, 5+) and treat each band as a separate factor level. With freMTPL2 you can do the same by binning `BonusMalus` before passing it to `C()`.

**Driver age banding.** `DrivAge` entered linearly cannot represent the U-shaped risk curve, with young drivers (<25) and elderly drivers (>70) both carrying higher frequency. The GLM approximation requires you to specify this structure explicitly before fitting, typically with a polynomial term or age bands. We do this:

```python
train = train.with_columns(
    pl.col("driver_age").cut(
        [25, 30, 40, 50, 60, 70],
        labels=["<25", "25-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    ).alias("age_band")
)

glm_banded = smf.glm(
    formula=(
        "claim_count ~ C(age_band) + veh_age + veh_power"
        " + bonus_malus + log_density"
        " + C(Region) + C(VehBrand) + C(VehGas)"
    ),
    data=train.to_pandas(),
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=log_exposure,
).fit()
```

To extract the age-band factor table:

```python
params = glm_banded.params
age_params = {k: v for k, v in params.items() if "age_band" in k}

age_table = pl.DataFrame({
    "band": list(age_params.keys()),
    "log_coeff": list(age_params.values()),
}).with_columns(
    pl.col("log_coeff").exp().alias("relativity")
)

print(age_table)
```

**Gini on holdout.** The Gini coefficient measures risk discrimination: how well the model separates high-risk from low-risk policies. Compute it on the holdout, not the training set.

```python
from sklearn.metrics import roc_auc_score

log_exposure_test = np.log(test["exposure"].to_numpy())
test_pd = test.to_pandas()
test_pd["log_exposure_offset"] = log_exposure_test

pred_glm = glm_banded.predict(test_pd, offset=test_pd["log_exposure_offset"])

# Gini = 2 * AUC - 1 (for a binary event, using claim indicator)
claim_indicator = (test["claim_count"] > 0).to_numpy().astype(int)
gini_glm = 2 * roc_auc_score(claim_indicator, pred_glm) - 1
print(f"GLM Gini: {gini_glm:.3f}")
```

On freMTPL2 a reasonably specified Poisson GLM with age banding and regional effects lands between 0.28 and 0.34 Gini depending on feature engineering. This is the baseline the GAM and GBM compete against.

---

## GAM: nonlinearity without black boxes

The main limitation of the GLM is that it requires you to specify nonlinear structure before fitting: you choose the age bands, you decide whether to include a polynomial term for vehicle age. When you guess wrong, you leave Gini on the table. When you do not know the right shape, you cannot know whether you have guessed wrong.

GAMs model the relationship between each feature and the target as a smooth function, estimated from the data. The output remains additive: the log-predicted frequency is a sum of per-feature shape functions, which makes it directly interpretable by a pricing actuary. There is no black box. The GAM's shape function for driver age is the U-shaped curve you would have needed to specify manually in the GLM.

For production pricing use, we prefer EBM (Explainable Boosting Machine) via [`insurance-gam`](https://github.com/burning-cost/insurance-gam). EBM builds the shape functions through boosting rather than spline regression, which makes it robust to the scale of features and gives it better performance than classical P-splines on large datasets.

```bash
uv add "insurance-gam[ebm]"
```

```python
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

features = ["driver_age", "veh_age", "veh_power", "bonus_malus", "log_density"]

X_train = train.select(features)
X_test  = test.select(features)
y_train = train["claim_count"].to_numpy()
exp_train = train["exposure"].to_numpy()

model_ebm = InsuranceEBM(loss="poisson", interactions="3x")
model_ebm.fit(X_train, y_train, exposure=exp_train)
```

The `interactions="3x"` setting tells EBM to consider pairwise interactions automatically. It will find the combinations that carry genuine signal; unimportant interactions end up with negligible contribution and can be filtered.

After fitting, inspect the shape functions directly:

```python
rt = RelativitiesTable(model_ebm)

# Driver age shape function: the U-shape the GLM had to assume
print(rt.table("driver_age"))

# BonusMalus: should show convex discount, steeper at low values
print(rt.table("bonus_malus"))

# Full summary of all features
print(rt.summary())
```

The `RelativitiesTable` output is a Polars DataFrame with `shape_value` and `relativity` columns. The relativity at each point is `exp(shape function value)`, normalised to 1.0 at the reference point. This is directly comparable to the factor table from the GLM.

**What EBM finds that the GLM missed.** Two features show clear nonlinearity in freMTPL2:

- Driver age. EBM recovers a U-shaped curve: claim frequency is highest below age 25, drops through the 30s and 40s, then rises again above 70. A linear term misses both tails; age banding approximates the shape but the actuary must choose cut-points manually.
- BonusMalus. The discount structure is convex, not linear. EBM finds the curve without any feature engineering. A linear term understates the discount for long-claim-free drivers.

**Gini comparison.** On freMTPL2, EBM with `interactions="3x"` typically produces 5-15 Gini points above the banded GLM. The exact gap depends on how thoroughly the GLM's feature engineering captured the true structure. Against a naively specified GLM, the gain is at the upper end of that range; against a carefully tuned GLM with extensive age banding and NCD transformations, it narrows.

The [insurance-gam benchmarks](https://github.com/burning-cost/insurance-gam/blob/main/benchmarks/) show EBM at +5-15pp Gini versus a linear+quadratic GLM on a 10,000-policy synthetic DGP. See [Your Model Is Either Interpretable or Accurate. insurance-gam Refuses That Trade-Off.](/2026/03/14/insurance-gam-interpretable-nonlinearity/) for the full treatment.

**Known limitation.** EBM's exposure calibration via `init_score` can inflate absolute Poisson deviance figures on some DGPs without affecting risk ordering. Use Gini as your primary comparison metric and validate calibration separately using a double-lift chart before loading EBM factors into a production tariff.

---

## GBM: CatBoost frequency model

The gradient boosted model is the performance ceiling. It makes no assumptions about the functional form of any feature relationship, finds interactions automatically, and on most insurance datasets it outperforms both the GLM and the GAM by a measurable margin in Gini.

The cost is interpretability. A CatBoost model is not a set of factor tables. It is a forest of trees. A pricing committee cannot review it in the same way they review a GLM coefficient. This is a real constraint in UK personal lines, where a model that cannot be explained to an FCA reviewer or challenged by an underwriter is difficult to deploy in production.

We use CatBoost with a Poisson objective. Exposure enters as a log-offset via the `baseline` parameter in `Pool`, which is the correct treatment for a Poisson count model: the baseline is added to the model's output before the loss is computed, exactly mirroring the log-exposure offset in statsmodels.

```python
import catboost

feature_cols = ["driver_age", "veh_age", "veh_power", "bonus_malus", "log_density"]
cat_cols     = ["VehBrand", "VehGas", "Region"]
all_cols     = feature_cols + cat_cols

X_train_cb = train.select(all_cols).to_pandas()
X_test_cb  = test.select(all_cols).to_pandas()

log_exp_train = np.log(train["exposure"].to_numpy())
log_exp_test  = np.log(test["exposure"].to_numpy())

pool_train = catboost.Pool(
    data=X_train_cb,
    label=train["claim_count"].to_numpy(),
    baseline=log_exp_train,    # log-exposure offset: added to output before Poisson loss
    cat_features=cat_cols,
)

pool_test = catboost.Pool(
    data=X_test_cb,
    label=test["claim_count"].to_numpy(),
    baseline=log_exp_test,
    cat_features=cat_cols,
)

cb_model = catboost.CatBoostRegressor(
    loss_function="Poisson",
    iterations=500,
    learning_rate=0.03,    # low learning rate: better generalisation for insurance
    depth=5,               # shallow trees: reduces variance, more interpretable features
    l2_leaf_reg=10,        # regularisation: important when some factor levels are sparse
    eval_metric="Poisson", # monitor holdout deviance during training
    early_stopping_rounds=50,
    random_seed=42,
    verbose=50,
)

cb_model.fit(pool_train, eval_set=pool_test)
```

Two hyperparameter choices matter most for insurance:

**Low learning rate.** Insurance datasets have relatively low signal-to-noise. A high learning rate (0.1+) causes the model to fit noise in the training year. We use 0.03; some teams go lower still (0.01) with more iterations. CatBoost's built-in early stopping on holdout deviance handles the tradeoff automatically.

**Shallow depth.** Trees of depth 5-6 are interpretable enough for SHAP analysis and regularised enough to generalise. Depth 8+ tends to overfit on the kind of thin segments (rare vehicle brands, low-density regions) that appear in every insurance portfolio.

**SHAP relativities.** The GBM predicts well but does not produce factor tables. The [`shap-relativities`](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/) library bridges this gap by computing SHAP values in log space and aggregating them by factor level to produce multiplicative relativities.

```bash
uv add "shap-relativities[all]"
```

```python
from shap_relativities import SHAPRelativities

X_test_pl = test.select(all_cols)

sr = SHAPRelativities(
    model=cb_model,
    X=X_test_pl,
    exposure=test["exposure"],
    categorical_features=cat_cols,
)
sr.fit()

# Validate that SHAP values reconstruct the model predictions
checks = sr.validate()
print(checks["reconstruction"])

# Per-level relativities for categorical features
region_rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"Region": "R11"},
)

# Continuous age relativity curve: finds the U-shape without binning
age_curve = sr.extract_continuous_curve(
    feature="driver_age",
    n_points=100,
    smooth_method="loess",
)
print(age_curve.head(10))
```

The `validate()` step is not optional. If the reconstruction check fails, the SHAP values are not correctly decomposing the model's predictions: usually a sign the explainer was built against the wrong model objective. Fix this before using the relativities for anything.

**Gini comparison.** CatBoost with native categoricals and early stopping typically lands 3-8 Gini points above the GLM on freMTPL2, and somewhat higher above a naively specified GLM. The shap-relativities benchmarks (measured on Databricks, 2026-03-21, 25,000 synthetic policies) show CatBoost Gini of approximately 0.411 versus 0.393 for a linear GLM. The magnitude depends on how well the GLM was specified.

**The governance question.** A 5-point Gini improvement is worth roughly 5% improvement in risk discrimination. For a book writing £200m GWP, that is a meaningful number. The additional governance burden is real: sign-off from a pricing committee, explanation to the FCA, documentation under PRA SS1/23. Whether that burden is worth carrying depends on the book and the team. Our view is that the GBM should be in the arsenal but should go to production via the distillation path described in the next section, not as a raw model in a rating engine. Once deployed, [conformal prediction intervals](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) wrap the model output to provide per-risk coverage guarantees without parametric distribution assumptions.

---

## Bringing it together

The GLM, GAM, and GBM are not three separate approaches to choose between. They are three stages of the same pipeline.

**Summary comparison**

| Model | Gini vs GLM baseline | Interpretability | Radar/Emblem compatible | Governance burden |
|-------|----------------------|------------------|------------------------|-------------------|
| Poisson GLM (banded) | Baseline | Full factor tables | Direct | Standard |
| EBM (insurance-gam) | +5-15pp | Shape functions | Via distillation | Moderate |
| CatBoost + SHAP | +3-8pp | SHAP relativities | Via distillation | Higher |

The Gini ranges assume freMTPL2-scale data with a non-linear DGP. On a book where the true DGP is already well-approximated by a linear GLM, the gaps narrow. Conversely, on a book with confirmed interactions (young drivers in dense urban areas, for instance), the GBM advantage grows.

**The distillation path.** CatBoost cannot be loaded into Radar or Emblem. The [`insurance-distill`](https://github.com/burning-cost/insurance-distill) library handles this by fitting a surrogate GLM on the GBM's predictions. The surrogate learns from CatBoost's denoised output rather than individual claim counts, which means it typically retains 90-97% of the GBM's Gini while producing standard multiplicative factor tables.

```bash
uv add "insurance-distill[catboost]"
```

```python
from insurance_distill import SurrogateGLM

surrogate = SurrogateGLM(
    model=cb_model,
    X_train=train.select(all_cols),
    y_train=train["claim_count"].to_numpy(),
    exposure=train["exposure"].to_numpy(),
    family="poisson",
)

surrogate.fit(
    max_bins=10,
    method_overrides={
        "bonus_malus": "isotonic",  # enforce monotone NCD discount
    },
)

report = surrogate.report()
print(report.metrics.summary())
# Gini (GBM):              ~0.41
# Gini (GLM surrogate):    ~0.39
# Gini ratio:              ~95%
# Max segment deviation:   ~8%

surrogate.export_csv("output/factors/", prefix="motor_freq_")
```

The surrogate's factor tables export directly to CSV in the format Radar and Emblem expect. See [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/) for the full distillation workflow including rating engine upload.

**Walk-forward CV.** Standard k-fold cross-validation is misleading on insurance data because it allows future information to leak into training. [`insurance-cv`](https://github.com/burning-cost/insurance-cv) implements walk-forward splits that respect accident year and IBNR development structure. On a 20,000-policy synthetic motor book with a +20%/year claims trend, random k-fold overestimates model quality by 13.2% versus the true prospective holdout; walk-forward overestimates by 6.3%.

```python
from insurance_cv import walk_forward_split
from insurance_cv.diagnostics import temporal_leakage_check

splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=12,
    test_months=6,
    step_months=6,
    ibnr_buffer_months=3,
)

check = temporal_leakage_check(splits, df, date_col="inception_date")
if check["errors"]:
    raise RuntimeError("\n".join(check["errors"]))
```

See [Walk-Forward Cross-Validation for Insurance GLMs in Python](/2026/03/24/walk-forward-cross-validation-insurance-glm-python/) for the full treatment.

**Production monitoring.** A model fitted in March 2026 is not the right model for October 2027. The portfolio ages, claims inflation shifts the severity distribution, new vehicle types enter the book. [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) tracks three failure modes separately: covariate shift (PSI per feature), calibration drift (A/E ratios with Poisson confidence intervals), and discrimination decay (Gini drift z-test from arXiv:2510.04556). The Gini drift test is the important one: it tells you whether to run a cheap recalibration or a full refit. An aggregate A/E dashboard cannot answer that question. See [Motor Model Mispricing Caught by Monitoring](/2026/03/23/motor-model-mispricing-caught-by-monitoring/) for a walkthrough.

**Proxy discrimination check.** Before finalising any factor table, run [`insurance-fairness`](/2026/03/20/fca-consumer-duty-pricing-fairness-python/). The FCA's Consumer Duty (PS22/9) requires firms to demonstrate fair value, and the FCA's TR24/2 thematic review (August 2024) found most insurers' Fair Value Assessments were insufficiently substantiated. Postcode is the factor most likely to be a proxy: Citizens Advice (2022) estimated a £280/year ethnicity penalty in UK motor insurance driven by postcode rating. `insurance-fairness` computes proxy R-squared, mutual information scores, and SHAP-linked price impact, and generates a Markdown audit report mapped to PRIN 2A and the Equality Act s.19.

---

## What to read next

These are not competing models. They are stages of a pipeline, and each stage has a dedicated library:

**Building the models**
- [GLMs for UK Insurance Pricing in Python](/2026/03/22/glm-insurance-python-uk-pricing-actuary-guide/): exposure as offset, NCD as factor, MTA handling, glum vs statsmodels
- [insurance-gam: EBM, ANAM, and PIN](/2026/03/14/insurance-gam-interpretable-nonlinearity/): when to use each interpretable architecture
- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/): the maths, the validation, the limitations for presenting to regulators

**From model to production**
- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/): the full distillation workflow
- [Walk-Forward Cross-Validation for Insurance GLMs in Python](/2026/03/24/walk-forward-cross-validation-insurance-glm-python/): why k-fold is lying to you, and what to use instead
- [Motor Model Mispricing Caught by Monitoring](/2026/03/23/motor-model-mispricing-caught-by-monitoring/): what systematic monitoring catches before the loss ratio does

**Governance**
- [insurance-fairness](https://github.com/burning-cost/insurance-fairness): FCA Consumer Duty compliance, proxy discrimination audit
- [insurance-governance](/2026/03/14/insurance-governance-unified-pra-ss123-validation/): PRA SS1/23 model validation reports

The IFoA tutorial established the framework in R. This post gives you the same framework in Python, with the libraries UK pricing teams are actually adopting. The pipeline runs from `fetch_openml` to Radar-compatible factor tables in well under 200 lines of code. The governance libraries add the audit trail that regulators require. None of this requires bespoke infrastructure: all libraries install via `uv add`, run on a laptop for development, and scale to Databricks for production.

---

*All benchmark numbers cited are from published library documentation measured on Databricks serverless compute. freMTPL2 results will differ from synthetic DGP results in the library benchmarks because the true DGP is unknown; treat the freMTPL2 Gini figures as indicative of relative improvement rather than absolute performance.*
