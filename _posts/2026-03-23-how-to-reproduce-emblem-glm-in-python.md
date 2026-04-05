---
layout: post
title: "How to Reproduce an Emblem GLM in Python"
date: 2026-03-23
author: Burning Cost
categories: [python, glm, insurance-pricing, tutorials]
description: "Reproduce an Emblem frequency-severity GLM in Python: factor tables, one-way plots, deviance residuals, and lift charts using statsmodels, CatBoost, and Polars."
tags: [glm, emblem, statsmodels, catboost, polars, frequency-severity, factor-tables, relativities, deviance-residuals, lift-chart, one-way-analysis, python, uk-insurance, motor-pricing, pricing-actuary, shap-relativities, insurance-cv]
---

This is the final post in our five-part "How to do X in Python" series for pricing actuaries. The other four cover the [exposure-weighted Gini coefficient](/2026/03/23/exposure-weighted-gini-coefficient-python/), [one-way analysis](/2026/03/23/one-way-analysis-python-pricing-actuary/), [factor tables to Excel](/2026/03/24/how-to-export-factor-table-to-excel-python/), and [double-lift charts](/2026/03/22/insurance-model-monitoring-gini-ae-double-lift-python/).

If you have spent any time in Emblem, you know the workflow: load data, define the model structure, fit Poisson and Gamma GLMs, inspect factor tables, run one-way diagnostics, check deviance residuals, produce a lift chart, iterate. This post reproduces that workflow end-to-end in Python using statsmodels for the GLMs, CatBoost as a GBM challenger, and Polars for data handling.

We are not claiming Python replaces Emblem's GUI or its integrated rating engine connectivity. The claim is narrower: every modelling diagnostic an Emblem user looks at during model development can be produced in Python with equivalent rigour, and in many cases with more flexibility.

---

## The data

We generate a synthetic UK motor portfolio: 60,000 policies, frequency and severity components with a known data-generating process. The true model has multiplicative structure, which is what a GLM with a log link recovers.

```python
import numpy as np
import polars as pl
from scipy.stats import gamma as gamma_dist

rng = np.random.default_rng(2024)
n = 60_000

# --- Rating factors ---
ncd_band    = rng.integers(0, 6, n)          # 0 = no NCD, 5 = max discount
veh_group   = rng.integers(1, 11, n)         # ABI groups 1-10
driver_age  = rng.integers(17, 80, n)        # years
area        = rng.choice(list("ABCDEF"), n)  # postcode area proxy
exposure    = np.where(rng.random(n) < 0.12,
                       rng.uniform(0.08, 0.99, n),
                       1.0)                  # ~12% short-term policies

# --- True relativities (multiplicative, log-additive) ---
ncd_effect  = np.array([1.40, 1.20, 1.05, 0.95, 0.85, 0.75])[ncd_band]
veh_effect  = 0.70 + 0.06 * veh_group       # linear-ish in group
age_effect  = np.where(
    driver_age < 25, 1.80,
    np.where(driver_age < 35, 1.20,
    np.where(driver_age > 65, 1.15, 1.00))
)
area_lkp    = {"A": 1.00, "B": 1.08, "C": 1.18,
               "D": 0.92, "E": 0.85, "F": 1.25}
area_effect = np.array([area_lkp[a] for a in area])

base_freq   = 0.07   # 7% base claim frequency
true_freq   = base_freq * ncd_effect * veh_effect * age_effect * area_effect

# --- Simulate claims ---
claim_count = rng.poisson(true_freq * exposure)

# --- Severity: Gamma, mean varies by vehicle group (£1,700 to £3,500) ---
true_sev = 1_500 + 200 * veh_group
severity = np.where(
    claim_count > 0,
    gamma_dist.rvs(a=4.0, scale=true_sev / 4.0, size=n, random_state=rng),
    0.0
)

df = pl.DataFrame({
    "ncd_band":    ncd_band,
    "veh_group":   veh_group,
    "driver_age":  driver_age,
    "area":        area,
    "exposure":    exposure,
    "claim_count": claim_count,
    "severity":    severity,
})
```

We include ~12% short-term policies because real books have them, and they matter for the [exposure-weighted Gini](/2026/03/23/exposure-weighted-gini-coefficient-python/) calculation. Severity is simulated at policy level for simplicity; in production you would work with a claim-level severity dataset.

---

## Fitting the frequency GLM

Emblem's default for claim frequency is Poisson with a log link and exposure as an offset. statsmodels provides the same specification:

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import numpy as np

# statsmodels works with pandas; bridge from Polars
df_pd = df.to_pandas()
df_pd["log_exposure"] = np.log(df_pd["exposure"])

# Treat ncd_band and veh_group as categorical factors
# In Emblem these would be defined as discrete in the factor definition
df_pd["ncd_f"] = df_pd["ncd_band"].astype(str)
df_pd["veh_f"] = df_pd["veh_group"].astype(str)
df_pd["area"]  = df_pd["area"].astype(str)

# Bin driver age into bands (Emblem does this in the factor definition screen)
df_pd["age_band"] = pd.cut(
    df_pd["driver_age"],
    bins=[16, 24, 34, 44, 54, 64, 80],
    labels=["17-24", "25-34", "35-44", "45-54", "55-64", "65-80"]
).astype(str)

# Set reference levels to match Emblem convention
# NCD 3 is a common base; area D is close-to-average for most books
freq_formula = (
    "claim_count ~ "
    "C(ncd_f, Treatment(reference='3')) + "
    "C(veh_f, Treatment(reference='5')) + "
    "C(age_band, Treatment(reference='35-44')) + "
    "C(area, Treatment(reference='D'))"
)

freq_model = smf.glm(
    formula = freq_formula,
    data    = df_pd,
    family  = sm.families.Poisson(link=sm.families.links.Log()),
    offset  = df_pd["log_exposure"],
).fit()

print(freq_model.summary2().tables[1].round(4))
```

The `offset` argument handles exposure correctly. The common mistake is passing exposure as `freq_weights` or `var_weights`, which reweights deviance contributions rather than entering the linear predictor directly. This gives subtly biased estimates whenever exposure correlates with your rating factors -- and in UK motor, it always does. NCD-5 policyholders are systematically older, more experienced, and more often annual policyholders than NCD-0 drivers.

Reference level choice does not affect fit quality. It only changes how the coefficients are expressed. Picking a base level that anchors near the middle of the book makes the relativities easier to review in a governance pack.

---

## Fitting the severity GLM

For severity, the standard actuarial approach is Gamma with a log link, fitted on policies with at least one claim:

```python
claims_pd = df_pd[df_pd["claim_count"] > 0].copy()
claims_pd["avg_sev"] = claims_pd["severity"] / claims_pd["claim_count"]

sev_formula = (
    "avg_sev ~ "
    "C(veh_f, Treatment(reference='5')) + "
    "C(age_band, Treatment(reference='35-44')) + "
    "C(area, Treatment(reference='D'))"
)

sev_model = smf.glm(
    formula      = sev_formula,
    data         = claims_pd,
    family       = sm.families.Gamma(link=sm.families.links.Log()),
    freq_weights = claims_pd["claim_count"],  # weight by claim count
).fit()

print(sev_model.summary2().tables[1].round(4))
```

NCD is absent from the severity model because NCD is a selection mechanism, not a causal driver of repair costs. Vehicle group goes into both models because it predicts both who claims and how much the repair costs. Whether `area` belongs in severity is a modelling judgement -- we include it because parts and labour costs do vary by region.

---

## Extracting factor tables (relativities)

Emblem's factor table view shows each level alongside its fitted relativity relative to the base. These are multiplicative relativities: `exp(coefficient)`.

```python
import polars as pl

def extract_factor_table(fitted_model, factor_name, base_label):
    """
    Extract a factor table from a statsmodels GLM result.
    Returns a Polars DataFrame: level, coefficient, relativity, 95% CI.
    """
    params = fitted_model.params
    conf   = fitted_model.conf_int()
    prefix = f"C({factor_name}"
    rows   = []

    for name, coef in params.items():
        if name.startswith(prefix):
            # Parameter names look like: C(ncd_f, Treatment(reference='3'))[T.0]
            level = name.split("[T.")[-1].rstrip("]")
            rows.append({
                "level":      level,
                "coef":       round(coef, 4),
                "relativity": round(np.exp(coef), 4),
                "ci_lo":      round(np.exp(conf.loc[name, 0]), 4),
                "ci_hi":      round(np.exp(conf.loc[name, 1]), 4),
            })

    # Add the base level row explicitly
    rows.append({
        "level":      base_label,
        "coef":       0.0,
        "relativity": 1.0000,
        "ci_lo":      1.0000,
        "ci_hi":      1.0000,
    })

    return pl.DataFrame(rows).sort("relativity")

ncd_table  = extract_factor_table(freq_model, "ncd_f", base_label="3")
veh_table  = extract_factor_table(freq_model, "veh_f", base_label="5")
area_table = extract_factor_table(freq_model, "area",  base_label="D")

print("NCD frequency relativities:")
print(ncd_table)
```

Output (approximate, given the simulated data):

```
shape: (6, 5)
┌───────┬─────────┬────────────┬────────┬────────┐
│ level ┆ coef    ┆ relativity ┆ ci_lo  ┆ ci_hi  │
│ ---   ┆ ---     ┆ ---        ┆ ---    ┆ ---    │
│ str   ┆ f64     ┆ f64        ┆ f64    ┆ f64    │
╞═══════╪═════════╪════════════╪════════╪════════╡
│ 5     ┆ -0.2624 ┆ 0.7692     ┆ 0.744  ┆ 0.795  │
│ 4     ┆ -0.1431 ┆ 0.8668     ┆ 0.840  ┆ 0.894  │
│ 3     ┆  0.0000 ┆ 1.0000     ┆ 1.000  ┆ 1.000  │
│ 2     ┆  0.0963 ┆ 1.1011     ┆ 1.067  ┆ 1.137  │
│ 1     ┆  0.1967 ┆ 1.2174     ┆ 1.179  ┆ 1.258  │
│ 0     ┆  0.3302 ┆ 1.3914     ┆ 1.349  ┆ 1.435  │
└───────┴─────────┴────────────┴────────┴────────┘
```

This is the same information Emblem shows in its factor table panel: the base level anchored at 1.0, relativities above and below, and confidence intervals. To format this for a governance review pack in Excel -- colour-coded by significance, one sheet per factor -- see [post #138 on factor tables to Excel](/2026/03/24/how-to-export-factor-table-to-excel-python/).

---

## One-way diagnostics

After fitting, the immediate check is one-way plots: does observed frequency by factor level align with what the model predicts? We covered the full mechanics of [one-way analysis in post #137](/2026/03/23/one-way-analysis-python-pricing-actuary/). Here we connect it to the fitted GLM.

Note: `fittedvalues` from a Poisson GLM are predicted claim counts. Divide by exposure to get frequency.

```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Attach model outputs to the main frame
df_pd["fitted_claims"] = freq_model.fittedvalues
df_pd["fitted_freq"]   = freq_model.fittedvalues / df_pd["exposure"]

def one_way_plot(df, factor_col, title):
    """
    One-way observed vs fitted frequency chart, weighted by exposure.
    Dual-axis: frequency on left, exposure bar chart on right.
    """
    grp = (
        df.groupby(factor_col, observed=True, sort=True)
          .apply(lambda g: pd.Series({
              "exposure":  g["exposure"].sum(),
              "obs_freq":  g["claim_count"].sum() / g["exposure"].sum(),
              "fit_freq":  g["fitted_claims"].sum() / g["exposure"].sum(),
          }), include_groups=False)
    )

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()

    levels = grp.index.astype(str)
    ax2.bar(levels, grp["exposure"], alpha=0.25, color="steelblue",
            label="Exposure (RHS)")
    ax1.plot(levels, grp["obs_freq"], "o-",  color="crimson",
             label="Observed", lw=2)
    ax1.plot(levels, grp["fit_freq"], "s--", color="navy",
             label="Fitted",   lw=2)

    ax1.set_ylabel("Claim frequency")
    ax2.set_ylabel("Earned exposure (car-years)")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()
    return fig

fig_ncd = one_way_plot(df_pd, "ncd_band", "Frequency one-way: NCD band")
fig_veh = one_way_plot(df_pd, "veh_group", "Frequency one-way: Vehicle group")
plt.show()
```

If the observed and fitted lines track each other across all factor levels, the GLM has captured that factor's effect. A systematic curved residual pattern -- say, observed curves upward at the high end of vehicle group while fitted is flat -- means the factor needs finer banding or an interaction term.

---

## Deviance residuals

Emblem displays residual diagnostics after fitting. The two standard checks are deviance residuals against fitted values, and a QQ plot.

```python
import scipy.stats as stats

dev_resid = freq_model.resid_deviance

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Deviance residuals vs fitted frequency
axes[0].scatter(
    freq_model.fittedvalues / df_pd["exposure"],
    dev_resid,
    alpha=0.15, s=8, color="steelblue"
)
axes[0].axhline(0,  color="black", lw=1)
axes[0].axhline( 2, color="red",   lw=1, ls="--")
axes[0].axhline(-2, color="red",   lw=1, ls="--")
axes[0].set_xlabel("Fitted frequency")
axes[0].set_ylabel("Deviance residual")
axes[0].set_title("Deviance residuals vs fitted frequency")
axes[0].set_xscale("log")

# QQ plot of deviance residuals
(osm, osr), (slope, intercept, r) = stats.probplot(dev_resid, dist="norm")
axes[1].plot(osm, osr, "o", alpha=0.2, ms=3, color="steelblue")
axes[1].plot(osm, slope * np.array(osm) + intercept, "r-", lw=2)
axes[1].set_xlabel("Theoretical quantiles")
axes[1].set_ylabel("Sample quantiles")
axes[1].set_title("Normal QQ: deviance residuals")

plt.tight_layout()
plt.show()

print(f"Deviance residual std dev: {dev_resid.std():.3f}")
print(f"Model deviance:            {freq_model.deviance:.1f}")
print(f"Null deviance:             {freq_model.null_deviance:.1f}")
print(f"Pseudo-R2 (McFadden):      "
      f"{1 - freq_model.deviance / freq_model.null_deviance:.4f}")
```

For a Poisson GLM on a large portfolio, deviance residuals should be approximately normal with standard deviation close to 1.0. Significantly above 1.0 is overdispersion -- the model is underestimating variance. If that is what you see, the next step is a quasi-Poisson or negative binomial specification.

A well-specified motor frequency GLM on clean UK data typically achieves a McFadden pseudo-R2 of 0.05-0.15. The pseudo-R2 measures the reduction in deviance from the null (intercept-only) model. Higher values are possible with more features; lower values on unusual book structures are common.

---

## Lift chart

The lift chart compares models by sorting policies into risk buckets and checking how well each model tracks observed frequency across those buckets. We sort by the challenger (CatBoost) to give it the home advantage.

First, fit the CatBoost frequency model:

```python
from catboost import CatBoostRegressor

X_cols       = ["ncd_band", "veh_group", "driver_age", "area"]
cat_features = ["area"]

cb_model = CatBoostRegressor(
    loss_function = "Poisson",
    iterations    = 400,
    learning_rate = 0.05,
    depth         = 5,
    cat_features  = cat_features,
    random_seed   = 42,
    verbose       = 0,
)

# CatBoost Poisson expects count targets, not rates.
# Pass claim_count as label and exposure as sample_weight.
# Divide predictions by exposure to recover the frequency scale.
from catboost import Pool

train_pool = Pool(
    data          = df_pd[X_cols],
    label         = df_pd["claim_count"],
    sample_weight = df_pd["exposure"],
    cat_features  = cat_features,
)

cb_model.fit(train_pool)

# predict() returns predicted counts; divide by exposure for frequency
df_pd["cb_freq"] = cb_model.predict(df_pd[X_cols]) / df_pd["exposure"]
```

Now build the lift chart:

```python
def lift_chart(df, n_bins=10):
    """
    Sort by CatBoost prediction, bin into deciles,
    compare observed vs GLM vs CatBoost within each bin.
    All quantities are exposure-weighted frequencies.
    """
    tmp = df.copy().sort_values("cb_freq")
    tmp["bin"] = pd.qcut(tmp["cb_freq"], q=n_bins, labels=False)

    grp = (
        tmp.groupby("bin")
           .apply(lambda g: pd.Series({
               "obs_freq": g["claim_count"].sum() / g["exposure"].sum(),
               "glm_freq": g["fitted_claims"].sum() / g["exposure"].sum(),
               "cb_freq":  (g["cb_freq"] * g["exposure"]).sum()
                           / g["exposure"].sum(),
           }), include_groups=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = grp.index.astype(str)
    ax.plot(bins, grp["obs_freq"], "o-",  color="crimson",     lw=2.5, label="Observed")
    ax.plot(bins, grp["glm_freq"], "s--", color="navy",        lw=2.0, label="Poisson GLM")
    ax.plot(bins, grp["cb_freq"],  "^-.", color="darkorange",  lw=2.0, label="CatBoost")
    ax.set_xlabel("Predicted frequency decile (low to high risk)")
    ax.set_ylabel("Claim frequency")
    ax.set_title("Lift chart: Poisson GLM vs CatBoost vs Observed")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()
    return fig

fig = lift_chart(df_pd)
plt.show()
```

A good lift chart shows all three lines tracking each other in the middle deciles, with the GBM typically closer to observed in the tails. If the GLM's line flattens out in the top decile while observed continues to rise, the model is leaving discrimination on the table -- usually because of non-linear or interaction effects the additive GLM structure cannot capture.

---

## Comparing model discrimination

The Gini coefficient gives a single-number summary of how well a model orders risk. We use the [exposure-weighted version](/2026/03/23/exposure-weighted-gini-coefficient-python/) for a fair comparison:

```python
def gini_coefficient(y_true, y_pred, exposure):
    """Exposure-weighted Gini coefficient."""
    order = np.argsort(y_pred)
    exp_s = exposure[order]
    y_s   = y_true[order]

    cum_exp   = np.concatenate([[0], np.cumsum(exp_s)]) / exp_s.sum()
    cum_claim = np.concatenate([[0], np.cumsum(y_s)])   / y_s.sum()

    auc = np.trapz(cum_claim, cum_exp)
    return 2 * auc - 1

glm_gini = gini_coefficient(
    df_pd["claim_count"].values,
    freq_model.fittedvalues.values,
    df_pd["exposure"].values,
)
cb_gini = gini_coefficient(
    df_pd["claim_count"].values,
    (df_pd["cb_freq"] * df_pd["exposure"]).values,
    df_pd["exposure"].values,
)

print(f"Poisson GLM Gini: {glm_gini:.4f}")
print(f"CatBoost Gini:    {cb_gini:.4f}")
print(f"Uplift:           {cb_gini - glm_gini:+.4f}")
```

On this synthetic dataset CatBoost will outperform the GLM by a few Gini points -- the true DGP has non-linear age effects the GLM's age banding only partially captures. On real portfolios, the gap between a well-specified GLM and a tuned GBM is typically 0.03-0.08 Gini points: enough to matter commercially, small enough that the GLM remains deployable when the rating engine requires it.

---

## Extracting GBM factor tables with shap-relativities

When the GBM wins on Gini, the natural question is whether its factor structure agrees with the GLM or whether there are effects the GLM factor banding is missing. We built [shap-relativities](/insurance-distill/) for this comparison:

```bash
uv add shap-relativities
```

```python
from shap_relativities import SHAPRelativities

sr = SHAPRelativities(cb_model, df_pd[X_cols])
sr.fit()

# extract_relativities returns all features; filter to ncd_band
all_rels = sr.extract_relativities()
ncd_gbm = all_rels.filter(all_rels["feature"] == "ncd_band")

# Compare GBM relativities against GLM relativities for NCD
glm_ncd = ncd_table.rename({"relativity": "glm_rel"})
comparison = (
    ncd_gbm
    .rename({"relativity": "gbm_rel"})
    .join(glm_ncd[["level", "glm_rel"]], on="level", how="left")
)
print(comparison)
```

If the GLM and GBM relativities agree closely, the GLM factor structure is capturing the main effects. Divergence at extreme levels -- say, the GBM giving NCD-0 a relativity of 1.65 where the GLM gives 1.39 -- is a signal that the GLM's base level needs recalibrating or that NCD is interacting with another factor.

---

## Temporal cross-validation

Emblem does not do temporal cross-validation. In Python, [insurance-cv](https://github.com/burning-cost/insurance-cv) provides proper time-ordered folds that respect the IBNR boundary:

```bash
uv add insurance-cv
```

```python
from insurance_cv import walk_forward_split, InsuranceCV

# Assumes df has an inception_date column
# splits = walk_forward_split(df, date_col="inception_date", min_train_months=24, test_months=6)
# cv = InsuranceCV(splits=splits, df=df)
# for split in cv.splits:
#     print(split.label, split.train_start, split.test_end)
```

Standard k-fold treats insurance observations as exchangeable across time. They are not: a model trained on 2022-2024 data and evaluated on 2022-2024 test policies will see the same trends, IBNR patterns, and seasonal effects in both splits. The correct approach holds out the most recent period(s) as test data and trains only on prior periods. We made the full case for this in [the post on why k-fold is wrong for insurance](/2026/03/21/why-k-fold-cv-is-wrong-for-insurance/).

---

## The combined pure premium

To get from frequency and severity models to a pure premium estimate:

```python
# Severity predictions on all policies (not just those with claims)
df_pd["fitted_sev"]       = sev_model.predict(df_pd)
df_pd["pure_premium"]     = df_pd["fitted_freq"] * df_pd["fitted_sev"]

summary = (
    pl.from_pandas(df_pd)
    .group_by("ncd_band")
    .agg([
        pl.col("exposure").sum(),
        pl.col("claim_count").sum(),
        (pl.col("claim_count").sum() / pl.col("exposure").sum()).alias("obs_freq"),
        pl.col("fitted_freq").mean(),
        pl.col("pure_premium").mean().alias("avg_pure_premium"),
    ])
    .sort("ncd_band")
)
print(summary)
```

This is the equivalent of Emblem's model score output: predicted pure premium by segment. A property of GLMs with canonical link and no regularisation is that the overall exposure-weighted average of fitted frequencies exactly equals the observed frequency. That is, the model is calibrated by construction at the portfolio level.

---

## What this covers and what it does not

This workflow reproduces the core of what an Emblem user does during model development: fit a frequency-severity GLM, extract factor tables with confidence intervals, run one-way diagnostics, check deviance residuals, compare models with a lift chart, and examine a GBM challenger.

What it does not reproduce:

- Emblem's interactive factor smoothing GUI. Smoothing can be done programmatically (cubic splines, isotonic regression, GAMs via [insurance-gam](https://github.com/burning-cost/insurance-gam)), but there is no point-and-click interface.
- Emblem's native Radar export. For that, see [insurance-distill](https://github.com/burning-cost/insurance-distill) and the [CatBoost factor table to Radar post](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/).
- Emblem's integrated data management and audit trail. In Python, version control is handled separately (git for code, DVC or equivalent for data).

The modelling is equivalent. The workflow is more code-heavy but also more reproducible -- the entire analysis above is a script you can re-run, diff, and review in a pull request.

---

## Installing dependencies

```bash
uv add statsmodels catboost polars shap-relativities insurance-cv
```

All five posts in this series use the same core stack. If you are building a Python pricing environment from scratch, the Python insurance pricing toolkit post covers the full dependency set and environment setup.
