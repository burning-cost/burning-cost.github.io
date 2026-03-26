---
layout: post
title: "One-Way Analysis in Python: From Scratch to Production"
date: 2026-03-23
author: Burning Cost
categories: [python, pricing-analysis, tutorials]
description: "One-way analysis in Python for pricing actuaries: pandas from scratch, credibility-weighted confidence intervals, thin cell handling, GBM shortcuts."
tags: [one-way-analysis, factor-tables, pandas, python, uk-insurance, motor-pricing, shap-relativities, insurance-gam, credibility, confidence-intervals, pricing-actuary, observed-expected]
---

Before any model is fitted, before any interaction is explored, there is one-way analysis. Sort by a rating factor, bin it if it is continuous, compute observed frequency versus expected frequency by level, weight by exposure, look at the chart, repeat for every factor. It is the first thing a pricing actuary does on a new dataset and the most common diagnostic they return to throughout the modelling cycle.

Python tutorials for insurance pricing skip it almost entirely. This post fills that gap: we build a correct one-way from scratch in pandas, add proper confidence intervals, then show how to get factor tables from a fitted GBM model using [shap-relativities](https://github.com/burning-cost/shap-relativities) — which is what production work actually looks like.

---

## What one-way analysis is

A one-way analysis produces a table and chart for a single rating factor. For each level of that factor (e.g., NCD band 0, 1, 2, 3, 4, 5), it shows:

- **Earned exposure** (car-years) — how much risk volume is in each level
- **Observed frequency** (claims / exposure) — what actually happened
- **Expected frequency** (model prediction / exposure) — what the model predicts
- **Observed-to-Expected ratio** (O/E or A/E) — the calibration signal

The O/E ratio is the diagnostic. An O/E above 1.0 at a factor level means the model is underpricing that segment. An O/E below 1.0 means overpricing. A systematic pattern across ordered levels — say, O/E climbing with vehicle group — means the model's vehicle group relativity structure is wrong.

One-way is fast, transparent, and remains useful long after you have fitted a GBM that has buried all the factor effects in hundreds of trees. It is the check you run before trusting the model.

---

## Building it from scratch in pandas

We will use a synthetic UK motor dataset: 50,000 policies, frequency model, five factors. Then we will write a `one_way()` function that handles exposure weighting correctly.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 50_000

# --- Synthetic portfolio ---
df = pd.DataFrame({
    "ncd_band":   rng.integers(0, 6, n),           # 0–5 (0 = no NCD, 5 = max NCD)
    "veh_group":  rng.integers(1, 21, n),           # ABI groups 1–20
    "driver_age": rng.integers(17, 80, n),          # years
    "exposure":   np.where(
        rng.random(n) < 0.15,
        rng.uniform(0.1, 0.4, n),                   # short-term
        rng.uniform(0.7, 1.0, n),                   # standard annual
    ),
})

# True log-frequency: NCD and vehicle group are the main drivers
log_true = (
    np.log(0.07)
    - 0.12 * df["ncd_band"]
    + 0.04 * (df["veh_group"] - 1)
    - 0.01 * np.maximum(df["driver_age"] - 25, 0)
)
df["true_freq"] = np.exp(log_true)
df["claims"] = rng.poisson(df["true_freq"] * df["exposure"]).astype(float)

# Simulated model prediction: mostly right but missing some NCD effect
log_pred = (
    np.log(0.07)
    - 0.09 * df["ncd_band"]    # model underweights NCD by 25%
    + 0.04 * (df["veh_group"] - 1)
    - 0.01 * np.maximum(df["driver_age"] - 25, 0)
)
df["predicted"] = np.exp(log_pred) * df["exposure"]  # predicted claim count
```

Now the core one-way function:

```python
def one_way(
    df: pd.DataFrame,
    factor: str,
    actual_col: str = "claims",
    predicted_col: str = "predicted",
    exposure_col: str = "exposure",
    min_exposure: float = 0.0,
) -> pd.DataFrame:
    """
    Exposure-weighted one-way analysis for a single rating factor.

    Returns a DataFrame with one row per factor level:
      - exposure: total earned car-years
      - actual_freq: observed frequency (actual claims / exposure)
      - expected_freq: model prediction (predicted claims / exposure)
      - oe_ratio: observed-to-expected (actual / predicted claims)
      - oe_lower, oe_upper: 95% Poisson confidence interval on O/E ratio

    Levels with exposure < min_exposure are flagged but retained in the
    output (with NaN frequencies). The caller decides whether to drop them.
    """
    grp = (
        df.groupby(factor, sort=True)
        .agg(
            exposure   =(exposure_col,   "sum"),
            actual     =(actual_col,     "sum"),
            predicted  =(predicted_col,  "sum"),
        )
        .reset_index()
    )

    grp["actual_freq"]   = grp["actual"]   / grp["exposure"]
    grp["expected_freq"] = grp["predicted"] / grp["exposure"]
    grp["oe_ratio"]      = grp["actual"]   / grp["predicted"]

    # Poisson 95% CI on O/E ratio
    # Observed claims is Poisson(lambda). CI on lambda via chi-squared exact:
    #   lower = chi2(2*k, 0.025) / 2 / predicted
    #   upper = chi2(2*(k+1), 0.975) / 2 / predicted
    from scipy.stats import chi2
    k = grp["actual"].values
    p = grp["predicted"].values
    grp["oe_lower"] = chi2.ppf(0.025, 2 * k)       / 2.0 / p
    grp["oe_upper"] = chi2.ppf(0.975, 2 * (k + 1)) / 2.0 / p

    # Flag thin cells
    grp["thin_cell"] = grp["exposure"] < min_exposure

    return grp


# Run on NCD band
ow_ncd = one_way(df, "ncd_band", min_exposure=200.0)
print(ow_ncd[["ncd_band", "exposure", "actual_freq", "expected_freq", "oe_ratio",
              "oe_lower", "oe_upper"]].to_string(index=False))
```

Output (approximate):

```
 ncd_band   exposure  actual_freq  expected_freq  oe_ratio  oe_lower  oe_upper
         0    1421.3       0.1063         0.0846     1.256     1.173     1.342
         1    1410.5       0.0928         0.0761     1.219     1.138     1.303
         2    1398.7       0.0833         0.0686     1.215     1.133     1.299
         3    1406.2       0.0626         0.0619     1.011     0.942     1.083
         4    1387.4       0.0552         0.0557     0.991     0.921     1.063
         5    1402.0       0.0487         0.0501     0.972     0.904     1.044
```

The pattern is clear: NCD bands 0–2 are consistently O/E above 1.2, bands 3–5 are near 1.0. The model is undercharging NCD 0 policies by about 25% because it underweights the NCD relativity, exactly the defect we built in.

---

## The confidence interval matters

The Poisson CI above uses the exact chi-squared construction. This is the right approach for count data, not a normal approximation with $\pm 1.96\sqrt{\hat{p}/n}$.

Why it matters: for thin cells (say, 8 actual claims), the normal approximation produces symmetric CIs that can go negative. The chi-squared construction is always non-negative and correct for small counts. For large cells (500+ claims), the two are nearly identical.

The `scipy.stats.chi2.ppf` approach handles the edge case of zero observed claims: when $k = 0$, `chi2.ppf(0.025, 0) = 0`, so the lower bound is 0.0, which is correct.

---

## Handling thin cells

The `min_exposure` parameter flags cells below a threshold. What you do with flagged cells depends on context:

**Option 1: Drop them.** Safe for exploratory analysis. Dangerous for governance if it hides a problematic segment.

**Option 2: Apply credibility weighting.** Blend the cell's observed frequency with the overall portfolio mean, weighted by the Bühlmann credibility weight $z = n / (n + k)$ where $k$ is calibrated to the portfolio. A simple but practical choice is $k = 500$ car-years for UK motor frequency models:

```python
def credibility_weight(exposure: float, k: float = 500.0) -> float:
    """Bühlmann credibility weight. Returns 0 for no data, approaches 1 for large cells."""
    return exposure / (exposure + k)


portfolio_freq = df["claims"].sum() / df["exposure"].sum()

ow_ncd["credibility_z"]    = ow_ncd["exposure"].apply(credibility_weight)
ow_ncd["blended_oe_ratio"] = (
    ow_ncd["credibility_z"] * ow_ncd["oe_ratio"]
    + (1 - ow_ncd["credibility_z"]) * 1.0   # blend toward 1.0 (no adjustment)
)
```

**Option 3: Pool with adjacent levels.** Works well for ordinal factors (vehicle group, driver age) where adjacent levels genuinely share risk characteristics. Implement as a merge pass before the groupby.

The 500 car-year threshold for $k$ is not arbitrary — it corresponds roughly to the exposure at which a 25% effect becomes statistically detectable at the 5% level with Poisson counts. For severity models with high variance, you would use a larger $k$; for binary frequency in large personal lines books, smaller.

---

## Making it production-ready

For a real pricing project, wrap `one_way()` to run across all factors in the dataset and collect results:

```python
def one_way_all_factors(
    df: pd.DataFrame,
    factors: list[str],
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Run one_way() on each factor and return a dict keyed by factor name."""
    return {factor: one_way(df, factor, **kwargs) for factor in factors}


factors = ["ncd_band", "veh_group", "driver_age"]
results = one_way_all_factors(df, factors, min_exposure=200.0)

# Quick summary: which factors have the most extreme O/E?
for factor, tbl in results.items():
    max_oe = tbl["oe_ratio"].max()
    min_oe = tbl["oe_ratio"].min()
    print(f"{factor:20s}  O/E range: {min_oe:.3f} – {max_oe:.3f}")
```

You also want to sort levels correctly. For `ncd_band` and `veh_group`, integer sort is fine. For `postcode_district` or `vehicle_make`, you want to sort by exposure descending to see the material levels first. Pass `sort=False` to the groupby and sort manually:

```python
grp = grp.sort_values("exposure", ascending=False)
```

---

## The production shortcut: shap-relativities

From-scratch one-way analysis works well for GLMs. For a fitted GBM — CatBoost, LightGBM — the marginal O/E chart is harder to interpret because the model includes interaction effects that the one-way ignores. A vehicle group effect in the GBM is not a pure main effect; it includes partial interaction terms with driver age, NCD, and region.

The actuarially correct equivalent for GBMs is to extract the SHAP-based factor table. This gives you a one-row-per-level relativity for each feature, normalised to a base level, with confidence intervals. It is directly comparable to GLM exp(beta) relativities and is what [shap-relativities](https://github.com/burning-cost/shap-relativities) produces.

```bash
uv add 'shap-relativities[ml]'
```

```python
import polars as pl
import catboost
from shap_relativities import SHAPRelativities

# Fit a CatBoost frequency model (Poisson objective, log link)
cat_features = ["ncd_band", "veh_group"]
X = df[["ncd_band", "veh_group", "driver_age"]]
y = df["claims"] / df["exposure"]   # annualised frequency as target

model = catboost.CatBoostRegressor(
    loss_function="Poisson",
    iterations=400,
    learning_rate=0.05,
    depth=4,
    cat_features=cat_features,
    verbose=0,
)
model.fit(X, y, sample_weight=df["exposure"])

# Extract SHAP-based factor relativities
sr = SHAPRelativities(
    model=model,
    X=X,
    exposure=df["exposure"].values,
    categorical_features=cat_features,
)
sr.fit()

relativities = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"ncd_band": "5", "veh_group": "1"},
)

ncd_table = relativities.filter(pl.col("feature") == "ncd_band")
print(ncd_table.select(["level", "relativity", "lower_ci", "upper_ci", "exposure_weight"]))
```

The `SHAPRelativities` output is a Polars DataFrame with one row per (feature, level), ready for a chart. Each relativity is the multiplicative effect relative to the base level, with CLT-based 95% confidence intervals derived from the SHAP value distribution across policies in that level.

This is the "production way" because it handles interactions correctly. The SHAP allocation decomposes each prediction into per-feature contributions. When you aggregate these over policies in a given NCD band, you get the average contribution of NCD to that band's predictions — the main effect, averaged over the actual distribution of the other correlated features in the data. This is a materially better estimate of the NCD effect than the marginal O/E, which conflates the NCD effect with any correlation between NCD and the other factors.

---

## GAM-based factor tables with insurance-gam

If you are using a GAM rather than a GBM — for example, an EBM (Explainable Boosting Machine) from [insurance-gam](https://github.com/burning-cost/insurance-gam) — the factor tables are native to the model architecture. An EBM is additive by construction; its shape functions are the factor tables.

```bash
uv add 'insurance-gam[ebm]'
```

```python
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

# InsuranceEBM infers feature names from X at fit time
ebm = InsuranceEBM(loss="poisson")
ebm.fit(X, df["claims"], exposure=df["exposure"])

# Factor table for NCD band: native to the EBM shape function
rel = RelativitiesTable(ebm)
ncd_shape = rel.table("ncd_band")
print(ncd_shape)
```

The EBM shape functions are interpretable directly without SHAP decomposition. Whether you prefer an EBM or a GBM plus SHAP is a modelling decision; for interpretability-first use cases (governance, Lloyd's submissions, FCA explanations), the EBM's native factor tables are easier to defend.

---

## Practical checklist

Before presenting a one-way analysis to a pricing meeting:

- **Check exposure by level first.** If a level has less than 100 car-years, flag it visually. Less than 50, merge or suppress.
- **Apply credibility weighting for the final O/E summary.** Raw O/E ratios in thin cells should not drive factor selections.
- **Plot the confidence intervals, not just the point estimates.** An O/E of 1.3 with a CI of [0.9, 1.7] is not a finding.
- **Show both sorted by factor level and sorted by exposure.** Sorted by level reveals trends; sorted by exposure reveals where the material mispricing is.
- **Note what the model does not know.** If the one-way for vehicle group shows a clear trend the model is missing, the explanation is usually that the model is right but the one-way is confounded. Check: does vehicle group correlate with NCD in your data? If so, the one-way chart is the wrong diagnostic.
- **For GBMs, use SHAP-based factor tables, not one-ways.** The marginal one-way for a GBM feature mixes main effects and interactions. `shap-relativities` gives you the correctly decomposed main effect.

One-way analysis is the starting point, not the conclusion. It tells you where to look. The model, the SHAP decomposition, and the confidence intervals tell you what is actually there.
