---
layout: post
title: "Estimating PCW Conversion Elasticity with Double Machine Learning"
date: 2026-03-26
categories: [causal-inference, demand-modelling]
tags: [pcw, dml, double-machine-learning, elasticity, instrumental-variables, catboost, insurance-causal, econml, doubleml, fca-ep25-2, uk-motor, conversion-model, gates]
description: "How to estimate a causally identified price elasticity from PCW quote data in Python, using commercial loading variation as an instrument and CatBoost nuisance models. The practical companion to our post on PCW endogeneity."
---

[Post 1 of this series]({% post_url 2026-03-26-the-pcw-endogeneity-problem-why-your-conversion-model-is-biased %}) established that the logistic regression your pricing team calls a demand model is biased in three distinct ways: price endogeneity from unobserved risk, the rank mediation trap, and competitor price simultaneity. This post implements the fix.

We will walk through the full estimation pipeline: data structure, instrument construction, the DML estimation itself, segment-level heterogeneous effects, and what the numbers mean for your pricing optimiser. Code examples use [`insurance-causal`](/insurance-causal/) and `doubleml`.

---

## What data you need

Start by being clear about the schema. PCW conversion estimation requires quote-level data, not policy-level data. Each row is one quote event:

```python
import polars as pl

# Minimum viable schema for DML conversion estimation
# One row per PCW quote (not per policy)
schema = {
    "quote_id":        pl.Utf8,    # unique quote identifier
    "quote_date":      pl.Date,    # date of quote — used to join rate review periods
    "converted":       pl.Int8,    # 1 = bought, 0 = did not buy
    "own_price":       pl.Float64, # your quoted premium (£)
    "tech_prem":       pl.Float64, # technical premium at quote time — CRITICAL
    "rate_review_id":  pl.Utf8,    # which quarterly rate review applies to this quote
    "log_cl":          pl.Float64, # log(own_price / tech_prem): the commercial loading
    # Risk confounders
    "age":             pl.Int32,
    "ncd_years":       pl.Int32,   # 0–5
    "vehicle_group":   pl.Utf8,    # A–F
    "region":          pl.Utf8,
    # Competitor prices (from Consumer Intelligence / eBenchmarkers)
    "comp_price_1":    pl.Float64, # cheapest competitor
    "comp_price_2":    pl.Float64, # second cheapest
    "comp_price_3":    pl.Float64,
    "rank":            pl.Int32,   # your rank on the results page — DO NOT USE AS PREDICTOR
}
```

A few points worth flagging. `tech_prem` must be the technical premium as it existed at quote time — not recalculated retroactively from current rate tables. Most insurers store only the final commercial price; if yours does, you need to reconstruct technical premium from rate review records, which is doable but tedious. `rank` is in the schema because you will need it for diagnostics, but do not put it in the confounder set — see post 1 for why.

You need at least three rate review periods with meaningful loading changes (> ±2%). A single flat period gives you no instrument. For a medium-sized motor book, one calendar year of PCW quotes gives 50,000–200,000 records with four rate review periods, which is enough.

---

## Step 1: Build the instrument

The identification strategy is Hausman-style IV via DML. The idea: commercial loading decisions are made at the segment/quarter level by underwriters and actuaries who do not know which specific customers will quote in that period. Loading variation across review periods is therefore orthogonal to individual unobserved risk — it shifts `own_price` without being caused by the customer-level confounders that contaminate raw price.

```python
import numpy as np
import polars as pl

def build_instruments(df: pl.DataFrame) -> pl.DataFrame:
    """
    Construct instruments for own-price endogeneity.

    Instrument 1: log commercial loading (log_cl = log(own_price / tech_prem))
        Variation comes from rate review decisions, not from customer risk.
        Validity requires: loading decisions not correlated with unobserved
        customer risk within the same quarter. This holds if the underwriting
        team applied loadings uniformly across the segment — standard practice.

    Instrument 2: competitor regional loading shift
        For each competitor, compute their average price in OTHER regions
        for the same vehicle group and NCD band. This "leave-one-out"
        competitor price is correlated with your relative rank (competitor
        pricing affects rank globally) but uncorrelated with the specific
        customer's unobserved risk in your region.
    """

    # Instrument 1: Own commercial loading — already in schema as log_cl
    # No transformation needed.

    # Instrument 2: Leave-one-out competitor price index
    # Average of competitor prices EXCLUDING the customer's own region.
    # Captures market-wide competitor cost shocks without the regional
    # claims inflation that also drives your conversion rate.

    df = df.with_columns([
        # Log of average competitor price (own region excluded via join, simplified here)
        pl.mean_horizontal("comp_price_1", "comp_price_2", "comp_price_3")
          .log()
          .alias("log_comp_avg"),
    ])

    # Region-demeaned competitor price: removes the common regional trend
    # that is correlated with claims environment (and therefore conversion)
    region_mean = (
        df.group_by("region")
          .agg(pl.col("log_comp_avg").mean().alias("log_comp_avg_region_mean"))
    )
    df = df.join(region_mean, on="region", how="left")
    df = df.with_columns([
        (pl.col("log_comp_avg") - pl.col("log_comp_avg_region_mean"))
          .alias("comp_price_instrument"),
    ])

    return df
```

For the IV to work, competitor prices must be from an external panel (Consumer Intelligence or eBenchmarkers), not scraped from the PCW mid-session. If you scraped them mid-session, the prices are conditional on your own rank being visible, which introduces selection bias.

---

## Step 2: Partial linear IV DML with `doubleml`

The `insurance-causal` library's [`RenewalElasticityEstimator`](https://github.com/pricing-frontier/insurance-causal) was designed for renewal data. We adapt it here for PCW new business conversion, and extend to the IV setting using `doubleml.DoubleMLPLIV` directly.

The partially linear IV regression (PLIV) model is:

```
Y = D·θ + g(X) + ε        (structural equation)
D = Z·π + m(X) + v         (first stage)
```

where `Y` = conversion (0/1), `D` = log(own\_price), `Z` = instruments, `X` = risk confounders, and `θ` is the causal elasticity we want.

```python
from catboost import CatBoostClassifier, CatBoostRegressor
import doubleml as dml
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def prepare_dml_arrays(df: pl.DataFrame, confounders: list[str]):
    """Convert polars DataFrame to the arrays doubleml expects."""
    df_pd = df.to_pandas()

    # Outcome: conversion (binary)
    Y = df_pd["converted"].values.astype(float)

    # Treatment: log own price (continuous)
    D = np.log(df_pd["own_price"].values)

    # Instruments: own commercial loading + competitor instrument
    Z = df_pd[["log_cl", "comp_price_instrument"]].values

    # Confounders: encode categoricals
    X_raw = df_pd[confounders].copy()
    cat_cols = X_raw.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        X_raw = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    X = X_raw.fillna(X_raw.mean()).values.astype(float)

    return Y, D, Z, X


def fit_pliv(df: pl.DataFrame) -> dml.DoubleMLPLIV:
    """
    Fit the partially linear IV regression via DML.

    Five-fold cross-fitting. CatBoost for all nuisance models.
    The treatment model (E[D|X]) is the first stage — its residuals
    are what the instrument needs to predict.
    """
    confounders = ["age", "ncd_years", "vehicle_group", "region"]
    Y, D, Z, X = prepare_dml_arrays(df, confounders)

    # DoubleML expects a DoubleMLData object
    dml_data = dml.DoubleMLData.from_arrays(
        x=X, y=Y, d=D, z=Z,
    )

    # Nuisance models
    # ml_l: E[Y|X] — outcome model (CatBoost classifier for binary Y)
    ml_l = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        verbose=0, random_seed=42,
        eval_metric="Logloss",
    )
    # ml_m: E[D|X] — treatment model (first stage denominator)
    ml_m = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        verbose=0, random_seed=42,
    )
    # ml_r: E[Z|X] — reduced form (instrument model)
    # With two instruments, doubleml fits one model per instrument column
    ml_r = CatBoostRegressor(
        iterations=500, depth=6, learning_rate=0.05,
        verbose=0, random_seed=42,
    )

    pliv = dml.DoubleMLPLIV(
        obj_dml_data=dml_data,
        ml_l=ml_l,
        ml_m=ml_m,
        ml_r=ml_r,
        n_folds=5,
        n_rep=1,
    )
    pliv.fit()
    return pliv
```

Running `pliv.summary` gives you the ATE — the portfolio-average semi-elasticity. A typical result for a UK motor PCW book might look like:

```
             coef  std err    t     P>|t|   2.5%   97.5%
d_log_price -0.31    0.048  -6.5   0.000  -0.40   -0.21
```

This says: a 1% increase in your quoted premium reduces PCW conversion probability by 0.31 percentage points on average. The naive logistic regression on the same data typically returns something like −0.55: the endogeneity bias inflates apparent price sensitivity by 40–80% in our experience, because the model is picking up the correlation between high prices and low-risk customers (who convert less because they are shopping harder for any deal, not because your specific price is too high).

---

## Step 3: Check the instrument

Before interpreting the elasticity estimate, validate the instrument. Weak instruments produce inflated standard errors and potentially sign-reversed point estimates.

```python
def check_instrument_strength(pliv: dml.DoubleMLPLIV) -> None:
    """
    Print first-stage diagnostics.

    The relevant statistic is the Cragg-Donald F-statistic (or its
    heteroskedasticity-robust Kleibergen-Paap equivalent). Rule of thumb:
    F > 10 is the floor; F > 20 is comfortable.

    With DML the 'first stage' is the partial correlation between the
    instrument residuals (after partialling out X) and the treatment
    residuals (after partialling out X). DoubleML does not report this
    directly, but you can compute it from the stored residuals.
    """
    # Extract residuals from the fitted model
    # pliv._models contains the cross-fitted nuisance models
    # pliv.predictions contains out-of-fold predictions

    # Simplified F-stat from partial correlation of IV residuals and D residuals
    D_resid = pliv.predictions["ml_m"]["d"].flatten()
    Z_resid = pliv.predictions["ml_r"]["IV1"].flatten()

    n = len(D_resid)
    k_z = 1  # number of instruments used (simplify to first)

    # Partial R² of instrument on residualised treatment
    from scipy import stats
    slope, intercept, r_value, p_value, _ = stats.linregress(Z_resid, D_resid)
    f_stat = (r_value**2 / k_z) / ((1 - r_value**2) / (n - k_z - 1))

    print(f"Partial R² (IV → D residual): {r_value**2:.4f}")
    print(f"First-stage F-statistic (approx): {f_stat:.1f}")
    if f_stat < 10:
        print("WARNING: Weak instrument. F < 10. Estimates are unreliable.")
        print("Consider: more rate review periods, or add competitor instrument.")
    elif f_stat < 20:
        print("Borderline instrument strength. Proceed with caution.")
    else:
        print("Instrument strength acceptable.")
```

If F < 10, you have two options: (a) extend the data to include more rate review periods with larger loading changes, or (b) add the competitor regional instrument. In practice, the loading-only instrument is often borderline for PCW data because pricing teams have been reluctant to make large loading changes since GIPP in 2022. The competitor regional instrument adds significant identifying power.

---

## Step 4: Segment-level GATES with `insurance-causal`

The portfolio-average elasticity is useful for a sanity check, but the pricing optimiser needs a CATE surface. For PLIV, heterogeneous effects require a different estimator — the AIPW-IV forest. The `insurance-causal` library's `RenewalElasticityEstimator` handles the renewal case with CausalForestDML; for the conversion case we use the same machinery adapted to the PCW schema.

```python
from insurance_causal.elasticity import RenewalElasticityEstimator

def estimate_conversion_elasticity(df: pl.DataFrame) -> RenewalElasticityEstimator:
    """
    Adapt RenewalElasticityEstimator to PCW new business conversion.

    The schema differences:
    - outcome: "converted" instead of "renewed"
    - treatment: "log_price_change" computed as log(own_price / tech_prem)
      — this is the commercial loading, which is our instrument. For CATE
      estimation we treat the loading as the treatment directly, which gives
      us the effect of commercial loading decisions on conversion. This is
      actually the policy-relevant quantity: actuaries set loadings, not
      absolute prices.

    Important: this estimates the effect of log(commercial loading) on
    conversion, not the full causal effect of log(price) on conversion.
    The PLIV step (Step 2) gives the latter. These are related but distinct:
    the CATE from this step conditions on the instrument being the treatment,
    which is valid for optimising loading decisions specifically.
    """
    # Rename schema to match RenewalElasticityEstimator expectations
    df_adapted = df.rename({
        "converted": "renewed",          # outcome column
        "log_cl": "log_price_change",     # treatment: log(price / tech_prem)
    })

    confounders = ["age", "ncd_years", "vehicle_group", "region"]

    est = RenewalElasticityEstimator(
        cate_model="causal_forest",
        binary_outcome=True,
        n_folds=5,
        n_estimators=200,        # must be divisible by n_folds * 2 = 10
        catboost_iterations=500,
        random_state=42,
    )

    est.fit(
        df_adapted,
        outcome="renewed",
        treatment="log_price_change",
        confounders=confounders,
    )

    return est
```

Once fitted, GATE estimation by segment is one call:

```python
def print_gates(est: RenewalElasticityEstimator, df: pl.DataFrame) -> None:
    """Print group-average treatment effects by key pricing segments."""

    # Adapt schema as above
    df_adapted = df.rename({"converted": "renewed", "log_cl": "log_price_change"})

    print("=== GATES by NCD band ===")
    gates_ncd = est.gate(df_adapted, by="ncd_years")
    print(gates_ncd)

    # Typical output for a UK motor PCW book:
    # ncd_years  elasticity  ci_lower  ci_upper      n
    #         0      -0.48     -0.61     -0.35    8420
    #         1      -0.41     -0.52     -0.30   11350
    #         2      -0.35     -0.45     -0.25   14200
    #         3      -0.29     -0.38     -0.20   18700
    #         4      -0.22     -0.31     -0.13   21300
    #         5      -0.14     -0.21     -0.07   26000

    print("\n=== GATES by region ===")
    gates_region = est.gate(df_adapted, by="region")
    print(gates_region)

    print("\n=== GATES by vehicle group ===")
    gates_vg = est.gate(df_adapted, by="vehicle_group")
    print(gates_vg)
```

The NCD band result is the most consistent finding across the insurers we have looked at. Customers with NCD 0–1 are roughly 3× more price-sensitive than those at NCD 5. This is not surprising — NCD 0 customers are predominantly young drivers who price-shop heavily on PCWs and have limited brand loyalty. But the *magnitude* of the gradient matters for optimisation: if your current loading is uniform across NCD bands, you are leaving conversion volume on the table at NCD 0–1 and over-discounting relative to elasticity at NCD 4–5.

---

## Step 5: The regulatory check

EP25/2's evaluation of pricing practices highlights the regulatory scrutiny applied to elasticity-based approaches — particularly where price sensitivity is correlated with protected or proxy-protected characteristics. The concern is clear even if EP25/2 does not prescribe specific controls: using elasticity as a pricing lever requires you to understand and document what the elasticity is proxying for.

NCD band, region, and vehicle group are all proxy-correlated with age. Before feeding the GATE surface into your pricing optimiser, run this:

```python
def regulatory_elasticity_audit(
    est: RenewalElasticityEstimator,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Flag segments where elasticity is correlated with proxy demographics.

    FCA concern: if NCD 0 customers are predominantly young (they are),
    and you price them more aggressively based on higher elasticity,
    you are using a proxy for age as a pricing lever even if age itself
    is not in the model. This is not unlawful in motor (age is explicitly
    permitted), but in home insurance it would be a problem. Know what
    your segment elasticities are proxying for.
    """
    df_adapted = df.rename({"converted": "renewed", "log_cl": "log_price_change"})

    # Get per-row CATE
    cates = est.cate(df_adapted)

    df_audit = df.with_columns([
        pl.Series("estimated_elasticity", cates),
    ])

    # Segment by age band (a demographic proxy) and check elasticity distribution
    df_audit = df_audit.with_columns([
        pl.when(pl.col("age") < 25).then(pl.lit("17-24"))
          .when(pl.col("age") < 35).then(pl.lit("25-34"))
          .when(pl.col("age") < 45).then(pl.lit("35-44"))
          .when(pl.col("age") < 55).then(pl.lit("45-54"))
          .when(pl.col("age") < 65).then(pl.lit("55-64"))
          .otherwise(pl.lit("65+"))
          .alias("age_band"),
    ])

    audit_result = (
        df_audit
        .group_by("age_band")
        .agg([
            pl.col("estimated_elasticity").mean().alias("mean_elasticity"),
            pl.col("estimated_elasticity").std().alias("sd_elasticity"),
            pl.len().alias("n"),
        ])
        .sort("age_band")
    )

    return audit_result
```

If `mean_elasticity` monotonically declines with age (younger = more elastic), and you are using elasticity as a pricing signal, you are de facto age-pricing even if age is not a direct input. In motor, that is permitted. In home, it is not. The audit output should go into your model governance documentation alongside the elasticity estimates themselves.

---

## What changes in your pricing optimiser

The GATE surface from this analysis replaces the elasticity coefficients in your existing optimisation setup. If you use `insurance-causal`'s `RenewalPricingOptimiser` or a custom loading optimiser, the input is the same: a per-segment semi-elasticity that maps commercial loading to conversion probability.

The practical differences from the naive model are usually:

**Segment gradient is steeper.** Causal elasticities show more variation across NCD bands and age groups than naive logistic regression estimates, because the naive model's endogeneity bias partly averages out across segments. The causal model says: you can load NCD 4–5 customers harder than your current model suggests, and you should discount NCD 0–1 customers more aggressively than your model suggests. Both adjustments increase portfolio NPV.

**Rank should not be in the optimiser objective.** Post 1 established that rank is a mediator, not a confounder. The total-effect elasticity estimated here — without rank in the confounder set — already includes the conversion improvement from rank improvement that comes with a price cut. If your optimiser also includes a rank-conversion curve and tries to model those effects separately, it will double-count.

**The instrument tells you something about pricing team behaviour.** The first-stage relationship between loading decisions and price is, by construction, what the instrument exploits. A strong first stage (F > 20) means your loading decisions are having their intended effect on prices. A weak first stage (F < 10) may mean that commercial loadings are being overridden by individual underwriter adjustments, or that loading decisions are smaller than they used to be — both operationally interesting findings independent of the elasticity estimation.

---

## What this does not fix

We said in post 1 that there are three problems, and this post addresses two of them fully and one partially.

**Price endogeneity:** fixed by the PLIV design with commercial loading as the instrument.

**Rank mediation trap:** fixed by excluding rank from the confounder set. The total-effect elasticity you estimate includes the rank-mediated pathway, which is the right quantity for pricing optimisation.

**Competitor price simultaneity:** partially addressed. The region-demeaned competitor instrument soaks up the common regional claims trend, but does not fully control for market-wide cost shocks that hit all regions simultaneously (a sustained repair cost inflation, for instance). For a full BLP-style demand system you would need structural price equations for each competitor — a substantial additional step. For most UK motor pricing teams, the two-instrument approach above gives a materially better estimate than naive logistic regression, and the remaining bias is small relative to the improvement from addressing endogeneity.

---

## Libraries and references

- [`insurance-causal`](https://github.com/pricing-frontier/insurance-causal): `RenewalElasticityEstimator`, `RenewalPricingOptimiser`
- [`doubleml`](https://docs.doubleml.org/): `DoubleMLPLIV` for the IV step
- [`econml`](https://econml.azurewebsites.net/): `CausalForestDML` (used internally by insurance-causal)
- Chernozhukov et al. (2018), *Econometrics Journal* 21(1) — the DML paper
- Chernozhukov et al. (2020), *Review of Economic Studies* 87(1) — GATES/BLP inference
- Schultz et al. (2023), arXiv:2312.15282 — causal forecasting for pricing, directly applicable
- Muijsson (2022), *Expert Systems with Applications* — DML for insurance renewal, the closest published insurance application
---

- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) — the `insurance-optimise` library that packages the DML elasticity pipeline for portfolio pricing, with the ENBP compliance checker
- [The PCW Endogeneity Problem: Why Your Conversion Model Is Biased](/2026/03/26/the-pcw-endogeneity-problem-why-your-conversion-model-is-biased/) — part 1 of this series: the causal structure behind the bias this post fixes

---

- FCA: [Evaluation Paper 25/2 (July 2025)](https://www.fca.org.uk/publications/corporate-documents/evaluation-paper-25-2-general-insurance-pricing-practices-remedies)
