---
layout: post
title: "Your Group Factors Are Not All Worth Modelling"
date: 2025-04-14
categories: [pricing, libraries]
tags: [multilevel-models, random-effects, credibility, catboost, reml, icc, broker, scheme, fleet, postcode, high-cardinality, variance-components, buhlmann-straub, insurance-multilevel, python]
description: "ICC diagnostics for multiple group factors in insurance pricing. When broker, scheme, fleet, and postcode sector effects are worth modelling with REML..."
---

There is a tier of problems just above "the model works" that pricing teams only reach after the first year with a new tool. The first year is spent getting the infrastructure right: integrating the library, validating the variance component estimates, convincing compliance that BLUP-derived broker factors are defensible. The second year is when the harder question arrives.

The harder question is: how many group factors should we be modelling this way?

Broker is obvious. Scheme is probably worth it. Fleet? Possibly. Postcode sector? The model has thousands of them. At what point does stacking REML random effects across multiple group levels stop helping and start producing noise?

The answer is the ICC — the Intraclass Correlation Coefficient — and using it properly is what separates teams that deploy multilevel models sensibly from teams that bolt random effects onto every categorical in the dataset.

---

## The problem with indiscriminate group effects

The intuition behind random effects is appealing: any time you have a categorical feature with many levels and uneven group sizes, credibility-weighted BLUPs beat dummies. So why not apply it everywhere?

Two reasons. First, if the between-group variance τ² is genuinely small relative to within-group noise σ², the REML estimator will correctly push all the BLUPs toward zero. Every group gets a multiplier close to 1.0. The random effects stage has added computation and governance complexity for essentially nothing. You cannot tell this without looking at the variance components.

Second, and less obvious: the REML estimate of τ² depends on how many groups you have and how much data each group carries. Postcode sector in UK motor might give you 9,000 groups with a median of 14 policies each. The REML estimator on such a distribution will struggle. The within-group variance σ² is estimated from thin data per group. The between-group variance τ² is estimated from 9,000 group means of questionable reliability. You can get plausible-looking output that conceals a poorly identified model.

The diagnostic workflow below gives you numbers to make this decision on, rather than intuition.

---

## Running the ICC diagnostic

After fitting a `MultilevelPricingModel`, examine the ICC for each group level before trusting the multipliers:

```python
from insurance_multilevel import (
    MultilevelPricingModel,
    variance_decomposition,
    high_credibility_groups,
)

model = MultilevelPricingModel(
    catboost_params={"iterations": 500, "loss_function": "Poisson"},
    random_effects=["broker_id", "scheme_id", "fleet_id", "postcode_sector"],
    min_group_size=5,
    reml=True,
)

model.fit(X_train, y_train, weights=exposure, group_cols=[
    "broker_id", "scheme_id", "fleet_id", "postcode_sector"
])

vd = variance_decomposition(model.variance_components)
print(vd)
```

A realistic UK motor portfolio might return:

```
┌──────────────────┬────────┬────────┬───────┬────────────┐
│ level            ┆ tau2   ┆ sigma2 ┆ icc   ┆ buhlmann_k │
╞══════════════════╪════════╪════════╪═══════╪════════════╡
│ broker_id        ┆ 0.041  ┆ 0.219  ┆ 0.158 ┆ 5.34       │
│ scheme_id        ┆ 0.063  ┆ 0.198  ┆ 0.241 ┆ 3.14       │
│ fleet_id         ┆ 0.008  ┆ 0.224  ┆ 0.034 ┆ 28.0       │
│ postcode_sector  ┆ 0.003  ┆ 0.226  ┆ 0.013 ┆ 75.3       │
└──────────────────┴────────┴────────┴───────┴────────────┘
```

The table tells you four distinct stories. Scheme is the dominant group effect: 24% of residual variance after CatBoost is attributable to scheme membership. That is not noise — that is selection effect from how schemes are constituted. A fleet scheme for professional services solicitors writes structurally different risk than a young driver scheme, and CatBoost cannot see this from vehicle type and postcode alone.

Broker is material: 16%. Worth modelling. The broker effect is partly selection (which types of customer each broker attracts) and partly underwriting culture (how stringent each broker is at point of sale). Both are unobservable from the policy features CatBoost receives.

Fleet is small: 3.4%. The Bühlmann k of 28 means a fleet needs 28 policy-years of exposure to reach 50% credibility. Very few fleets in a personal lines book get anywhere near that. The BLUPs will be close to zero for almost every fleet. The random effects stage is working correctly — it is shrinking almost everything to the portfolio mean — but it is doing so because there is almost no signal to preserve. You are better off excluding `fleet_id` from the random effects list and saving the REML computation.

Postcode sector is noise: 1.3%. ICC below 2% is a signal that the between-group variance is within estimation error. If you are already modelling postcode risk via a spatial component or postcode district in CatBoost, the sector-level residual variation is a mix of genuine micro-area signal and sampling noise on thin cells. Do not model it as a random effect. The BYM2 spatial model from [`insurance-spatial`](https://github.com/burning-cost/insurance-spatial) is the right tool if postcode geography is genuinely material — it borrows spatial structure to stabilise the thin cells that REML treats independently.

---

## The ICC threshold question

The question of what ICC is "worth modelling" does not have a universal answer. It depends on your portfolio size, the number of groups, and what you are going to do with the multipliers.

A practical guide:

| ICC   | Interpretation | Recommendation |
|-------|---------------|----------------|
| > 0.15 | Material between-group variance; group membership is a meaningful pricing signal | Model with REML random effects |
| 0.05–0.15 | Moderate; worth modelling if groups are large enough to get Z > 0.5 for most | Model if median group size supports credibility; skip if most groups are very thin |
| 0.02–0.05 | Marginal; only the largest groups will reach meaningful credibility | Probably not worth the governance overhead unless a few large groups dominate the portfolio |
| < 0.02 | Negligible; within estimation error | Exclude from random effects |

These are starting points, not rules. A specialist fleet book where a handful of fleets each write thousands of policies might have an ICC of 0.06 but still produce useful multipliers because the group sizes are large. A broker market with 400 brokers and a median of 8 policies each might have an ICC of 0.12 but still produce unreliable multipliers because the data is so thin that τ² itself is poorly estimated.

The check that resolves ambiguity is the credibility weight distribution:

```python
from insurance_multilevel import high_credibility_groups

summary = model.credibility_summary("broker_id")

# How many brokers have Z > 0.7?
hc = high_credibility_groups(summary, min_z=0.7)
print(f"{len(hc)} of {len(summary)} brokers reach 70% credibility")

# Distribution of credibility weights
print(summary.select("credibility_weight").describe())
```

If most groups have Z < 0.3, the random effects stage is producing multipliers that are 70%+ shrinkage toward the portfolio mean. That is not necessarily wrong — it is what the data warrants — but it means the stage is contributing very little commercially, and you should consider whether the governance cost (explaining BLUPs in pricing reviews, managing model documentation) is justified.

---

## Composing effects across levels

When you include multiple group levels with material ICC, the multipliers compose on the log scale:

```python
# Broker effect: exp(b_broker) = 1.14 (14% loading)
# Scheme effect: exp(b_scheme) = 0.91 (9% discount)
# Final multiplier: 1.14 × 0.91 = 1.037

# This is handled automatically by model.predict()
premiums = model.predict(X_test, group_cols=["broker_id", "scheme_id"])
```

The library fits a separate `RandomEffectsEstimator` per group level. Broker effects and scheme effects do not interact in the model — each is an independent one-way random effects problem on the Stage 1 residuals. This is clean and auditable, but it means the model cannot capture a situation where Broker A's business through Scheme C is systematically different from what you would predict by multiplying their individual BLUPs.

In practice, broker-by-scheme interaction effects are usually small relative to the main effects unless your book has a specific scheme that one broker dominates. If you suspect a material interaction, the diagnostic is to look at the residuals from the two-stage model, broken down by broker-scheme cell:

```python
residuals = model.log_ratio_residuals(X_train, y_train)

# Check residuals by broker-scheme combination
import polars as pl

df_diag = X_train.with_columns([
    pl.Series("residual", residuals)
]).group_by(["broker_id", "scheme_id"]).agg([
    pl.count("residual").alias("n"),
    pl.mean("residual").alias("mean_resid"),
])

# Flag cells with large residuals and enough exposure
suspicious = df_diag.filter(
    (pl.col("n") > 30) & (pl.col("mean_resid").abs() > 0.15)
)
print(suspicious.sort("mean_resid", descending=True))
```

A broker-scheme cell with 80 policies and a mean residual of +0.22 after accounting for both the broker and scheme BLUPs separately is something your underwriting team should know about. The model is giving both broker and scheme a clean bill of health individually, but the combination is running hot. That is underwriting intelligence, not a model failure.

---

## High-cardinality limits: postcode sector at scale

Postcode sector in UK personal lines is the obvious case where the numbers break down. England and Wales have approximately 9,000 postcode sectors (the SW1A part of a postcode). A motor book of 200,000 policies might have a median of 22 policies per sector, with a heavy tail: 50 sectors covering the major conurbations with 500+ policies, and thousands of rural sectors with fewer than five.

The REML model with `min_group_size=5` will exclude a large fraction of these sectors from variance estimation. The τ² estimate will be dominated by the well-populated urban sectors. The resulting Bühlmann k will reflect the behaviour of those sectors, which may not be representative of the rural tail.

The honest answer here is that REML one-way random effects are not the right tool for postcode sector in a standard personal lines book. The geographic correlation structure matters: a sector with thin data but surrounded by sectors with bad experience should borrow from its neighbours, not from the portfolio mean. That is the BYM2 model.

The two-level approach that works:

1. Use CatBoost Stage 1 with postcode *district* (the SW1 part) as a feature. At ~3,000 districts with a median of ~65 policies each, district-level effects are learnable from the data.
2. Use REML Stage 2 for broker and scheme effects, which are the group factors where credibility weighting is genuinely useful.
3. If postcode sector micro-area effects are material for your book, add a BYM2 spatial component from [`insurance-spatial`](https://github.com/burning-cost/insurance-spatial) as a separate stage.

This is not a limitation of the credibility approach — it is the correct application of it. Different group factors have different statistical structures, and using the right model for each is better than forcing everything through the same framework.

---

## Checking whether Stage 2 earns its place

Once you have decided which group factors to model, the final check is whether Stage 2 actually improves predictions on held-out data:

```python
from insurance_multilevel import lift_from_random_effects

stage1_preds = model.stage1_predict(X_test)
final_preds  = model.predict(X_test, group_cols=["broker_id", "scheme_id"])

lift = lift_from_random_effects(
    y_test,
    stage1_preds,
    final_preds,
    weights=exposure_test,
)

print(lift)
```

```
{
  'rmse_stage1':       0.4231,
  'rmse_final':        0.3894,
  'rmse_improvement_pct':  7.97,
  'malr_stage1':       0.4118,
  'malr_final':        0.3801,
  'malr_improvement_pct':  7.70,
}
```

A 7–8% improvement in MALR from Stage 2 alone — after a well-tuned CatBoost Stage 1 — is a substantial gain. This is consistent with the scheme ICC of 0.24 and broker ICC of 0.16 from the diagnostic table above. The lift from random effects is roughly proportional to the ICC: if total residual variance has 40% attributable to group membership, you expect a correspondingly large improvement from modelling it correctly.

On a portfolio where the only material group factor was broker with ICC = 0.05, the lift would be much smaller — probably 1–2% in MALR. That is still real money at scale, but whether it justifies the model governance overhead of two additional group effect tables is a decision for the pricing team, not the algorithm.

---


The credibility framework gets misused in two directions. Some teams apply it nowhere, relying instead on manual spreadsheet loadings that are applied at full credibility regardless of the data behind them. Others, having seen the framework work for broker, apply it to every high-cardinality categorical in the dataset, producing random effects stages that are doing nothing because the ICC is 0.01 and every BLUP is 0.98 × 0 = 0.

The ICC diagnostic is the tool that keeps you in the sensible middle. Broker and scheme effects in UK personal lines are genuinely material — the actuarial literature on broker effects in motor has documented this for decades, and the two-stage REML model quantifies it rigorously. Fleet effects are usually small because fleet policies in personal lines tend to be structured by vehicle count, not by a fixed organisation ID that persists across renewals. Postcode sector effects exist but require spatial borrowing, not independent random effects.

The decision about which group factors to model at Stage 2 should be driven by the ICC and the credibility weight distribution, not by the list of high-cardinality columns in your dataset. The ICC is available from `variance_decomposition()` in four lines of code. Run it first. The list of high-cardinality columns is not a model specification.

---

`insurance-multilevel` is open source under MIT at [github.com/burning-cost/insurance-multilevel](https://github.com/burning-cost/insurance-multilevel). Install with `pip install insurance-multilevel`. Requires Python 3.10+, CatBoost, Polars, and SciPy.

- [Your Broker Adjustments Are Guesswork](/2026/03/13/your-broker-adjustments-are-guesswork/) — the 2026 introduction to the two-stage CatBoost + REML architecture
- [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/) — the closed-form approach when you do not need a GBM Stage 1
- [Spatial Territory Ratemaking with BYM2](/2026/03/09/spatial-territory-ratemaking-bym2/) — geographic random effects with spatial borrowing, the right tool for postcode sector
