---
layout: post
title: "When Credibility Meets CatBoost: Choosing Between Classical and Modern Approaches"
date: 2026-02-28
categories: [pricing, libraries, guides]
tags: [credibility, buhlmann-straub, catboost, gbm, multilevel, random-effects, reml, insurance-credibility, insurance-multilevel, shap-relativities, uk-motor, scheme-pricing]
description: "Bühlmann-Straub vs CatBoost vs two-stage multilevel for UK motor pricing: when each wins and how insurance-credibility and insurance-multilevel combine them."
---

There is a version of this question that comes up at least once on every pricing project involving schemes, brokers, or MGA portfolios: do we apply credibility adjustments to the GBM output, or do we just feed the group identifiers into the model and let it figure it out?

The question sounds like it has a simple answer. It does not. The right approach depends on how many groups you have, how thin the data is, what your regulator expects, and whether you need to explain the group adjustment separately from the base rate. We have built separate libraries for the two ends of this spectrum and a third for the middle ground. This post explains when to use each.

---

## Classical Bühlmann-Straub: what it does well

The Bühlmann-Straub model (1970) solves a specific problem. You have a set of groups -- schemes, territories, NCD classes, fleet categories -- each with its own observed loss rate across several periods, and each with a different volume of data. You want to blend each group's observed experience with the portfolio mean, but the blend weight should reflect how credible that experience actually is.

The mathematics is clean. Three structural parameters govern everything:

- **mu**: the collective portfolio mean loss rate
- **v**: expected process variance within groups (noise)
- **a**: variance of the hypothetical means across groups (signal)

From these, Bühlmann's k = v/a gives you the noise-to-signal ratio. The credibility factor for group i is Z_i = w_i / (w_i + k), where w_i is total exposure. A large k means the portfolio is relatively homogeneous -- trust the collective. A small k means groups genuinely differ -- trust the individual experience.

The practical output is a credibility premium for each group: a weighted blend of that group's own loss rate and the portfolio mean, where the blend weight is derived entirely from the data.

Our [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility) library implements this as `BuhlmannStraub`, fitting structural parameters non-parametrically from a panel of (group, period) observations:

```python
import polars as pl
from insurance_credibility import BuhlmannStraub

# One row per (scheme, accident year) -- loss rate and earned exposure
scheme_panel = pl.DataFrame({
    "scheme":     ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    "year":       [2021, 2022, 2023] * 3,
    "loss_rate":  [0.55, 0.60, 0.58, 0.80, 0.75, 0.82, 0.40, 0.42, 0.38],
    "exposure":   [1000, 1200, 1100,  300,  350,  320, 5000, 4800, 5200],
})

bs = BuhlmannStraub()
bs.fit(scheme_panel, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")

summary = bs.summary()
# scheme  exposure  loss_rate_observed  credibility_weight  credibility_premium
# A       3300      0.578               0.76                0.561
# B       970       0.789               0.40                0.674
# C       15000     0.401               0.96                0.406
```

Scheme B has only 970 earned exposures across three years. Its observed loss rate of 78.9% is pulled toward the portfolio mean. Scheme C, with 15,000 exposures, earns 96% credibility and its premium barely moves. This is exactly the behaviour you want.

The limitations are equally clear. Bühlmann-Straub operates on a univariate loss rate. It cannot handle interactions between driver age and vehicle group. It has no view of the policy-level rating factors that drive the underlying risk -- it only sees aggregate loss experience at the group level. If Scheme B's bad experience is explained by a high proportion of young drivers, Bühlmann-Straub cannot see that. It will apply a partial credibility adjustment for what is actually a compositional difference.

This is the right tool for scheme-level adjustments when you have aggregate data and need an auditable, defensible blend with a clear mathematical interpretation. It is not the right tool for building the base rate structure.

---

## CatBoost for base rate structure: what it does well

The GBM, fitted at policy level on all rating factors, solves a different problem. Given driver age, vehicle group, postcode sector, NCD, and fifty other factors, it learns the non-linear, interacting structure of claim frequency and severity across the portfolio. CatBoost handles high-cardinality categoricals natively, which matters a great deal when vehicle group alone has 400 values and postcode sector has 2,800.

On a UK personal lines motor portfolio of 500,000 policies, a well-tuned CatBoost model typically outperforms a GLM with manually engineered interactions by 4-7 Gini points on holdout data. That gap is not noise. It represents genuine rating structure that the GLM's additive assumptions cannot capture.

The problem arises when you want to add group-level adjustments on top. Feeding `broker_id` or `scheme_id` directly into CatBoost only works when groups are large enough for the tree to learn stable effects. With 200 brokers and an average of 2,500 policies each, a high-cardinality categorical in CatBoost will produce unreliable leaves for the smaller brokers and will not naturally shrink toward the portfolio mean the way Bühlmann-Straub does. You can tune the regularisation parameters to get some shrinkage, but you cannot inspect the effective credibility weight given to each group, and you certainly cannot explain it to a regulator.

Where shap-relativities fits here: once you have a CatBoost base rate model, you still need to produce factor tables for your pricing committee and for Radar or Emblem. Our [`shap-relativities`](https://github.com/burning-cost/shap-relativities) library extracts exposure-weighted relativities in multiplicative format from any CatBoost model:

```python
from shap_relativities import SHAPRelativities

sr = SHAPRelativities(
    model=catboost_model,
    X=X_train,
    exposure=exposure,
    categorical_features=["driver_age_band", "vehicle_group", "ncd_years"],
)
sr.fit()
tables = sr.extract_relativities(
    base_levels={"driver_age_band": "30-39", "vehicle_group": 1, "ncd_years": 5},
)
# tables is a Polars DataFrame; filter per feature for individual tables
age_table = tables.filter(tables["feature"] == "driver_age_band")
```

The resulting relativities are what go into Radar. The CatBoost model is the source of truth; the factor tables are the governance-compatible representation of it. This workflow -- CatBoost in training, GLM-format tables in production -- is how the Gini gap actually gets into production.

---

## The hybrid: two-stage multilevel modelling

The right architecture for most UK motor scheme and MGA portfolios is neither pure credibility nor pure GBM. It is sequential: fit CatBoost on individual risk factors to learn the base rate structure, then apply REML-estimated random effects to capture group-level departures from that structure.

Our [`insurance-multilevel`](https://github.com/burning-cost/insurance-multilevel) library implements this as `MultilevelPricingModel`:

**Stage 1**: CatBoost is fitted on all features *excluding* group columns. The group exclusion is deliberate and critical. If `broker_id` is included in Stage 1, the GBM partially absorbs the group signal. Stage 2 then sees only the residual, underestimates the between-group variance tau2, and applies insufficient shrinkage. Excluding group columns from Stage 1 preserves identifiability.

**Stage 2**: Log-ratio residuals r_i = log(y_i / f_hat_i) are fitted with REML random intercepts per group. The BLUP for each group -- the Best Linear Unbiased Predictor -- is exactly a credibility-weighted adjustment: Z_g * (r_bar_g - mu_hat), where Z_g = tau2 / (tau2 + sigma2/n_g) is the Bühlmann credibility weight.

**Final premium**: f_hat(x) * exp(b_hat_group). Multiplicative structure throughout, compatible with standard UK personal lines rating.

```python
from insurance_multilevel import MultilevelPricingModel

model = MultilevelPricingModel(
    catboost_params={"iterations": 500, "loss_function": "Poisson", "depth": 6},
    random_effects=["broker_id", "scheme_id"],
    min_group_size=10,
)

model.fit(
    X_train,                          # all features including broker_id, scheme_id
    y_train["claim_count"],
    weights=y_train["exposure"],
    group_cols=["broker_id", "scheme_id"],
)

premiums = model.predict(X_test, group_cols=["broker_id", "scheme_id"])

# Inspect credibility weights per group
summary = model.credibility_summary()
# group_col   group_id   n_obs  tau2   sigma2   credibility_weight  blup_adjustment
# broker_id   BRK_007    4200   0.031  0.284    0.82                1.043
# broker_id   BRK_119    180    0.031  0.284    0.17                1.008
# scheme_id   SCH_003    8900   0.018  0.219    0.88               0.961
```

Broker 007 has 4,200 policies. Its 82% credibility weight means the BLUP adjustment is heavily anchored to its own experience -- a 4.3% uplift, probably a worse-than-average book. Broker 119 has only 180 policies: 17% credibility weight, its adjustment barely departs from zero. The shrinkage is automatic, principled, and inspectable.

This is the notation that mirrors Bühlmann-Straub deliberately. The tau2/sigma2/k/credibility_weight output from `credibility_summary()` maps directly to v/a/k/Z in the classical model. A pricing actuary who knows credibility theory can read this output without any new conceptual framework.

---

## When each approach wins

**Use Bühlmann-Straub (`insurance-credibility`) when:**
- You have aggregate data only -- loss ratios or burning cost by scheme and year, not policy-level data
- You need a fully auditable, closed-form calculation that a regulator or external auditor can verify by hand
- You are adjusting scheme-level rates on top of an existing base rate structure built elsewhere
- Groups number in the tens or low hundreds and data history stretches across at least three years per group

**Use CatBoost base rate plus shap-relativities when:**
- You have rich policy-level data and want to capture non-linear interactions across rating factors
- The group structure is simple enough that group effects are not the primary modelling challenge
- Your governance process requires factor tables for a rating engine
- The group identifiers (broker, scheme) have enough data each to trust the GBM's leaf estimates

**Use insurance-multilevel when:**
- You have policy-level data *and* a high-cardinality group structure (50+ brokers, 100+ schemes)
- Groups span a wide range of sizes -- some with 10,000 policies, others with 50 -- and you need appropriate shrinkage across that range
- You want the base rate and the group adjustment to be modelled and governable separately
- The group adjustment needs to be explainable in credibility terms: "Scheme X has 76% credibility and is running 8% adverse relative to its expected risk mix"

The diagnostic you want from insurance-multilevel is the ICC -- the intraclass correlation coefficient -- which tells you what fraction of total premium variance sits between groups rather than within them. If ICC is below 2%, the group structure is probably not worth the additional complexity. If it is above 10%, treating groups as random effects will materially improve both accuracy and stability.

```python
from insurance_multilevel import diagnostics

vc = model._variance_components["broker_id"]
icc = diagnostics.icc(vc, group_col="broker_id")
# 0.087 -- 8.7% of variance between brokers, worth modelling separately
```

---

## The actual workflow for UK motor

For a UK personal lines motor portfolio with direct and broker channels mixed together, the approach we use is:

1. Fit CatBoost on individual risk factors (driver, vehicle, geography, NCD) excluding any channel or group identifiers. Extract factor tables via shap-relativities for the pricing committee. This is the base rate model.

2. On the residuals from Step 1, fit Bühlmann-Straub at scheme level if you only have aggregate scheme data. Or fit insurance-multilevel at broker and scheme level if you have policy-level data with group identifiers. The BLUP outputs become the scheme and broker loading tables.

3. The final premium is base rate (from the CatBoost factor tables) multiplied by the scheme loading multiplied by the broker loading. Each component is separately auditable and separately governable.

Step 3 is what makes this architecture work in practice. The base rate review and the scheme rate review happen on different cycles and involve different stakeholders. Separating them structurally -- rather than baking everything into one model that nobody can decompose -- is what allows the pricing function to actually manage the portfolio.

The credibility approach did not disappear when GBMs arrived. It got a better Stage 1.

---

[`insurance-credibility`](https://github.com/burning-cost/insurance-credibility) | [`insurance-multilevel`](https://github.com/burning-cost/insurance-multilevel) | [`shap-relativities`](https://github.com/burning-cost/shap-relativities)

---

## Related articles

- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/)
- [Bayesian Hierarchical Models for Thin-Data Pricing](/2026/02/17/bayesian-hierarchical-models-for-thin-data-pricing/)
- [When Group Effects Are Worth Modelling: ICC Diagnostics for High-Cardinality Factors](/2026/03/06/multilevel-group-factors/)
