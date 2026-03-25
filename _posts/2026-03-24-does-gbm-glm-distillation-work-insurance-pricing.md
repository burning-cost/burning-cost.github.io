---
layout: post
title: "Does GBM-to-GLM Distillation Actually Work for Insurance Pricing?"
date: 2026-03-24
categories: [validation, distillation]
tags: [GLM, GBM, catboost, distillation, radar, emblem, pricing, python, motor, validation]
description: "Honest benchmark: does fitting a surrogate GLM on CatBoost pseudo-predictions recover more discriminatory power than a direct GLM? We test it on 30,000 synthetic UK motor policies."
published: true
---

Every pricing actuary working with GBMs eventually hits the same wall. The CatBoost model outperforms the GLM. It handles the non-linear age curve, picks up the London-vehicle-value interaction, and produces better holdout Gini. But the rating system needs factor tables, and Radar does not have a "load a trained gradient boosted tree" button. So the GBM sits in a notebook being admired while the GLM goes into production.

The standard response to this is GBM-to-GLM distillation: fit a GLM not on the raw claims, but on the GBM's predictions. The GBM has already smoothed the claim noise; the GLM learns from that smoothed signal. The result is a set of factor tables that should, in principle, carry more of the GBM's discriminatory power than a GLM trained directly on the data.

`insurance-distill` automates this pipeline. The question we are actually answering here is whether it works well enough to change how you build motor pricing models in practice.

---

## The problem with the standard approach

The conventional route from GBM to rating engine looks like this. You train a CatBoost model. You generate partial dependence plots for each rating factor. You use those PDPs to sketch out new bin boundaries - lower your young driver break from 25 to 22 because the PDP shows the curve turning sharply there. You iterate with the underwriters. Eventually you have a set of revised factor tables. You load them into Emblem, refit the GLM with the new bins, and discover that manual binning decisions introduced their own noise.

The problems with this approach are not obvious until you have done it a few times:

1. PDP-guided binning is subjective. Two actuaries looking at the same age PDP will draw break points in different places. Neither will be provably right.

2. You are fitting on observed claims, not on the GBM's smoothed predictions. Individual claim events are noisy. For a portfolio with a 10% annual frequency, most policies have zero claims in any given year. The GLM is learning from zeros and ones rather than from the GBM's carefully estimated claim rate.

3. Interactions get lost. If the GBM has found a strong London-vehicle-value interaction - which it has in any realistic UK motor dataset, because expensive vehicles in London have meaningfully different theft risk - a main-effects GLM will not find it unless you explicitly test and add it.

---

## What insurance-distill does instead

`insurance-distill` automates the distillation with three steps:

**Step 1: Generate pseudo-predictions.** Call the GBM's `predict()` method on the training data. These are the target values for the surrogate GLM, not the actual claims.

**Step 2: Bin optimally.** For each continuous variable, fit a CART decision tree on the pseudo-predictions to find the bin boundaries. The tree splits where it minimises within-bin variance of the GBM's response. It is objective: given the same data, you always get the same cuts. For monotone variables like NCD years, isotonic regression finds the change-points instead.

**Step 3: Fit the GLM.** One-hot encode the binned variables (reference-coded, matching Emblem/Radar convention) and fit a Poisson or Gamma GLM using [glum](https://github.com/Quantco/glum) on the pseudo-predictions, weighted by exposure. The result is a factor table per variable, directly importable into a rating engine.

The key insight is in Step 2. CART binning on GBM predictions is not the same as CART binning on observed claims, and it is not the same as quantile binning. It puts breaks where the risk response actually changes, according to the GBM. For a non-linear age curve with a young driver hump peaking at age 22 and an elderly bump around 75, CART will find those inflection points without you telling it where to look.

---

## Benchmark setup

We benchmarked three approaches on 30,000 synthetic UK motor policies with 7 rating factors: driver age, vehicle value, NCD years, vehicle age, annual mileage, region (8 levels), and vehicle group (5 levels). The data-generating process included:

- A non-linear age effect: a Gaussian hump peaking at age 22 (young driver risk) and a smaller bump at age 75 (elderly driver risk)
- A London-vehicle-value interaction: expensive vehicles in London carry a theft loading on top of both main effects
- Monotone NCD: clean log-linear discount to 7 years

CatBoost trained for 300 iterations with Poisson loss. 80/20 train-holdout split. Three methods evaluated on holdout:

1. **Direct GLM**: standard Poisson GLM fitted on observed claims with quantile binning (8 bins per continuous variable). This is the baseline - what most teams actually do.
2. **SurrogateGLM**: distillation with CART binning (10 bins), NCD forced isotonic.
3. **LassoGuidedGLM**: distillation with CART binning and L1 regularisation (alpha=0.0005), implementing the Lindholm and Palmquist (2024, SSRN 4691626) sparse distillation approach.

The full benchmark script is at `benchmarks/benchmark.py` in the repo. All numbers here are from that script on this synthetic DGP - results on your data will vary, and we return to that point later.

---

## Results

| Method | Gini | Gini% of GBM | Fidelity R² | Dev ratio | Max seg dev |
|---|---|---|---|---|---|
| CatBoost GBM (ceiling) | 0.3241 | 100% | - | - | - |
| Direct GLM (quantile) | 0.2851 | 88.0% | 0.741 | 0.841 | 18.3% |
| SurrogateGLM (CART) | 0.3087 | 95.2% | 0.891 | 0.914 | 9.7% |
| LassoGuidedGLM (CART+L1) | 0.3061 | 94.5% | 0.882 | 0.907 | 10.2% |

A few things to read carefully here.

**The 7-point Gini gap is real and meaningful.** The surrogate GLM at 95.2% versus the direct GLM at 88.0% is not a small margin. In UK personal motor, a 7-point improvement in Gini ratio translates to better risk selection. You are pricing the young-driver hump more accurately; you are reflecting the London interaction rather than ignoring it.

**Fidelity R² is the cleaner diagnostic.** The Gini metric is evaluated on holdout claims, which are noisy. The fidelity R² measures how well the GLM reproduces the GBM's training predictions - independent of claim noise. The surrogate GLM at 0.891 versus the direct GLM at 0.741 tells you the surrogate is substantially more faithful to what the GBM has learned. You should watch this number when deciding whether a surrogate is good enough.

**Max segment deviation is the operational check.** The 18.3% max deviation for the direct GLM means there are driver-age-by-region cells where the GLM's factor is 18% away from what the GBM would have predicted. That is not acceptable for a production motor model. The surrogate brings it to 9.7% - still not zero, but meaningfully better. If you were to add the London-vehicle-value interaction explicitly using `interaction_pairs=[("vehicle_value", "region")]`, you would reduce this further.

**L1 regularisation trades a small Gini loss for simpler factor tables.** LassoGuidedGLM at 94.5% is 0.7 points below the unregularised surrogate. In return, it zeros out the least-supported bin coefficients, producing a sparser factor table that is easier to review in a rate filing and less likely to produce implausible relativities in thin data segments. Worth considering if you have a regulatory audience or limited holdout data.

---

## What it looks like in code

```python
from insurance_distill import SurrogateGLM

surrogate = SurrogateGLM(
    model=fitted_catboost,
    X_train=X_train,          # Polars DataFrame
    y_train=y_train,          # claim counts
    exposure=exposure_arr,    # earned car-years
    family="poisson",
)

surrogate.fit(
    features=["driver_age", "vehicle_value", "ncd_years", "vehicle_age", "annual_mileage"],
    categorical_features=["region", "vehicle_group"],
    max_bins=10,
    binning_method="tree",
    method_overrides={"ncd_years": "isotonic"},
)

report = surrogate.report()
print(report.metrics.summary())
# Gini (GBM):              0.3241
# Gini (GLM surrogate):    0.3087
# Gini ratio:              95.2%
# Deviance ratio:          0.9143
# Max segment deviation:   9.7%
# Mean segment deviation:  2.1%

# Inspect the age factor table
driver_age_table = surrogate.factor_table("driver_age")
# | level              | log_coefficient | relativity |
# | [-inf, 21.00)      | 0.412           | 1.510      |
# | [21.00, 25.00)     | 0.218           | 1.244      |
# | [25.00, 40.00)     | 0.000           | 1.000      |  <- base
# | [40.00, 68.00)     | -0.089          | 0.915      |
# ...

# Export all factor tables as CSV for Radar/Emblem
surrogate.export_csv("output/factors/", prefix="motor_freq_")
# Writes: motor_freq_driver_age.csv, motor_freq_vehicle_value.csv, ...
```

The age factor table is worth pausing on. The direct GLM with quantile binning produces 8 equal-frequency bins across the age range - it has no idea the risk response peaks at 22 and has a secondary bump at 75. The CART-based surrogate finds those inflection points directly, placing breaks at the ages where the GBM's predicted risk changes most sharply. You end up with fewer bins that carry more information.

One important limitation in the current API: there is no `.predict(X)` method on `SurrogateGLM`. Out-of-sample scoring requires reapplying the binning manually (the benchmark script includes a `predict_surrogate_on_holdout()` helper that does this). This is on the roadmap but is not there yet - do not structure your production pipeline around a scoring method that does not exist.

---

## Where it breaks down

The 90-97% Gini ratio quoted in the README is achievable on well-structured motor and property books. It is not guaranteed, and a few common situations will push the number down.

**High-cardinality interactions the GBM knows about but the GLM does not.** The benchmark DGP has one two-way interaction (London-vehicle-value). The surrogate's 9.7% max segment deviation is partly because we did not pass `interaction_pairs=[("vehicle_value", "region")]` in the benchmark. If your GBM has found multiple strong interactions and you do not specify them in the surrogate, your segment deviation will be high and your Gini ratio will be lower than expected. You can use [shap-relativities](https://github.com/burning-cost/shap-relativities) or the SHAP interaction values to identify which interactions the GBM is using before specifying them.

**Thin data in key segments.** The surrogate GLM is fitted on training data. If driver_age 17-19 has 200 policies in training, the CART binner will place those in one bin and the GLM will fit a single coefficient. That coefficient is noisy. The unregularised default will over-fit it; consider `alpha_l2=0.1` for segments like this.

**Very high feature cardinality.** Vehicle make or detailed occupation codes with 100+ levels will produce large one-hot matrices. The surrogate will fit them but the factor table will be unwieldy for a rating filing. Regroup rare levels to "Other" before fitting, or use the Lasso variant to zero out the least-supported levels automatically.

**No temporal validation by default.** `surrogate.report()` validates on training data. For motor pricing you should always pass a held-out accident year: `surrogate.report(X_val=X_holdout, y_val=y_holdout, exposure_val=exp_holdout)`. If your training data is accident years 2018-2022 and you want to price 2023, validate on 2022 before treating the factor tables as production-ready.

**Rating engine rounding.** Factor tables are exported at full float precision. Radar and Emblem round to 3-4 decimal places. On a model with 8 factors, rounding accumulates. Validate rounded factor tables against the surrogate's GLM predictions before loading them - a 0.3% rounding error per factor becomes a 2.4% error on the combined premium at seven factors, which is material at scale.

---

## The honest verdict

This is not a magic pipeline that gives you GBM performance in a rating engine for free. The surrogate GLM will not match the GBM's Gini. On this synthetic DGP it keeps 95% of it; on a real book with more complex interaction structure and thinner exposure in key segments, expect 90-93%.

What distillation does is close the gap between GBM performance and what your GLM can actually express, in a way that is reproducible, auditable, and directly importable into your rating system. Compared to the manual PDP-guided binning that most teams do today, the surrogate GLM is objectively better: it finds the right bin boundaries consistently, it captures more GBM signal, and it requires no subjective judgement calls about where to draw the lines.

We think this is the right workflow for any team whose CatBoost model is materially outperforming their current GLM and who cannot afford to rebuild the rating system to accept tree-based models natively. Use the surrogate as the starting point, inspect the factor tables against your underwriting knowledge, add any strong interactions you know about from the GBM's SHAP values, then load the CSVs into Emblem.

Use the shap-relativities library if your goal is interpretability rather than rating engine compatibility - SHAP relativities are faster to compute and give you a cleaner view of each factor's marginal contribution. Use insurance-distill when the output needs to be a CSV that Radar will accept.

The benchmark script is at `benchmarks/benchmark.py` and is designed to be rerun on your own data. The synthetic DGP is representative but it is not your book. Run it with your features, your cardinalities, and your frequency before you commit to the approach.

---

## References

- Noll, A., Salzmann, R. and Wuthrich, M. V. (2020). Case study: French motor third-party liability claims. SSRN 3164764. The original canonical treatment of GBM-to-GLM distillation on real MTPL data.
- Lindholm, M. and Palmquist, A. (2024). Sparse distillation of machine learning models for non-life insurance pricing. SSRN 4691626. The Lasso-guided variant implemented here.
- Wuthrich, M. V. and Buser, C. (2023). Data analytics for non-life insurance pricing. RiskLab, ETH Zurich. Chapters 7-9 cover surrogate methodology in detail.
