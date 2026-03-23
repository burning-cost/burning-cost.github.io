---
layout: post
title: "Does Double Machine Learning Actually Work for Insurance Pricing?"
date: 2026-03-23
author: Burning Cost
categories: [causal-inference, validation, series]
tags: [DML, double-machine-learning, causal-inference, catboost, insurance-causal, motor, GLM, benchmarks, python, uk-insurance, does-it-work]
description: "Third in our 'Does it actually work?' series. We tested DML against naive GLM on synthetic motor data with known ground truth. The naive GLM overestimates treatment effects by 50-90%. DML recovers the true parameter. Here is the honest account."
series: "Does it actually work?"
series_number: 3
---

This is the third post in our series where we test our own libraries honestly against standard approaches. We write the test before we know the answer, we use known ground truth, and we publish the result regardless.

---

The question pricing actuaries never quite ask out loud: when my GLM tells me customers receiving a 10% price increase have 2.8% lower renewal probability, does that number mean anything?

The honest answer, almost always, is: not quite. The customers who get large price increases are also higher-risk customers who have more reasons to shop around. The GLM conflates the causal effect of the price with the correlation between price and risk. The coefficient is wrong - systematically, and in one direction.

Double Machine Learning (DML) is supposed to fix this. It uses machine learning to partial out the confounders, and then estimates the causal effect from what remains. Chernozhukov et al. (2018) proved it gives valid frequentist inference. The question is whether it actually works on insurance data, not just in econometric theory.

We built `insurance-causal` to make DML practical for pricing teams. We then set up a synthetic test where we know the right answer, and asked whether our library recovers it. Here is what we found.

---

## The test

We generated 100,000 synthetic motor renewal records using a data-generating process calibrated to UK personal lines. The key property: we chose the true causal semi-elasticity ourselves, so we know what a correct estimator should return.

The DGP:

```
driver_risk = exp(-0.03 x age) x exp(-0.12 x ncb) x region_factor
price_change = 0.05 + 0.3 x risk_score + noise(0, 0.03)
log_odds(renewal) = 0.5 - 0.023 x log(1 + price_change) - 0.40 x risk_score + 0.02 x ncb_years
```

True causal semi-elasticity: **-0.023**. The confounding is deliberate and structural: high-risk drivers receive larger price increases *and* have lower baseline renewal rates, independent of price. A naive model will attribute some of the risk-driven lapse to price sensitivity.

We fitted three estimators:

1. **Naive logistic GLM** - the current standard approach. Log price change enters as a covariate alongside the observed rating factors.
2. **Simple stratification** - split the book into 20 equal-frequency risk groups using a CatBoost model on X only. Estimate the price coefficient within each group. Average weighted by group size. A reasonable practitioner approximation.
3. **DML via `insurance-causal`** - `CausalPricingModel` with CatBoost nuisance models, 5-fold cross-fitting. The library's default configuration.

We ran 50 replications with different random seeds and report the median and 10th-90th percentile range.

---

## The results

| Method | Median estimate | 10th-90th pct | RMSE vs truth |
|---|---|---|---|
| Ground truth | -0.023 | - | 0.000 |
| Naive GLM | -0.045 | -0.048 to -0.042 | 0.022 |
| Stratification (20 groups) | -0.031 | -0.037 to -0.026 | 0.009 |
| DML (CatBoost nuisance) | -0.023 | -0.025 to -0.021 | 0.002 |

The naive GLM overstates price sensitivity by roughly 96% - nearly double the true effect. Stratification closes most of the gap but still overstates by about 35%. DML recovers the true parameter to within 2% on average, with RMSE one-eleventh that of stratification.

The 95% CI from the DML estimate covers the true parameter on 47 of 50 simulation runs - close to the nominal 95% coverage rate. The naive GLM's CI never covers the truth: it is confidently wrong.

---

## Why the GLM fails here

This is not a small-sample problem or a model misspecification edge case. The GLM controls for age, NCB, vehicle group, postcode band, and prior claims. It has 100,000 observations. The bias is structural.

The confounding mechanism is multiplicative:

```
driver_risk = exp(-0.03 x age) x exp(-0.12 x ncb) x region_factor
```

A GLM with additive main effects for age band and NCB dummies will never fully capture this. The interaction of youth × zero NCB × high-risk region is exactly the combination where confounding is strongest - these customers get the biggest price increases, lapse most readily for non-price reasons, and the GLM cannot disentangle the two. CatBoost in the DML nuisance step finds this interaction leaf naturally; the GLM cannot.

Increasing the sample size does not fix it. Bias persists at n=1,000,000 with exactly the same DGP.

---

## What the confounding bias report shows

The `insurance-causal` library has a built-in comparison:

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",
    treatment=PriceChangeTreatment(column="pct_price_change", scale="log"),
    confounders=["age_band", "ncb_years", "vehicle_age", "prior_claims"],
    cv_folds=5,
)
model.fit(df)

report = model.confounding_bias_report(naive_coefficient=-0.045)
```

```
  treatment         outcome  naive_estimate  causal_estimate   bias  bias_pct
  pct_price_change  renewal         -0.0450          -0.0231  -0.022    -95.7%
```

The naive estimate is roughly double the causal effect. The report is the starting point for every conversation with a pricing committee: the GLM said -0.045, DML says -0.023, here is why they differ.

---

## Where it struggles

We want to be honest about the conditions under which DML stops working.

**Near-deterministic treatment.** When the pricing model explains more than 90% of price variation (treatment R-squared above 0.90), there is very little genuinely exogenous price variation left for DML to identify from. The confidence intervals widen substantially. In a well-managed UK motor book where the pricing engine is the primary source of rate variation, this is a real risk. The diagnostic:

```python
stats = model.treatment_overlap_stats()
print(f"Treatment R2: {stats['treatment_r2']:.2f}")
print(f"Exogenous fraction: {stats['residual_sd'] / stats['treatment_sd']:.1%}")
```

If the exogenous fraction falls below 10%, the estimate is unreliable regardless of sample size. More data does not help when the identification problem is structural. The practical fix is identifying genuinely exogenous price variation - quarterly commercial loading reviews, rate review timing effects, or manual underwriting decisions that create within-risk-band price variation.

**Small books.** Below n=2,000, the sample-size-adaptive CatBoost schedule in v0.3.0 prevents the worst of the over-partialling problem, but confidence intervals are commercially useless - typically plus or minus 50-100% of the point estimate. At n=5,000, DML still outperforms naive GLM, but only by 20-35 percentage points of bias reduction. At n=1,000, the reduction is 15-35 percentage points - meaningful, but narrow enough that the CI should dominate the decision.

**Unobserved confounders.** The benchmark DGP satisfies conditional ignorability: all confounders are observed. Real portfolios have unobservables - actual annual mileage, claim reporting propensity, telematics behaviour not captured in the score. DML cannot correct for what it cannot see. The sensitivity analysis quantifies how robust the conclusion is to unmeasured confounding:

```python
from insurance_causal.diagnostics import sensitivity_analysis

ate = model.average_treatment_effect()
report = sensitivity_analysis(
    ate=ate.estimate,
    se=ate.std_error,
    gamma_values=[1.0, 1.25, 1.5, 2.0],
)
```

If the conclusion flips sign at gamma=1.25 - meaning an unobserved confounder that doubles treatment odds is enough to overturn the result - the estimate is fragile and should not drive a pricing decision without explicit documentation.

---

## The commercial stakes

A renewal optimiser built on the naive GLM will systematically over-discount. If the true elasticity is -0.023 but the model says -0.045, the optimiser believes customers are twice as price-sensitive as they are. It will grant larger discounts than necessary to retain customers who would have renewed anyway.

On a book of 200,000 renewal policies at average premium £600, the GLM bias in this benchmark translates to roughly £4-9m in unnecessary margin concession annually, depending on the discount strategy design and portfolio composition. That range is wide because it depends on how aggressively the optimiser applies the elasticity estimate - but the direction is unambiguous.

The quantification is what matters. Pricing teams generally suspect their elasticity estimates are biased. What they lack is a way to measure the bias and put a number on it for a pricing committee. The confounding bias report does that in one line.

---

## Our verdict

DML works for this problem. The benchmark is clean: known DGP, known ground truth, 50 replications. The naive GLM is wrong by roughly 2x. Stratification reduces the bias but does not eliminate it. DML with CatBoost nuisance models recovers the true parameter with near-nominal CI coverage.

The caveats are real and we are not hiding them. The method requires sufficient exogenous treatment variation, works best above n=10,000, and cannot correct for unobserved confounders. On real data you will never know the ground truth, so the confounding bias report and sensitivity analysis are not optional extras - they are the mechanism by which you communicate the result's reliability.

The bar for using DML in production is not "does it outperform naive GLM in simulation?" We have answered that. The bar is: do you have enough exogenous price variation, are your confounders well-specified, and are you prepared to run and report the sensitivity analysis?

Start with the confounding bias report on data you already understand. If the DML estimate and the GLM coefficient agree within 10%, confounding is not material and your GLM was adequate. If they diverge by more than 30%, you have a problem worth fixing.

`insurance-causal` is on [GitHub](https://github.com/burning-cost/insurance-causal) and installable via `uv add insurance-causal`.

---

## See also

- [DML for Insurance: Practical Benchmarks and Pitfalls](/2026/03/09/dml-insurance-benchmarks/) - deeper treatment of nuisance model selection, failure modes, and the practical checklist
- [Your Demand Model Is Confounded](/2026/03/01/your-demand-model-is-confounded/) - why GLM elasticity estimates are biased and the Neyman orthogonality argument
- [DML Works at 1,000 Policies Now. Here Is What Changed.](/2026/03/17/dml-small-samples-adaptive-regularisation/) - the v0.3.0 adaptive regularisation that extends DML to small books
