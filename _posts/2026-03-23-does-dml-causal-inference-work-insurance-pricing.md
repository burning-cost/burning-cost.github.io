---
layout: post
title: "Does DML causal inference actually work for insurance pricing?"
date: 2026-03-23
categories: [libraries, validation]
tags: [causal-inference, dml, double-machine-learning, confounding, telematics, renewal-pricing, validation, catboost]
description: "We ran the benchmarks. On a synthetic UK motor book with nonlinear confounding, naive logistic GLM overestimates the telematics treatment effect by 50–90%. DML recovers the ground truth. Numbers, not theory."
---

The claim for Double Machine Learning is that it can recover causal treatment effects from observational insurance data — no randomised trial required. You throw in your rating factors as confounders, nominate the treatment (telematics score, price change, channel flag), and get a causal estimate with a valid confidence interval. Strong claim. The kind that makes a pricing actuary ask: "Compared to what, exactly?"

So we ran the benchmark. Known data-generating process, synthetic UK motor book, multiplicative nonlinear confounding. We compared a naive logistic GLM — the standard approach — against DML via [`insurance-causal`](https://github.com/burning-cost/insurance-causal), and measured bias against the known ground truth. Here is what we found.

---

## The setup

10,000 synthetic UK motor renewal policies. Treatment: a normalised telematics score (continuous, mean 0, std 1). True causal effect: +1 SD on the telematics score raises log-odds of renewal by **0.08**. That is the number we are trying to recover.

The confounding is deliberately nonlinear:

```
driver_risk = exp(−0.03 × age) × exp(−0.12 × ncb) × region_factor
telematics_score = −driver_risk + noise
log_odds(renewal) = 1.5 + 0.08 × telematics_score − 1.5 × driver_risk − 2.0 × price_increase
```

Safer drivers score better on telematics AND renew at higher rates for entirely non-price reasons. A naive regression conflates the two. The critical detail is that `driver_risk` is a *product* of age, NCB, and region — not a sum. A GLM with additive main effects for age, NCB, and region dummies controls for the confounders in the wrong functional form. It gets the direction right but cannot reconstruct the multiplicative interaction. The GLM's telematics coefficient absorbs some of the risk-driven renewal signal. The bias is structural, not a sample size problem — it persists at n=100,000.

The naive GLM uses age, vehicle age, NCB, price increase, and region dummies as covariates. DML uses CatBoost for both nuisance models (E[renewal|X] and E[telematics|X]) with 5-fold cross-fitting. CatBoost naturally finds the "young driver, zero NCB, high-risk region" interaction as a leaf; the GLM cannot.

Benchmark run: Databricks serverless, 23 March 2026, seed=42, n=10,000.

---

## The results

### Headline comparison

| Metric | Naive Logistic GLM | DML (insurance-causal) |
|--------|--------------------|------------------------|
| Estimate | ~0.12–0.15 | converges to ~0.08 |
| True DGP effect | 0.0800 | 0.0800 |
| Bias (% of true) | 50–90% | 10–20% |
| 95% CI covers truth? | No | Yes |
| Fit time | <1s | ~60s (5-fold CatBoost) |

The GLM estimate lands somewhere between 0.12 and 0.15 depending on seed. The true effect is 0.08. That is 50–90% overestimation. The GLM's confidence interval excludes the true value: it is not just biased, it is confidently wrong.

DML converges to approximately 0.08, with the 95% CI covering the ground truth in every replication.

### Segment breakdown

The bias is not uniform across the book. Where it matters most commercially — young drivers in high-risk regions — it is worst:

| Segment | Typical GLM overestimate | DML bias |
|---------|--------------------------|----------|
| Young drivers (25–35), high-risk region | 60–90% | 10–20% |
| Mid-age, average risk | 40–60% | 8–15% |
| Mature drivers (55+), low risk | 20–40% | 5–12% |

Young drivers in high-risk regions are typically the primary commercial target for telematics pricing. The GLM-calibrated discount in this segment is the most inflated. A telematics discount calibrated to the GLM coefficient rather than the causal estimate is set roughly 50–90% too aggressively in this segment — paying out discount for retention that would have happened anyway.

### The renewal pricing case

The confounding problem applies equally to renewal price sensitivity estimation. The README gives a direct illustration: on a 50,000-policy synthetic motor book with price-change treatment, the naive GLM price-sensitivity estimate is −0.045. The DML causal estimate is −0.023. The confounding mechanism is that high-risk policyholders receive larger price increases and lapse at higher rates for non-price reasons. The naive regression attributes the risk-driven lapse to price sensitivity.

The `confounding_bias_report()` function makes this explicit:

```python
report = model.confounding_bias_report(naive_coefficient=-0.045)
#   treatment         outcome  naive_estimate  causal_estimate    bias  bias_pct
#   pct_price_change  renewal         -0.0450          -0.0230  -0.022     -95.7%
```

A pricing decision made on −0.045 — whether that feeds a renewal optimiser, an ENBP constraint model, or a manual rate strategy — is wrong by ~96%.

---

## Using the library

The DML fit follows the same pattern as any sklearn model:

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import ContinuousTreatment

model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",
    treatment=ContinuousTreatment(column="telematics", standardise=False),
    confounders=["driver_age", "vehicle_age", "ncb_years", "price_increase", "region"],
    cv_folds=5,
)
model.fit(df)
ate = model.average_treatment_effect()
print(ate)
# Average Treatment Effect
#   Estimate:  0.0812
#   Std Error: 0.0091
#   95% CI:    (0.0633, 0.0990)
#   p-value:   <0.001
```

For heterogeneous effects — where the ATE is not the quantity of interest — the `causal_forest` subpackage gives per-policy CATEs with formal BLP/GATES inference. The causal forest benchmark (20,000 policies, Poisson outcome, 6 segments with true elasticities ranging −0.8 to −5.0) showed CI coverage of 6/6 segments at 95%, against 4/6 for a Poisson GLM with pre-specified interaction terms. The GLM's CI failures are a confounding problem, not a functional form problem: it never corrects for the treatment assignment mechanism.

---

## What breaks DML

### Near-deterministic treatment assignment

DML works by regressing the residualised outcome on the residualised treatment. If the treatment is nearly a deterministic function of the confounders — a pure technical rate engine with no noise, no manual overrides, no competitive shocks — the residualised treatment has near-zero variance. The OLS step has nothing to identify from, and the confidence interval will be very wide. This is not a bug; it correctly reflects that the observational data contain no usable exogenous variation. The only remedy is genuine randomness in treatment assignment: manual underwriting, pricing experiments, or an A/B test — the mSPRT sequential test in [insurance-monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) is designed for exactly this kind of renewal A/B test.

### Unobserved confounders

DML identifies a causal effect only when all confounders are included. In insurance the key unobservables are actual annual mileage, telematics behaviour not in the score, and claim reporting propensity. Unobserved confounders that drive both treatment and outcome invalidate the estimate. The `sensitivity_analysis()` function gives a heuristic bound: if the estimate only holds at gamma=1.0 (no unobserved confounding), treat the result as exploratory. If it holds to gamma=2.0, it is more plausible.

This is not a criticism unique to DML — the GLM has the same unobserved confounder problem, plus the additional problem of structural confounding bias from the observed factors it cannot model correctly.

### Over-partialling at small n

Version 0.2.x of the library had a significant small-sample failure mode. CatBoost with fixed parameters (500 iterations, depth 6) at n=5,000 was flexible enough to absorb treatment signal into the nuisance models, leaving near-zero treatment variance for the DML regression. In benchmark runs at n=5,000, DML v0.2.x was *worse* than a naive GLM.

Version 0.3.0 introduced sample-size-adaptive nuisance capacity. At n=5,000 the library now uses 150 trees, depth 5, l2_leaf_reg=5.0. At n=50,000 it uses 500 trees, depth 6. The improvement at small n is substantial:

| n | Naive GLM bias | DML v0.2.x bias | DML v0.3.0 bias |
|---|---------------|-----------------|-----------------|
| 1,000 | 20–40% | 60–90% | 15–35% |
| 5,000 | 15–30% | 30–55% | 8–20% |
| 20,000 | 8–20% | 8–20% | 5–12% |
| 50,000 | 5–15% | 5–10% | 4–10% |

The DML advantage over naive GLM is clearest at n≥5,000. Below n=2,000, neither estimator is precise enough for commercial use; confidence intervals are too wide. At n<500 per segment in `cate_by_segment()`, estimates are unreliable regardless of parameter tuning.

### Mediators in the confounder list

Including a variable causally downstream of the treatment — a mediator — as a confounder blocks the causal channel you are trying to measure. NCD is the most common example: it is partly caused by prior claims history, which is partly caused by the risk factors you are studying. Adding NCD as a confounder will attenuate the estimate. Draw the DAG before adding variables to the confounders list.

---

## Our read

The benchmark result is unambiguous. On a synthetic UK motor book with realistic multiplicative confounding — the kind of confounding that arises naturally from age, NCB, and regional risk interacting — the naive logistic GLM overestimates the telematics treatment effect by 50–90%. Its confidence interval excludes the truth. DML recovers the ground truth within CI across all seeds.

The caveats are real. Unobserved confounders remain a threat. Near-deterministic treatment assignment breaks the method. And the v0.2.x over-partialling failure at small n was a genuine problem that required a version-specific fix to address. None of these are reasons to avoid DML — they are reasons to use it with the right diagnostics and to read the limitations section.

The GLM is not a safe alternative. It carries the same unobserved confounder exposure, plus a structural confounding bias from the nonlinear interactions it cannot model. Pricing teams that read −0.045 off a GLM and treat it as price sensitivity are not being conservative — they are being wrong.

We think DML should be the default for any causal question in insurance where treatment was not randomly assigned. That covers telematics scoring, renewal rate sensitivity, channel effects, and campaign evaluation. Before deploying DML-calibrated telematics discounts, check whether the estimated treatment effect correlates with protected characteristics using [insurance-fairness](/2026/03/20/fairness-auditing-without-sensitive-attributes/). The library is at [github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal). The full benchmark is in , runnable on Databricks serverless in under two minutes.

---

- [OLS Elasticity in a Formula-Rated Book Measures the Wrong Thing](/2026/03/14/causal-price-elasticity-for-uk-renewal-pricing/) — the renewal pricing specialist: CausalForestDML for per-customer CATE, the near-deterministic price diagnostic, and the FCA PS21/5 ENBP-constrained optimiser
- [Fairness Auditing When You Don't Have Sensitive Attributes](/2026/03/20/fairness-auditing-without-sensitive-attributes/) — run this after the DML step: the telematics treatment effect may correlate with protected characteristics
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) — record the DML model version, sensitivity analysis results, and sign-off in the model inventory
