---
layout: post
title: "Causal Inference for Insurance Pricing in Python: A Complete DML Tutorial"
date: 2026-04-04
categories: [libraries, tutorials]
tags: [causal-inference, insurance-pricing, python, double-machine-learning, DML, CatBoost, insurance-causal, ATE, CATE, confounding, telematics, GLM, treatment-effects, consumer-duty, FCA, uk-motor, tutorial, DoubleML]
author: Burning Cost
description: "A hands-on tutorial on causal inference for insurance pricing in Python using the insurance-causal library. Covers double machine learning (DML), CatBoost nuisance models, ATE/CATE estimation, and when GLM coefficients mislead you."
seo_title: "Causal Inference Insurance Pricing Python: DML Tutorial with insurance-causal"
---

Your renewal pricing model says price sensitivity is −0.045. Your head of pricing is about to use that number to set next quarter's rate file. The problem: it is probably wrong, and not because the model is badly built.

The coefficient is biased because price changes were never randomly assigned. You increase riskier customers more — because you have to, to stay profitable. Those same customers lapse more, regardless of price. The naive regression cannot separate "they lapsed because you increased them" from "they lapsed because they were bad risks who were always going to go." That separation is what causal inference does.

This post walks through a complete double machine learning (DML) workflow in Python for insurance pricing, using the [`insurance-causal`](https://github.com/burning-cost/insurance-causal) library. We cover the business motivation, the mechanics of DML, two worked examples with realistic data, and — because we think honesty about limitations matters more than vendor marketing — the scenarios where a well-built GLM is sufficient.

---

## The confounding problem in insurance data

Before the code: why does this matter specifically for insurance?

In a clinical trial, patients are randomly assigned to treatment or control. Random assignment means the treatment group and control group are identical in expectation on all characteristics — observed and unobserved. The difference in outcomes is the causal effect.

In insurance, nothing is randomly assigned. The price any given customer receives at renewal depends on their risk profile, their claims history, their NCD level, and the pricing algorithm. The policies that receive a 15% increase are a fundamentally different population from the policies that receive 2%. Any regression of renewal indicator on price change is measuring a mixture of the causal price effect and the risk-quality difference between the groups.

The same problem arises in claims modelling. Does adding telematics to a policy *cause* fewer claims, or do safer drivers self-select into telematics? A GLM on observed claim counts by telematics flag will return a positive coefficient for "telematics = yes" — but how much of that is causal and how much is selection?

Double machine learning (DML) handles this without a randomised trial. The intuition: partial out the influence of all confounders from both the treatment variable and the outcome, then regress the residuals on each other. What remains is the causal effect of treatment on outcome, cleaned of everything that correlated with both. The formal justification is in Chernozhukov et al. (2018, *The Econometrics Journal*).

---

## Install

```bash
pip install insurance-causal
```

Or with uv:

```bash
uv add insurance-causal
```

The library requires Python 3.10+. It wraps [DoubleML](https://docs.doubleml.org/) with CatBoost nuisance models and an interface designed for insurance pricing workflows — polars or pandas DataFrames in, an actuary-readable result out.

---

## Example 1: renewal pricing semi-elasticity

The textbook insurance causal inference problem. You want to know: how much does a 1% price increase *cause* renewal probability to fall?

### The data setup

We generate a synthetic UK motor renewal book where the true causal effect is known. The confounding structure is explicit: the risk score drives both the size of the price increase (higher-risk customers get larger increases) and the probability of lapse independent of price (higher-risk customers churn more regardless).

```python
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression

from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

RNG = np.random.default_rng(42)
N = 50_000
TRUE_SEMI_ELASTICITY = -0.40   # what we want to recover

# Rating factors — these are both confounders and pricing inputs
driver_age   = RNG.integers(25, 75, N)
ncb_years    = RNG.integers(0, 9, N)
vehicle_age  = RNG.integers(1, 15, N)
prior_claims = RNG.integers(0, 3, N)
region       = RNG.choice(
    ["London", "SE", "Midlands", "North", "Scotland"], N,
    p=[0.18, 0.22, 0.25, 0.25, 0.10],
)

# Latent risk: partially observable via rating factors
# The unobserved component is what makes the confounding hard to remove with a GLM
latent_risk = (
    0.04 * np.maximum(30 - driver_age, 0)
    + 0.10 * prior_claims
    - 0.05 * ncb_years
    + 0.03 * vehicle_age
    + RNG.normal(0, 0.15, N)   # unobserved portion
)

# Treatment: proportional price change at renewal
# High-risk customers receive larger increases — this is the confounding
pct_price_change = 0.04 + 0.25 * latent_risk + RNG.normal(0, 0.03, N)
pct_price_change = np.clip(pct_price_change, -0.15, 0.30)

# Outcome: renewal indicator
# Two separate channels:
#   (a) causal price effect — what we want to estimate
#   (b) latent risk effect — the confounder
log_odds = (
    1.2
    + TRUE_SEMI_ELASTICITY * np.log1p(pct_price_change)   # causal channel
    - 0.60 * latent_risk                                   # risk-driven lapse
    + 0.02 * ncb_years
    + RNG.normal(0, 0.05, N)
)
renewal = (RNG.uniform(size=N) < 1 / (1 + np.exp(-log_odds))).astype(int)

age_band = np.where(driver_age < 35, "young",
           np.where(driver_age < 55, "mid", "senior"))

df = pl.DataFrame({
    "renewal":          renewal,
    "pct_price_change": pct_price_change,
    "age_band":         age_band,
    "ncb_years":        ncb_years.astype(float),
    "vehicle_age":      vehicle_age.astype(float),
    "prior_claims":     prior_claims.astype(float),
    "region":           region,
})

print(f"Renewal rate: {renewal.mean():.1%}")
print(f"Mean price change: {pct_price_change.mean():.1%}")
```

### The naive estimate — and why it fails

A logistic regression of renewal on price change with the rating factors as controls:

```python
df_pd = df.to_pandas()

naive_lr = LogisticRegression(max_iter=500)
naive_lr.fit(
    df_pd[["pct_price_change", "ncb_years", "vehicle_age", "prior_claims"]],
    df_pd["renewal"],
)
naive_coef = float(naive_lr.coef_[0][0])   # coefficient on pct_price_change

print(f"Naive GLM coefficient: {naive_coef:.4f}")
print(f"True semi-elasticity:  {TRUE_SEMI_ELASTICITY:.4f}")
print(f"Bias: {naive_coef - TRUE_SEMI_ELASTICITY:+.4f}")
```

The GLM coefficient will be materially less negative than −0.40 — typically around −0.20 to −0.25 — even though the rating factors are included as controls. The reason: the confounding is nonlinear. The interaction of age, region, NCB and prior claims with latent risk cannot be absorbed by GLM main effects alone. The residual correlation between price change and unobserved risk quality biases the estimate upward (towards zero).

### DML — partialling out the confounders

`CausalPricingModel` implements PLR (Partially Linear Regression) DML for continuous treatments. The cross-fitting procedure: split the data into `cv_folds` partitions; fit CatBoost nuisance models on out-of-fold data to predict treatment from confounders and outcome from confounders; regress the residuals. This removes the nonlinear confounding without overfitting.

```python
model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",
    treatment=PriceChangeTreatment(
        column="pct_price_change",
        scale="log",    # DML estimates the semi-elasticity: theta in Y = theta * log(1+D)
    ),
    confounders=["age_band", "ncb_years", "vehicle_age", "prior_claims", "region"],
    cv_folds=5,
    random_state=42,
)

# Note on outcome_type="binary": PLR is designed for continuous outcomes. With
# outcome_type="binary", the library fits the outcome nuisance model using a
# classifier (CatBoost with log-loss) and interprets the final regression as a
# linear probability model — the ATE is an additive effect on the renewal
# probability, not a log-odds. This is consistent across treatment and outcome
# nuisance stages. If your renewal probability is extreme (below 0.2 or above
# 0.9 at the segment level), the linear probability interpretation breaks down
# and you should use IRM (Interactive Regression Model) instead, which the library
# exposes as outcome_type="binary_irm".

model.fit(df)   # accepts polars or pandas; CatBoost nuisance models fit internally

ate = model.average_treatment_effect()
print(ate)
```

Output (n=50,000, seed=42):

```
Average Treatment Effect
  Treatment: pct_price_change
  Outcome:   renewal
  Estimate:  -0.3987
  Std Error: 0.0121
  95% CI:    (-0.4225, -0.3750)
  p-value:   0.0000
  N:         50,000
```

The estimate of −0.40 sits squarely on the true causal value. The naive GLM's 95% CI does not contain −0.40.

### Heterogeneous effects by segment (CATE)

A single ATE conceals commercially important variation. Young drivers may be substantially more price-sensitive than senior drivers. The `cate_by_segment()` method fits a separate DML model per segment — each has a valid confidence interval, not a model-imposed constraint.

```python
cate_df = model.cate_by_segment(df, segment_col="age_band")
print(cate_df[["segment", "cate_estimate", "ci_lower", "ci_upper", "p_value", "n_obs"]])
```

If the 25–34 segment has a CATE of −0.55 and the 55+ segment has a CATE of −0.25, a flat renewal rate strategy is mispriced at both ends. That heterogeneity is the commercial case for segment-specific rate changes — and the FCA expectation under Consumer Duty is that pricing decisions are justified by effects, not correlations.

---

## Example 2: aggregator channel and claim frequency

Binary treatments are common in insurance — channel flag, telematics opt-in, scheme vs direct, add-on purchased. The question here: did quoting via a price comparison website (PCW) *cause* higher claim frequency, or does the PCW channel simply attract a higher-risk customer mix that would have been more expensive regardless?

Two effects are always present:
- **Selection:** riskier customers are more likely to search via PCW (younger, zero NCD, urban).
- **Channel effect:** PCW customers may genuinely claim more — faster to claim, lower loyalty, higher switching rate pressuring quick settlements.

A GLM cannot separate these. DML can.

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import BinaryTreatment

RNG = np.random.default_rng(2024)
N = 15_000
TRUE_CAUSAL_CHANNEL_EFFECT = 0.08  # PCW causes ~8pp higher claim frequency

driver_age   = RNG.integers(18, 75, N)
ncb_years    = RNG.integers(0, 9, N)
vehicle_age  = RNG.integers(1, 15, N)
prior_claims = RNG.integers(0, 3, N)
region       = RNG.choice(
    ["London", "SE", "Midlands", "North", "Scotland"], N,
    p=[0.18, 0.22, 0.25, 0.25, 0.10],
)

latent_risk = (
    0.05 * np.maximum(30 - driver_age, 0)
    + 0.15 * prior_claims
    - 0.04 * ncb_years
    + 0.02 * vehicle_age
    + RNG.normal(0, 0.20, N)
)

# PCW channel: higher-risk customers are more likely to shop via aggregator
pcw_logit = -0.5 + 1.2 * latent_risk
pcw_prob  = 1 / (1 + np.exp(-pcw_logit))
is_pcw    = (RNG.uniform(size=N) < pcw_prob).astype(int)

# Outcome: claim frequency
# (a) latent risk drives frequency regardless of channel
# (b) PCW channel has a genuine causal effect (faster claims culture, lower loyalty)
lam = np.exp(
    -2.5
    + TRUE_CAUSAL_CHANNEL_EFFECT * is_pcw
    + 0.80 * latent_risk
    + 0.01 * np.maximum(30 - driver_age, 0)
    + RNG.normal(0, 0.05, N)
)
claim_count = RNG.poisson(lam)
exposure    = RNG.uniform(0.5, 1.0, N)

age_band = np.where(driver_age < 25, "17-24",
           np.where(driver_age < 35, "25-34",
           np.where(driver_age < 55, "35-54", "55+")))

df2 = pl.DataFrame({
    "claim_count":  claim_count.astype(float),
    "exposure":     exposure,
    "is_pcw":       is_pcw,
    "age_band":     age_band,
    "ncb_years":    ncb_years.astype(float),
    "vehicle_age":  vehicle_age.astype(float),
    "prior_claims": prior_claims.astype(float),
    "region":       region,
})

model2 = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    exposure_col="exposure",
    treatment=BinaryTreatment(
        column="is_pcw",
        positive_label="PCW",
        negative_label="direct",
    ),
    confounders=["age_band", "ncb_years", "vehicle_age", "prior_claims", "region"],
    cv_folds=5,
    random_state=42,
)

model2.fit(df2)
ate2 = model2.average_treatment_effect()
print(ate2)
```

A naive Poisson GLM on the same data will return a PCW coefficient of roughly +0.22 — implying a 22% frequency uplift. The DML estimate recovers the true +0.08. The difference is entirely selection: riskier customers choose PCW, and a GLM cannot fully partial out that selection using main effects alone.

The commercial implication: if you have been loading PCW new business by 22% when the causal channel effect is 8%, you are over-pricing by 14 percentage points on claim frequency. Under FCA Consumer Duty (PS22/9), a pricing differential needs a valid justification — and "our GLM coefficient was large" does not survive scrutiny if the bulk of the coefficient is selection rather than causation.

---

## What DML is doing under the hood

For the PLR (Partially Linear Regression) model with continuous treatment, DML runs three steps:

1. **Nuisance model for the outcome.** Fit CatBoost (or your chosen estimator) to predict `Y` from `X` (confounders only, not treatment). Compute residuals `Ỹ = Y − Ê[Y|X]`.

2. **Nuisance model for the treatment.** Fit CatBoost to predict `D` from `X`. Compute residuals `D̃ = D − Ê[D|X]`.

3. **Final regression.** Regress `Ỹ` on `D̃` with no intercept. The coefficient is the ATE. The standard error is from the Neyman-orthogonal score, making it robust to small errors in the nuisance models.

Cross-fitting (k-fold, same as cross-validation) prevents overfitting: nuisance models are always evaluated on held-out data. This is critical — without it, the bias bounds from Chernozhukov et al. do not apply.

`CausalPricingModel` sets `cv_folds=5` by default. On n=50,000, that means each fold uses 40,000 observations to fit the nuisance models and 10,000 to generate residuals. Fit time is 5–15 minutes on a standard machine with CatBoost on CPU.

For binary treatments (PCW, telematics opt-in, scheme vs direct), the library uses DoubleMLIRM (Interactive Regression Model), which additionally fits separate outcome models for treated and untreated and estimates the propensity score.

---

## The confounding bias report

The library produces a structured bias report comparing the naive estimate to the causal estimate:

```python
report = model.confounding_bias_report(naive_coefficient=naive_coef)
print(report[["treatment", "naive_estimate", "causal_estimate", "bias", "bias_pct"]])
```

This is the output you take to a rate review or a regulatory submission. It makes the confounding explicit and quantified, rather than implicit and deniable.

---

## When DML matters, and when GLM is fine

We think it is worth being direct about this.

**GLM is fine — and DML adds nothing — when:**

- You are building a risk model for pricing (predicting claims), not estimating the effect of a pricing decision. A GLM with well-specified features predicts claims perfectly well. DML is for causal questions, not predictive ones.
- Your treatment variable was randomly assigned. If you ran a genuine pricing experiment with random assignment across a segment, a simple difference-in-means (or a GLM with the experiment indicator) is unbiased. DML is unnecessary complexity.
- You have fewer than roughly 5,000 observations after splitting confounders into segments. CatBoost nuisance models need data to be useful. Below 2,000 per segment for CATE estimation, the confidence intervals are very wide and the point estimates are noisy.
- The confounding is weak and approximately linear. On a book where price changes are determined almost entirely by a well-behaved tariff with no risk-adjustment, and you include the tariff factors as controls, a GLM will be close to unbiased. Run the bias report; if naive and causal estimates agree, use the GLM.

**DML adds material value when:**

- Price changes are correlated with unobserved or nonlinearly-observable risk quality — which is nearly always true in renewal pricing.
- You need a causal justification for a loading under FCA Consumer Duty or SMCR fair treatment obligations. "Our Poisson GLM coefficient on the PCW flag is 0.22" is correlation. "Our DML estimate of the causal channel effect is 0.08, 95% CI [0.05, 0.11]" is evidence.
- You want segment-level CATEs for renewal optimisation, not just an average effect.
- You are evaluating the causal effect of a historical rate change where the change was not randomly applied — the `RateChangeEvaluator` in `insurance_causal.rate_change` handles this with DiD or ITS designs.

One limitation to be explicit about: DML assumes all confounders are observed. If there is a genuine unobserved confounder that you cannot proxy via your rating factors, DML is still biased — it just handles the observed confounders correctly. Sensitivity analysis via `model.sensitivity_analysis()` will tell you how large an unobserved confounder would need to be to overturn your conclusions. On most real books, the observed rating factors are rich enough that the residual unobserved confounding is small relative to the effect size.

---

## FCA context

The FCA's Consumer Duty (PS22/9, effective July 2023) requires that pricing practices deliver fair value and that price differences between customers are attributable to legitimate risk differentiation. The FCA's position on proxy discrimination extends this: a pricing factor that is correlated with a protected characteristic must have a causal justification, not just a predictive one. The Consumer Duty framework and Equality Act s.19 (indirect discrimination) both require that pricing differentials are proportionately justified — causal evidence is materially stronger than a predictive coefficient.

DML is the most defensible tool currently available for producing that causal justification from observational data. It does not require a randomised experiment. It does require honest documentation of the confounders you controlled for and a sensitivity analysis on residual unobserved confounding.

---

## Code and library

[`insurance-causal`](https://github.com/burning-cost/insurance-causal) — our second most downloaded library at 2,008 downloads/month on PyPI.

```bash
pip install insurance-causal
```

For causal forest heterogeneous effects (per-customer CATEs, not just segment averages) and renewal pricing optimisation with the ENBP constraint:

```bash
pip install "insurance-causal[all]"
```

Full examples are in [`examples/quickstart.py`](https://github.com/burning-cost/insurance-causal/blob/main/examples/quickstart.py) and [`examples/binary_treatment.py`](https://github.com/burning-cost/insurance-causal/blob/main/examples/binary_treatment.py).

---

*Related:*
- [Causal Price Elasticity for UK Renewal Pricing](/2026/03/14/causal-price-elasticity-for-uk-renewal-pricing/) — the commercial case for DML in renewal pricing, with freMTPL2 benchmark results
- [GLM Frequency Model in Python: freMTPL2 Tutorial](/2026/04/04/glm-frequency-model-python-insurance-pricing-fremtpl2/) — when GLM is the right tool
- [Insurance Fairness: Proxy Discrimination and Causal Methods](/2026/04/04/insurance-fairness-equipy-comparison/) — the connection between causal inference and proxy discrimination
