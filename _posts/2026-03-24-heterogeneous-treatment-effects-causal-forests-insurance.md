---
layout: post
title: "Your Pricing Model Knows the Average Effect. That Is Not Enough."
date: 2026-03-24
categories: [techniques, pricing]
tags: [causal-inference, causal-forest, heterogeneous-treatment-effects, hte, dml, telematics, econml, insurance-causal, uk-motor]
description: "Using causal forests and GATES/CLAN/RATE inference to find which customers respond most to a price or discount change — not just the average effect."
---

Double/debiased machine learning gives you a clean average treatment effect (ATE): the mean semi-elasticity of renewal probability with respect to a 10% price increase is −0.18, say. That is a useful number. But it is also the wrong number for most of the decisions you actually make.

When you decide whether to offer a telematics discount, you are not offering it to the portfolio average. You are offering it to specific customers, each of whom has their own elasticity. The 22-year-old in inner-city Manchester on a PCW may respond three times more strongly to a £50 discount than the 55-year-old direct renewer in Surrey. Treating them identically — either by offering the same discount or by using the same retention assumption in a pricing optimisation — is a pricing error. The question is whether you can measure the difference.

Conditional average treatment effects (CATEs) are the answer. The new `causal_forest` module in [`insurance-causal`](https://github.com/burning-cost/insurance-causal) provides a full pipeline: estimate per-risk CATEs with a causal forest, then run the formal Chernozhukov et al. (2020/2025) inference suite — BLP, GATES, CLAN — plus the Yadlowsky et al. (2025 JASA) RATE test for targeting validity.

```bash
uv add insurance-causal
```

---

## The problem with ATE for a heterogeneous portfolio

Suppose you are evaluating a telematics discount programme. The ATE tells you that across the book, offering a telematics device reduces claims frequency by 8%. Decent result. But the mechanism is not uniform: younger drivers change their behaviour under observation more than older ones; urban drivers have more scope to reduce peak-hour exposure; high-mileage drivers get more signal from telematics. If the discount costs you the same per policy, you want to concentrate it on the customers where the effect is largest.

DML — the approach in the `autodml` subpackage — correctly identifies the ATE but by construction averages out heterogeneity. `CausalForestDML` from EconML (Athey, Tibshirani & Wager, 2019, *Annals of Statistics* 47(2)) instead estimates a separate treatment effect for every point in covariate space. The forest uses "honest" sample splitting: trees are grown on one half of the data and treatment effects estimated on the other, which prevents the in-sample bias that would otherwise inflate heterogeneity findings.

The CATE estimate at a given covariate value x is:

```
tau_hat(x) = E[Y(1) - Y(0) | X = x]
```

For continuous treatment (log price change rather than a binary discount), this becomes a semi-elasticity: the expected change in renewal probability per unit increase in log price, conditional on risk profile x.

---

## Fitting the estimator

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
    TargetingEvaluator,
    make_hte_renewal_data,
)

df = make_hte_renewal_data(n=50_000, seed=42)

confounders = ["age", "ncd_years", "vehicle_group", "channel", "region"]

est = HeterogeneousElasticityEstimator(
    n_estimators=400,
    min_samples_leaf=20,   # larger than econml default; sparse rating cells need it
    catboost_iterations=500,
    random_state=42,
)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)

# ATE with 95% CI
ate, lb, ub = est.ate()
print(f"ATE: {ate:.3f}  [{lb:.3f}, {ub:.3f}]")
# -> ATE: -0.189  [-0.201, -0.177]

# Per-risk CATEs
cates = est.cate(df)
print(f"CATE range: {cates.min():.3f} to {cates.max():.3f}")
# -> CATE range: -0.312 to -0.089
```

The CATEs span from −0.31 to −0.09 across the portfolio. The portfolio average (−0.19) is somewhere in the middle. The question is whether that spread is genuine heterogeneity or estimation noise — and if genuine, which customers sit at the extremes.

The `min_samples_leaf=20` default matters. EconML's default is 5. For insurance data, rating segments can be sparse — vehicle group E in Scotland may have 200 policies in a 50,000-row training set. Leaves with fewer than 20 observations produce CATE estimates with intervals too wide to act on; 20 is the minimum we would use in production.

---

## GATES: does the spread matter, and how large is it?

Group Average Treatment Effects (GATES) partition the portfolio into quintiles by predicted CATE and estimate the average effect within each group. If the causal forest has identified real heterogeneity, Group 5 (most price-elastic) should have a substantially larger effect than Group 1 (least elastic). If the spread across quintiles is indistinguishable from zero, the CATE variation is noise.

```python
inf = HeterogeneousInference(n_splits=100, k_groups=5)
result = inf.run(df, estimator=est, cate_proxy=cates)

print(result.summary())
```

```
Heterogeneous Treatment Effect Inference
==================================================
N observations: 50,000
N data splits:  100

--- BLP (Best Linear Predictor) ---
  ATE (beta_1):    -0.1887  (SE=0.0031)
  HTE (beta_2):     0.7841  (SE=0.0892)
  H0: beta_2=0 -> p=0.0000  [HETEROGENEITY DETECTED]

--- GATES (Group Average Treatment Effects) ---
  Group 1: GATE=-0.0991  (SE=0.0044)  n=10,003
  Group 2: GATE=-0.1523  (SE=0.0038)  n=9,998
  Group 3: GATE=-0.1889  (SE=0.0037)  n=9,997
  Group 4: GATE=-0.2301  (SE=0.0040)  n=10,001
  Group 5: GATE=-0.2998  (SE=0.0046)  n=10,001
  Monotone increasing: True
```

Group 5's estimated GATE (−0.30) is roughly three times the magnitude of Group 1's (−0.10). The BLP beta_2 is positive and highly significant, confirming the CATE proxy explains real variation — not forest artefacts.

The `n_splits=100` matters. The inference procedure (Chernozhukov et al. 2020) runs 100 random splits of the data, fits a propensity model on the auxiliary half, and runs the BLP regression on the main half each time, then aggregates t-statistics via the median. Single-split inference would be anti-conservative; 100 splits stabilises the size.

---

## CLAN: who are the most and least affected?

GATES tells you the effect size varies by quintile. CLAN (Classification Analysis) tells you what those quintiles look like — which covariates differ most between Group 5 and Group 1.

```python
result.plot_clan()
```

The CLAN table in `result.clan` is a polars DataFrame with columns `feature`, `mean_top`, `mean_bottom`, `diff`, `t_stat`, `p_value`. The top differentiating features in the synthetic data (which is calibrated to NCD band):

| Feature | Mean (Group 5) | Mean (Group 1) | Diff | p |
|---|---|---|---|---|
| ncd_years | 0.4 | 4.7 | −4.3 | <0.001 |
| age | 24.1 | 56.8 | −32.7 | <0.001 |
| channel_pcw | 0.71 | 0.21 | +0.50 | <0.001 |

Group 5 is disproportionately NCD=0 or 1, under-25, and on PCW. Group 1 is NCD=4/5, over-50, direct channel. This is exactly what you would expect: the most price-elastic customers are those with least investment in the relationship and the most alternatives visible to them.

For a telematics programme evaluation, this output directly answers "who should we target?" without requiring the team to specify the segmentation in advance.

---

## RATE: does the ranking actually work?

Knowing that heterogeneity exists is step one. Step two is asking whether your CATE estimates produce a useful targeting rule. RATE (Rank-Weighted Average Treatment Effect, Yadlowsky et al. 2025 *JASA* 120(549)) measures whether customers ranked as most elastic by `tau_hat` genuinely receive larger effects than the average.

```python
evaluator = TargetingEvaluator(method="autoc", n_bootstrap=500)
rate_result = evaluator.evaluate(df=df, estimator=est, cate_proxy=cates)
print(rate_result.summary())
```

```
Targeting Evaluation (RATE)
========================================
N observations: 50,000

AUTOC: 0.0312  (SE=0.0041)
  95% CI: [0.0232, 0.0392]
QINI:  0.0187

Interpretation:
  AUTOC significantly positive — CATE ranking is informative for targeting.
```

AUTOC integrates TOC(q)/q across the q-grid, upweighting targeting precision at small q (i.e., if you can only offer the discount to your top 10%, does the model identify the right 10%?). A significantly positive AUTOC means the ranking adds real information over random assignment.

If AUTOC is not significant — which happens when treatment variation is too small for the forest to identify heterogeneity reliably — the right response is to not use individual CATEs for targeting decisions. The result is not a failure of the method; it is a useful finding that avoids acting on noise.

---

## A note on claim frequency

The synthetic data above uses renewal (binary outcome). For a claim frequency outcome with exposure, `HeterogeneousElasticityEstimator` accepts an `exposure_col` argument:

```python
est_freq = HeterogeneousElasticityEstimator(
    binary_outcome=False,
    exposure_col="exposure",
    n_estimators=400,
)
est_freq.fit(
    df,
    outcome="claim_count",
    treatment="log_price_change",
    confounders=confounders,
)
```

When `exposure_col` is set, the outcome is divided by exposure (converting counts to rates) and exposure values are passed as sample weights to the nuisance models. The interpretation of `tau_hat(x)` then becomes the change in claim rate per unit log price change — a less intuitive quantity than renewal elasticity, but useful for adverse selection modelling where you want to know whether price changes shift the risk mix differently across segments.

---

## What this is not

A few things worth being explicit about.

The causal forest, like any DML estimator, requires an identification assumption: conditional on the confounders, price changes must be partially random. In a re-pricing exercise with tight technical models, almost all of the price variation is explained by the confounders and there is very little exogenous price variation left to identify the elasticity. The library will warn if treatment residual variance is near zero, but it cannot recover identification from data that does not have it. You need genuine price variation — A/B test randomisation, historical pricing grid quirks, or competitor shock variation — for any of this to be valid.

The CATE estimates are also conditional on the forest specification. The default `min_samples_leaf=20` means that very sparse rating cells may be averaged over broader neighbourhoods than ideal. Validate the GATES against known subgroups (NCD band, age decile) before relying on the CLAN output for operational decisions.

---

## The full workflow in brief

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
    TargetingEvaluator,
)

# 1. Fit
est = HeterogeneousElasticityEstimator(n_estimators=400)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)
cates = est.cate(df)

# 2. Test heterogeneity and characterise (BLP + GATES + CLAN)
result = HeterogeneousInference(n_splits=100, k_groups=5).run(
    df, estimator=est, cate_proxy=cates
)
print(result.summary())
result.plot_gates()
result.plot_clan()

# 3. Validate the ranking (RATE)
rate = TargetingEvaluator(n_bootstrap=500).evaluate(
    df=df, estimator=est, cate_proxy=cates
)
print(rate.summary())
rate.plot_toc()
```

Four method calls after fitting. The output tells you whether heterogeneity is real (BLP), how large it is by quintile (GATES), who the extreme groups are (CLAN), and whether the ranking is actionable (RATE). That is the complete answer to "should I be using the ATE here, or does segment matter?"

For most UK motor portfolios with sufficient price variation, it matters considerably.
