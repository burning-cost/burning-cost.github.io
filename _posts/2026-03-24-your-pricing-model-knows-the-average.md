---
layout: post
title: "Your Pricing Model Knows the Average. Your Customers Don't Care About the Average."
date: 2026-03-24
author: Burning Cost
categories: [pricing, causal-inference, heterogeneous-treatment-effects, libraries]
description: "DML gives you the ATE. But the ATE is the mean of a wide distribution. Causal forests estimate the full CATE distribution and tell you which customers are 5x more elastic than the average. Here is how to use insurance-causal's causal_forest subpackage to find them."
canonical_url: "https://burning-cost.github.io/2026/03/24/your-pricing-model-knows-the-average/"
tags: [causal-inference, hte, causal-forest, dml, gates, clan, rate, elasticity, insurance-causal, renewal, motor, uk-motor, python, polars, econml]
---

Your DML model tells you the average treatment effect: a 10% price increase reduces renewal probability by X percentage points on average across the portfolio. That number is real, it is properly identified, and it is useless for deciding how hard to push on any individual segment.

The ATE is the mean of a distribution. In insurance renewal data, that distribution is wide. The customer on NCD year 0 who bought through a comparison website is a completely different renewal proposition from the customer on NCD year 5 who set up a direct debit three years ago and has never shopped. Applying a flat rate change treats them identically. One of them will leave; the other barely notices. The question the ATE cannot answer is which one.

This is the problem that heterogeneous treatment effect estimation exists to solve.

---

## What causal forests actually estimate

A causal forest does the same residualisation as DML — it regresses out the confounders from both the outcome and the treatment — but instead of estimating a single pooled coefficient, it fits a forest that produces a **conditional average treatment effect (CATE)** for each observation. The CATE at $x$ is an estimate of how much the treatment effect differs from the ATE for customers with characteristics $x$.

The machinery is CausalForestDML (Athey, Tibshirani & Wager 2019, *Annals of Statistics*), which handles the cross-fitting correctly and uses honest splitting — the trees used to partition the feature space are trained on a separate sample from the ones used to estimate the leaf-level effects. Without honesty, the CATEs overfit and the confidence intervals are wrong.

The CATE estimates on their own are not sufficient. You need to know three things:

1. **GATES**: Are the CATE estimates correctly ordered? Does the group we call "most elastic" actually have a higher average treatment effect than the group we call "least elastic"?
2. **CLAN**: Which covariates separate the most elastic customers from the least? This is the question that tells you the economic story.
3. **RATE**: If you built a targeting rule using the CATE ranking — offer retention discounts only to the top-quintile elastic customers — does the ranking add actual value, or is it noise?

These three form a progression from "does heterogeneity exist?" to "who are they?" to "can you act on it?".

---

## The workflow

We use `insurance-causal` v0.5.2.

```bash
uv add insurance-causal
```

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
    TargetingEvaluator,
    CausalForestDiagnostics,
    make_hte_renewal_data,
)
```

For this walkthrough we use the synthetic renewal dataset from the library. In production you would substitute your own dataframe — polars or pandas, both accepted.

```python
df = make_hte_renewal_data(n=20_000, seed=42)

confounders = ["age", "ncd_years", "vehicle_group", "channel"]

est = HeterogeneousElasticityEstimator(
    n_estimators=200,
    min_samples_leaf=20,    # default; important for sparse rating segments
    catboost_iterations=500,
    random_state=42,
)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)
```

`min_samples_leaf=20` is the insurance-specific default, larger than EconML's default of 5. Rating segments can be sparse in the tails — very young drivers, high-value vehicles — and leaves with fewer than 20 observations produce CATE estimates wide enough to be useless.

Get the ATE and the per-customer CATEs:

```python
ate, ate_lb, ate_ub = est.ate()
print(f"ATE: {ate:.4f}  95% CI [{ate_lb:.4f}, {ate_ub:.4f}]")
# ATE: -0.1842  95% CI [-0.2104, -0.1580]

cates = est.cate(df)
print(f"CATE std dev: {cates.std():.4f}")
print(f"CATE 10th pct: {cates[cates.argsort()][:2000].mean():.4f}")
print(f"CATE 90th pct: {cates[cates.argsort()][-2000:].mean():.4f}")
```

The ATE is -0.18: a 10% price increase (log 0.10) reduces renewal probability by 18 percentage points on average. The distribution tells a different story. The bottom decile averages around -0.08; the top decile averages around -0.37. That is roughly a 4-5x difference in elasticity across the portfolio.

---

## Diagnostics first

Before trusting the CATEs, run the diagnostic checks. The near-deterministic price problem is real in insurance data: if your price changes are almost entirely explained by risk factors (technical premium changes applied segment-by-segment with no noise), the residual variation after conditioning on covariates approaches zero and the CATEs are unreliable.

```python
diag = CausalForestDiagnostics(n_splits=20, random_state=42)
report = diag.check(df, estimator=est, cates=cates)
print(report.summary())
```

```
Causal Forest Diagnostics
=============================================
N observations: 20,000

[PASS] Overlap / propensity
       Propensity SD:        0.0431

[PASS] Residual variation
       Var(D-e)/Var(D):       0.3817  (threshold: >=0.10)

[PASS] BLP calibration
       beta_2 (HTE):          0.7234  (p=0.0001)

All checks passed. CATE estimates should be reliable.
```

The residual variation ratio is the key number. 0.38 means 38% of the variation in log price changes is not explained by the covariates — there is real exogenous pricing variation to identify from. If this were 0.04, the causal forest is trying to find heterogeneity in near-deterministic price assignments, and you should stop here.

---

## GATES: how large is the spread?

GATES partitions customers into quintiles by estimated elasticity and estimates the average treatment effect in each group. The test is whether the groups are monotone — groups 1 through 5 should have increasing (in magnitude) treatment effects if the CATE proxy is correctly ordering heterogeneity.

```python
inf = HeterogeneousInference(n_splits=100, k_groups=5, random_state=42)
result = inf.run(df, estimator=est, cate_proxy=cates)
print(result.summary())
```

```
Heterogeneous Treatment Effect Inference
==================================================
N observations: 20,000
N data splits:  100

--- BLP (Best Linear Predictor) ---
  ATE (beta_1):    -0.1837  (SE=0.0135)
  HTE (beta_2):     0.7191  (SE=0.0881)
  H0: beta_2=0 -> p=0.0000  [HETEROGENEITY DETECTED]

--- GATES (Group Average Treatment Effects) ---
  Group 1: GATE=-0.0801  (SE=0.0214)  n=4,000
  Group 2: GATE=-0.1344  (SE=0.0189)  n=4,000
  Group 3: GATE=-0.1822  (SE=0.0201)  n=4,000
  Group 4: GATE=-0.2491  (SE=0.0177)  n=4,000
  Group 5: GATE=-0.3619  (SE=0.0193)  n=4,000
  Monotone increasing: True
```

Group 5 has a GATE of -0.36 versus Group 1 at -0.08. The most elastic quintile has 4.5x the treatment effect of the least elastic quintile. The groups are monotone, which validates that the CATE proxy is correctly ordering them.

The BLP `beta_2` of 0.72 (p < 0.0001) is the formal test: the CATE proxy predicts significant variation in realised treatment effects. This is not trivially true — you can fit a causal forest on any data and get CATE estimates; beta_2 tells you whether those estimates correspond to real heterogeneity in the data.

---

## CLAN: who are the elastic customers?

Knowing the CATE spread is useful; knowing who sits in Group 5 is actionable.

```python
# Top features from CLAN
clan_table = (
    result.clan.table
    .sort("p_value")
    .head(8)
    .select(["feature", "mean_top", "mean_bottom", "diff", "p_value"])
)
print(clan_table)
```

```
┌────────────────────┬──────────┬─────────────┬────────┬──────────┐
│ feature            │ mean_top │ mean_bottom │ diff   │ p_value  │
│ ---                │ ---      │ ---         │ ---    │ ---      │
│ str                │ f64      │ f64         │ f64    │ f64      │
╞════════════════════╪══════════╪═════════════╪════════╪══════════╡
│ channel_PCW        │ 0.8341   │ 0.1203      │ 0.7138 │ 0.000000 │
│ ncd_years          │ 1.2104   │ 4.6831      │ -3.473 │ 0.000000 │
│ age                │ 28.4     │ 44.7        │ -16.3  │ 0.000001 │
│ vehicle_group_G    │ 0.4102   │ 0.2341      │ 0.1761 │ 0.000034 │
│ vehicle_group_E    │ 0.1882   │ 0.2701      │ -0.082 │ 0.000812 │
└────────────────────┴──────────┴─────────────┴────────┴──────────┘
```

The CLAN compares covariate means between the most elastic group (Group 5) and the least elastic (Group 1). The signal is unambiguous:

- **Channel**: 83% of the most elastic customers came through a PCW versus 12% of the least elastic. This is the dominant signal.
- **NCD years**: the most elastic group averages 1.2 years NCD; the least elastic averages 4.7 years. Low NCD customers are building their discount — they are newer, less committed, and actively price-shopping.
- **Age**: the most elastic group averages 28; the least averages 45. Younger drivers are disproportionately price-sensitive at renewal.

This is not a surprise to anyone who has worked motor books for long enough. But there is a difference between knowing directionally that PCW buyers are more elastic and having a model-free estimate of the *size* of that heterogeneity with confidence intervals on it. A flat 5% rate increase applied to this portfolio is extracting value from the loyal direct NCD-5 customers whilst pushing out the PCW NCD-0 customers at a rate 4.5x faster than the ATE would imply.

---

## RATE: does the targeting rule actually work?

GATES and CLAN tell you the distribution is wide and what drives it. RATE (Yadlowsky et al. 2025, *JASA*) asks a harder question: if you built a retention discount rule targeting only the top-quintile elastic customers, would the CATE ranking actually identify the right ones?

```python
evaluator = TargetingEvaluator(method="autoc", n_bootstrap=500, random_state=42)
rate_result = evaluator.evaluate(df=df, estimator=est, cate_proxy=cates)
print(rate_result.summary())
```

```
Targeting Evaluation (RATE)
========================================
N observations: 20,000

AUTOC: 0.0412  (SE=0.0089)
  95% CI: [0.0237, 0.0587]
QINI:  0.0318

Interpretation:
  AUTOC significantly positive — CATE ranking is informative for targeting.
```

The AUTOC is significantly positive: the lower bound of the confidence interval is 0.024, well above zero. Customers ranked highly by the causal forest do receive meaningfully larger treatment effects than the portfolio average. The targeting rule works.

If the AUTOC CI had included zero, the correct conclusion would be to stop here: the CATE estimates have no targeting value, likely because the sample is too small or the treatment variation too limited to identify reliable individual-level effects.

---

## When not to use this

Causal forests require sample sizes the ATE does not. Honest splitting divides the training data in two, and each cross-fitting fold operates on a further fraction. Below around 10,000 observations the CATEs become unreliable — the diagnostics will warn you (`n < 5,000` triggers an explicit warning; treat anything below 10,000 as exploratory).

The near-deterministic price problem is more serious than it looks. If your renewal pricing is fully determined by risk factors — last year's premium × technical change factor, applied identically across every policy in a rating cell — then the residual variation ratio will be near zero and there is nothing for the causal forest to learn from. You need real exogenous variation: pricing A/B tests, staggered roll-outs, or natural experiments where policies within the same rating cell received different prices due to timing or system quirks. If that variation does not exist in your data, the right tool is a structural demand model, not a causal forest.

The library enforces the `min_samples_leaf=20` default, which is larger than EconML's default of 5. You can reduce this if your data is dense, but the warning about small groups in `gate()` and `_run_gates()` exists for a reason: CATE estimates in segments with fewer than 500 observations are wide enough to be misleading.

Finally: do not use individual CATEs to set individual renewal prices without further validation under FCA pricing fairness rules. The CATE is a causal estimate of price sensitivity, not a risk-adequate premium. The appropriate use is to inform targeting rules (who gets a retention discount offer) and to understand the size of cross-subsidisation effects, not to feed raw CATE values into the rating engine.

---

## Summary

The ATE tells you the population-average effect of a price change. The CATE distribution tells you that this average conceals a 4-5x range across your book — and that the customers at the elastic end are systematically identifiable by channel and NCD. GATES quantifies the spread. CLAN names the drivers. RATE validates whether the ranking adds targeting value.

`insurance-causal` v0.5.2 is on PyPI. Install with `uv add insurance-causal`. The full worked example is at [`causal_forest_hte_walkthrough.py`](https://github.com/burning-cost/burning-cost-examples/blob/main/examples/causal_forest_hte_walkthrough.py).
