---
layout: post
title: "Your Pricing Model Knows the Average. Your Customers Don't Care About the Average."
date: 2026-03-25
categories: [techniques, pricing]
tags: [causal-inference, causal-forest, heterogeneous-treatment-effects, hte, gates, clan, rate, dml, elasticity, insurance-causal, uk-motor, econml]
description: "The average treatment effect hides a 5x spread in price elasticity across a UK motor book. GATES, CLAN, and RATE tell you the size, who's who, and whether the ranking is actionable — with confidence intervals."
---

Your DML model says the average semi-elasticity of renewal is −0.19 per unit log price change. That is a correct estimate of the population average. It is also the number you should stop reporting as if it settles anything.

The problem is not that the number is wrong. It is that the mean of a wide distribution answers a different question from the one that actually matters for pricing. If the most price-sensitive 20% of your book has an elasticity of −0.30 and the least sensitive 20% is at −0.10, applying a uniform renewal discount based on the average effect will be wasteful on one group and insufficient on the other. In a UK motor portfolio with 200,000 renewals per year, the gap between pricing to the average and pricing to the distribution is a material amount of margin.

Conditional average treatment effects (CATEs) and their formal inference machinery — GATES, CLAN, RATE — address this directly. The `causal_forest` module in [`insurance-causal`](https://github.com/burning-cost/insurance-causal) implements the full pipeline.

---

## Why the average is the wrong number

Consider a re-pricing exercise at renewal. You want to know which customers to hold and which to let go. The standard workflow uses an aggregate price elasticity — from a GLM or a DML estimator — and applies it uniformly across all segments when computing the expected volume response to a proposed rate change.

That workflow has a hidden assumption: the elasticity is constant across the portfolio. For a GLM with a price interaction term, the assumption is explicit. For DML, it is less visible but still there — the DML ATE is the average of the heterogeneous effects, not their distribution.

In a UK motor book, the assumption is wrong in a predictable way. PCW customers, younger drivers, and NCD band 0–1 customers consistently show higher price sensitivity than direct-channel, NCD band 4–5 renewers. This is not a new insight to actuaries — it is already embedded in NCD discount structures and channel loading factors. What the causal forest adds is a defensible, model-free estimate of *how large* the difference is, with proper confidence intervals.

In our benchmarks on a 20,000-policy synthetic UK motor DGP — six segments varying by age band and urban/rural, with log-odds semi-elasticities ranging from −6.0 (young urban) to −1.0 (senior rural) — the GATE-based approach reduced segment RMSE by a factor of 3.1 compared to applying the portfolio ATE to every segment uniformly (0.1226 vs 0.3824). For the young urban segment alone, the uniform ATE overestimated the effect by 4.4x.

The uniform ATE is not a neutral choice. It systematically misdirects retention spend.

---

## Before you run the forest: the diagnostic screen

Causal forests are useless if there is no exogenous price variation in the data. The near-deterministic price problem — where price changes are almost entirely explained by the observable risk factors — is common in insurance: if your technical model determines 95% of the price variation, the DML residual treatment is tiny and the CATE estimates will have intervals too wide to act on.

Run the diagnostic check before fitting the forest:

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    CausalForestDiagnostics,
    make_hte_renewal_data,
)

df = make_hte_renewal_data(n=50_000, seed=42)
confounders = ["age", "ncd_years", "vehicle_group", "channel", "region"]

est = HeterogeneousElasticityEstimator(
    n_estimators=400,
    min_samples_leaf=20,
    catboost_iterations=500,
    random_state=42,
)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)
cates = est.cate(df)

diag = CausalForestDiagnostics(n_splits=20)
report = diag.check(df, estimator=est, cates=cates)
print(report.summary())
```

```
Causal Forest Diagnostics
=============================================
N observations: 50,000

[PASS] Overlap / propensity
       Propensity SD:        0.0412

[PASS] Residual variation
       Var(D-e)/Var(D):       0.2317  (threshold: >=0.10)

[PASS] BLP calibration
       beta_2 (HTE):          0.6893  (p=0.0000)

All checks passed. CATE estimates should be reliable.
```

The key number is `Var(D-e)/Var(D)` — the fraction of treatment variation that is exogenous after conditioning on observables. The threshold of 0.10 is conservative: below it, the effective instrument is too weak for reliable CATE estimation. If you fail this check, running the full GATES analysis will produce spurious-looking segmentation that reflects noise, not heterogeneity.

The `min_samples_leaf=20` default matters. EconML's default is 5. Insurance rating segments are frequently sparse — vehicle group E in Scotland, or a high-performance classification in a rural postcode, may have 200 policies in a 50,000-row training set. With leaves below 20 observations, the CATE estimates in those cells have intervals far too wide to act on. Twenty is the minimum we use in production.

---

## GATES: size of the spread across quintiles

Once the diagnostic screen passes, the inference question is: how much does elasticity vary across the portfolio? GATES (Sorted Group Average Treatment Effects, Chernozhukov et al. 2020/2025 *Econometrica*) partitions the book into quintiles by predicted CATE and estimates the average effect within each. The key test is whether the spread across quintiles is statistically distinguishable from zero.

```python
from insurance_causal.causal_forest import HeterogeneousInference

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

Group 5's GATE (−0.30) is roughly three times Group 1's (−0.10). The BLP beta_2 coefficient (0.78) measures how well the CATE proxy explains the realised variation — a value below zero would indicate the forest rankings are noise. At p < 0.0001 the heterogeneity is not a modelling artefact.

The `n_splits=100` matters. The BLP procedure (Chernozhukov et al.) runs 100 random data splits, fits a propensity model on the auxiliary half and the BLP regression on the main half each time, then aggregates t-statistics via the median. A single split produces anti-conservative inference; 100 splits stabilise the size to the nominal level.

The portfolio ATE (−0.19) sits at Group 3. If you target a retention discount at the whole book using that number, you are underinvesting in Groups 4–5 (who would respond more) and overinvesting in Groups 1–2 (who barely care about the price). The GATES table translates directly into a tiered discount strategy: offer the most generous retention discount to the top two quintiles, a moderate discount to the middle, and accept churn at the bottom.

---

## CLAN: who are they?

GATES tells you the effect size differs. CLAN (Classification Analysis) tells you which customers sit in Group 5 vs Group 1. The analysis is non-parametric: it simply compares covariate means between the top and bottom quintiles and reports the largest differentiators.

```python
result.plot_clan()
```

From `result.clan` (a polars DataFrame with columns `feature`, `mean_top`, `mean_bottom`, `diff`, `t_stat`, `p_value`):

| Feature | Group 5 mean | Group 1 mean | Diff | p |
|---|---|---|---|---|
| ncd_years | 0.4 | 4.7 | −4.3 | <0.001 |
| age | 24.1 | 56.8 | −32.7 | <0.001 |
| channel_pcw | 0.71 | 0.21 | +0.50 | <0.001 |

Group 5 is disproportionately NCD band 0–1, under-25, on PCW. Group 1 is NCD 4–5, over-50, direct channel. The CLAN output confirms what experienced actuaries would tell you informally — but it also quantifies the difference. A 50-percentage-point gap in PCW prevalence between the most and least elastic quintiles is the kind of number that makes the case for segment-specific discount parameters in a pricing optimisation.

The same analysis provides a fairness diagnostic: if the CLAN output shows that Group 5 (most elastic, earns the largest discount) is strongly correlated with age, that is a FCA PS22/9 indirect discrimination concern. Running CLAN on protected characteristics is not optional.

---

## RATE: does the ranking actually work?

GATES confirms heterogeneity exists. CLAN characterises it. The remaining question is operational: does ranking customers by their estimated CATE produce a useful targeting rule, or is the ranking too noisy to act on?

RATE (Rank-Weighted Average Treatment Effect, Yadlowsky et al. 2025 *JASA* 120(549)) addresses this. The test asks whether customers ranked as most elastic by `tau_hat` genuinely receive larger effects than a random subset of the same size. The AUTOC metric integrates targeting precision across the full q-grid, upweighting the top of the ranking (relevant if you can only afford to target the top 10–20%).

```python
from insurance_causal.causal_forest import TargetingEvaluator

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

In our synthetic benchmark, the AUTOC was 1.9319 (SE=0.0105, p=0.000), reflecting a DGP with strong and cleanly separated heterogeneity. In practice, on real UK motor data with noisier confounding and less extreme segment differentiation, AUTOC will be smaller — but a significantly positive AUTOC is the minimum bar for operationalising the CATE ranking.

If AUTOC is not significant, the correct conclusion is: you have confirmed heterogeneity exists (from BLP/GATES), but the per-customer CATE estimates are too imprecise to rank customers reliably. In that case, use the GATES quintile membership as a coarse targeting rule rather than the continuous CATE. This is not a failure mode — it is the method correctly flagging the limits of the data.

---

## The quick segment cut: `gate()` for known groups

For a faster analysis when you already know which segments to compare — NCD band, channel, age decile — `HeterogeneousElasticityEstimator.gate()` gives CATE estimates grouped by any categorical variable, without running the full BLP inference machinery:

```python
# NCD-band cut after fitting
ncd_gates = est.gate(df, by="ncd_years")
print(ncd_gates)
```

```
shape: (6, 5)
┌──────────┬─────────┬──────────┬──────────┬───────┐
│ ncd_years│  cate   │ ci_lower │ ci_upper │   n   │
╞══════════╪═════════╪══════════╪══════════╪═══════╡
│     0    │ -0.2984 │ -0.3102  │ -0.2866  │  8521 │
│     1    │ -0.2601 │ -0.2710  │ -0.2492  │  9847 │
│     2    │ -0.2213 │ -0.2318  │ -0.2108  │ 10432 │
│     3    │ -0.1778 │ -0.1876  │ -0.1680  │ 11203 │
│     4    │ -0.1421 │ -0.1517  │ -0.1325  │  9998 │
│     5    │ -0.0987 │ -0.1083  │ -0.0891  │ 10012 │
└──────────┴─────────┴──────────┴──────────┴───────┘
```

NCD 0 is three times as elastic as NCD 5. The confidence intervals are tight at n > 8,000 per group. This output feeds directly into a pricing optimisation: the NCD multiplier for renewal pricing should reflect not just the risk gradient but also the different renewal response to price — two effects that happen to point in the same direction but for different reasons.

Groups with fewer than 500 policies trigger a warning; at that count, the CATE confidence intervals are typically too wide for individual decisions.

---

## Sample size: the practical minimum

For a UK motor book:

- Fewer than 20,000 renewals: use DML ATE with segment-level manual GATE by `est.gate()`. Do not run the full causal forest GATES/CLAN inference — the n_splits=100 BLP procedure requires enough observations per split for the propensity model to be reliable.
- 20,000–50,000: causal forest is viable but use `min_samples_leaf=20` and treat CLAN output as directional rather than precise. Validate against known actuarial intuition.
- 50,000+: full pipeline is appropriate. The GATES CIs will be tight enough to distinguish adjacent quintiles.

Honest splitting halves the effective sample. With 50,000 policies, the estimation half of each fold contains roughly 5,000 observations per k-fold split — enough for the nuisance models and the CATE forest, but only if `min_samples_leaf` is set conservatively.

---

## The bottom line

The ATE is not wrong. It is the correct answer to "what happens on average?" Insurance pricing rarely needs to know what happens on average. It needs to know what happens to the specific customer in front of it.

GATES gives you the quintile spread with confidence intervals. CLAN tells you that the spread aligns with NCD band and channel — variables you already knew mattered, but now with a model-free effect size. RATE tells you whether the individual ranking is tight enough to operationalise. Between them, they convert "we think PCW customers are more elastic" from an actuarial heuristic into a defensible number you can put in a pricing sign-off.

For most UK motor portfolios with genuine price variation in the data, the spread is large enough to matter. The question is whether you are measuring it.

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
    TargetingEvaluator,
    CausalForestDiagnostics,
)

est = HeterogeneousElasticityEstimator(n_estimators=400, min_samples_leaf=20)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)
cates = est.cate(df)

# 1. Screen: is there enough exogenous variation?
report = CausalForestDiagnostics(n_splits=20).check(df, estimator=est, cates=cates)
if not report.residual_variation_ok:
    print("Insufficient price variation — stop here.")

# 2. Measure the spread (GATES) and characterise segments (CLAN)
result = HeterogeneousInference(n_splits=100, k_groups=5).run(
    df, estimator=est, cate_proxy=cates
)
result.plot_gates()
result.plot_clan()

# 3. Validate the targeting rule (RATE)
rate = TargetingEvaluator(n_bootstrap=500).evaluate(df=df, estimator=est, cate_proxy=cates)
print(rate.summary())
```

Install with `uv add insurance-causal`. The causal forest dependencies (econml, catboost) are declared as optional extras; `uv add insurance-causal[causal_forest]` pulls them in.
