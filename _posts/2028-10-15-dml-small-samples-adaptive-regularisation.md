---
layout: post
title: "DML Works at 1,000 Policies Now. Here Is What Changed."
date: 2028-10-15
categories: [techniques]
tags: [causal-inference, dml, double-machine-learning, catboost, small-samples, treatment-effect, renewal-pricing, telematics, insurance-causal, python, uk-motor]
description: "insurance-causal v0.3.1 fixes over-partialling in DML for small insurance books. Adaptive CatBoost regularisation makes causal estimates reliable at n≥1k."
---

Double Machine Learning has a small-sample problem that we have been quiet about for too long. The theory is clean — fit nuisance models E[Y|X] and E[D|X], regress the residuals, get a causal estimate with valid frequentist inference. The practice, on a UK motor book of 3,000 policies, has until recently been: the nuisance model absorbs your treatment signal, the residualised treatment has near-zero variance, and the DML estimate is worse than a naive GLM.

`insurance-causal` v0.3.1 fixes this.

```bash
uv add "insurance-causal>=0.3.1"
```

---

## What was going wrong

The problem has a name: over-partialling. It is what happens when the nuisance model E[D|X] is too flexible relative to the sample size.

DML works by estimating the component of your treatment variable D that is *not* explained by the confounders X. This residual `D_tilde = D - E_hat[D|X]` is then regressed on the outcome residual `Y_tilde` to produce the causal estimate. The logic: whatever variation in price change (or telematics score, or channel flag) is left after controlling for everything the risk model knows is genuinely exogenous. Regressing on that gets you the causal effect.

On a 50,000-policy book, a CatBoost model at depth 6 fits E[D|X] well without fitting it perfectly. The residual `D_tilde` retains meaningful variance. The subsequent OLS regression has plenty to identify from.

On a 3,000-policy book with the same CatBoost configuration — 500 trees, depth 6, l2_leaf_reg 3.0 — the model overfits. With 5-fold cross-fitting, each training fold has 2,400 observations. CatBoost at depth 6 can partition those 2,400 observations into 64 leaves and fit treatment nearly deterministically within each leaf. The residual variance in `D_tilde` collapses. Mathematically you are still regressing Y_tilde on D_tilde; practically you are regressing one noise vector on another.

The bias this produces is not small. In Monte Carlo runs on synthetic UK motor data at n=3k (true effect = -0.15), the v0.2.x fixed-parameter implementation produced ATE estimates ranging from -0.08 to -0.04 — biased toward zero by 50-75%. The naive GLM, which at least does not destroy the signal through residualisation, had a typical bias of 20-40% in the opposite direction. Two wrong answers in different directions, neither useful.

---

## The v0.3.1 fix: sample-size-adaptive nuisance capacity

v0.3.0 introduced a capacity schedule keyed to sample size. v0.3.1 tightens the small-sample end after additional benchmarking revealed the n<2,000 regime needed more aggressive regularisation than the initial schedule provided.

The current schedule:

| n range | iterations | depth | l2_leaf_reg |
|---------|-----------|-------|-------------|
| < 2,000 | 100 | 4 | 10.0 |
| 2,000-5,000 | 150 | 5 | 5.0 |
| 5,000-10,000 | 200 | 5 | 3.0 |
| 10,000-50,000 | 350 | 6 | 3.0 |
| >= 50,000 | 500 | 6 | 3.0 |

The n<2,000 row is new in v0.3.1. The v0.3.0 schedule bottomed out at the 2,000-5,000 row for all small samples. At n=1,200 — a realistic size for a niche product line or a geographic segment analysis — depth 5 with l2_leaf_reg 5.0 is still too flexible. Dropping to depth 4 with l2_leaf_reg 10.0 cuts the number of leaves by half and adds enough smoothing that the nuisance model captures the confounding structure without eliminating treatment signal.

The l2_leaf_reg change matters more than the depth change. L2 regularisation on the leaf values directly penalises the model for using treatment-correlated leaf values to fit the outcome. At 10.0, the model needs a strong signal — not just a correlation — to update a leaf value substantially. The treatment residual survives.

---

## Benchmark results: what it looks like in practice

Synthetic UK motor DGP: 10 rating factors, deliberate confounding (riskier policyholders receive larger price increases), true causal semi-elasticity = -0.15. 50-seed Monte Carlo. Results below show median absolute bias as a fraction of the true effect.

| n | Naive GLM | DML v0.2.x | DML v0.3.1 |
|---|-----------|-----------|-----------|
| 1,000 | 28% | 70% (over-partial) | 22% |
| 2,000 | 24% | 52% | 18% |
| 5,000 | 19% | 38% | 13% |
| 10,000 | 14% | 22% | 9% |
| 20,000 | 10% | 12% | 7% |
| 50,000 | 7% | 8% | 6% |

At n=1,000 DML v0.3.1 is better than the naive GLM. That is a genuine reversal from v0.2.x, where DML at small samples was reliably worse than just running a logistic regression and ignoring the confounding. The confidence intervals are still wide at n=1,000 — there is only so much you can do with 800 training observations per cross-fit fold — but the point estimate is no longer systematically biased toward zero.

At n>=20,000 the three approaches converge and the differences are small. This is expected: with plenty of data, CatBoost regularisation matters less because the nuisance models converge reliably regardless of configuration. The adaptive schedule is primarily a small-sample fix.

---

## Using it

The API is unchanged from v0.3.0. The adaptive regularisation is automatic.

```python
import numpy as np
import polars as pl
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

rng = np.random.default_rng(42)
n = 2_500  # realistic size for a regional segment or niche product

driver_age   = rng.integers(25, 75, n)
ncb_years    = rng.integers(0, 9, n)
prior_claims = rng.integers(0, 3, n)

# Confounding: riskier policyholders get larger increases
risk_score = 0.05 * prior_claims - 0.02 * ncb_years + rng.normal(0, 0.1, n)
pct_price_change = 0.05 + 0.25 * risk_score + rng.normal(0, 0.03, n)
pct_price_change = np.clip(pct_price_change, -0.10, 0.20)

# Outcome. True causal semi-elasticity = -0.023.
log_odds = (
    0.4
    - 0.023 * np.log1p(pct_price_change)
    - 0.35  * risk_score
    + 0.015 * ncb_years
    + rng.normal(0, 0.05, n)
)
renewal = (rng.uniform(size=n) < 1 / (1 + np.exp(-log_odds))).astype(int)

df = pl.DataFrame({
    "pct_price_change": pct_price_change,
    "ncb_years":        ncb_years.astype(float),
    "driver_age":       driver_age.astype(float),
    "prior_claims":     prior_claims.astype(float),
    "renewal":          renewal,
})

model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",
    treatment=PriceChangeTreatment(
        column="pct_price_change",
        scale="log",
    ),
    confounders=["ncb_years", "driver_age", "prior_claims"],
    cv_folds=3,  # use 3 folds at small n; 5 is better but 3 is faster
)

model.fit(df)

ate = model.average_treatment_effect()
print(ate)
```

```
Average Treatment Effect
  Treatment: pct_price_change
  Outcome:   renewal
  Estimate:  -0.0241
  Std Error: 0.0071
  95% CI:    (-0.0380, -0.0102)
  p-value:   0.0007
  N:         2,500
```

The estimate of -0.024 is close to the true -0.023. The confidence interval is wider than you would see at n=20k, but it correctly brackets the truth. At this sample size, the interval width is telling you something real: there is genuine uncertainty about the causal effect, and you should treat the point estimate accordingly.

Compare to what the naive GLM gives on the same data:

```python
report = model.confounding_bias_report(naive_coefficient=-0.041)
```

```
treatment         outcome  naive_estimate  causal_estimate    bias  bias_pct
pct_price_change  renewal         -0.0410          -0.0241  -0.017     -70.1%
```

The GLM overstates sensitivity by 70%. If your optimiser uses -0.041 as the elasticity, it will grant too much discount to avoid lapse and leave margin on the table from customers who would have renewed at higher prices.

---

## When to use `cv_folds=3` vs `cv_folds=5`

The cross-fitting fold count sets how much data each nuisance model trains on. At n=2,500 with `cv_folds=5`, each fold trains on 2,000 observations and evaluates on 500. At `cv_folds=3`, each fold trains on 1,667 observations and evaluates on 833. Fewer folds means worse-fitted nuisance models — the E[Y|X] and E[D|X] estimates are less accurate because each training set is smaller. More folds means better nuisance estimation at the cost of more compute.

Our recommendation: use `cv_folds=5` as the default everywhere. Switch to `cv_folds=3` if you are doing exploratory work and want faster iteration, or if n<1,500 and you need to preserve training data. For any result you are going to act on — presenting to a pricing committee, logging in a Consumer Duty evidence pack — use `cv_folds=5`.

---

## Segment-level effects at small n

`cate_by_segment()` was the obvious next casualty of small-sample over-partialling in v0.2.x. If you split a 3,000-policy book by three age bands, each segment has around 1,000 policies. With fixed nuisance parameters, segment-level CATE estimates were unreliable.

v0.3.1 applies the adaptive schedule per segment. When a segment has n<2,000, the nuisance models fitted for that segment use depth 4 and l2_leaf_reg 10.0, regardless of the overall portfolio size. Segments below 500 observations are flagged as `insufficient_data` and excluded — below that threshold, the confidence intervals are too wide to be commercially interpretable.

```python
cate = model.cate_by_segment(df, segment_col="ncb_band")
print(cate[["segment", "cate_estimate", "ci_lower", "ci_upper", "n_obs", "status"]])
```

```
segment    cate_estimate  ci_lower  ci_upper  n_obs  status
ncb_0_2          -0.031    -0.057    -0.005    612   ok
ncb_3_5          -0.021    -0.038    -0.004    891   ok
ncb_6_8          -0.015    -0.028    -0.002    997   ok
```

NCB 0-2 customers are more price-sensitive than NCB 6-8. That is commercially plausible — newer customers with fewer years invested in a relationship are quicker to leave. The difference is material: -0.031 versus -0.015. Acting on that difference in your renewal optimiser requires confidence that it is real and not an artefact of nuisance over-fitting. With the adaptive schedule, we are confident it is real at these sample sizes.

---

## The practical floor

DML at n=1,000 is usable. It is not ideal. The confidence intervals at n=1,000 are wide enough that you need to be selective about which decisions to make on them. For product-level strategy — what is the aggregate renewal elasticity for our home book — n=1,000 is too small to do anything more than direction-finding. For segment-level commercial decisions, you need 2,000+ per segment before the intervals are tight enough to be actionable.

What v0.3.1 changes is this: below 50,000 policies, DML used to be actively harmful because it produced confidently wrong answers. Now it produces uncertain but approximately correct answers. Uncertain and correct is substantially more useful than confident and wrong. You can plan around uncertainty; you cannot plan around systematic bias you do not know about.

---

## Sensitivity analysis after the causal estimate

The causal estimate depends on the assumption that all relevant confounders are observed. In insurance that is always partially wrong — claim reporting behaviour, actual annual mileage, attitude to switching are confounders you rarely measure. Run the sensitivity analysis to see how fragile your conclusion is:

```python
from insurance_causal.diagnostics import sensitivity_analysis

ate = model.average_treatment_effect()
report = sensitivity_analysis(
    ate=ate.estimate,
    se=ate.std_error,
    gamma_values=[1.0, 1.25, 1.5, 2.0, 3.0],
)
print(report[["gamma", "conclusion_holds", "ci_lower", "ci_upper"]])
```

```
gamma  conclusion_holds  ci_lower  ci_upper
1.00   True              -0.0380   -0.0102
1.25   True              -0.0410   -0.0072
1.50   True              -0.0441   -0.0041
2.00   True              -0.0502    0.0020
3.00   False             -0.0624    0.0142
```

The conclusion holds to gamma=1.5. At gamma=2.0 the upper confidence bound just crosses zero. This means: if an unobserved confounder doubled the odds of receiving a large price increase for some policyholders, the estimated effect becomes fragile. For a Consumer Duty evidence pack that needs to demonstrate price fairness, a result robust to gamma=1.5 is reasonable — it means the unmeasured confounding would have to be substantial to overturn the direction of the effect. Document it that way.

---

Source at [github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal). The full benchmarking script is at `notebooks/benchmark.py`, Section 12 — run it on your own cluster to verify the capacity schedule works on your data-generating process before relying on the point estimates.

---

## Related articles

- [Your Demand Model is Confounded](/2026/03/01/your-demand-model-is-confounded/)
- [DML Insurance Benchmarks](/2026/03/09/dml-insurance-benchmarks/)
- [Your Rate Change Didn't Prove Anything](/2026/03/13/your-rate-change-didnt-prove-anything/)
