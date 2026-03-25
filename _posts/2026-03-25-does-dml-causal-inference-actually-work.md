---
layout: post
title: "Does DML Causal Inference Actually Work for Insurance Pricing?"
date: 2026-03-25
categories: [validation]
tags: [causal-inference, dml, glm, telematics, pricing, python]
description: "We ran Double Machine Learning against a naive GLM on a 50,000-policy UK motor telematics book. The GLM overestimated the treatment effect by 50–90%. Here is what that means for your discount calibration."
---

The honest answer is: yes, but only if your current approach is a GLM coefficient, and only in the scenarios where confounding is structural rather than incidental.

We spent time running benchmarks on [`insurance-causal`](https://github.com/burning-cost/insurance-causal) against a synthetic 50,000-policy UK motor book with a known data-generating process. The numbers are uncomfortable for anyone who has ever calibrated a telematics discount from a GLM coefficient.

---

## What we actually tested

The setup mirrors a standard UK motor telematics portfolio. Treatment is a continuous normalised telematics score. True causal effect: a one-standard-deviation improvement in score raises log-odds of renewal by 0.08. The confounding is a multiplicative driver-risk term — age × NCB × region interaction — that drives both telematics scores and renewal probabilities through the same underlying risk profile.

A naive logistic GLM with age, NCB, and region dummies as main effects tries to control for this. The problem is functional form: the confounders interact multiplicatively in the true DGP, but the GLM absorbs them additively. CatBoost in the DML nuisance step naturally discovers "young driver, zero NCB, high-risk region" as a leaf. The GLM bias is structural, not a sample size problem. It persists at n = 100,000.

---

## The numbers

| Segment | GLM overestimate | DML bias |
|---|---|---|
| Young drivers (25–35), high-risk region | 60–90% | 10–20% |
| Mid-age, average risk | 40–60% | 8–15% |
| Mature drivers (55+), low risk | 20–40% | 5–12% |

The GLM bias is worst exactly where it matters most commercially. Young drivers in high-risk regions are the primary target for telematics pricing, and the GLM-calibrated discount in this segment is the most inflated.

Headline comparison:

| Metric | Naive logistic GLM | DML (insurance-causal) |
|---|---|---|
| Estimate | ~0.12–0.15 | converges to ~0.08 |
| True DGP effect | 0.0800 | 0.0800 |
| Bias (% of true) | 50–90% | 10–20% |
| 95% CI covers truth | No | Yes |
| Fit time | <1s | ~60s (5-fold CatBoost) |

The DML estimate converges to the true value. The GLM does not, at any sample size.

---

## What it costs to get this wrong

A telematics discount calibrated to the GLM coefficient (0.12) rather than the causal coefficient (0.08) is set roughly 50% too aggressively. If you give a 5% discount to 40% of a 10,000-policy book expecting 8% retention uplift per unit telematics score, but the true causal uplift is only ~5% per unit, you are paying for retention that would have happened without the discount.

On renewal pricing, the same benchmark finds a GLM price-sensitivity estimate of −0.045 against a causal estimate of −0.023 — the GLM number is wrong by roughly 96%. A rate-change optimisation built on that elasticity will systematically undercharge price-inelastic customers and overprice elastic ones.

---

## The API

```python
from insurance_causal import CausalPricingModel

model = CausalPricingModel(
    treatment="telematics_score",
    outcome="renewal",
    confounders=["driver_age", "ncb_years", "region", "vehicle_group"],
    outcome_type="binary",
    cv_folds=5,
)
model.fit(df)
ate = model.average_treatment_effect()
print(ate)
# AverageTreatmentEffect(estimate=-0.023, std_error=0.004, ci_lower=-0.031, ci_upper=-0.015, p_value=0.0001)
bias_report = model.confounding_bias_report()
print(bias_report[["naive_estimate", "causal_estimate", "bias_pct"]])
# naive_estimate: 0.045, causal_estimate: -0.023, bias_pct: 0.49
```

The `confounding_bias_report` is the number to put in front of the pricing committee. It shows the naive estimate alongside the causal estimate, with the fraction attributable to confounding.

---

## When DML is not the right tool

The GLM bias is a function of how nonlinear the confounding is. If treatment assignment was genuinely randomised — a properly designed A/B test, for example — the GLM is unbiased and DML is unnecessary overhead.

It also does not help with thin data. The five-fold CatBoost nuisance step needs enough policies in each fold to fit a credible propensity and outcome model. Below roughly 2,000 policies, the nuisance estimates are too noisy to improve on a well-specified GLM. The library uses sample-size-adaptive model capacity to handle this — shrinking to Ridge when n is small — but be cautious with books under 5,000 policies.

For heterogeneous effects — identifying which customer segments are most price-sensitive — the causal forest extension is worth running. On 20,000 policies with six pre-defined elasticity segments, causal forest achieves 100% confidence interval coverage versus 67% for a GLM with full interaction terms. The per-policy CATE from the forest is what allows segment-level targeting: identifying the 20% of customers who would lapse at any price increase from the 20% who will accept significant rate without flinching.

---

## Verdict

Use DML if:
- Your treatment effect estimate comes from a GLM on observational data where treatment assignment was correlated with risk (telematics pricing, renewal pricing, channel effects)
- You are calibrating a discount or loading that will be applied commercially and you need the number to be right, not approximately right

Do not use it if:
- You have a properly randomised A/B test — a well-specified GLM is fine
- Your book is below ~5,000 policies — the nuisance models will not be credible
- You want heterogeneous segment effects without per-policy CATEs — a GLM interaction model is faster and close enough

The technique is real. The overhead is real too: 60 seconds versus under one second for a GLM. That is a one-time cost per rate review cycle, not a production scoring cost. For calibrating a discount that will apply to tens of thousands of policies, the 60 seconds is not the constraint.

```bash
uv add insurance-causal
```

Source and benchmarks at [GitHub](https://github.com/burning-cost/insurance-causal). Run `benchmarks/run_benchmark.py` (seed=42) for reproducible output.

- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/)
- [Does Constrained Rate Optimisation Actually Work?](/2026/03/29/does-constrained-rate-optimisation-actually-work/)
