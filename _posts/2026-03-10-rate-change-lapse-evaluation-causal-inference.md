---
layout: post
title: "Rate Change Evaluation: Did the Premium Increase Cause the Lapses?"
date: 2026-03-10
author: Burning Cost
categories: [pricing, causal-inference, libraries, case-study]
description: "A 12% rate increase on young motor drivers. An 8% lapse spike three months later. Here is how to tell whether the rate change caused it  -  using synthetic difference-in-differences."
tags: [causal-inference, sdid, synthetic-control, did, rate-change, lapses, retention, fca-tr242, insurance-causal-policy, motor, uk-motor, python, polars, consumer-duty]
---

Q3 2025. Your pricing team applied a 12% technical increase to young driver segments  -  17-25, direct channel, annual policies, minimum NCD. The increase went live in July. By October, the lapse rate on that segment had jumped from 11% to 19%. Your head of pricing is in front of the underwriting committee being asked the obvious question: did we cause this?

The honest answer, if you have only done a before/after comparison, is that you do not know. You know that lapses went up after the rate change. You do not know whether the rate change caused them to go up.

This matters for two reasons. First, Consumer Duty: FCA TR24/2 (August 2024) specifically called out insurers who could not demonstrate causal attribution between pricing interventions and customer outcomes. Lapse spikes following rate changes are exactly the kind of outcome the FCA expects you to be able to explain. Second, the business decision: if young driver lapses were going to spike anyway  -  market repricing, competitor behaviour, end-of-aggregate-scheme  -  applying a remedial rate cut to stop the bleed is the wrong response.

This post walks through how to answer the question properly, using [`insurance-causal-policy`](https://github.com/burning-cost/insurance-causal-policy). The scenario is synthetic but the structure mirrors situations we have seen on UK motor books.

---

## Why the before/after comparison fails here

The instinctive analysis is to plot the lapse rate on your young driver segment from January 2025 through December 2025 and annotate the July rate change. The lapse rate is flat at 10–12% through Q1 and Q2, then climbs to 19% in Q3.

That chart proves correlation. It does not prove causation. Three alternative explanations fit the same data equally well:

**Market-wide lapse pressure.** Q3 2025 was when several aggregator platforms updated their eligibility criteria, increasing the range of quotes available to young drivers. Renewal quote diversity increased across the board. A segment that was paying 10% lapses in a thin market might pay 18% when comparison shopping becomes easier  -  regardless of your rate action.

**Seasonal effects.** Young drivers have higher lapse rates in Q3 and Q4 as a matter of normal portfolio seasonality  -  end of summer coincides with life events (moving, car sales, income changes) that drive mid-term cancellations and non-renewals. If your rate change happened to land in July, the Q3 spike looks like causation when it is partly seasonality.

**Competitor repricing.** If a competitor cut rates on young drivers in June 2025  -  which at least one comparison site confirmed for tier-1 providers  -  your rate increase was doubly punishing: you raised while others fell. The lapse spike reflects the relative price change, not just your absolute increase. A before/after chart on your own data cannot separate this.

The only way to separate your causal effect from these alternatives is to construct a counterfactual: what would the lapse rate have been on this segment if you had not changed rates? And the only credible way to construct that counterfactual is with a control group  -  segments that did not receive the rate change and whose trends you can use to infer what would have happened to the treated segment.

---

## The setup: retention as the outcome variable

The outcome metric matters  -  and for lapse evaluation it is not loss ratio.

Loss ratio is the natural outcome for evaluating rate adequacy. But for lapse evaluation, the outcome is **retention rate**: the proportion of policies eligible for renewal that renew. Alternatively, for mid-term lapses, it is the cancellation rate within the policy year.

`insurance-causal-policy` supports retention directly as a built-in outcome. You need two additional columns in your segment-period panel: `policies_renewed` and `policies_due`.

```bash
uv add insurance-causal-policy
```

```python
import polars as pl
from insurance_causal_policy import (
    PolicyPanelBuilder,
    SDIDEstimator,
    compute_sensitivity,
    plot_event_study,
    plot_synthetic_trajectory,
    FCAEvidencePack,
)

# policy_df: one row per segment per quarter
# Columns: segment_id, period, earned_premium, earned_exposure,
#          policies_renewed, policies_due
#
# claims_df: not used for retention outcome but still required for panel build
# rate_log_df: segment_id, first_treated_period

builder = PolicyPanelBuilder(
    policy_df=policy_df,
    claims_df=claims_df,
    rate_log_df=rate_log_df,
    outcome="retention",   # uses policies_renewed / policies_due
    min_exposure=50.0,     # exclude cells with fewer than 50 exposure units
)
panel = builder.build()
print(builder.summary())
```

The panel summary tells you what you are working with before you fit anything:

```
{
  'n_segments': 87,
  'n_treated_segments': 22,    # young driver segments that got the rate increase
  'n_control_segments': 65,    # older driver segments, unaffected by the action
  'n_periods': 10,             # Q1 2023 through Q2 2025 (8 pre) + Q3, Q4 2025 (2 post)
  'n_cells': 870,
  'pct_nonzero_exposure': 98.6,
  'pct_treated_cells': 25.3,
}
```

22 treated segments and 65 controls is a good ratio for SDID  -  the constrained weight optimisation has enough donors to construct a credible synthetic control. The 98.6% non-zero exposure cells tells you the panel is well-populated; sparse cells (if present) are typically fringe segments such as very young drivers in thin geographic areas.

---

## The parallel trends assumption and pre-treatment test

SDID's identifying assumption is approximate parallel trends: in the absence of the rate change, treated and control segments would have evolved similarly. Before interpreting any result, you need to test whether this holds in the pre-treatment window.

```python
est = SDIDEstimator(
    panel,
    outcome="retention",
    inference="placebo",
    n_replicates=200,
)
result = est.fit()

# Pre-treatment diagnostic first
from insurance_causal_policy import pre_trend_summary
pt = pre_trend_summary(result)
print(pt)
```

```
Pre-treatment parallel trends test
-----------------------------------
Periods tested: Q1 2023 – Q2 2025 (8 periods)
Joint F-statistic: 1.34  (p = 0.267)
Max single-period deviation: -0.008 (Q3 2024, not significant)
Verdict: PASS
```

A p-value of 0.267 on the joint test is reassuring. The treated and control segments were tracking together before the rate change, once SDID's reweighting has been applied. If this had failed  -  p below 0.10  -  you would need to investigate whether there is a structural reason the young driver segments were diverging before Q3 2025, and whether you can fix it (additional controls, a different comparison group) or whether the causal design is simply not valid.

The event study plot is the visual complement:

```python
plot_event_study(result, zero_at="treatment")
```

This shows the period-by-period gaps between treated and synthetic control. In the pre-treatment window, the gaps should scatter around zero. Post-treatment, you are looking for a step change. If the pre-treatment gaps already show a trend, the parallel trends test result was too generous  -  look at the plot, not just the p-value.

---

## Fitting SDID and reading the result

```python
print(result.summary())
```

```
SDID Estimation Results
-----------------------
Outcome:          retention
Treated segments: 22
Control segments: 65 (effective: 28)
Pre-treatment:    8 periods
Post-treatment:   2 periods

ATT:   -0.0631
SE:     0.0189
95% CI: [-0.1001, -0.0261]
p-value: 0.0008

Pre-trend test: PASS (p=0.267)
Inference: placebo (200 replicates)
```

The ATT is -0.0631. In plain terms: the rate increase caused a 6.3 percentage point reduction in retention on the treated segments, after accounting for everything that would have happened without the rate change. The 95% confidence interval is [-10.0pp, -2.6pp]  -  the effect is real, it is statistically well-identified, and it is not trivially small.

The nominal lapse spike was 8pp (11% to 19%). The SDID estimate says 6.3pp of that is attributable to the rate change. The remaining 1.7pp would have happened anyway  -  market conditions, seasonality, and the competitor repricing effect.

This matters for the decision. Management asked whether the rate change caused the lapse spike. The answer is: yes, most of it. Not all of it. The "effective control segments" count of 28  -  out of 65 donors  -  reflects how SDID concentrates weight: 28 control segments were doing most of the work in constructing the synthetic counterfactual. Check which ones:

```python
from insurance_causal_policy import plot_unit_weights
plot_unit_weights(result, top_n=15)
```

If the high-weight controls are segments you would endorse as valid comparators  -  similar underlying risk profiles, just not subject to the young driver increase  -  the construction is defensible. If the top controls are implausible (e.g., commercial fleet segments), that is a signal to revisit the donor pool definition.

---

## Visualising the counterfactual

The most intuitive output for a committee presentation is the synthetic trajectory plot:

```python
plot_synthetic_trajectory(
    result,
    title="Young driver retention: treated vs synthetic control",
    treatment_label="Post rate increase (Q3 2025+)",
)
```

This plots three lines: the actual retention rate of the treated segments, the synthetic control trajectory (what we estimate would have happened without the rate change), and the gap between them. The gap in the post-treatment window is the ATT. The synthetic control continues the pre-treatment pattern of the control group  -  in this case, a mild seasonal dip through Q3 that returns in Q4, which is what the young driver segments would have experienced without the intervention.

The chart makes the counterfactual argument concrete. The lapse rate was going to dip slightly regardless. The rate change pushed it substantially further.

---

## Sensitivity analysis: how robust is the conclusion?

An ATT of -6.3pp with a tight confidence interval is encouraging. But the inference rests on parallel trends approximately holding in the post-treatment window  -  the very window we cannot directly test because the treatment happened. The sensitivity analysis asks: how large would a post-treatment violation need to be before the conclusion reverses?

```python
sens = compute_sensitivity(result, m_values=[0.0, 0.5, 1.0, 1.5, 2.0])
print(sens.summary())
```

```
Sensitivity Analysis (Rambachan-Roth 2023)
-------------------------------------------
M = 0.0:  ATT bounds [-0.1001, -0.0261]  (identifies zero? No)
M = 0.5:  ATT bounds [-0.1096, -0.0166]  (identifies zero? No)
M = 1.0:  ATT bounds [-0.1191, -0.0071]  (identifies zero? No)
M = 1.5:  ATT bounds [-0.1286,  +0.0024] (identifies zero? Yes)
M = 2.0:  ATT bounds [-0.1381,  +0.0119] (identifies zero? Yes)

Breakdown point M*: ~1.4 pre-period SDs
```

The result is robust to post-treatment parallel trends violations up to 1.4 times the observed pre-period variation. Only when you allow violations 1.5x larger than anything observed in the pre-treatment window does the confidence interval start to include zero.

For FCA purposes, this is the right level of robustness to demonstrate. The claim is not that parallel trends holds exactly  -  it is that the conclusion only changes under an implausibly large violation.

---

## The FCA evidence pack

The analysis above answers the business question. For Consumer Duty outcome monitoring documentation, you need it in a structured format that a compliance reviewer can sign off.

```python
pack = FCAEvidencePack(
    result=result,
    sensitivity=sens,
    product_line="Motor",
    rate_change_date="2025-Q3",
    rate_change_magnitude="+12% technical premium, young drivers (17-25) direct channel",
    analyst="Pricing Team",
    panel_summary=builder.summary(),
)

# Markdown for internal docs
report_md = pack.to_markdown()

# JSON for audit trail / system ingestion
report_json = pack.to_json()
```

The Markdown report includes: the ATT estimate with CI; the pre-trend test result; the sensitivity breakdown point; a structured caveats section covering IBNR lag (not relevant here since we are using retention, not loss ratio), thin cells, and the parallel trends assumption; and a plain-English conclusion paragraph.

The FCA has not endorsed SDID as a specific evidence standard. What TR24/2 asked for is a credible counterfactual and documented assumptions. This pack provides both.

---

## What the result actually tells management

The analysis produces a defensible answer to the original question:

The 12% rate increase on young drivers in Q3 2025 caused approximately 6.3 percentage points of the observed 8 percentage point lapse spike. The remaining 1.7pp would have occurred regardless, driven by market conditions including increased quote availability on comparison platforms and competitor repricing in the segment.

The rate change was the primary cause of the lapses. It was not the sole cause.

That distinction matters for the remedial decision. If the full 8pp spike was attributable to the rate change, pricing team credibility depends on either justifying the increase or partially reversing it. If 1.7pp was going to happen anyway, a partial reversal to bring the increment-attributable lapse rate back to an acceptable level requires a smaller rate correction than the naive view implies. The causal estimate gives you the number to plug into the demand model.

It also matters for the FCA narrative. "Our rate increase caused some additional lapses and we have quantified the effect and it is within our Consumer Duty acceptable outcome range" is a defensible regulatory position. "Our loss ratios justified the increase, the lapses happened, we are not sure why" is not.

---

## When to use SDID vs DRSC

The scenario above has 65 control segments  -  large enough for SDID's constrained weight optimisation (CVXPY) to be well-identified. The benchmark on 40 donors shows SDID and DRSC performing equivalently on RMSE.

If your young driver book is smaller and you only have 8–12 plausible control segments, use `DoublyRobustSCEstimator` instead:

```python
from insurance_causal_policy import DoublyRobustSCEstimator

est_dr = DoublyRobustSCEstimator(
    panel,
    outcome="retention",
    inference="bootstrap",
    n_replicates=500,
)
result_dr = est_dr.fit()
print(result_dr.summary())
# DRSC estimate: -0.0614 decrease in retention (95% CI: -0.0941 to -0.0287, p=0.000)
```

DRSC uses unconstrained OLS weights (negative weights are permitted), which reduces variance when the donor pool is small and the SC weight problem is poorly conditioned. The DR property  -  consistency under either parallel trends or synthetic control identification  -  is the safety net when you cannot be sure which assumption is the right one.

The decision rule from the benchmarks (2026-03-21, Databricks serverless, 100 simulations):

- Fewer than 10 control segments: use DRSC
- 20 or more control segments: use SDID
- 10–19 control segments: run both; if the point estimates differ by more than one standard error, examine the SC weight diagnostics before drawing conclusions

---

## What to do when parallel trends fails

One more situation to handle honestly: the pre-treatment test fails. This happens when:

- The rate change was applied precisely to the segments that were already deteriorating  -  adverse selection into treatment
- A concurrent event (GIPP pricing reform, Ogden rate change) affected young drivers differently from older drivers in the pre-treatment window
- Your segment definitions are not clean  -  some "control" segments were adjacent to treated ones and absorbed price-sensitive risks via mid-term adjustments

If the pre-trend test fails, the SDID estimate is still the best available estimate of the treatment effect, but you should not present it to the FCA as a clean causal result. The sensitivity analysis becomes more important: even with a pre-trend failure, you can ask how large the violation must be to overturn the conclusion. If the breakdown point is generous (M* > 2), the result may still be usable with appropriate caveats. If the breakdown point is below 0.5  -  violations smaller than half the pre-period variation reverse the sign  -  the causal design is not reliable and you need a different approach (external benchmarks, structural demand model, or a different control group definition).

---

## Case study series

This is the third post in the case study series:

- [Case study #1: Extracting rating factors from CatBoost for Radar](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/)  -  SHAP relativities to factor tables, end-to-end distillation workflow
- [Case study #2: Motor model mispricing caught by monitoring](/2026/03/23/motor-model-mispricing-caught-by-monitoring/)  -  how PSI, A/E, and Gini drift caught a Thatcham reclassification failure before the half-year review
- Case study #3 (this post): causal evaluation of whether a rate change caused a lapse spike

The common thread: decisions that look obvious from aggregate data fall apart under scrutiny, and the right tool changes what you conclude.

---

`insurance-causal-policy` is on PyPI at v0.2.0. Install with `uv add insurance-causal-policy`. Full worked example at [`causal_rate_change_evaluation.py`](https://github.com/burning-cost/burning-cost-examples/blob/main/examples/causal_rate_change_evaluation.py).

- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/)  -  the `insurance-optimise` library for prospective demand modelling; this post covers retrospective causal evaluation of rate changes already made
- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/)  -  the monitoring framework that catches model deterioration between formal causal evaluations
