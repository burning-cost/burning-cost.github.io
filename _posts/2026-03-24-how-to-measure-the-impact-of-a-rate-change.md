---
layout: post
title: "How to Measure the Impact of a Rate Change"
date: 2026-03-24
categories: [techniques]
tags: [causal-inference, rate-change, did, monitoring, sdid, insurance-causal-policy, difference-in-differences, parallel-trends, uk-motor, consumer-duty]
description: "Six months after your 5% rate increase, the loss ratio looks better. Here is how to find out whether the rate change caused it — and how to avoid the analysis mistakes that produce the wrong answer."
---

Six months ago you pushed a 5% rate increase across your motor book. The board presentation is due. Your loss ratio has improved 4 points. Someone has made a slide with a before/after chart.

Here is what you do not yet know: whether the rate change caused the improvement.

That is not pedantry. If claims frequency was already falling — driven by changing driving patterns, competitor exits, or a quiet claims winter — your loss ratio would have improved regardless of what you did. You are about to present correlation as evidence. The FCA's TR24/2 (August 2024) reviewed how UK insurers evaluate pricing interventions and found exactly this: before/after comparisons without counterfactuals do not constitute adequate evidence under Consumer Duty outcome monitoring.

This post is about how to actually measure what happened.

---

## Why the before/after comparison fails

The before/after comparison assumes everything else stayed constant. In insurance, almost nothing stays constant over a six-month window.

Suppose you applied a 5% rate increase to personal lines motor in October 2025. By April 2026, your loss ratio on that book is down 3.5 points. The before/after slide looks compelling. But consider what else happened between October and April:

- Claims frequency in UK motor fell 4% in Q4 2025 due to changes in driving patterns post-COVID correction (ONS transport survey, December 2025).
- Two major aggregator price comparison algorithms changed ranking weights in November, reducing adverse selection across the board.
- Your book mix shifted: the rate increase caused higher-risk segments to lapse at a higher rate, so the surviving book is lighter on risk even before the rate mechanics play through to loss ratio.

Each of those is a plausible confound. The before/after comparison cannot distinguish the causal effect of your rate change from any of them.

The only way to separate your causal effect from the counterfactual trend is to have a comparison group — a segment of your book (or a comparable line) that did not receive the rate change and that was subject to the same market forces. You observe what happened to that group over the same window, infer what would have happened to your treated segment without the rate change, and the difference is your estimate of the causal effect.

This is Difference-in-Differences. Done properly, it is the right method for post-hoc rate change evaluation.

---

## When you have a natural control group

Most rate changes are not portfolio-wide. You change rates on motor direct but not on motor broker. You increase rates on young drivers but hold on mature segments. You push through on home but not on motor.

If you have segments that did not receive the rate change and that face similar market conditions, you have a natural control group. Use it.

The key assumption is **parallel trends**: in the absence of the rate change, your treated and control segments would have followed the same trend in the outcome metric. This is an assumption, not a fact, and it cannot be proven — but you can test whether pre-treatment trends were parallel, which gives you evidence that the assumption is plausible.

[`insurance-causal-policy`](https://github.com/burning-cost/insurance-causal-policy) implements Synthetic Difference-in-Differences (SDID, Arkhangelsky, Athey, Hirshberg, Imbens and Wager, 2021) for exactly this setup. SDID improves on standard DiD by constructing optimal weights over control units and time periods, which handles the common insurance problem that your treated and control segments have different mean loss ratios — standard DiD with a fixed control group is biased when treated and controls start at different levels; SDID's unit weights absorb level differences so the identification comes from trend similarity, not level similarity.

### Setting up the panel

SDID requires a balanced segment × period panel. Your raw data is policy-level. The `PolicyPanelBuilder` handles the aggregation:

```python
import polars as pl
from insurance_causal_policy import PolicyPanelBuilder, SDIDEstimator

# policy_df: one row per policy, with segment_id, period,
#            earned_premium, earned_exposure
# claims_df: one row per policy-period, with segment_id, period,
#            incurred_claims, claim_count
# rate_log_df: one row per treated segment, with segment_id,
#              first_treated_period

builder = PolicyPanelBuilder(
    policy_df,
    claims_df,
    rate_log_df,
    outcome="loss_ratio",
    min_exposure=50.0,   # warn on thin cells
)
panel = builder.build()
print(builder.summary())
# {'n_segments': 84, 'n_treated_segments': 31, 'n_control_segments': 53,
#  'n_periods': 12, 'pct_treated_cells': 28.3, ...}
```

Loss ratio here is `incurred_claims / earned_premium` aggregated at the segment-period level — not the mean of individual policy loss ratios. The distinction matters when exposure varies across cells, which it always does.

### Fitting the SDID estimator

```python
est = SDIDEstimator(panel, outcome="loss_ratio", inference="placebo")
result = est.fit()
print(result.summary())
# ATT: -0.035  SE: 0.009  CI: [-0.052, -0.018]  p=0.0001
# Pre-trend test: PASS (p=0.412)
# Treated: 31 segments, Control (effective): 22 segments
# Pre-treatment periods: 8, Post-treatment: 4
```

The ATT of -0.035 means the rate change caused a 3.5 percentage point reduction in the loss ratio, after accounting for what would have happened without the rate change. The pre-trend test passing (p=0.412) gives you evidence that the parallel trends assumption is plausible — the treated and control segments were moving together before the rate change.

The 95% CI of [-0.052, -0.018] excludes zero. The improvement is real. The before/after chart was right that something improved; now you know whether it was the rate change.

### Sensitivity analysis

Parallel trends cannot be proven, only made plausible. How robust is the estimate to violations of the assumption?

```python
from insurance_causal_policy import compute_sensitivity, plot_sensitivity

sens = compute_sensitivity(result)
# M=0.0: ATT bounds [-0.052, -0.018]  (no violation allowed)
# M=1.0: ATT bounds [-0.061, -0.009]  (still excludes zero)
# M=1.9: ATT bounds [-0.070,  0.001]  (zero enters identified set at M≈1.8)
# Breakdown point M*: ~1.8 pre-period standard deviations
plot_sensitivity(sens)
```

The breakdown frontier is 1.8 pre-period standard deviations of parallel trends violation. If you believe any post-treatment confounding is smaller than 1.8 times the pre-treatment trend discrepancies you observed, the sign of the effect is robust. In practice this means you document the assumption, flag the breakdown point, and let the reader assess whether the threats to validity are plausible.

---

## When there is no natural control group

Sometimes the rate change was portfolio-wide. You increased rates across all channels, all segments, all regions. There is no untouched segment to use as a control.

This is a harder identification problem. You cannot run SDID without control units. Three approaches are available, each with honest limitations.

**Use an external benchmark.** Market-wide loss ratio indices (Lloyd's, ABI Motor Statistics, or broker-reported aggregates) give you a counterfactual trend independent of your book. The assumption is that your book would have tracked the market in the absence of the rate change. This assumption is stronger than parallel trends within your own book: your book is not the market, and it differs on risk profile, channel mix, and underwriting appetite. Document the assumption carefully and bound the estimate by the plausible deviation from market trend.

**Use a closely related but untreated product line.** If you increased motor rates but held home rates flat, home might serve as a control — if the same macro drivers (claims environment, operational costs, competitor behaviour) affect both lines similarly. The validity depends on whether your treated and control lines genuinely share a common trend. Underwriters typically have a strong prior on this. Use it.

**Use the Doubly Robust Synthetic Control estimator with a thin donor pool.** When you have even a small number of untreated segments (5–15), `insurance-causal-policy` includes a `DoublyRobustSCEstimator` that provides formal double robustness: it gives a consistent ATT estimate if either the synthetic control weights correctly reproduce the treated counterfactual trend, or if parallel trends holds after reweighting. The DR property is a safety net when the donor pool is small and SC weights are imprecisely estimated.

```python
from insurance_causal_policy import DoublyRobustSCEstimator

est = DoublyRobustSCEstimator(
    panel,
    outcome="loss_ratio",
    inference="bootstrap",
)
result = est.fit()
print(result.summary())
# ATT: -0.031  SE: 0.012  CI: [-0.054, -0.008]  p=0.009
# DR property: active (parallel trends AND SC weights both contribute)
```

What you should **not** do is run an interrupted time series model on a single time series — your own portfolio's monthly loss ratio — and interpret the level shift or slope change as the causal effect of the rate change. An ITS on a single portfolio time series is a before/after comparison dressed up with regression. It cannot separate market-wide trends from your intervention unless the model explicitly includes external controls, in which case you are doing DiD by another name. We are not recommending this approach.

---

## Staggered rate changes

In practice, rate changes are rarely applied at a single point in time across a clean set of segments. More commonly, you phase a rate increase: direct motor went live in Q1, broker motor in Q2, fleet in Q3. Each cohort has a different treatment date.

Standard two-way fixed effects (TWFE) is known to produce badly biased estimates — sometimes with the wrong sign — under staggered treatment adoption with heterogeneous treatment effects (Callaway and Sant'Anna, 2021). If your pricing team is running a simple panel regression with treatment dummies across a staggered rollout, the estimate is likely to be wrong.

`insurance-causal-policy` includes a `StaggeredEstimator` that uses the Callaway and Sant'Anna approach: it estimates a group-time ATT for each cohort × post-treatment period combination, using only clean controls (never-treated or not-yet-treated segments), and aggregates them to an overall ATT.

```python
from insurance_causal_policy import StaggeredEstimator

est_s = StaggeredEstimator(panel, outcome="loss_ratio")
result_s = est_s.fit()
print(f"Overall ATT: {result_s.att_overall:.4f} (SE: {result_s.se_overall:.4f})")
print(f"Cohorts: {result_s.n_cohorts}")
# Overall ATT: -0.0339 (SE: 0.0091)
# Cohorts: 3
```

The cohort-time estimates are also available, which lets you check whether the effect is homogeneous across cohorts or whether early-treated segments saw a different effect than later-treated ones. In a price increase scenario, heterogeneity often reflects differences in competitor response timing.

---

## Practical warnings

**Minimum pre-treatment periods.** The SDID pre-trend test needs at least 8 pre-treatment periods to be credible. With 4 or fewer, the parallel trends assumption cannot be meaningfully assessed. The library will warn at fewer than 3 periods. If you only have 2 quarters of pre-treatment data, you should report this as a limitation, not suppress the warning.

**Do not evaluate too early.** Six months after a rate change is probably too early for loss ratio evaluation. Claims IBNR is substantial at 6 months for liability and bodily injury lines, and incurred loss ratios at that development point understate ultimate claims materially. For a motor book, evaluating at 12–18 months with a development factor applied to recent periods is more honest. At 6 months, lapse rate and new business volume are better-measured outcomes than loss ratio.

**Spillover effects.** If your treated segments compete for the same policyholders as your control segments, the rate change on treated segments might affect behaviour in control segments — policyholders migrate, competitors reprice to fill the gap. In this case the control segment is contaminated and parallel trends is violated by construction. Check whether your treated and control segments draw from the same distribution channel before treating them as independent.

**Thin segments.** The outcome metric is noisy when earned exposure is low. SDID weights on a control segment with 200 car-years per period will be imprecise. Use `min_exposure` to flag thin cells and treat estimates from thinly-capitalised segments with appropriate scepticism. The panel quality summary from `builder.summary()` shows the exposure distribution before you commit to the analysis.

**The regulatory context.** FCA TR24/2 and EP25/2 (July 2025, the FCA's evaluation of GIPP remedies) both use counterfactual comparison as the standard of evidence. If you are producing outcome monitoring evidence for Consumer Duty purposes, a before/after chart is not adequate. The `FCAEvidencePack` in `insurance-causal-policy` generates structured documentation — ATT, sensitivity analysis, panel quality metrics, caveats — designed to address what TR24/2 was looking for.

```python
from insurance_causal_policy import FCAEvidencePack

pack = FCAEvidencePack(
    result=result,
    sensitivity=sens,
    product_line="Motor",
    rate_change_date="2025-Q4",
    rate_change_magnitude="+5% technical premium",
    analyst="Pricing Team",
    panel_summary=builder.summary(),
)
report_md = pack.to_markdown()
```

---

## The decision framework

The choice of method depends on your data situation:

- **You have segments you did not change, with 8+ pre-treatment periods:** use `SDIDEstimator`. Run the pre-trend test. Report the breakdown frontier.
- **You have a thin donor pool (5–15 control segments):** use `DoublyRobustSCEstimator`. Acknowledge the DR property means you are relying on either SC or parallel trends, not both simultaneously.
- **Your rate change was staggered across cohorts:** use `StaggeredEstimator`. Do not use TWFE.
- **The change was genuinely portfolio-wide with no internal controls:** use an external market benchmark as a proxy control, document the assumption explicitly, and report wider uncertainty bounds. This is a weaker identification strategy and should be labelled as such.

What you should not do is run a before/after comparison, find a 3.5 point improvement, and conclude the rate change delivered 3.5 points. The market might have given you 2.5 points regardless. The rate change might have given you 5 points and selection effects took back 1.5. You do not know unless you construct the counterfactual.

The methods above let you construct it.

---

- [Synthetic DiD for Rate Change Evaluation](/2026/03/13/your-rate-change-didnt-prove-anything/) — covers the SDID methodology in depth, including the Arkhangelsky et al. (2021) weight optimisation and FCA regulatory context
- [Rate Change Evaluation: Did the Premium Increase Cause the Lapses?](/2026/03/10/rate-change-lapse-evaluation-causal-inference/) — worked case study on a young driver segment with a known lapse spike
- [How Much of Your GLM Coefficient Is Actually Causal?](/2026/03/01/your-demand-model-is-confounded/) — DML for price elasticity estimation; applies when the rate change was continuous rather than a clean step function
- [Champion/Challenger Testing with an Audit Trail](/2026/03/13/your-champion-challenger-test-has-no-audit-trail/) — prospective evaluation design that avoids the post-hoc measurement problem entirely
