---
layout: post
title: "Synthetic Difference-in-Differences for Rate Change Evaluation"
date: 2026-03-13
categories: [regulation, libraries, causal-inference]
tags: [sdid, causal-inference, difference-in-differences, fca-tr242, fca-ep252, rate-change-evaluation, python, insurance-causal-policy]
description: "DiD and Callaway-Sant'Anna for rate change attribution. insurance-causal-policy quantifies what your rate change actually achieved, TR24/2 compliant."
---

Your loss ratio went down six months after the rate increase. The pricing presentation shows a before/after chart. Management is pleased. The FCA compliance team has filed it as a positive outcome.

None of this is evidence that the rate change caused the improvement.

This is not a pedantic distinction. FCA TR24/2 (August 2024) reviewed how UK insurers evaluate the outcomes of their pricing interventions and found that 28 firms failed to demonstrate causal attribution between rate changes and observed outcomes. The FCA's language was explicit: before/after comparisons without counterfactuals do not constitute adequate evidence under Consumer Duty outcome monitoring. They compared outcomes between treated and comparison groups in their own EP25/2 evaluation specifically because they knew the before/after approach cannot isolate the treatment effect from market-wide trends.

We built [`insurance-causal-policy`](https://github.com/burning-cost/insurance-causal-policy) to close this gap. It is on PyPI at v0.1.4, 170 tests passing, and it produces the kind of structured causal evidence that TR24/2 was looking for.

---

## What the before/after analysis gets wrong

When you apply a 12% rate increase to a motor segment in Q1 2023 and the loss ratio improves by 8 points over the following three quarters, you are observing a combination of:

1. The causal effect of the rate increase on the loss ratio
2. Market-wide claims trends that affected you and your competitors equally
3. Segment-level trends that were already in motion before Q1 2023
4. Random fluctuation, particularly for thinner segments

A before/after chart conflates all four. If claims frequency was falling across the market due to changing driving patterns or Ogden rate expectations, your loss ratio would have improved regardless of the rate action. Selection into treatment creates bias in the before/after comparison that makes your rate change look more effective than it was.

The only way to separate effect 1 from effects 2, 3, and 4 is a counterfactual: what would have happened to this segment if the rate change had not been applied? That counterfactual is not in your data. It has to be constructed.

---

## Synthetic Difference-in-Differences

Synthetic Difference-in-Differences (SDID, Arkhangelsky, Athey, Hirshberg, Imbens and Wager, 2021, *American Economic Review* 111(12): 4088–4118) is the right method for this problem.

The core idea: instead of comparing your treated segments to a fixed control group, SDID constructs a synthetic control by finding a weighted combination of untreated segments whose pre-treatment trajectory matches your treated segments as closely as possible. It also reweights time periods, emphasising the pre-treatment periods that best predict the counterfactual. The ATT estimate comes from a weighted two-way fixed effects regression using these optimised weights.

This is an improvement over standard DiD in three specific ways for insurance books:

**Unit weights absorb level differences.** Your treated segments and control segments have different mean loss ratios (that is partly why you changed the rates). Standard DiD with a fixed control group is biased when treated and controls start at different levels. SDID's unit weights include an intercept term that absorbs level differences; what matters is whether the trends were parallel, not whether the levels were equal.

**Time weights focus on informative pre-treatment periods.** If the most recent pre-treatment quarters best predict the post-treatment control trajectory (common in insurance when claims patterns shift gradually), SDID down-weights earlier periods automatically. A standard DiD with equal time weights gives stale observations the same influence as informative ones.

**The estimator is better-behaved under heterogeneous treatment effects.** Standard two-way fixed effects (TWFE) is well known to produce nonsensical estimates when treatment effects vary. SDID is substantially more robust, and the staggered adoption extension handles it properly.

---

## What the library does

`insurance-causal-policy` has five components:

**PolicyPanelBuilder** converts your raw insurance tables into the balanced segment × period panel that SDID requires. The panel problem in insurance is non-trivial: raw policy data is inherently unbalanced (policies start and end, some segment-period cells have zero claims), and the outcome needs to be exposure-weighted correctly. Loss ratio is `incurred_claims / earned_premium`, not the mean of individual policy loss ratios: this distinction matters when exposure varies substantially across cells.

```python
from insurance_causal_policy import PolicyPanelBuilder

builder = PolicyPanelBuilder(
    policy_df,      # segment_id, period, earned_premium, earned_exposure
    claims_df,      # segment_id, period, incurred_claims, claim_count
    rate_log_df,    # segment_id, first_treated_period
    outcome="loss_ratio",
    min_exposure=50.0,  # warn on thin cells
)
panel = builder.build()
print(builder.summary())
# {'n_segments': 84, 'n_treated_segments': 31, 'n_control_segments': 53,
#  'n_periods': 12, 'pct_treated_cells': 28.3, ...}
```

**SDIDEstimator** runs the SDID estimation. Unit and time weights are solved via CVXPY using the constrained least-squares formulation from the Arkhangelsky et al. paper. Three inference options: placebo (fastest, requires more control units than treated units), bootstrap (robust, slower), jackknife (leave-one-out, expensive for large panels).

```python
from insurance_causal_policy import SDIDEstimator

est = SDIDEstimator(panel, outcome="loss_ratio", inference="placebo")
result = est.fit()
print(result.summary())
# ATT: -0.0812  SE: 0.0231  CI: [-0.1265, -0.0359]  p=0.0004
# Pre-trend test: PASS (p=0.431)
# Treated: 31 segments, Control (effective): 18 segments
# Pre-treatment periods: 8, Post-treatment: 4
```

An ATT of -0.0812 on loss ratio means the rate change caused an 8.1 percentage point reduction in the loss ratio, after accounting for what would have happened without the rate change.

**StaggeredEstimator** handles books where rate changes were applied to different segments at different times, as is common in practice when rate actions are phased across risk segments over multiple quarters. Standard TWFE is known to produce sign-reversal estimates under staggered adoption with heterogeneous treatment effects; StaggeredEstimator uses Callaway and Sant'Anna (2021) to estimate a group-time ATT for each cohort × period combination, using only clean controls (never-treated or not-yet-treated segments).

```python
from insurance_causal_policy import StaggeredEstimator

est_s = StaggeredEstimator(panel, outcome="loss_ratio")
result_s = est_s.fit()
print(f"Overall ATT: {result_s.att_overall:.4f} (SE: {result_s.se_overall:.4f})")
print(f"Cohorts: {result_s.n_cohorts}")
# Overall ATT: -0.0741 (SE: 0.0198)
# Cohorts: 4
```

**compute_sensitivity** implements a Python approximation of HonestDiD (Rambachan and Roth, 2023, *Review of Economic Studies*, rdad018). The question it answers: how large must the post-treatment parallel trends violation be before the sign of the ATT reverses? That violation size is the breakdown frontier. If your ATT of -0.08 only reverses to zero when post-treatment confounding is three times larger than any pre-treatment deviation you observed, the result is robust.

```python
from insurance_causal_policy import compute_sensitivity, plot_sensitivity

sens = compute_sensitivity(result)
# M=0.0: ATT bounds [-0.1265, -0.0359]  (no violation allowed)
# M=1.0: ATT bounds [-0.1496, -0.0128]  (still excludes zero)
# M=2.0: ATT bounds [-0.1727,  +0.0103] (zero enters identified set)
# Breakdown point M*: ~1.8 pre-period SDs
plot_sensitivity(sens)
```

**FCAEvidencePack** generates a structured regulatory document (Markdown, JSON, or PDF) that combines the estimation results, sensitivity analysis, panel quality statistics, and a standard caveats section. The output is designed to address the specific gaps TR24/2 identified: counterfactual construction, parallel trends evidence, and documented limitations.

```python
from insurance_causal_policy import FCAEvidencePack

pack = FCAEvidencePack(
    result=result,
    sensitivity=sens,
    product_line="Motor",
    rate_change_date="2023-Q1",
    rate_change_magnitude="+12% technical premium",
    analyst="Pricing Team",
    panel_summary=builder.summary(),
)
report_md = pack.to_markdown()
report_json = pack.to_json()
```

---

## The end-to-end workflow on synthetic data

The library ships a synthetic motor panel generator so you can validate the methodology before applying it to live data. `make_synthetic_motor_panel` creates a segment × period panel with a known true ATT — which lets you verify that the estimator is recovering the right answer.

```python
from insurance_causal_policy import (
    make_synthetic_motor_panel,
    PolicyPanelBuilder,
    SDIDEstimator,
    compute_sensitivity,
    FCAEvidencePack,
)

# Generate synthetic data with true ATT = -0.08
policy_df, claims_df, rate_log_df = make_synthetic_motor_panel(
    n_segments=100,
    n_periods=12,
    true_att=-0.08,
    seed=42,
)

# Build panel
builder = PolicyPanelBuilder(policy_df, claims_df, rate_log_df)
panel = builder.build()

# Estimate
est = SDIDEstimator(panel, outcome="loss_ratio", inference="placebo")
result = est.fit()
# ATT: -0.0812, true: -0.08. Close enough.

# Sensitivity
sens = compute_sensitivity(result)

# Evidence pack
pack = FCAEvidencePack(
    result=result,
    sensitivity=sens,
    product_line="Motor",
    rate_change_date="2023-Q1",
    rate_change_magnitude="+12%",
    panel_summary=builder.summary(),
)
print(pack.to_markdown())
```

The full workflow from raw tables to regulatory document takes under 20 lines of code and under a minute to run on a panel with 100 segments and 12 periods.

---

## The regulatory context

We want to be precise about what TR24/2 actually found, because the finding is specific.

The FCA reviewed outcomes monitoring across general insurance firms in August 2024. It found that firms routinely reported outcome metrics — loss ratios, complaint rates, claims acceptance rates — without demonstrating that their pricing interventions caused the observed changes. The failure was methodological: before/after comparisons without counterfactuals, no parallel trends testing, no sensitivity analysis. This is not the FCA demanding peer-reviewed econometrics. It is the FCA saying that "our loss ratio improved" is not the same as "our rate change worked."

EP25/2 (July 2025) is the FCA's own evaluation of GIPP remedies. It used conditional DiD — the FCA built a control group of comparable firms that did not implement the remedied practices and compared outcomes over the policy year following implementation. The methodology section of EP25/2 explicitly discusses parallel trends testing and counterfactual validity. The FCA is applying a higher standard to its own evaluations than it is seeing from regulated firms.

The gap is not about resources; the calculations in `insurance-causal-policy` run on a laptop in under a minute. The gap is that no Python package previously combined SDID, staggered adoption, insurance-specific panel construction, and FCA-oriented output in a form that a pricing team could run without an econometrics background. We hope this closes it.

---

## What SDID cannot do

We should be direct about the limitations, because the regulatory context creates a temptation to oversell the methodology.

SDID gives you the Average Treatment Effect on the Treated under the assumption that parallel trends hold after reweighting. This assumption cannot be tested from the data; the pre-trend test (and the HonestDiD sensitivity analysis) give you evidence about whether the assumption is plausible, not whether it is true.

For insurance specifically, three threats to identification are worth naming. First, **market-wide shocks**: if the Ogden rate changed, or the FCA introduced GIPP reforms, or there was a severe weather event, during your post-treatment window and this differentially affected your treated and control segments, the SDID estimate is biased. SDID's time weights partially mitigate this but cannot eliminate it. Second, **IBNR**: incurred loss ratios for periods within 12–18 months of the analysis date understate ultimate claims. This means the apparent ATT in recent periods is likely an underestimate of the true effect. Third, **thin segments**: if your treated segments have low earned exposure — say, fewer than 500 car-years per period — the outcome metric itself is noisy, and the SDID weights will struggle to find a synthetic control that matches the pre-treatment trajectory. The `min_exposure` parameter and panel quality warnings exist for this reason.

None of these mean the method is not worth using. The output needs to be interpreted with domain knowledge, and the caveats section of the evidence pack should be read as carefully as the headline ATT.

---

## Installation

```bash
uv add insurance-causal-policy
```

Python 3.10 or later. Dependencies: Polars, NumPy, SciPy, CVXPY (with Clarabel solver), Pandas. The optional `differences` package enables the full Callaway-Sant'Anna implementation; without it, the library falls back to a native doubly-robust estimator.

The source is at [github.com/burning-cost/insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy). The tests cover 170 scenarios including synthetic data recovery, edge cases (single treated unit, zero-claim periods, staggered cohorts), and the FCA evidence pack output format.

---

A before/after chart is not evidence. It is a description of what happened. The question of whether the rate change caused what happened requires a counterfactual. This library builds the counterfactual.

- [How Much of Your GLM Coefficient Is Actually Causal?](/2026/03/01/your-demand-model-is-confounded/)
- [When exp(beta) Lies: Confounding in GLM Rating Factors](/2026/03/05/your-rating-factor-might-be-confounded/)
- [Champion/Challenger Testing with ICOBS 6B.2.51R Compliance](/2026/03/13/your-champion-challenger-test-has-no-audit-trail/)
