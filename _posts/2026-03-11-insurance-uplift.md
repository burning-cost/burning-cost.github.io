---
layout: post
title: "Your Retention Campaign Has No Targeting Rule"
date: 2026-03-11
categories: [libraries, pricing, causal]
tags: [uplift-modelling, HTE, CATE, retention, renewal, CausalForestDML, Qini, AUUC, Consumer-Duty, ENBP, GIPP, PS21-5, insurance-uplift, python, EconML]
description: "Every insurer runs discount campaigns at renewal. Most target by propensity to lapse — who will leave? The correct question is who will respond to a discount. Those are not the same people. insurance-uplift estimates per-customer treatment effects on renewal probability, classifies the portfolio into Persuadables, Sure Things, Lost Causes, and Do Not Disturbs, and clips recommendations to ENBP. 127 tests."
---

Every UK personal lines insurer runs some version of a retention campaign at renewal. A model predicts who is likely to lapse. Someone in the retention team decides on a discount threshold. An outbound call or a reduced renewal quote goes to whoever is above the threshold.

This is a propensity model. It answers the question: "who will leave?" That is not the right question for a discount campaign. The right question is: "who will change their decision because of the discount?" These are different populations, and targeting the wrong one costs real money.

Consider what a propensity model sends you towards. A customer who was going to lapse regardless of price — they are gone whether you discount or not. A customer who was going to renew regardless of price — they take the discount and you just gave away margin for nothing. The customers you actually want to find are the ones sitting on the fence: price-sensitive enough that a discount flips their renewal decision, but not so price-insensitive that they renew whatever you charge.

Uplift modelling estimates individual treatment effects — specifically, who moves from lapse to retain when you reduce the price. [`insurance-uplift`](https://github.com/burning-cost/insurance-uplift) provides the full pipeline for UK personal lines renewal data: panel construction, CATE estimation via EconML's CausalForestDML, Qini curve evaluation, a four-customer taxonomy, ENBP-constrained recommendations, and a Consumer Duty fairness audit. 127 tests, MIT-licensed, on PyPI.

```bash
uv add insurance-uplift
```

---

## The four customers

Guelman, Guillén & Pérez-Marín (2012–2015) described the four customer types that every retention campaign implicitly creates. The partition is exhaustive.

**Persuadable** — lapse without the discount, retain with it. τ̂(x) strongly negative: the causal effect of a price increase on renewal probability is large in magnitude. These are the only customers worth targeting with a discount. A 5% reduction changes their decision.

**Sure Thing** — renew regardless of price. τ̂(x) near zero. They would have renewed at your existing renewal rate. Discounting them gives away margin for no retention benefit. The renewal team's intuition here is wrong: the customer accepts the discount, they renew, the campaign looks like it worked. The counterfactual — they would have renewed anyway — is invisible.

**Lost Cause** — lapse regardless of price. τ̂(x) near zero, but the customer has already decided to leave. No discount changes this. A price-insensitive churner has made up their mind for reasons the discount does not address (a better quote elsewhere, life change, broker move).

**Do Not Disturb** — τ̂(x) positive. A price reduction triggers comparison shopping. The discount reminds a contented customer that they should check the market. This is the group that makes the wrong targeting rule actively harmful, not just wasteful. Sending them a discount invitation is worse than sending nothing.

The taxonomy is not just a classification exercise. It tells you what to do:

| Segment | Action |
|---|---|
| Persuadable | Offer a targeted discount |
| Sure Thing | Hold rate — they will renew |
| Lost Cause | No intervention — spend the budget elsewhere |
| Do Not Disturb | Do not contact — leave them alone |

A propensity model conflates Persuadables and Sure Things (both show high lapse propensity without a discount) and misses Do Not Disturbs entirely.

---

## Why propensity modelling gets this wrong

The standard retention model is a binary classifier: `P(lapse | X)`. It ranks customers by probability of lapsing and flags the top decile for intervention.

The problem is selection bias. Customers who received discounts in the past renewed at higher rates, but that tells us nothing about whether the discount caused the retention. Maybe those customers were going to renew anyway — they are Sure Things who also happened to receive a discount. The propensity score for a Sure Thing is high (they are predicted to lapse without intervention) precisely because they are inelastic and have historically received discounts to hold them.

Causal inference separates the effect of the price from the confounding driven by correlated customer characteristics. DML (Double Machine Learning) residualises out the variation in price changes that can be predicted from customer features — the annual blanket increases, the segment-level strategies — and estimates the treatment effect from the residual variation that is orthogonal to those confounders. CausalForestDML grows trees that maximise treatment effect heterogeneity rather than outcome prediction, which gives you per-customer CATE estimates τ̂(x) with valid confidence intervals via honest splitting.

---

## Building the panel

The first step is constructing a renewal panel from a policy extract. `RetentionPanel` handles the essential accounting: computing the log price ratio treatment, flagging censored policies, and computing earned exposure.

```python
import polars as pl
from insurance_uplift.data import RetentionPanel

panel_obj = RetentionPanel(
    policy_df=df,
    renewal_premium_col='renewal_premium',
    expiring_premium_col='expiring_premium',
    renewal_indicator_col='renewed',
    start_date_col='start_date',
    end_date_col='end_date',
    enbp_col='enbp',
    censor_date=date(2025, 12, 31),
)
panel = panel_obj.build()
```

The treatment is `log(renewal_premium / expiring_premium)`. Positive treatment is a price increase; negative is a price decrease. This continuous treatment maps directly to the log-linear elasticity form and is symmetric: a -10% price change is log(0.90) ≈ -0.105, which is the same magnitude as a +10% increase in the other direction.

Censored policies — those with `end_date > censor_date` — are flagged and excluded from binary outcome modelling with a warning. A 6-week pre-renewal extract censors roughly 12–15% of a typical motor book. The library warns you, excludes them cleanly, and documents what was dropped.

```python
# Check treatment variation — DML requires within-cell variation
variation = panel_obj.treatment_variation_report(
    confounder_cols=['age_band', 'ncd_band', 'region']
)
# Cells with coefficient of variation < 0.05 are flagged
# A blanket 10% increase applied uniformly across the book
# will fail this check — you need real price differentiation.

panel_clean = panel.filter(pl.col('censored_flag') == 0)
```

The treatment variation check matters. DML residualisation works by estimating `E[T|X]` and using the residual `T - E[T|X]`. If your pricing strategy applied a flat 10% increase to every customer in a segment, there is no within-segment variation for the model to learn from. The check flags this before you fit, not after.

---

## Estimating CATE

```python
from insurance_uplift.fit import RetentionUpliftModel

model = RetentionUpliftModel(
    estimator='causal_forest',   # CausalForestDML with honest CIs
    n_estimators=2000,
    n_folds=5,
    min_samples_leaf=20,
)

model.fit(
    panel=panel_clean,
    confounders=['age', 'ncd', 'vehicle_age', 'region', 'tenure', 'prior_claims'],
)

# Per-customer CATE: effect of 1-unit log price increase on renewal probability
tau = model.cate(panel_clean)

# With 95% confidence intervals (honest, via Bootstrap-of-Little-Bags)
tau_hat, lower_95, upper_95 = model.cate_inference(panel_clean)

# Population ATE
ate, lo, hi = model.ate()
print(f"ATE: {ate:.4f} ({lo:.4f}, {hi:.4f})")
# Typical result: ATE: -0.31 (-0.38, -0.24)
# Interpretation: a 1-unit log price increase reduces renewal probability by 0.31 pp
# A 10% price increase reduces renewal probability by 0.31 × 0.095 ≈ 2.9 pp on average

# Group-level effects by NCD band
gate = model.gate(panel_clean, by='ncd_band')
print(gate)
# ncd_band 0:  gate = -0.52 (most price-sensitive — new policyholders)
# ncd_band 4:  gate = -0.18 (established customers — less sensitive)
# ncd_band 9+: gate = -0.07 (long-tenure customers — most inelastic)
```

The `x_learner` option is available for books where fewer than 10% of customers received an explicit retention discount. Cross-fitted T-learner estimates are more efficient under heavy treatment imbalance. `dr_learner` is the fall-back for low-overlap panels — when discounts were concentrated in a narrow segment and propensity overlap with the control group is poor.

---

## Evaluating targeting quality

Standard ML metrics do not measure targeting quality. A model with lower CATE MSE can produce worse campaign outcomes than one with higher MSE, if it fails to rank the Persuadables above the Sure Things. Qini curves measure what matters: for each fraction of the customer base targeted in descending order of τ̂(x), how many incremental renewals do you get?

```python
from insurance_uplift.evaluate import qini_curve, auuc, uplift_at_k, segment_types, plot_qini

# Qini coefficient (area between curve and random targeting diagonal)
score = auuc(panel_clean['renewed'], panel_clean['treatment'], tau)
print(f"AUUC: {score:.4f}")

# Uplift at top 30%: what fraction of achievable gain does the top 30% capture?
u30 = uplift_at_k(panel_clean['renewed'], panel_clean['treatment'], tau, k=0.3)
print(f"Uplift@30: {u30:.3f}")
# A value of 0.78 means targeting 30% of the book captures 78% of achievable gain

# Four-customer taxonomy
segments = segment_types(panel_clean['renewed'], panel_clean['treatment'], tau)
print(segments)
#  segment_type    n  fraction  avg_tau
#  Persuadable   4821   0.241   -0.487
#  Sure Thing   11203   0.560    0.042
#  Lost Cause    2891   0.145   -0.031
#  Do Not Disturb  1085   0.054    0.183

# Plot
ax = plot_qini(panel_clean['renewed'], panel_clean['treatment'], tau)
```

In a real motor book, Persuadables are typically 20–30% of the renewal portfolio. Sure Things run at 50–60%. Lost Causes are smaller — customers who are leaving for non-price reasons. Do Not Disturbs are the dangerous minority: targeting them with a discount initiative triggers the comparison shopping that causes the lapse you were trying to prevent.

The continuous treatment binarisation is transparent: customers who received below-median price changes are the "treated" group; above-median are "control". This is the standard adaptation for observational panels without explicit A/B assignment.

---

## Optimal targeting rules with PolicyTree

The Qini curve tells you how many to target. PolicyTree tells you who — expressed as a decision tree that is implementable in an underwriting system.

```python
from insurance_uplift.segment import PolicyTree, SegmentSummary

tree = PolicyTree(
    uplift_model=model,
    max_depth=2,         # depth-2 trees are implementable in most rating systems
    backend='sklearn',   # sklearn approximation; use 'policytree_r' for welfare-optimal
)
tree.fit(panel_clean)

# Binary recommendation: 1 = offer discount
recommend = tree.recommend(panel_clean)

# Expected welfare gain vs uniform rate
print(f"Welfare gain vs uniform: {tree.welfare_gain():.2f} percentage points retention")

# Exportable rules — import into pricing engine
rules = tree.export_rules()
# [{'node': 0, 'feature': 'ncd', 'threshold': 2,
#   'left': 'DO_NOT_DISCOUNT', 'right': 'CHECK_AGE'},
#  {'node': 1, 'feature': 'age', 'threshold': 45,
#   'left': 'DISCOUNT', 'right': 'DO_NOT_DISCOUNT'}]

# Segment summary: readable table for the pricing committee
summary = SegmentSummary(uplift_model=model, max_depth=3)
summary.fit(panel_clean)
print(summary.segment_table())
```

The depth-2 tree trades a small welfare loss against implementability. For governance presentations where you need to demonstrate optimality, the `policytree_r` backend calls the Athey & Wager (2021) exhaustive welfare-maximisation algorithm via rpy2 + the R `policytree` package. The greedy sklearn approximation is typically within 5% welfare loss at depth 2 and does not require R.

---

## ENBP constraint and Consumer Duty audit

Any retention pricing recommendation in the UK must pass through ICOBS 6B.2 before it is acted on. The ENBP rule (from FCA PS21/5, effective January 2022) prohibits the renewal offer price from exceeding what the same customer would be quoted as a new business applicant. You cannot extract margin from inelastic customers beyond this floor.

```python
from insurance_uplift.constrain import ENBPConstraint, FairnessAudit, ROIReport

# Clip recommendations to ENBP
constraint = ENBPConstraint(enbp_col='enbp', expiring_premium_col='expiring_premium')

# Convert targeting decision to a rate change recommendation
recommended_rate_change = pl.Series([
    -0.05 if r == 1 else 0.0 for r in recommend.to_list()
])

clipped = constraint.apply(panel_clean, recommended_rate_change)
report = constraint.audit_report(panel_clean, recommended_rate_change)
# Columns: policy_id, expiring_premium, enbp, recommended_renewal,
#          clipped_renewal, was_clipped, clip_amount_pct
```

The ENBP column is pre-computed from the new business rating system and included in the renewal extract. Any recommendation that would produce a renewal offer above ENBP is silently reduced to the ENBP ceiling. The audit report documents what was clipped, by how much, and by segment.

Consumer Duty (PRIN 2A.4) adds a second requirement that sits above technical ENBP compliance. Price insensitivity correlates with vulnerability in personal lines: older customers are simultaneously the most inelastic and the most likely to have Consumer Duty vulnerability characteristics. The `FairnessAudit` class checks whether the segments you are targeting — or holding rates on — are disproportionately composed of vulnerable proxies.

```python
audit = FairnessAudit(
    protected_proxies=['age_band', 'postcode_income_decile'],
    vulnerability_threshold_age=70,
)
audit.fit(panel_clean.select(['age_band', 'postcode_income_decile']), tau)
fairness_table = audit.audit()

# Flagged rows: groups that are BOTH inelastic (avg_tau > 0) AND vulnerability proxies
flagged = fairness_table.filter(pl.col('flagged_as_vulnerable'))
print(flagged.select(['proxy_variable', 'group', 'avg_tau', 'regulatory_note']))

# age_band: 70-79, avg_tau = 0.14 → flagged
# postcode_income_decile: 1, avg_tau = 0.09 → flagged
```

A group with positive τ̂(x) is commercially attractive — they will renew even if you raise the price. If that group is older customers or low-income postcodes, charging them more may be technically ENBP-compliant but is likely to fail the Consumer Duty fair value outcome. The audit output is designed to be handed to compliance with a `regulatory_note` field explaining the specific concern.

---

## ROI before you commit the budget

```python
roi = ROIReport(
    discount_cost_per_unit=0.0,      # no admin cost per policy
    policy_premium_avg=650.0,        # average motor premium
)

results = roi.compute(
    df=panel_clean,
    tau_hat=tau,
    recommended_treatment=recommend,
    discount_size=0.05,              # 5% discount offered to targeted customers
)

print(f"Policies targeted:              {results['n_treated']:,}")
print(f"Expected additional renewals:   {results['expected_additional_renewals']:.1f}")
print(f"Expected discount cost (£):     {results['expected_discount_cost']:,.0f}")
print(f"Expected revenue gain (£):      {results['expected_additional_premium_revenue']:,.0f}")
print(f"Net ROI (£):                    {results['net_roi']:,.0f}")
print(f"ROI (%):                        {results['roi_pct']:.1f}%")
print(f"Break-even retention rate:      {results['break_even_retention_rate']:.3f}")
```

The ROI calculation links the statistical CATE estimates to business outcomes. The expected additional renewals from targeting the Persuadable segment are the model's τ̂(x) × log(1 - discount_size) summed across targeted customers — the expected change in renewal probability for each customer multiplied by the price change they receive. The break-even retention rate is the minimum additional retention the campaign needs to achieve to cover the discount cost. If that number is well below the model's expected incremental retention, the campaign is commercially justified.

---

## The pipeline in full

The three Burning Cost causal inference libraries address adjacent questions on the same renewal dataset:

- **insurance-causal:** "What is the average causal effect of a price change on retention?" Output: scalar ATE with CIs.
- **insurance-elasticity:** "How does price sensitivity vary by customer segment?" Output: CATE surface, elasticity heatmap by rating factor.
- **insurance-uplift:** "Which customers should receive a discount, how many incremental renewals will that produce, and what is the ROI?" Output: targeting recommendation, Qini curve, ROI report, fairness audit.

The natural handoff: use insurance-elasticity to understand the elasticity landscape, then bring the CATE estimates into insurance-uplift to build the campaign decision.

The ENBP constraint and FairnessAudit are not optional add-ons. FCA EP25/2 (December 2025) confirmed that GIPP remedies are working as intended: the motor price walking market has been largely eliminated, and ENBP is now the accepted floor. Consumer Duty vulnerability reviews in 2025 found persistent gaps for customers with multiple vulnerability characteristics. Any retention targeting tool deployed in the UK must address both.

---

**[insurance-uplift on GitHub](https://github.com/burning-cost/insurance-uplift)** — 127 tests, MIT-licensed, PyPI. Library #46.

---

**Related articles from Burning Cost:**
- [Demand Modelling for Insurance Pricing](/2026/02/25/demand-modelling-for-insurance-pricing/)
- [Your Renewal Pricing Is Flying Blind](/2026/03/11/your-renewal-pricing-is-flying-blind/)
- [Your Demand Model Is Confounded](/2026/03/01/your-demand-model-is-confounded/)
