---
layout: post
title: "The FCA Is Investigating Home and Travel Insurers: Is Your Pricing Model Defensible?"
date: 2026-03-19
author: Burning Cost
categories: [compliance, pricing, fairness]
description: "Active Consumer Duty investigations in home and travel insurance. What a defensible pricing model actually requires under PRIN 2A, and what the FCA thematic review said about most firms' current evidence."
tags: [FCA, Consumer-Duty, fairness, proxy-discrimination, insurance-fairness, compliance, home-insurance, travel-insurance, TR24/2, PS22/9, pricing, python, uk-insurance]
---

Following the Which? super-complaint on home and travel insurance, the FCA has indicated - according to market commentary and its own published supervisory communications - that enforcement activity is under way on fair value grounds in the home and travel sector. Reports suggest this includes restrictions on book growth for at least one firm, skilled person reviews into claims handling systems, and remedial commitments from senior managers, including consideration of whether redress may be due to customers.

This is not a hypothetical compliance risk.

The question for any home or travel pricing team is whether their model governance would survive scrutiny. Not whether it looks reasonable internally, but whether it would hold up in front of an FCA supervision team who has already found that the majority of firms are producing inadequate evidence. The answer for most teams, based on what the FCA has said publicly, is no.

---

## What Consumer Duty actually requires

Consumer Duty came into force in July 2023. The rules sit in PRIN 2A, with the detailed framework set out in PS22/9 and finalised guidance in FG22/5.

PRIN 2A.4.6 requires firms to monitor whether their products and services provide fair value and whether different groups of customers are experiencing poor outcomes. This is not satisfied by a paragraph in a board paper. The monitoring must be ongoing, it must be segmented by customer group, and it must produce quantitative evidence that the pricing model does not systematically disadvantage any group.

The key word is "evidence." The Consumer Duty is not a principles-based obligation where good intentions count. The FCA expects firms to be able to demonstrate, with data, that their pricing produces fair value outcomes across the relevant customer population.

---

## What the FCA found when it looked

TR24/2, the FCA's general insurance thematic review published in August 2024, examined how firms were implementing their Consumer Duty monitoring obligations. The finding was direct: most firms were producing "high-level summaries with little substance or relevant information."

That phrase is doing a lot of work. It means the FCA looked at what firms called fair value assessments and found them to be prose documents without quantitative backing. No statistical tests. No segment-level disparity analysis. No evidence that the pricing team had actually interrogated whether outcomes differed across groups.

The FCA's expectation is explicit in FG22/5: firms must monitor outcomes across groups of customers with different characteristics, including protected characteristics under the Equality Act 2010. That means your fair value assessment needs to show, with numbers, what is happening to female policyholders relative to male, to younger versus older customers, and to any other dimension along which your model might produce systematically different outcomes.

The Equality Act dimension matters independently. Section 19 prohibits indirect discrimination, which means a pricing factor that is neutral on its face but has a disproportionate adverse effect on a protected group is unlawful unless justified. Postcode in home insurance is the obvious candidate: it predicts risk, but it also correlates with ethnicity, religious composition, and in some cases disability prevalence. The interaction between the Consumer Duty monitoring obligation and the Section 19 indirect discrimination test means that a complete fair value assessment needs to address proxy discrimination, not just headline outcome ratios.

---

## What "defensible" looks like in practice

A defensible pricing model is one where, if the FCA asked tomorrow, you could produce structured evidence with the following components:

**Proxy detection across all rating factors.** For each factor in your tariff, you need to show whether it acts as a proxy for a protected characteristic. Mutual information scores, CatBoost proxy R-squared, and SHAP proxy decomposition are the standard tools. A factor with postcode proxy R-squared above 0.10 is contributing discriminatory variation to prices, even if the actuarial justification for the factor itself is clean.

**Calibration by group with statistical tests.** A/E ratios segmented by protected characteristic, with credibility-weighted standard errors. The test is not whether A/E looks similar across groups on visual inspection: it is whether the difference is statistically significant given the exposure in each cell. A pricing committee paper that says "A/E ratios are broadly similar" is not evidence. A confidence interval around the ratio difference is.

**Multicalibration analysis.** Standard A/E by group checks whether the model is unbiased conditional on group membership. Multicalibration checks whether it is unbiased conditional on both group membership and predicted premium level. A model that looks fair overall may be systematically overcharging a specific protected group at lower premium levels and undercharging them at higher ones. Those effects cancel in aggregate but the individual-level discrimination is real.

**A documented audit trail, not a summary.** The FCA thematic finding was that firms produced summaries. What the FCA wants is the underlying analysis: the test statistics, the sample sizes, the methodology, the finding, and the response where a finding is adverse.

---

## What insurance-fairness provides

[`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) is designed specifically for this problem. The library produces structured evidence rather than summaries.

`FairnessAudit` runs the complete proxy detection and group outcome analysis in a single call. Given a fitted CatBoost model, a policy-level dataset, and a list of protected characteristic columns, it computes mutual information scores and CatBoost proxy R-squared for all rating factors, runs calibration-by-group with exposure weighting, and produces a Markdown report with explicit Consumer Duty regulatory mapping. The report is structured for a pricing committee pack or an FCA file review. It does not produce a paragraph of interpretation: it produces tables, confidence intervals, and test results.

```python
uv add insurance-fairness
```

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=catboost_model,
    data=df,
    protected_cols=["gender", "age_band"],
    prediction_col="predicted_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=["postcode_district", "vehicle_group", "ncd_years", "build_year"],
)
report = audit.run()
report.to_markdown("fair_value_evidence_2026q1.md")
```

`MulticalibrationAudit` handles the within-premium-band analysis. It segments policyholders into predicted premium deciles, then tests for systematic A/E bias within each (decile, group) cell. The correction mechanism applies credibility-weighted adjustments where cells fail the test, following the Denuit, Michaelides and Trufin (2026) framework adapted for UK insurance conventions.

`WassersteinCorrector` handles the harder case: where a proxy factor has been identified and needs to be corrected rather than just flagged. It computes discrimination-free premiums using Wasserstein barycenter correction, following the Lindholm-Richman-Tsanakas-Wuthrich (2022) framework. The output is a set of corrected predictions that can go straight into a GLM tariff as multiplicative relativities.

`PrivatizedFairnessAudit` handles the case where protected attribute data is not available at all, using local differential privacy. We cover this in a separate post.

---

## What you should not do

Two things to avoid.

First, do not write a Consumer Duty fair value assessment that is a prose summary of why your pricing is reasonable. The FCA has already said, in TR24/2, that this is not adequate. A document describing your actuarial methodology and asserting it produces fair outcomes is not evidence that it does.

Second, do not overclaim the regulatory position on any specific metric or threshold. Consumer Duty is law. The obligation to monitor and evidence fair outcomes is not negotiable. But the FCA has not specified that a particular statistical test or threshold constitutes compliance. What the FCA has said is that it expects quantitative evidence, not high-level summaries, and that it will scrutinise the substance of that evidence. Build the evidence base first; it will survive any reasonable regulatory interpretation.

---

## The timeline pressure

The FCA's 2026 supervisory plan commits to expanding its home and travel review in Q2 and Q3 2026. Firms that were reviewed in the 2025 thematic are already aware of their standing. Firms that were not reviewed should not assume their absence from the sample was a clean bill of health.

The investigation approach the FCA is now using is described as "assertive, supervision-led, earlier intervention" rather than lengthy enforcement. That language means the FCA is prepared to act before it has a complete enforcement case. Adequate fair value evidence, produced before a supervisory engagement begins, is substantially better than evidence produced under pressure once contact has been made.

If your pricing team cannot today produce a structured, quantitative, Consumer Duty-aligned fair value assessment for your home or travel book, the time to build that capability is not after a section 166 notice arrives.

---

**Related posts:**
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) -- the detection methodology in detail: mutual information, CatBoost proxy R-squared, SHAP decomposition
- [Discrimination-Free Pricing in Python: Causal Paths, Optimal Transport, and the FCA](/2026/03/10/insurance-fairness-ot/) -- the correction step after detection: WassersteinCorrector and causal path decomposition
- [Fairness Auditing When You Don't Have Sensitive Attributes](/2026/03/20/fairness-auditing-without-sensitive-attributes/) -- what to do when you cannot observe the protected characteristic at all
