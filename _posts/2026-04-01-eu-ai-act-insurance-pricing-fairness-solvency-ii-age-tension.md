---
layout: post
title: "The EU AI Act's Age Problem: Why Solvency II and Article 21 Cannot Both Be Right"
date: 2026-04-01
categories: [fairness, regulation]
tags: [eu-ai-act, fairness, solvency-ii, age-discrimination, Chouldechova, calibration, impossibility-theorem, FCA-Consumer-Duty, UK-Equality-Act, life-insurance, health-insurance, algorithmic-fairness, Meding, arXiv-2501-12962, non-discrimination]
description: "Solvency II requires age-based pricing. EU AI Act Article 21 lists age as a prohibited characteristic. Chouldechova's impossibility theorem shows you cannot satisfy both fairness and calibration across age groups simultaneously. What should a UK pricing actuary actually do?"
seo_title: "EU AI Act Insurance Pricing Fairness: The Solvency II Age Contradiction"
math: true
author: burning-cost
---

The EU AI Act has been covered extensively — scope, conformity assessment, what counts as a high-risk AI system, whether your GLM is even an AI system at all. We have written those posts. This one is about a different, less-discussed problem: a genuine legal contradiction that the Act creates for life and health insurance pricing, and the statistical impossibility that sits underneath it.

Here is the tension in one sentence: Solvency II requires you to price on actuarial evidence, which means using age because age causally drives claims; the EU AI Act's non-discrimination framework lists age as a prohibited characteristic whose use requires specific justification; and Chouldechova's impossibility theorem proves mathematically that you cannot simultaneously achieve calibration and statistical fairness across groups that differ in their underlying claim rates.

No amount of documentation resolves this. It is a genuine conflict between three incompatible requirements: actuarial soundness, regulatory compliance, and mathematical consistency. The Act does not tell you how to resolve it.

---

## What each framework actually requires

**Solvency II.** Article 21 of the Solvency II Directive permits — and in practice requires — the use of age, sex (subject to Test-Achats restrictions), health status, and other actuarially relevant characteristics in pricing. The requirement flows from Article 77 and the actuarial function obligations under Article 48: the insurer must calculate technical provisions using "prudent, reliable and objective" information, with assumptions that reflect "all the relevant experience available." An actuary who ignores age in life expectancy modelling for a life annuity product is not being prudent. They are introducing systematic mispricing that creates solvency risk.

The UK position post-Brexit is equivalent. Article 272(2) of the retained Solvency II Directive (now embedded in UK law via SI 2019/1212) preserves the actuarial exception. The PRA's actuarial function requirements under Solvency II Article 48 (and PRA SS18/16) require the Chief Actuary to sign off on pricing assumptions. Signing off on a model that deliberately ignores age for a product where mortality or morbidity is the primary driver would require extraordinary justification.

**EU AI Act Article 10.** For high-risk AI systems — which includes individual life and health insurance pricing under Annex III, paragraph 5(c) — Article 10(2)(f) requires that training, validation, and testing datasets are "relevant, sufficiently representative, and to the best extent possible, free of errors and complete." Article 10(5) permits processing of special category data, including health data, "strictly necessary for the purpose of ensuring bias monitoring, detection, and correction in relation to the high-risk AI systems."

More directly, Recital 44 states that "high-risk AI systems should not lead to discrimination prohibited by Union law." The relevant Union law here is Article 21 of the EU Charter of Fundamental Rights: "any discrimination based on any ground such as sex, race, colour, ethnic or social origin, genetic features, language, religion or belief, political or any other opinion, membership of a national minority, property, birth, disability, age or sexual orientation shall be prohibited."

Age is in that list. So is sex. So are genetic features — highly relevant for health and life insurance.

**The actuarial exception and where it stops.** EU law does recognise an actuarial exception for insurance. Council Directive 2004/113/EC (the Gender Goods and Services Directive), as interpreted post-Test-Achats (Case C-236/09), eliminated sex as a permitted general insurance rating factor from December 2012 onwards. The Age Discrimination aspects of Directive 2000/78/EC (the Employment Equality Directive) do not directly apply to insurance pricing as they target employment relationships. The Framework Equality Directive and Racial Equality Directive (2000/43/EC) create restrictions on ethnicity-based pricing — which feeds into the postcode proxy problem.

The important gap is this: none of these older directives was drafted with Article 10 bias monitoring of high-risk AI systems in mind. The EU AI Act does not contain an explicit actuarial exception equivalent to the UK Equality Act 2010, Schedule 3, paragraph 20. The UK statute says, in terms, that age-related differences in insurance are permissible if they are "effected by reference to actuarial or other data from a source on which it is reasonable to rely, and [the act] is reasonable having regard to the data and any other relevant factors." The EU AI Act says nothing equivalent.

Whether this gap will be filled by Commission guidance, by EIOPA interpretation, or by litigation is genuinely uncertain. Meding (arXiv:2501.12962, January 2026) identifies this as one of the three core unresolved tensions in EU AI Act compliance for insurance. We agree with that assessment.

---

## Why Chouldechova matters here

Chouldechova (2017) — "Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments" — proved that for any binary classifier, three fairness conditions cannot simultaneously hold unless base rates (i.e. outcome prevalences) are equal across groups:

1. **Calibration**: the predicted probability of an event equals the actual probability for each subgroup
2. **False positive rate parity**: the rate at which negatives are misclassified as positives is equal across groups
3. **False negative rate parity**: the rate at which positives are misclassified as negatives is equal across groups

For insurance pricing, re-read those three conditions in actuarial terms. Calibration means your expected loss rate for each age band matches the actual loss rate. False positive rate parity (in the pricing context: the rate at which low-risk individuals are overpriced) being equal across age groups. False negative rate parity (the rate at which high-risk individuals are underpriced) being equal across age groups.

The impossibility result says: if 25-year-olds have a different claim rate than 65-year-olds — and they do — you cannot achieve all three simultaneously. You must choose which fairness criterion to prioritise, and every choice sacrifices one of the others.

The EU AI Act does not specify which fairness criterion to use. Article 10's reference to "free of bias" is not operationally defined. The implementation guidance does not resolve the Chouldechova impossibility. The Act implicitly assumes you can satisfy non-discrimination requirements without specifying what those requirements mean in a world where base rates differ between protected groups.

For age, this is not a marginal problem. The claim rate differential between a 20-year-old and a 60-year-old for critical illness insurance is not a few percentage points. It is an order of magnitude. A model that achieves false positive rate parity across those groups will do so by systematically underpredicting risk for the older cohort and overpredicting it for the younger cohort. That is an actuarially unsound model. It will also misprice in the other direction if you chase false negative rate parity instead.

Calibration is the only criterion that survives contact with basic actuarial science. A calibrated model, by definition, charges each individual their expected cost (plus loadings). That is also the definition of the actuarial exception under the UK Equality Act: using "actuarial or other data" to price differential risk. These two things are the same.

But calibration is incompatible with demographic parity — the requirement that the average predicted value be equal across protected groups. And demographic parity is what many fairness frameworks, including some readings of the GDPR and the EU AI Act, implicitly assume when they talk about "removing discrimination."

The EU regulatory frameworks are not aligned with each other on this point. They cannot all be right.

---

## The UK position: FCA Consumer Duty as the operational framework

UK insurers operating solely in the UK domestic market are not subject to the EU AI Act. Post-Brexit, the Act does not apply to UK-regulated entities writing UK business. But two parallel tensions exist.

**The Equality Act 2010.** Schedule 3, paragraph 20 provides the actuarial exception explicitly: different terms for insurance "by reason of" a protected characteristic (which includes age under the Act) are not unlawful if the difference is "effected by reference to actuarial or other data from a source on which it is reasonable to rely" and the act "is reasonable having regard to the data and any other relevant factors." This is a proportionality test but it is well-understood and courts have applied it repeatedly in insurance contexts.

Schedule 3, paragraph 22 provides the equivalent exception for sex — now narrowly scoped following the Test-Achats-equivalent domestic analysis: sex-based pricing is prohibited for general insurance (motor, home), but the exception remains for certain life and health products where actuarial evidence justifies the differential.

The UK Equality Act's actuarial exception is significantly more legally coherent for insurance purposes than the EU AI Act's approach. It acknowledges the problem directly and provides a proportionality framework. The EU AI Act creates obligations without resolving the tension.

**FCA Consumer Duty (PS22/9).** The FCA's Consumer Duty, in force from 31 July 2023, does not ban differential pricing by age. It requires that the price offered represents fair value relative to the benefit provided. If an older customer genuinely has higher expected claims, charging them more is not a Consumer Duty breach. The breach would be charging them more than actuarial evidence supports, or treating the higher price as a justification for providing worse service or support.

The Consumer Duty's "fair value" requirement does, however, create a fairness audit obligation. The FCA expects firms to be able to demonstrate that their pricing approach produces outcomes that are fair across customer cohorts. That requires the kind of calibration monitoring that Chouldechova's theorem identifies as the only consistent fairness metric when base rates differ.

Our practical view: calibration-by-cohort monitoring, documented against the actuarial exception under the Equality Act 2010 and the Consumer Duty fair value requirement, is the defensible UK approach. It does not resolve the EU law tension — but it is the framework that UK regulators have actually endorsed.

---

## What the EU AI Act Article 10 obligation actually requires in practice

For firms in scope — primarily those with life, health, or income protection pricing AI systems in the EU, including UK insurers with Lloyd's Brussels operations or cross-border EEA policyholders — Article 10 requires demonstrable bias monitoring. Here is what we think that requires in practice, based on the current state of the law:

**1. Document your fairness metric choice.** The Act does not specify a metric. Choose calibration, explain why (Chouldechova's theorem; actuarial soundness; the only metric compatible with the Solvency II actuarial exception), and record this in your model card. The documentation must explain what metrics you considered and what the trade-offs are. This is a proportionality argument, not a guarantee of mathematical fairness.

**2. Run calibration monitoring by age band.** For each relevant age cohort in your life or health pricing model, compute the ratio of actual-to-expected claims (the A/E ratio). Document the monitoring frequency, trigger thresholds, and remediation actions. Monthly A/E monitoring by age band, with triggers at ±10% deviation from 1.0, is a defensible standard for an annual-repricing product.

**3. Document the actuarial justification for age use explicitly.** This should be in the model card. The justification is the causal relationship between age and claim incidence — not a correlational one. Age causes increased mortality, increased cancer incidence, and increased critical illness claims. It is not a proxy for something else. This is the distinction the EU Charter Article 21 framework does not draw clearly but that the UK Equality Act does.

**4. Do not run demographic parity as your primary fairness metric.** Demographic parity — equal average predictions across age groups — is incompatible with calibration when claim rates differ by age. Running it as a metric will produce numbers that look good but are actuarially meaningless. Worse, optimising for it will misprice.

**5. Consider the EU guidance gap as a risk, not a solved problem.** The Commission has not issued guidance resolving the tension between the actuarial exception and Article 10. EIOPA's Final Opinion (EIOPA-BoS-25-360, August 2025) addresses AI governance but does not provide a legal opinion on the Article 21 tension. Any firm applying for a conformity assessment under Article 43(2) should document this gap explicitly and state that it constitutes a legal risk that the regulator has not resolved.

---

## A practical worked example

Consider a UK life insurer writing individual term assurance for the EEA market through a Lloyd's Brussels vehicle. The pricing model uses a GBM with age, sum assured, smoker status, BMI, and medical history as inputs.

The model is a high-risk AI under Annex III 5(c). It must comply with Articles 9–15 by 2 August 2026.

Age is a direct input. The model is calibrated: for 40-year-olds, predicted mortality closely matches observed population and portfolio mortality for that age cohort. For 70-year-olds, the same. The age gradient in the model reflects the actual age gradient in mortality — log-mortality increases roughly linearly with age in the 30–70 range.

An Article 10 bias monitor that flags age as a "prohibited characteristic" and recommends reducing the model's sensitivity to age would be recommending actuarial malpractice. A model less sensitive to age would systematically underprice 70-year-old applicants and overprice 40-year-olds. The actuarial function sign-off under Article 48 of Solvency II would need to flag this as a material pricing risk.

The correct documentation path is:
1. Record that age is a direct input
2. Document the causal actuarial justification (WHO/ONS mortality tables, portfolio A/E analysis by age cohort)
3. Record that demographic parity is incompatible with calibration for this product, per Chouldechova (2017), and that calibration is the chosen primary fairness metric
4. Document the legal uncertainty regarding Article 21 vs the actuarial exception
5. Note that the UK Equality Act Schedule 3 para 20 actuarial exception provides the closest available legal framework, and apply it analogically pending EU-specific guidance

That is not a complete solution to the legal tension. It is the most defensible approach available given current guidance.

---

## The FCA 2026 AI review context

The FCA's 2026 multi-firm AI review — announced in the FCA's Business Plan published in April 2025 — has a specific focus on demographic outcomes. The FCA is examining whether pricing models produce systematically worse outcomes for customers sharing protected characteristics. This is distinct from the EU AI Act but creates parallel UK obligations.

The FCA has been clear, in its Consumer Duty guidance and in its published supervisory work on insurance pricing, that "fair value" requires firms to be able to demonstrate that differential pricing is actuarially justified. The word "demonstrate" matters. Saying "we use age because mortality increases with age" is not sufficient. You need A/E monitoring by cohort, documented and reviewable, showing that the model's age gradient reflects actual experience.

This is where the UK framework is actually more actionable than the EU AI Act. The FCA has a specific, operational expectation — demonstrate actuarial justification through empirical monitoring — rather than the Act's open-ended "bias monitoring" obligation that does not account for the actuarial exception.

---

## The bottom line

UK pricing actuaries should not panic about the EU AI Act fairness tension — but they should not ignore it either. The legal position for UK domestic business is manageable under the Equality Act 2010 actuarial exception, reinforced by the Consumer Duty. For EU business, the tension between Solvency II's actuarial pricing requirements and the EU AI Act's non-discrimination obligations is genuinely unresolved, and no current guidance fills the gap.

The statistician's answer — that calibration is the only fairness metric consistent with accurate risk pricing and the Chouldechova impossibility result — is correct. The question is whether EU regulators will reach the same conclusion in their eventual guidance. Based on how similar tensions have been resolved in data protection law (where the legitimate interests balancing test has generally favoured actuarial use of sensitive data), we think they will. But until that guidance arrives, document everything.

---

*Post 1 covers the legal and statistical tensions. For the monitoring tooling — calibration-by-cohort monitoring with formal statistical tests — see our [formal statistical monitoring post](/2026/04/01/formal-statistical-monitoring-insurance-pricing-models-drift-cusum/). For intersectional fairness auditing across combined protected characteristics, see our [CCdCov post](/2026/04/01/intersectional-fairness-distance-covariance-insurance-pricing/).*
