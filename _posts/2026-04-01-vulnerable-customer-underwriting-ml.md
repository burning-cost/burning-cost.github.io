---
layout: post
title: "What Machine Learning Can (and Cannot) Do for Vulnerable Customer Underwriting"
date: 2026-04-01
categories: [regulation, fairness]
tags: [vulnerable-customers, fca, consumer-duty, mental-health, travel-insurance, social-renters, proxy-discrimination, fairness, underwriting, fca-fg21-1, equality-act-2010, uk-insurance, insurance-fairness, access-to-insurance]
description: "The FCA has flagged travel insurance for mental health conditions and contents insurance for social renters as supervisory priorities. Here is what ML can genuinely help with, what it cannot fix, and what good practice looks like under Consumer Duty."
---

On 24 February 2026, the FCA published its first annual Regulatory Priorities report for the insurance sector — replacing the old portfolio letter format with something it intends to be clearer and more binding in tone. Two items on the access agenda stand out for pricing actuaries: the FCA's commitment to review travel insurance underwriting decisions for consumers with pre-existing mental health conditions, and its work with Fair4All Finance to increase home contents insurance uptake among social renters.

Neither problem is new. Both have become supervisory priorities because the industry has not solved them voluntarily.

## What the FCA actually said

The access-to-insurance section of the February 2026 report names 67% of social renters as lacking contents insurance — a figure from the FCA's own Financial Lives survey. The FCA will work with Fair4All Finance, insurers, and social housing providers to pilot opt-in (and potentially opt-out) schemes to close that gap. Separately, it is conducting an independent review of travel underwriting decisions affecting people with mental health conditions, where premiums can be "prohibitively expensive" or cover is simply unavailable.

This sits inside a broader Consumer Duty framing. The FCA's March 2025 review of vulnerable customer treatment found that only four in ten vulnerable consumers had disclosed their circumstances to providers, with roughly a quarter actively uncomfortable doing so. It found firms monitoring process outputs rather than customer outcomes, and little evidence of vulnerability considerations being embedded in product design from the outset. The regulator declined to update FG21/1 — the 2021 guidance on fair treatment of vulnerable customers — but the February 2026 priorities make clear it expects firms to act on the existing framework.

The signposting rules for travel insurance with medical conditions were already strengthened: from 1 January 2026, the premium trigger for mandatory signposting to specialist providers rose from £100 to £200 (indexed to CPI, reviewed every five years). The FCA estimates the original 2020 rules generated an additional 21,000 policy sales — a number that suggests the counterfactual without intervention was worse, but also that the intervention moved a modest share of the affected population.

## Where ML can genuinely help

### Granular segmentation to avoid blanket exclusions

The core problem with travel insurance for mental health conditions is not that the risks are uninsurable — it is that they have historically been priced via crude, binary underwriting rules. "Has the applicant ever been treated for depression?" admits no gradation between someone who had a six-week episode ten years ago and someone who is currently hospitalised. Gradient boosted models trained on claims data can surface the actual predictive variables — recency, severity, treatment status, comorbidities — and price them individually rather than applying a single loading or exclusion.

The practical constraint is data. Most travel insurers have limited historical claims data linking mental health diagnosis codes to trip-level outcomes. Reinsurers with broader books are better placed, but the individual insurer faces a cold-start problem. Pooled industry data schemes, which the ABI has historically been reluctant to sponsor for competitive reasons, would help more than any modelling sophistication applied to a thin dataset.

### Alternative data for social renters

The social renter problem is partly about willingness to pay, partly about product design, and partly about underwriting. Contents insurance in social housing has historically been unprofitable: high theft rates in certain estates, damp-related claims, difficulty reaching customers at renewal. Insurers who have tried to serve this market have often retreated.

ML can help in a specific, limited way: better use of property-level data (building age, type, location at postcode sector level) and tenancy-level data (tenure duration as a proxy for stability) can sharpen individual risk estimates and reduce the cross-subsidy from low-risk payers to high-risk ones. Pay-as-you-go products built around verified tenancy data from housing associations could reduce adverse selection by anchoring the risk pool to a defined population rather than self-selection.

Fair4All Finance's pilot is exploring automatic opt-in through social housing providers — essentially using the landlord as a distribution channel and population anchor. That is a product and distribution innovation rather than an ML one, but it creates the claims history that makes better pricing possible over time.

### Personalised pricing that does not exclude

There is a legitimate argument that finer risk segmentation helps vulnerable customers. If a pooled rate cross-subsidises high-risk from low-risk policyholders, it is the high-risk customers who price out first when insurers adjust downward for the low-risk and maintain or raise rates for the high-risk. Granular pricing can, in principle, hold rates lower for customers who are lower-risk within a group that was previously treated as homogeneous.

The evidence for this in practice is mixed. Côté, Côté & Charpentier (2025) formalise this tension through a proxy vulnerability score: the gap between a premium estimated with and without a protected attribute. A high proxy vulnerability score means the model is implicitly pricing the protected characteristic through correlated variables — which is both a fairness problem and, under the Equality Act 2010, potentially unlawful for characteristics such as disability. Granularity without fairness constraints does not solve the access problem; it may worsen it.

## What ML cannot solve

### The data gap is structural, not technical

Machine learning requires labelled training data. For rare outcomes in underserved populations, we often have neither the claims history nor the exposure denominator. A travel insurer who has historically declined to cover applicants with bipolar disorder has no claims data for that population — because they were never on risk. Survival bias in training data produces models that extrapolate confidently into territory where they have no information. This is not a modelling problem that more compute addresses.

### Adverse selection when cover becomes mandatory

If the FCA moves toward mandated provision — either through direct regulation or through Consumer Duty pressure that makes refusal to cover a difficult-to-defend outcome — the adverse selection dynamic changes sharply. Mandated cover at regulated rates without risk selection concentrates the highest-risk cases in the book. Insurers respond by either pricing up the entire pool (reducing uptake further) or restricting on dimensions that are not prohibited (geography, occupation, travel destination). Neither outcome serves the stated policy objective.

This is not an argument against mandated provision. It is an argument that mandated provision requires a cross-subsidy mechanism — either explicit (government scheme, reinsurance facility) or implicit (spread across the broader book) — and that pretending ML alone can make the economics work is misleading.

### The fundamental tension between actuarial fairness and access

Risk-based pricing is coherent from an actuarial standpoint: like risks should pay like premiums. But insurance has always involved solidarity — the healthy subsidise the sick, the cautious subsidise the reckless, the young subsidise the old in certain lines. The question is not whether solidarity exists but who bears it and by whose choice.

FG21/1 defines a vulnerable customer as someone who, due to their personal circumstances, is especially susceptible to harm. The Equality Act 2010 imposes direct and indirect discrimination protections for disability and other protected characteristics. Consumer Duty's PS22/9 requires firms to deliver good outcomes for all customers, including those in vulnerable circumstances. None of these frameworks resolves the underlying economic tension. They constrain how firms can express that tension in pricing and product design.

Pricing actuaries need to be honest with their boards about what that constraint costs. Requiring an insurer to cover a population at rates that do not reflect expected loss either produces underwriting losses or requires cross-subsidisation. That cross-subsidy has a source, and identifying it is a commercial and regulatory question, not a technical one. ML does not make it disappear.

## What good practice looks like under the current framework

Under Consumer Duty, firms should be reviewing their product ranges against the needs of vulnerable consumers. That means:

- Stress-testing decline and exclusion rates against vulnerability characteristics, not just protected ones under the Equality Act. Are you declining materially more often when mental health history is declared? What is the actuarial justification for the exclusion versus a loading?
- Monitoring outcomes, not processes. The FCA's March 2025 review found firms tracking whether vulnerability flags were logged, not whether vulnerable customers received cover at fair value.
- Documenting the underwriting rationale for blanket exclusions. An exclusion that cannot be defended with claims data becomes much harder to maintain under Consumer Duty.
- Engaging with the FCA's travel insurance review proactively. The FCA has said it will conduct an independent review of underwriting decisions. Firms with defensible, data-driven underwriting methodologies are in a better position than those relying on undocumented heuristics.

On the social renter question specifically: the Fair4All Finance pilot creates an opportunity to acquire the data that makes future pricing improvement possible. Participating in the pilot, even at breakeven, is an investment in a dataset.

## The honest summary

ML can help by replacing coarse underwriting rules with granular, evidence-based risk assessment — but only where the claims data exists to train on. It can help by identifying proxy discrimination in existing models and correcting for it. It can support product design that reaches underserved populations through better segmentation and targeted distribution.

ML cannot create insurable data where none exists. It cannot resolve the adverse selection that follows mandated provision without a cross-subsidy. It cannot make the tension between risk-based pricing and access disappear by being clever enough.

The FCA's February 2026 priorities are a signal that supervisory patience with industry inaction on both these fronts is running out. The response that serves firms best — commercially and from a regulatory standpoint — is genuine engagement with the underwriting evidence and honest documentation of where the economic constraints lie. Pointing to ML capability as a reason the problem will solve itself is not that response.

---

*References: FCA Regulatory Priorities for Insurance, February 2026 ([PDF](https://www.fca.org.uk/publication/regulatory-priorities/insurance-report.pdf)); FCA Travel Insurance Signposting Review, 2025 ([fca.org.uk](https://www.fca.org.uk/publications/multi-firm-reviews/travel-insurance-signposting-rules-consumers-medical-conditions-review)); FCA Finalised Guidance FG21/1, February 2021; FCA Policy Statement PS22/9 (Consumer Duty), July 2022; Equality Act 2010; Côté, Côté & Charpentier, 'Proxy Vulnerability Score', 2025; Fair4All Finance contents insurance pilot, 2026.*
