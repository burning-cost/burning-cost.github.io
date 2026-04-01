---
layout: post
title: "FCA 2026 Insurance Priorities: A Pricing Actuary Checklist"
date: 2026-04-01
categories: [regulation, blog]
tags: [FCA, consumer-duty, pricing-actuary, AI, premium-finance, access-to-insurance, SMCR, mental-health, proxy-discrimination, insurance-fairness, insurance-monitoring, insurance-governance, UK-insurance, checklist, fair-value, MS24-2]
description: "The FCA published its first annual insurance regulatory priorities document in February 2026. Here is everything a pricing actuary needs to do in response — ten action items with the context that makes each one unavoidable."
seo_title: "FCA 2026 Insurance Priorities: Pricing Actuary Checklist"
---

The FCA published its first annual insurance regulatory priorities document on 24 February 2026 — signed by Sarah Pritchard (Deputy CEO) and Graeme Reynolds (Interim Director of Insurance). It replaces over 40 Dear CEO letters. The format change is significant: instead of topic-specific letters that individual business lines could treat as someone else's problem, there is now one document that covers the whole picture and is addressed to the whole firm.

We have been writing about the individual pieces — premium finance, access, AI governance, SMCR — since the document landed. This post is the consolidation: what a pricing actuary specifically needs to have done, or be doing, by the end of 2026.

Ten items. Roughly ordered by urgency.

---

## 1. Run a proxy discrimination audit on travel underwriting

This is the most time-sensitive item on the list. The FCA has committed to "independently review travel underwriting decisions affecting consumers with pre-existing mental health conditions." That review begins in 2026. "Independently" means the FCA is not waiting for self-certification — they will look at the models.

The actuarial position is uncomfortable. The Money and Mental Health Policy Institute mystery-shopped 15 travel insurers in 2023: 60% decline rate for a severe bipolar profile. Which? documented a 1,159% premium increase in May 2025 for a bipolar plus personality disorder case. There is no published UK actuarial study specifically linking mental health conditions to travel insurance claim rates. Premiums of 27 times and decline rates of 60% without that data are legally vulnerable under Equality Act 2010 Schedule 3 Para 21 — which requires that differential treatment be based on "actuarial or other reliable data" that is "reasonable in all the circumstances."

The specific modelling risk is proxy discrimination. Travel models using prior claims history, occupation codes, or geographic variables may be picking up mental health conditions indirectly. Run `insurance-fairness` `FairnessAudit` with mental health condition indicators (if available) or with correlated proxies as `protected_cols`. If you cannot run the audit because the data does not exist, that is itself the finding — and it needs to reach the board.

Full detail, including the non-disclosure bias problem that corrupts claims data for this segment, is in our post: [FCA Access to Insurance: What Pricing Teams Actually Need to Do](/2026/03/28/fca-access-to-insurance-what-pricing-teams-need-to-know/).

---

## 2. Audit your mental health loading and decline rules against condition-specific data

Connected to item 1 but distinct. Even if you pass the proxy discrimination screen, blanket mental health loading rules may not survive the FCA's review.

The Equality Act test requires the justification to be condition-specific. A loading applied to "any mental health condition or history" that treats managed depression on stable medication the same as active psychosis is not satisfying the test — even if some actuarial data exists for the bucket as a whole. If your underwriting rules classify by severity, stability, or treatment status, the actuarial evidence needs to correspond to those classifications. If it does not, the rules need to change before the FCA review arrives.

Check also whether your signposting implementation is compliant. The ICOBS 6A.4 trigger rose to £200 from January 2026. The FCA's October 2025 post-implementation review found broken links and ambiguous referrals. Signposting does not fix the access problem, but non-compliance with the signposting rules is an additional exposure.

---

## 3. Assess housing tenure as a proxy for disability in contents pricing

The second access priority — home contents insurance for social renters — is less immediate than travel, but it creates a specific modelling problem pricing teams own.

67% of social renters in England do not hold contents insurance (FCA Financial Lives 2024). If your contents pricing model uses housing tenure as a rating factor, you may be amplifying that access barrier rather than reflecting it. Social tenancy correlates with disability — long-term health conditions, including mental health conditions, are disproportionately represented in social housing. A tenure variable in a contents model can therefore function as a proxy for disability even when built only to predict claims frequency.

The Fair4All Finance opt-out pilot is collecting expressions of interest from participating firms from Q1 2026. Findings are due by end 2027. If you are considering participation, the pricing question is acute: what is price elasticity at household incomes below £20,000 a year? Most contents models were not calibrated at that income level because the product was not historically sold there at scale.

Run `FairnessAudit` with `housing_tenure` as a `protected_col`. The output will not tell you whether to use tenure as a rating factor — that is a business decision. But it will tell you whether the variable produces materially different outcomes for protected groups, which is the question the FCA review will start from.

---

## 4. Produce a fair value assessment for premium finance specifically

The FCA MS24/2 final report landed on 3 February 2026. The headline number is the encouraging one: APRs fell an average 4.1% since 2022 due to Consumer Duty pressure, rising to 8% where the FCA directly challenged firms. Estimated annual consumer savings of £157 million.

The detail is less comfortable. The FCA found three categories of weak fair value methodology: no FVA document at all; analysis exists but lacks a firm basis; policies exist but are not followed. Premium finance is now explicitly named in the 2026 regulatory priorities as an active supervision priority. Firms with APRs materially higher than the market average can expect direct supervisory engagement.

Premium finance is used in approximately 23 million UK policies — roughly half of all motor and home. If your firm offers monthly payment and your FVA does not specifically cover the finance product (separate from the insurance product), that is the gap the FCA is looking for.

The FVA for premium finance needs to show that the APR reflects the actual cost structure: funding cost, bad debt, servicing cost, and margin. "We benchmarked against the market" is not sufficient if the benchmark set was selected to include only favourable comparators. You cannot outsource Consumer Duty accountability to the finance provider — PROD 4 sits with the insurer.

The pricing team's specific piece of this: segment the APR by risk profile and payment method. If monthly payers are systematically worse risks and your APR does not reflect credit risk, you are cross-subsidising in the wrong direction. If monthly payers are not worse risks and your APR is materially above the annual-equivalent rate, the FVA cannot close without addressing that gap. Both scenarios have come up in FCA direct engagements under MS24/2.

More on the mechanics: [FCA Premium Finance Market Study MS24/2: What Pricing Teams Need to Know](/2026/03/28/fca-premium-finance-market-study-ms24-2-pricing-teams/) and [Premium Finance APR as a Pricing Problem](/2026/03/25/premium-finance-apr-pricing-problem/).

---

## 5. Build out your AI outcome monitoring evidence

The FCA has committed to "evaluate the risks and opportunities of AI for insurance, assess how firms are using AI in their operations" throughout 2026. An industry report is expected. The Bank of England and FCA joint AI survey (published December 2025) found 46% of firms have only "partial understanding" of the AI they are using — which is precisely the gap the 2026 evaluation is designed to expose.

No new AI rules are coming. The FCA has said this repeatedly and clearly. What is coming is supervisory scrutiny under the frameworks that already exist — Consumer Duty, SMCR, PS21/5, and the Equality Act — applied by a supervisor who now has the technical capacity to read a model card and identify what is missing.

What a prepared pricing team needs to produce on request:

- A model inventory with version numbers, training data dates, last validation dates, and named accountable owners (SMCR SMF)
- For each production model: a calibration summary (A/E ratios segmented, not just aggregate), discrimination screening results, and a model risk tier classification
- Monitoring evidence showing drift detection is running in production, not just at deployment

The `insurance-governance` library covers the model inventory and governance documentation. `insurance-monitoring` covers A/E, calibration stability (`PITMonitor`), Gini drift (`GiniDriftBootstrapTest`), and PSI/CSI. If you are running AI models in production without these running continuously, that is the gap the FCA evaluation will identify.

The FCA's specific concern about "AI-enabled hyper-personalisation rendering some customers uninsurable" (quoted directly from the priorities document) maps to a proxy discrimination screen, not just a fairness metric at the portfolio level. A model can pass an aggregate fairness test and still create uninsurable profiles for a narrow protected group. The `DoubleFairnessAudit` in `insurance-fairness` runs both aggregate and individual-level screens.

Full context: [FCA AI Evaluation 2026: What the Regulator Is Looking at](/2026/03/25/fca-ai-evaluation-insurance-pricing/).

---

## 6. Check your Consumer Duty evidence pack covers AI-specific risks

This is an extension of item 5 but sits under a different governance process. The annual Consumer Duty fair value assessment needs to cover AI-specific risks explicitly, not just model performance metrics.

PRIN 2A.4.6 requires ongoing monitoring — not a point-in-time annual snapshot. The FCA's multi-firm review of Consumer Duty implementation described the evidence most firms were producing as "high-level summaries with little substance." The FCA has six active Consumer Duty investigations open as of March 2026. Consumer Duty enforcement is no longer hypothetical.

The pricing team's contribution to the evidence pack needs to cover: (1) model calibration with confidence intervals, (2) discrimination monitoring results, (3) drift detection status, and (4) the specific Consumer Duty price and value outcome — are products delivering fair value to the target market, including any sub-segments that AI personalisation has created?

The 12-step technical process for the pricing actuary's portion of the Consumer Duty fair value assessment is set out in full in: [Consumer Duty Fair Value Evidencing: A 12-Step Technical Checklist](/2026/03/25/consumer-duty-fair-value-evidencing-12-step-checklist-pricing-actuaries/).

---

## 7. Understand what SMCR Phase 1 actually changes — and what it does not

The FCA's commitment to halve the administrative burden of SMCR has generated a lot of commentary. Browne Jacobson called it an "unprecedented rollback." It is not.

CP25/21 Phase 1 changes are administrative: streamlined Statements of Responsibility, de-duplication of certifications, threshold inflation adjustments. No SMF roles are being removed. No prescribed responsibilities are changing. SS1/23 (which establishes model risk governance via the SMF4 CRO role) is not being amended. The impact on pricing governance is near-zero.

Phase 2 — where structural SMCR changes could in principle occur — requires HM Treasury legislation. There is no confirmed timeline. Speculating about what Phase 2 might do and adjusting governance frameworks in anticipation is premature and creates its own regulatory risk.

What you should not do: redraw SMCR accountability maps, remove named individuals from model governance documentation, or reduce the level of evidence produced for SMF sign-off on the basis of Phase 1. What you should do: map your Consumer Duty compliance cost for non-UK lines now, before the Q2 2026 consultation on disapplying Consumer Duty to non-UK business lands. If that disapply goes through, you will want to have a clear picture of what you are no longer required to do — and what you may choose to keep anyway because it is good practice regardless.

Full analysis: [FCA SMCR Rollback: What It Actually Means for Pricing Governance](/2026/03/26/fca-smcr-rollback-consumer-duty-disapply-pricing-governance/).

---

## 8. Monitor the pure protection market study — final report due Q3 2026

The FCA MS24/1 interim report (29 January 2026) is the pricing team's early warning for term assurance, critical illness, and income protection. Feedback deadline was 31 March 2026. The final report is expected Q3 2026.

The interim findings are significant: the FCA found that pricing competition in pure protection is limited, that some mortality assumptions are materially worse than population experience without adequate justification, and that commission structures may distort consumer outcomes. The Q3 final report will set the regulatory framework for the segment for the next several years.

If you price or validate pure protection products, the interim report is worth reading directly. The FCA's specific concern about mortality assumptions maps directly to the actuarial justification test — the same Equality Act test that applies to mental health in travel applies here where conditions or demographics are involved in the underwriting. [FCA Pure Protection Market Study: What Pricing Teams Should Prepare For](/2026/03/25/fca-pure-protection-market-study-what-pricing-teams-should-prepare-for/).

---

## 9. Keep EU AI Act scope correct in your compliance documentation

This is a quality-control item for firms operating across UK and EU. The widespread assumption in the market is that motor and property insurance pricing AI falls within the EU AI Act's high-risk classification. It does not.

Annex III of the Act lists "AI systems used to evaluate the creditworthiness of natural persons or establish their credit score" as high-risk. Premium finance involves credit — that AI application is potentially in scope. Motor vehicle liability insurance is listed separately in Annex III para 5(b) but only for the risk assessment used in access-to-credit decisions, not for the pricing AI used in the insurance product itself.

The practical consequence: general insurance pricing models (frequency, severity, demand) in motor and property are not high-risk AI under the Act. Running them through high-risk conformity assessment processes is unnecessary and creates misleading documentation. The governance standard that applies in the UK is Consumer Duty outcomes monitoring and SMCR — not EU Act Article 9 risk management.

Full analysis: [EU AI Act and Insurance Pricing: What You Actually Need to Know](/2026/03/28/eu-ai-act-insurance-pricing-what-you-need-to-know/).

---

## 10. Watch motor and home claims handling — it is where the FCA started

The first of the FCA's four 2026 priorities is consumer understanding and claims handling quality. The FCA identified 270,000 motor total loss compensation cases requiring remediation. A value measures review for motor and home concludes Q4 2026. A sales process analysis completed Q1 2026.

Claims is not typically a pricing team responsibility. But pricing and claims are connected in ways the FCA review will expose: if claims handling is poor in specific segments, the claims frequency and severity data those segments generate is itself compromised. A model trained on under-settled total loss claims is not predicting real claims cost — it is predicting the cost of inadequate settlement. If the FCA's remediation programme changes how total loss settlements are calculated, that changes the severity distribution the pricing model is built on.

The specific action: identify whether your training data includes total loss claims from the remediation period. If it does, assess the materiality of the settlement understatement on your severity model's A/E before the Q4 2026 value measures review surfaces it.

---

## The shape of the year

The FCA's 2026 priorities are enforcement-led in a way the previous Dear CEO letters were not. Six Consumer Duty investigations are active. Premium finance firms with outlier APRs are being contacted directly. The AI evaluation is generating the evidence base for future guidance. The travel underwriting review is happening independently, not through firms.

The common thread across all ten items is evidentiary. The FCA is not asking firms to certify intent. They are asking for data — on outcomes, on model performance, on the actuarial basis for underwriting rules. Pricing teams that can produce credible, documented, segmented evidence are in a substantially different position from those that cannot.

The libraries that cover the technical work — `insurance-fairness`, `insurance-monitoring`, `insurance-governance` — are linked throughout. The regulatory framework posts are below. The actuarial work is yours.

---

**Related posts:**
- [FCA 2026 Consolidated Insurance Regulatory Priorities](/2026/03/25/fca-2026-consolidated-regulatory-priorities-insurance/)
- [FCA AI Evaluation 2026: What the Regulator Is Looking at](/2026/03/25/fca-ai-evaluation-insurance-pricing/)
- [FCA Access to Insurance: What Pricing Teams Actually Need to Do](/2026/03/28/fca-access-to-insurance-what-pricing-teams-need-to-know/)
- [FCA Premium Finance Market Study MS24/2: What Pricing Teams Need to Know](/2026/03/28/fca-premium-finance-market-study-ms24-2-pricing-teams/)
- [FCA SMCR Rollback: What It Actually Means for Pricing Governance](/2026/03/26/fca-smcr-rollback-consumer-duty-disapply-pricing-governance/)
- [Consumer Duty Fair Value Evidencing: A 12-Step Technical Checklist](/2026/03/25/consumer-duty-fair-value-evidencing-12-step-checklist-pricing-actuaries/)
- [EU AI Act and Insurance Pricing: What You Actually Need to Know](/2026/03/28/eu-ai-act-insurance-pricing-what-you-need-to-know/)
- [FCA Pure Protection Market Study: What Pricing Teams Should Prepare For](/2026/03/25/fca-pure-protection-market-study-what-pricing-teams-should-prepare-for/)
