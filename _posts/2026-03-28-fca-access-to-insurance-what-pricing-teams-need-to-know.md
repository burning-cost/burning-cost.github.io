---
layout: post
title: "FCA Access to Insurance: What Pricing Teams Actually Need to Do"
date: 2026-03-28
categories: [regulation, fairness]
tags: [FCA, consumer-duty, equality-act, access-to-insurance, mental-health, social-renters, proxy-discrimination, insurance-fairness, fair-value, pricing, actuarial, UK-insurance, travel-insurance, contents-insurance]
description: "The FCA's 2026 access priorities are not a compliance problem — they are an actuarial evidence problem. The question is whether the data exists to justify what your models are doing. We go through what that means in practice."
seo_title: "FCA Access to Insurance 2026: What Pricing Actuaries and Data Scientists Need to Do"
---

The FCA published its first annual sector-specific insurance regulatory priorities on 24 February 2026. There are four priorities. One of them — Increasing Access to Insurance — is described in every legal and regulatory newsletter as "firms should review their Consumer Duty obligations." That is not wrong, exactly, but it misses the point by a wide margin.

The access priority is not primarily a Consumer Duty problem. It is an actuarial evidence problem. The FCA is asking whether the data exists to justify current mental health loadings and declines in travel insurance. For social renters, it is asking whether granular pricing models have inadvertently created the access barrier the FCA is now trying to fix. Those are questions for pricing teams, not compliance teams. This post covers what you should actually be doing.

---

## What the FCA said, precisely

The priorities document names two specific access commitments:

- To "work with trade bodies, insurers and consumer groups to increase contents insurance uptake for social renters"
- To "independently review travel underwriting decisions affecting consumers with pre-existing mental health conditions"

The second one is the most important for pricing teams. Note the word "independently" — the FCA is not asking firms to self-certify. They are going to look at the underwriting models themselves.

This is not a new rule. There is no Dear CEO letter, no consultation paper, no new ICOBS requirement. The mechanism is supervisory: multi-firm reviews, data requests, and enforcement via Consumer Duty and the Equality Act where standards fall short. But "no new rule" does not mean low stakes. The FCA is explicitly linking this work to the HM Treasury Financial Inclusion Strategy (November 2025) and the Motor Insurance Taskforce Final Report (December 2025), which is an unusual degree of alignment between FCA supervision and government social policy. This is not a tick-box exercise.

---

## Social renters: where the pricing model is part of the problem

67% of social renters in England do not hold contents insurance. The comparison figure is 11% of homeowners. That is not a small gap — it is structural.

The IFoA and Fair By Design named insurance as "the biggest contributor to the poverty premium in essential services in the UK" in their September 2021 joint report. Their analysis of the mechanism is precise and uncomfortable: the shift from broad risk pooling to granular individual-level pricing has produced more accurate models and worse access. A model that identifies social renters as higher risk — higher theft rates in some postcode areas, higher proportion in flats with shared entry — is doing what we built it to do. The problem is that the people who need insurance most are being priced out of it.

The IFoA report recommended a Flood Re-style scheme for priced-out vulnerable consumers in 2021. What actually emerged, four years later, is the Fair4All Finance opt-out pilot: tenants in social housing auto-enrolled in contents insurance at "a few pounds a month," with an active opt-out required to leave the scheme. Expressions of interest from participating firms opened Q1 2026, funded by dormant assets. Findings are due by end 2027.

For pricing teams, the practical question the opt-out pilot creates is this: what will the opt-out rate be? The answer is almost entirely a function of how the product is priced. If the auto-enrolled premium is set above what social renters find affordable at their income level, opt-out will be high and the pilot will fail. The parameter that matters is price elasticity at household incomes below £20,000 a year, which is roughly the social renter median. Most contents pricing teams do not have a demand model calibrated at this income level because the product has historically not been sold into this segment at scale.

There is a second modelling question that is easier to miss: if your pricing model uses housing tenure as a rating factor, you are already in the access problem. Social tenancy is predictive of claim frequency in some datasets, and using it is actuarially defensible. But social tenancy is also correlated with disability — people with long-term health conditions, including mental health conditions, are disproportionately represented in social housing. A tenure variable in a home or contents model can therefore function as a proxy for disability even when that is not the intent. Under the Equality Act 2010 Schedule 3 Para 21, a pricing variable that produces materially worse outcomes for a disabled person needs actuarial justification — specifically, that the differential is based on "actuarial or other reliable data" and is "reasonable in all the circumstances." The fact that the variable is predictive is necessary but not sufficient. It has to be predictive of the insured risk, not of disability itself.

---

## Mental health travel insurance: the evidential gap

The Money and Mental Health Policy Institute mystery-shopped 15 travel insurers in 2023 with a severe bipolar disorder profile. Nine of them declined — a 60% decline rate. For those who accepted, premiums ran up to 27 times higher than the same policy without a mental health condition. Which? confirmed in May 2025 that the picture is unchanged: one insurer charged a 1,159% premium increase for a bipolar plus personality disorder profile.

These numbers are dramatic. The regulatory problem they create is specific: under Equality Act 2010 Schedule 3 Para 21, an insurer may treat a disabled person differently in insurance, but only where that treatment is based on "actuarial or other reliable data" and is "reasonable in all the circumstances." Mental health conditions with substantial long-term effects on daily life meet the legal definition of disability under Section 6 of the Act.

The FCA's independent review will look for the actuarial data behind each loading and decline rule. And here is the problem: there is no published UK actuarial study specifically linking mental health conditions to travel insurance claim rates. This is not speculation — the MMHPI, the ABI working group convened in Q1 2026, and the FCA's own review are collectively trying to establish whether such data exists within firms. If it does not, then premiums of 27 times and decline rates of 60% are built on something other than actuarial evidence.

There are two further complications that make the current practice hard to defend even if some actuarial data exists.

First, blanket mental health loadings treat clinically distinct conditions as equivalent. Bipolar disorder, schizophrenia, managed depression on a stable medication regime, and a single episode of anxiety three years ago are not the same risk. The Equality Act test requires the justification to be condition-specific. A loading applied to "any mental health condition or history" that captures the resolved past episode in the same bracket as active psychosis is almost certainly not satisfying the legal test.

Second, non-disclosure corrupts the claims data. 45% of people with mental health conditions never disclose to their travel insurer, according to MMHPI survey data. The population of disclosed mental health claimants is therefore a self-selected subset — either people who disclosed despite high loadings, or people with serious active conditions who had no choice. The apparent claim rate for disclosed mental health is systematically inflated relative to the true population rate. Any actuarial study built on insurer records without adjusting for this selection bias is measuring the wrong thing.

The signposting regime (ICOBS 6A.4) raised the trigger threshold from £100 to £200 from January 2026. The FCA's October 2025 post-implementation review found approximately 21,000 additional policy sales from the signposting rules — but also found non-compliance, broken links, and ambiguous referrals. Signposting is not a solution to the access problem; it is a patch on a system that is declining and overcharging in ways that may not be legally justified.

---

## The Equality Act structure: what the independent review will test

The insurance exception in Equality Act 2010 Schedule 3 Part 5 is narrower than most underwriters think. The relevant provisions are Paragraphs 20 and 21.

For risk-assessment products — which includes travel insurance — the exception applies where:

1. The different treatment is based on "relevant information from a source on which it is reasonable to rely"
2. The different treatment is "reasonable in all the circumstances"
3. For products involving mortality, injury or illness risk: "actuarial or other reliable data" specifically

The burden of justification sits with the insurer. A blanket policy based on stereotype or convention, without a source document, fails all three tests. The FCA's independent review is effectively an audit of whether the third test is being met. Firms who cannot produce an actuarial justification document will be in the worst possible position.

What the review will not do: require firms to accept risks outside their appetite, mandate cross-subsidisation, or override actuarially justified differentials. The Consumer Duty point is similar — Consumer Duty does not create a right of access to any product. What it does require is that where a firm cannot serve a consumer, they communicate this clearly and signpost effectively.

---

## What pricing teams should actually build

### 1. Audit the actuarial justification for every mental health loading and decline

Pull the data source behind each rule:

- Is there a source document? When was it compiled? Is it UK-specific?
- Is it condition-specific — does it separate bipolar from depression from anxiety from historical conditions now resolved?
- Does it link the specific condition to the specific travel insurance peril (medical expenses abroad, cancellation, curtailment)?
- Is antidepressant use coded as a proxy for mental health condition? Antidepressants are prescribed for conditions ranging from mild anxiety to severe psychosis — they are not a reliable proxy for claim risk

If the data source is "industry convention" or "underwriting judgment" with no document: this is the live risk. The FCA review will be looking for exactly this gap. Document now what you have, and document clearly where the gap is.

### 2. Check your proxy variables for correlation with disability

For any model using variables that could correlate with mental health conditions or disability:

- Housing tenure — social renters over-represented in the disability population
- Payment method — monthly payment correlates with lower income, which correlates with stress and mental health conditions
- Credit score — mental health conditions are associated with financial difficulty
- Postcode deprivation indices — mental health prevalence increases with deprivation

The test is not whether the variable is predictive. It may be. The test is whether the model produces materially worse outcomes for the disability population and whether that outcome is actuarially justified as required by Schedule 3 Para 21.

`insurance-fairness` provides the tooling for this audit. `FairnessAudit` with `protected_cols` can run demographic parity and counterfactual fairness checks. If you do not have direct observation of mental health status, you can run the audit against proxies — housing tenure, deprivation decile — to identify whether the model produces systematically different outcomes for those groups. That will not be a definitive Equality Act assessment, but it will tell you whether there is a signal worth investigating further.

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=travel_pricing_model,
    data=df_quotes,
    protected_cols=["housing_tenure"],          # proxy for disability
    prediction_col="quoted_premium",
    outcome_col="claims_incurred",
    exposure_col="exposure",
)
report = audit.run()
print(report.summary())
```

A high demographic parity ratio between social renters and owner-occupiers in travel insurance premiums is not, by itself, unlawful. But it is a finding you want to understand before the FCA does.

### 3. Start logging decline rates

If you are not already logging:

- Decline rates by product line
- Loadings applied and their stated basis
- Whether a mental health condition or disability was declared

The FCA independent review will request this data. Firms that cannot produce it are in a weaker position on every question the review asks. This is not a complex infrastructure problem — it is a decision to record data that you are currently discarding.

### 4. Run a fair value assessment for products with mental health exclusions

If a product excludes mental health conditions but is priced comparably to a full-cover product: what proportion of the target market has a mental health condition? The Pure Protection Market Study interim report (MS24/1, January 2026) found "a protection gap persists particularly for consumers with pre-existing medical conditions." If the mental health population is material to the target market, an exclusion-only product at full-cover pricing is a candidate fair value failure under Consumer Duty Price and Value outcome.

### 5. Build a demand model for the social renter segment

For anyone involved in the Fair4All Finance pilot or preparing to be: the opt-out rate is a function of price. Building a price sensitivity model for household incomes below £20,000 a year requires different training data than most insurers' existing elasticity models. The work to do now is to identify whether that segment is represented in your quote data, and if it is, to estimate elasticity at low income levels before the pilot forces the question empirically.

---

## What is genuinely new in 2026

It is worth being precise about which parts of this are new and which are longstanding.

**New:** The FCA commitment to independently review travel underwriting decisions — not just signposting compliance, but the actuarial models themselves. The explicit linkage of FCA supervision to HM Treasury financial inclusion policy. The ABI working group on mental health underwriting, convening Q1 2026, which is the first formal forum specifically on underwriting (not just communication standards). The Fair4All Finance opt-out pilot with FCA backing.

**Longstanding but now under more scrutiny:** The 67% social renter uninsured figure — this has been in the FCA Financial Lives Survey data for years. The MMHPI decline rate and loading data — the 2023 report remains the primary source, with Which? 2025 confirming it is still current. The ABI voluntary Mental Health Standards have existed since 2020; the independent review will provide external verification of whether they have made any difference.

**Not yet resolved:** No published UK actuarial data linking specific mental health conditions to travel insurance claim rates. No finalised product design for minimum-value contents insurance for social renters. No regulatory decision on whether opt-out enrolment is lawful under ICOBS — the pilot is testing this.

---

## The core tension pricing actuaries should name clearly

The IFoA 2021 analysis makes the underlying problem explicit: models that are more accurate at the individual level can produce worse societal outcomes. A better model identifies social renters as higher risk and prices them accordingly. That is the model doing its job. The societal consequence is that people who need insurance — and who, when asked, say they value their possessions at over £5,000, with 25% valuing them above £25,000 — cannot afford or cannot obtain it.

Consumer Duty does not resolve this tension. It requires firms to justify value and communicate clearly, but it does not mandate cross-subsidisation. The FCA is not, in this priorities document, asking firms to unprice risk.

What it is asking is narrower and more tractable: whether the actuarial data exists to justify what the models are actually doing. For mental health travel insurance, the answer in many firms is probably no. For social renters contents insurance, the question is whether the pricing model has created a problem that the industry is now being asked to fix by other means.

Pricing teams are in the centre of both questions. The work is not primarily regulatory compliance — it is getting the actuarial evidence in order for decisions that are already being made, and understanding what the models are actually doing to the people they are pricing.

---

*The insurance-fairness library implements proxy discrimination auditing and fairness-accuracy tradeoff analysis. Source and documentation at [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness). FCA Insurance Regulatory Priorities 2026: [fca.org.uk/publication/regulatory-priorities/insurance-report.pdf](https://www.fca.org.uk/publication/regulatory-priorities/insurance-report.pdf). Equality Act 2010 Schedule 3: [legislation.gov.uk/ukpga/2010/15/schedule/3/part/5](https://www.legislation.gov.uk/ukpga/2010/15/schedule/3/part/5).*
