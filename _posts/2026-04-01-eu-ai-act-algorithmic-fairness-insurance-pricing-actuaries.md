---
layout: post
title: "The EU AI Act and Insurance Pricing Fairness: The Tension Nobody Has Resolved"
date: 2026-04-01
categories: [regulation, fairness]
tags: [eu-ai-act, fairness, non-discrimination, algorithmic-fairness, age-rating, solvency-ii, gender-directive, test-achats, equality-act, fca-consumer-duty, proxy-discrimination, insurance-fairness, life-insurance, health-insurance, eiopa, meding-2026, arXiv-2501-12962, UK-insurance, annex-iii, protected-characteristics]
description: "Age is a legally permitted rating factor under Solvency II. Age is also a protected characteristic under EU law. The EU AI Act imposes non-discrimination obligations on high-risk AI systems without resolving this contradiction. We map the tension, explain what it means for UK pricing teams, and give a concrete audit approach."
author: Burning Cost
---

Kristof Meding published a paper in January 2026 (arXiv:2501.12962) with a title that is an accurate description of its subject: "It's complicated." The paper maps EU non-discrimination law onto the algorithmic fairness requirements in the EU AI Act for high-risk AI systems. Insurance pricing is explicitly one of those systems. The paper's core finding is not a condemnation of the Act — it is a careful legal analysis showing that the Act's non-discrimination obligations and the existing body of EU insurance law are in genuine tension, and the Act does not resolve that tension.

This post is our interpretation for pricing actuaries. We are not restating what we have already written about the Act's scope, the definition of AI systems, or the August 2026 compliance timeline — all of that is in our [definitive guide](/2026/03/28/eu-ai-act-insurance-pricing-what-you-need-to-know/). This post focuses on the specific problem Meding identifies: what happens when you run a high-risk AI pricing system and the EU's non-discrimination law meets the EU's insurance pricing law and they say different things.

If you work in UK life, health, or income protection pricing — and especially if your firm has EEA operations, Lloyd's Brussels exposure, or cross-border EU policyholders — the August 2026 deadline is not just a documentation exercise. It is the point at which a legal contradiction you have been operating inside for years becomes subject to active enforcement.

---

## The contradiction at the centre

Here is the core problem.

EU non-discrimination law prohibits discrimination on the basis of age. Article 21 of the EU Charter of Fundamental Rights (binding since the Lisbon Treaty) establishes age as a prohibited ground of discrimination. The Employment Equality Framework Directive (2000/78/EC) operationalises this in employment contexts; for goods and services including insurance, the prohibition applies through the Charter directly and through national implementing legislation. Age is a protected characteristic.

EU insurance law explicitly permits age as an actuarially justified rating factor. The Gender Goods and Services Directive (2004/113/EC) allows member states to permit sex-based differentiation where actuarial data justifies it, subject to the constraints confirmed by the Test-Achats ruling (Case C-236/09, March 2011). For age, the position has been clearer: Solvency II (Article 20 of Directive 2009/138/EC) requires that technical provisions and premium calculations reflect the risk profile of the insured population, and actuarial practice universally uses age as one of the primary drivers of that profile.

The EU AI Act adds a third layer. Article 10(5) requires that, where strictly necessary for the purposes of bias detection and correction in high-risk AI systems, providers may process special categories of personal data including data relating to age. Articles 10(2)(f) and 10(5) together instruct that training and validation data should be relevant, representative, and free from "errors and biases" that could produce discriminatory outcomes prohibited by EU law.

The problem is that "discriminatory outcomes prohibited by EU law" is not defined by the AI Act. The Act cross-references the applicable non-discrimination framework but does not specify which framework governs for insurance. Is it the general EU Charter prohibition on age discrimination? The specific carve-outs in Solvency II? The Test-Achats framework for sex? The national implementation of the Racial Equality Directive? Meding's analysis shows these frameworks do not produce the same result, and the AI Act leaves the resolution to future case law or regulatory interpretation.

---

## Three specific tensions for insurance pricing

**1. Age as a rating factor**

A GBM life insurance pricing model trained on age, smoker status, sum insured, and medical history will use age as one of its most important features. Age is causally related to mortality. Using it produces actuarially accurate prices. Under Solvency II, the insurer is *required* to reflect the risk in premiums. Under general EU non-discrimination principle, using age to produce materially different prices for individuals is age discrimination.

The resolution in current EU law is that actuarial justification creates an exception — but the AI Act does not incorporate that exception explicitly. Article 10's requirement to detect and mitigate "errors and biases" does not say "except where the bias is actuarially justified." The European Commission has not published guidance on how the actuarial exception interacts with the AI Act's fairness requirements.

Meding makes the point plainly: a life pricing model trained to be accurate will produce outcomes that are measurably disparate by age, because that is what the data requires. Standard ML fairness metrics — demographic parity, equalised odds, individual fairness — will all flag this model as unfair. The model is not unfair in any actuarial sense. It is actuarially mandatory.

**2. Sex-based pricing and the Test-Achats aftermath**

The Test-Achats ruling prohibited the use of sex as a rating factor in insurance from 21 December 2012 — effective across all EU member states. UK insurers apply this by default given the historical integration. A pricing model that uses sex directly violates the Gender Directive as confirmed by that ruling.

But the AI Act's bias detection requirements extend to *indirect* effects. A life pricing model that does not use sex as an input variable can still produce outcomes that differ by sex if the other variables are correlated with sex. Occupation, smoker status, body mass index, and certain medical history patterns are all sex-correlated in population data. A model trained to maximise predictive accuracy on historical data will absorb some of the sex signal through those proxies.

The question Meding identifies, and which the Act does not answer, is: does Article 10's requirement to detect and correct bias extend to disparate impact through correlated variables when those variables are individually actuarially justified? If yes, the requirement is computationally and actuarially very demanding. If no, the proxy discrimination problem is unaddressed by the Act.

EIOPA addressed this partially in its Final Opinion (EIOPA-BoS-25-360, August 2025). Footnote 13 states: *"Proxy discrimination arises when sensitive customer characteristics (e.g. ethnicity), whose use is not permitted, are indirectly inferred from other customer characteristics (e.g. location) that are considered legitimate."* EIOPA's framing treats proxy discrimination as a risk to be monitored rather than a bright-line prohibition. The EIOPA opinion, however, applies to *non-high-risk* AI — the EU AI Act's own provisions for high-risk systems are a different standard, and EIOPA has not issued equivalent guidance for them.

**3. Ethnicity and postcode**

The third tension is the one UK pricing teams have been grappling with domestically for longer. A model using postcode, driving behaviour, and vehicle type can produce materially different prices for policyholders from different ethnic backgrounds — because those variables correlate with ethnicity in UK (and EU) population data. Citizens Advice documented a £280 per year average difference in motor insurance premiums in postcodes where more than 50% of residents are people of colour (2022 analysis; estimated total: £213 million annually).

Under EU law, the Racial Equality Directive (2000/43/EC) prohibits discrimination on grounds of racial or ethnic origin in access to goods and services including insurance. A pricing model that produces racially disparate outcomes through ostensibly neutral variables may violate this directive even if no one intended it to. The EU AI Act's Article 10 obligations around bias monitoring are designed to surface exactly this. But the Act does not define what level of disparity constitutes prohibited discrimination — it defers to applicable non-discrimination law, which itself does not specify a quantitative threshold.

---

## What the UK framework says — and where it is clearer

The UK position after Brexit is that the EU AI Act does not apply as domestic law. The relevant UK frameworks are the Equality Act 2010, FCA Consumer Duty (PS22/9, in force July 2023), FCA PS22/3 (home and motor pricing practices), and PRA SS1/23.

For age specifically, the UK Equality Act 2010 includes an insurance exception — Schedule 3, paragraph 20 — that permits age-related differences in insurance premiums and benefits where they are based on actuarial or other relevant data that it is reasonable to rely on. This is a cleaner resolution than the EU AI Act provides: the actuarial exception is statutory and explicit. A UK life insurer using age as a rating factor with actuarial justification has a defined legal basis for doing so.

For sex, the UK also relies on the insurance exception (Schedule 3, paragraph 22), which allows sex-based pricing differences supported by relevant actuarial data, subject to a reasonable reliance test. This is more permissive than the post-Test-Achats EU position, which bans sex as a direct factor with no exception for new business. UK firms writing new EU business must comply with the EU rule. UK domestic business can still use the Schedule 3 exception — though in practice very few UK insurers use sex as a direct rating factor in new business given regulatory scrutiny and the availability of sex-correlated telematics and other proxies.

For ethnicity and proxy discrimination, the UK Equality Act's indirect discrimination provisions (Section 19) do apply. There is no actuarial exception for proxy discrimination of the kind Citizens Advice documented. The FCA's position (van Bekkum et al. cite it as: firms using AI that "embed or amplify bias leading to worse outcomes for some consumer groups might not be acting in good faith unless differences can be objectively justified") does not provide a numerical threshold either. But the FCA framework is at least coherent: justify or remove the disparity, and be audit-ready.

The FCA Consumer Duty adds an operational layer. PRIN 2A.4 requires that pricing must provide fair value. FCA TR24/2 (August 2024) found that most fair value assessments reviewed lacked granularity to identify whether specific customer groups received poor value at the sub-group level. The direction of FCA supervision is towards model-level evidence that protected groups are not systematically disadvantaged — evidence that has to exist before the supervisor asks for it, not after.

The practical advantage of the UK framework for pricing actuaries is that, while it is genuinely demanding, it is at least intelligible. The Equality Act's actuarial exception tells you what justification looks like. FCA Consumer Duty tells you what monitoring looks like. The EU AI Act's Article 10 leaves those questions open.

---

## The computational feasibility problem

Meding identifies a second concern in the paper: computational feasibility. The EU AI Act requires providers of high-risk AI systems to assess and mitigate bias in training data and model outputs. The range of algorithmic fairness metrics available for this assessment is large — demographic parity, equalised odds, calibration within groups, counterfactual fairness, individual fairness — and these metrics routinely conflict with each other. A model that satisfies demographic parity will, in most real datasets, fail equalised odds, because the two require different things.

For insurance pricing, this is not an abstract problem. Consider a health insurance pricing model assessed on cancer claim probability. Demographic parity requires that the model's prediction be statistically independent of age. Calibration requires that the model's prediction be accurate — that the predicted probability matches the observed claim rate within age groups. These cannot both hold simultaneously if cancer incidence varies by age (which it does, substantially).

The Act does not tell you which metric to prioritise when they conflict. It requires you to detect and correct for bias without specifying what bias means in a context where different conceptions of fairness are mathematically incompatible. Meding's diagnosis is that this represents a genuine gap in the Act's technical requirements — not a political compromise or an oversight, but an unresolved question about what fairness actually requires when the ground truth is itself unequally distributed across groups.

For life and health pricing actuaries in scope of the August 2026 deadline, this matters concretely. Your Article 10 compliance documentation must address bias detection. You will need to choose metrics. You should choose them carefully, document why you chose them, and explain the trade-offs. The documentation showing you considered and deliberately chose a calibration-within-groups approach over a demographic parity approach is the kind of reasoning the conformity assessment requires. If you have not made that choice consciously, you cannot document it.

---

## What UK teams should actually do

Split by situation.

**UK insurer with no EU exposure**

The EU AI Act does not apply. But the underlying fairness questions do not go away. The FCA's 2026 insurance supervisory priorities (published 24 February 2026) explicitly flag AI/ML pricing model governance and monitoring for bias. The six open Consumer Duty investigations mean the FCA has demonstrated enforcement appetite. For life and health pricing, the actuarial fairness question — which protected characteristics does your model disparately affect, and what is the actuarial justification for each — should be live regardless of EU Act scope.

Run a proxy discrimination audit using `insurance-fairness` `FairnessAudit`. For life and health models, the relevant protected characteristics under the Equality Act are age (with actuarial exception potentially available), sex (with actuarial exception potentially available for UK domestic business), disability, and race/ethnicity (no actuarial exception). The output tells you the magnitude of disparity; your actuarial team needs to determine which disparities are justified, which require mitigation, and which require escalation. That analysis belongs in your model documentation.

**UK insurer with EEA subsidiary or Lloyd's Brussels exposure**

The EU AI Act applies to your life and health pricing models used for EU-domiciled policyholders, with a compliance deadline of 2 August 2026. The Article 10 bias detection requirement means you need documented evidence that you have assessed the model for discriminatory outcomes prohibited by EU law.

Our recommendation: use the FCA framework as your baseline (it is more operationally defined) and document how you have mapped it to the EU Act requirements. Specifically: document which fairness metrics you are using, document the known conflicts between metrics (calibration versus demographic parity, for example), document your rationale for prioritising calibration-within-groups for actuarially justified factors, and document the residual disparities you are monitoring. This creates the Article 10 audit trail. It is also what EIOPA's proportionality principle would expect of a competent insurer.

On the age question specifically: document the Solvency II actuarial requirement explicitly. Note that age is being used because it is actuarially required, not because it was selected as a targeting variable. Include the empirical justification — the mortality or morbidity table, the statistical analysis. If the EU Commission subsequently issues guidance clarifying that the actuarial exception extends to the AI Act context, you have the documentation ready. If they do not, you have a defensible record of having taken the question seriously.

**Lloyd's syndicates**

Lloyd's Brussels is within EU jurisdiction. Actuarial Function Guidance (Lloyd's, March 2025) already requires formal validation of pricing methodologies. The EU AI Act compliance adds Article 9–15 documentation requirements for any life or health high-risk AI system writing EU-domiciled business. The Lloyd's Pricing Maturity Matrix (November 2024) and the actuarial function requirements provide the governance backbone — the AI Act compliance layer is documentation uplift on top of a framework that should already exist.

The fairness question for Lloyd's is most live for any syndicate writing international health, travel insurance, or life reinsurance with EU cedants. For these books, proxy discrimination auditing is both an EU AI Act obligation and a sensible risk management practice.

---

## A note on insurance-fairness

Our `insurance-fairness` library ([PyPI](https://pypi.org/project/insurance-fairness/), [GitHub](https://github.com/burning-cost/insurance-fairness)) includes tools designed for exactly this audit:

- `FairnessAudit`: compute a range of fairness metrics simultaneously across protected groups. The output explicitly shows where metrics conflict — a demographic parity score alongside a calibration-within-groups score alongside an equalised odds score. Making the conflict visible is the first step to documenting your rationale for choosing among them.
- `detect_proxies`: identify rating factors that are statistically correlated with protected characteristics above a specified threshold. Particularly relevant for the ethnicity/postcode proxy issue.
- `IntersectionalFairnessAudit` (v1.0.0, CCdCov method from Lee et al. 2025): detects disparate impact on intersectional subgroups — combinations like young female or elderly disabled — that marginal audits miss. Fairness gerrymandering is a real pattern in insurance model testing; intersectional auditing is the correction.

Running these tools does not itself produce EU AI Act compliance. It produces the evidence base that Article 10 compliance documentation requires. The tool outputs, with your team's interpretation and the actuarial justification for residual disparities, are what goes into the conformity assessment.

---

## The honest answer to "is our model compliant?"

The honest answer is that no one can give you a clean yes right now. Meding is right that the Act creates fairness obligations for high-risk AI in insurance without resolving the tension with the insurance law that permits actuarially justified differentiation. Until the European Commission issues interpretive guidance, or an EU supervisory authority brings an enforcement action that produces a ruling, the boundary is not precisely defined.

What that means practically: document your reasoning, document your metrics, document the trade-offs you are making and why. If your life or health pricing model uses age because age drives mortality, the documentation of that actuarial rationale is your compliance evidence. A regulator assessing your conformity assessment will look at whether you have engaged seriously with the fairness question, not whether you have achieved demographic parity on a metric that actuarial reality makes impossible.

The UK FCA framework is actually better calibrated to this problem — its combination of a statutory actuarial exception and Consumer Duty's outcome-focused monitoring is more actionable than the Act's cross-references to unresolved non-discrimination law. If you are a UK-regulated firm with EU compliance obligations as well, our advice is to build the UK compliance framework first (it is more precise) and then layer the EU AI Act documentation on top. The evidence you need overlaps substantially. You are not doing double work; you are documenting the same thing in two formats.

The August 2026 deadline is four months away. For life and health teams in scope, that is not a comfortable margin if you are starting from zero. For teams with a functioning model governance framework already, it is achievable. The critical first task — before any of the documentation work — is the model inventory: which of your pricing models are AI systems, which are in scope for EU customers, and which of those are for life or health at the individual level. Everything else depends on getting that right.

---

## Related reading

- [EU AI Act and Insurance Pricing — What You Actually Need to Know](/2026/03/28/eu-ai-act-insurance-pricing-what-you-need-to-know/) — the definitive guide to scope, AI system definition, conformity assessment, and the four-month timeline
- [Intersectional Fairness in Insurance Pricing: Why Auditing Age and Gender Separately Is Not Enough](/2026/04/01/intersectional-fairness-distance-covariance-insurance-pricing/) — the fairness gerrymandering problem and CCdCov implementation
- [Proxy Discrimination in UK Motor Pricing](/2026/03/03/your-pricing-model-might-be-discriminating/) — the `insurance-fairness` library for proxy discrimination testing
- [FCA 2026 Insurance Priorities: A Pricing Actuary Checklist](/2026/04/01/fca-2026-insurance-priorities-pricing-actuary-checklist/) — ten action items from the FCA's February 2026 regulatory priorities document
