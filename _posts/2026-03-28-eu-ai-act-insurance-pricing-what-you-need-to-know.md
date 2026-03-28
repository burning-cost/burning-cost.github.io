---
layout: post
title: "EU AI Act and Insurance Pricing — What You Actually Need to Know"
date: 2026-03-28
categories: [regulation]
tags: [eu-ai-act, regulation, model-governance, annex-iii, high-risk-ai, glm, gbm, life-insurance, health-insurance, motor-pricing, eiopa, pra-ss123, conformity-assessment, compliance]
description: "Most industry guidance on the EU AI Act and insurance pricing is wrong in ways that matter. Motor and property pricing is not high-risk AI. Traditional GLMs may not be AI systems at all. Conformity assessment is self-assessment, not third-party audit. This is the corrected, accurate guide — built from the primary sources."
---

Most industry commentary on the EU AI Act and insurance pricing is wrong in ways that matter. Not wrong about the deadline — 2 August 2026 is real and is not moving. Not wrong about the penalties — up to €35 million or 7% of global annual revenue for non-compliance with the high-risk obligations. Wrong about *who is in scope*.

The widespread assumption, including in Big 4 insurance sector guides and most consultancy briefings, is that all insurance pricing AI is high-risk under the Act. Read Annex III. It is not.

This post is the accurate guide. We have worked from the primary sources: the Act itself (Regulation (EU) 2024/1689, OJ 12.7.2024), the European Commission Guidelines on the definition of AI systems (C/2025/3554, 6 February 2025), and the EIOPA Final Opinion on AI governance (EIOPA-BoS-25-360, August 2025). We flag where the law is clear, where it is genuinely ambiguous, and what each category of insurer should actually do.

---

## Start here: the three things most guides get wrong

**1. Motor and property pricing is not high-risk AI under the Act.**

Annex III, paragraph 5(c) — the operative provision — classifies as high-risk: *"AI systems intended to be used for risk assessment and pricing in relation to natural persons in the case of life and health insurance."*

That is the complete text. Life and health. Not motor. Not property. Not liability, household, commercial lines, or anything else. The Act's drafters drew the line at life and health because the fundamental rights case is strongest there: denial of life cover or unaffordable health insurance has severe and often irreversible effects on individuals. The same logic does not hold for motor third-party liability. Paragraph 5(b), which some guides cite, covers creditworthiness AI — not insurance pricing.

**2. Traditional GLMs may not be AI systems at all.**

The Act applies only to AI systems as defined in Article 3(1). The European Commission's February 2025 Guidelines state explicitly (paragraph 42): *"systems used to improve mathematical optimisation or to accelerate and approximate traditional, well established optimisation methods, such as linear or logistic regression methods, fall outside the scope of the AI system definition."*

EIOPA cited this passage directly in footnote 11 of its Final Opinion. A fitted Poisson GLM with fixed distribution family, pre-specified link function, and no automated structure selection may not be an AI system under the Act. If a model is not an AI system, no obligations apply at all.

**3. Conformity assessment is self-assessment.**

Article 43(2) is unambiguous: AI systems in Annex III, points 2 to 8 — which includes point 5, insurance pricing — follow the conformity assessment procedure based on internal control (Annex VI). No notified body. No third-party audit. The insurer conducts and documents the assessment themselves. This is considerably less onerous than most commentary implies.

---

## What Annex III paragraph 5(c) actually covers

Let us be precise about scope. High-risk AI for insurance pricing requires all three of:

1. An AI system (not excluded from the Article 3(1) definition)
2. Intended for risk assessment and pricing of *life or health insurance*
3. In relation to *natural persons* (individual-level, not portfolio or book)

This captures:
- Individual life insurance pricing models using GBMs, neural networks, or random forests
- Individual health and PMI pricing models using the same model classes
- Critical illness and income protection pricing at individual level
- Annuity pricing models where they generate individual quotes based on individual risk characteristics

This does not capture:
- Motor pricing (GBM, GLM, or anything else)
- Property and household pricing
- Commercial lines pricing
- Liability pricing
- Group insurance pricing where the unit of pricing is not the individual natural person
- Portfolio-level reserving or planning models
- Reinsurance pricing (counterparty is not a natural person)

The "intended purpose" test in Article 3(17) means it is the purpose the system is built for, not how it is actually used. A GBM built for motor pricing that someone then uses to experiment with a life product is still a motor pricing system for classification purposes. Equally, a life pricing GBM that performs no better than random is still high-risk AI if it was intended for that use.

---

## The AI system definition: GLMs, GBMs, and the grey zone

Even for life and health pricing, the obligations only apply if the model is an AI system. The Article 3(1) definition requires inference from input data to generate outputs "with varying levels of autonomy." The Commission's February 2025 Guidelines build on this: linear and logistic regression methods fall outside the definition because they solve a well-understood mathematical optimisation problem rather than inferring their own structure.

Work through the categories:

**Likely outside the AI system definition:**
- Traditional GLMs with pre-specified distribution family, link function, and feature set (the actuary specifies the structure; the algorithm solves for coefficients)
- Logistic regression frequency models where all modelling decisions are made before fitting
- GAMs with explicitly pre-specified smooth terms, basis types, and fixed knot positions
- Credibility/Bühlmann-Straub models with fixed update rules

**Likely inside the AI system definition:**
- Gradient boosted trees (XGBoost, LightGBM, CatBoost) — these infer complex feature interactions from data autonomously; the tree structure is not pre-specified
- Neural networks of any architecture
- Random forests
- Stacked ensembles containing any of the above components
- AutoML pipelines

**Genuinely ambiguous — document and assess:**
- GAMs with automated smoothing parameter selection (mgcv with REML-selected lambda values) — the smoothing structure is being inferred, not specified
- GLMs with automated variable selection (LASSO, elastic net, stepwise AIC) — the model structure is learned from data
- Regularised regression where the feature set and penalty emerge from cross-validation

The Commission's guidance is non-binding. The ultimate legal test is autonomy in inferring structure. A LASSO GLM sits closer to a GBM than to a traditional GLM on that test. Legal certainty will not arrive before the August 2026 deadline. Teams with genuinely ambiguous models should document the assessment either way and be prepared to defend it.

**The practical implication:** A life pricing team using a pure traditional Poisson GLM — pre-specified structure, no automated selection, fixed form determined before fitting — has a credible, citable basis to argue the model falls entirely outside the Act. That assessment should be documented explicitly, with reference to Commission Guidelines paragraph 42 and EIOPA-BoS-25-360 footnote 11. If the assessment holds, no Articles 9–15 obligations apply.

---

## What the high-risk obligations actually require

For life and health pricing teams who are in scope, the requirements in Articles 9–15 are substantial but not irrational. The Act assumes a level of model governance that good pricing functions should already have. The gap, for most teams, is documentation rather than practice.

### Article 9: Risk management system

A documented, continuous risk management process covering identification of known and foreseeable risks (discriminatory pricing, proxy discrimination, distribution shift, data quality failures, automation bias, model degradation, feedback loops), estimation and evaluation of risks, and adoption of risk management measures. "Continuous" means updated through the model's life, not produced once at launch.

Article 9(10) is worth noting: where providers are already subject to risk management processes under other Union law, the Article 9 requirements may be *combined* with existing procedures. A Solvency II insurer already has data quality obligations under Delegated Regulation Article 260(1)(a)(ii). The AI Act risk management process can be layered onto that structure rather than built independently.

### Article 10: Data governance

Training, validation, and test datasets require documented governance: collection methodology, origin, preparation steps, known limitations, representativeness assessment, and — critically — examination for biases that could affect health and safety, fundamental rights, or lead to prohibited discrimination.

For insurance, this means documenting why the training period was chosen, what exclusions were applied (large losses, incomplete policy years, selection from past underwriting), and what known biases exist. Article 10(2)(f) feedback loop warning applies directly: if pricing feeds into retention, and retention affects the next training dataset, that chain must be documented and managed.

### Article 11: Technical documentation

Annex IV specifies nine categories of required documentation, to be completed *before* the system is placed in service. The structure maps closely onto what a thorough PRA SS1/23 validation report already covers: model rationale, design choices, training methodology, validation results, dataset descriptions, monitoring plan, risk management summary. The gap for most teams is not that the work has not been done — it is that the documentation does not exist in a form that satisfies Annex IV's completeness requirements.

### Article 12: Record-keeping and logging

High-risk AI systems must technically allow for automatic logging of events over the system's lifetime. For pricing, this is system-level traceability: model version identifiers, training run logs, monitoring alerts, change control records, and consequent actions. The Act does not require logging of every individual quote — unlike biometric identification systems, pricing models are not subject to that granularity. GDPR Article 22 creates separate individual rights obligations if pricing decisions are fully automated, but that is a distinct question from Article 12.

### Article 13: Transparency to deployers

The system must be designed so deployers (underwriters, distribution teams, actuarial function) can interpret its output and use it appropriately. This is an operational model card — not the technical documentation of Article 11, but the document a non-technical committee member or underwriter can use to understand what the model does, where it performs poorly, what inputs it requires, and when to escalate.

Required content includes: declared accuracy level with stated metrics and known performance limitations; accuracy for specific person groups or subpopulations where applicable (the fairness disclosure); input data specifications; and documentation of human oversight measures. If the model performs worse for applicants under age 25 due to thin data, that must be in the Article 13 document.

### Article 14: Human oversight

High-risk AI systems must be designed with human-machine interface tools so they can be effectively overseen by natural persons. The Act does not require a human to approve every individual premium quote. It requires that a human *can* intervene: anomaly detection, override capability, the ability to suspend automated pricing, and training for oversight persons on automation bias.

EIOPA's Final Opinion (para 3.30) maps this to the three lines of defence structure: Board/AMSB takes ultimate responsibility and must have sufficient AI knowledge; the actuarial function (already responsible for underwriting opinions under Solvency II Article 48) takes the primary technical oversight role; compliance and audit verify legal requirements; the DPO handles GDPR.

### Article 15: Accuracy, robustness, cybersecurity

Accuracy metrics must be declared in the Article 13 instructions-of-use document. No mandatory benchmarks exist yet — the Commission is commissioning standards work, but nothing is finalised. For insurance, typical declared metrics include Gini coefficient or AUC-ROC, deviance or log-loss calibration, out-of-time validation results with confidence intervals, and calibration ratios by segment. The system must perform consistently throughout its lifecycle — which is an ongoing monitoring obligation, not a one-time validation.

On feedback loops (Article 15(4)): if a model continues to learn after deployment, or if pricing systematically affects the population that will form future training data, these loops must be documented and mitigated. Reweighting training data by propensity model is a standard mitigation.

---

## Conformity assessment: what self-assessment actually involves

Article 43(2) routes insurance pricing directly to Annex VI internal control. The steps are:

1. Verify that the Quality Management System per Article 17 is in place — this is the overarching governance framework that contains Articles 9–15
2. Examine the technical documentation (Annex IV) and assess that Articles 9–15 have been satisfied
3. Verify that design, development, and post-market monitoring are consistent with the technical documentation

The provider's own authorised representative signs the Declaration of Conformity under Article 47. Before deployment, the system must also be registered in the EU AI database under Article 71.

A "substantial modification" under Article 83 triggers a new conformity assessment. The boundary that matters in practice: retraining on new data with the same architecture and within pre-specified performance tolerances is probably not a substantial modification. Changing the algorithm (GLM to GBM), changing the target variable, or changing the intended purpose is a substantial modification.

---

## UK position and extraterritorial reach

The EU AI Act does not apply as domestic UK law. The FCA and PRA stated in April 2024 they will not introduce equivalent AI-specific rules in the near term. The existing UK framework for pricing model governance — PRA SS1/23, SMCR accountability, FCA Principles including Consumer Duty — creates substantively overlapping obligations without the Act's specific structure.

The extraterritorial reach of Article 2 is real and cannot be ignored: the Act applies to providers and deployers in third countries where the AI output is *used* in the EU. For UK insurers, this means:

- **Pure domestic insurer, no EU-domiciled policyholders:** EU AI Act does not apply. SS1/23 and FCA Principles govern.
- **UK insurer with EEA subsidiary:** The subsidiary is within EU jurisdiction. Pricing models used for the subsidiary's policyholders are in scope.
- **UK parent running the model, EU subsidiary using the output:** UK parent is the provider under Article 2. The EU AI Act applies to those systems.
- **Lloyd's managing agents:** Brussels-written business and European syndicate operations are within EU jurisdiction. Pricing that feeds Brussels is in scope.
- **Cross-border service arrangements:** Any UK insurer providing services to EU-domiciled policyholders, even without a subsidiary, where the pricing model output is used in EU territory.

The UK AI Regulation Bill (re-introduced to the House of Lords on 4 March 2025) would create an AI Authority and codify governance principles as binding duties. As of March 2026 it remains a Private Member's Bill without government backing. We would not architect compliance around it.

One practical conclusion worth stating plainly: building to EU AI Act standard is the right approach regardless of whether you are strictly in scope. The governance requirements — documented risk management, technical documentation, data governance, human oversight, performance monitoring — are what the FCA and PRA expect under existing frameworks. Firms building this infrastructure are not over-complying; they are building what should already exist.

---

## What EIOPA's Final Opinion adds

EIOPA-BoS-25-360 is widely cited as evidence that the EU AI Act covers general insurance. It does not. Read the opening paragraphs: the opinion explicitly applies to AI systems in insurance that are *not* high-risk under the Act. It is the governance framework for lower-risk use cases — the vast majority of insurance AI. The high-risk obligations under Articles 9–15 sit separately and are not the subject of the opinion.

Where the EIOPA opinion is worth reading carefully:

The proxy discrimination treatment (footnote 13) is specific: *"Proxy discrimination arises when sensitive customer characteristics (e.g. ethnicity), whose use is not permitted, are indirectly inferred from other customer characteristics (e.g. location) that are considered legitimate."* EIOPA cites the Commission's Test-Achats guidelines: price differentiation by car engine size is permitted; differentiation by body size (correlated with gender) is not. This is relevant to any pricing team using postcode, driving behaviour, or other geographic or behavioural features.

The explainability section (paras 3.25–3.28) endorses SHAP and LIME explicitly, while noting that their assumptions and limitations must be documented. It also allows for statistical rather than deterministic explanations "when duly justified and documented" — relevant for teams where SHAP values are computed at segment rather than individual level.

The EIOPA record-keeping table (Annex I) is more granular than most teams expect. For higher-risk uses, it requires: algorithm rationale and exact library version references; pseudo-random seeds for reproducibility; documented performance thresholds that define when retraining is triggered. These are reasonable documentation requirements for any production model, EU AI Act scope notwithstanding.

The proportionality principle runs through the entire opinion: a simple GLM motor pricing model at a small insurer has different governance expectations to a neural network health pricing system at a major composite. EIOPA is not expecting identical documentation depth regardless of materiality. But "low risk" is not an exemption from governance — it is a calibration of effort.

---

## A practical decision sequence for March 2026

Work through this in order.

**Step 1: Determine if the model is an AI system.**

Is it a traditional GLM with pre-specified structural form and no automated structure selection? Document an assessment citing Commission Guidelines paragraph 42 and EIOPA footnote 11. If the assessment holds, no AI Act obligations apply. If it is a GBM, neural network, or ensemble: it is an AI system.

**Step 2: Does the EU AI Act apply to you?**

Are you placing AI systems on the EU market, or are your model outputs used by EU-domiciled policyholders? If purely domestic with no EU exposure: the Act does not apply. If you have an EEA subsidiary, Lloyd's Brussels exposure, or cross-border EU policyholders: the Act applies to the systems used for those customers.

**Step 3: Is the model high-risk?**

High-risk means: AI system (Step 1) + life or health insurance + individual natural person pricing. Motor, property, commercial, and group pricing: not high-risk. Life, health, PMI, critical illness, income protection, annuities at individual level: high-risk.

**Step 4: If high-risk, build the Article 9–15 documentation stack.**

The timeline from now to 2 August 2026 is approximately four months. For a life or health pricing team starting from a good SS1/23 baseline, that is achievable. For a team without systematic model governance, it is extremely tight.

The practical sequence:

*Now to end of April:* Complete the model inventory. Every production model in scope gets documented: intended purpose, model class, population served, data provenance, known limitations. Determine which are AI systems, which are high-risk, which are genuinely ambiguous. This is the foundation; everything else depends on it.

*May to June:* Build the Article 9–15 documentation for each high-risk model. Risk management system (Article 9). Data governance documentation (Article 10). Annex IV technical documentation (Article 11). Logging capability (Article 12). Deployer transparency document (Article 13). Oversight procedures with named owners (Article 14). Accuracy declarations and monitoring plan (Article 15).

*July:* Conduct the internal conformity assessment. Check the quality management system is in place. Review the documentation for completeness. Identify and close gaps. Draft the EU Declaration of Conformity.

*By 2 August:* Sign the Declaration of Conformity. Register in the EU AI database under Article 71.

For teams using SS1/23-aligned model governance already: much of this work is documentation uplift, not new work. The model card exists; it needs to be checked against Annex IV requirements. The validation report exists; it needs accuracy metric declarations added. The monitoring system exists; logging events need to be wired to the model record.

For teams without systematic governance: the compliance gap is structural. Four months is not enough time to build governance infrastructure and produce compliant documentation simultaneously. The pragmatic approach is to prioritise the highest-GWP, most customer-facing models, get those to compliant standard, and document the remediation timeline for the rest.

---

## What the Act does not require

The Act does not require models to be accurate by any absolute standard. It requires declared accuracy, validated accuracy, and ongoing monitoring. A model with a Gini of 0.28 is compliant if that is documented, validated, and in the transparency pack. A model with a Gini of 0.50 is non-compliant with no documentation.

It does not require explainability at the individual quote level. Article 13 requires transparency to deployers. Article 86 creates individual rights to explanation, but that is a separate provision and a separate obligation.

It does not prohibit risk-based pricing. The Act explicitly accommodates actuarially justified pricing differentiation where it is non-discriminatory. That is the same standard the FCA has applied since the Gender Directive and the Test-Achats ruling. The Act does not change the underlying fairness obligations; it adds governance requirements around documenting and monitoring compliance with them.

It does not require model-free pricing or remove actuarial discretion. The human oversight requirements (Article 14) presuppose that humans remain in the loop. The Act's architecture is built on the assumption that actuaries and underwriters exercise judgment, not that AI replaces it.

---

## Related reading

- [EU AI Act update: motor pricing probably isn't high-risk (and your GLMs may not be AI)](/2026/03/28/eu-ai-act-motor-pricing-not-high-risk-glm-scope/) — the corrections post that preceded this guide
- [EU AI Act August 2026: What Your Pricing Team Must Do](/2026/03/25/eu-ai-act-august-2026-insurance-pricing/) — the original post, accurate for life/health teams, overstated for general insurance
- [The PRA Just Named AI a Supervisory Theme](/2026/04/10/pra-ai-supervisory-theme-pricing-models/) — the UK domestic governance picture
- [Proxy Discrimination in UK Motor Pricing](/2026/03/03/your-pricing-model-might-be-discriminating/) — the `insurance-fairness` library for EIOPA proxy discrimination requirements
