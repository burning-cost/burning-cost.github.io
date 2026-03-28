---
layout: post
title: "The PRA Just Named AI a Supervisory Theme. What That Means for Your Pricing Models."
date: 2026-04-10
categories: [regulation, governance, pricing]
tags: [pra, fca, consumer-duty, model-risk, ai-regulation, eu-ai-act, frc, tas-100, proxy-discrimination, insurance-fairness, insurance-governance, insurance-monitoring, insurance-causal, model-validation]
description: "In January 2026 the PRA named AI as an insurance supervisory theme for the first time. The FCA published a bias research note in December 2024. The FRC updated TAS 100 to make bias assessment explicit for ML models. And the EU AI Act compliance deadline for high-risk systems is August 2026. For UK pricing teams, the regulatory direction of travel is now clear enough to act on."
---

In January 2026 the PRA named AI as an insurance supervisory theme for the first time. That sentence is worth reading slowly. Not a mention in the appendix. Not a footnote about emerging risks. A first-tier supervisory priority, sitting alongside capital adequacy and climate in the Dear CEO letter sent to every regulated insurer at the start of the year.

That is new. And it has practical consequences for how pricing teams should be running their ML models, documenting their governance, and auditing for bias.

This post sets out what has changed, what the regulatory expectations actually are (rather than what vendors will tell you they are), and what pricing teams should do about it.

---

## What changed in January 2026

The PRA's annual Dear CEO letter published in January 2026 contains three passages that matter for pricing model owners.

First, AI is named explicitly as an enabler of innovation with novel risks attached — specifically data quality, third-party dependencies, and cyber exposure. The PRA expects firms to "adopt new technologies responsibly, ensuring deployment does not compromise safety and soundness, data integrity, or fair treatment of policyholders." That last phrase is doing material work. Fair treatment of policyholders is a Consumer Duty obligation, and the PRA is now linking it explicitly to AI deployment.

Second, the PRA is intensifying scrutiny of model assumptions. The specific concern is internal models that assume future underwriting performance improvements "not supported by historical experience." That is aimed at capital models, but the logic applies directly to ML pricing models: if your gradient boosting model assumes it will generalise beyond the training distribution in ways that your historical validation does not support, you have a documentation problem. The PRA has signalled it will "intensify engagement" with firms where gaps between assumed and realised profitability are material.

Third, data quality is framed as foundational, with investment in "fit-for-purpose systems" expected. Not aspired to. Expected.

The context is important. The PRA held two banking-sector CRO roundtables on AI in October 2025 with 21 SS1/23-regulated firms and published findings in November 2025 noting that ML models are "particularly prone to overfitting due to high parameter counts" and that this vulnerability is "amplified when datasets fail to represent the target population adequately." Those sessions were run under SS1/23 — they were for banks, not insurers — but the PRA has now folded the same concerns into its insurance supervision priorities.

What the PRA has NOT done: there is no insurer equivalent of SS1/23 (the banks' Model Risk Management supervisory statement) as of Q1 2026. No dedicated AI survey for insurers. No thematic findings on insurance pricing model governance. The pressure is being applied via the supervision letter, not hard rules. This matters for how you calibrate your response — the ask is demonstrably better governance, not a specific form submission.

---

## The FCA is building the intellectual case for enforcement

The FCA's headline position is unchanged: no AI-specific rules. Principles-based regulation via Consumer Duty, SM&CR, and the FCA Handbook. But in December 2024 the FCA published something that deserves more attention than it has received in pricing circles.

The December 2024 research note — *A literature review on bias in supervised machine learning* — is the first in what the FCA describes as a series. It explicitly references insurance claim prediction as a use case. Its core findings: historical data embeds past exclusion practices; sampling issues and variable selection choices during modelling introduce additional bias; mitigation techniques exist but need human review and contextual judgment.

That is not an academic exercise. That is the FCA building the intellectual foundation it will need to justify enforcement or formal guidance on algorithmic pricing. When the FCA subsequently wants to take action against a firm for discriminatory pricing outcomes, it will cite this research note as establishing that the risks were well-understood and publicly documented. The fact that it was published in December 2024 means it is already in the record.

The mechanism for enforcement already exists. FCA Guidance FG22/5 (Consumer Duty) warns that using algorithms, including machine learning or artificial intelligence, which embed or amplify bias could lead to worse outcomes for some groups of customers — and that doing so might not constitute acting in good faith toward consumers. That provision has been in force since July 2023. What the December 2024 research note does is signal that the FCA is actively thinking about how to apply it.

The February 2026 FCA insurance priorities publication adds a further signal: the FCA will "evaluate the risks and opportunities of AI for insurance, assess how firms are using AI in their operations." It specifically flags that AI-enabled hyper-personalisation of insurance "runs the risk of rendering some customers uninsurable, or even potential discrimination." This is not a fringe concern internally — it is a stated 2026 priority.

The Treasury Committee's January 2026 report is more pointed still. It criticised the FCA and PRA for a "wait-and-see" approach to AI and called for the FCA to publish comprehensive guidance on consumer protection rules applied to AI by end of 2026. That guidance, if and when it arrives, will almost certainly reference algorithmic pricing bias. The FCA has a 90-day deadline to respond to the Committee's recommendations.

---

## FRC updated TAS 100. Your ML models are in scope.

The Financial Reporting Council updated TAS 100 (Technical Actuarial Standards) in October 2024 specifically to address AI and ML. The headline is simple: existing actuarial professional standards are not waived for algorithmic models.

TAS 100 now explicitly covers model bias — examples and identification — understanding and communication, governance frameworks, and model stability, all as they apply to AI/ML. The FRC held a joint webinar with the IFoA in March 2025 to explain the guidance.

The practical consequence: if you are a Fellow of the IFoA working with pricing ML models, bias assessment and explainability are not optional extras you can defer until there is a specific regulatory mandate. They are professional standards obligations you already have. This has been true since October 2024.

For Lloyd's syndicates, there is an additional layer. Lloyd's Actuarial Function Guidance published in March 2025 requires formal Syndicate Actuarial Function opinions on methodology, explicit documentation of model limitations and governance, and compliance with IEULR and RARC monitoring protocols. ML pricing models are in scope. The Pricing Maturity Matrix (November 2024) establishes tiered expectations for model sophistication — syndicates at higher tiers are expected to have correspondingly more rigorous validation infrastructure.

---

## The EU AI Act deadline is August 2026 — and it applies to some UK firms now

The EU AI Act came into force in August 2024. The full compliance deadline for high-risk AI systems under Articles 9–49 is **2 August 2026**. That is four months away.

Under Annex III, AI systems for "risk assessment and pricing in relation to natural persons in the case of life and health insurance" are explicitly classified as high-risk. Obligations include conformity assessments, technical documentation, quality management systems, automated operational logs, Fundamental Rights Impact Assessments before deployment, human oversight requirements, and minimum six-month log retention.

Two important caveats. First, this is life and health pricing only — P&C pricing AI is not explicitly high-risk under Annex III. Second, UK firms without EU exposure are not legally subject. But UK firms with EU operations, EU-admitted business, or EU customer exposure face direct compliance obligations regardless of where they are headquartered.

The UK government has not issued guidance on EU AI Act compliance for UK firms. The joint April 2024 BoE/FCA/PRA statement confirmed no new AI-specific UK rules. This creates a deliberate regulatory divergence — UK supervisors are relying on principles-based frameworks while the EU has prescriptive technical requirements. UK firms with EU exposure have to navigate both simultaneously, which in practice means the stricter EU requirements define the floor.

---

## What pricing teams should actually do

The regulatory picture adds up to the same practical ask from three directions: document your models better, audit for bias systematically, and monitor performance continuously. This is not conceptually novel — it is what good model governance has always meant. The change is that the bar for "demonstrably adequate" has moved, and three of our open-source libraries address it directly.

### Proxy discrimination auditing under Consumer Duty

The FCA's Consumer Duty obligation is the most immediate practical pressure. Your model does not use gender, ethnicity, or religion as rating factors. But if your rating factors are correlated with those protected characteristics — and area-based variables, vehicle type, and even telematics behaviour all carry some such correlation in practice — the proxy discrimination risk is real.

[`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) implements the FCA's ask. It runs FairnessAudit to measure demographic parity and calibration by group, IndirectDiscriminationAudit to detect proxy relationships between rating factors and protected characteristics, and MulticalibrationAudit to test whether calibration holds not just at the population level but across intersecting subgroups. We demonstrated what this finds on 67,856 real motor policies [in a previous post](/2026/03/27/fairness-audit-real-insurance-data-ausprivauto/) — driver age was flagged as a proxy for gender, which is actuarially justified under Section 19 of the Equality Act, but the flagging itself is what you need in your governance documentation to show the analysis was done. "We checked, here is what we found, here is our objective justification" is the Consumer Duty evidential standard.

### SS1/23-style validation documentation

The PRA cannot compel insurers to follow SS1/23 (it applies to banks). But the January 2026 Dear CEO letter expects the same rigour in a different wrapper: robust justification for model assumptions, documented evidence of model performance, challenge processes that leave a paper trail.

[`insurance-governance`](https://github.com/burning-cost/insurance-governance) produces a ModelValidationReport aligned to SS1/23's principles — discrimination metrics (Gini, Lorenz), calibration testing (Hosmer-Lemeshow with RAG status), stability checks, and assumption documentation. We benchmarked this on 677,991 freMTPL2 policies in [an earlier post](/2026/03/28/model-validation-fremtpl2-real-data-governance/) and found three things that change how teams should set RAG thresholds on real data. The library produces output directly usable in a model sign-off pack or a CRO briefing. The PRA supervisor who asks "how do you know your model is performing as expected" gets a structured answer rather than a collection of ad hoc analyses.

### Ongoing model performance monitoring

Both the FCA (February 2026) and the PRA (January 2026) expect firms to monitor outcomes continuously, not just at deployment. The FCA specifically asks firms to "monitor consumer outcomes closely and mitigate identified risks."

[`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) implements PSI-based drift detection, performance tracking, and monitoring dashboards designed for production pricing models. We published findings on drift detection on freMTPL2 data [here](/2026/03/27/model-drift-detection-fremtpl2-insurance-monitoring/). The honest finding from that work: PSI catches distributional drift in input features before it shows up in loss ratios, which means you can act before the model has already damaged your book. The monitoring obligation from regulators is easy to discharge if you have the infrastructure. Most pricing teams do not have it as a formal, documented process — it exists as informal actuary-watches-the-numbers. That is not sufficient any more.

### Causal inference for spurious correlation

The least obvious but arguably most important tool in this context. A gradient boosting model trained on historical data will pick up correlations that are actuarially spurious — geographic proxies for unmeasured risk factors, temporal autocorrelations in claim data, selection biases from the firm's own underwriting history. Those spurious correlations produce two problems: they create the proxy discrimination risk the FCA is concerned about, and they undermine the "robust historical justification" the PRA expects for model assumptions.

[`insurance-causal`](https://github.com/burning-cost/insurance-causal) implements Double Machine Learning for estimating causal effects in pricing models — separating genuine risk relationships from spurious correlations driven by confounding. We demonstrated this on freMTPL2 bonus-malus data [in an earlier post](/2026/03/28/causal-inference-fremtpl2-bonusmalus-dml/). The application to regulatory risk is direct: if your pricing model's area variable is a causal driver of claims, you can defend it. If it is a proxy for something you cannot observe, a causal analysis will help you identify that before the FCA asks.

---

## What this is not

This is not a prediction that the FCA will take enforcement action against a specific firm for algorithmic pricing discrimination in 2026. We do not know that. What we do know is that three separate regulatory bodies — PRA, FCA, and FRC — have all moved in the same direction in the last eighteen months: naming AI risk explicitly, building the intellectual and evidential framework for oversight, and signalling that model governance expectations have risen.

For a Head of Pricing briefing their CRO, the message is: the regulatory environment for ML pricing models has materially changed since 2023. The ask is not a specific form submission or a new data return. It is demonstrable governance — bias audits with documented findings, validation reports that can withstand a supervisor's challenge, continuous monitoring with clear escalation triggers, and model assumptions that are grounded in historical evidence rather than hoped-for improvements.

None of that is unreasonable. Most of it should have been standard practice already. The regulatory direction of travel means it needs to be on paper now.

---

Related posts:
- [What a Real Fairness Audit Finds: Gender Bias Testing on 67,856 Motor Policies](/2026/03/27/fairness-audit-real-insurance-data-ausprivauto/)
- [What PRA SS1/23 Validation Looks Like on Real Data: 677K French Motor Policies](/2026/03/28/model-validation-fremtpl2-real-data-governance/)
- [Model Drift Detection on freMTPL2: Does PSI Actually Catch Pricing Model Drift?](/2026/03/27/model-drift-detection-fremtpl2-insurance-monitoring/)
- [Causal Inference on freMTPL2: Bonus-Malus and Double Machine Learning](/2026/03/28/causal-inference-fremtpl2-bonusmalus-dml/)
