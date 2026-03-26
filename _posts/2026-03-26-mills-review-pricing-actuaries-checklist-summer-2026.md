---
layout: post
title: "The Mills Review: What Pricing Actuaries Must Do Before Summer 2026"
date: 2026-03-26
categories: [regulation, governance]
tags: [fca, mills-review, consumer-duty, smcr, tr24-2, model-governance, algorithmic-accountability, insurance-governance, insurance-fairness, insurance-monitoring, ai, motor, uk-pricing, fair-value]
description: "The FCA's Mills Review Board report lands this summer. Two regulatory exposure points exist right now — before the report publishes. This is what pricing actuaries should do about them."
---

On 27 January 2026, the FCA published a Call for Input launching the Mills Review — a formal inquiry led by Sheldon Mills, Executive Director of Consumers and Competition, into the long-term impact of AI on UK retail financial services. The Board report is expected this summer. External publication follows.

Most pricing teams read that and thought: summer 2026, watch this space. We think that is the wrong response, and this post explains why.

There are two live regulatory exposure points that exist today, before the Review reports. Closing them is the right thing to do regardless of what the Review recommends. If the Review's summer report raises the evidence standard — which is its most likely practical effect — firms that have done the work will be in a structurally different position from those that haven't.

---

## What the Mills Review actually is

The Review is not a consultation on current practice. It is the FCA thinking through what the regulatory framework needs to look like as AI systems move from assistive to agentic. Mills outlined three capability stages in his January speech: assistive AI that explains and compares; advisory AI that recommends switching or adjusting cover; and autonomous AI — his 2030 scenario — where agents act on consumer instructions, moving funds, negotiating renewals, reallocating savings.

Four themes structure the inquiry: how AI technology will evolve toward autonomous and agentic systems; what that does to market structure and competition; how consumer behaviour shifts when decision-making is delegated to agents; and whether the existing regulatory framework is adequate or needs to adapt.

On the last point, Mills was specific about the problem the FCA faces. The frameworks it has were built on three assumptions that AI already violates: that models update occasionally; that their behaviour is predictable; that accountability sits with identified humans. A GBM that retrains monthly, incorporates features from third-party providers, and produces outputs that the pricing team cannot fully trace violates all three.

Mills confirmed the Review will not create new rules before summer. The FCA stated explicitly in December 2025 that no AI-specific regulation is coming. But the Report will define what the FCA expects to see — and that is the point.

---

## Exposure point one: the SM&CR accountability gap

The most pointed question Mills raised in his speech was this: what do "reasonable steps" mean under the Senior Managers and Certification Regime when a pricing model incorporates components the firm did not build, retrains on a schedule the SMF holder did not approve, and shifts behaviour as new data arrives?

He did not answer it. The Review is intended to generate the answer. But the question itself carries the obligation. Under existing SM&CR rules, senior managers cannot delegate accountability to a model or to a vendor. If a firm's AI-related controls are inadequate, the accountable SMF holder — likely the CRO, CTO, or in some firms the Chief Actuary — faces personal exposure.

The gap is specific: no current guidance defines what "reasonable steps" looks like for a pricing model that retrains continuously. The Mills Review's summer recommendations will likely address this. Until they do, any pricing team that cannot answer four questions is sitting with an SM&CR gap:

1. Which named SMF holder has accountability for the current pricing model in production?
2. Who approved the last retraining run, and what sign-off record exists?
3. What components in the model come from third parties, who controls their updates, and has the firm documented that chain?
4. What happened to consumer outcomes by segment between the last two model versions?

A team that cannot produce clear answers to those four questions has the gap under existing rules — before the Review changes anything. The summer recommendations will not create the obligation. They will clarify its scope and raise the evidence standard.

[`insurance-governance`](https://github.com/burning-cost/insurance-governance) automates the model inventory and run audit trail that sits underneath this: every training run generates a run ID, JSON metadata sidecar, and HTML validation report with the sign-off chain. The outputs slot directly into a model risk management system and give an SMF holder an auditable record to point to.

---

## Exposure point two: TR24/2 fair value assessment failures

TR24/2 is the FCA's Thematic Review of general insurance product governance, published August 2024. It covered 28 manufacturers and 39 distributors across 10 product lines. Its findings are relevant to every pricing team because they describe the current enforcement posture — not a future one.

The FCA found that most manufacturers were not adequately assessing and evidencing that their products deliver fair value. Information produced by manufacturers was "often too high level and lacked granularity." Firms could not demonstrate that products provided intended value to customers outside standard segments. The FCA required firms to assess whether those shortcomings applied to their own activities and to remediate promptly, with the explicit warning that intervention — action plans, product withdrawals, or skilled person reviews — would follow continued failure.

That review was published 19 months ago. The FCA's follow-up from TR24/2 has not concluded.

TR24/2 does not mention AI pricing models. But an AI pricing model that lacks segment-level outcome monitoring is precisely the failure TR24/2 describes. The connection Mills makes is direct: the Review will raise the evidence standard for what "adequate" fair value assessment means when AI is involved. Firms with inadequate FVAs now, and AI pricing models now, have a layered gap.

The practical fix is not complicated. Most firms already produce some form of fair value assessment — the problem TR24/2 identified was granularity. Segmenting the monitoring by demographic and geographic groups, and documenting what the model produces across those segments, is the difference between a high-level statement and an evidenced assessment. [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) runs the Lindholm, Richman, Tsanakas and Wüthrich proxy discrimination framework against postcode-linked ONS Census 2021 ethnicity data. The outputs are audit-ready and directly address the FCA's documented concern about what adequate outcome evidence looks like.

---

## The agentic AI risk that belongs on the risk register now

The four Review themes include a competition analysis that pricing teams have largely ignored. Mills raised the possibility that AI agents acting for consumers could concentrate market power in whoever controls the consumer interface. If agentic AI that monitors, switches, and negotiates automatically becomes material, the insurer becomes a component supplier — distribution controlled by the agent operator. Price comparison websites already intermediate a substantial share of UK motor. Agentic AI intensifies that dynamic.

The pricing risk is specific and modellable. A pricing team's GWP model estimates demand elasticity from observed switching behaviour — humans who shop at renewal when the premium increase exceeds a threshold. An agent that switches automatically based on its own objective function has a different sensitivity function. It may optimise for factors the insurer does not observe, switch at smaller price differentials than human behaviour implies, and aggregate switching across a large book simultaneously — creating correlated lapse risk that the demand model does not capture.

Mills' concrete example from the speech was "Sarah," a working parent whose AI agent manages household finances: moves savings automatically, flags uncompetitive renewals, switches current accounts. If even 10 to 20 per cent of the motor book switches via agent rather than human decision, the empirical elasticity estimates built on PCW switching data are no longer valid.

This is not a 2030 problem requiring a 2030 response. It belongs on the model risk register now, with a note that the assumption of human price sensitivity will need to be revisited when agentic switching becomes observable. The FCA's coordination with the CMA on AI-mediated distribution means the scrutiny is coming from two directions simultaneously.

---

## The FCA's bias research note: documenting the governance decision

One piece that connects the Mills Review to current pricing practice is the FCA's December 2024 Research Note on bias in supervised machine learning. It identified three sources of bias — data (historical practices of exclusion, sampling issues), modelling (variable selection, feature engineering), and human interpretation (how practitioners use model outputs) — and confirmed that "technical mitigation strategies may affect model accuracy and could have unintended consequences for model bias on other groups."

That last sentence is the FCA acknowledging the accuracy-fairness Pareto frontier. You cannot simultaneously maximise predictive accuracy and demographic fairness. Mitigating proxy discrimination through feature removal or post-processing degrades the Gini. Where to operate on that frontier is a governance decision, not a technical one.

The implication for pricing teams is that this governance decision needs to be documented. Running the proxy discrimination audit is step one. Deciding what to do about the results — and recording that decision with appropriate sign-off — is the step most teams skip. If the FCA investigates and finds that the pricing team ran a disparate impact test, found elevated impact on a protected-characteristic-correlated segment, and then did nothing because the Gini would fall by 0.4 points, that is a governance failure. If the team documented the tradeoff, escalated it to the SMF holder, and recorded the outcome-based justification for maintaining the feature, that is a defensible position.

---

## Five actions before summer 2026

The Mills Review recommendations land this summer. The FCA AI Live Testing second cohort (applications closed 24 March 2026) starts testing in April; its evaluation report is Q1 2027. Consumer Duty and SM&CR are already in force. Here is what a pricing team should do in the window before the Review publishes.

**1. Map the accountability chain.** Identify the named SMF holder for the pricing model. Document who approves retraining runs, who reviews model version changes, and what the sign-off record looks like. If that chain is unclear or undocumented, the gap exists now. Use [`insurance-governance`](https://github.com/burning-cost/insurance-governance) to automate the run inventory.

**2. Audit third-party data inputs.** The FCA has stated explicitly that firms must gain assurance that third-party data used in pricing does not discriminate against customers with protected characteristics. For each third-party feature — geodemographic scores, credit proxies, telematics data feeds — document what disparate impact testing was done and when. If none has been done, do it.

**3. Run the proxy discrimination audit.** Use the Lindholm/Richman/Tsanakas/Wüthrich framework via [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness). Link postcode features to ONS Census 2021 ethnicity data. Quantify disparate impact by segment. Document what you found and the governance decision made in response. The FCA's December 2024 Research Note confirms the accuracy-fairness tradeoff is real — document that your team understands it and that the decision was made with appropriate oversight.

**4. Fix the fair value assessment granularity.** TR24/2 found that most firms fail this now. High-level FVAs are a current enforcement risk. Segment your fair value monitoring by demographic and geographic groups. If the FCA comes back to TR24/2 follow-up while the Mills Review is in progress, the firms that have already addressed granularity are in a different position from those that haven't.

**5. Deploy production monitoring.** If the model retrains, document that drift detection, calibration monitoring, and disparate impact change tracking are in place — and that someone reviews them. [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) covers this: data drift in inputs, calibration shift in outputs, performance monitoring across segments. The monitoring needs to be both operational and evidenced — the FCA's AI Live Testing programme is testing whether firms can demonstrate control of deployed AI, not just that models produce good holdout metrics.

---

## On timing

The firms that will face scrutiny first when the Mills Review recommendations land are not the ones building the most sophisticated AI systems. They are the ones already identified under TR24/2 as having inadequate fair value assessments — firms the FCA has already looked at and found wanting. TR24/2 was August 2024. The follow-up is overdue.

The summer 2026 window is genuinely a first-mover advantage. The Mills Review recommendations will not create new legal obligations. They will define what "adequate" looks like under SM&CR and Consumer Duty when AI is involved. Firms that can demonstrate they already meet that standard when the FCA comes to ask have a fundamentally different conversation. Firms that treat the summer report as the starting gun will be building governance infrastructure while already under scrutiny.

---

Primary sources: [Sheldon Mills speech, 28 January 2026](https://www.fca.org.uk/news/speeches/fca-long-term-review-ai-retail-financial-services-designing-unknown) — [Mills Review Call for Input, 27 January 2026](https://www.fca.org.uk/publications/calls-input/review-long-term-impact-ai-retail-financial-services-mills-review) — [TR24/2, 21 August 2024](https://www.fca.org.uk/publications/thematic-reviews/tr24-2-general-insurance-pure-protection-product-governance) — [FCA Research Note on ML Bias, December 2024](https://www.fca.org.uk/publications/research-notes/research-note-literature-review-bias-supervised-machine-learning)

Related posts:
- [The Mills Review: What the FCA's Long-Term AI Inquiry Means for Pricing Teams](/2026/03/26/fca-long-term-ai-review/) — the four themes and agentic AI scenarios
- [Consumer Duty Fair Value Evidencing: 12-Step Checklist for Pricing Actuaries](/2026/03/25/consumer-duty-fair-value-evidencing-12-step-checklist-pricing-actuaries/) — TR24/2 compliance in practice
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) — Consumer Duty proxy discrimination analysis using `insurance-fairness`
- [Your Champion Model Is Unchallenged](/2026/03/17/champion-model-unchallenged/) — model governance and sign-off under SS1/23
