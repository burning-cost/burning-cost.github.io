---
layout: post
title: "The Mills Review: What the FCA's Long-Term AI Inquiry Means for Pricing Teams"
date: 2026-03-26
categories: [regulation]
tags: [fca, mills-review, consumer-duty, ai, algorithmic-accountability, insurance-governance, insurance-fairness, insurance-monitoring, smcr, model-governance, motor, uk-pricing]
description: "The FCA's Mills Review is not another compliance checklist — it is the regulator thinking aloud about how UK financial services might look in 2030. Pricing teams should read it carefully."
---

On 28 January 2026, Sheldon Mills — the FCA's Executive Director of Consumers and Competition — stood up at the Supercharged Sandbox Showcase event and announced what is now called the Mills Review: a formal inquiry into the long-term impact of AI on retail financial services, with recommendations to the FCA Board expected in summer 2026 and an external publication to follow.

The call for input closed on 24 February. Most pricing teams missed it. That is a mistake they should not repeat, because the Review is not a consultation about current practice — it is the regulator thinking aloud about where the next five years go, and what it will need to do when they get there.

---

## What the Review is actually asking

The Mills Review has four stated themes: how AI technology will evolve (specifically toward agentic systems); how that evolution changes market structure and competition; how consumers will behave differently; and how the regulator itself will need to change. The document that captures this is the Call for Input published on 27 January 2026, which is available on the FCA website alongside Mills' speech text.

Mills was unusually candid about the regulatory problem. He observed that the FCA's existing framework was built on assumptions that modern AI already violates: that models update infrequently, that their behaviour is predictable, and that accountability sits clearly with identifiable humans. None of this is true for a production GBM that retrains on new data every month, incorporates features from third-party data providers, and produces outputs that the pricing team cannot fully trace.

The SM&CR question he raised directly: what do "reasonable steps" mean under the Senior Managers and Certification Regime when your pricing model incorporates components you did not build and updates on a schedule you did not approve? He does not answer this. The Review is intended to generate the answer. But the question itself is pointed enough to be a prompt for action.

---

## Three AI scenarios the FCA is planning around

Mills sketched a progression that is worth understanding, because it frames how the regulator is thinking about the risk surface.

**Assistive AI** is where most firms are now: tools that help consumers understand products, flag risks, or compare options. The regulatory challenge is manageable — firms control the outputs, they are interpretable, Consumer Duty's outcome-testing regime applies.

**Advisory AI** is where some firms are moving: systems that recommend concrete actions, from switching insurer to adjusting cover levels. This is the territory where the FCA is watching whether recommendations are genuinely aligned with consumer interests or whether they optimise for conversion. The conflicts of interest can be subtle.

**Autonomous AI** is the 2030 scenario: agents acting on behalf of consumers within parameters consumers have set. In insurance, this could mean an agent that automatically switches cover at renewal, negotiates terms, or adjusts a telematics-based policy in real time. Mills explicitly flagged that the FCA's current accountability frameworks were not designed for this, and that the Review needs to determine whether they can be adapted or whether new approaches are required.

For pricing teams, the third scenario is the most consequential. If consumers delegate their purchasing to agents, the traditional demand model — where price sensitivity is estimated from observed switching behaviour — breaks. The agent is not sensitive to price the same way a human is. It has its own objective function.

---

## The AI Lab Live Testing programme

Alongside the Mills Review, the FCA has been running a separate but related initiative: AI Live Testing, which started as a proposal under the AI Lab and became operational in 2025.

The feedback statement is FS25/5, published 9 September 2025. The second cohort of applications opened on 19 January 2026 and closed 24 March 2026. Testing starts April 2026. The FCA's technical partner for the programme is Advai, which specialises in adversarial testing of AI systems.

What AI Live Testing actually offers is unusual: firms submit a mature proof-of-concept and work directly with FCA regulatory and technical staff to test it in live market conditions before full deployment. The FCA provides tailored support; Advai tests robustness and bias. In exchange, the FCA gets empirical evidence about how real AI systems behave in production — which feeds back into the Mills Review.

This matters for a pricing team because the programme reveals what the FCA considers important enough to test for. The stated focus is output-driven validation, real-world impact on consumers, and mitigation of risks identified during testing. The FCA is not primarily concerned with whether the model scores a good Gini on a holdout set. It is concerned with what the model does to consumer outcomes at scale.

---

## What this means for Consumer Duty and explainability

The FCA has been explicit that no AI-specific regulation is coming. This was confirmed in December 2025 and restated by Mills in January. The review is about preparing the regulatory posture for an uncertain future, not about introducing new rules now.

But the Mills Review does sharpen a problem that Consumer Duty already creates for pricing teams: the explainability gap.

PRIN 2A — the Consumer Duty framework in force since July 2023 — requires firms to demonstrate that their products provide fair value and that outcomes are monitored across customer groups. The FCA has assessed this in practice: TR24/2, published August 2024, found that most firms' fair value assessments were too high-level and lacked adequate granularity to evidence good outcomes across protected characteristic groups. That is a finding from 18 months ago. The supervisory follow-up from that report has not concluded.

The Mills Review extends the same logic: if an advisory or autonomous AI system produces a consumer outcome that turns out to be poor, who is accountable? What evidence does the firm hold that it understood the system's behaviour before deployment? The existing Consumer Duty framework already requires that evidence. The Review is asking whether the framework is sufficient as systems become more complex, not whether it applies — it already does.

For a pricing model team, the practical translation is: "we cannot explain this GBM" is not a regulatory position you can sustain if the GBM is making material decisions about what consumers pay. That is the case now, not in 2030.

---

## The competition structure risk

The piece of the Mills Review that has received least attention from insurers is the competition analysis. Mills raised the possibility that AI could concentrate market power in the hands of firms or platforms that control consumer interfaces and data. The concern is specific: if an agentic AI system is switching consumers between insurers automatically, the entity that controls that agent controls the distribution channel. The insurer becomes a component supplier rather than a brand.

This is not purely hypothetical. Price comparison websites already intermediate a substantial portion of UK motor insurance. AI agents that go further — monitoring, switching, and negotiating automatically — would intensify that dynamic. For a pricing team, the implication is that the demand-side model of your portfolio may need to account for agent-driven switching behaviour that is qualitatively different from human price sensitivity.

The FCA's CMA coordination on this is explicitly mentioned in the Mills Review documents. Competition concerns about AI-mediated distribution are on the regulatory agenda. They should be on the pricing strategy agenda too.

---

## What a pricing team should do now

The Mills Review will produce recommendations in summer 2026. The external publication will follow. Firms that have done the underlying work before the report lands will be better placed to respond than those that treat it as a future compliance exercise.

Three things are worth doing now.

**Document your accountability chain.** Mills' SM&CR question — who is responsible when the model updates and incorporates uncontrolled components — requires an answer at your firm. Map it: which SMF holder has accountability for the pricing model, what sign-off process exists for retraining runs, and what record exists of each model version in production. [`insurance-governance`](https://github.com/burning-cost/insurance-governance) automates the validation trail and model inventory that sits underneath this, generating auditable HTML reports with run IDs and JSON sidecars that can go into your MRM system.

**Run a proxy discrimination audit on your current model.** The Consumer Duty explainability gap is already open. If the Mills Review recommendations tighten the evidence standard — which is the most likely outcome — firms that can demonstrate they measured and addressed proxy discrimination will be in a different position to those that cannot. [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) implements the Lindholm, Richman, Tsanakas and Wüthrich framework and produces audit-ready documentation. The ONS Census 2021 postcode-to-ethnicity linkage is publicly available; the analysis runs on a standard laptop.

**Put monitoring in place before it is required.** The advisory and autonomous AI scenarios Mills describes require robust monitoring infrastructure to be regulatory defensible. If your firm is experimenting with agentic components — even internally — you need to know what those systems are doing and be able to demonstrate that knowledge. [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) covers drift detection, calibration monitoring, and performance tracking in production.

None of this is speculative preparation. Consumer Duty and SS1/23 already require it. The Mills Review will, at most, raise the documentation standard that the FCA expects when it comes to ask about your AI systems. The firms that are already meeting that standard have less to do when it does.

---

## On what we do not know

The Mills Review recommendations have not been published. The Call for Input closed in February; the FCA Board report is expected in summer; external publication follows. We do not know whether the recommendations will propose new supervisory guidance, new rule amendments, or simply a statement of intent. Mills was careful to say the review "does not change our regulatory approach." It may simply confirm it with more specificity.

What we do know is that the FCA is building an evidence base from AI Live Testing (second cohort starting April 2026, evaluation report expected Q1 2027) and from the Mills Review simultaneously. These two strands will inform each other. The outputs will shape how the FCA supervises AI in retail financial services through the rest of the decade. Being aware of both now — and having the governance infrastructure to respond to whatever emerges — is the sensible position.

---

Primary sources: [Sheldon Mills speech, 28 January 2026](https://www.fca.org.uk/news/speeches/fca-long-term-review-ai-retail-financial-services-designing-unknown) — [Mills Review Call for Input](https://www.fca.org.uk/publications/calls-input/review-long-term-impact-ai-retail-financial-services-mills-review) — [FS25/5: AI Live Testing feedback statement, 9 September 2025](https://www.fca.org.uk/publications/feedback-statements/fs25-5-ai-live-testing)

Related posts:
- [FCA AI Evaluation 2026: What the Regulator Is Looking at](/2026/03/25/fca-ai-evaluation-insurance-pricing/) — what the FCA's current supervision programme is testing for
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) — Consumer Duty proxy discrimination analysis using `insurance-fairness`
- [Your Model Validation Is a Checklist, Not a Test](/2025/07/13/model-validation-pra-ss123/) — SS1/23-compliant validation with `insurance-governance`
