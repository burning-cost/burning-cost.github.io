---
layout: post
title: "Building a regulatory-grade AI pricing model: what the FCA's Live Testing cohort actually requires"
date: 2026-04-05
categories: [regulation, governance, libraries]
tags: [FCA, AI-Live-Testing, consumer-duty, insurance-governance, insurance-fairness, insurance-conformal, uncertainty-quantification, explainability, bias-detection, outcome-monitoring, mills-review, model-risk, UK-insurance, pricing, ai, SMF, PRIN-2A, PS22-9, FCA-CP6-24]
description: "FCA AI Live Testing cohort 2 opened in January 2026. Testing is live from April. Firms in scope for AI underwriting and pricing need to demonstrate four things: outcome monitoring, bias detection, uncertainty quantification, and explainability. Here is what that requires technically and what the Mills Review wildcard changes about the timeline."
author: Burning Cost
---

The FCA's AI Live Testing initiative began its second cohort in January 2026. Testing runs from April 2026, with an evaluation report due at year-end. The FCA has explicitly named AI in underwriting and pricing as within scope.

This is not hypothetical. Cohort 2 is live now.

What follows is a technical account of what firms inside the sandbox — and firms watching from outside — actually need to demonstrate. Not what the regulator says in press releases. What you need to have built and documented.

---

## What the Live Testing programme is

The AI Live Testing sandbox is a supervised environment in which the FCA works directly with firms on live AI deployments. It sits alongside the existing regulatory sandbox and the Digital Sandbox. The distinction matters: this is not pre-commercial testing. Firms bring live AI systems. The FCA observes, asks questions, and — for cohort 2 — focuses specifically on how firms' AI models interact with consumers in underwriting, claims, and pricing.

Applications for cohort 2 closed in early 2026. Firms that got in are now running. The evaluation report the FCA publishes at year-end will, in effect, set the supervisory expectations for every firm in the market that did not participate. That is the mechanism: sandbox results become thematic guidance, thematic guidance becomes supervisory expectation, supervisory expectation becomes enforcement risk.

The FCA has been clear that no new AI-specific rulebook is coming. The framework is Consumer Duty (PRIN 2A), the Equality Act 2010, FCA SYSC, and — for model governance — the principles emerging from CP6/24 (the PRA's model risk consultation, which applies to insurers under equivalent proportionality). The sandbox is how the FCA tests whether existing rules are sufficient or need supplementing. Which means firms in the sandbox are not just demonstrating compliance with current rules. They are shaping what the next set of rules will be.

---

## The four things you need to demonstrate

Four requirements have emerged clearly from FCA communications, Treasury Select Committee questioning, and what participating firms in cohort 1 have described:

### 1. Outcome monitoring

The FCA's Consumer Duty requires firms to demonstrate that AI produces consumer-duty-aligned outcomes — not just that the model is technically sound. This distinction is doing real work. A model can have excellent lift, calibrated A/E ratios, and a clean validation report, and still fail outcome monitoring if you cannot show what outcomes different consumer groups are actually receiving.

Outcome monitoring for a pricing model means, at minimum:

- Segmented claims experience by consumer group, updated at a frequency that would catch drift before it becomes systemic. Quarterly is the FCA's implicit expectation based on the outcome testing framework published in 2024.
- Loss ratio and claims frequency tracked separately for customers who renewed, churned, and are new-to-book — because the Consumer Duty outcome is different for each.
- A defined escalation process: who sees the monitoring output, who decides whether a threshold breach is material, and under which SMCR function that accountability sits.

The last point matters in Live Testing. The FCA will ask to see the escalation trail, not just the monitoring dashboard.

### 2. Bias detection

FCA Director Sheldon Mills stated in February 2026 that the FCA has not yet found systemic bias in pricing AI. That statement is more careful than it sounds. It means the FCA's current investigations have not produced a finding of systemic bias. It does not mean the FCA is satisfied that bias does not exist. The Live Testing cohort is partly designed to determine whether current methods for detecting bias in pricing AI are adequate.

What this means for a participating firm:

You cannot produce a single demographic parity ratio and call it done. The FCA's own review of Consumer Duty implementation (TR24/2, August 2024) found that most insurers' fair value assessments were "high-level summaries with little substance." The bias detection evidence needs to be:

- **Exposure-weighted**: unweighted metrics misrepresent where the portfolio actually sits.
- **Segmented, not aggregate**: a model can pass aggregate calibration while systematically mispricing a protected sub-group. The FCA's Consumer Duty supervisory work, including the December 2025 Research Note on motor insurance pricing and local area ethnicity, specifically concerns proxy discrimination — where a rating factor that does not name a protected characteristic is nonetheless correlated with one. EP25/2 (July 2025) evaluates the GIPP price-walking remedies and is a separate document.
- **Proxy-tested**: postcode is the canonical example. Citizens Advice estimated a £280 per year ethnicity penalty in UK motor, totalling £213 million annually, running through postcode and similar proxy factors. If your model uses postcode, you need proxy correlation evidence and an Equality Act Section 19 proportionality argument for why the factor is retained.
- **Version-tracked**: the test needs to be linked to the model version, run at deployment, and repeated on a documented schedule. Not a one-off report run when someone asks for it.

The `insurance-fairness` library covers the proxy detection problem directly. Its CatBoost proxy R² approach returns R² ≈ 0.62 for postcode-to-ethnicity correlation on UK data where Spearman correlation returns |r| ≈ 0.10 and finds nothing. The difference is not marginal — it is the difference between a test that catches the problem and one that does not.

### 3. Uncertainty quantification

The FCA's guidance on AI — due to be formalised by end 2026, in part informed by the Live Testing evaluation — is expected to require that AI-driven pricing decisions come with human-in-the-loop protocols where the model's uncertainty is high. That requires the uncertainty to be quantified in the first place.

Most pricing models produce point estimates. A GLM or gradient boosting model tells you the expected frequency or severity. It does not tell you how confident it is, how that confidence varies across the risk distribution, or whether the estimate is within its reliable operating range for a given risk.

Parametric prediction intervals from a Tweedie GLM assume the variance scales as mu^p across the book. That assumption breaks exactly where the stakes are highest: unusual, high-value risks where the tail heteroscedasticity is largest. The aggregate coverage might look fine; the per-decile coverage for the top risk decile will not.

Conformal prediction fixes this without parametric assumptions. The guarantee is P(y in interval) ≥ 1 - α for any data distribution, as long as calibration and test data are exchangeable. The `insurance-conformal` library implements the correct nonconformity score for insurance data — |y - ŷ| / ŷ^(p/2), which normalises by the Tweedie standard deviation rather than treating a £1 error on a £100 risk identically to a £1 error on a £10,000 risk. On a heterogeneous UK motor portfolio, this produces intervals that are 13% narrower than parametric Tweedie while maintaining the coverage guarantee.

For a firm in Live Testing, uncertainty quantification needs to produce something operational: a flag on pricing decisions where the model is operating in a low-confidence region, a defined protocol for what happens to those decisions (manual review, conservative loading, escalation), and a log showing the protocol was followed.

### 4. Explainability

The Treasury Select Committee raised AI transparency in credit and insurance as a concern in 2025, and the FCA is now required to address it. The question is: can a customer who receives an adverse pricing outcome understand why?

This is not the same as model interpretability for internal validation. SHAP values on a gradient boosting model satisfy a technical audience. They do not satisfy a consumer who has been declined or heavily loaded, and they do not satisfy an FCA reviewer who wants to see that the firm could have explained the decision to the consumer in terms they could understand and challenge.

Explainability in the Live Testing context means:

- **Per-decision audit trails**: not a general model explanation, but a record of which features drove this specific decision for this specific consumer, retainable for however long the FCA might look back (the motor finance scheme's seventeen-year retrospective is a useful reference point).
- **Consumer-accessible explanations**: not SHAP values, but the narrative version of what drove the premium. "Your vehicle is in a higher-risk group" is an explanation. A feature importance chart is not.
- **Counterfactual recourse**: if a consumer is declined or loaded, can you tell them what would need to change for the outcome to be different? The FCA's algorithmic recourse work (Consumer Duty Outcome 4) makes this expectation explicit.

The `insurance-governance` library's `ExplainabilityAuditLog` class handles per-decision SHAP logging to a structured store, with SMF-tagged accountability mapping at each decision point. The explainability audit trail post from April 2026 walks through the implementation in detail if you need it.

---

## Building the stack in practice

For a firm entering Live Testing — or preparing for the supervisory expectations that will follow from it — the technical checklist is:

**Outcome monitoring**
- [ ] Segmented A/E ratios by consumer group, at policy level, updated quarterly
- [ ] Loss ratio tracking split by renewal / new business / lapse cohort
- [ ] Named SMF accountable for monitoring output and escalation decisions
- [ ] Monitoring logs retained with timestamps and reviewer sign-offs

**Bias detection**
- [ ] Proxy correlation tests at deployment, version-tagged (CatBoost R², mutual information — not just Spearman)
- [ ] Exposure-weighted demographic parity ratios and A/E by group
- [ ] Section 19 Equality Act proportionality argument documented for each rating factor with proxy correlation above threshold
- [ ] Scheduled re-audit frequency documented and adhered to

**Uncertainty quantification**
- [ ] Conformal prediction intervals replacing or supplementing parametric intervals
- [ ] Per-decile coverage verification, not just aggregate
- [ ] Operating boundary defined: what confidence threshold triggers manual review?
- [ ] Log of decisions flagged as high-uncertainty, and what happened to them

**Explainability**
- [ ] Per-decision SHAP values logged to structured store at point of decision
- [ ] Consumer-facing explanation template tested against plain English criteria
- [ ] Counterfactual recourse available for declined/loaded decisions
- [ ] Audit trail retained for regulatory look-back period

The `insurance-governance`, `insurance-fairness`, and `insurance-conformal` libraries address the technical implementation of the second, third, and fourth pillars respectively. They are not magic — you still need to design the escalation process, write the consumer explanations, and get the SMF accountability mapping signed off by someone with an actual SMF designation. But the measurement infrastructure underneath those decisions needs to be automated and documented, not built case-by-case when the FCA asks.

---

## The Mills Review wildcard

Recommendations from the Mills Review — the FCA's formal inquiry into the long-term impact of AI on retail financial services — are due in summer 2026 and will be published externally. They will set supervisory expectations for years.

We do not know what the recommendations will say. What we do know is:

- Firms in Live Testing cohort 2 will have shaped them. The FCA will use what it observed in the sandbox as evidence for what is achievable. If cohort 2 firms demonstrate working uncertainty quantification and per-decision audit trails, the Mills Review will cite them as proof of concept and set them as expectations for the rest of the market.
- The timeline is tight. Summer 2026 recommendations plus the cohort 2 evaluation report at year-end means that by early 2027, any firm without the four pillars in place will be visibly out of step with a published supervisory framework.
- CP6/24 (the PRA's insurance model risk consultation) is also live. The interaction between the Mills Review recommendations and CP6/24 final rules will determine exactly which AI governance requirements are formal regulatory obligations versus supervisory expectations. The difference matters for enforcement, not for what you need to build.

The right response to that uncertainty is not to wait. The four requirements described above are not experimental. They are the minimum for a defensible position under current Consumer Duty obligations, regardless of what the Mills Review concludes. Building them now means you are ready for whatever framework lands in 2027, and you have the documentation history that shows you were operating appropriately during the 2025–2026 implementation period.

---

## Practical next steps

If you are inside cohort 2: the checklist above is what the FCA is evaluating. Each item needs to be implemented, documented, and linked to a named accountable individual. The evaluation report will distinguish firms that have the infrastructure from those that have the assertions.

If you are outside cohort 2: the evaluation report at year-end is the document to watch. It will tell you what the FCA found, what it expects, and where firms fell short. The gap between "firms in the sandbox built it" and "firms outside the sandbox have not" is not a gap you want to be on the wrong side of when the Mills Review recommendations land.

Either way, the technical work is the same. The four pillars need to be in place, automated, logged, and accountable. None of it is novel — the methods exist, the regulatory obligations that underpin them are clear, and the question is whether the implementation is in production or still on a planning document.

The FCA's direction of travel since 2021 has been consistent: it wants firms to demonstrate outcomes, not just processes. AI in pricing does not change that expectation. It raises the stakes, because algorithmic decisions at scale are harder to audit after the fact than manual decisions at the time. The point of the Live Testing programme is to establish, before the fact, what adequate looks like.

---

## References

- FCA AI Live Testing: [fca.org.uk/firms/innovation/ai-live-testing](https://www.fca.org.uk/firms/innovation/ai-live-testing)
- FCA TR24/2: Multi-firm review of Consumer Duty implementation in product governance, August 2024
- FCA EP25/2: Evaluation of General Insurance Pricing Practices remedies, July 2025
- FCA Consumer Duty (PRIN 2A), effective 31 July 2023
- PRA CP6/24: Model risk management principles for insurers (consultation)
- Equality Act 2010, Section 19 (indirect discrimination)
- Mills Review: FCA inquiry into long-term AI impact on retail financial services, recommendations expected summer 2026

Related on Burning Cost:

- [The FCA's AI Evaluation Is Coming: Do You Have Per-Decision Audit Trails?](/2026/04/03/fca-ai-explainability-audit-trail-insurance-pricing/) — per-decision explainability implementation
- [FCA AI Evaluation 2026: What the Regulator Is Looking at](/2026/03/25/fca-ai-evaluation-insurance-pricing/) — the broader evaluation context
- [The Mills Review: What the FCA's Long-Term AI Inquiry Means for Pricing Teams](/2026/03/26/fca-long-term-ai-review/) — summer 2026 recommendations
- [Motor Finance Redress: What PS26/3 Tells Insurance Pricing Teams About Their Own Exposure](/2026/04/04/motor-finance-redress-ps263-insurance-pricing-governance/) — the retrospective enforcement precedent
- [Mean Parity Is Not Tail Parity](/2026/04/04/demographic-parity-tails-regression-insurance-fairness/) — why aggregate fairness checks miss the exposure
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — uncertainty quantification foundation
