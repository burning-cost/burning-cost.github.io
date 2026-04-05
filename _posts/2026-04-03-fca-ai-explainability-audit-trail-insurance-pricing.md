---
layout: post
title: "The FCA's AI Evaluation Is Coming: Do You Have Per-Decision Audit Trails?"
date: 2026-04-03
author: Burning Cost
categories: [compliance, governance, libraries]
description: "Most pricing teams have model documentation. Almost none have per-decision audit trails. The FCA's Q2 2026 AI evaluation will test exactly this gap. Here is what the FCA actually expects, and what insurance-governance already covers."
tags: [FCA, Consumer-Duty, AI-governance, SHAP, explainability, audit-trail, SM&CR, SYSC, PRIN-2A, insurance-governance, python, uk-insurance, model-risk]
---

The FCA confirmed in its 2026 insurance supervisory priorities that it will "evaluate AI deployment across the insurance value chain — including underwriting, claims and consumer services" in Q2 2026. That supervisory assessment is now weeks away.

Most pricing teams we speak to have model documentation. They have a model card, a validation report, a risk committee submission. What they do not have is a per-decision audit trail: a record of which features drove a specific customer's premium on a specific date, what the model predicted, what was charged, and who was responsible.

That gap is the one that matters in a Section 166 skilled person review, a data subject access request, or a Consumer Duty enforcement investigation. "We have good documentation" is not the same as "we can show you what happened to this policyholder."

---

## What the FCA actually expects

The FCA has not introduced AI-specific rules — it confirmed as much in December 2025. But the existing framework creates concrete obligations for AI pricing models:

**PRIN 2A.3 (consumer understanding outcome):** If your model cannot explain a premium at the individual decision level, you have a potential breach. The consumer understanding outcome requires that customers can understand material decisions affecting them. A black-box premium increase of £200 at renewal with no attributable factors is not defensible under this test.

**SM&CR:** SMF24 (Chief Operations) and SMF4 (Chief Risk) carry personal liability for technology risk and model risk respectively. Personal accountability demands that you can reconstruct what a model did, for whom, and when. Immutable audit records that map predictions to named responsible individuals are the evidence base for discharging that accountability.

**SYSC:** AI pricing models are material systems. Appropriate system and controls documentation is not optional.

**Data (Use and Access) Act 2025:** The Act relaxes some GDPR Article 22 restrictions on automated decision-making but introduces rights to human intervention and challenge. Responding to those rights requires per-decision records — you cannot explain a historical premium decision if you did not record what drove it.

None of this requires the FCA to issue new AI-specific rules. The obligations already exist. The Q2 2026 evaluation will test whether firms are meeting them.

---

## The five artefacts the FCA will look for

Based on FCA supervisory communications and market commentary from Browne Jacobson and Buchanan Ingersoll, a defensible posture for a Section 166 review requires five documentation artefacts:

**1. System Design and Architecture Protocol.** Purpose, training data provenance, data minimisation methodology. Updated on major version changes, not annually.

**2. Bias and Fairness Audit Reports.** Statistical evidence of outcomes across demographic segments, produced pre-deployment and at least bi-annually. Must address proxy discrimination — postcode and occupation acting as proxies for ethnicity are the obvious cases in motor and home.

**3. Consumer Duty Impact Assessment.** Formal evaluation evidencing all four Consumer Duty outcomes. Updated annually or on material market shifts.

**4. Performance and Model Drift Logs.** PSI and A/E tracking demonstrating continuous monitoring and showing that recalibration decisions were made on schedule.

**5. Incident Response and Human Override Plan.** A protocol for suspending the model if it is found to be discriminating or miscalibrated, tested annually with evidence of that testing.

Note what is not on that list: "a model development document." Artefacts 1 and 2 overlap with model development documentation, but artefacts 3, 4, and 5 are distinct governance processes that your model development artefacts cannot substitute for.

---

## Where insurance-governance already covers this

[`insurance-governance`](https://github.com/burning-cost/insurance-governance) (v0.3.1, ~10,470 LOC across 32 source files) covers three of the five artefacts directly:

**Artefact 1 — System Design and Architecture Protocol:** `Article13Document` implements EU AI Act Article 13(3) in full, with fields for intended purpose, input specification, output interpretation, training data provenance, and planned change notification. For life and health models within EU AI Act scope, this is the Article 13 compliance document. For motor and property models — which are not high-risk under Annex III and not subject to Articles 12-14 — it serves as the equivalent UK governance document.

**Artefact 2 — Bias and Fairness Audit:** `DiscriminationReport` produces disparate impact tests and renewal cohort A/E analysis by demographic segment. `ModelValidationReport` produces Gini, calibration, and PSI statistics. Together these cover the quantitative evidence requirement.

**Artefact 3 — Consumer Duty Impact Assessment:** `GovernanceReport` generates HTML/JSON packs for model risk committee review. The Consumer Duty outcome mapping — fair value, consumer understanding, products and services, consumer support — is part of the standard output.

Artefacts 4 and 5 are partially covered: `MRMModelCard` has monitoring fields and the risk tier framework, but there is no continuous drift log that links model performance metrics to governance decisions over time.

---

## The gap that v0.3.1 fills: runtime audit trails

The v0.2.0 package covered what you document before deployment. It did not record what happens during deployment — nothing logged what occurred at prediction time for a specific customer: which feature values were present, what the SHAP decomposition looked like, what the model output was, what premium was charged, whether a human reviewed or overrode the output, and who was the responsible SMF.

That gap is closed in v0.3.1. But it is worth understanding why the gap mattered — because firms still running on v0.2.0 have exactly this exposure.

This matters for three specific scenarios:

A **Section 166 reviewer** asks: "Show me the audit trail for the renewals of customers in postcode district X between January and March 2026." Without per-decision records, you have nothing to show. Model documentation tells the reviewer how the model works in general; it tells them nothing about what it did to this cohort.

A **consumer complaint** arrives: "My renewal premium increased by £180 and nobody explained why." Under PRIN 2A.3 and the Data Use Act 2025, you need to be able to reconstruct which factors drove that premium and express them in plain language. "The model increased the premium" is not an explanation.

A **data subject access request** arrives. The right to a meaningful explanation of automated decisions — even with the GDPR Article 22 relaxations under the Data Use Act — requires per-decision records.

`Article13Document.explanation_tools` was a free-text field in the static documentation layer. There was no class that logged what occurred at prediction time for a specific customer — which features were present, what the SHAP decomposition looked like, what the model output was, and who was the responsible SMF. V0.3.1 changes this.

---

## What the audit subpackage covers

The audit subpackage in v0.3.1 has five components:

**`ExplainabilityAuditEntry`** — a dataclass that captures, per prediction: input features, signed SHAP values, model output, final premium, whether a human reviewed it, the reviewer's SM&CR identifier if so, whether an override was applied, and the basis for the decision. Each entry carries a SHA-256 hash computed from its own content, so post-hoc modification is detectable.

**`ExplainabilityAuditLog`** — an append-only JSONL backend. One line per decision. `verify_chain()` recomputes all hashes and reports integrity failures — that is your Section 166 evidence that the log has not been tampered with. `export_period()` extracts a date range for regulatory submission.

**`SHAPExplainer`** — uses `TreeExplainer` for gradient boosting models, `LinearExplainer` for GLM, `KernelExplainer` otherwise. Returns per-prediction dicts mapping feature names to SHAP values. Populates `Article13Document.explanation_tools` with a structured description of the method used. Optional dependency on `shap`; graceful fallback if not installed.

**`PlainLanguageExplainer`** — converts signed SHAP values (model units) to pound-denominated factor attributions:

```
Your motor premium of £487.50 is calculated as follows:
- Base rate: £320.00
- Your 3 years of no-claims discount reduced this by £45.20 (-14.1%)
- Your vehicle group (Category 12) added £32.40 (+10.1%)
- Your postcode area added £18.70 (+5.8%)
- Your 10 years of driving experience reduced this by £23.40 (-7.3%)
- Other factors: +£22.00
Final premium: £487.50 [includes IPT at 12%]
```

This is the PRIN 2A.3 consumer understanding implementation. It is also what the IAIS recommended in 2025: customers should receive a "breakdown of factors influencing premium calculations." The research on SHAP-based explanations for automated decisions (ScienceDirect, 2025) is clear that the compliance gap is not the SHAP computation — it is the communication format. Feature attributions in model units are not explanations. Pounds are.

**`AuditSummaryReport`** — an HTML/JSON summary of the audit log over a period, for model risk committee packs. Includes prediction volume, override rate, SHAP feature importance distribution across the portfolio, and the `verify_chain()` integrity result.

---

## What we are not claiming

A few things worth being honest about.

The FCA's detailed guidance on audit trails and explainability, which it indicated would arrive in late 2026, has not yet been published. We have designed the `ExplainabilityAuditTrail` classes to be adaptable when that guidance appears. We are not hard-coding assumptions about what the FCA will ultimately require.

The EU AI Act Articles 12-14 — which include specific logging and transparency requirements — do not apply to motor or property ML pricing models. They apply to life and health models. The audit trail is good governance and the right response to existing FCA obligations, not an Article 12 legal requirement for motor/property.

SHAP has real limitations. The BIS FSI 2024 paper on AI explainability is direct about this: SHAP values can be inaccurate, unstable, and susceptible to misleading interpretation. We are not claiming that SHAP-based explanations are perfectly faithful to model behaviour. We are claiming they are the industry standard (81% of UK insurance firms use explainability tools, with SHAP the most cited, per the BoE/FCA 2024 AI survey), they are auditable and reproducible, and they are vastly better than no explanation at all.

---

## Starting point

```bash
uv add insurance-governance
```

The audit subpackage shipped in v0.3.1. `ExplainabilityAuditEntry`, `ExplainabilityAuditLog`, `SHAPExplainer`, `PlainLanguageExplainer`, and `AuditSummaryReport` are all available now. The static documentation classes — `Article13Document`, `ModelValidationReport`, `DiscriminationReport`, `MRMModelCard`, `GovernanceReport` — remain in place and cover artefacts 1 through 3.

If your model risk committee is asking whether you can respond to the FCA's Q2 evaluation with something more than a model development document, the answer on the runtime audit side is: v0.3.1 has it. The documentation artefacts have been there since v0.2.0. The per-decision audit trail is the piece that takes the most effort to integrate into a live pricing stack — that is the implementation work ahead, not the library work.

Worked examples on the FréMTPL2 dataset are available in the repository.
