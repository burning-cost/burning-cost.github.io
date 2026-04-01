---
layout: post
title: "EIOPA's AI Governance Opinion: What UK Pricing Teams Need to Do"
date: 2026-04-01
categories: [regulation]
tags: [eiopa, ai-governance, consumer-duty, proxy-discrimination, model-documentation, fairness, solvency-ii, actuarial-function, shap, lime]
seo_title: "EIOPA AI Governance Opinion EIOPA-BoS-25-360 — Implications for UK Insurance Pricing"
description: "EIOPA published its AI Governance Opinion (EIOPA-BoS-25-360) in August 2025. It names the actuarial function as responsible for AI controls, endorses SHAP and LIME explicitly, and sets out a fairness metrics framework that directly contradicts actuarial fairness. Here is what it means for UK pricing teams, including those who think Brexit makes it irrelevant."
---

In August 2025 EIOPA published its Opinion on AI Governance and Risk Management (EIOPA-BoS-25-360). Most UK commentary dismissed it within a week: "addressed to EU supervisors, not firms; Brexit means we can ignore it." That framing is wrong in two ways, and we think it is worth explaining why.

The first reason is that most large UK pricing teams sit inside EU groups — Aviva, AXA, Allianz UK, Zurich UK, RSA/Intact — where the group AI governance framework will be built to the highest common standard. If your EU entity must comply, the framework lands on your desk too. The second reason is that EIOPA's Annex I documentation template is the closest thing any regulator has ever published to a worked example of what model documentation for non-life pricing AI should actually contain. It is worth reading regardless of jurisdiction.

This post works through the opinion in the order a pricing team would actually use it: scope first (do your models even qualify?), then the documentation template, then the fairness problem EIOPA named but did not solve, then the comparison to UK regulation, and finally what to do.

---

## 1. Scope: GLMs are probably out, GBMs are probably in

The opinion operates on a specific definition of "AI system" that follows the Commission AI Office Guidelines. Footnote 11 is explicit: "systems used to improve mathematical optimisation or to accelerate and approximate traditional, well established optimisation methods, such as linear or logistic regression methods, fall outside the scope of the AI system definition."

That is a meaningful boundary. A pure GLM pricing model — frequency × severity with a log-linear structure, Poisson and gamma errors, estimated via IRLS — is almost certainly not an AI system under this definition. Your aggregate pricing curve is not in scope.

GBM, XGBoost, LightGBM, neural networks: these almost certainly qualify. They are not approximations of linear methods. They are distinct algorithmic approaches. If you are running a gradient-boosted model in any part of your pricing stack — risk selection, fraud scoring, telematics, demand elasticity — it is an AI system for the purposes of this opinion.

The opinion also explicitly excludes high-risk AI, which is reserved for the EU AI Act (Articles 9–15). Life and health insurance pricing AI falls under EU AI Act Annex III point 5(c) as high-risk and is not addressed here. Non-life and P&C pricing AI that does not meet the high-risk threshold is squarely in scope of EIOPA-BoS-25-360.

The practical first step is therefore a triage of your model inventory. Separate the GLMs (probably out), the gradient-boosted and neural net models (probably in), and anything genuinely borderline (random forests and similar ensemble methods — apply judgment).

---

## 2. The Annex I documentation template

Section 3 of the opinion (paras 3.22–3.24) requires records sufficient for reproducibility and traceability. For higher-risk AI systems, EIOPA provides Annex I, a nine-field documentation template. We are going to go through it field by field because this is where the opinion earns its keep.

**Field 1: Business objective.** Why does this AI system exist? What decision does it support or automate? This is not a technical question; it is a governance question. If you cannot answer it in two sentences without mentioning "ML pipeline," your documentation is already in trouble.

**Field 2: IT integration.** How does the model connect to production systems? What data flows in, what decisions flow out, what systems does it touch? API connections to other AI systems are explicitly flagged as a security risk in paras 3.34–3.38 — if your GBM is receiving features from another model, document that dependency.

**Field 3: Staff roles.** Who owns the model? Who can change it? What are the escalation procedures? Para 3.30 explicitly names the actuarial function as responsible for AI system controls within its remit — underwriting policy and technical provisions. This is new. It means the actuarial function needs a documented mandate that covers ML model oversight, not just reserve sign-off.

**Field 4: Data collection.** Training and test data provenance, ground truth construction, and bias removal steps. The ground truth construction requirement is significant: if your model targets claims cost, you need to document how you constructed that target variable, including truncation choices, development factors, and any exclusions. "We used paid claims at 12 months" is not sufficient without documenting what that choice implies.

**Field 5: Data preparation.** Variable domain ranges, feature engineering with intention records. What was each feature created to capture? Why did you include postcode sector but not full postcode? This is where proxy discrimination controls get documented — see section 3 below.

**Field 6: Technical choices.** Algorithm selection rationale and, critically, library versions with exact version references. Not "we used LightGBM." "We used LightGBM 4.3.0 with scikit-learn 1.4.2 and SHAP 0.45.1." The reason for specificity is reproducibility: if you need to audit or retrain a model eighteen months later, you need to know exactly what software produced the original results.

**Field 7: Code and hyperparameters.** Including pseudo-random seeds for high-impact applications. If your model uses any stochastic element — random subsampling, dropout, bootstrapped confidence intervals — the seed must be recorded. This is the difference between a result being auditable and a result being approximately auditable.

**Field 8: Model performance.** KPIs with satisfactory performance levels defined in advance, and scheduled retraining frequency. "We monitor the Gini coefficient" is insufficient without specifying what Gini level triggers a retraining decision.

**Field 9: Model security.** Adversarial resilience, data poisoning risk, fall-back plans.

If you are building a new ML model for pricing and you document it against these nine fields, you will have better documentation than the majority of models currently in production at UK insurers. That is not a controversial claim.

---

## 3. The fairness problem EIOPA named but did not solve

Paras 3.12–3.16 cover fairness and ethics. The proxy discrimination text (footnote 13) is precise: "Proxy discrimination arises when sensitive customer characteristics (e.g. ethnicity), whose use is not permitted, are indirectly inferred from other customer characteristics (e.g. location) that are considered legitimate."

EIOPA provides two motor pricing examples from the Test-Achats context. Engine size: permitted, because it has direct actuarial justification independent of gender correlation. Body size or weight correlated with gender: not permitted, because the only plausible mechanism through which it affects motor risk is gender. The test is whether you can construct an actuarial causal chain that does not run through the protected characteristic.

So far, workable. The problem comes in Annex I, where EIOPA sets out a fairness metrics table. The metrics listed are:

- **Demographic parity**: positive outcomes at proportionally equal rates across protected groups
- **Calibration**: equal predictive values across subgroups
- **Equalized odds**: equal true positive and true negative rates across subgroups
- **Equalized opportunities**: equal error rates for the favourable outcome

EIOPA then adds, clearly aware of what it has just done: group fairness metrics "could contradict the concept of actuarial fairness in insurance underwriting."

That is an understatement. Demographic parity in insurance pricing means charging identical premiums to groups with different expected claims costs. That is cross-subsidisation. It is also, in the UK context, what price walking prohibits — but at the level of renewal versus new business, not protected characteristics. Equalized odds in a fraud detection context means flagging the same proportion of claims as fraudulent across ethnic groups regardless of actual fraud rates. If actual fraud rates differ across groups, this is not achievable without accepting worse calibration.

EIOPA does not resolve this tension. It says firms should choose metrics "appropriate to their use case." We think the more honest framing is: for pricing AI, calibration is the relevant metric (does the model produce unbiased cost predictions for each subgroup?), and demographic parity is not, because enforcing it would require intentional mispricing. For claims AI — fraud detection, claims handling — the choice is harder, and the actuarial function needs to make an explicit documented decision about which metric it is optimising and why.

The failure to solve this is not a criticism of EIOPA. It is genuinely an unsolved problem. But firms need to notice that the metrics table does not come with instructions.

---

## 4. How this compares to UK regulation

**FCA Consumer Duty (PS22/3, effective July 2023).** Consumer Duty requires good outcomes across four areas: products and services, price and value, consumer understanding, and consumer support. EIOPA's fairness and ethics requirements are largely convergent with Consumer Duty in outcome — both want customer-centric AI, explainability on request, and bias controls. EIOPA is more specific on technical implementation (SHAP/LIME, fairness metrics). Consumer Duty is more specific on pricing fairness for vulnerable customers. Meeting both simultaneously is not significantly more work than meeting either one well.

**PRA SS1/23 (Model Risk Management Principles).** SS1/23 formally applies to banks, not insurers. There is a direction of travel toward insurer extension, but no confirmed date as of April 2026. Where SS1/23 is notably stronger than EIOPA is independent model validation: SS1/23 requires ongoing and event-driven revalidation by a function separate from model development. EIOPA's opinion is vaguer on validation independence. Where EIOPA is stronger is AI fairness — SS1/23 has no equivalent proxy discrimination framework or fairness metrics table.

**Equality Act 2010.** The Equality Act prohibits both direct and indirect discrimination on protected characteristics. EIOPA's proxy discrimination framework is addressing the same underlying risk through a model governance lens. The enforcement routes differ: the Equality Act gives customers a right of action; EIOPA gives supervisors an expectation to enforce via NCAs. For a UK pricing team, the Equality Act risk is the more immediately material one — it is live, it gives customers standing to sue, and there is no gap between protected characteristics under the Act and the proxy discrimination concept in EIOPA footnote 13.

**EU AI Act (covered in prior posts).** The EIOPA opinion explicitly excludes high-risk AI to avoid double regulation with the EU AI Act. A UK non-life insurer with no EU operations faces no EU AI Act obligations for its pricing models (non-life pricing is not in Annex III). It faces EIOPA-BoS-25-360 if it operates through an EU entity. It faces FCA Consumer Duty and the Equality Act regardless.

---

## 5. What is genuinely new

Before we get to the action list, it is worth being clear about what EIOPA-BoS-25-360 actually adds beyond repackaging existing requirements.

The following are new in the sense that no prior EIOPA primary document contained them:

- Explicit endorsement of LIME and SHAP as acceptable explainability tools for insurance, with the qualification that assumptions and limitations must be documented
- The three-audience explainability framework: regulators/auditors get global model explanation; customers get simple language explanation of material decisions on request; intermediaries must be notified when AI drives a decision affecting their obligations — these are three distinct requirements, not one
- The fairness metrics table with the actuarial fairness tension warning, which at minimum forces firms to make a documented choice between metrics rather than ignoring the question
- Pseudo-random seed recording as an explicit requirement for high-impact models
- Exact library version references in technical documentation
- Ground truth construction as a required documentation field
- Explicit naming of the actuarial function as responsible for AI controls within its remit (para 3.30)

The documentation requirements in paras 3.22–3.24 and Annex I are not new in the sense that Solvency II Article 258 has always required adequate records. What is new is the specificity: a named template with nine fields, specific sub-requirements within each field, and explicit reproducibility and traceability as the test.

---

## 6. Seven things a UK pricing team should do now

**1. Triage your model inventory against the AI system definition.** GLMs and logistic regression: probably out. GBMs, XGBoost, LightGBM, neural nets: probably in. Random forests and ensemble methods: apply judgment. Record your decisions.

**2. Score each in-scope AI system for impact.** EIOPA's proportionality logic means governance requirements scale with impact. Assessment criteria: large-scale data processing, use of sensitive data, number of customers affected (including vulnerable), level of automation, consumer-facing vs internal, whether the line is compulsory. Higher scores need the full Annex I treatment.

**3. Implement Annex I record-keeping for your highest-scoring models.** All nine fields. Start with field 6 (library versions) and field 7 (pseudo-random seeds) because these are the most frequently missing in existing documentation and the most annoying to reconstruct retrospectively.

**4. Document your proxy discrimination controls explicitly.** For each pricing feature in an in-scope model, record: what the feature is intended to capture; what known protected characteristic correlations exist; and why the actuarial causal chain for the feature is independent of those characteristics. "We reviewed features for discrimination risk" is not documentation. "Engine size is included as a proxy for vehicle performance and driver behaviour; we reviewed its correlation with gender and found r=0.18; we conclude the actuarial mechanism is independent of gender because [reason]" is.

**5. Adopt the three-audience explainability framework.** For your highest-impact pricing models: (a) produce a global model explanation (SHAP summary plots with documented assumptions) for internal and regulator use; (b) define what a customer-facing explanation looks like for a material pricing decision, in plain English; (c) define what intermediaries need to know when the model drives a decision that affects their obligations.

**6. Give the actuarial function a formal documented mandate for ML model oversight.** Para 3.30 names it; most UK actuarial function charters do not yet cover it explicitly. This is a governance gap that the FCA or PRA could probe under Consumer Duty or Solvency II equivalence assessments.

**7. Review third-party vendor contracts.** Para 3.11 is unambiguous: ultimate responsibility rests with the insurer regardless of who built the model. Where IP prevents full explainability — which is common with vendor telematics scoring platforms and some PCW demand models — you need contractual clauses guaranteeing audit access, external testing rights, and bias assessment obligations. Check your Guidewire and Duck Creek contracts while you are at it.

---

EIOPA will review supervisory convergence across member states in August 2027 — two years after publication. Between now and then, NCAs will be forming views on what "good" looks like for AI governance in insurance. The Annex I template is the clearest signal yet of what that means in practice.

UK teams in EU groups are already living in this framework. Pure UK domestic teams have a window to implement it on their own terms rather than reactively. We think Annex I is the right place to start.

---

*EIOPA-BoS-25-360 was published 6 August 2025. The companion impact assessment is EIOPA-BoS-25-363. EIOPA signals further guidance on specific use cases at para 4.3 — non-life pricing is a candidate for dedicated treatment.*
