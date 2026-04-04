---
layout: post
title: "EU AI Act update: motor pricing probably isn't high-risk (and your GLMs may not be AI)"
date: 2026-03-28
categories: [regulation]
tags: [eu-ai-act, regulation, model-governance, mrm, compliance, annex-iii, high-risk-ai, glm, gbm, motor-pricing, life-insurance, eiopa, extraterritoriality]
description: "We published a post three weeks ago treating all insurance pricing models as potentially high-risk under the EU AI Act. After reading the primary sources, we need to correct three significant assumptions: motor pricing is not captured by Annex III, traditional GLMs may be outside the AI system definition entirely, and conformity assessment is self-assessment only."
---

We published a post three weeks ago arguing that pricing teams needed to treat their models as high-risk AI under the EU AI Act. The framing was broadly correct for life and health insurers. It was wrong, or at least substantially overstated, for the majority of UK general insurance pricing teams. After working through the primary sources — the Act itself, the European Commission's February 2025 Guidelines on the AI system definition, and the EIOPA Final Opinion published in August 2025 — we need to correct three assumptions.

This is not a minor clarification. The difference between "your motor GBM is high-risk AI" and "the EU AI Act probably does not apply to your motor GBM at all" is, in compliance terms, substantial.

---

## Correction 1: Motor and property pricing is not high-risk AI

The original post cited Annex III of the Act as catching general insurance pricing under the "essential services" category. That was wrong. Read the primary text.

Annex III, paragraph 5(c) classifies as high-risk: *"AI systems intended to be used for risk assessment and pricing in relation to natural persons in the case of life and health insurance."*

That is the complete operative text. Life and health insurance. Not motor. Not property. Not liability, commercial lines, or household. The phrase "life and health insurance" is not illustrative — it is the scope limiter. Motor pricing models are not captured by Annex III paragraph 5 of the EU AI Act.

There is no catch-all for general insurance in paragraph 5. The Act's drafters made a specific choice to draw the line at life and health, where AI-driven pricing decisions have the most severe and least reversible effects on individuals — denial of life cover, unaffordable health premiums, refusal of critical illness policies. The same logic does not hold for motor third-party liability or household contents, where the fundamental rights analysis is materially weaker.

Paragraph 5(b), which we cited, covers AI used to evaluate creditworthiness or establish credit scores. This captures credit reference AI, not insurance pricing AI — these are distinct systems with distinct regulatory treatment, even if actuaries sometimes use similar methods.

**The practical consequence:** A UK motor insurer with no EU policyholders and no EU subsidiaries is not subject to the EU AI Act high-risk obligations at all. A UK life insurer pricing individual protection policies for UK customers is not directly subject to the Act as domestic law — but see Correction 3 on extraterritoriality.

---

## Correction 2: Traditional GLMs may be outside the AI system definition entirely

The original post treated GLMs and GBMs as equivalent for compliance purposes. They are not. The distinction matters because the entire EU AI Act applies only to *AI systems* as defined in Article 3(1). If a model is not an AI system under that definition, no obligations apply — not data governance, not technical documentation, not conformity assessment.

The European Commission published non-binding Guidelines on the definition of AI systems on 6 February 2025. Paragraph 42 states explicitly: *"systems used to improve mathematical optimisation or to accelerate and approximate traditional, well established optimisation methods, such as linear or logistic regression methods, fall outside the scope of the AI system definition."*

EIOPA cited this passage directly in the Final Opinion (EIOPA-BoS-25-360, footnote 11), giving it regulatory acknowledgement at the European insurance supervisory level.

The Commission's reasoning follows the structure of Article 3(1): an AI system must infer from input data how to generate outputs "with varying levels of autonomy." A traditional GLM with a pre-specified link function and fixed structural form does not infer its own structure from data. The actuary specifies the distribution family, the link function, and the features. The algorithm solves a well-understood optimisation problem. This is not "autonomy" in any meaningful sense.

Working through the implications:

**Likely outside the AI system definition:**
- Traditional GLMs (Tweedie, Gamma, Poisson) with pre-specified link functions
- Logistic regression frequency models where the structural form is fixed in advance
- Additive models with structure specified before fitting (GAMs with pre-specified smoothing basis and fixed knot positions)
- Credibility models with fixed Bayesian update rules

**Likely inside the AI system definition:**
- Gradient boosted trees (XGBoost, LightGBM, CatBoost) — these infer complex feature interactions autonomously from data; the structure is not pre-specified
- Neural networks of any architecture
- Random forests
- Stacked ensembles containing any of the above
- AutoML pipelines that select models or features from the data

**Grey area requiring assessment:**
- GAMs with automated knot/basis selection (e.g. mgcv's automatic smoothing via restricted maximum likelihood) — the smoothing structure is inferred, not specified
- GLMs with automated variable selection (LASSO, elastic net, stepwise) — the model structure is learned
- Regularised regression where the penalty and feature set emerge from cross-validation

The Commission's guidance is non-binding. The ultimate legal test is whether the system infers structure from data with varying levels of autonomy. A LASSO GLM is arguably closer to a GBM than to a traditional GLM on that test. Legal certainty will not arrive until case law develops.

**The practical consequence for UK pricing teams:** Most UK motor pricing functions are running GBMs on top of GLM-equivalent tariffs. The GBMs infer structure; they are likely AI systems. But because motor pricing is not high-risk under Annex III (see Correction 1), this analysis is moot for motor pricing if you have no EU exposure. For life and health pricing teams using pure traditional GLMs with no automated structure selection, a documented assessment that the model falls outside the AI system definition may be defensible — and if it holds, removes all Act obligations for that model.

---

## Correction 3: Conformity assessment is self-assessment, not third-party audit

The original post implied that compliance would require external audit or notified body certification. For insurance pricing, this is wrong.

Article 43 of the Act sets out conformity assessment procedures. High-risk AI systems listed in Annex III, Section 5 (which is where life/health pricing sits) are assessed under Annex VI — internal control. This means the provider conducts and documents the conformity assessment themselves. No third-party notified body is involved. This is considerably less onerous than many feared when the Act was first published.

Third-party audit under Annex VII applies only to Annex III systems that are also subject to EU harmonised product legislation (the machinery directive, medical devices regulation, etc.). Insurance pricing models are not covered by any such harmonised legislation. The self-assessment route applies.

Self-assessment still requires rigorous documentation — technical documentation per Annex IV, an EU declaration of conformity per Article 47, and registration in the EU database under Article 71. But the person signing off is the provider's own authorised representative, not an external auditor.

---

## Is your model in scope? A decision sequence

The following sequence applies to UK pricing teams as of 28 March 2026.

**Step 1: Is the system an AI system?**

If the model is a traditional GLM with a pre-specified structural form and no automated structure selection: documented self-assessment that it falls outside the Article 3(1) definition may be defensible, citing Commission Guidelines para 42. If it is a GBM, neural network, or ensemble: it is an AI system.

**Step 2: Does the EU AI Act apply to you?**

The Act applies to: (a) providers placing AI systems on the EU market; (b) providers or deployers in third countries where the AI output is *used* in the EU. If you are a purely domestic UK insurer with no EU-domiciled policyholders, no EU subsidiaries, and no Lloyd's European operations, the Act does not apply to you as a matter of direct obligation. FCA and PRA stated in April 2024 they will not introduce equivalent domestic AI-specific rules in the near term — existing model governance expectations apply instead (SS1/23 for banks; Solvency II Articles 120–126 and PRA CP6/24 for insurers).

If you have EU-domiciled policyholders, an EEA subsidiary, Lloyd's Brussels operations, or cross-border service arrangements, the Act applies to the pricing systems used for those customers.

**Step 3: Is the system high-risk?**

High-risk means: the system is an AI system (Step 1), and it is used for risk assessment and pricing of *life or health insurance* at the *individual* level. Motor, property, liability, and commercial lines are not high-risk under Annex III. Portfolio-level or book-level pricing (not individual natural persons) is not captured.

If yes to both: Articles 9–15 apply in full. Self-assessment conformity assessment under Annex VI.

If no: the system is not high-risk AI. No Articles 9–15 obligations. The EIOPA governance opinion still applies (data quality, documentation, human oversight, explainability) — but these are Solvency II-based expectations, not AI Act obligations, and most already map onto good model governance practice — whether codified as SS1/23 (for banks) or Solvency II Articles 120–126 (for insurers).

---

## What the EIOPA opinion actually says — and where it goes further than the Act

EIOPA-BoS-25-360, published August 2025, is sometimes cited as if it reinforces the EU AI Act's scope over general insurance. It does not. The opinion states explicitly that it applies to AI systems in insurance that are *not* high-risk under the Act — it is the governance framework for lower-risk AI use cases. The high-risk obligations (Articles 9–15) sit separately.

Where the EIOPA opinion is worth reading carefully:

The proxy discrimination treatment (footnote 13) is specific and useful: *"Proxy discrimination arises when sensitive customer characteristics (e.g. ethnicity), whose use is not permitted, are indirectly inferred from other customer characteristics (e.g. location) that are considered legitimate."* EIOPA cites the Commission's Test-Achats guidelines in the same footnote — price differentiation by car engine size is permitted; differentiation by body size (correlated with gender) is not. This is relevant to any UK pricing team using postcode or driving behaviour telemetry.

The explainability section (paras 3.25–3.28) explicitly endorses SHAP and LIME as tools, while requiring that their assumptions and limitations be documented. This aligns directly with the validation expectations in PRA SS1/23.

The governance opinion is more conservative than the Act in one important respect: it encourages insurers to apply its principles *regardless of whether the model technically qualifies as an AI system* under the Article 3(1) definition. The FCA/PRA direction of travel points the same way. Building governance to the EIOPA standard is good practice for UK pricing functions irrespective of EU AI Act scope questions.

---

## The UK regulatory gap

The EU AI Act does not apply as domestic UK law. The FCA and PRA have said they will not replicate it. But the extraterritorial reach of Article 2 is real: any UK insurer pricing EU-domiciled policyholders is a provider subject to the Act for those systems. Lloyd's managing agents with Brussels-written business and UK parents of EEA insurance subsidiaries are in scope.

For purely domestic UK firms, the relevant governance framework is: for insurers, Solvency II Articles 120–126 and PRA CP6/24; for banks and investment firms, SS1/23. All firms are subject to the SMCR (accountable SMF) and FCA Principles — particularly consumer outcomes. The obligations these create are substantively similar to EU AI Act Articles 9–12. Firms building governance to the EU AI Act standard are simultaneously satisfying UK expectations; the converse is largely true as well.

The UK AI Regulation Bill (re-introduced to the House of Lords in March 2025) would create an AI Authority and codify governance principles into binding duties. As of March 2026 it remains a Private Member's Bill without government backing. We would not plan compliance architecture around it.

---

## What we got right, what we got wrong

The original post was correct that:
- The EU AI Act compliance deadline of 2 August 2026 is real and the penalties (up to €35 million or 7% of global revenue) are not theoretical
- Life and health pricing teams are genuinely in scope and the compliance gap is structural, not just documentation
- The Digital Omnibus extension proposal is speculative and should not be planned around
- The governance requirements (risk management, technical documentation, human oversight, monitoring) represent good practice regardless of legal obligation

The original post was wrong that:
- General insurance pricing (motor, property, liability) is captured by Annex III paragraph 5
- GLMs and GBMs face equivalent compliance obligations under the Act
- Conformity assessment would require third-party notified body review

The error in the original post came from relying on secondary analysis rather than reading Annex III directly. We should have done that first.

---

If you are a life or health pricing actuary using GBMs or neural networks for individual-level pricing, and you have EU policyholders or operate under an EEA structure: the August 2026 deadline applies to you and the requirements are substantial. The original post's practical guidance on Articles 9–15 remains accurate for your situation.

If you run motor or property pricing models: the EU AI Act high-risk obligations almost certainly do not apply to your work. Your governance obligations flow from Solvency II Articles 120–126 (insurers) or SS1/23 (banks), together with FCA Principles — not from the EU AI Act. That is not a reason to do less — it is a reason to be clear about *why* you are doing it.
