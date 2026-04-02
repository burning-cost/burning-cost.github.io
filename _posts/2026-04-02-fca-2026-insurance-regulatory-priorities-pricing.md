---
layout: post
title: "FCA 2026 Insurance Regulatory Priorities: What Pricing Teams Need to Know"
date: 2026-04-02
categories: [regulation]
tags: [fca, consumer-duty, pricing, pet-insurance, pmi, ai-underwriting, renewal-pricing, social-renters, fairness, actuarial]
description: "The FCA published its first consolidated insurance priorities report on 24 February 2026. It replaces individual portfolio letters and signals where regulatory action is coming in pricing. Here is what matters."
author: burning-cost
---

On 24 February 2026, the FCA published its first consolidated annual priorities report for insurance — a single document replacing the portfolio letters it previously sent to different insurance sub-sectors. The [full report is here](https://www.fca.org.uk/publication/regulatory-priorities/insurance-report.pdf).

The consolidation is a meaningful change in how the FCA communicates expectations. A single report creates a clearer record of what the regulator said and when. That matters for governance documentation. It also means pricing-relevant signals sit alongside conduct and claims signals in one place, which requires pricing teams to actually read it rather than waiting for the GI-specific extract from compliance.

Here is what pricing teams need to take from it.

---

## Pet insurance and PMI: enforcement is coming

The FCA was not subtle. The report explicitly flags "price increases and consumer understanding issues" in pet insurance and private medical insurance, and states that "there may be regulatory action in 2026."

That phrasing — "there may be regulatory action" — is not standard regulatory boilerplate. The FCA uses it deliberately when it has already begun internal work and wants firms to know the clock is running.

For pricing teams, this means two things. First, the substantive issue: if your pet or PMI pricing has seen material rate increases over the past two to three years, you should have Consumer Duty fair value documentation that explains why those increases were justified. "Claims inflation" is an answer. "Claims inflation plus margin expansion" is not a defensible answer if the margin expansion is material and your value assessment has not been updated to reflect it.

Second, methodology. The FCA is looking at consumer understanding alongside price. If your pricing model has opacity that makes it difficult to explain why a specific customer's premium is what it is — particularly at renewal — that is a dual problem: fair value and consumer understanding. A model that segments 50 risk factors down to a quoted premium with no tractable explanation for the movement year-on-year will attract questions.

We have seen some teams treat the Consumer Duty fair value assessment as a once-per-year compliance exercise. That is the wrong posture. For products currently under regulatory scrutiny, it needs to be a live document with pricing decisions logged against it.

---

## AI in underwriting and pricing: the FCA is watching, standards are not set yet

The report states that the FCA will evaluate AI risks in insurance, and that pricing using AI must demonstrate Consumer Duty good outcomes. It does not define "AI" precisely — a normal GLM with a GBM uplift factor would not typically attract the label, but a model where the dominant pricing signal comes from a neural network or an LLM-derived feature almost certainly would.

What matters practically: no technical standards have been published. Guidance is expected later in 2026. This is actually the window to get ahead of it rather than wait.

The FCA's AI concerns in insurance cluster around three things based on their broader statements: proxy discrimination (protected characteristics being proxied through seemingly neutral variables), lack of explainability at the individual decision level, and governance gaps where the team deploying the model does not understand the signals it is acting on.

If you are using a black-box component in production pricing, the question to ask now is not "can we explain this model in general" but "can we explain why this specific customer's premium is £X rather than £Y." That is a different and harder question. Shapley values give you a decomposition. Whether that decomposition makes intuitive sense, and whether you can document it as a fair outcome, is a separate matter.

The [insurance-fairness](https://pypi.org/project/insurance-fairness/) library includes tooling for proxy discrimination testing — identifying whether protected characteristic proxies are doing unexpected work in a model. It is not a compliance certification; it is a diagnostic. Running it now, before FCA guidance crystallises, gives you a baseline and a record that you looked.

---

## General insurance pricing rules: renewal ban continues, interventions will be strengthened

The 2022 renewal pricing rules — the ban on charging renewing customers more than new customers for equivalent risk — remain in force and will not be loosened. The FCA is explicit that it will strengthen interventions if the rules are not delivering fair outcomes in practice.

The notable operational change: the FCA has deleted three pricing practices data returns as part of its simplification work. This is not a relaxation of the underlying rules; it is a reduction in the form-filling that surrounds them. The substantive obligation is unchanged.

For pricing teams, "strengthen interventions" is the phrase to focus on. The FCA has already collected two years of data under the new regime. It knows which firms are compliant and which are achieving technical compliance while still producing renewal premium paths that penalise inertia through other mechanisms — ancillary products, policy terms changes, risk re-underwriting. If your pricing has been working around the renewal rules through these mechanisms, the time to review that approach is now.

The renewal pricing rules also interact with AI governance in an interesting way. If your renewal premium is set partly by a model you cannot fully explain, and that model produces prices that disadvantage renewing customers relative to new-business equivalents, demonstrating compliance becomes substantially harder. The two issues are linked.

---

## Home contents for social renters: fairness audit implications

The FCA has flagged home contents insurance access for social renters as a priority — specifically, pricing models that exclude or significantly overprice cover for people in social housing.

This is not primarily a revenue issue for most insurers. Social renters are typically underwritten in specialist schemes or declined outright by mainstream providers. The regulatory pressure here is about market access.

The practical implication for pricing teams with home contents products is whether your model treats property tenure as a risk factor in a way that, even where it is correlated with claims experience, produces outcomes that are disproportionate or that proxy socioeconomic vulnerability. That distinction — between "this group has higher claims" and "this group is being excluded from affordable cover" — is where Consumer Duty fairness analysis becomes necessary rather than optional.

If you do not currently offer cover to social renters, you are not required by the FCA to start. But you should be able to document the actuarial and commercial basis for the exclusion. "We have always done it this way" is not that documentation.

---

## Claims handling and fair value

The report treats claims handling as a Consumer Duty output metric. The FCA will monitor service quality through outsourced claims handlers, not just the insurer directly. For pricing teams, this matters because fair value assessments are supposed to account for the full customer outcome — not just the premium-to-expected-loss ratio but the quality of the claims process the customer receives when they need it.

If your pricing model implicitly assumes a standard service quality that is not being delivered by your outsourced claims operation, your fair value assessment is wrong. This is not a hypothetical: the combination of labour shortages and claims inflation over 2023-25 has degraded claims handling service across the market, and in some lines the degradation has been material.

---

## What pricing teams should do now

Five things, in approximate order of urgency.

**If you write pet or PMI:** Pull your fair value assessment. Check that rate increase justification is documented decision-by-decision and that consumer understanding of pricing is addressed. Do this before the end of Q2. If the documentation is thin, that is a gap to fix in the next two months, not something to leave until a supervisory review forces it.

**If you use any form of ML or AI in production pricing:** Run a proxy discrimination test now and document the results. Establish a baseline against which FCA guidance, when published later in 2026, can be assessed. The [insurance-fairness](https://pypi.org/project/insurance-fairness/) library is a starting point. Having no documentation when guidance arrives is worse than having imperfect documentation.

**On renewal pricing:** Audit your renewal pricing paths for the past 12 months. Not just the headline premium comparison — the full picture including ancillary products and terms changes. If you see patterns that disadvantage renewing customers through mechanisms other than the headline rate, review them with your pricing governance framework before the FCA does.

**On social renters:** If you exclude or heavily load social housing tenure, document the actuarial basis explicitly. Be specific about claims experience by tenure type and the commercial rationale. Generic references to "higher risk" will not be sufficient if this comes under scrutiny.

**On AI governance generally:** If you use the [insurance-governance](https://pypi.org/project/insurance-governance/) library for PRA SS1/23 model validation, ensure your AI-component models are covered in the validation scope. The FCA and PRA expectations are converging on requiring that AI pricing components go through the same governance as statistical models, not lighter-touch sign-off. If your governance framework treats neural network components differently from GLMs, that distinction will need a documented rationale.

The FCA's consolidated report format makes it easier to track what they said and when. The flip side is that it also makes your documentation easier to audit against. These priorities will not change materially before the end of 2026. Treat them as the frame within which every significant pricing decision this year needs to be logged.
