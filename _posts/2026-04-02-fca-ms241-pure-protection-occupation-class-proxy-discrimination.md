---
layout: post
title: "FCA MS24/1 and Income Protection: What Occupation Class Is Actually Doing to Your Premiums"
date: 2026-04-02
author: Burning Cost
categories: [fairness, regulation]
tags: [fca, ms24-1, income-protection, occupation-class, proxy-discrimination, fairness, insurance-fairness, EqA-2010, consumer-duty, multi-state, pure-protection]
description: "The FCA's pure protection market study interim report landed in January 2026. The final report is due Q3 2026. For income protection pricing teams, the central question is whether occupation class encodes proxy discrimination through the transition rate structure — and whether you can demonstrate it does not."
---

The FCA's pure protection market study (MS24/1) is not a theoretical exercise. The interim report, published in January 2026, contains specific concerns about whether income protection and critical illness pricing produces fair outcomes across protected characteristic groups. The final report is expected in Q3 2026. That is roughly six months for protection pricing teams to form a view on a question most of them have not formally examined: whether their occupation class pricing encodes proxy discrimination, and whether they can demonstrate otherwise.

This post is not about the theory of multi-state fairness. We covered that [last week]({{ site.baseurl }}{% post_url 2026-04-01-multi-state-transition-fairness-ltc-ci-insurance-pricing %}). This post is about what the FCA is actually asking, what a pricing team needs to produce before Q3 2026, and why the standard analytical workflow is likely to give the wrong answer.

---

## What MS24/1 is actually scrutinising

The FCA's concern in MS24/1 is not that income protection is mispriced in a simple sense. The concern is structural: whether the pricing methodology systematically produces worse outcomes for groups with protected characteristics under the Equality Act 2010.

For IP, the primary proxy risk is occupation class. Occupation is a legitimate risk factor — a category A (professional) occupation has materially lower disability inception rates than a category D (heavy manual) occupation. This is actuarially robust and has been standard IP practice for decades. The FCA is not asking firms to remove occupation from their models.

The problem the regulator is concerned about is different. Occupation in the UK correlates with protected characteristics — race, disability status, sex — in well-documented ways. The ONS Labour Force Survey data shows that Black and minority ethnic workers are overrepresented in manual and semi-skilled occupations (categories C and D) relative to professional occupations (category A) after controlling for age and education. Disabled workers are significantly underrepresented in professional occupations. These are not small effects.

An IP pricing model that uses occupation class as a rating factor is, therefore, pricing some protected groups at materially higher rates than others — not because of any deliberate discriminatory intent, but because the rating factor is a partial proxy for protected characteristics. Under EqA 2010 s.29, indirect discrimination in service provision does not require intent. It requires a provision, criterion, or practice that puts persons with a protected characteristic at a particular disadvantage.

The FCA's specific question, as we read MS24/1, is: firms are using occupation classification — can they demonstrate that the pricing differential it produces is proportionate to legitimate actuarial aim, and cannot be achieved by less discriminatory means?

---

## Why your current analysis will not answer this question

The standard IP pricing governance workflow involves checking that rated premiums are within a reasonable range across occupation classes, that experience emerges as expected, and that the rating factors have statistically significant predictive power. None of this addresses the FCA's question.

What the FCA is asking is a counterfactual: if you removed the occupation-related component of your premium that tracks the protected characteristic distribution, would the residual premium still adequately reflect risk? If the answer is yes, the current pricing practice may not survive EqA proportionality scrutiny. If the answer is no — the occupation risk differential is real and not reducible to protected characteristic proxies — you need to be able to demonstrate this with data.

The distinction matters. Occupation class captures two things simultaneously:

1. **Genuine occupational disability risk** — a miner's physical loading hazard, a surgeon's manual dexterity exposure, a driver's road accident risk. These are real risk differentials unrelated to protected characteristics.
2. **Socioeconomic and demographic correlates** — the occupation distribution itself reflects historical and current barriers that are correlated with race, disability, and sex. A model trained on occupation absorbs both signals.

Lindholm-style discrimination-free pricing separates these by marginalising out the protected characteristic component. Applied to a single-period product, the calculation is straightforward. Applied to IP, it is not — because the premium is a function of multiple transition rates (healthy-to-disabled, disabled-to-recovered, disabled-to-dead), and applying Lindholm to the aggregate premium rather than each transition rate produces a mathematically incorrect result.

The April 1 post explains the linear algebra of why this is wrong. The practical consequence is: if your team has run any kind of fairness analysis on IP premiums by applying marginalisation at the premium level, the results are unreliable. The correction must happen at the transition rate level.

---

## The occupation class rate ratio problem

To see how significant this is in practice, consider the rate ratio structure in a standard IP model with four occupation classes.

Before discrimination-free correction, a typical per-transition rate ratio structure might look like this (Class 4 / Class 1):

- Healthy → Disabled inception rate: 1.8–2.5x
- Disabled → Recovery rate: 0.65–0.80x (lower recovery, i.e., longer disability spells)
- Disabled → Death: 1.2–1.4x

These ratios are actuarially meaningful. But how much of the 1.8x inception ratio for Class 4 versus Class 1 is genuine occupational loading, and how much is the occupation-based selection effect on who ends up in each class?

The discrimination-free correction applied per transition answers this. It marginalises out the component of each transition rate that tracks the occupation class distribution, producing corrected rates $\lambda^*_{H \to D}$, $\lambda^*_{D \to H}$, $\lambda^*_{D \to \text{Dead}}$ that reflect risk variation across age, sex, and smoking status but not across the distribution of occupation classes treated as a sensitive attribute.

In the `insurance-fairness` v1.1.0 implementation:

```python
import polars as pl
from insurance_fairness.multi_state import MultiStateTransitionFairness

mstf = MultiStateTransitionFairness(
    states=["H", "D", "Dead"],
    absorbing_states=["Dead"],
    sensitive_col="occupation_class",
    feature_cols=["age_band", "sex", "smoker", "benefit_amount"],
    discount_rate=0.03,
)

report = mstf.fit_transform(
    panel_df=ip_panel_df,
    benefit_state="D",
    benefit_type="annuity",
    term_years=20,
)

# Per-transition rate ratios: what remains after correction?
print(report.rate_ratios_by_transition)
```

The `rate_ratios_by_transition` output shows you, for each transition, the rate ratio across occupation classes before and after correction. If the correction brings the H→D ratio from 1.85 to 1.02 for Class 4 versus Class 1, that is evidence that most of the rated disability inception differential is tracking the protected characteristic distribution, not genuine occupational risk. If it brings it from 1.85 to 1.60, most of the differential is attributable to genuine risk factors.

Either result is actuarially useful. The first result is regulatorily significant: it tells you your occupation class premium differential is substantially a proxy for protected characteristics, and a proportionality defence is very difficult to construct. The second result supports proportionality: the differential persists after removing protected characteristic components, indicating it reflects genuine risk.

The point is that without running this analysis, you do not know which situation you are in.

---

## What the FCA will want to see

The FCA does not currently specify the exact methodology for demonstrating non-discrimination in protection pricing. MS24/1 final report will presumably contain more specific guidance. But the EqA proportionality standard — which is the legal standard underlying the regulatory concern — requires firms to show:

1. A legitimate actuarial aim for the rating practice (satisfied by standard disability risk arguments for occupation class)
2. That the practice is proportionate to that aim — i.e., that it is not more discriminatory than necessary to achieve the aim
3. Ideally, that less discriminatory alternatives were considered and rejected

The `MultiStateFairnessReport` provides the documentary evidence for point 2. The before/after premiums by occupation class, the per-transition rate ratios, the deviance cost of the correction — these are the numbers that go into a fair value assessment. The deviance cost is particularly important: if the discrimination-free premium produces only a 1.7% increase in model deviance, the argument that the current discrimination is "necessary" for accurate pricing is weak. If it produces a 25% deviance increase, the argument is much stronger.

The audit trail also matters for point 3. If you have run the analysis, shown the rate ratio structure before and after correction, and documented why the residual differentials are proportionate to genuine risk factors, that is a defensible position under EqA s.29. If you have not run the analysis, you are in a much weaker position — you cannot demonstrate proportionality without having measured the extent of discrimination in the first place.

---

## The sensitive attribute problem

One complication specific to disability insurance: disability is simultaneously a protected characteristic under EqA s.6 and a model state. The IP multi-state model has "Disabled" as a state. When you specify `occupation_class` as the sensitive attribute in `MultiStateTransitionFairness`, the correction marginalises out the occupation component of each transition rate. But the disability state itself is part of the model structure.

This creates a tension that is not resolved in the Lim/Xu/Zhou paper or in insurance-fairness v1.1.0. The framework assumes the sensitive attribute is fixed at policy issue. For an attribute like race or sex, this is appropriate. For disability status, the attribute can change over the lifetime of the policy — someone who was not disabled at issue may become disabled and then recover.

For the purposes of EqA compliance in IP pricing, we think the correct sensitive attribute is not disability status (which is a rated outcome, not an underwriting characteristic) but rather the socioeconomic proxies that correlate with protected characteristics: occupation class, postcode sector, educational attainment if rated. Occupation class is the highest-priority proxy in the UK IP market and the one MS24/1 appears most concerned with.

The `bias_correction='kl'` option in `LindholmCorrector` handles sparse groups — "Other" ethnicity categories in a UK IP portfolio may have small sample sizes and the KL correction reduces instability in the marginal distribution estimate.

---

## Timeline and practical steps

If the FCA's final MS24/1 report lands in Q3 2026 with specific requirements for demonstrating non-discrimination in IP pricing, firms that have done the analysis will have a significant advantage in response time. The steps are:

**Now:** Obtain IP panel data in the format `TransitionDataBuilder` expects — one row per individual-wave, with state, age, and feature columns. If your data is in annual renewal format rather than panel format, `TransitionDataBuilder.build()` handles interval splitting. The data preparation is likely the largest time investment.

**Before Q2 2026:** Run `MultiStateTransitionFairness` with occupation class as the sensitive attribute. Produce the `MultiStateFairnessReport` and review the per-transition rate ratios. Understand which transitions are driving the occupation-class premium differential and whether those differentials persist after discrimination-free correction.

**Before Q3 2026:** Document the proportionality argument. If the rate ratios persist after correction (genuine risk), document why. If they collapse after correction (proxy), decide whether to adjust pricing methodology before the FCA asks you to.

The tool is in production. It requires polars, numpy, and scipy. The data requirement is the limiting factor for most teams — IP panel data in an appropriate format is not something all insurers have readily accessible from their policy administration systems.

---

## Install

```bash
uv add insurance-fairness==1.1.0
```

The multi-state module is in `insurance_fairness.multi_state`. Full pipeline documentation and a worked example on synthetic ELSA-format panel data are in the [multi-state notebook](https://github.com/burning-cost/insurance-fairness/blob/main/notebooks/multi_state_income_protection.ipynb).

The FCA will ask these questions. The only variable is when.
