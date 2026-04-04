---
layout: post
title: "Multi-State Fairness: Four Things arXiv:2602.04791 Doesn't Tell You"
date: 2026-04-03
categories: [fairness, life-insurance]
tags: [fairness, income-protection, critical-illness, multi-state, lindholm, poisson, kolmogorov, age-conditional-ot, demographic-parity, equalised-odds, disability, ELSA, HRS, insurance-fairness, FCA, ms24-1, Lim-Xu-Zhou, arXiv-2602-04791, UK-personal-lines, python]
description: "The per-transition Lindholm correction for multi-state models is sound. But arXiv:2602.04791 has four gaps that matter before you build on it: no accuracy degradation figures, demographic parity only, the disability-as-state paradox, and age-conditional OT that has no public implementation. Here is what to watch for."
math: true
author: Burning Cost
---

We have covered the core argument of Lim, Xu, and Zhou (arXiv:2602.04791) twice on this site — [in March](https://burning-cost.github.io/fairness/regulation/life-insurance/2026/03/25/fair-pricing-long-term-insurance/) and [again in April](https://burning-cost.github.io/fairness/insurance-pricing/2026/04/01/lindholm-fairness-is-not-enough-for-income-protection-pricing/). The central claim — apply Lindholm marginalisation per transition, not on the aggregate premium, then rerun Kolmogorov — is correct. If you price income protection or critical illness and you are thinking about fairness compliance, this is the paper that matters.

But after a careful read of the full paper, including the appendices, there are four things it does not tell you that you need to know before you build.

---

## 1. Accuracy degradation is not reported

The paper demonstrates that its corrections close the racial premium gap in the HRS long-term care case study. What it does not report is what the corrections cost in predictive accuracy.

This is a significant omission. The discrimination-free transition rates $\lambda^*_m(z, x)$ are marginalised over the sensitive attribute distribution. By construction, they compress rate variation: you are replacing condition-specific predictions with a weighted average across conditions. Deviance increases. AIC worsens. The question is by how much, and the paper declines to answer it.

For a UK regulatory context this matters in two ways.

First, under Consumer Duty, the actuarial function of an insurer must demonstrate that pricing supports fair value. If discrimination-free corrections materially degrade the predictive accuracy of inception and recovery rate models, the insurer has a legitimate question to answer about whether the resulting premiums are appropriately risk-reflective — or whether they are systematically mispricing risk for some groups in a way that creates cross-subsidies. The FCA's pure protection market study (MS24/1, interim January 2026) cares about both discrimination and appropriate pricing.

Second, the Solvency II best estimate liability calculation uses multi-state transition models as inputs. If the 'fair' transition rates diverge materially from the best estimate rates, you have a tension between the internal model (which should use best estimates under SII) and the pricing model (which should use discrimination-free rates under Consumer Duty). This tension is not novel — it exists for single-period models too — but it is more acute in multi-state models because the premium is a nonlinear function of multiple corrected rates, and each correction compounds.

Our implementation in `insurance-fairness` v1.1.0 addresses this directly: `MultiStateFairnessReport` records per-transition deviance before and after correction, and the aggregate premium impact. If you are building this for a real IP book, run those numbers and put them in your model validation document before anyone asks.

```python
report = mstf.fit_transform(panel_df=ip_panel, ...)

# Per-transition deviance comparison
for transition, stats in report.deviance_stats.items():
    pct_change = (stats.deviance_after - stats.deviance_before) / stats.deviance_before * 100
    print(f"{transition}: deviance +{pct_change:.1f}%")

# Aggregate premium shift
print(f"Portfolio average premium before: £{report.mean_premium_before:.0f}")
print(f"Portfolio average premium after:  £{report.mean_premium_after:.0f}")
print(f"Balance check (should be ~0):     {report.balance_error_pct:.3f}%")
```

The balance check matters: discrimination-free pricing does not automatically preserve portfolio balance. Lindholm marginalisation is balance-preserving when $\hat{P}(S = s_j)$ in the correction uses the same population distribution as the book. If your IP book has a non-representative occupational mix — which UK IP books typically do, skewing toward professional and clerical classes — the portfolio-average premium will shift. This is expected and should be documented, not treated as a bug.

---

## 2. Demographic parity only — and the paper is honest about this, but the implications run deep

The paper implements and demonstrates the demographic parity criterion: after correction, premiums do not vary by sensitive group. It acknowledges that equalised odds and predictive rate parity exist as alternative criteria, and it defines them formally in the appendix. It does not provide correction procedures for either.

For most UK insurance applications under the Equality Act 2010 s.29, demographic parity is the relevant standard. Indirect discrimination in service provision means charging a protected group more than a comparator group without proportionate justification. Demographic parity removes the group-mean price differential.

But equalised odds is sometimes the right criterion, and for IP the argument is not frivolous.

Equalised odds requires that the model's error rates are equal across sensitive groups: the rate of incorrectly predicting no disability for people who will become disabled is the same across occupation classes (true positive rate parity), and the rate of predicting disability for people who will remain healthy is the same across groups (false positive rate parity). If your disability inception model systematically under-predicts for certain occupational groups — which it might, if your training data underrepresents long-term disability claims for those groups — demographic parity correction will adjust the premium without fixing the underlying miscalibration.

The practical test: run Lindholm per-transition correction and then check within-group A/E ratios on a holdout. If the corrected rates are still miscalibrated for specific groups — A/E materially different from 1.0 — you have a calibration problem layered on top of the discrimination problem, and demographic parity alone has not resolved it.

```python
from insurance_fairness.multi_state import MultiStateTransitionFitter
from insurance_fairness import calibration_by_group

# After correction, check per-group calibration for H→D transition
h_to_d_cal = calibration_by_group(
    df=holdout_df.filter(pl.col("from_state") == "H").filter(pl.col("to_state") == "D"),
    protected_col="occupation_class",
    prediction_col="corrected_rate_h_to_d",
    outcome_col="event",
    exposure_col="exposure",
)
print(h_to_d_cal.by_group())
```

If the A/E ratio for Class 4 (heavy manual) is 1.23 after the demographic parity correction, your model is underpredicting disability inception for that group. The correction has made the premium demographically neutral but actuarially inaccurate for that subpopulation. These are different problems with different solutions.

---

## 3. The disability-as-state paradox

This is the theoretical gap in the paper that we have not seen addressed anywhere in the fairness literature.

Disability is a protected characteristic under the Equality Act 2010, s.6. In the Lim/Xu/Zhou framework, disability is also a model state. The multi-state model for income protection explicitly contains a "Disabled" state; the H→D transition is the inception rate; the D→H transition is the recovery rate.

The framework treats the sensitive attribute $S$ as fixed at policy issue — race, occupation class, or a socioeconomic proxy. The correction marginalises over $P(S)$ to produce discrimination-free rates.

But for an individual currently in the Disabled state, disability is simultaneously:

- Their current model state (determines which transition rates apply)
- A protected characteristic (disability status is protected under EqA s.6)

The question the paper does not answer: should the D→Dead transition rate, conditional on being currently disabled, be corrected for a sensitive attribute that is itself correlated with whether someone is in the Disabled state? And more pointedly: if occupational class predicts both the H→D inception probability and the D→H recovery probability, and occupation is used as the sensitive attribute proxy, are we correcting the right thing?

The paper's framework assumes $S$ is observed once at issue and held fixed. In a real IP model fitted on panel data, the same individual appears in both the H-risk-of-D rows (exposure contributing to H→D estimation) and potentially in the D-risk-of-H rows (exposure contributing to D→H estimation). The sensitive attribute is the same; the model state is different; the correction is applied independently per transition. Whether this is theoretically correct when $S$ correlates with being in state D is an open question.

For practical UK compliance purposes, the current implementation is defensible: correct the occupation proxy effect at each transition independently, document the per-transition rate ratios, flag the limitation. But if your IP portfolio has a high proportion of mental health disability claims — a protected characteristic under EqA s.6 — and mental health conditions are predictable from occupation, you are in territory the paper has not mapped.

---

## 4. Age-conditional OT is the genuinely novel contribution, and nobody has implemented it

The paper makes three claims of novelty: post-processing per-transition Lindholm (not novel — trivial extension of Lindholm 2022), adversarial in-processing per transition (moderate novelty), and age-conditional optimal transport pre-processing (genuinely novel).

The age-conditional OT is the contribution that does not reduce to a prior method. Here is why it is non-trivial.

Standard OT pre-processing maps the covariate distribution $P(Z | S = s)$ to a common distribution $P(Z^{\perp_S})$ that is independent of $S$. Wüthrich-style pre-processing and the existing `WassersteinCorrector` in `insurance-fairness` do this. For annual GI this is fine — age appears in the model as a rating factor and the transport does not materially distort the age signal.

For a multi-state insurance model, age plays a dual role:

1. Age drives transition rates: older policyholders have materially higher disability inception rates, lower recovery rates, and higher mortality. The age effect is causal and should be preserved.
2. Age determines the remaining term over which the benefit is computed: a 35-year-old with a 20-year IP policy has a different expected present value from a 55-year-old with the same policy. The age effect enters the premium calculation through the Kolmogorov equations directly.

Unconditional OT applied to features including age will partially transport age variation when age correlates with the sensitive attribute. In the UK, age correlates with occupation class (different generations entered different occupational structures), with socioeconomic status, and with the composition of non-white-British populations by industry sector. Running unconditional OT removes some of the age signal alongside the sensitive attribute signal.

Lim/Xu/Zhou's solution is a separate transport map $T_{x^*}: Z \to Z^{\perp_S}$ fitted within each issue age cohort $x^*$. The map achieves $Z^{\perp_S} \perp S \mid X = x^*$: independence from $S$ conditional on age, not unconditionally. Age itself is excluded from the transported feature set and enters the transition models and the Kolmogorov calculation directly.

This requires fitting one OT map per age band — say, five-year cohorts from 25 to 65 — rather than one map for the whole portfolio. For a typical UK IP book this means eight to ten separate transport maps. The computational cost is not prohibitive. The implementation complexity is higher: you need to ensure that the transported features retain meaningful variation within age bands rather than collapsing to the cohort mean.

The paper specifies the approach but provides no code and no numerical results for the pre-processing path. The case study in the paper uses only post-processing. There is no public Python implementation of age-conditional OT for insurance multi-state models, including in `insurance-fairness` v1.1.0. We have this tracked as a v1.2 item, but it requires careful validation against the paper's specification before shipping.

If your team wants pre-processing fairness for income protection now, the honest position is: the theoretically correct tool does not exist in production-ready form. Post-processing per-transition Lindholm is available and covers the main regulatory requirement. Age-conditional OT is the right long-term answer but is currently a research implementation, not a product.

---

## What this means for a UK IP pricing team in Q2 2026

The FCA pure protection market study final report is expected Q3 2026. The question you want to be able to answer before that report lands is: "have you assessed whether your transition rate models embed proxy discrimination, and what did you find?"

The per-transition Lindholm correction in `insurance-fairness` v1.1.0 gives you the technical means to answer that question. You need:

1. Panel data from your claims history with enough duration to fit per-transition Poisson GLMs — this is a data availability question, not a modelling question.
2. A proxy for the sensitive attribute. UK IP insurers rarely hold ethnicity data; the practical proxy is occupation class mapped to a socioeconomic or deprivation score, or MOSAIC/Acorn segment linked to postcode. The choice of proxy is the most consequential methodological decision you will make.
3. The `MultiStateTransitionFairness` pipeline, documented in the [multi-state notebook](https://github.com/burning-cost/insurance-fairness/blob/main/notebooks/multi_state_income_protection.ipynb).
4. Per-transition deviance before and after correction, and the balance check. Both go in the model validation document.

What you do not yet have: age-conditional OT pre-processing, adversarial in-processing, or a framework for handling disability as both state and protected characteristic. These are real gaps. For a first-pass Consumer Duty fairness assessment on a UK IP book, they are not blockers. For a firm that wants to claim it has implemented the complete Lim/Xu/Zhou framework, they are.

The paper is good work. It makes the right architectural argument. But it is a research paper, not an implementation specification, and the gap between its claims and what you can put in production today is larger than the enthusiastic summaries suggest.

---

- [Multi-State Fairness in Income Protection and Critical Illness: Why Lindholm on the Aggregate Premium Is Wrong](/fairness/life-insurance/2026/03/31/multi-state-fairness-income-protection-critical-illness-poisson-decomposition/) — the full technical argument
- [Lindholm Fairness Is Not Enough for Income Protection Pricing](/fairness/insurance-pricing/2026/04/01/lindholm-fairness-is-not-enough-for-income-protection-pricing/) — `insurance-fairness` v1.1.0 implementation and worked example
- [Sequential Optimal Transport for Multi-Attribute Fairness](/fairness/machine-learning/2026/03/31/sequential-ot-fairness-multi-attribute-insurance-pricing/) — extending OT pre-processing to multiple sensitive attributes simultaneously
- [Fairness-Accuracy Pareto Frontiers with NSGA-II](/tutorials/fairness/2026/03/28/fairness-accuracy-pareto-nsga2-insurance-pricing/) — quantifying the tradeoff explicitly for governance
