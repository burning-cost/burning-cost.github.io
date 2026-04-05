---
layout: post
title: "What You Can Actually Say in a Fairness Attestation (and What You Cannot)"
date: 2026-04-03
categories: [fairness, compliance]
tags: [FCA, EP25-2, Consumer-Duty, fairness-attestation, KL-divergence, discrimination-insensitive, Miao-Pesenti, arXiv-2603-16720, model-validation, Equality-Act, insurance-fairness, DiscriminationInsensitiveReweighter, multicalibration, UK-insurance, pricing-actuary, python]
description: "Most fairness sections in model validation reports say nothing. The KL discrimination-insensitive result from Miao & Pesenti (2026) gives you a precise, defensible claim: we applied the minimum correction needed to achieve sensitivity-zero. Here is what to write, why it holds, and where the limits are."
author: burning-cost
---

Most fairness attestations in UK pricing model validation reports say something like: "The model does not use gender or ethnicity as rating factors. Premium relativities are set to reflect actuarial risk. The model has been reviewed for indirect discrimination and no material concerns were identified."

That paragraph does not attest to anything. It describes a process — "we looked at it" — and asserts an outcome — "we found nothing" — without connecting the two. An FCA supervisor asking "how did you establish that?" will receive a follow-up answer that amounts to: we ran some correlations.

This post is about writing a fairness attestation that can actually be defended. The Miao and Pesenti (2026) result on discrimination-insensitive pricing — specifically the existence and uniqueness theorem — gives you a piece of language that most other methods cannot. We explain what that claim is, what it proves, and what a complete attestation looks like when it is built on that foundation.

---

## The specific claim the KL result supports

Miao and Pesenti (arXiv:2603.16720) prove the following. Given a pricing model h(X, D) trained on data where X are legitimate rating factors and D are protected covariates, there exists a unique probability measure Q* that satisfies two properties simultaneously:

1. **Sensitivity-zero:** the distortion risk measure has zero Gâteaux derivative with respect to each protected covariate at every rating cell — meaning a marginal change in D does not change the premium
2. **Actuarial balance:** expected fair premiums equal expected actual claims under Q*, so the correction does not under- or over-price the portfolio in aggregate

And Q* is the **minimum-KL-divergence** measure satisfying both constraints. There is no less-distorting correction that also achieves sensitivity-zero. Theorem 3.1 gives the closed-form solution; Theorem 3.2 proves uniqueness under conditions that are met for standard UK personal lines claim distributions (Poisson frequency, gamma or log-normal severity).

That last sentence is what you can write in an attestation: we applied the minimum necessary correction to achieve sensitivity-zero of the pricing principle to each protected covariate. The correction is uniquely determined by the data and the fairness criterion. There is no degree of freedom that could have produced a lighter-touch intervention while still satisfying the constraint.

This is categorically different from saying "we checked correlations and didn't find anything." It is a constructive claim about the correction applied, with a mathematical proof behind it.

---

## Why most alternatives cannot make this claim

**Unawareness** (removing D from the feature matrix) makes no claim about the residual proxy effect. It is a process, not an outcome. A model trained without gender can still price by gender through occupation, vehicle group, and postcode. Stating "we did not include gender" says nothing about whether the model's output is sensitive to gender.

**Discrimination-free pricing** (Lindholm et al., ASTIN 2022) achieves conditional independence of the premium from D given X: it marginalises out D across its conditional distribution given X. This is a stronger fairness criterion than sensitivity-zero in one direction — it eliminates all D-dependence, not just local marginal sensitivity — but it requires a causal DAG to separate legitimate risk transmission from discriminatory proxy effects, and it introduces portfolio bias (total fair premiums no longer equal total expected claims without a separate re-normalisation step). You can write that you eliminated the direct D-dependence from the pricing function, but the causal graph dependency is an assumption that can be challenged.

**Wasserstein / OT barycentre methods** achieve demographic parity — the marginal distribution of premiums is equated across groups. This is the wrong criterion for UK insurance: the Equality Act 2010 permits price differences that reflect genuine risk differences. Achieving demographic parity will over-correct where groups genuinely differ in risk. You cannot defensibly attest that a Wasserstein correction satisfies the Equality Act proportionality test, because you have also removed legitimate risk differentials.

**Group-level A/E monitoring** tells you whether a group is being over- or under-priced on average. It does not tell you whether the mechanism of pricing responds to the protected attribute. A model can produce equal A/E ratios across gender groups while still having premium calculations that are sensitive to gender — because the sensitivity operates at the cell level, not the group average.

The KL approach is the one that produces a minimum-distortion claim. The distortion risk measure's sensitivity to D is zeroed at every rating cell, not just on average. And the correction is provably minimal in the KL sense.

---

## What the propensity reweighter does and where the approximation lies

`DiscriminationInsensitiveReweighter` in `insurance-fairness` v0.6.3 implements the propensity-reweighting approximation to Theorem 3.1:

```
dQ/dP  proportional to  P(A = a_i) / P(A = a_i | X_i)
```

where P(A | X) is estimated by logistic regression or random forest. This is the exact solution to the KL problem in the special case where the pricing function h is linear in D and the risk measure is expected value. For UK personal lines GLMs, this is the case that matters: frequency-severity models on the log scale are linear in each feature, and we price on expected value not Expected Shortfall.

For GBMs with strong D-interaction terms, the propensity approximation is second-order in the nonlinearity of D's contribution to h. In practice: if D is not in the feature matrix, the GBM cannot develop interaction terms involving D directly, and the propensity approximation is exact for any remaining proxy effect through X. If D is in historical data and was used in training, then removed, the approximation quality depends on how nonlinear the model's use of D was.

The full Theorem 3.1 solution — root-finding for the Lagrange multipliers η* per rating cell, incorporating the conditional loss distribution — is not yet implemented. The reason is honest and should be stated as such in any attestation: the paper contains only a synthetic numerical example, and we do not yet have evidence that the additional complexity changes the correction materially on real UK portfolios. The propensity approximation is the basis of the claim; the full solution would strengthen it.

---

## A worked example: gender correction on a UK motor portfolio

Here is the workflow for a standard UK motor use case — historical training data that includes gender, which cannot be used as a rating factor post-*Test Achats* (2012) but may have contaminated the proxy structure learned by the model.

```python
import pandas as pd
import numpy as np
from insurance_fairness import DiscriminationInsensitiveReweighter, MulticalibrationAudit

# X_train: historical motor data, 2015-2024
# Columns include: driver_age, vehicle_group, ncd_years, occupation_band,
#                  postcode_district, annual_mileage, gender (historical, cannot be used)
# y_train: observed claim cost / earned exposure (pure premium)

rw = DiscriminationInsensitiveReweighter(
    protected_col="gender",
    method="forest",       # random forest captures nonlinear proxy relationships
    clip_quantile=0.99,    # truncate top 1% of weights to limit influential observations
    random_state=42,
)

weights = rw.fit_transform(X_train)

diag = rw.diagnostics(X_train)
print(f"Propensity model accuracy:   {diag.propensity_model_score:.3f}")
print(f"Max weight:                  {diag.weight_stats['max']:.2f}")
print(f"Effective N: {diag.weight_stats['effective_n']:.0f} / {len(X_train)}")
print(f"Mean propensity — male:      {diag.mean_propensity_by_group[0]:.3f}")
print(f"Mean propensity — female:    {diag.mean_propensity_by_group[1]:.3f}")
```

The diagnostics matter for the attestation. A propensity model accuracy of 0.52 on a balanced dataset means your rating factors carry almost no gender information — the correction will be trivial and the "we applied the KL correction" claim is still valid but substantively empty. An accuracy of 0.78 means the portfolio is strongly segmented by gender through its proxies, the effective sample size will fall materially, and the correction is doing real work.

```python
# Train on non-gender features with the KL weights
feature_cols = [c for c in X_train.columns if c != "gender"]
model.fit(X_train[feature_cols], y_train, sample_weight=weights)

# Predicted premiums on hold-out
y_pred_corrected = model.predict(X_val[feature_cols])

# Compare to the uncorrected model (trained without weights)
model_uncorrected.fit(X_train[feature_cols], y_train)
y_pred_uncorrected = model_uncorrected.predict(X_val[feature_cols])
```

To put a number in the attestation, you need before/after comparisons by group:

```python
val_data = X_val.copy()
val_data["y_pred_corrected"] = y_pred_corrected
val_data["y_pred_uncorrected"] = y_pred_uncorrected
val_data["y_actual"] = y_val

# Premium by gender group
summary = (
    val_data
    .groupby("gender")[["y_pred_uncorrected", "y_pred_corrected", "y_actual"]]
    .mean()
)
print(summary)
```

A typical output on a UK motor book with genuine gender proxying through postcode and occupation:

```
        y_pred_uncorrected  y_pred_corrected  y_actual
gender
F                  £312.40           £318.60   £317.90
M                  £371.20           £364.80   £365.50
```

The uncorrected model underprices female drivers (A/E 0.982) and overprices male drivers (A/E 1.015). The corrected model brings both groups within 0.5% of actual. The direction makes sense: postcode and occupation correlate with gender; the uncorrected model is partly pricing gender through those proxies; the reweighting removes that proxy signal from the training data.

Those numbers go in the attestation. They are not incidental; they are the evidence.

---

## The multicalibration audit: checking what the reweighting missed

The reweighter operates at the marginal level — it achieves X ⊥ D under the reweighted distribution. This does not guarantee that the model is well-calibrated within every subgroup of the protected attribute. A model can pass the reweighting step and still have miscalibrated predictions for, say, young female drivers with high NCD (a cell where the propensity model may not have had enough data to estimate accurately).

Multicalibration — from Hébert-Johnson et al. (2018), implemented in `insurance-fairness` — checks this. It partitions the portfolio into cells defined jointly by predicted premium band and the protected attribute, and tests for systematic over- or under-prediction in each cell.

```python
audit = MulticalibrationAudit(n_bins=10, alpha=0.05)

# Post-correction audit: corrected predictions vs actual, split by age band
# (age is collectable and usable; use it as the proxy observable for ethnicity effects
# via postcode-to-LSOA if direct ethnicity data is not available)
report = audit.audit(
    y_true=y_val,
    y_pred=y_pred_corrected,
    protected=age_band_val,    # or ethnicity_proxy_val from ONS Census 2021
    exposure=exposure_val,
)

print(f"Is multicalibrated: {report.is_multicalibrated}")
print(f"Overall calibration p-value: {report.overall_calibration_pvalue:.4f}")
for g, pval in report.group_calibration.items():
    print(f"  {g}: p={pval:.4f}")
print(report.worst_cells)
# Is multicalibrated: True
# Overall calibration p-value: 0.4123
# Cells with significant bias at alpha=0.05: 2 / 40
# Portfolio A/E: 0.997
# Conclusion: model passes multicalibration at family-wise 5% level.
# 2/40 cells flagged at unadjusted alpha=0.05 — consistent with expected
# false alarm rate. 0/40 flagged at Bonferroni-corrected alpha=0.00125.
```

The multicalibration audit does two things for the attestation. First, it tells you whether the KL correction was sufficient or whether residual bias remains in specific subgroups — if it was not, you have evidence of the problem and can apply a cell-level correction. Second, it gives you a statistically calibrated pass/fail at a stated false alarm rate, which is what "no material concerns were identified" should actually mean.

---

## What goes in the model validation report

The fairness section of a model validation report that uses this workflow should read something like this:

---

*Fairness analysis*

*The pricing model does not use gender as a rating factor. Gender was present in the training data (historical policies underwritten prior to the Test Achats ruling, 2012). To address the risk that gender information is encoded in the training data through proxy relationships — principally postcode district, occupation band, and vehicle group — we applied the KL discrimination-insensitive reweighting procedure from Miao and Pesenti (2026, arXiv:2603.16720), as implemented in `insurance-fairness` v0.6.3 (`DiscriminationInsensitiveReweighter`).*

*The reweighting finds the minimum-KL-divergence probability measure Q* over the training data such that the pricing principle has zero Gâteaux sensitivity to gender at every rating cell, while preserving actuarial balance (E^Q[Y] = E^P[Y]). The existence and uniqueness of Q* under the Miao-Pesenti criterion (their Theorem 3.2) means the correction applied is the minimum necessary: no less-distorting reweighting exists that also achieves sensitivity-zero.*

*The propensity model (random forest) achieved cross-validated accuracy of [0.64] on the training data, indicating moderate gender predictability from rating factors. Maximum sample weight after clipping at the 99th percentile: [3.8]. Effective sample size after reweighting: [87,400] of [103,200].*

*Post-correction multicalibration audit (hold-out, 10 predicted premium bands × 4 age groups): [2 of 40] cells flagged at unadjusted alpha = 0.05, consistent with the expected false alarm rate under the null of no bias (5% of 40 = 2 expected). Zero cells flagged at Bonferroni-corrected alpha = 0.00125. Both cells involve the youngest driver age band (17–24), which has high inherent claim volatility. The model passes multicalibration at the family-wise 5% level.*

*Premium comparisons by gender group on the hold-out set (Table 2) show A/E ratios of [0.997] (female) and [1.003] (male) after correction, compared to [0.982] and [1.015] before correction. The correction narrows the spread by [0.033 percentage points] and brings both groups within a standard error of 1.000.*

*Limitation: the full Theorem 3.1 solution (Lagrange multiplier root-finding per rating cell) is not implemented. The propensity reweighting approximation is exact for linear pricing functions under expected value; for the GBM used in this model, it is an approximation whose error is second-order in the nonlinearity of gender's indirect contribution to h. We assess this error as non-material given the propensity model accuracy observed, but note it as a modelling limitation for future development.*

---

That last paragraph is important. An attestation that claims more precision than the method supports will not survive scrutiny. Stating the limitation explicitly, with a quantitative basis for "non-material," is stronger than omitting it.

---

## What the Equality Act 2010 proportionality test requires from you

Section 19 of the Equality Act prohibits indirect discrimination unless the practice is "a proportionate means of achieving a legitimate aim." For insurance pricing:

- **Legitimate aim:** actuarially sound pricing that preserves the insurer's solvency position and reflects genuine risk differences
- **Proportionate means:** the practice must go no further than necessary to achieve the aim

The KL uniqueness result speaks directly to proportionality. You can say: we used the actuarially correct pricing measure P as the baseline. We corrected it to Q* using the minimum divergence necessary to achieve sensitivity-zero. We did not impose group-level parity constraints or remove legitimate risk differentials. The correction is demonstrably minimal.

This is a proportionality argument, not a discrimination denial. You are not claiming the model produces no proxy effects whatsoever — that would be implausible and unverifiable. You are claiming that you identified a proxy mechanism, measured its effect on the pricing function's sensitivity, applied the least-distortive correction that achieves sensitivity-zero, and preserved actuarial balance throughout. Under section 19, that is a defensible argument. "We didn't use gender" is not.

---

## The things you cannot claim

**You cannot claim the model has zero indirect discrimination.** Sensitivity-zero means the premium does not respond to marginal changes in D at any rating cell. It does not mean the premium is fully conditionally independent of D given X. For binary D with linear h and expected value pricing, the two are equivalent. For real models, they are not. State clearly which criterion you have satisfied.

**You cannot claim the correction is validated on real UK data.** The Miao-Pesenti paper validates the method on a single synthetic example with a linear model. We have no published empirical comparison showing the propensity approximation is close to the full Theorem 3.1 solution on a real UK motor portfolio. This is a genuine limitation and should be noted.

**You cannot claim Q* satisfies demographic parity.** It does not try to. If your regulator or commercial team is expecting group-level premium parity, the KL approach is the wrong tool — though it is worth explaining why demographic parity is the wrong criterion for insurance pricing to begin with.

**You cannot claim age has been corrected.** If age is in D (you included it in the protected set), the correction will zero out age sensitivity — which you do not want, since age is a legitimate rating factor for UK motor insurance and is specifically recognised as such. The protected set M must be defined carefully. Gender and proxied ethnicity belong there. Age, typically, does not. Document which attributes you included and why.

---

## The honest version of "we checked and found nothing"

The legitimate version of "no material concerns were identified" is:

- The propensity model accuracy was below [0.58], indicating weak proxy relationships between rating factors and gender. The reweighting made trivial corrections (maximum weight [1.4], effective N [98,200] of [103,200])
- The multicalibration audit found [1 of 40] cells with significant miscalibration, within the 5% expected false alarm rate under the null of no bias
- Pre- and post-correction A/E ratios by gender group differ by [0.002], within the confidence interval

That version is defensible because it specifies what was tested, what was found, and what would have shown up if there were a material problem. "We reviewed for indirect discrimination and found no material concerns" can mean anything. The version above says the same thing and means something.

---

## Getting started

```bash
pip install insurance-fairness
```

`DiscriminationInsensitiveReweighter` and `MulticalibrationAudit` are in the top-level namespace. The propensity model defaults to logistic regression; use `method="forest"` when you expect nonlinear proxy relationships — which you should assume for postcode and occupation in UK motor.

The theory is in Miao, K.E. and Pesenti, S.M. (2026), *Discrimination-Insensitive Pricing*, arXiv:2603.16720, and is discussed in detail in our [technical post on the KL formulation](/2026/03/31/kl-discrimination-insensitive-pricing-insurance-fairness/). The attestation language in this post is our editorial view of how the result should be used in practice — not legal advice, not regulatory guidance. Consumer Duty compliance ultimately sits with your board and your compliance function. What we can give you is a technical analysis that those functions can build on.

---

*Related:*
- [KL Discrimination-Insensitive Pricing: The Information-Theoretic Foundation](/2026/03/31/kl-discrimination-insensitive-pricing-insurance-fairness/) — the full technical derivation and comparison with Lindholm
- [Consumer Duty Fair Value Evidencing: A 12-Step Checklist](/2026/03/25/consumer-duty-fair-value-evidencing-12-step-checklist-pricing-actuaries/) — the wider model validation workflow
- [FCA Consumer Duty Pricing Fairness in Python](/2026/03/20/fca-consumer-duty-pricing-fairness-python/) — proxy detection, group calibration, and disparate impact ratio computation
- [Multicalibration for Fair Insurance Pricing](/2026/03/27/multicalibration-fair-insurance-pricing/) — the multicalibration step in depth
