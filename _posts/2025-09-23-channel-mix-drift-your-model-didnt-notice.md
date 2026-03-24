---
layout: post
title: "Your New Business Mix Changed. Your Model Didn't Notice."
date: 2025-09-23
categories: [model-validation, pricing, techniques]
tags: [covariate-shift, density-ratio, catboost, channel-mix, fca-consumer-duty, portfolio-drift, python, uk-motor]
description: "When a new aggregator partnership or competitor exit changes your new business mix, models trained on the old distribution misprice silently."
---

Something happened to your new business mix last quarter. Maybe a competitor pulled out of a price comparison site and their policyholders came to you. Maybe you struck a new affinity deal and started writing a segment you barely touched before. Maybe you ran a TV campaign targeting a different demographic. The channel mix shifted. Your portfolio is not the population it was when the model was last trained.

The model doesn't know. It scores every incoming risk the way it always has, against the distribution of risks it learned from. If the new channel skews older — say, over-55s who cancelled with the competitor — and the model was trained predominantly on a younger, PCW-driven book, the model will overprice the newcomers on frequency and underprice them on severity. Not uniformly; in the pockets of feature space that are suddenly denser in your new business than they were in training data.

We have seen a motor portfolio where a new PCW partnership skewed the channel significantly younger, lower NCD, and urban. The model trained on the existing direct-channel book — where the median driver age was 47 — started overpredicting frequency for the under-30 cohort by around 8%. Not because the underlying risk relationship changed. Because the model had seen very few young, low-NCD, urban drivers and was extrapolating into sparse feature space.

This post is about detecting that shift before it shows up in loss ratios, and correcting predictions using only the covariate information you already have.

---

## Why this is not a calibration problem

The tempting fix is recalibration. Take the past three months of new business, wait for early claims development, and fit a GLM offset. Platt scaling on top of the model outputs. A credibility-weighted blend.

These all treat the problem as a scalar correction. The issue is that covariate shift creates _segment-specific_ bias, not a flat scalar offset. Recalibrating the model globally will reduce the overall loss ratio error, but it will redistribute bias across segments rather than remove it. In our motor example: a 6% global uplift to correct for the overall shortfall makes the over-55 segment (which the model was already pricing accurately) now 6% overpriced, while the under-30 segment goes from 8% underpredicted to roughly 2% underpredicted. You have moved the error around.

The correct fix is segment-specific reweighting. And the way to do that without waiting months for target labels is density ratio estimation.

---

## The diagnosis

Install the library:

```bash
pip install insurance-covariate-shift
```

You need two covariate matrices: `X_historical` (the policies that were in the training window — call this the source distribution) and `X_new_business` (the last three months of new business — the target distribution). No loss labels required at this stage.

```python
from insurance_covariate_shift import CovariateShiftAdaptor

# categorical_cols takes column indices, not names
# here indices 2, 3, 4 correspond to vehicle_group, area_code, cover_type
adaptor = CovariateShiftAdaptor(
    method='catboost',
    categorical_cols=[2, 3, 4]
)
adaptor.fit(X_historical, X_new_business, feature_names=feature_names)

report = adaptor.shift_diagnostic(
    source_label='Historical book (Jan 2025–Jun 2026)',
    target_label='New business (Jul–Sep 2026)'
)
print(report.verdict)
# MODERATE
print(f"ESS ratio: {report.ess_ratio:.3f}")
# ESS ratio: 0.51
print(f"KL divergence: {report.kl_divergence:.3f}")
# KL divergence: 0.31
```

The ESS (effective sample size) ratio of 0.51 means the historical book is worth roughly half its nominal size as a training set for the new business distribution — the other half of the data is in regions of feature space that aren't well represented in what you're now writing. A SEVERE verdict (ESS ≤ 0.30) would mean you should retrain. MODERATE means correction is viable.

The per-feature importance from the report tells you _where_ the shift is concentrated:

```python
for feat, importance in report.feature_importance().items():
    print(f"{feat}: {importance:.3f}")
# driver_age:     0.41
# ncd_band:       0.28
# area_code:      0.19
# vehicle_group:  0.09
# cover_type:     0.03
```

Driver age and NCD band account for 70% of the detected shift. That is consistent with the channel mix story: new PCW partnership, younger drivers with less NCD.

---

## The correction

Once you have the adaptor fitted, importance weights are available immediately:

```python
weights = adaptor.importance_weights(X_new_business)
# array of shape (n_new_business,), clipped at 99th percentile
```

The weight `w(x)` for a given policy is the density ratio `p_target(x) / p_source(x)`. A weight of 2.3 on a young, low-NCD, urban driver means that profile is 2.3 times more common in the new business than it was in training.

To produce a corrected A/E ratio for the current book, compute the weights for the _source_ calibration set (the held-out portion of your training data with known claims), then use those weights in a weighted frequency calculation:

```python
import numpy as np

# Weights for the source calibration set, pulled towards the target distribution
source_weights = adaptor.importance_weights(X_cal)

# Importance-weighted A/E on the source calibration set
# This estimates how well the model performs on the *target* distribution
naive_ae = model.predict(X_cal).sum() / y_cal.sum()
weighted_ae = (
    (model.predict(X_cal) * source_weights).sum()
    / (y_cal * source_weights).sum()
)
print(f"Naive A/E: {naive_ae:.3f}")
print(f"IW-corrected A/E: {weighted_ae:.3f}")
# Naive A/E: 1.083  (model overpredicts 8% on the shifted distribution)
# IW-corrected A/E: 1.011  (near-flat after reweighting)
```

For segment-level diagnosis, the weights also reveal which cohorts are most affected:

```python
# Segment by driver age band
for age_band, mask in age_band_masks.items():
    seg_weight = weights[mask].mean()
    print(f"{age_band}: mean_weight={seg_weight:.2f}")

# 17-25: mean_weight=2.31
# 26-35: mean_weight=1.74
# 36-50: mean_weight=0.97
# 51-65: mean_weight=0.61
# 66+:   mean_weight=0.44
```

The weights confirm the shift structure. Young drivers are significantly underrepresented in training relative to the new channel — mean weight 2.31 for the 17–25 band. The model has less information where it now needs it most. Over-55s are overrepresented in training relative to new business — the model calibrated well for them and the new channel brings fewer of them. Without correction, the pricing for the 17–25 cohort is based on a distribution that does not match who is actually coming through the door.

---

## FCA Consumer Duty

There is a regulatory angle here that does not get enough attention. Consumer Duty (PS22/9) requires firms to ensure their products and services deliver good outcomes for customers, including on price and value. If your model is systematically overpredicting frequency for a specific demographic cohort because the training distribution is stale, that is not a random error — it is a directional bias affecting identifiable groups of customers.

The FCA expects firms to monitor whether pricing models remain appropriate for the customers they are serving. A model trained 18 months ago on a substantially different channel mix is, by any reasonable reading, not appropriate for the current book without adjustment. The fact that the model has not been changed is not a defence; the book has changed.

The `ShiftDiagnosticReport` produces a plain-language governance summary:

```python
print(report.fca_sup153_summary())
```

```
SHIFT DIAGNOSTIC SUMMARY
Verdict: MODERATE
ESS ratio: 0.51 (threshold: 0.30 SEVERE, 0.60 MODERATE)
KL divergence: 0.31 (threshold: 0.50 SEVERE)
Primary shift drivers: driver_age (0.41), ncd_band (0.28), area_code (0.19)

RECOMMENDATION: Importance weighting correction is appropriate. Consider
flagging to model risk for review at next validation cycle. Under FCA Consumer
Duty (PS22/9), verify that pricing outcomes remain fair across affected
segments. A SEVERE verdict would require material change notification under
FCA SUP 15.3.
```

A MODERATE verdict does not by itself trigger SUP 15.3 notification, but it should appear in the model risk log and the next annual model validation. A SEVERE verdict (ESS ≤ 0.30 or KL ≥ 0.50) is a different matter: that is the level of shift where a model trained on the original distribution is materially misspecified for the current book, and FCA Principle 11 (relations with regulators) comes into play alongside Consumer Duty.

---

## When correction is not enough

Importance weighting corrects for covariate shift: the case where the feature distribution `P(X)` has changed but the conditional loss distribution `P(Y|X)` has not. It cannot correct for concept drift — when the risk relationship itself has changed, not just who is buying. A new PCW bringing in younger drivers is covariate shift if those drivers have the same underlying frequency-given-characteristics as the young drivers in your existing book. If the new channel is systematically adverse-selecting risks within each cell — customers who have been declined elsewhere, or who specifically chose this aggregator because of how it ranked them — that is concept drift and weighting will not save you.

The diagnostic helps here too. If you have even a few months of target labels — claims from the new channel — you can test the covariate shift assumption directly: fit the model using source labels and importance weights, compare to a model fitted on target labels, and check whether the weighted source model's predictions track the target model's in the segments where you have target data. If they diverge systematically, concept drift is the more likely culprit.

That is the point where you need labelled target data and a transfer learning approach rather than pure importance weighting. We cover that in [insurance-thin-data](https://github.com/burning-cost/insurance-thin-data) and [insurance-transfer](https://github.com/burning-cost/insurance-transfer).

---

## What to actually do

The practical workflow, in order:

1. Quarterly, compute the covariate shift diagnostic between your current training window and the last 90 days of new business. This takes under five minutes if you have the data ready.
2. If MINOR, document and move on. If MODERATE, apply importance weighting and log the correction in your model governance record. If SEVERE, escalate to model risk and begin the retraining process.
3. For MODERATE shifts, track whether importance-weighted predictions are tracking actual A/E better than unweighted predictions as claims develop. This is your early warning that concept drift may also be present.
4. Do not wait for the annual model validation to notice this. By then you have a year of mispriced risks on the book.

The library is at [github.com/burning-cost/insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift). The demo notebook in [`burning-cost-examples`](https://github.com/burning-cost/burning-cost-examples/tree/main/notebooks/insurance_covariate_shift_demo.py) runs the full scenario on synthetic UK motor data.

---

## Related articles

- [Your Pricing Model is Drifting](/2026/03/03/your-pricing-model-is-drifting/)
- [Recalibrate or Refit?](/2026/05/14/recalibrate-or-refit/)
- [Your Book Has Shifted and Your Model Doesn't Know](/2026/09/14/your-book-has-shifted-and-your-model-doesnt-know/)
