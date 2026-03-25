---
layout: post
title: "Your New Book Doesn't Look Like Your Old Book. Your Model Doesn't Care."
date: 2026-02-04
author: Burning Cost
description: "Density ratio correction for portfolio composition shift — CatBoost classifier, importance-weighted evaluation, insurance-covariate-shift library"
tags: [covariate-shift, model-validation, density-ratio, catboost, fca]
---

Your frequency model passed validation. Gini looks fine. A/E on holdout is 0.987. The governance committee signed it off and it went live.

Six months later, the loss ratio is climbing and nobody can explain why. The underwriting team points at weather. The claims team points at fraud. The pricing team looks at the monitoring dashboard and sees nothing alarming.

Here is what the dashboard is not showing you: the book you are pricing today is not the book your model was trained on. The loss ratio is not telling you the model is wrong. It is telling you the model is right — for a book you no longer write.

---

## The composition shift problem

Portfolio composition shifts faster than governance cycles. Consider what happened to a typical UK motor book between 2025 and 2027: comparison site algorithm changes rerouted younger, urban traffic; EV adoption shifted vehicle mix in ways that interacted with NCB patterns; broker consolidation pushed older, rural risks onto books that had historically been direct-channel. None of these changes altered the underlying risk relationships. A 23-year-old in an inner-city postcode is still riskier than a 45-year-old in rural Lincolnshire. What changed is the proportions.

This matters because your model was trained to minimise prediction error across the 2025 book distribution. When the 2027 book has three times as many under-25s and half the rural risks, the model's calibration on those segments is no longer tested by the training data's natural distribution. Under-25s were 9% of training observations; they are 28% of the live book. The model's predictions for that segment are based on a thin slice of training data, often fitted at the boundary of the tree structure where extrapolation is happening rather than interpolation.

The damage is asymmetric. The model is well-calibrated on the segments that dominate the training data. It is miscalibrated on the segments that dominate the current book. And your monitoring workflow, which tests model performance on a holdout drawn from the same 2025 distribution, will not catch this.

---

## Why backtesting misses it

The standard backtesting workflow is:

1. Train on Q1–Q3 data
2. Validate on Q4 holdout from the same period
3. Track A/E, Gini, and PSI against that holdout over time

Step 2 is the problem. The Q4 holdout is drawn from the same distribution as the training data. It tests whether the model generalises within the 2025 book. It does not test whether the model generalises to the 2027 book.

PSI catches the symptom late. Population Stability Index measures whether the model score distribution has shifted. By the time PSI crosses 0.20 (the conventional "investigate" threshold), the score distribution has already drifted substantially — which means you have been pricing on a miscalibrated model for the months it took the score distribution to move. PSI is a lagging indicator. You want a leading one.

A/E on live claims is worse still. You need claims to develop before you can compute it, and claims development on a motor book takes months. You are looking at the past when you need to look at the present.

---

## Density ratio estimation: the fix

The density ratio `w(x) = p_target(x) / p_source(x)` is the correct tool. For each risk profile `x`, it asks: how much more (or less) common is this profile in today's book than in the training book? A value of 3.5 for young urban EV drivers means those risks appear 3.5x as frequently now. A value of 0.3 for older rural risks means they appear at one-third the frequency.

You do not need claims data from the new book to estimate this. You only need the feature matrix of risks being scored — what they look like, not what they do. The density ratio is estimated from two unlabelled datasets: source (training distribution) and target (current deployment).

Three methods, each suited to different data:

**CatBoost classifier**: train a binary classifier to distinguish source from target observations, then convert the predicted probabilities to density ratios via `w(x) = P(target | x) / P(source | x)`. Handles postcode district, vehicle code, occupation — the high-cardinality categoricals that are standard in UK personal lines data — natively, without preprocessing. The correct choice for most motor books.

**RuLSIF** (Relative Unconstrained Least-Squares Importance Fitting): a kernel method with a closed-form solution, fast, no hyperparameter grid search. Mathematically clean — Yamada et al. (2013) showed the relative formulation is more numerically stable than the original LSIF. Use it when your feature set is all-continuous: model scores, continuous rating factors, demographic indices. It does not handle high-cardinality categoricals well.

**KLIEP** (Kullback-Leibler Importance Estimation Procedure): explicitly enforces the normalisation constraint `E_source[w(x)] = 1`. Slower than RuLSIF and requires bandwidth selection. Useful primarily as a cross-check: if RuLSIF and KLIEP agree on the verdict, you can trust it.

For a UK motor book, start with `catboost`. If you have a pure-scoring context (re-weighting model scores rather than raw features), `rulsif` is faster.

---

## insurance-covariate-shift in ten lines

```python
import polars as pl
from insurance_covariate_shift import CovariateShiftAdaptor

# X_2025: feature matrix of training book (no labels needed)
# X_2027: feature matrix of current book (no labels needed)
adaptor = CovariateShiftAdaptor(
    method="catboost",
    categorical_cols=[3, 4],  # postcode_district, vehicle_abi_group
    clip_quantile=0.99,
)
adaptor.fit(X_2025, X_2027)

report = adaptor.shift_diagnostic(
    source_label="Direct 2025",
    target_label="Mixed Channel 2027",
)
print(report.verdict)       # MODERATE
print(f"ESS ratio: {report.ess_ratio:.2f}")         # 0.51
print(f"KL divergence: {report.kl_divergence:.3f}") # 0.34 nats
print(report.feature_importance())
# {'driver_age': 0.41, 'postcode_district': 0.31, 'vehicle_age': 0.18, 'ncb': 0.10}
```

The Effective Sample Size ratio is the headline number. It measures what fraction of source observations are effectively contributing to estimates on the target distribution. ESS = 1.0 means the two books look identical. ESS = 0.51 means roughly half your source data is contributing — the other half represents risks that are substantially over- or under-represented in the current book.

The library's thresholds are calibrated to practical action:

- ESS >= 0.60 and KL <= 0.10 nats: **NEGLIGIBLE**. Deploy as-is, standard monitoring.
- ESS 0.30–0.60 or KL 0.10–0.50 nats: **MODERATE**. Apply importance weighting; schedule a retrain review.
- ESS < 0.30 or KL > 0.50 nats: **SEVERE**. Retrain before deploying.

MODERATE is not a disaster. It is actionable. The model is usable but your evaluation metrics need correcting.

---

## Importance-weighted evaluation

The correction is straightforward. Get importance weights for each holdout observation, then re-weight your standard metrics.

```python
import numpy as np

# Holdout drawn from source (2025) book
weights = adaptor.importance_weights(X_holdout)

# Standard A/E: what your monitoring currently reports
ae_standard = y_holdout.sum() / predicted_holdout.sum()

# Importance-weighted A/E: what performance looks like on the 2027 book
ae_weighted = (
    np.average(y_holdout, weights=weights)
    / np.average(predicted_holdout, weights=weights)
)

print(f"Standard A/E:  {ae_standard:.3f}")   # 0.987 — looks fine
print(f"Weighted A/E:  {ae_weighted:.3f}")   # 1.064 — 6.4% underpriced on current mix
```

The weighted A/E does not require a single labelled observation from the 2027 book. It reweights the source holdout to reflect the current distribution. The claims data comes entirely from 2025. The only input from 2027 is the unlabelled feature matrix.

That 6.4% is not an estimate of the model's fundamental error. It is an estimate of how the existing error distributes across the current book, given that the book now over-indexes on segments where the model is less well-calibrated. You were not wrong; you were right in the wrong places.

---

## When to reweight and when to retrain

Importance weighting buys time. It does not replace retraining. The correction is only as good as the density ratio estimate, and the estimate degrades as the shift grows.

**Reweight when:**
- ESS ratio is MODERATE (0.30–0.60)
- The shift is driven by distributional changes in features the model already uses
- Retraining timeline is more than two model refreshes away
- You are correcting evaluation metrics for governance documentation, not changing pricing logic

**Retrain when:**
- ESS ratio is SEVERE (< 0.30): the weights are high-variance and the correction is directionally useful but imprecise
- The shift introduces feature categories the model has never seen (new postcode districts from an acquisition, a vehicle group that launched after training)
- More than 18 months of data from the target distribution are available
- The importance-weighted A/E by decile shows non-uniform bias — systematic over-pricing in some deciles and under-pricing in others suggests the underlying relationships have shifted, not just the mix

The second bullet on retrain is commonly missed. If your book composition shifted because you acquired a broker portfolio that brought in vehicle groups with fewer than 50 training observations, importance weighting cannot fix that. The model has no useful prediction for those risks regardless of the mix correction. KLIEP and RuLSIF will return extremely high weights for those observations, which is the correct statistical answer: those risks are badly underrepresented in your training data. The practical answer is a specialist rate for those segments until you have enough data to train on them.

A rough rule: if the shift is in the distribution of existing feature values, reweight. If the shift introduces new feature support (values the model has never seen), retrain or exclude.

---

## FCA SUP 15.3: what this means for model approval

FCA SUP 15.3 requires notification of material changes to pricing methodology. Using a model trained on a materially different book distribution without adjustment is, in our reading, exactly the kind of change the rule is aimed at. The question is whether the change is material.

The library's `fca_sup153_summary()` gives you the quantitative basis to answer that question:

```python
print(report.fca_sup153_summary())
```

```
COVARIATE SHIFT DIAGNOSTIC REPORT
Source: Direct 2025 (n=48,300)
Target: Mixed Channel 2027 (n=63,100)
Generated: 2028-05-15

VERDICT: MODERATE
ESS ratio: 0.51 (threshold: 0.30 SEVERE, 0.60 NEGLIGIBLE)
KL divergence: 0.34 nats (threshold: 0.50 SEVERE, 0.10 NEGLIGIBLE)

Principal shift drivers:
  driver_age          41% of total shift
  postcode_district   31%
  vehicle_age         18%
  ncb                 10%

Recommended action: Apply importance weighting to evaluation metrics.
Retrain within 6 months or when ESS ratio falls below 0.35.
Monitor driver_age and postcode_district distribution monthly.
```

FCA PRIN 2A.2 requires that models used in pricing are appropriate for the population being scored. MODERATE verdict means the model is still appropriate but requires documented correction. SEVERE means appropriateness is questionable without retraining.

What the report does not write for you is the actuarial judgement: what caused the shift, what the commercial implications are, and what the remediation plan is. The library provides the quantitative basis; your governance note provides the context. That is the right division of labour.

---

## The monitoring cadence

Run the shift diagnostic monthly against a fixed reference window (12 months of training data). The ESS ratio is a single number you can track on a dashboard alongside A/E and Gini.

```python
# Monthly: update X_target_current with latest month's risks
report = adaptor.shift_diagnostic(
    source_label="Training 2025",
    target_label=f"Live Book {current_month}",
)
log_to_monitoring(month=current_month, ess=report.ess_ratio, kl=report.kl_divergence)
```

When ESS drops below 0.60, start reporting importance-weighted metrics alongside standard metrics. When it drops below 0.40, open the retraining conversation with governance. When it hits 0.30, stop the conversation and start the retrain.

The point is not that the model becomes unusable at 0.30. The point is that by 0.30, you have already had months of warning. A team monitoring ESS monthly will never be caught by a composition shift. A team monitoring PSI on score distributions will catch it six months after the model stopped being fit for purpose on the current book.

---

`insurance-covariate-shift` is open source under Apache 2.0 at [github.com/burning-cost/insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift). Requires Python 3.10+, CatBoost 1.2+, NumPy 1.24+.

**Related posts:**
- [Your Model Was Trained on Last Year's Book](/2028/02/15/covariate-shift-detection-book-mix-changes/) — detection, ESS ratio, conformal intervals under shift, and FCA filings
- [Model Validation Is a Checklist, Not a Test](/2027/09/15/model-validation-pra-ss123/) — PRA SS1/23 model validation and what a defensible sign-off process looks like
- [Transfer Learning for Thin Segments](/2027/07/15/transfer-learning-for-thin-segments/) — when the target book has sparse data, importance weighting pairs with transfer learning
