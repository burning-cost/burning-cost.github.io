---
layout: post
title: "Competing Risks Calibration: Why Your Fine-Gray Validation Is Wrong"
date: 2026-03-31
categories: [survival-analysis, model-validation]
tags: [competing-risks, calibration, Fine-Gray, D-calibration, Aalen-Johansen, survival-analysis, lapse-modelling, retention-modelling, GIPP, insurance-survival, Alberge-2026, probability-integral-transform, recalibration]
description: "D-calibration and ICI are mathematically invalid for competing-risks models. If F_k(inf|x) < 1 — which is always true for lapse, claim, and MTA competing causes — the probability integral transform does not hold. Two new metrics from Alberge et al. (AISTATS 2026, arXiv:2602.00194) fix this, and are now in insurance-survival v0.3.9."
author: burning-cost
---

If you are validating a Fine-Gray retention model and your calibration check involves D-calibration or an ICI-style plot, you are running a test that does not work. The mathematics is wrong, not just approximate. This post explains why, and what to use instead.

---

## The problem with F_k(infinity) < 1

D-calibration exploits the probability integral transform. For a single-event survival model, if T* is the true event time for individual i and F_hat(t|x_i) is the model's predicted CDF, then under correct calibration:

```
F_hat(T*_i | x_i) ~ Uniform(0, 1)
```

This holds because for a continuous distribution, the CDF evaluated at a draw from that distribution is uniform. D-calibration tests whether the empirical distribution of {F_hat(T*_i | x_i)} across observed events is close to uniform. If the model is well-calibrated, it should be.

For competing risks, this breaks immediately. Consider a retention model with four causes: lapse (k=1), claim-triggered cancellation (k=2), mid-term adjustment (k=3), natural expiry (k=4). For any individual cause k:

```
F_k(infinity | x) = P(event k occurs | X = x)  !=  1
```

A high-risk lapse segment might have F_lapse(inf|x) = 0.45. When you evaluate F_lapse(T*|x) for a policy that actually lapses, the result lives on the support [0, 0.45], not [0, 1]. D-calibration compares these truncated values to Uniform(0,1) and produces a statistic that reflects the truncation rather than the calibration. The failure mode is insidious: a correctly calibrated Fine-Gray model will look miscalibrated, while a model that systematically overestimates lapse probability can look well-calibrated.

ICI has the same problem. The standard construction groups predictions into quantile bins and compares mean predicted CIF against the Aalen-Johansen estimate within each bin. But in bins with high predicted lapse risk, F_lapse(inf|x_i) might average 0.60. In low-risk bins, it might average 0.20. The AJ CIF within each bin converges to a different terminal value, making cross-bin comparisons incoherent. A calibration curve can show apparent perfect calibration on a model with systematic probability bias.

---

## Why this matters for UK pricing

Fine-Gray subdistribution hazard models have been increasingly standard in UK retention modelling since around 2021 — any decent lapse model now treats claim, MTA, and natural expiry as competing causes rather than treating them all as censoring. This is the right modelling choice. The calibration tooling has not kept up.

The consequences are practical. GIPP renewal pricing requires that premium adequacy can be demonstrated over the full policy lifetime. If your lapse model is systematically miscalibrated, your expected renewal population is wrong, your CLV calculations are wrong, and your expected claims development from the retained book is wrong. For lapse-sensitive products — income protection, long-term disability — the reserve adequacy problem is direct.

The current `calibration_curve()` in insurance-survival's competing_risks module gives a useful picture at a single time point but has no formal test statistic, is not time-integrated, and is not normalised for the terminal probability issue. A pricing actuary using it as their primary calibration validation pass is producing a model governance artefact that cannot detect systematic probability bias.

---

## Two proper calibration metrics

Alberge, Aghbalou, Sabourin, Mozharovskyi and d'Alché-Buc (arXiv:2602.00194, AISTATS 2026) prove that both D-calibration and ICI are improper for competing risks, and provide two replacements with formal properness guarantees.

### cal_K^alpha — time-integrated marginal calibration error

For cause k, compute the time-varying gap between the mean predicted CIF and the Aalen-Johansen marginal estimate:

```
cal_k(t) = |E_X[F_hat_k(t|X)] - F_k^AJ(t)|
```

Then integrate over the full observation window:

```
cal_K^alpha = (1/T_max) * integral_0^{T_max} cal_k(t)^alpha dt
```

With alpha=2 this is the L2 mean calibration error over the time axis. A perfectly calibrated model returns 0.0. The plug-in estimator uses the AJ CIF on a held-out calibration set as the reference, and Proposition 14 in the paper proves consistency.

The practical interpretation is direct: if cal_K^2 = 0.03, the model is off by an average of 0.03 CIF units over the observation window, integrated across time. In a retention model where typical CIF values run from 0 to 0.45, an error of 0.03 is material.

### CR D-calibration — normalised PIT test

The correct generalisation of D-calibration for competing risks. Instead of testing whether F_k(T*|X) is uniform, test whether the normalised PIT:

```
PIT_i = F_hat_k(T*_i | X_i) / F_hat_k(t_max | X_i)
```

is uniform over [0, 1]. The normalisation by F_hat_k(t_max|x) — the CIF asymptote — maps each subject's value to the correct support regardless of their terminal probability. Subjects in high-lapse-risk segments and low-lapse-risk segments are now comparable.

Theorem 8 in the paper proves that CR D-cal = 0 if and only if the model is marginally calibrated. Theorem 15 shows that cal_K^alpha calibration implies CR D-calibration.

The test is a chi-squared goodness-of-fit test on the normalised PIT values against a uniform distribution over n_bins bins, using only subjects who actually experienced cause k.

---

## AJ recalibration: fix without retraining

The paper also describes post-hoc recalibration that corrects miscalibration while preserving the C-index exactly.

The additive correction is:

```
delta(t) = F_k^AJ(t; D_cal) - mean(F_hat_k(t | X_i))
F_hat_k_recal(t | x) = clip(F_hat_k(t | x) + delta(t), 0, 1)
```

Fit delta(t) on a held-out calibration set (not the training set). At test time, add the same time-varying correction to every individual. Because delta(t) depends only on t, not on x, the relative ordering of predictions at each time is unchanged — the C-index is preserved exactly (Theorem 17).

For UK pricing: you need a calibration holdout of roughly 500 policies per competing cause to get stable AJ estimates. For a four-cause retention model that means around 2,000 held-out policies. If your training set is smaller than that, consider a temporal holdout: train on years 1–4, calibrate on year 5, score on year 6.

---

## Using the implementation

These metrics are in `insurance-survival` v0.3.9:

```python
pip install "insurance-survival>=0.3.9"
```

```python
import numpy as np
from insurance_survival.competing_risks import (
    compute_cal_k_alpha,
    AJRecalibrator,
    CRDCalibration,
)

# predicted_cif: shape (n_samples, n_times)
# event_times: observed time, shape (n_samples,)
# event_indicators: 0=censored, 1=lapse, 2=claim, 3=MTA
# times: evaluation grid, shape (n_times,)

# 1. Measure miscalibration on a holdout set
cal_error = compute_cal_k_alpha(
    predicted_cif=cif_cal,
    times=times,
    event_times=T_cal,
    event_indicators=E_cal,
    event_k=1,   # lapse
    alpha=2,
)
print(f"cal_K^2 (lapse) = {cal_error:.4f}")
# e.g. 0.0312 — model under-predicts lapse CIF by 0.031 on average

# 2. CR D-calibration test
cr_dcal = CRDCalibration(n_bins=10)
cr_dcal.compute(
    predicted_cif=cif_cal,
    times=times,
    event_times=T_cal,
    event_indicators=E_cal,
    event_k=1,
)
print(cr_dcal.summary())
# {'statistic': 24.3, 'p_value': 0.004, 'n_events': 612, ...}
# p < 0.05: evidence of miscalibration

# 3. Fit the recalibrator on the calibration set
recal = AJRecalibrator()
recal.fit(
    predicted_cif=cif_cal,
    times=times,
    event_times=T_cal,
    event_indicators=E_cal,
    event_k=1,
)

# Inspect the time-varying correction
print(f"Mean correction: {recal.delta_.mean():.4f}")

# 4. Apply to test predictions — C-index unchanged
cif_test_recal = recal.transform(predicted_cif=cif_test, times=times)

# 5. Verify: cal_K^2 should be near zero on the recalibrated predictions
cal_after = compute_cal_k_alpha(
    predicted_cif=cif_test_recal,
    times=times,
    event_times=T_test,
    event_indicators=E_test,
    event_k=1,
)
print(f"cal_K^2 after recalibration = {cal_after:.4f}")
# e.g. 0.0021 — essentially zero
```

The `delta_` attribute on a fitted `AJRecalibrator` is a 1-D array over the time grid — the full correction curve. Inspect it before deploying. If it is not approximately monotone and well-behaved, your calibration holdout may be too small or your AJ estimate is noisy.

---

## What to check in the validation pack

For a Fine-Gray retention model, the validation pack now needs:

1. **C-index** (discrimination — existing, correct as-is)
2. **cal_K^2** per competing cause on a held-out test set — report the raw number with a threshold (we use 0.05 as a flag for review)
3. **CR D-calibration** per cause — report the chi-squared statistic and p-value, confirm n_events per cause is adequate for the test (at least 50 events per bin, so 500 events for the default 10-bin test)
4. **delta(t) plot** from the `AJRecalibrator` — this is the bias function and should go into the validation pack alongside the corrected predictions
5. **Post-recalibration cal_K^2** to confirm the correction worked

The Brier score and IBS are not affected by the competing-risks calibration issue — they are proper scoring rules and give the right answer for Fine-Gray models. Keep them in the pack; they measure something slightly different (overall predictive accuracy) from the calibration metrics above.

One thing that does not change: the underlying Fine-Gray model does not need to be retrained. AJ recalibration is a post-hoc correction applied at scoring time. You can retrofit it to models already in production.

---

**Paper:** Alberge, J., Aghbalou, A., Sabourin, A., Mozharovskyi, P. & d'Alché-Buc, F. (2026). 'Calibration of Survival Models.' arXiv:2602.00194. AISTATS 2026, Tangier.

**Library:** [github.com/burning-cost/insurance-survival](https://github.com/burning-cost/insurance-survival) — `pip install "insurance-survival>=0.3.9"`.
