---
layout: post
title: "Is Your Model Miscalibrated, or Is That Just Noise?"
date: 2026-04-02
categories: [model-monitoring, insurance-pricing]
tags: [calibration, discrimination, score-decomposition, hypothesis-testing, mincer-zarnowitz, hac, p-value, insurance-monitoring, actuarial, arXiv-2603.04275, mcb, dsc, unc, scoring-rules, python]
description: "Most pricing teams eyeball calibration plots or track PSI. Neither tells you whether observed miscalibration is statistically significant. ScoreDecompositionTest in insurance-monitoring v1.2.0 gives you a p-value for MCB and DSC, based on Dimitriadis & Puke (arXiv:2603.04275, March 2026)."
author: burning-cost
---

Your motor frequency model shows an A/E ratio of 1.04 on the latest quarterly monitoring window. Last quarter it was 1.01. Is that miscalibration, or is it noise in 18,000 policies?

Most pricing teams cannot answer that question. They run calibration plots, track PSI, compute A/E ratios — and then exercise judgement about whether the numbers are "material". The judgement is necessary because none of those tools attach a standard error to anything. There is no test. There is no p-value. There is no formal decision rule.

[insurance-monitoring v1.2.0](https://pypi.org/project/insurance-monitoring/) ships `ScoreDecompositionTest`, which fixes this. It decomposes any proper scoring rule into three components — Miscalibration (MCB), Discrimination (DSC), and Uncertainty (UNC) — and attaches asymptotically valid p-values to MCB and DSC. The method is Dimitriadis & Puke (arXiv:2603.04275, March 2026), with our addition of insurance-specific exposure weighting.

---

## The decomposition

For any consistent scoring function S, any forecast F, and observed outcomes y, the following identity holds exactly:

```
S(F, y) = MCB + UNC − DSC
```

**UNC — Uncertainty.** The score you get by predicting the portfolio mean on every risk. It reflects the difficulty of the data. It cannot be improved by any model — it is determined by the claim frequency distribution alone.

**DSC — Discrimination.** How much better than the portfolio mean is your optimally recalibrated forecast? This is the pure ranking signal: the model's ability to separate cheap risks from expensive ones. A model that correctly identifies that a 23-year-old male on a hot hatch is ten times riskier than a 55-year-old woman in a Civic has high DSC. A model that correctly identifies the overall mean but has no view on who is above or below it has DSC near zero.

**MCB — Miscalibration.** How much worse is your actual forecast than your recalibrated forecast? A perfectly calibrated model has MCB = 0. Non-zero MCB means a simple linear adjustment — a slope or intercept correction — would improve the score. The recalibration is a Mincer-Zarnowitz regression: regress observed y on your forecast F to get F* = aF + b. MCB = S(F, y) − S(F*, y).

The MZ slope is particularly diagnostic. A slope below 1 means your relativities are compressed — the model does not spread far enough across risk segments. A slope above 1 means the spread is too wide. An intercept materially different from zero means there is a systematic level shift. These are actionable: a slope near 0.85 tells a pricing actuary that a 15% expansion of the rating relativities is warranted without any model refit.

---

## Why A/E is not the same thing

A/E tells you whether the model is globally balanced. It is the ratio of actual claims to expected claims at portfolio level. A/E = 1.04 means the model is slightly under-priced on average.

MCB is more informative in two ways. First, it uses the full scoring framework — it measures the cost of miscalibration in deviance units, which is directly linked to pricing adequacy. Second, it now comes with a standard error and a p-value.

An A/E of 1.04 on 18,000 policies might be entirely consistent with a well-calibrated model and ordinary Poisson sampling variation. Or it might represent a real and statistically significant pricing gap. Without a hypothesis test, you cannot know which. With `ScoreDecompositionTest`, you can.

---

## Using it in a monitoring workflow

```python
import numpy as np
from insurance_monitoring.calibration import ScoreDecompositionTest

# Quarterly monitoring window: 18,000 policies
# y: observed claim counts / exposure
# y_pred: model predictions (pure premium or frequency)
# vehicle_years: exposure for each policy

sdi = ScoreDecompositionTest(score_type='mse', exposure=vehicle_years)
result = sdi.fit_single(y, y_pred)

print(result.summary())
```

Output:

```
Score decomposition (n=18,000)
  Score (S):        0.003841
  MCB:              0.000092  (SE=0.000041, p=0.024)
  DSC:              0.001204  (SE=0.000089, p=0.000)
  UNC:              0.004953

  MZ regression:    y_hat* = 0.0019 + 0.923 * y_hat

  MCB=0 test:   FAIL (miscalibrated)  p=0.024
  DSC=0 test:   PASS (has skill)      p=0.000
```

That tells you three things at once. MCB is statistically significant at 5% — the observed calibration drift is real, not noise. The MZ slope of 0.923 tells you the relativities need expanding by about 8%. And DSC is large and highly significant — the model's ranking ability is intact. This is a recalibrate situation, not a refit.

Compare that to what a plain A/E ratio gives you: 1.04. No standard error. No decomposition. No indication of whether the ranking signal has degraded. No decision.

---

## The monitoring workflow with p-values

The decision logic becomes mechanical once you have the tests:

| MCB p-value | DSC p-value | Decision |
|-------------|-------------|----------|
| > 0.05 | < 0.05 | Monitor: calibration acceptable, discrimination present |
| < 0.05 | < 0.05 | Recalibrate: apply MZ correction to rating factors |
| > 0.05 | > 0.05 | Investigate: model has no ranking skill (data issue?) |
| < 0.05 | > 0.05 | Refit: miscalibrated and no discrimination to preserve |

The last row is the worst case. MCB is significant (your prices are systematically wrong) and DSC is not (your model does not meaningfully rank risks). A rate review will not fix it. The model needs rebuilding.

This maps directly onto the `REDEPLOY / RECALIBRATE / REFIT` framework in `ModelMonitor` (insurance-monitoring v1.0.0). `ScoreDecompositionTest` provides the inference layer underneath that decision tree.

---

## HAC standard errors and temporal dependence

Insurance monitoring data is not iid. Policies in the same quarter experience the same weather patterns, the same economic conditions, the same changes in claims inflation. Score contributions for claims in the same period are correlated.

Standard OLS standard errors would understate the true uncertainty — they would make your MCB p-value look more significant than it is. `ScoreDecompositionTest` uses Newey-West (Bartlett kernel) HAC standard errors, which correct for this dependence. The lag order is selected automatically using the Newey-West (1994) rule: `floor(4*(n/100)^(2/9))`. For a typical quarterly window of 18,000 policies, that gives around 7 lags.

For portfolios with strong annual seasonality — a boat insurance book where all policies renew in spring, for example — you can override with `hac_lags=12` to ensure seasonal autocorrelation is absorbed.

---

## Quantile scores and reserve models

The default `score_type='mse'` covers frequency and pure premium models benchmarked against squared error. For reserve models targeting a specific loss percentile, use `score_type='quantile'` with the appropriate `alpha`.

```python
# Reserve model targeting 75th percentile of claims distribution
sdi_q = ScoreDecompositionTest(
    score_type='quantile',
    alpha=0.75,
    exposure=vehicle_years
)
result_q = sdi_q.fit_single(y_reserve, y_pred_reserve)
```

For quantile scores, the MZ step uses quantile regression rather than OLS. This is the technical contribution of Dimitriadis & Puke that did not previously exist: asymptotically valid standard errors for the MCB and DSC components of a pinball loss decomposition, using the Galvao-Yoon (2024) HAC covariance estimator for non-smooth loss functions. Before March 2026, there was no way to formally test miscalibration of a reserve model at a specific quantile level. Now there is.

---

## What this does not replace

`ScoreDecompositionTest` uses linear MZ recalibration. The existing `murphy_decomposition()` in insurance-monitoring uses isotonic recalibration (PAVA) — which captures any monotone miscalibration, not just linear. For a GBM with non-linear miscalibration across age bands, PAVA-based MCB will be larger. Linear MZ understates it.

Run both. `murphy_decomposition()` gives you the point estimate with the most complete recalibration. `ScoreDecompositionTest` gives you the p-value. The two are complementary, not competing.

---

## Installation

```
uv add insurance-monitoring>=1.2.0
```

The class is available at `insurance_monitoring.calibration.ScoreDecompositionTest`.

**Paper:** [arXiv:2603.04275](https://arxiv.org/abs/2603.04275) — Timo Dimitriadis & Marius Puke (Heidelberg University, March 2026) | **Library:** [insurance-monitoring on PyPI](https://pypi.org/project/insurance-monitoring/)
