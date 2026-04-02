---
layout: post
title: "Testing Whether Your Challenger Model Actually Discriminates Better"
date: 2026-04-02
categories: [insurance-pricing, model-monitoring]
tags: [calibration, discrimination, score-decomposition, hypothesis-testing, mincer-zarnowitz, hac, glm, gbm, insurance-monitoring, actuarial, pra-ss1-23, fca-consumer-duty, arXiv-2603.04275]
description: "Dimitriadis & Puke (2026) solve how to attach p-values to the MCB/DSC/UNC score decomposition. insurance-monitoring v1.2.0 implements it. When your GBM beats your GLM on deviance, this tells you whether that gap is in calibration, discrimination, or noise."
author: burning-cost
---

A challenger GBM beats the champion GLM by 0.8% on Poisson deviance across a 120,000-policy motor holdout. The team debates deployment. The model owner says the improvement is real. The validation actuary says it might be sampling noise. Nobody can say with certainty whether the GBM is actually better at ranking risks, or whether it is simply better calibrated, or whether those 0.8 percentage points are statistically distinguishable from zero.

This is not a hypothetical. It is the standard state of model validation in UK personal lines pricing as of 2026.

PRA SS1/23 requires firms to demonstrate that their pricing models are fit for purpose — adequate calibration and adequate discrimination — and to show that deterioration in either is detected before it affects pricing adequacy. Most monitoring frameworks do neither rigorously. They report Gini, A/E, and deviance as point estimates with no standard errors and no decomposition of which part of model behaviour has changed.

[insurance-monitoring v1.2.0](https://pypi.org/project/insurance-monitoring/) ships `ScoreDecompositionTest`, based on Dimitriadis & Puke (arXiv:2603.04275, March 2026). It decomposes any proper scoring rule into Miscalibration and Discrimination components and attaches asymptotic p-values to each.

---

## The decomposition

For any proper scoring rule S, a forecast F, and observed outcomes y:

```
S(F, y) = MCB(F, y) + UNC(y) − DSC(F, y)
```

Three components:

**UNC — Uncertainty.** The score you would achieve by predicting the grand mean on every risk. It is a property of the data, not the model.

**MCB — Miscalibration.** How much worse than a linearly recalibrated version of your own forecast are you? A perfectly calibrated model has MCB = 0. Non-zero MCB means a simple linear adjustment — changing the overall mean level or applying a slope correction — would improve the score without touching the underlying risk ranking.

**DSC — Discrimination.** How much better than the grand-mean forecast is your recalibrated forecast? This is the pure ranking signal. A model that separates high-risk from low-risk policies well has high DSC. Random relativities give DSC near zero.

The identity is exact, not approximate. It holds for MSE, Poisson deviance, quantile scores, and interval scores.

The recalibration step is a Mincer-Zarnowitz (MZ) regression: regress observed outcomes y on your forecast F. For mean scores (MSE, Poisson deviance), this is OLS — the classical MZ regression from econometrics (1969). The recalibrated forecast is F* = aF + b. An MZ slope near 1 and intercept near 0 means your model is well-calibrated. A slope significantly below 1 means your model's relativities are compressed — it does not move enough across risk segments. MCB quantifies the cost of that miscalibration in score units.

For quantile scores (pinball loss, reserve models targeting a specific loss percentile), the MZ step uses quantile regression rather than OLS. This is the genuine technical contribution of Dimitriadis & Puke: extending the inference to non-smooth loss functions via the Galvao-Yoon (2024) HAC covariance estimator. Standard errors for quantile regression MZ — properly corrected for serial dependence in policy-period data — did not exist before this paper.

---

## The GLM versus GBM question

Returning to the opening scenario: GLM champion versus GBM challenger, 0.8% deviance improvement, 120,000 policies.

The decomposition for a realistic motor frequency book might look like this:

| Component | GLM | GBM | Difference | p-value |
|-----------|-----|-----|------------|---------|
| Score (Poisson deviance) | 0.4821 | 0.4782 | 0.0039 | 0.003 |
| MCB | 0.0043 | 0.0008 | 0.0035 | 0.001 |
| DSC | 0.1187 | 0.1191 | −0.0004 | 0.61 |
| UNC | 0.5965 | 0.5965 | — | — |

The GBM's total score is significantly better (p = 0.003). But the decomposition shows the entire gain sits in MCB: the GBM is better calibrated by 0.0035 deviance units (p = 0.001), while its discrimination is statistically indistinguishable from the GLM's (p = 0.61).

This matters for the deployment decision. The GBM does not rank risks more accurately. It is better calibrated — probably because its more flexible functional form fits the training data mean more closely across age and vehicle bands where the GLM uses coarser bins. You could achieve a comparable MCB improvement by recalibrating the existing GLM: run the MZ regression, apply the slope and intercept corrections as rating adjustments. That takes a day and does not require deploying a new model, a new IT build, and a new governance sign-off.

This is not an argument against GBMs. It is an argument for knowing what you are buying when you deploy one.

---

## Why Diebold-Mariano misses this

The standard two-sample forecast comparison is the Diebold-Mariano test: is S(A, y) − S(B, y) significantly non-zero? It is a single test on the total score difference. It cannot say which component drives it.

Dimitriadis & Puke prove that the component tests have strictly higher power than DM when models differ in only one component. If your challenger improves calibration but discrimination is identical, DM mixes the calibration signal with the noise from the discrimination comparison. The MCB component test isolates the calibration effect. In the table above, DM correctly finds a significant result at p = 0.003; without the decomposition, you would not know the significant effect is entirely in calibration and the discrimination comparison is noise at p = 0.61.

---

## Using ScoreDecompositionTest

```python
from insurance_monitoring.calibration import ScoreDecompositionTest

sdi = ScoreDecompositionTest(score_type='mse', exposure=vehicle_years)

# Single model
result = sdi.fit_single(y_test, y_pred_glm)
print(f"MCB: {result.miscalibration:.4f} (p={result.mcb_pvalue:.3f})")
print(f"DSC: {result.discrimination:.4f} (p={result.dsc_pvalue:.3f})")
print(f"MZ slope: {result.recalib_slope:.3f}  intercept: {result.recalib_intercept:.4f}")

# Champion vs challenger
comp = sdi.fit_two(y_test, y_pred_glm, y_pred_gbm)
print(f"ΔMCB: {comp.delta_mcb:.4f}  p={comp.delta_mcb_pvalue:.3f}")
print(f"ΔDSC: {comp.delta_dsc:.4f}  p={comp.delta_dsc_pvalue:.3f}")
```

Pass `exposure=vehicle_years` for frequency models — the weighted MZ regression handles variable policy durations. For severity or reserve models, `score_type='quantile'` with an `alpha` parameter covers pinball loss at any quantile level.

The `summary()` method produces a plain-text report for governance packs. The `plot()` method draws a stacked MCB/DSC/UNC bar chart with p-value annotations.

---

## How this fits the monitoring stack

insurance-monitoring already runs PSI/CSI on input feature distributions, `check_auto_calibration()` on cohort-level A/E with bootstrap MCB, `murphy_decomposition()` for MCB/DSC/UNC point estimates via isotonic recalibration, and `GiniDriftTest` for rank-based drift detection.

`ScoreDecompositionTest` adds inference to the MCB and DSC components. The existing `murphy_decomposition()` uses PAVA (non-parametric isotonic recalibration), which captures any monotone miscalibration. The new class uses linear MZ recalibration — more conservative, understates MCB for models with non-linear miscalibration, but yields a tractable asymptotic distribution. Run both: `murphy_decomposition()` for the complete point-estimate picture, `ScoreDecompositionTest` for the p-values.

The governance chain that ties this together: `GiniDriftTest` tells you the Gini has fallen by 0.4 percentage points since deployment. `ScoreDecompositionTest` tells you whether that fall is statistically significant and whether it is in calibration (a rate review fixes it) or discrimination (requires a refit). A PRA reviewer or an internal CRO should expect a firm to be able to answer both questions. Most currently cannot.

---

## Installation

```
uv add insurance-monitoring>=1.2.0
```

The `model-diagnostics` package (PyPI, v1.4.3, December 2025) provides the MCB/DSC/UNC point estimates without inference. The R `SDI` package by the paper's authors implements the full test suite. Our implementation follows their algorithm with insurance-specific exposure weighting added.

**Paper:** [arXiv:2603.04275](https://arxiv.org/abs/2603.04275) — Dimitriadis & Puke (2026) | **Library:** [insurance-monitoring on PyPI](https://pypi.org/project/insurance-monitoring/)
