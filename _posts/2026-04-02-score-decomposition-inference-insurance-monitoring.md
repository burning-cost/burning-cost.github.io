---
layout: post
title: "Your Model Comparison Has No Error Bars"
date: 2026-04-02
categories: [insurance-pricing, model-monitoring]
tags: [calibration, discrimination, score-decomposition, hypothesis-testing, mincer-zarnowitz, hac, gini, insurance-monitoring, actuarial, fca-consumer-duty, arXiv-2603.04275]
description: "insurance-monitoring v1.2.0 ships ScoreDecompositionTest, based on Dimitriadis & Puke (2026). Decomposes any scoring rule into Miscalibration and Discrimination components with asymptotic p-values. If your challenger model beats champion by 0.3% Gini, this tells you whether that gap is statistically significant — and where it comes from."
author: burning-cost
---

Most pricing teams run this comparison at least once per review cycle. The actuary runs the champion and challenger through holdout validation. The challenger beats the champion on Gini by 0.3 percentage points and on deviance by 0.8%. The team debates whether to deploy. Nobody asks whether the difference is statistically significant, because there is no standard tooling to answer the question. The challenger goes live.

Sometimes that is fine. Often it is not, and it becomes apparent only after the next monitoring cycle when performance reverts.

There is a second problem that is subtler and worse. Even when teams do ask whether a model is better, the question they actually need to answer is different. A Gini improvement could mean the challenger ranks risks more accurately. It could equally mean the challenger is better calibrated — its mean predictions are closer to observed loss ratios — and ranking ability is identical. These two improvements have entirely different remediation paths. If the problem is calibration, you can fix the existing model without refitting. If the problem is discrimination, you cannot.

[insurance-monitoring v1.2.0](https://pypi.org/project/insurance-monitoring/) ships `ScoreDecompositionTest`, which answers both questions with p-values.

---

## The decomposition

For any proper scoring rule S, a forecast F, and observed outcomes y, the Dimitriadis & Puke (arXiv:2603.04275, March 2026) decomposition gives:

```
S(F, y) = MCB(F, y) + UNC(y) − DSC(F, y)
```

Three components:

**UNC — Uncertainty.** The score you would achieve predicting the grand mean every time. It measures the inherent difficulty of the prediction problem, not the model. You cannot improve it.

**MCB — Miscalibration.** How much worse than a linearly recalibrated version of your own forecast are you? A perfectly calibrated model has MCB = 0. Non-zero MCB means your predictions are biased in a way that a simple linear adjustment of your existing scores would fix.

**DSC — Discrimination.** How much better than predicting the grand mean is your linearly recalibrated forecast? This is the pure ranking signal — the ability to separate risks, independent of any calibration bias.

The recalibration step is a Mincer-Zarnowitz regression: regress y on F (OLS for mean scores, quantile regression for quantile scores). The recalibrated forecast F* = aF + b is the best linear re-weighting of your model's output. MCB measures how far your raw forecast is from F*. DSC measures how far F* is from the trivial forecast.

This identity holds for MSE, MAE, Poisson deviance, quantile scores, and interval scores. It is not GLM-specific. It applies to any scoring rule you already use for model evaluation.

The paper's contribution over prior decompositions (Murphy 1977, Bröcker 2009, Wüthrich 2025) is the inference layer: asymptotic distributions for MCB and DSC under HAC standard errors, so you can test H₀: MCB = 0 and H₀: DSC = DSC_baseline with proper p-values. Temporal dependence between policy periods is handled automatically by the HAC covariance estimator.

---

## Why the two-sample test matters

The single-model tests are useful but the two-sample comparison is what pricing teams actually need.

Given champion model A and challenger model B evaluated on the same holdout set, the standard question is whether S(B, y) < S(A, y) — is B's total score better. The Diebold-Mariano test answers this. What it cannot tell you is whether B is better because it has lower MCB, higher DSC, or both.

Dimitriadis & Puke prove that the two-sample component tests have strictly higher power than DM in scenarios where models differ in only one component. If your challenger improves calibration but discrimination is identical, DM tests the combined signal and may find nothing. The component test on MCB will find it.

For pricing governance, this distinction matters beyond statistical power. "The challenger is better because it ranks risks more accurately" is a different and more specific claim than "the challenger is better." The latter is a number. The former is evidence about what the model is actually doing.

---

## Using ScoreDecompositionTest

```python
from insurance_monitoring.calibration import ScoreDecompositionTest

sdi = ScoreDecompositionTest(score_type='mse')
result = sdi.fit_single(y_test, y_pred)
print(f"MCB: {result.miscalibration:.4f} (p={result.mcb_pvalue:.3f})")
print(f"DSC: {result.discrimination:.4f} (p={result.dsc_pvalue:.3f})")

# Two-sample comparison
comp = sdi.fit_two(y_test, y_pred_champion, y_pred_challenger)
print(f"ΔMCB p-value: {comp.delta_mcb_pvalue:.3f}")
print(f"ΔDSC p-value: {comp.delta_dsc_pvalue:.3f}")
```

`score_type` accepts `'mse'`, `'mae'`, and `'quantile'` (with an additional `alpha` parameter for the quantile level). For frequency models with exposure weighting, pass `exposure=vehicle_years` as a keyword argument.

Interpreting the two-sample output: a small `delta_mcb_pvalue` means the challenger's miscalibration is statistically distinguishable from the champion's. A small `delta_dsc_pvalue` means the challenger's discrimination ability is statistically distinguishable. You can have both, neither, or one.

The case to look out for: challenger shows lower total score, `delta_mcb_pvalue` is highly significant, `delta_dsc_pvalue` is 0.6. The challenger is better calibrated. It is not better at ranking risks. If you are deploying it for segmentation purposes, you are buying nothing you could not get by recalibrating the champion.

For quantile scores, the underlying Mincer-Zarnowitz regression uses quantile regression rather than OLS, with HAC standard errors via the Galvao & Yoon (2024) method for non-smooth loss functions. This makes the tool applicable to reserve models targeting a specific quantile of the loss distribution, not just mean predictions.

---

## The FCA angle

Consumer Duty requires insurers to demonstrate that pricing models produce fair outcomes for customers. The regulator's expectation is not that models be perfect but that firms have governance processes capable of detecting when they are not.

The current state in most firms is: model monitoring reports point estimates of Gini, A/E ratio, and occasionally deviance. There are no standard errors. The monitoring team reports "Gini fell from 42.1% to 41.7%" with no assessment of whether that movement is within sampling variation or a genuine signal. Decisions on whether to recalibrate or refit are based on thresholds that were set by experience rather than statistical reasoning.

`ScoreDecompositionTest` gives monitoring teams a defensible answer to "is this decline statistically significant?" and "is the problem in our calibration, our risk ranking, or both?" These are questions any competent regulator should expect a firm to be able to answer. At present, most cannot.

The class integrates with `insurance-monitoring`'s existing `DriftReport`. If you are already running `GiniDriftTest` and `CalibrationCUSUM`, `ScoreDecompositionTest` adds the component-level inference that those tools lack. `GiniDriftTest` tells you the Gini has moved. `ScoreDecompositionTest` tells you which part of the model's behaviour has changed and whether that change is significant.

---

## A note on the existing Python ecosystem

Before building this we checked what already existed. The `model-diagnostics` package (PyPI, v1.4.3, December 2025) implements the MCB/DSC/UNC point-estimate decomposition for mean and quantile forecasts. It is well-implemented and we would recommend it for exploratory analysis.

What it does not have: hypothesis tests, HAC standard errors, or two-sample comparison. scikit-learn issue #23767, open since 2022, tracks a request for calibration decomposition with no implementation. The R `SDI` package by the paper's authors implements everything — our Python implementation follows their algorithm closely.

The other gap is insurance-specific: exposure weighting for frequency models and integration with Poisson deviance scoring. `ScoreDecompositionTest` handles these natively because insurance-monitoring already carries the exposure weighting infrastructure from its `murphy_decomposition()` function.

---

## Installation

```
uv add insurance-monitoring>=1.2.0
```

**Paper:** [arXiv:2603.04275](https://arxiv.org/abs/2603.04275) — Dimitriadis & Puke (2026) | **Library:** [insurance-monitoring on PyPI](https://pypi.org/project/insurance-monitoring/)
