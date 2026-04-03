---
layout: post
title: "Reserve Models Have a Calibration Problem Nobody Is Testing"
date: 2026-04-03
categories: [reserving, model-monitoring]
tags: [score-decomposition, calibration, discrimination, quantile-score, pinball-loss, mincer-zarnowitz, reserving, solvency-ii, glm, gbm, insurance-monitoring, mcb, dsc, quantile-regression, hac, arXiv-2603.04275, Dimitriadis, Puke, actuarial, reserve-adequacy]
description: "Frequency model monitoring has a growing toolkit. Reserve models — which target specific quantiles of the loss distribution, not the mean — have almost nothing. ScoreDecompositionTest in insurance-monitoring v1.2.0 fills the gap with HAC-robust p-values on miscalibration and discrimination for any quantile target."
author: burning-cost
math: true
---

When a pricing actuary monitors a frequency model, they have a reasonable toolkit: A/E ratios, calibration plots, Gini drift tests, and now `ScoreDecompositionTest` with its p-values on MCB and DSC. The tooling was built for mean forecasts, and the statistics work cleanly.

Reserve models are different. A reserving model does not typically target the mean of the claims distribution — it targets a specific quantile. In a Solvency II context, the one-year value-at-risk at 99.5% is the regulatory headline number. In practice, a UK GI reserving team might be monitoring whether their stochastic reserve model is correctly estimating the 75th or 90th percentile of outstanding claims — the level that informs the risk margin and the internal model capital requirement.

For that question — is this reserve model well-calibrated at the quantile level it is targeting? — almost no statistical tooling exists. A/E ratios are mean comparisons. Gini coefficients measure discrimination of a mean model. Reserve triangles and bootstrapping frameworks tell you about process variance but not about whether your conditional quantile estimates are systematically biased.

`ScoreDecompositionTest` in [insurance-monitoring v1.2.0](https://pypi.org/project/insurance-monitoring/) covers this case through the quantile score path. The underlying methodology is Dimitriadis & Puke (arXiv:2603.04275, March 2026). The result is a formal hypothesis test for whether a reserve model is miscalibrated at its target quantile — with a p-value, not a governance judgement call.

---

## Why mean-based tools do not work for reserve quantiles

A model targeting the 80th percentile of the claims distribution has a different failure mode from a model targeting the mean. Mean miscalibration shows up as a systematic level shift: the model says £1,200 and the average observed loss is £1,300, and a Poisson A/E ratio catches this immediately.

Quantile miscalibration is more subtle. A reserve model targeting the 80th percentile is well-calibrated if approximately 20% of observed losses exceed the model's prediction. That is the coverage property. If 28% of observations exceed the model, the model is understating the quantile — not understating the mean. These are different problems. A well-calibrated mean forecast can be a badly calibrated 80th percentile forecast, and vice versa.

The scoring rule that matches a quantile target is the pinball (tick) loss at level $\alpha$:

$$\ell_\alpha(f, y) = (y - f)\left(\alpha - \mathbf{1}[y < f]\right)$$

For $\alpha = 0.80$, this assigns weight 0.8 to under-predictions and weight 0.2 to over-predictions — exactly the asymmetry needed to incentivise correct 80th percentile estimation. This is a proper scoring rule: a model that truthfully reports its best estimate of the 80th percentile minimises expected pinball loss.

The decomposition `S = MCB + UNC - DSC` applies to pinball loss exactly as it does to squared error. The MCB component measures how much worse the model's 80th percentile estimate is than its optimally recalibrated version. The DSC component measures how much better the recalibrated estimate is than just predicting the portfolio's empirical 80th percentile for every risk. The statistical inference layer is new: Dimitriadis & Puke (2026) extend Mincer-Zarnowitz theory to non-smooth scoring functions using subgradient arguments and the Galvao-Yoon (2024) HAC covariance estimator. Before March 2026, formally testing whether a quantile forecast was miscalibrated — as opposed to eyeballing coverage rates — had no valid asymptotic framework in Python.

---

## The Mincer-Zarnowitz regression for quantile forecasts

For mean forecasts, MZ regression is OLS: regress observed outcomes on predictions and test whether slope = 1, intercept = 0. The generalisation to quantile forecasts replaces OLS with quantile regression at the same level $\alpha$:

```
q_alpha(y | y_hat) = b + a * y_hat
```

If the model is well-calibrated, $a \approx 1$ and $b \approx 0$ — the conditional $\alpha$-quantile of actual losses given predicted losses should equal the predicted losses. The MCB test is a test that the MZ quantile regression coefficients are (0, 1). The DSC test asks whether the recalibrated quantile forecast beats the unconditional portfolio quantile.

The slope interpretation carries over from mean models with one important caveat: a slope below 1 means the model is not spreading its quantile estimates far enough across risks. A model with a slope of 0.85 at the 80th percentile is telling you that when it predicts a risk's 80th percentile loss at £5,000, the actual 80th percentile conditional on that prediction is closer to £4,250. The model is compressing its range. That has direct implications for capital allocation: the highest-risk policies in the portfolio are less capitally-intensive than the model implies, and the lowest-risk policies are more so.

---

## Running this in practice

```python
import numpy as np
from insurance_monitoring.calibration import ScoreDecompositionTest

# Reserve model monitoring: 3,000 settled claims from accident year 2024
# y_settled: ultimate settled claim amounts (£)
# y_pred_80: model's 80th percentile reserve estimate per claim

sdt_80 = ScoreDecompositionTest(
    score_type='quantile',
    alpha=0.80,
)

result_80 = sdt_80.fit_single(y_settled, y_pred_80)
print(result_80.summary())
```

Output:

```
Score decomposition — quantile score (alpha=0.80), n=3,000
  Score (S):        0.00612
  MCB:              0.00089  (SE=0.00031, p=0.004)
  DSC:              0.00147  (SE=0.00052, p=0.005)
  UNC:              0.00670

  MZ quantile regression:  q_0.80(y | y_hat) = 183 + 0.847 * y_hat

  MCB=0 test:  FAIL (miscalibrated at 80th percentile)  p=0.004
  DSC=0 test:  PASS (has ranking skill at 80th percentile)  p=0.005
```

The output says the model is significantly miscalibrated (p=0.004): its 80th percentile estimates are systematically compressed (slope 0.847, intercept £183). The model has genuine discrimination skill (p=0.005) — it ranks risks correctly by quantile. This is a recalibrate situation, not a refit.

The actionable correction: multiply the non-intercept component of each prediction by 1/0.847 ≈ 1.18 and add £183. That is a parametric correction to the reserve model's output — not a refit of the underlying model structure.

---

## Comparing reserve models

The two-sample test matters when you are choosing between a chain-ladder bootstrap and a GLM-based stochastic reserve model, or between model vintages. The `fit_two` method runs the 6×6 joint HAC covariance across both models' score vectors and tests delta-MCB and delta-DSC separately.

```python
# Model A: chain-ladder bootstrap (current)
# Model B: GLM stochastic reserve (challenger)
# Both targeting the 80th percentile of claim ultimate

comparison = sdt_80.fit_two(y_settled, y_pred_a_80, y_pred_b_80)

print(f"Delta-MCB (B - A): {comparison.delta_mcb:.5f}  (p={comparison.delta_mcb_pvalue:.3f})")
print(f"Delta-DSC (B - A): {comparison.delta_dsc:.5f}  (p={comparison.delta_dsc_pvalue:.3f})")
print(f"DM overall:        {comparison.delta_score:.5f}  (p={comparison.delta_score_pvalue:.3f})")
print(f"Combined IU:                                     (p={comparison.combined_pvalue:.3f})")
```

If the GLM challenger posts lower pinball loss overall (DM significant) but delta-MCB explains the entire gap and delta-DSC is not significant, you are looking at a calibration advantage, not a structural one. The chain-ladder bootstrap with a recalibrated output layer recovers the same performance. That is a meaningful finding for an internal model validation committee.

The Diebold-Mariano test alone — testing whether total pinball loss is equal — cannot make this distinction. It is consistent with either story. When two models differ primarily in one component, component-specific tests have higher power than the DM test: this is explicitly demonstrated in the paper's simulation study.

---

## What this does not cover

**IBNR and reporting lags.** `ScoreDecompositionTest` requires settled claims — outcomes must be observed to compute the score. For accident years still developing, you have at best partially developed ultimates. The test is valid only once you have reasonably complete settlement. Running it on immature development year data is not wrong statistically, but the "outcomes" you are scoring against are estimates, not observations, which adds a layer of model risk that the test does not capture.

**Correlation across accident years.** HAC standard errors handle temporal autocorrelation within a monitoring window — claims from the same period experience the same macro conditions. They do not automatically handle spatial correlation across policies in the same postcode or bodily injury claims from the same solicitor network. For portfolios with strong spatial dependence, `hac_lags` should be set conservatively.

**Non-parametric miscalibration.** Linear MZ recalibration only removes affine miscalibration — a scale and shift. A reserve model that is correctly calibrated on average but understates the 80th percentile for large commercial risks whilst overstating it for personal lines will show MCB near zero by the MZ criterion despite being systematically wrong by segment. For the full diagnostic picture, pair this with `murphy_decomposition()`, which uses isotonic recalibration (PAVA) and will surface segment-level miscalibration that linear MZ misses.

```python
from insurance_monitoring.calibration import murphy_decomposition

# Full calibration curve — catches non-linear miscalibration
murphy = murphy_decomposition(y_settled, y_pred_80, distribution="gamma")
murphy.plot()  # calibration and discrimination curves by risk decile

# Inference tests — catches linear bias, provides p-values
result = sdt_80.fit_single(y_settled, y_pred_80)
print(f"MCB p-value: {result.mcb_pvalue:.3f}")
```

Run both. `murphy_decomposition()` gives you the complete diagnostic picture of where the reserve model is wrong. `ScoreDecompositionTest` gives you the evidence for the governance decision.

---

## The governance case

Solvency II Technical Provisions (Articles 76–86) require that technical provisions use "best estimate" assumptions with appropriate risk margins. The PRA expects that stochastic reserve models are validated against actual experience — which means testing whether the model's quantile estimates are correct, not just whether the mean is correct.

The standard validation approach — plotting fan charts of model percentiles against actual development factors — is visual, not statistical. "The fan looks broadly right" is not a validation finding. A `ScoreDecompositionTest` result that says `MCB p=0.84, DSC p=0.002` is: the model is not significantly miscalibrated at the 80th percentile, and it does have statistically significant discrimination skill. That is a validation finding you can put in a model validation report.

For firms with an internal model approved by the PRA, the validation framework is part of the regulatory capital calculation. PRA CP6/24 (insurance model risk guidance) expects quantitative validation evidence. A p-value from a formally grounded test is quantitative evidence. A fan chart is a picture.

The reference implementation is [insurance-monitoring on PyPI](https://pypi.org/project/insurance-monitoring/). `ScoreDecompositionTest` is at `insurance_monitoring.calibration.ScoreDecompositionTest`. Install with:

```bash
uv add "insurance-monitoring>=1.2.0"
```

The paper is Dimitriadis, T. & Puke, M. (2026), "Statistical Inference for Score Decompositions", arXiv:2603.04275. The quantile MZ extension is in Section 2.3; the HAC covariance derivation for non-smooth scoring rules is in the appendix.

---

*The R reference implementation for the SDI method is [marius-cp/SDI](https://github.com/marius-cp/SDI). Our Python implementation adds exposure weighting and insurance-specific scoring function wrappers absent from the R package.*
