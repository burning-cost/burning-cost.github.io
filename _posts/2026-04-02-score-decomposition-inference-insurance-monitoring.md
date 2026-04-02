---
layout: post
title: "Your Model Comparison Has No Error Bars"
date: 2026-04-02
categories: [libraries]
tags: [insurance-monitoring, calibration, scoring-rules, inference, governance, fca, consumer-duty, arXiv-2603.04275, Dimitriadis, Puke, mcb, discrimination]
description: "insurance-monitoring v1.2.0 adds ScoreDecompositionTest: HAC-robust p-values on the miscalibration and discrimination components of any scoring rule. When your pricing team argues about whether Model A is better than Model B, they need more than two RMSE numbers."
author: burning-cost
---

Every model comparison in a UK pricing team's governance pack has the same structure: Model A achieved RMSE 0.847, Model B achieved RMSE 0.831. Someone declares Model B the winner. The meeting moves on.

Nobody asked: is the difference statistically meaningful? And if it is real, is Model B better because it ranks risks more accurately, or because it is better calibrated — or both?

These are not pedantic questions. A model that improves on RMSE purely through calibration can be recalibrated away by a competitor. A model that improves through discrimination is structurally better at pricing. Under FCA Consumer Duty, the obligation is to demonstrate that model governance produces good outcomes — and "our RMSE went down" is not that demonstration.

[insurance-monitoring v1.2.0](https://pypi.org/project/insurance-monitoring/) adds `ScoreDecompositionTest`, implementing Dimitriadis & Puke (arXiv:2603.04275, March 2026). It gives you HAC-robust p-values on the components that matter.

---

## The decomposition

For any proper scoring rule S — MSE, MAE, quantile score — there is an exact additive decomposition:

```
S(F, y) = MCB(F, y) + UNC(y) - DSC(F, y)
```

- **MCB** (miscalibration): how much worse your forecasts are than their best linear recalibration. If MCB > 0, your model is making systematic errors — the direction and magnitude could be corrected by a simple affine transformation of the outputs.
- **UNC** (uncertainty): the irreducible difficulty of the problem. Fixed for a given dataset. Depends only on the distribution of actuals, not on your model.
- **DSC** (discrimination): how much better your recalibrated forecasts are than just predicting the mean. This is the component that measures genuine ranking ability.

A good model has low MCB and high DSC. UNC cancels when you compare two models on the same dataset.

The paper extends the standard Mincer-Zarnowitz regression framework — originally developed for mean forecasts — to non-smooth scoring functions including quantile scores. This matters for reserving teams working with VaR or ES targets. HAC standard errors handle autocorrelation in monitoring windows where consecutive months overlap.

---

## Why single-number comparisons mislead

Consider two motor frequency models, both fitted to the same development dataset, evaluated on 18 months of hold-out data:

| | RMSE | MCB | DSC |
|---|---|---|---|
| Model A (GLM, current) | 0.847 | 0.018 | 0.129 |
| Model B (GBM, challenger) | 0.831 | 0.041 | 0.147 |

Model B has lower RMSE and looks like the clear winner. But break it down: Model A is better calibrated (lower MCB at 0.018 vs 0.041), whilst Model B discriminates more (higher DSC at 0.147 vs 0.129). The RMSE improvement comes entirely from discrimination. The calibration picture has actually got worse. Now the question is whether that DSC gap is statistically real, or whether Model B just got lucky on this hold-out window. A committee armed with p-values makes a different decision than one looking at two RMSE numbers.

The Diebold-Mariano test, the standard tool for comparing forecast accuracy, tests whether S(A, y) - S(B, y) = 0 in expectation. It misses scenarios where models differ on only one component. Two models can have identical overall score whilst one is better calibrated but worse at ranking, and the DM test will correctly say "no difference" whilst the DSC test will catch the divergence.

---

## The API

```python
import numpy as np
from insurance_monitoring.calibration import ScoreDecompositionTest

rng = np.random.default_rng(42)
n = 5_000

# Simulated motor frequency data
true_rate = rng.uniform(0.04, 0.18, n)
actual = rng.poisson(true_rate).astype(float)
exposure = rng.uniform(0.5, 1.0, n)

# Model A: GLM — well calibrated, modest discrimination
pred_a = true_rate * rng.lognormal(0, 0.05, n)

# Model B: GBM challenger — slightly miscalibrated, stronger discrimination
pred_b = true_rate * rng.lognormal(0.03, 0.02, n) * (1 + 0.4 * (true_rate - true_rate.mean()))

sdt = ScoreDecompositionTest(score_type="mse", exposure=exposure)

# Single-model decomposition
result_a = sdt.fit_single(actual, pred_a)
print(f"Model A — MCB: {result_a.miscalibration:.4f} (p={result_a.mcb_pvalue:.3f}), "
      f"DSC: {result_a.discrimination:.4f} (p={result_a.dsc_pvalue:.3f})")

result_b = sdt.fit_single(actual, pred_b)
print(f"Model B — MCB: {result_b.miscalibration:.4f} (p={result_b.mcb_pvalue:.3f}), "
      f"DSC: {result_b.discrimination:.4f} (p={result_b.dsc_pvalue:.3f})")

# Two-model comparison: tests delta-MCB and delta-DSC separately
comparison = sdt.fit_two(actual, pred_a, pred_b)
print(f"\nDelta-MCB (B - A): {comparison.delta_mcb:.4f} (p={comparison.delta_mcb_pvalue:.3f})")
print(f"Delta-DSC (B - A): {comparison.delta_dsc:.4f} (p={comparison.delta_dsc_pvalue:.3f})")
print(f"Combined IU p-value: {comparison.combined_pvalue:.3f}")

sdt.summary(comparison)
```

The `fit_two` method tests delta-MCB and delta-DSC independently, then combines them via an intersection-union test. The combined p-value is the maximum of the two component p-values. To reject the joint null — that both MCB and DSC are equal — you need both individual tests to be significant at your chosen alpha. Conservative, but valid: it controls size without requiring any correlation assumption between the two test statistics.

For frequency models, pass `exposure` as a vector and the regressions use WLS automatically. The HAC lag order defaults to `floor(4 * (n/100)^(2/9))`, which works for most monitoring windows; override with `hac_lags=k` if your experience data is more serially dependent.

---

## Pairing with murphy_decomposition

`ScoreDecompositionTest` uses linear recalibration (the MZ regression) to define MCB and DSC. `murphy_decomposition` uses isotonic regression (PAVA) — a non-parametric approach that gives a more complete calibration picture at the cost of not producing hypothesis tests.

We recommend running both. `murphy_decomposition` gives you the full calibration curve and a granular view of where the model is wrong. `ScoreDecompositionTest` tells you whether the differences you are seeing are statistically credible and whether model replacement is warranted, not just model recalibration.

```python
from insurance_monitoring.calibration import murphy_decomposition

# Run murphy for the diagnostic picture
murphy_result = murphy_decomposition(actual, pred_b, distribution="poisson")
murphy_result.plot()  # calibration and discrimination curves

# Run SDT for the inference question
result = sdt.fit_single(actual, pred_b)
print(f"MCB p-value: {result.mcb_pvalue:.3f}")
# p > 0.05: calibration deviation is not statistically significant
# Safe to recalibrate rather than refit
```

---

## Governance angle

FCA Consumer Duty SS1/23 model governance frameworks — increasingly adopted by UK insurers beyond the banking sector where they originated — require documented evidence that model monitoring detects calibration and discrimination degradation as distinct phenomena. A single RMSE trend chart does not satisfy that requirement. A committee pack that shows MCB stable at p=0.42 and DSC declining with p=0.03 says something precise: the model's ranking ability is degrading but its bias is under control. That is actionable, and it is auditable.

The quantile score path (`score_type="quantile"`, `alpha=0.05` for a 5th percentile) is directly applicable to reserve adequacy monitoring where the target is not the mean but a specific quantile of the loss distribution. The inference framework is the same: HAC-robust, distribution-free, valid under temporal dependence.

---

## Install

```bash
pip install "insurance-monitoring>=1.2.0"
```

`ScoreDecompositionTest` is in `insurance_monitoring.calibration`. The class, result types, and `summary()` and `plot()` methods are all documented in the [API reference](https://burning-cost.github.io/insurance-monitoring/api/).

The paper is Dimitriadis, T. & Puke, M. (2026), "Statistical Inference for Score Decompositions", arXiv:2603.04275. The R reference implementation is the `SDI` package.
