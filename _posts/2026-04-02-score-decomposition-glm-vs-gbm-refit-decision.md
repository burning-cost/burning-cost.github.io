---
layout: post
title: "Your Challenger GBM Beats the GLM — But Does It Deserve Deployment?"
date: 2026-04-02
categories: [insurance-pricing, model-monitoring]
tags: [score-decomposition, calibration, discrimination, glm, gbm, model-selection, insurance-monitoring, mcb, dsc, mincer-zarnowitz, governance, arXiv-2603.04275, fca, consumer-duty, actuarial]
description: "When a challenger GBM outperforms the production GLM on deviance, the improvement is often entirely in miscalibration — meaning the GLM could be recalibrated to match without a full refit. ScoreDecompositionTest in insurance-monitoring v1.2.0 separates the two."
author: burning-cost
---

Every UK motor pricing cycle produces the same conversation. The data science team presents a GBM challenger. It posts a Gini of 34.2% against the production GLM's 32.8%. Deviance is down 3.4%. The GBM is better. The business case for deployment looks straightforward.

Except the GLM was fitted on data from 2021-2023. It is now 2026. Claims inflation has run at 8-12% annually. The GLM's absolute level is wrong — it has not been recalibrated since the last model refresh. The GBM, fitted on more recent data, has the right absolute level baked in. It is better calibrated. But once you account for calibration, is it actually better at ranking risks?

The answer matters enormously. If the GBM is better calibrated and has the same discrimination as the recalibrated GLM, you are comparing a model refresh (expensive) against a balance correction (cheap). If the GBM genuinely improves discrimination — separating cheap risks from expensive ones more accurately — deployment is justified on its merits. These are different decisions, with different governance implications and different commercial consequences.

[insurance-monitoring v1.2.0](https://pypi.org/project/insurance-monitoring/) adds `ScoreDecompositionTest`, implementing Dimitriadis & Puke (arXiv:2603.04275, March 2026). It decomposes any scoring rule into Miscalibration (MCB) and Discrimination (DSC) components with asymptotically valid p-values on each. For this specific decision — refit the model, or recalibrate it — it gives you the statistical evidence you need.

---

## The decomposition and what it means for model selection

For a proper scoring rule S, forecast F, and observed outcomes y:

```
S(F, y) = MCB + UNC − DSC
```

**UNC** is the same for any model evaluated on the same dataset. It cancels entirely in model comparisons.

**MCB** measures systematic bias: how much worse is your forecast than its own best linear recalibration? The recalibration is a Mincer-Zarnowitz (MZ) regression of outcomes on predictions:

```
y = b + a * y_hat + epsilon
```

A perfectly calibrated model has a=1, b=0. MCB=0. Non-zero MCB means a simple affine adjustment would improve the score — no refitting, no new features.

**DSC** measures genuine ranking ability: how much better than the portfolio mean is your model after any calibration errors are removed? DSC is the discrimination signal stripped of level effects.

The practical implication for model comparison:

- If model B has lower S than model A, but delta-MCB explains the entire gap and delta-DSC is not significant, then B's advantage is purely calibration. Recalibrate A and the advantage disappears.
- If delta-DSC is significant — B has demonstrably better ranking ability — deployment is justified independently of calibration.

The Diebold-Mariano (DM) test, the standard tool for comparing forecast accuracy, tests whether S(A) − S(B) = 0 in expectation. It cannot distinguish these two cases. A significant DM test is consistent with either story.

---

## A worked example: 120,000-policy motor holdout

The following figures are from a realistic simulation. A production GLM was fitted in 2022, with a credibility-corrected A/E adjustment applied in 2024 but no structural refit. A challenger GBM was trained on 2023-2025 experience data and evaluated on a 120,000-policy holdout from Q4 2025.

| | S (Poisson deviance) | MCB | DSC |
|---|---|---|---|
| Production GLM (recalibrated 2024) | 0.04183 | 0.00312 | 0.02141 |
| Challenger GBM | 0.03891 | 0.00041 | 0.02252 |
| Delta (GBM − GLM) | −0.00292 | −0.00271 | +0.00111 |
| Delta p-value | 0.003 | 0.001 | 0.048 |

The GBM is better on total deviance (p=0.003, significant). Decomposing it: the calibration gap (delta-MCB = −0.00271, p=0.001) accounts for 93% of the total deviance improvement. The discrimination gain (delta-DSC = +0.00111, p=0.048) is statistically significant but small.

The MCB gap is telling you something concrete. The GLM MZ regression returns a slope of 0.961 and an intercept of 0.0021 — the relativities are slightly compressed and the overall level is slightly high. The 2024 recalibration has not fully corrected for subsequent inflation. The GBM, trained on 2023-2025 data, has the current claims environment baked in.

Armed with this, the governance decision looks different. Deploying the GBM buys you a 3.4% deviance improvement, 93% of which is recoverable by recalibrating the GLM's intercept and expanding its relativities by approximately 4%. The remaining 7% — the genuine discrimination improvement — is real but small. Whether that 0.00111 DSC improvement justifies the time, cost, and regulatory friction of a full model replacement is a business call, not a statistical one.

---

## Running the test

```python
import numpy as np
from insurance_monitoring.calibration import ScoreDecompositionTest

# Holdout data: 120,000 policies
# y: observed claim counts / exposure (annualised frequency)
# exposure: vehicle years
# pred_glm: production GLM predictions
# pred_gbm: challenger GBM predictions

sdt = ScoreDecompositionTest(score_type="mse", exposure=vehicle_years)

# Single-model diagnostics first
result_glm = sdt.fit_single(y, pred_glm)
result_gbm = sdt.fit_single(y, pred_gbm)

print("GLM:")
print(f"  MCB: {result_glm.miscalibration:.5f}  (p={result_glm.mcb_pvalue:.3f})")
print(f"  DSC: {result_glm.discrimination:.5f}  (p={result_glm.dsc_pvalue:.3f})")
print(f"  MZ:  slope={result_glm.recalib_slope:.3f}, intercept={result_glm.recalib_intercept:.4f}")

print("\nGBM:")
print(f"  MCB: {result_gbm.miscalibration:.5f}  (p={result_gbm.mcb_pvalue:.3f})")
print(f"  DSC: {result_gbm.discrimination:.5f}  (p={result_gbm.dsc_pvalue:.3f})")

# Head-to-head comparison — tests delta-MCB and delta-DSC separately
comparison = sdt.fit_two(y, pred_glm, pred_gbm)

print("\nComparison:")
print(f"  delta_MCB: {comparison.delta_mcb:.5f}  (p={comparison.delta_mcb_pvalue:.3f})")
print(f"  delta_DSC: {comparison.delta_dsc:.5f}  (p={comparison.delta_dsc_pvalue:.3f})")
print(f"  DM test:   {comparison.delta_score:.5f}  (p={comparison.delta_score_pvalue:.3f})")
print(f"  Combined IU p-value: {comparison.combined_pvalue:.3f}")
```

The `fit_two` method estimates all six score series jointly and computes their 6×6 HAC covariance matrix. This matters: the GLM and GBM predictions are correlated — they are evaluated on the same claims experience. A naive z-test treating the two models as independent would understate uncertainty. The joint HAC covariance accounts for this.

For frequency models, pass exposure as a vector and the MZ regression uses WLS. HAC lag order defaults to `floor(4*(n/100)^(2/9))` — around 8 lags for a 120,000-policy holdout. Override with `hac_lags=k` if your data has strong annual seasonality; `hac_lags=12` covers the case where policies renew in the same month each year.

---

## What the MZ slope tells you about the GLM

The MZ regression output deserves more attention than it typically gets. When `sdt.fit_single()` returns `recalib_slope=0.961`, it is saying: your GLM relativities are compressed by about 4%. The model correctly identifies direction — young drivers, high-performance vehicles, urban postcodes — but is not spreading them far enough. A slope below 1 means you are underpricing your most expensive risks relative to your cheapest ones.

This is directly actionable without a model refit. If your pricing engine runs on a GLM rating table, you can multiply the non-base relativities by 1/0.961 ≈ 1.04 and adjust the base rate to preserve balance. The effect is to expand the spread of the rating structure whilst maintaining the same overall premium level. The MZ intercept tells you whether the base rate also needs adjusting.

Compare this with what a standard Gini coefficient gives you: 32.8% for the GLM, 34.2% for the GBM. The Gini is a pure discrimination metric — it measures ranking ability ignoring calibration entirely. A Gini improvement of 1.4 points says nothing about whether the improvement is recoverable by recalibration. The Gini and the DSC test answer the same underlying question, but the DSC test comes with a p-value.

---

## The governance question: when is a DSC improvement enough to justify refit?

There is no universal threshold. What score decomposition does is reframe the question from "is the GBM better?" (trivially yes, by construction of the challenger process) to "how much of the GBM's advantage is structural, and is that structural advantage worth the cost of deployment?"

For a UK motor book with 300,000-policy renewal volume, a DSC improvement of 0.00111 Poisson deviance units corresponds roughly to a 0.3% reduction in pure premium variance — call it £4-6 per policy in expected pricing accuracy improvement, assuming a £1,500 average premium and claim rates typical of a mixed urban/rural book. Against typical model development costs (6-9 months of actuarial and data science time, IT integration, regulatory validation), this is a marginal case. If the DSC improvement were 0.003-0.004 — a change that shows up as 2-3 Gini points — the calculation looks different.

What is certain: deploying the GBM and recalibrating the GLM give you the same calibration outcome. The question is whether the discrimination improvement justifies the additional work. Score decomposition puts a number on it.

Under FCA Consumer Duty PS22/9, firms must document that pricing decisions are based on legitimate risk factors and produce fair outcomes. A governance pack that says "we replaced the GLM with a GBM because Gini improved by 1.4 points" is weakly evidenced — it does not separate the calibration component (which reflects economic drift, not model quality) from the discrimination component (which reflects genuine model improvement). A pack that says "the GBM shows a statistically significant discrimination improvement of delta-DSC=0.00111 (p=0.048), whilst the calibration gap (delta-MCB=0.00271, p=0.001) is attributable to economic drift and has been corrected independently" is a materially stronger piece of evidence.

---

## When score decomposition is less useful

Linear MCB is the right calibration measure when your model errors are approximately affine — a systematic level shift, or a compressed/expanded spread. For GBMs with miscalibration that varies non-linearly across risk segments, the linear MZ recalibration understates the full MCB. A model could have MCB=0 by the MZ criterion whilst being systematically wrong for 18-year-olds and 60-year-olds simultaneously, the errors cancelling at the portfolio level.

The `murphy_decomposition()` function in insurance-monitoring uses isotonic (PAVA) recalibration, which captures any monotone miscalibration pattern — not just linear. We recommend running both. `murphy_decomposition()` gives you the full calibration curve and a diagnostic view of where the model is wrong by risk segment. `ScoreDecompositionTest` gives you the hypothesis tests for the governance decision.

For small holdouts (fewer than 200 policies), the HAC asymptotic standard errors are unreliable. The test is designed for production monitoring windows — quarterly holdouts of 5,000 or more policies are typical in UK personal lines. For specialty lines with thin data, the bootstrap MCB test in `check_auto_calibration()` is more appropriate.

---

## Installation

```
uv add insurance-monitoring>=1.2.0
```

`ScoreDecompositionTest` is at `insurance_monitoring.calibration.ScoreDecompositionTest`. The `fit_two` method is the relevant entry point for champion-challenger decisions.

**Paper:** [arXiv:2603.04275](https://arxiv.org/abs/2603.04275) — Timo Dimitriadis & Marius Puke (Heidelberg University, March 2026) | **Library:** [insurance-monitoring on PyPI](https://pypi.org/project/insurance-monitoring/)
