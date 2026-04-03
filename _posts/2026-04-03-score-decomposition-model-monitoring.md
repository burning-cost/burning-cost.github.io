---
layout: post
title: "Your Gini Is Stable. Your Model Isn't."
date: 2026-04-03
categories: [model-monitoring, insurance-pricing]
tags: [score-decomposition, calibration, discrimination, mcb, dsc, unc, gini, auc, model-monitoring, motor-insurance, mincer-zarnowitz, hac, quarterly-monitoring, arXiv-2603.04275, Dimitriadis, Puke, actuarial, insurance-monitoring]
description: "A stable Gini coefficient is not evidence that a model is performing well. It is evidence that the model is still ranking risks in the same order. Score decomposition separates discrimination from miscalibration, so you know whether a degrading model needs recalibration or a refit."
author: burning-cost
math: true
---

The quarterly model monitoring pack lands in the committee's inbox. The Gini on the motor frequency model is 33.1%, essentially unchanged from 33.4% twelve months ago. The slide deck says "performance remains stable". The committee moves on.

Meanwhile, claims inflation has run at 9% for three years. The book has shifted composition — more young drivers, fewer fleet policies. The model was last refitted in 2022. But the Gini is stable, so all is well.

This is a trap. The Gini coefficient — and by extension, any rank-order discrimination metric — is blind to calibration. A model that correctly identifies which risks are cheaper and which are more expensive will maintain its Gini even as every single one of its absolute predictions drifts further from reality. You can have a stable Gini and a model that is mispricing every policy in the book.

The decomposition that makes this concrete is:

$$S(F, y) = \text{MCB} + \text{UNC} - \text{DSC}$$

where MCB (miscalibration) and DSC (discrimination) are orthogonal. They can move independently. Watching Gini — which maps onto DSC — tells you nothing about whether MCB has crept up. You need both.

---

## What the three components actually measure

For any proper scoring rule S — squared error, Poisson deviance, pinball loss — the decomposition holds exactly. Not approximately. Exactly.

**UNC** is the score of the "climate" forecast: predicting the portfolio mean for every risk. It measures the inherent difficulty of the prediction problem. For a motor frequency book, it reflects how variable claim rates are across the portfolio. UNC changes slowly and only when the book composition changes materially. It is not a model property; it is a data property.

**DSC** (discrimination) measures how much better than climate you can get once the forecast is optimally recalibrated. It is the pure ranking signal: how accurately does the model separate cheap risks from expensive ones? Gini and AUC are monotone transformations of DSC — they measure the same thing. A model with high DSC correctly identifies that a 19-year-old on a sports hatchback is materially riskier than a 52-year-old on a city car, regardless of whether its absolute predictions are correct.

**MCB** (miscalibration) is the excess score attributable to wrong predicted levels. The recalibration that defines MCB is a Mincer-Zarnowitz (MZ) regression: fit the linear model $y = b + a \hat{y}$ on observed outcomes and predictions. If $a \approx 1$ and $b \approx 0$, the model is well-calibrated; MCB $\approx 0$. If claims inflation has eroded the model's level, the intercept will be non-zero and MCB will be positive. If the rating relativities have drifted — perhaps because the risk composition of certain segments has changed — the slope will deviate from 1.

The key point: DSC is insensitive to the values of $a$ and $b$. A model with slope 0.7 and intercept 0.15 can have the same DSC as a perfectly calibrated model, provided the ranking of risks is preserved. Gini tracks DSC. Gini does not track MCB. They are different things.

---

## A motor book tracked over eight quarters

Consider a motor frequency model fitted in mid-2023 and deployed into production. We track the quarterly decomposition over the following two years. The data below is simulated from a plausible scenario: 20,000 policies per quarter, claim rates drifting upward with inflation, model not recalibrated since fitting.

| Quarter | Score (S) | MCB | DSC | UNC | Gini (approx) | MCB p | DSC p |
|---------|-----------|-----|-----|-----|----------------|-------|-------|
| Q3 2023 | 0.00384 | 0.00009 | 0.00120 | 0.00495 | 33.4% | 0.31 | <0.001 |
| Q4 2023 | 0.00387 | 0.00012 | 0.00118 | 0.00493 | 33.2% | 0.21 | <0.001 |
| Q1 2024 | 0.00391 | 0.00018 | 0.00121 | 0.00494 | 33.5% | 0.08 | <0.001 |
| Q2 2024 | 0.00395 | 0.00024 | 0.00123 | 0.00496 | 33.8% | 0.03 | <0.001 |
| Q3 2024 | 0.00401 | 0.00031 | 0.00119 | 0.00493 | 33.1% | 0.007 | <0.001 |
| Q4 2024 | 0.00406 | 0.00038 | 0.00118 | 0.00490 | 32.9% | 0.001 | <0.001 |
| Q1 2025 | 0.00412 | 0.00045 | 0.00117 | 0.00492 | 32.7% | <0.001 | <0.001 |
| Q2 2025 | 0.00418 | 0.00051 | 0.00116 | 0.00491 | 32.4% | <0.001 | <0.001 |

Three things are happening simultaneously, and only score decomposition reveals all three.

**1. Gini is wobbling, not collapsing.** From Q3 2023 to Q2 2025, Gini has moved from 33.4% to 32.4% — a one-point drift that most committee packs would label "stable". DSC confirms this: discrimination has barely moved (0.00120 to 0.00116). The model still ranks risks correctly.

**2. MCB is rising monotonically, and it is now statistically significant.** In Q3 2023, MCB is 0.00009 with p=0.31 — not significant, consistent with a well-calibrated model. By Q2 2024 (p=0.03), it crosses the conventional threshold. By Q4 2024 (p=0.001), it is unambiguous. By Q2 2025 (p<0.001), the model is materially miscalibrated.

**3. The overall score (S) has risen by 8.9% over two years, almost entirely due to MCB.** An A/E ratio would have caught some of this — the model is systematically underpricing. But the A/E has no standard error, and it cannot tell you that discrimination is unaffected. Score decomposition gives you both facts simultaneously: yes, recalibrate; no, do not refit.

---

## What the MZ regression is telling you

At Q2 2025, the MZ regression for this model returns approximately:

```
y_hat* = 0.0031 + 0.891 * y_hat
```

The intercept of 0.0031 is the base rate shift. Two years of uncorrected claims inflation have added about 0.3 percentage points to the portfolio mean frequency — the model's absolute predictions are low. The slope of 0.891 is the relativity compression: the model is not spreading far enough. Risks it predicts at the 95th percentile are actually about 11% riskier than the model implies; risks at the 5th percentile are about 11% less risky.

Both corrections are actionable without a model refit. Multiply all rating relativities by $1/0.891 \approx 1.12$ and adjust the base rate to restore balance. This is a recalibration exercise, not a modelling exercise.

The MZ slope below 1 is common in models that have aged without recalibration: the risk segments that were already over-represented in the training data have become relatively safer (or more profitable) over time, whilst the under-represented segments have deteriorated. The model learned correlations from 2021-2023 data that no longer hold in the same form. But the rank ordering — which segments are cheaper and which are more expensive — has been preserved. That is why DSC is stable.

---

## Running the analysis

The code is straightforward. For each quarterly monitoring window:

```python
import numpy as np
from insurance_monitoring.calibration import ScoreDecompositionTest

# Quarterly monitoring window
# y: observed claim frequency (claims / vehicle years)
# y_pred: model predictions
# vehicle_years: exposure per policy

sdt = ScoreDecompositionTest(score_type='mse', exposure=vehicle_years)
result = sdt.fit_single(y, y_pred)

print(result.summary())
```

Output for Q2 2025:

```
Score decomposition (n=20,412, exposure-weighted)
  Score (S):        0.00418
  MCB:              0.00051  (SE=0.00013, p<0.001)
  DSC:              0.00116  (SE=0.00009, p<0.001)
  UNC:              0.00483

  MZ regression:    y_hat* = 0.0031 + 0.891 * y_hat

  MCB=0 test:   FAIL (miscalibrated)  p<0.001
  DSC=0 test:   PASS (has skill)      p<0.001
```

To track the decomposition over time, store the result each quarter:

```python
from dataclasses import asdict
import polars as pl

results = []
for quarter, (y_q, y_pred_q, exp_q) in monitoring_windows.items():
    sdt = ScoreDecompositionTest(score_type='mse', exposure=exp_q)
    r = sdt.fit_single(y_q, y_pred_q)
    results.append({
        "quarter": quarter,
        "mcb": r.miscalibration,
        "dsc": r.discrimination,
        "unc": r.uncertainty,
        "mcb_pvalue": r.mcb_pvalue,
        "dsc_pvalue": r.dsc_pvalue,
        "mz_slope": r.recalib_slope,
        "mz_intercept": r.recalib_intercept,
    })

history = pl.DataFrame(results)
```

Plot MCB and DSC separately over time, with a horizontal band at zero and markers when the p-value crosses 0.05. That is the monitoring pack slide — not a Gini trend line.

---

## When to recalibrate versus when to refit

The decomposition frames the decision cleanly.

**MCB significant, DSC unaffected:** recalibrate. The model's ranking is intact. Apply the MZ correction (multiply relativities by $1/a$, adjust the base rate by $b/a$). No new modelling required. This is the scenario in the motor example above — MCB has grown steadily whilst DSC has held.

**DSC declining, MCB stable:** the model is losing discrimination. Something has changed in the risk structure that the model's features no longer capture. A new risk factor has emerged, or an existing one has changed its relationship with claims. Recalibration will not fix this. A refit — or at minimum a feature review — is warranted.

**Both MCB and DSC degrading:** the model is in trouble on both dimensions. The usual cause is a structural shift: a new peril that the model was never exposed to, a regulatory change that altered the risk mix (e.g., the FCA pricing reform reshaping the renewal book), or a macro shock like the post-2022 claims inflation environment. A refit is likely unavoidable, though recalibration may buy time whilst the development programme runs.

**DSC falls to near-zero, MCB significant:** the model has no residual ranking skill. It is not only wrong in its absolute predictions — it is also no longer separating risks meaningfully. This is the worst case. The model should not be pricing without human review.

---

## Why Gini alone misses this

Gini — and its near-equivalent, the Lorenz curve AUC — is insensitive to affine transformations of the predictions. Multiply every prediction by 1.5, add 0.01, the Gini is unchanged. This is by design: Gini is a pure discrimination metric. It answers "does the model rank risks correctly?" and ignores "does the model predict the right numbers?".

In a well-functioning monitoring programme, this would not matter — a calibration check (A/E ratio, double-lift plots) would catch the level drift separately. In practice, many teams lean on Gini as their primary health indicator precisely because it is easy to track and tends to be stable. Its stability is reassuring. That reassurance is unwarranted.

The post-2022 claims inflation environment illustrated this exactly. Books that had gone through the FCA pricing reform in 2022 had well-constructed models with reasonable Gini coefficients. But three years of 8-12% annual claims inflation — medical costs, repair costs, used car values — had severely eroded absolute calibration. The models were still ranking correctly. They were pricing at 2022 levels in 2025. The Gini told committees nothing was wrong.

Score decomposition would have shown MCB trending upward from early 2023. The recalibration could have been done in any quarterly cycle. Instead, many firms ended up doing emergency rate corrections late in the cycle.

---

## The governance argument

FCA Consumer Duty requires that pricing decisions are fair and based on up-to-date risk assessment. A miscalibrated model — one systematically underpricing or overpricing relative to current claims experience — is not producing outcomes consistent with fair value, even if its discrimination is intact.

MCB has a direct link to fair value in a way that Gini does not. A model with MCB significantly greater than zero is charging the wrong prices. DSC degradation is a warning about future model relevance. Both matter, and they matter for different reasons.

The practical ask: add two numbers to the quarterly monitoring pack. MCB with its p-value and trend. DSC with its p-value and trend. Keep the Gini if the committee likes it — it is a useful intuition check. But stop treating a stable Gini as evidence that the model is performing well. It is evidence of one thing only, and calibration is not it.

---

## A note on HAC standard errors

Insurance monitoring data is correlated. Claims from the same quarter share common macro shocks — the same spell of winter weather, the same claims inflation trend, the same batch of solicitor-driven PI claims. Standard OLS residuals would understate uncertainty, producing MCB p-values that are too small and trigger false alarms.

`ScoreDecompositionTest` uses Newey-West (Bartlett kernel) HAC standard errors, which correct for temporal autocorrelation. The lag order defaults to $\lfloor 4(n/100)^{2/9} \rfloor$ — roughly 7 lags for a 20,000-policy quarterly window. For books with strong annual seasonality (a boat insurer, a flood-exposed home book), override with `hac_lags=12` to absorb calendar-year cyclicality.

The HAC correction matters most precisely when you most want to trust the p-value: in a monitoring window shortly after a macro event, when you are trying to determine whether a jump in MCB is real or statistical noise.

---

*Score decomposition for insurance model monitoring is based on Dimitriadis, T. & Puke, M. (2026), "Statistical Inference for Score Decompositions", arXiv:2603.04275. The R reference implementation is [marius-cp/SDI](https://github.com/marius-cp/SDI). The `ScoreDecompositionTest` class is in [insurance-monitoring](https://pypi.org/project/insurance-monitoring/) (v1.2.0+) at `insurance_monitoring.calibration`.*
