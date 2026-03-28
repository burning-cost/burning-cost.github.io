---
layout: post
title: "Why Random Cross-Folding Is Wrong for Time-Series Causal Inference"
date: 2026-03-27
categories: [causal-inference]
tags: [dml, double-machine-learning, time-series, cross-fitting, rate-change, causal-inference, python]
description: "Ciganovic et al. (March 2026) show that standard DML cross-fitting leaks future information when your data is a time series. Their fix  -  Reverse Cross-Fitting  -  has direct implications for evaluating insurance rate changes over time."
author: Burning Cost
---

The core promise of Double Machine Learning is that cross-fitting removes regularisation bias: fit nuisance models on one fold, predict on another, and the out-of-sample residuals carry no bias from in-sample overfit. That promise depends entirely on one assumption: that observations in the training fold are independent of observations in the evaluation fold.

When your data is a time series, that assumption fails.

---

## The contamination problem

In a standard `KFold(shuffle=True)` split, future observations end up in the training fold when past observations are in the evaluation fold. The nuisance model  -  say, `E[Y | D, X]` at time `t`  -  is fitted on data from `t+1, t+2, ...`. It learns the future.

This is not a small problem. In economic time series with strong autocorrelation, a nuisance model trained on future data absorbs almost all residual variation, leaving the second-stage DML estimate with nothing to work with. The bias is proportional to how much predictive power the future has over the past  -  substantial in any mean-reverting or trending series.

---

## Reverse Cross-Fitting

Ciganovic, D'Amario, and Tancioni (arXiv:2603.10999, March 2026) formalise this and propose a solution exploiting time-reversibility.

For a strictly stationary process, the forward sequence `(Y_1, ..., Y_T)` and the backward sequence `(Y_T, ..., Y_1)` have the same joint distribution. The scheme uses this symmetry:

1. Split the series at its midpoint into forward and backward halves.
2. Train nuisance models on the backward half; generate residuals on the forward half.
3. Train nuisance models on the forward half; generate residuals on the backward half.
4. Combine residuals and proceed with the DML second stage.

Training on `(t = T/2+1, ..., T)` and predicting on `(t = 1, ..., T/2)` creates no temporal contamination  -  the nuisance model has not seen the future of the evaluation observations. The paper proves root-T consistency and asymptotic normality under mild mixing conditions.

The paper also derives a **Goldilocks calibration rule** for nuisance regularisation: tune the nuisance learner to minimise the DML second-stage MSE, not standalone prediction error on a held-out fold. These are different criteria, and conflating them inflates bias in typical pipelines.

A third contribution  -  LP+DML  -  combines DML residualisation with Jordà Local Projections to estimate impulse response functions causally, applied to bank capital requirement shocks propagating into lending over multiple quarters.

---

## What this means for `insurance-causal`

Our [`insurance-causal`](https://github.com/burning-cost/insurance-causal) library uses `KFold(shuffle=True)` inside `cross_fit_nuisance()`  -  the standard random fold scheme. For `PremiumElasticity` on a cross-sectional renewal book where policies are independent observations, this is fine. Policies renewed in January 2023 do not cause policies renewed in March 2023.

But the `rate_change` module works with temporal aggregates: quarterly loss ratios, quarterly retention rates, time-indexed outcomes. When you want to residualise a temporal confound from a rate change trajectory using DML, you need temporal cross-fitting.

The natural extension is a `temporal_cross_fit_nuisance()` function in `autodml/_crossfit.py` with the same interface  -  returning `g_hat, alpha_hat, fold_indices, nuisance_models`  -  but with fold construction that respects time order via a `t_index` parameter.

The LP+DML approach also maps directly onto rate change impulse responses: instead of the current ITS segmented regression in `_its.py` (which uses Newey-West standard errors to handle autocorrelation but does not residualise confounders nonparametrically), you project DML residuals onto lagged rate change indicators. This would give causal impulse responses rather than associational segmented regression slopes.

---

## Where it breaks down

Reverse Cross-Fitting requires stationarity. Quarterly UK motor claim frequencies are not stationary  -  they have been trending hard since 2022 driven by parts inflation and claims severity. You need to detrend before applying this, or the time-reversibility argument does not hold.

The two-fold scheme also halves effective sample size. With 20 quarters of data  -  typical for a UK personal lines book  -  each nuisance model trains on 10 observations. Forest-based learners (`ForestRiesz`, `catboost` backend) are useless at that scale. You want `LinearRiesz` and the `linear` backend, with the Goldilocks calibration rule to set the regularisation strength.

For portfolios with 40+ quarters of data and stable data collection (rare before 2018 for most UK insurers), the method becomes genuinely tractable.

---

## Citation

Ciganovic, N., D'Amario, V., & Tancioni, M. (2026). *Double Machine Learning for Time Series*. arXiv:2603.10999. [https://arxiv.org/abs/2603.10999](https://arxiv.org/abs/2603.10999)
