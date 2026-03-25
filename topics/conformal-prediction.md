---
layout: page
title: "Conformal Prediction for Insurance Pricing"
description: "Distribution-free prediction intervals for insurance GBMs. Finite-sample coverage guarantees, exposure weighting, and non-exchangeable claims series. Python library and tutorials for UK pricing actuaries."
permalink: /topics/conformal-prediction/
---

Standard insurance models produce point estimates. A Tweedie GBM gives you expected loss cost — not how confident the model is in that estimate. That distinction matters. A standard motor risk with ten thousand similar policies behind it is not the same as a borderline fleet risk in a sparse corner of feature space. The model outputs a number in both cases. Without uncertainty quantification you cannot tell them apart, and you will misprice the risks you are least sure about most consistently.

The classical response is parametric confidence intervals: bootstrap the GLM coefficients, or propagate uncertainty through the Tweedie dispersion. Both depend on distributional assumptions. If the model is misspecified — and it is always at least partially misspecified — the intervals are wrong in ways that are difficult to characterise.

Conformal prediction offers a different kind of guarantee. It makes no assumptions about the error distribution and produces intervals with a finite-sample coverage guarantee: the true value falls within the interval at least 90% of the time (or whatever coverage level you set), unconditionally. The [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) library applies this to insurance models, handling the exposure weighting, heteroscedasticity, and non-exchangeable claims series that standard implementations ignore.

**Library:** [insurance-conformal on GitHub](https://github.com/burning-cost/insurance-conformal) &middot; `pip install insurance-conformal`

---

## Tutorials and introductions

- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the core tutorial: split conformal on a Tweedie GBM with exposure weighting and heteroscedasticity correction
- [Causal AI for Pricing Actuaries: A Practical Guide](/2026/03/25/causal-ai-for-pricing-actuaries/) — includes conformal intervals as part of the uncertainty quantification workflow
- [Which Uncertainty Quantification Method? A Decision Framework for Insurance Pricing Actuaries](/2026/03/25/which-uncertainty-quantification-method-decision-framework/) — when to use conformal vs GAMLSS vs credibility vs bootstrap

---

## Techniques and extensions

- [Conformalised Quantile Regression: Prediction Intervals That Actually Adapt to Risk](/2026/03/24/conformalised-quantile-regression-insurance-prediction-intervals/) — CQR for heteroscedastic claims, better interval efficiency than split conformal
- [Frequency and Severity Are Two Outputs. You Have One Prediction Interval.](/2026/03/13/insurance-multivariate-conformal/) — joint conformal prediction over frequency and severity simultaneously
- [Coverage Is the Wrong Guarantee for Pricing Actuaries](/2026/03/13/insurance-conformal-risk/) — conformal risk control for premium sufficiency rather than interval coverage
- [Your Conformal Intervals Are Wrong When the Claims Series Has Trend](/2025/12/30/conformal-prediction-for-non-exchangeable-claims-time-series/) — adaptive conformal inference for non-exchangeable (trending) claims data
- [GAMLSS vs Conformal: Head-to-Head on the Same Dataset](/2026/03/25/gamlss-vs-conformal-head-to-head/) — empirical comparison of parametric and distribution-free uncertainty methods
- [Your Reserve Range Has No Frequentist Guarantee](/2026/02/14/reserve-range-conformal-guarantee/) — applying conformal coverage guarantees to reserving range estimation
- [Conformal Prediction for Solvency II Capital Requirements: Model-Free SCR Estimation](/2026/03/25/conformal-prediction-solvency-ii-scr-model-free-estimation/) — using conformal intervals for capital modelling without distribution assumptions
- [Monthly Covariate Shift Monitoring: When to Reweight and When to Retrain](/2026/03/15/covariate-shift-detection-book-mix-changes/) — conformal validity as a monitoring signal for book mix changes

---

## Benchmarks and validation

- [Does Conformal Prediction Actually Work for Insurance Claims?](/2026/03/26/does-conformal-prediction-actually-work-for-insurance-claims/) — empirical coverage results on held-out UK motor data
- [Does conformal prediction actually work for insurance pricing?](/2026/03/23/does-conformal-prediction-work-insurance-pricing/) — validation of coverage guarantees under realistic insurance conditions
- [Which Uncertainty Quantification Method Should You Use?](/2026/03/25/which-uncertainty-quantification-method/) — benchmark comparison across methods on the same dataset

---

## Library comparisons

- [MAPIE vs insurance-conformal: Why Generic Conformal Prediction Breaks on Insurance Data](/2026/03/20/mapie-vs-insurance-conformal-prediction-intervals/) — what goes wrong with MAPIE on Tweedie models and exposure-weighted portfolios
