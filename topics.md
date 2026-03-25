---
layout: page
title: "Topics"
description: "Topic hubs for UK insurance pricing: conformal prediction, causal inference, model monitoring, SHAP relativities, and fairness/proxy discrimination. Open-source Python libraries with tutorials, benchmarks, and comparisons."
permalink: /topics/
---

Each hub collects every post on a topic in one place — tutorials, benchmarks, library comparisons — alongside the open-source library that implements it.

---

### [Conformal Prediction for Insurance](/topics/conformal-prediction/)

Distribution-free prediction intervals with finite-sample coverage guarantees. No distributional assumptions, no model-specific variance estimation. Works on Tweedie GBMs, frequency-severity models, and reserve ranges.

Library: [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal)

---

### [Causal Inference for Insurance Pricing](/topics/causal-inference/)

Double machine learning for deconfounding rating factors and estimating price elasticity. Causal forests for heterogeneous treatment effects. Synthetic DiD for rate change evaluation and FCA evidence packs.

Library: [`insurance-causal`](https://github.com/burning-cost/insurance-causal)

---

### [Insurance Model Monitoring](/topics/model-monitoring/)

Exposure-weighted PSI and CSI, A/E ratios with Poisson confidence intervals, Gini drift z-tests, and double-lift monitoring for deployed pricing models. Layered detection that fires months before your A/E ratio moves.

Library: [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring)

---

### [SHAP Relativities for Insurance](/topics/shap-relativities/)

Multiplicative rating factor tables from CatBoost models in GLM exp(beta) format. Confidence intervals, exposure weighting, and direct export to Radar and Emblem. GBM-to-GLM distillation for rating engine deployment.

Library: [`shap-relativities`](https://github.com/burning-cost/shap-relativities)

---

### [Insurance Fairness and Proxy Discrimination](/topics/fairness-proxy-discrimination/)

Proxy discrimination auditing aligned to FCA EP25/2 and Consumer Duty. Optimal transport discrimination-free pricing. Fairness testing without sensitive attribute data. Structured documentation output.

Library: [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness)
