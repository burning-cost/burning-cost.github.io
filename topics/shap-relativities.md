---
layout: page
title: "SHAP Relativities for Insurance: GBM Factor Tables in GLM Format"
description: "Extract multiplicative rating factor tables from CatBoost models using SHAP. Same exp(beta) format as a GLM. Compatible with Radar, Emblem, and any rating engine that accepts factor imports. Python library."
permalink: /topics/shap-relativities/
---

Many UK pricing teams have a GBM sitting on a server somewhere that outperforms the production GLM. The model is not in production because nobody can get the relativities out of it. The regulator wants a factor table. Radar needs an import file. The pricing committee wants to challenge the numbers in terms they recognise — a relativity of 1.42 for young male drivers, not a SHAP waterfall plot.

SHAP values are the natural tool for this. The Shapley axioms guarantee that SHAP values sum exactly to the model output — which means you can exponentiate them to get multiplicative contributions, band them by rating factor, and average across exposure to produce a factor table in the same format as exp(beta) from a GLM. The [`shap-relativities`](https://github.com/burning-cost/shap-relativities) library does this in a way that is production-ready: confidence intervals, exposure weighting, consistency checks against the model's raw predictions, and output formats for Radar and Emblem import.

Extracting relativities is step one. The next step — for teams who want to move the GBM itself to production — is GBM-to-GLM distillation: fitting a surrogate GLM to the GBM's predictions and exporting a multiplicative factor schedule that a rating engine can execute directly. The [`insurance-distill`](https://github.com/burning-cost/insurance-distill) library handles the distillation step.

**Library:** [shap-relativities on GitHub](https://github.com/burning-cost/shap-relativities) &middot; `pip install shap-relativities`

**Related library:** [insurance-distill on GitHub](https://github.com/burning-cost/insurance-distill) &middot; `pip install insurance-distill`

---

## Tutorials and introductions

- [SHAP Relativities for Insurance GBMs: GLM-Format Factor Tables in Python](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/) — the foundational tutorial: extracting a motor frequency factor table from a CatBoost model
- [How to Extract GLM-Style Rating Factors from a CatBoost Model](/2026/03/02/how-to-extract-rating-factors-from-catboost/) — step-by-step guide with exposure weighting and confidence intervals
- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/) — end-to-end workflow from fitted model to Radar-importable factor schedule

---

## GBM-to-GLM distillation

- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/) — distillation walkthrough using `insurance-distill`
- [From GBM to Radar: A Complete Databricks Workflow for Pricing Actuaries](/2026/02/21/from-gbm-to-radar-databricks-workflow/) — production-grade workflow including MLflow tracking and Unity Catalog registration
- [Blending GLMs and GBMs for UK Pricing: Cross-Validated Weights, Not a Choice Between Them](/2026/02/28/your-gbm-and-glm-are-not-competitors/) — using SHAP to blend GLM and GBM predictions with cross-validated mixing weights

---

## Validation and benchmarks

- [Does GBM-to-GLM Distillation Actually Work for Insurance Pricing?](/2026/03/24/does-gbm-glm-distillation-work-insurance-pricing/) — empirical comparison: distilled GLM lift versus raw GBM lift on held-out data
