---
layout: page
title: "shap-relativities"
description: "Convert GBM SHAP values into GLM-style multiplicative factor tables. Monotonicity constraints, credibility-weighted smoothing, RADAR-compatible output. The missing link between gradient boosting and insurance pricing sign-off."
permalink: /shap-relativities/
---

Convert gradient boosting SHAP values into GLM-style multiplicative relativity tables. The standard workflow for UK motor pricing teams migrating from GLMs to GBMs: fit CatBoost or XGBoost, extract SHAP, then produce the factor table format your pricing sign-off process expects.

Includes monotonicity enforcement, credibility-weighted smoothing via Whittaker-Henderson, RADAR-compatible export, and A/E diagnostics by factor level.

**[View on GitHub](https://github.com/burning-cost/shap-relativities)** &middot; **[PyPI](https://pypi.org/project/shap-relativities/)**
