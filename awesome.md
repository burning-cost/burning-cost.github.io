---
layout: page
title: "Open-Source Tools for Insurance Pricing in Python"
description: "A curated list of open-source Python tools for non-life insurance pricing — rating factor analysis, fairness testing, causal inference, model monitoring, conformal prediction, credibility, and more."
permalink: /awesome/
---

The Python ecosystem for insurance pricing has grown substantially since 2022. This page lists the tools we consider worth knowing about — our own libraries alongside the broader open-source landscape. We include external tools where they are genuinely the best available option, and we note where gaps remain.

All tools listed are open-source and Python-based unless stated otherwise. Links go to GitHub repositories.

---

## Rating Factor Analysis & Interpretability

Tools for extracting, inspecting, and communicating rating factors from pricing models.

| Tool | Description |
|---|---|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) (Burning Cost) | Extracts GLM-style multiplicative rating relativities from CatBoost GBMs using SHAP values. Benchmarked at +2.85pp Gini lift over direct GLM on synthetic UK motor data. |
| [insurance-distill](https://github.com/burning-cost/insurance-distill) (Burning Cost) | GBM-to-GLM distillation — fits a surrogate Poisson/Gamma GLM to GBM predictions and exports multiplicative factor tables for Radar/Emblem rating engines. 90–97% R² match on benchmarks. |
| [insurance-gam](https://github.com/burning-cost/insurance-gam) (Burning Cost) | EBM and Neural Additive Model for interpretable pricing — shape functions per rating factor give the transparency of a GLM with GBM-level predictive power. Includes exact Shapley values and factor table output. |
| [SHAP](https://github.com/shap/shap) | The standard Python library for Shapley value explanations. Works with CatBoost, XGBoost, LightGBM, and scikit-learn models. The starting point before reaching for anything more specialised. |
| [interpret](https://github.com/interpretml/interpret) | Microsoft's Explainable Boosting Machine (EBM) implementation. GAM with automatic pairwise interaction detection. Directly usable in insurance pricing without a wrapper. |
| [glum](https://github.com/Quantco/glum) | QuantCo's high-performance GLM library. The correct choice for Poisson/Gamma/Tweedie GLMs in Python — faster than statsmodels, proper exposure offsets, L1/L2/elastic-net regularisation, formula interface. ~130k PyPI downloads/month as of March 2026. v3.2.0 adds Polars support. |
| [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools) (Burning Cost) | GLM tooling — nested GLM embeddings, R2VF factor level clustering, territory banding, SKATER spatial clustering. Complements glum rather than replacing it. |

---

## Fairness & Discrimination Testing

Tools for proxy discrimination auditing and algorithmic fairness in pricing.

UK context: the FCA's Consumer Duty and Equality Act 2010 create specific obligations around indirect discrimination. Generic fairness libraries were built for binary classification without exposure weighting — they require adaptation for Poisson/Gamma pricing models.

| Tool | Description |
|---|---|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) (Burning Cost) | Proxy discrimination auditing aligned to FCA Consumer Duty and Equality Act 2010. Exposure-weighted bias metrics in multiplicative (log-space) models. Proxy R² method catches postcode proxies that Spearman correlation misses entirely (r²=0.78 vs r=0.06 in benchmarks). |
| [Fairlearn](https://github.com/fairlearn/fairlearn) | Microsoft's fairness library. Covers demographic parity, equalized odds, and mitigation algorithms. Built for binary classification — the metrics require adaptation for insurance regression, and there is no exposure weighting. Useful as a secondary check on classification sub-models. |
| [AIF360](https://github.com/Trusted-AI/AIF360) | IBM's AI Fairness 360 toolkit. 70+ fairness metrics, pre/in/post-processing bias mitigations. Same caveat as Fairlearn: designed for classification. Documentation and maintenance quality has declined since 2023. |

---

## Causal Inference

Tools for moving beyond correlation — deconfounding rating factors, measuring price elasticity, and evaluating rate changes.

| Tool | Description |
|---|---|
| [insurance-causal](https://github.com/burning-cost/insurance-causal) (Burning Cost) | Double machine learning (DML) for deconfounding rating factors, plus causal forest for heterogeneous treatment effects. Use DML for portfolio-level average effects; causal forest for segment-level CATEs with n≥2,000 per group. Includes price elasticity estimation. |
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) (Burning Cost) | Synthetic difference-in-differences for evaluating rate changes — event study, HonestDiD sensitivity bounds, FCA evidence pack output. v0.2.0 adds DoublyRobustSCEstimator: 24% lower RMSE than SDID with few comparison groups. |
| [EconML](https://github.com/py-why/EconML) | Microsoft's econometric causal ML library. The reference implementation of double machine learning, causal forests, and IV methods. insurance-causal builds on EconML and adds insurance-specific wrappers, confounding diagnostics, and documentation for pricing use cases. |
| [DoWhy](https://github.com/py-why/dowhy) | PyWhy's causal reasoning library. DAG-based identification, refutation tests, and causal discovery. Useful for formalising causal assumptions before fitting a DML model. |
| [CausalML](https://github.com/uber/causalml) | Uber's uplift modelling library — T-learner, S-learner, X-learner, causal forests. Primarily built for marketing/conversion uplift; works for price response estimation with adaptation. |

---

## Model Monitoring & Drift Detection

Tools for detecting when a deployed pricing model has gone stale.

Insurance-specific note: generic drift tools do not implement exposure-weighted PSI, actual-vs-expected ratios, or Gini drift tests. They are useful for feature drift detection but insufficient for full pricing model monitoring.

| Tool | Description |
|---|---|
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) (Burning Cost) | Exposure-weighted PSI/CSI, A/E ratios with Garwood CIs, Gini drift z-test, and PITMonitor for calibration drift via e-process martingale. mSPRT sequential testing holds 1% FPR where peeking t-tests reach 25%. InterpretableDriftDetector attributes drift to feature interactions with BH FDR control. v0.8.0. |
| [Evidently](https://github.com/evidentlyai/evidently) | Apache 2.0. 100+ metrics for data quality, feature drift (PSI, KS, Wasserstein), and model performance. Dashboard UI, MLflow integration. v0.7.20 as of January 2026. Pivoting toward LLM observability — insurance teams are not the primary audience. Useful for feature distribution monitoring. |
| [NannyML](https://github.com/nannyml/nannyml) | Apache 2.0. Confidence-Based Performance Estimation (CBPE) estimates model performance without ground-truth labels. Univariate and multivariate drift. v0.13.1, July 2025. CBPE applies to calibrated classifiers — not directly to Poisson/Gamma regression. Useful for binary conversion model monitoring. |

---

## Conformal Prediction

Distribution-free prediction intervals with finite-sample coverage guarantees.

| Tool | Description |
|---|---|
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) (Burning Cost) | Five non-conformity scores tuned for Tweedie and Poisson claims. The `pearson_weighted` score gives 13.4% narrower intervals than parametric Tweedie at identical 90% coverage. Frequency-severity conformal intervals, online retrospective adjustment (RetroAdj), Solvency II SCR bounds. v0.6.0. |
| [insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts) (Burning Cost) | Conformal prediction for non-exchangeable claims time series — ACI, EnbPI, SPCI, MSCP, Poisson/NB non-conformity scores. |
| [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) | The standard Python conformal prediction library. v1.3.0. Works with any sklearn-compatible estimator. The natural starting point — insurance-conformal adds the Tweedie/Poisson-specific non-conformity measures and insurance regulatory output that MAPIE does not include. |
| [crepes](https://github.com/henrikbostrom/crepes) | Conformal regressors and predictive systems. Simpler API than MAPIE, useful for normalised conformal prediction and Mondrian (conditional coverage) approaches. |

---

## Model Governance & Validation

Tools for structured model validation and regulatory documentation.

| Tool | Description |
|---|---|
| [insurance-governance](https://github.com/burning-cost/insurance-governance) (Burning Cost) | PRA SS1/23-compliant model validation reports. Bootstrap Gini CI, Poisson A/E CI, double-lift charts, renewal cohort test. HTML/JSON output structured for model risk committees. Catches miscalibration that manual checklists miss. |

No comparable open-source tools exist for insurance-specific model governance as of March 2026. Generic ML model cards (e.g., Google's Model Card Toolkit) do not cover actuarial validation tests.

---

## Credibility & Experience Rating

Classical actuarial credibility methods in Python.

| Tool | Description |
|---|---|
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) (Burning Cost) | Bühlmann-Straub credibility with mixed-model equivalence checks, Bayesian experience rating, and individual experience rating (static, dynamic, surrogate, and deep attention variants). 6.8% MAE improvement on thin schemes in benchmarks. |

No significant external open-source Python credibility libraries exist. `chainladder-python` (CAS) covers claims reserving but not rating credibility.

---

## Portfolio Optimisation

Constrained rate optimisation subject to profitability, retention, and regulatory constraints.

| Tool | Description |
|---|---|
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) (Burning Cost) | SLSQP constrained rate optimisation with analytical Jacobians, FCA ENBP constraints, efficient frontier, and ParetoFrontier for multi-objective optimisation across profit, retention, and fairness. Includes demand modelling (conversion/retention elasticity). Benchmarked at +143.8% profit lift over flat rate loading. |

No open-source Python insurance rate optimisation library existed before insurance-optimise. Financial portfolio optimisers (PyPortfolioOpt, skfolio) use Markowitz/HRP methods that do not transfer to insurance pricing constraints.

---

## Cross-Validation for Insurance

| Tool | Description |
|---|---|
| [insurance-cv](https://github.com/burning-cost/insurance-cv) (Burning Cost) | Temporal walk-forward cross-validation respecting policy time structure, IBNR buffers, and sklearn-compatible scorers. Walk-forward detects 10.5% optimism that k-fold hides on insurance data. |

Standard scikit-learn `TimeSeriesSplit` does not handle IBNR buffers or exposure-weighted insurance scoring. insurance-cv adds these on top of a sklearn-compatible API.

---

## Model Deployment & Champion/Challenger

| Tool | Description |
|---|---|
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) (Burning Cost) | Champion/challenger routing with shadow mode, SHA-256 deterministic assignment, SQLite quote log, bootstrap LR test, and ENBP audit trail. |

MLflow handles experiment tracking and model registry but does not implement champion/challenger routing with insurance-specific KPI tracking. BentoML and Seldon handle serving but not A/B routing with quote-level logging.

---

## Datasets

Public datasets for benchmarking insurance pricing models.

| Tool / Dataset | Description |
|---|---|
| [insurance-datasets](https://github.com/burning-cost/insurance-datasets) (Burning Cost) | Synthetic UK motor and home portfolios with known data-generating process parameters. Use this to verify that your model recovers true relativities before applying it to real data. Polars output supported. |
| [freMTPL2freq / freMTPL2sev](https://www.openml.org/search?type=data&id=41214) via OpenML | French motor third-party liability dataset. The standard public benchmark for insurance GLMs and GBMs. ~678k policies, claim frequency and severity. Available via `sklearn.datasets.fetch_openml` or directly from OpenML. |
| [CASdatasets](http://cas.uqam.ca/) | R package containing 40+ actuarial datasets (motor, property, health). Python users must read the CSV exports. Maintained by Charpentier (UQAM). No Python package. |

---

## General ML Tools Commonly Used in Pricing

These are not insurance-specific, but any pricing team working in Python will use them.

| Tool | Description |
|---|---|
| [CatBoost](https://github.com/catboost/catboost) | Yandex's gradient boosting library. The GBM of choice for insurance pricing — native categorical variable handling without one-hot encoding, built-in Tweedie/Poisson/Gamma objectives, fast CPU training. Consistently outperforms LightGBM on high-cardinality categorical data common in motor pricing. |
| [LightGBM](https://github.com/microsoft/LightGBM) | Microsoft's GBM library. Faster training than CatBoost on large datasets with few categoricals. The standard alternative when CatBoost is slow. |
| [glum](https://github.com/Quantco/glum) | Listed above under Rating Factor Analysis — worth repeating here. If you are fitting GLMs in Python, use glum, not statsmodels. |
| [Polars](https://github.com/pola-rs/polars) | Fast DataFrame library in Rust. Handles the 10M+ row portfolios where pandas is slow. Several Burning Cost libraries support `polars=True` output. v1.x API is stable. |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | Pipeline infrastructure, preprocessing, model selection, and scoring. The scaffolding most pricing libraries are built on. |
| [PyMC](https://github.com/pymc-devs/pymc) | Bayesian modelling in Python. Used in insurance-spatial (BYM2 territory models), bayesian-pricing, and insurance-credibility for hierarchical models. v5.x. |

---

*This list is maintained by [Burning Cost](https://burning-cost.github.io). Corrections and additions welcome — open an issue on any of our repositories.*
