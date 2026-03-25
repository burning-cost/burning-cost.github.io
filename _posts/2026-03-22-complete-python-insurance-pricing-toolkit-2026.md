---
layout: post
title: "The Complete Python Insurance Pricing Toolkit: Every Library You Need in 2026"
date: 2026-03-22
categories: [guides]
tags: [python, insurance-pricing, ecosystem]
---

The Python insurance pricing ecosystem has grown fast enough that it is genuinely difficult to keep track of what exists. This is our attempt at a comprehensive, honest map - general-purpose tools alongside insurance-specific ones, with plain assessments of when each is the right choice.

We cover roughly 45 tools across 14 categories. For each: what it does, who maintains it, maturity, and when to reach for it instead of the alternative.

---

## 1. Core statistical modelling

These are the tools most teams use daily.

**statsmodels** - GLMs, GEEs, mixed models, and time-series in one package. Maintained by the statsmodels community; established. The standard choice for Poisson and Gamma GLMs: transparent, numerically stable, and produces the p-values and standard errors your technical reviewers expect. Its `GLM` class is the reference implementation for multiplicative pricing.

```bash
pip install statsmodels
```

**scikit-learn** - `TweedieRegressor`, `HistGradientBoostingRegressor`, and the full pipeline infrastructure. Maintained by a large INRIA-led community; established. Use it for GBM pipelines and when you need sklearn's ecosystem (`Pipeline`, `GridSearchCV`, cross-validators). Its `TweedieRegressor` is good for Tweedie-family GLMs but lacks the inference machinery statsmodels provides - you cannot get standard errors out of it.

```bash
pip install scikit-learn
```

**CatBoost** - gradient boosted trees with native ordered boosting and first-class categorical support. Maintained by Yandex; established. Our preferred GBM for insurance. Ordered boosting reduces target leakage from exposure-correlated categoricals (postcode, occupation) that standard gradient boosting handles poorly. Pass `loss_function='Tweedie'` for combined ratio targets or `loss_function='Poisson'` for frequency directly. CatBoost handles high-cardinality categoricals (ABI group, postcode sector) without preprocessing.

```bash
pip install catboost
```

**LightGBM** - fast gradient boosting with histogram approximation. Maintained by Microsoft; established. Faster to train than CatBoost on most datasets; competitive accuracy. The ordered boosting advantage goes to CatBoost for insurance data with correlated categoricals. Use LightGBM when training speed is the binding constraint or when you need its leaf-wise growth for large feature spaces.

```bash
pip install lightgbm
```

**XGBoost** - the original competitive GBM. Maintained by a large open-source community; established. Still widely used; strong distributed training story. We reach for CatBoost or LightGBM first in insurance contexts, but XGBoost's `Poisson` objective and `reg:tweedie` are production-grade and well-understood. Choose it if your team is already fluent in it - the differences are marginal on tabular insurance data.

```bash
pip install xgboost
```

---

## 2. Data handling

**Polars** - fast, memory-efficient dataframes backed by Rust and Apache Arrow. Maintained by Ritchie Vink and the Polars team; growing fast. Our recommendation for new work. Lazy evaluation means query plans are optimised before execution, which matters when you are joining 5m policy records with monthly claims triangles. The syntax is more explicit than pandas - which is a feature when you are debugging a complex frequency severity join.

```bash
pip install polars
```

**pandas** - the universal Python dataframe library. Maintained by a large community; established. Still the right choice when you need interoperability: most ML libraries produce or consume pandas DataFrames, and most example code is written against pandas. We use polars for heavy lifting and convert to pandas at the ML library boundary.

```bash
pip install pandas
```

**PyArrow** - columnar in-memory format and interop layer. Maintained by the Apache Arrow project; established. You need this whenever you are moving data between systems (Parquet files, Spark, Polars). Not something you typically use directly in a pricing model, but a required dependency of anything that reads Parquet.

```bash
pip install pyarrow
```

**insurance-datasets** (Burning Cost) - synthetic UK motor and home insurance datasets with published data-generating processes. New. Two datasets: UK personal lines motor (18 columns, Poisson-Gamma DGP) and household (16 columns). Both have known true coefficients, so you can verify that your model implementation recovers them. Use this for testing GLM or GBM implementations before your production data is available.

```bash
pip install insurance-datasets
```

---

## 3. Interpretability and factor extraction

**SHAP** - Shapley value attribution for any model. Maintained by Scott Lundberg and the SHAP team; established. The standard for understanding what a GBM is doing. Use `TreeExplainer` for CatBoost/LightGBM/XGBoost - it is exact and fast. The main insurance limitation: raw SHAP values are additive in log-space for log-link models, not in the multiplicative factor space that actuaries work in. That is the gap the next tool fills.

```bash
pip install shap
```

**shap-relativities** (Burning Cost) - converts SHAP values from a CatBoost model into GLM-style multiplicative factor tables. New. Extracts continuous relativity curves for numeric variables (e.g. a full driver age curve) and discrete loading tables for categoricals, in the same format as Emblem or Radar output. The honest assessment: this does not improve the model's predictive performance - the SHAP extraction introduces a small approximation (typically 5–7% mean error vs the true DGP on our synthetic benchmark). Its value is entirely in communicating GBM behaviour to actuaries and rating engine teams.

```bash
pip install shap-relativities
```

**EBM / InterpretML** - Explainable Boosting Machines: a GAM fitted by boosting that is inherently interpretable. Maintained by Microsoft; growing. Each feature gets its own shape function, shown as a graph. Strong on non-linear single-variable effects; interaction detection requires explicit configuration. This sits between a GLM and a black-box GBM in both accuracy and interpretability. We find it useful for initial factor analysis where you want to see the shape of a continuous rating variable without committing to a bin structure.

```bash
pip install interpret
```

**insurance-gam** (Burning Cost) - three production-grade interpretable alternatives to GLMs: Actuarial NAM (Neural Additive Model fine-tuned for exposure-weighted insurance data), PIN (Poisson Interaction Network for explicit two-way interactions), and EBM integration. New. Use NAM when the GLM's linear additive structure is leaving predictive performance on the table but the team needs to see individual factor shapes. Use PIN when you have a specific interaction hypothesis to test.

```bash
pip install "insurance-gam[ebm]"
```

---

## 4. Model distillation

**insurance-distill** (Burning Cost) - fits a Poisson or Gamma GLM surrogate to a GBM's predictions, with optimal binning, and exports factor tables in a format that Radar or Emblem can consume. New. In our testing on synthetic UK motor data the surrogate GLM retained 90–97% of the GBM's Gini coefficient. This is a pragmatic tool for teams whose rating engine cannot consume a GBM directly. If your organisation can deploy arbitrary Python scoring, use the GBM directly.

```bash
pip install insurance-distill
```

---

## 5. Fairness and discrimination

**Fairlearn** - general-purpose toolkit for assessing and mitigating algorithmic fairness issues. Maintained by Microsoft; established. Covers demographic parity, equalised odds, and a range of mitigation algorithms. The insurance gap: it has no concept of proxy discrimination (the scenario the FCA specifically investigates - where a factor correlates with a protected characteristic even when that characteristic is not in the model).

```bash
pip install fairlearn
```

**AI Fairness 360 (AIF360)** - IBM's broader fairness toolkit, covering more metrics and preprocessing/postprocessing mitigations than Fairlearn. Maintained by IBM Research; established. More comprehensive than Fairlearn; also heavier. Neither covers insurance-specific proxy discrimination detection out of the box.

```bash
pip install aif360
```

**insurance-fairness** (Burning Cost) - FCA-specific proxy discrimination analysis: proxy R-squared per rating factor, mutual information scores, SHAP-linked proxy impact, and a `FairnessAudit.run()` method that produces a Markdown report structured around FCA Consumer Duty expectations. New. Use this when you need to answer the specific question the FCA is asking about proxy discrimination in personal lines pricing. The general-purpose tools do not answer that question in the form regulators expect.

```bash
pip install insurance-fairness
```

**EquiPy** - equality of opportunity metrics for insurance. Community-maintained; dormant since 2023. We mention it for completeness but would not build on it given the maintenance status.

---

## 6. Causal inference

**EconML** - heterogeneous treatment effects via Double Machine Learning (DML), causal forests, and related methods. Maintained by Microsoft; growing. The right tool for estimating how a price or product change affected claims frequency when you have observational data and confounders. We use EconML's `LinearDML` and `CausalForestDML` for rate change evaluation where a controlled experiment was not run.

```bash
pip install econml
```

**DoWhy** - causal graph specification and identification. Maintained by Microsoft; growing. Useful for making the causal assumptions explicit (drawing the DAG) before running estimation. We use it as a conceptual layer on top of EconML rather than as a standalone estimation tool.

```bash
pip install dowhy
```

**insurance-causal** (Burning Cost) - DML with CatBoost for estimating price elasticity and treatment effects in insurance renewal portfolios, including a Riesz representer approach for continuous treatment (actual premium charged) that avoids the generalised propensity score instabilities common in renewal data. New. The Riesz representer is a practical advance over standard EconML for the renewal pricing case; we would not reach for this library for a general DML problem where EconML is sufficient.

```bash
pip install insurance-causal
```

**insurance-causal-policy** (Burning Cost) - Synthetic Difference-in-Differences (SDID) for evaluating the effect of rate changes across portfolio segments. New. SDID uses pre-treatment data to construct a synthetic control that removes market-wide inflation bias - a significant problem in naive before-after analysis of rate changes. On our synthetic benchmark (100 segments, true ATT = -0.08 on loss ratio), SDID bias was 0.0052 vs plain DiD bias of 0.0034 and naive before-after bias of 0.0378. For most well-designed rate changes, DiD is adequate; this library is the right choice when you cannot identify a clean control group.

```bash
pip install insurance-causal-policy
```

---

## 7. Model monitoring and drift detection

**Evidently** - open-source ML monitoring: data drift, model quality, and data quality reports. Maintained by Evidently AI; established. Comprehensive general-purpose monitoring. The insurance gap: it has no exposure-weighting (a PSI calculated on policy count is not the same as one calculated on exposure), and its out-of-the-box metrics are not aligned to insurance KPIs (Gini drift, A/E by segment).

```bash
pip install evidently
```

**NannyML** - performance monitoring without ground truth labels, using CBPE (Confidence-Based Performance Estimation). Maintained by NannyML; growing. Useful when you want to estimate model performance degradation before claims have developed. Again, no insurance-native metrics or exposure weighting.

```bash
pip install nannyml
```

**Alibi Detect** - statistical drift detection: MMD, KS, LSDD, and several others. Maintained by Seldon; established. Strong on the statistical testing side. More a toolkit than a monitoring system - you compose tests rather than getting reports.

```bash
pip install alibi-detect
```

**insurance-monitoring** (Burning Cost) - exposure-weighted PSI/CSI, Gini drift z-tests, A/E by segment, and structured audit trail output aligned with PRA SS3/17 model risk documentation requirements. New. If you are running standard drift detection on an insurance book and reporting to a model risk committee, this is the tool. For general ML monitoring outside insurance, Evidently is more mature.

```bash
pip install insurance-monitoring
```

---

## 8. Conformal prediction

**MAPIE** - Model Agnostic Prediction Interval Estimator: split conformal and cross-conformal prediction intervals for regression and classification. Maintained by Quantmetry; growing. The general-purpose implementation. Well-documented, scikit-learn compatible. For insurance: it supports custom non-conformity scores, which you need because the standard absolute residual score is wrong for skewed claim distributions.

```bash
pip install mapie
```

**insurance-conformal** (Burning Cost) - conformal prediction intervals with Tweedie and Poisson non-conformity scores, exposure weighting, and a locally-weighted variant for risk-segment-level calibration. New. On our benchmark (50,000 policies, heteroskedastic Gamma DGP), the parametric Tweedie approach had 13–14% wider intervals than conformal and systematically under-covered the top risk decile. Use this if you need per-segment coverage guarantees; MAPIE with a custom score is a reasonable alternative.

```bash
pip install insurance-conformal
```

**insurance-conformal-ts** (Burning Cost) - conformal prediction for insurance claims time series where the exchangeability assumption fails: seasonal loss patterns, market hardening, IBNR development. New. Standard conformal gives invalid intervals when the test distribution shifts - exactly what happens during market dislocations. This library implements methods for the non-exchangeable case.

```bash
pip install insurance-conformal-ts
```

---

## 9. Bayesian methods

**PyMC** - probabilistic programming in Python, with NUTS sampling. Maintained by the PyMC team; established. The standard Python Bayesian modelling library. Use it for hierarchical models (territory ratemaking, credibility blending) where you want full posterior distributions rather than point estimates and standard errors.

```bash
pip install pymc
```

**CmdStanPy** - Python interface to Stan, the mature probabilistic programming language. Maintained by the Stan development team; established. Stan's HMC sampler is often faster than PyMC for complex hierarchical models. The tradeoff is Stan's separate modelling language. We use CmdStanPy when the model is complex enough that Stan's performance advantage outweighs the DSL overhead.

```bash
pip install cmdstanpy
```

**insurance-spatial** (Burning Cost) - BYM2 (Besag-York-Mollié 2) Bayesian spatial model for UK territory ratemaking, implemented in PyMC. Includes spatially weighted conformal prediction intervals for geographic coverage guarantees under Consumer Duty. New. The alternative to grouping postcode sectors by k-means on loss ratios - BYM2 borrows strength across adjacent sectors and quantifies what fraction of geographic variation is genuinely spatial vs idiosyncratic noise.

```bash
pip install insurance-spatial
```

---

## 10. Survival analysis and retention

**lifelines** - survival analysis in Python: Kaplan-Meier, Cox regression, parametric models. Maintained by Cameron Davidson-Pilon; established. The standard Python survival library. Strong on the core methods. The insurance-specific gaps: `MixtureCureFitter` is univariate only (no covariate adjustment on the cure fraction), no Fine-Gray competing risks regression, no shared frailty for recurrent events.

```bash
pip install lifelines
```

**insurance-survival** (Burning Cost) - extends lifelines with three subpackages that fill confirmed Python ecosystem gaps: `cure` (covariate-adjusted mixture cure models for never-lapse estimation), `competing_risks` (Fine-Gray regression with IPCW weighting for lapse modelling where death and cancellation compete), and `recurrent` (shared frailty models for repeat-claim policyholders). New. If you are running standard time-to-lapse or time-to-first-claim analysis, lifelines is sufficient. Use this library when you hit one of those three specific gaps.

```bash
pip install insurance-survival
```

---

## 11. Governance and deployment

**MLflow** - experiment tracking, model registry, and deployment tooling. Maintained by Databricks; established. The standard for tracking model training runs, comparing Gini coefficients across experiments, and registering production models. Widely deployed in insurance IT stacks. Nothing insurance-specific, but if you are not using it you probably should be.

```bash
pip install mlflow
```

**insurance-governance** (Burning Cost) - model validation report generator (Gini, PSI, Hosmer-Lemeshow, lift charts in self-contained HTML), model risk management framework with `RiskTierScorer` (0-100 composite score mapping to Tier 1/2/3), `ModelInventory`, and `GovernanceReport` output aligned with UK insurance model governance requirements. New. The validation report is the most immediately useful part - it runs the standard suite of validation tests in one call. Use MLflow for experiment tracking; use this for the regulatory documentation layer.

```bash
pip install insurance-governance
```

---

## 12. Specialised pricing methods

**insurance-credibility** (Burning Cost) - Bühlmann-Straub group credibility (for fleet schemes and large accounts) and Bayesian individual experience rating (for NCD-like credibility at policy level). New. The Bühlmann-Straub formula is standard actuarial practice; this is the first clean Python implementation we are aware of.

```bash
pip install insurance-credibility
```

**insurance-dispersion** (Burning Cost) - Double GLM (DGLM) for joint modelling of mean and dispersion. A second regression model estimates the dispersion parameter phi per risk rather than assuming a single scalar across the book. New. Use this when you have systematic heteroskedasticity - broker vs direct business, fleet vs personal lines - that your single-dispersion model is misfitting.

```bash
pip install insurance-dispersion
```

**insurance-whittaker** (Burning Cost) - Whittaker-Henderson smoothing for experience rating tables. New. Every UK pricing team smooths age and vehicle-group curves. Most do it in Excel. This is the Python implementation, with automatic lambda selection via cross-validation.

```bash
pip install insurance-whittaker
```

**insurance-frequency-severity** (Burning Cost) - Sarmanov copula joint frequency-severity modelling. New. The standard two-GLM approach multiplies frequency and severity predictions assuming independence. When claim count and average severity are correlated (they are, for large fleets and commercial lines), this introduces systematic bias. The Sarmanov copula captures the dependency structure.

```bash
pip install insurance-frequency-severity
```

**insurance-telematics** (Burning Cost) - HMM-based driving regime classification and telematics scoring pipeline. New. Three-state HMM (cautious/normal/aggressive driving regimes) fitted to trip-level speed and acceleration data, with a `TelematicsScoringPipeline` that produces per-driver risk scores and a `TripSimulator` for prototyping without real telematics data.

```bash
pip install insurance-telematics
```

**insurance-eqrn** (Burning Cost) - Extreme Quantile Regression Neural Networks (Pasche & Engelke 2024, *Annals of Applied Statistics*) for covariate-dependent GPD parameters. New. For XL pricing and reinsurance layer costing where you need per-segment tail estimates rather than a single pooled GPD. If you only need portfolio-level VaR, standard EVT (scipy's `genpareto`) is sufficient.

```bash
pip install insurance-eqrn
```

**insurance-nflow** (Burning Cost) - normalising flows for severity distribution modelling. New. Drops the parametric family assumption (lognormal, Gamma) and learns the full conditional distribution from data, including bimodal structures and heavy tails via the Hickling-Prangle Tail Transform Flow (ICML 2025). The relevant use case is UK motor bodily injury severity, which is genuinely bimodal and poorly fit by any single parametric family.

```bash
pip install insurance-nflow
```

---

## 13. Optimisation

**insurance-optimise** (Burning Cost) - constrained rate optimisation with demand curve estimation (DML-based elasticity), ENBP (Expected Net Benefit of Pricing) metric, and FCA Consumer Duty constraint enforcement. New. The honest numbers from our benchmark: DML elasticity estimation on typical portfolio data with small price variation is imprecise. The optimiser's value is in the constrained maximisation step - even with an imprecise elasticity estimate, demand-curve-aware pricing outperformed flat loading by +143% mean profit per segment on synthetic PCW data.

```bash
pip install insurance-optimise
```

---

## 14. Cross-validation

**insurance-cv** (Burning Cost) - temporal walk-forward cross-validation for insurance pricing models, with splits that respect policy year, accident year, and IBNR development structure. New. Standard k-fold CV gives you overoptimistic scores on insurance data because it allows future accident years to appear in training folds. On our benchmark (20,000 policies, +20%/year claims trend), random k-fold overestimated CV score by approximately 8–12 Gini points relative to a true out-of-time holdout; temporal CV matched the holdout. Use this; k-fold on insurance data is not meaningful.

```bash
pip install insurance-cv
```

---

## How to choose

A few decision rules that apply across most of these categories:

**Use the general-purpose tool unless there is a specific insurance reason not to.** Fairlearn, Evidently, lifelines, MAPIE, and MLflow are all mature and well-documented. The Burning Cost libraries fill confirmed gaps; they are not replacements for the broader ecosystem.

**CatBoost for the primary GBM.** Ordered boosting and native categorical handling address genuine problems in insurance data. The training speed disadvantage vs LightGBM is real but usually not the binding constraint.

**Polars for data preparation.** The performance difference over pandas is large enough to matter on multi-million row policy files; the API is cleaner on joins and aggregations.

**Temporal cross-validation is not optional.** If your CV setup allows future data to appear in training, your reported Gini is wrong.

---

*This post will be updated as the ecosystem evolves. If we have missed a tool that deserves coverage, the repository is open.*

---

**Deep dives on the Burning Cost libraries:**

- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) - the first open-source Python implementation of the So & Valdez ASTIN 2024 Best Paper approach; per-risk CoV for safety loading and referrals
- [Quantile GBMs for Insurance: TVaR, ILFs, and Large Loss Loadings](/2026/03/07/insurance-quantile/) - conditional quantile regression for the tail: year-end large loss loadings and per-segment TVaR
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) - distribution-free coverage guarantees; the case for Tweedie non-conformity scores over raw residuals
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) - the FCA-specific proxy audit that Fairlearn does not cover
- [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/) - the joint optimisation problem that manual reconciliation cannot solve, with shadow prices and an efficient frontier
- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) - why exposure-weighted PSI and the Murphy decomposition matter; what Evidently and NannyML miss
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) - structured validation documentation and model inventory management aligned to UK regulatory expectations
- [Your Model Is Either Interpretable or Accurate. insurance-gam Refuses That Trade-Off.](/2026/03/14/insurance-gam-interpretable-nonlinearity/) - EBMs, ANAMs, and PIN as alternatives to the GLM/GBM binary
- [Whittaker-Henderson Smoothing for Insurance Pricing](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/) - REML lambda selection and confidence intervals for rating table graduation
- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) - why OLS renewal elasticity is structurally confounded and what DML does about it
