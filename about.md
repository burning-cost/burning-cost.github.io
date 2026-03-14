---
layout: page
title: About
description: "Burning Cost is on the forefront of machine learning and data science research in UK personal lines insurance. We help pricing teams adopt best practice, best-in-class tooling, and Databricks."
permalink: /about/
---

Burning Cost is on the forefront of machine learning and data science research in UK personal lines insurance. We help pricing teams adopt best practice, best-in-class tooling, and Databricks.

The name comes from a basic actuarial concept: burning cost is claims incurred divided by premium earned. Simple, direct, no mystification. That is how we think about tooling.

---

## What we have built

35 Python libraries covering the full pricing workflow. See the [full library index with pip install commands](/tools/).

UK pricing teams have adopted GBMs (CatBoost is now the dominant choice for most new builds) but many are still taking GLM outputs to production because the GBM outputs are not in a form that rating engines, regulators, or pricing committees can work with. The tools here are about closing that gap — from raw data through to a signed-off rate change with an audit trail. All of it runs on Databricks.

**Data & Validation**

- [`insurance-cv`](https://github.com/burning-cost/insurance-cv) - temporal walk-forward cross-validation with IBNR buffers and sklearn-compatible scorers
- [`insurance-datasets`](https://github.com/burning-cost/insurance-datasets) - synthetic UK motor data with a known data-generating process, for testing and teaching
- [`insurance-synthetic`](https://github.com/burning-cost/insurance-synthetic) - vine copula synthetic portfolio generation preserving multivariate dependence structure
- [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) - distribution-free prediction intervals for insurance GBMs with finite-sample coverage guarantees
- [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) - exposure-weighted PSI/CSI, actual-vs-expected ratios, and Gini drift z-tests for deployed models
- [`insurance-governance`](https://github.com/burning-cost/insurance-governance) - structured PRA SS1/23 model validation reports and model risk management, output as HTML and JSON

**Model Building**

- [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility) - Buhlmann-Straub credibility in Python with mixed-model equivalence checks
- [`bayesian-pricing`](https://github.com/burning-cost/bayesian-pricing) - hierarchical Bayesian models for thin-data pricing segments using PyMC 5
- [`insurance-spatial`](https://github.com/burning-cost/insurance-spatial) - BYM2 spatial models for postcode-level territory ratemaking, borrowing strength from neighbours
- [`insurance-multilevel`](https://github.com/burning-cost/insurance-multilevel) - CatBoost combined with REML random effects for high-cardinality categorical groups
- [`insurance-trend`](https://github.com/burning-cost/insurance-trend) - loss cost trend analysis with structural break detection and regime-aware projections
- [`insurance-gam`](https://github.com/burning-cost/insurance-gam) - interpretable GAM models: EBM tariffs, actuarial NAM, and pairwise interaction networks with exact Shapley values
- [`insurance-interactions`](https://github.com/burning-cost/insurance-interactions) - automated GLM interaction detection using CANN, NID, and SHAP-based methods
- [`insurance-survival`](https://github.com/burning-cost/insurance-survival) - shared frailty models for recurrent claims, cure models, competing risks, and retention modelling

**Interpretation**

- [`shap-relativities`](https://github.com/burning-cost/shap-relativities) - multiplicative rating factor tables from CatBoost models via SHAP, in the same format as exp(beta) from a GLM
- [`insurance-causal`](https://github.com/burning-cost/insurance-causal) - causal inference via double machine learning for deconfounding rating factors

**Tail Risk & Distributions**

- [`insurance-distributional-glm`](https://github.com/burning-cost/insurance-distributional-glm) - GAMLSS for Python: model ALL distribution parameters as functions of covariates, seven families, RS algorithm
- [`insurance-severity`](https://github.com/burning-cost/insurance-severity) - severity modelling toolkit for UK non-life insurance: full predictive distributions per risk
- [`insurance-quantile`](https://github.com/burning-cost/insurance-quantile) - quantile and expectile GBMs for tail risk, TVaR, and increased limit factors
- [`insurance-distributional`](https://github.com/burning-cost/insurance-distributional) - distributional GBMs with Tweedie, Gamma, ZIP, and negative binomial objectives

**Commercial**

- [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) - constrained rate change optimisation with efficient frontier between loss ratio target and movement cap constraints
- [`insurance-demand`](https://github.com/burning-cost/insurance-demand) - demand modelling for conversion and retention, price response curves
- [`insurance-elasticity`](https://github.com/burning-cost/insurance-elasticity) - causal price elasticity estimation via CausalForestDML, DR-Learner, and Automatic Debiased ML
- [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility) - NCD and bonus-malus systems for UK motor, including claiming threshold optimisation
- [`insurance-survival`](https://github.com/burning-cost/insurance-survival) - cure models, customer lifetime value, lapse tables, and MLflow wrapper for retention modelling

**Compliance & Governance**

- [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) - proxy discrimination auditing, optimal transport discrimination-free pricing, and FCA Consumer Duty documentation support
- [`insurance-causal-policy`](https://github.com/burning-cost/insurance-causal-policy) - synthetic difference-in-differences for causal rate change evaluation and FCA evidence packs
- [`insurance-governance`](https://github.com/burning-cost/insurance-governance) - model risk management: ModelCard, ModelInventory, and GovernanceReport generation
- [`insurance-deploy`](https://github.com/burning-cost/insurance-deploy) - champion/challenger framework with shadow mode, rollback, and full audit trail

**Infrastructure**

- [`burning-cost`](https://github.com/burning-cost/burning-cost) - the Burning Cost CLI; orchestration for pricing model pipelines

---

## The problem we are solving

UK pricing teams have been building GBMs for years, mostly CatBoost. The models are better than the production GLMs. But many teams are still taking the GLM to production, because the GBM outputs are not in a form that a rating engine, regulator, or pricing committee can work with.

The issue is not technical skill. It is tooling. There is no standard Python library that extracts a multiplicative relativities table from a GBM. There is no standard library that does temporally-correct walk-forward cross-validation with IBNR buffers. There is no standard library that builds a constrained rate optimisation a pricing actuary can challenge. There is no standard library that generates a PRA SS1/23-compliant model validation report.

We wrote those libraries because we needed them. Then we kept going. Everything is built to run on Databricks — that is where UK pricing teams are working, and where our research demonstrates its best practice.

---

## Contact

**Email:** [pricing.frontier@gmail.com](mailto:pricing.frontier@gmail.com)

**GitHub:** [github.com/burning-cost](https://github.com/burning-cost)
