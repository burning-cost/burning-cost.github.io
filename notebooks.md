---
layout: page
title: "Databricks Notebook Archive"
description: "10 production-ready Databricks notebooks for UK insurance pricing. Download the full archive or browse individual notebooks for Bayesian models, causal inference, fairness audits, and more."
permalink: /notebooks/
---

Ten Databricks notebooks from the Burning Cost library stack, covering the full pricing workflow from data generation to model deployment. Each notebook runs top-to-bottom on synthetic UK motor data with no external dependencies beyond what it installs with `%pip`.

<a href="/assets/notebooks/burning-cost-databricks-notebooks.zip" class="download-btn">Download all 10 notebooks (.zip, 53 KB)</a>

---

## How to use these

1. Download a `.py` file below (or the full `.zip`)
2. In Databricks: **Workspace &rarr; Import &rarr; File** — select the `.py` file
3. Databricks will recognise the `# Databricks notebook source` header and render the markdown cells
4. Run `Run All` — the first cell does `%pip install` so no cluster setup is needed

Notebooks are tested on Databricks Runtime 14.x+ with serverless SQL compute. They work on any standard cluster with Python 3.11+.

---

## The notebooks

### Foundation

**[01 — End-to-end motor pricing workflow](/assets/notebooks/01_end_to_end_motor_pricing.py)**
The full pipeline in one script: synthetic portfolio generation, CatBoost frequency model, SHAP relativities, PRA validation report, and champion/challenger deployment. Start here if you want to see how the libraries connect.
Uses: `insurance-synthetic`, `catboost`, `shap-relativities`, `insurance-validation`, `insurance-deploy`

**[02 — Synthetic portfolio generation](/assets/notebooks/02_synthetic_portfolio_generation.py)**
Build a realistic UK motor book using vine copulas that preserve multivariate dependence structure. Fit the synthesizer, generate 50k policies, and assess fidelity with `SyntheticFidelityReport`. Useful for sharing data with vendors without moving real policyholder records.
Uses: `insurance-synthetic`

---

### Frequency and severity modelling

**[03 — Bayesian hierarchical frequency model](/assets/notebooks/03_bayesian_hierarchical_frequency.py)**
Hierarchical Bayesian model with partial pooling for sparse rating cells. Pathfinder for fast variational inference, NUTS for the posterior you actually want. Extracts multiplicative relativities in rate-table format and flags thin segments for manual review.
Uses: `bayesian-pricing`, `pymc`, `arviz`

**[04 — Bühlmann-Straub credibility](/assets/notebooks/04_buhlmann_straub_credibility.py)**
Polars-native Bühlmann-Straub credibility for account-level and segment-level experience rating. The actuarial workhorse for blending individual experience with the collective prior. Sklearn-compatible scorer included.
Uses: `credibility`

---

### Model structure

**[05 — Automated GLM interaction detection](/assets/notebooks/05_glm_interaction_detection.py)**
CANN + Neural Interaction Detection (NID) pipeline that finds the interactions your Poisson GLM missed. Ranks candidates by NID score, validates survivors with likelihood-ratio tests. Built into data: `age_band × vehicle_group` is a real interaction — watch the library find it.
Uses: `insurance-interactions`

---

### Causal inference

**[06 — Causal deconfounding via double machine learning](/assets/notebooks/06_causal_deconfounding.py)**
DML for separating the causal effect of a rating factor from confounding driven by correlated features. Recovers the true ATE against a known data-generating process. The confounding bias report shows how much naive GLM estimates are distorted.
Uses: `insurance-causal`

---

### Compliance and governance

**[07 — Fairness and proxy discrimination audit](/assets/notebooks/07_fairness_proxy_audit.py)**
FCA Consumer Duty framing: proxy detection, disparate impact metrics, and counterfactual fairness tests. Produces the structured audit report you need before a rate change sign-off.
Uses: `insurance-fairness`

**[08 — Model drift monitoring](/assets/notebooks/08_model_drift_monitoring.py)**
CSI heatmap across rating factors, actual-vs-expected calibration by segment, and Gini drift z-test (arXiv 2510.04556). Scenario: frequency model fitted on 2022-2023 data, monitored on Q1 2025 data.
Uses: `insurance-monitoring`

**[09 — Champion/challenger deployment](/assets/notebooks/09_champion_challenger_deploy.py)**
Model registry, routing verification, shadow mode logging, KPI dashboard, bootstrap loss ratio comparison, and power analysis. ICOBS 6B.2.51R ENBP audit report included.
Uses: `insurance-deploy`

---

### Spatial

**[10 — BYM2 spatial territory ratemaking](/assets/notebooks/10_bym2_spatial_territory.py)**
Besag-York-Mollié BYM2 model for postcode-level territory rates. Borrows strength from neighbouring areas to handle sparse geographic cells. Full INLA-style workflow in Python.
Uses: `insurance-spatial`

---

All notebooks are MIT-licensed. Source repos live under [github.com/burning-cost](https://github.com/burning-cost). If a notebook is broken or out of date, raise an issue on the relevant repo.
