---
layout: page
title: "Causal Inference for Insurance Pricing"
description: "Double machine learning, causal forests, and synthetic DiD for UK insurance pricing. Deconfound rating factors, estimate price elasticity, and evaluate rate changes with statistical rigour. Python library."
permalink: /topics/causal-inference/
---

Correlation is cheap in insurance data. Every rating factor correlates with every other rating factor, and untangling which of them causes risk versus which merely predicts risk is not a statistical exercise you can do with a GLM. Double machine learning (DML) lets you estimate the causal effect of a specific factor — telematics score, NCD level, vehicle group — while flexibly controlling for confounders using a GBM. The result is a deconfounded coefficient with a standard error you can actually interpret.

The application that drives most adoption is price elasticity for PS21/5 compliance. OLS elasticity on formula-rated renewal data does not give you what the FCA needs. The premium affects who lapses, and lapse propensity is correlated with risk — so OLS conflates demand response with adverse selection. CausalForestDML separates these and gives you heterogeneous treatment effects: the elasticity varies by customer segment, which is exactly the information a pricing actuary needs to set a compliant renewal uplift.

A third application is rate change evaluation. Standard before/after analysis of a rate change cannot isolate the premium increase from everything else that was changing at the same time. Synthetic difference-in-differences identifies a control group from your own book — policies that received a different rate movement — and estimates the causal impact of the treated cohort's uplift on lapses, conversions, and loss ratio.

**Library:** [insurance-causal on GitHub](https://github.com/burning-cost/insurance-causal) &middot; `pip install insurance-causal`

---

## Tutorials and introductions

- [Your Elasticity Estimate Is Biased and You Already Know Why](/2026/03/15/causal-price-elasticity-tutorial/) — the foundational tutorial: DML price elasticity on a UK renewal book, with FCA PS21/5 compliance framing
- [Causal AI for Pricing Actuaries: A Practical Guide](/2026/03/25/causal-ai-for-pricing-actuaries/) — broader introduction to causal methods for pricing teams: DML, causal forests, DiD, interrupted time series
- [OLS Elasticity in a Formula-Rated Book Measures the Wrong Thing](/2026/03/14/causal-price-elasticity-for-uk-renewal-pricing/) — why naive elasticity estimation fails and how DML fixes it

---

## Techniques and extensions

- [DML Works at 1,000 Policies Now. Here Is What Changed.](/2026/03/17/dml-small-samples-adaptive-regularisation/) — adaptive regularisation for thin segments where standard DML overfits the nuisance models
- [Your Pricing Model Knows the Average Effect. That Is Not Enough.](/2026/03/24/heterogeneous-treatment-effects-causal-forests-insurance/) — causal forests for heterogeneous treatment effects across the portfolio
- [Your Pricing Model Knows the Average. Your Customers Don't Care About the Average.](/2026/03/25/heterogeneous-price-elasticity-causal-forests-insurance-pricing/) — GATES and CLAN for characterising which segments are most price-sensitive
- [Heterogeneous Lapse Effects with Bayesian Causal Forests: Beyond the Average Elasticity](/2026/03/12/insurance-bcf/) — BCF with BART for credible intervals on heterogeneous effects in small samples (now in [insurance-causal](https://github.com/burning-cost/insurance-causal) as the `causal_forest` module)
- [Rate Change Evaluation: Did the Premium Increase Cause the Lapses?](/2026/03/10/rate-change-lapse-evaluation-causal-inference/) — applying DiD to lapse attribution after a renewal rate change
- [Synthetic Difference-in-Differences for Rate Change Evaluation](/2026/03/13/your-rate-change-didnt-prove-anything/) — SDiD using `insurance-causal-policy` for FCA evidence packs

---

## Benchmarks and validation

- [Double Machine Learning for Insurance Pricing: Benchmarks and Pitfalls](/2026/03/09/dml-insurance-benchmarks/) — simulation results on synthetic UK motor data with known treatment effects
- [Does DML causal inference actually work for insurance pricing?](/2026/03/25/does-dml-causal-inference-actually-work/) — empirical validation of DML on realistic insurance data
- [Does DML Causal Inference Actually Work for Insurance Pricing?](/2026/03/25/does-dml-causal-inference-actually-work/) — extended benchmark with GLM comparison

---

## Library comparisons

- [DoWhy vs insurance-causal: Which Causal Inference Library Should Insurers Use?](/2026/03/18/dowhy-vs-insurance-causal-inference-insurance-pricing/) — DoWhy's DAG-based approach versus DML for insurance pricing tasks
- [EconML vs insurance-causal: Causal Inference for Insurance Pricing](/2026/03/19/econml-vs-insurance-causal-inference-pricing/) — EconML is the upstream dependency; this explains what insurance-causal adds on top
