---
layout: post
title: "Our 12-Module Insurance Pricing Course Is Now Free and Open Source"
date: 2026-03-25
categories: [course, announcement, tutorial]
tags: [course, python, databricks, GLM, GBM, fairness, causal-inference, monitoring, governance, spatial, BYM2, open-source, free, insurance-causal, insurance-fairness, insurance-monitoring, insurance-governance, insurance-gam, insurance-optimise, tutorial]
description: "Modern Insurance Pricing with Python and Databricks - all 12 modules, free, on GitHub. GLMs through causal elasticity, fairness auditing, spatial BYM2 territory models, and model governance. For UK pricing teams who know Excel but want Python."
---

We have released the complete "Modern Insurance Pricing with Python and Databricks" course - all twelve modules - as a free, open-source repository on GitHub.

The course was previously sold as a paid product. We have removed the paywall. Everything is now at [github.com/burning-cost/course](https://github.com/burning-cost/course), MIT-licensed, no registration required.

---

## Why we built it

The standard data science curriculum does not prepare you for insurance pricing. Kaggle tutorials do not cover exposure offsets. Coursera MLOps content does not cover FCA Consumer Duty constraints. The textbooks cover GLM theory but not how to get a Tweedie GLM with temporal cross-validation past a model governance committee in 2026.

There is a specific gap between "knows Python" and "can run a pricing function in Python" — if you want an overview of the full stack before diving into modules, [the Python insurance pricing stack](/2026/03/14/the-python-insurance-pricing-stack/) maps out how the pieces fit together - and that gap costs pricing teams months of painful, trial-and-error self-education. We wanted to close it properly, not with a YouTube playlist.

The course is deliberately ground-zero. You need working Python. You do not need prior insurance experience. By the end you will have run a complete end-to-end pipeline: raw tabular data to filed rates, with MLflow tracking, fairness audit, and a governance sign-off trail.

---

## Who it is for

UK personal lines pricing actuaries and analysts who know Emblem, Radar, or Excel-based GLM workflows and want to move to Python. Also useful for data scientists joining insurance teams who understand ML but have not encountered insurance-specific constraints before: exposure offsets, temporal cross-validation, FCA ENBP limits, PS21/5 renewal pricing rules.

---

## The 12 modules

Every module uses real APIs from our open-source libraries. Nothing is invented for pedagogical convenience.

1. **Python and Polars for pricing data** - Working with large tabular policy and claims datasets using Polars. DataFrame operations, lazy evaluation, and the patterns that come up constantly in pricing workflows.

2. **Exploratory data analysis for insurance** - Frequency/severity decomposition, exposure-weighted distributions, bivariate lift charts, and the diagnostics that reveal data quality problems before they contaminate a model.

3. **GLMs in Python** - statsmodels Tweedie GLM with exposure offset, factor diagnostics, and lift charts. Everything Emblem does, in Python, with the same interpretation conventions actuaries expect.

4. **Gradient boosting with CatBoost** - Frequency/severity GBMs on freMTPL2, temporal walk-forward cross-validation, and hyperparameter search. Covers the validation approach that actually matters for pricing: out-of-time, not out-of-fold.

5. **Model monitoring** - PSI-based covariate drift detection, segment-level A/E monitoring, Gini stability over time, and mSPRT sequential testing for detecting when a rate change has had its full effect. Uses `insurance-monitoring`.

6. **Fairness auditing** - Proxy discrimination detection, demographic parity and equalised odds analysis under FCA Consumer Duty, and the difference between protected characteristics and actuarially justified risk factors. Uses `insurance-fairness`.

7. **Causal inference for pricing** - Double ML causal estimation of own-price elasticity for renewal pricing. Why naive retention models give biased elasticity estimates and how to fix it. Uses `insurance-causal`.

8. **End-to-end pipeline** - Raw tabular data to filed rates in a single Databricks workflow: feature engineering, GLM, GBM, blending, MLflow tracking, model registry, and a signed-off audit trail.

9. **Demand elasticity and rate optimisation** - Constrained SLSQP optimisation hitting a loss ratio target while respecting movement caps and FCA ENBP limits. Efficient frontier between loss ratio and volume. Uses `insurance-optimise`.

10. **Model governance** - What a model risk management framework looks like for a pricing model in a UK regulated insurer. PRA SS1/23 obligations, model owner documentation, and producing a governance record that survives review. Uses `insurance-governance`.

11. **Model deployment monitoring** - Post-deployment monitoring beyond training-time validation: production drift detection, alert thresholds, and the governance loop that connects a live model back to its validation record.

12. **Spatial and BYM2 territory modelling** - Geospatial smoothing with the BYM2 intrinsic CAR prior, Voronoi banding from postcode centroids, and borough-level risk surfaces. The right approach for territory rating when you have thin data at small geographies. Uses `insurance-spatial`.

---

## What is in the repository

Each module contains a Databricks notebook (`.ipynb`), a written tutorial in Markdown, and exercises with worked solutions. All notebooks have been verified against the current APIs of the relevant libraries - `insurance-causal`, `insurance-fairness`, `insurance-monitoring`, `insurance-governance`, `insurance-gam`, `insurance-optimise`, and `insurance-spatial`.

The notebooks run on Databricks. Most also run locally in Jupyter if you install the dependencies, but we have not tried to paper over the few places where Databricks-specific features (Unity Catalog, MLflow tracking server, Delta tables) are the right tool.

---

## Getting started

Clone the repository:

```bash
git clone https://github.com/burning-cost/course.git
cd course
```

The `README.md` has setup instructions for Databricks and for local Jupyter. Start with Module 1 if you are new to Polars. Start with Module 3 if you already work with tabular data in Python and want to get to the insurance-specific content quickly.

The [course landing page](/course/) gives an overview of all twelve modules.

---

We built this because we think pricing teams should be spending their time on hard technical problems, not on figuring out how to translate a textbook GLM into a Databricks notebook. The course is our attempt to compress that translation layer into something reusable.

If something is wrong, unclear, or out of date, open an issue on [GitHub](https://github.com/burning-cost/course).
