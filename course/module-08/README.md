# Module 8: End-to-End Pricing Pipeline

**Part of: Modern Insurance Pricing with Python and Databricks**

---

## What this module covers

Everything you have built across Modules 1–7 — Delta tables, walk-forward cross-validation, CatBoost GBMs, conformal prediction intervals, constrained rate optimisation — runs in isolation in its own notebook. This module ties it together into a single reproducible pipeline: one notebook, one MLflow experiment, one Unity Catalog schema, one audit record that lets you reproduce any output at any future date.

The pipeline covers every stage of a UK personal lines rate review: loading 200,000 synthetic motor policies across four accident years, feature engineering with a shared transform layer (the same functions used at training time and scoring time), walk-forward cross-validation with a six-month IBNR buffer using the `insurance-cv` library, Optuna hyperparameter tuning, final CatBoost Poisson frequency and Tweedie severity models logged to MLflow, conformal prediction intervals calibrated and validated using `insurance-conformal`, a constrained rate optimisation using `insurance-optimise`, and a pipeline audit record that ties every Delta table version and every MLflow run ID to the same run date.

The audit record is not an afterthought. Consumer Duty (PS 22/9, effective July 2023) and PS 21/5 together require that pricing teams can reproduce the model, the inputs, and the decision trail for any price charged in the past three years. The pipeline is designed so that this is automatic: every table write carries the MLflow run ID; every MLflow run carries the Delta table version used for training; the audit record captures all of this in a single queryable row. To reproduce any historical output, you query the audit table, read the Delta table at the logged version, and load the MLflow model at the logged run ID.

The feature engineering discipline — all transforms defined as pure functions in a single dictionary, applied identically at training time and scoring time — is the most practically important contribution of this module. The single most common failure mode in production ML-based pricing is a mismatch between training-time and scoring-time feature engineering. A model trained with NCB encoded as a string categorical and scored with NCB as an integer will produce systematically wrong predictions, silently, with no exception raised. The pipeline's transform layer prevents this structurally.

---

## Prerequisites

- All previous modules completed, or equivalent working knowledge:
  - Module 1: Databricks, Delta, Unity Catalog, MLflow basics
  - Module 2: GLMs in Python with statsmodels
  - Module 3: GBMs with CatBoost, Poisson frequency, Tweedie severity
  - Module 4: SHAP relativities (helpful but not required for the pipeline itself)
  - Module 5: Conformal prediction intervals with `insurance-conformal`
  - Module 6: Credibility and Bayesian pricing (context, not a code dependency)
  - Module 7: Constrained rate optimisation with `insurance-optimise`
- A Databricks workspace with a cluster running DBR 14.0 ML or later
- Unity Catalog enabled on your workspace (required for the Delta table writes)

---

## Estimated time

5–6 hours for the tutorial plus exercises. The notebook runs in 45–60 minutes on a 4-core cluster (Standard_DS3_v2 or equivalent). The Optuna tuning is the longest step; set `N_OPTUNA_TRIALS = 10` for a faster initial run.

---

## Files

| File | Purpose |
|------|---------|
| `tutorial.md` | Main written tutorial — read this first |
| `notebook.py` | Databricks notebook — full nine-stage pipeline on 200,000 synthetic policies |
| `exercises.md` | Four exercises covering failure modes, Delta versioning, and pipeline governance |

---

## Libraries

All five Burning Cost libraries plus CatBoost, Optuna, and MLflow:

```bash
pip install \
  catboost \
  "insurance-conformal[catboost]" \
  insurance-cv \
  insurance-optimise \
  polars \
  optuna \
  mlflow \
  --quiet
```

Sources:
- [github.com/burning-cost/insurance-cv](https://github.com/burning-cost/insurance-cv)
- [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal)
- [github.com/burning-cost/insurance-optimise](https://github.com/burning-cost/insurance-optimise)

---

## What you will be able to do after this module

- Structure a pricing pipeline so that all configurable values (catalog, schema, LR target, conformal alpha) live in one cell and everything else flows from them
- Define feature engineering as a shared transform layer — a list of pure functions applied identically at training and scoring time — and understand why this prevents the most common production failure in ML pricing
- Run walk-forward cross-validation with an IBNR buffer on multi-year claims data, and explain why random splits produce metrics that do not reflect out-of-time generalisation
- Tune CatBoost hyperparameters with Optuna on a temporal validation set and log the full search to MLflow
- Train final frequency and severity models, log them to MLflow with explicit provenance (training table version, feature list, CV strategy), and load them back for scoring
- Calibrate and validate conformal prediction intervals using `insurance-conformal` and interpret coverage-by-decile diagnostics
- Chain model outputs into the rate optimiser: convert frequency predictions to technical premiums, build the renewal portfolio, and run the ENBP-constrained SLSQP solve
- Write a pipeline audit record that ties every table version and model run ID to a single queryable row, enabling exact reproduction of any historical output
- Adapt the pipeline for a different line of business by changing the configuration cells and the data generation cell, leaving all other stages unchanged

---

## Why this is worth your time

The individual modules are valuable in isolation. A better GLM, a calibrated GBM, an ENBP-compliant rate optimisation — each of these is a genuine improvement over the status quo. But the sum is worth more than the parts only if the pipeline holds together.

In practice, the failure modes that matter most in production pricing are not modelling failures — they are integration failures. The GBM trained with NCB as a categorical variable, scored with NCB as an integer. The conformal intervals calibrated on one year's data, applied to a portfolio whose claim severity has inflated 15% since calibration. The rate optimisation that assumes a renewal portfolio that does not match what the model actually predicted. These failures are silent. They do not raise exceptions. They produce numbers that look plausible and are wrong.

The pipeline in this module is designed specifically to prevent those failures. Feature engineering is defined once and applied everywhere. Conformal intervals are validated before use. The rate optimiser ingests actual model predictions, not a parallel calculation. Every output is tagged with the exact version of the data it was produced from.

This is what FCA-ready pricing infrastructure looks like: not a collection of well-built components, but a well-built system in which the components are connected in a way that makes failures detectable and outputs reproducible.
