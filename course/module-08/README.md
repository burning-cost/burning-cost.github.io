# Module 8: End-to-End Pricing Pipeline

**Part of: Modern Insurance Pricing with Python and Databricks**

---

## The one that ties it together

Modules 1 through 7 each cover a distinct part of the pricing workflow. Delta tables. GLMs. Temporal cross-validation. SHAP relativities. Conformal uncertainty. Credibility blending. Rate optimisation. Each works in isolation. None of them, alone, is a pricing pipeline.

This module builds the pipeline.

We go from raw data in Unity Catalog to a rate change recommendation ready for a pricing committee. Every component from the earlier modules appears: the feature transform layer, the CatBoost frequency-severity models, walk-forward CV, SHAP-derived relativities, conformal intervals, Buhlmann-Straub credibility blending, and the constrained rate optimiser with FCA compliance built in.

The notebook you produce at the end of this module is not a demonstration. It is a working template for a real motor pricing project. We have structured it so that you can swap out the synthetic data for your own portfolio and run the pipeline with minimal modification.

---

## What you will build

- A Unity Catalog schema with Delta tables for raw data, feature views, model outputs, and the rate change pack
- A reproducible feature transform layer: all transforms defined as pure functions, versioned alongside the model
- Walk-forward cross-validation using `insurance-cv` with IBNR buffers, to validate frequency and severity models on temporally correct splits
- Separate CatBoost Poisson (frequency) and CatBoost Gamma (severity) models, hyperparameter-tuned with Optuna and tracked in MLflow
- SHAP relativities extracted with `shap-relativities`, one table per rating factor
- Conformal prediction intervals on pure premium using `insurance-conformal`, calibrated on a held-out conformity set
- Credibility-blended relativities using `credibility` (Buhlmann-Straub), balancing model-derived relativities against incumbent rates
- A constrained rate optimisation using `rate-optimiser` with the FCA PS21/5 ENBP constraint and the efficient frontier
- A final output pack written to Delta tables: relativities table, rate change summary, frontier data, model diagnostics

---

## Prerequisites

- Completed Modules 1-7, or comfortable with: GLM/GBM pricing, Delta tables and Unity Catalog, SHAP for tree models, credibility theory, and basic constrained optimisation
- Python comfortable. You will be reading and adapting code, not writing from scratch
- A Databricks workspace with Unity Catalog enabled. Databricks Free Edition works for everything here

If you have not done the earlier modules, the tutorial explains every component as it appears. You will not be lost. But the earlier modules give you the why behind each decision; this one concentrates on the how of connecting them.

---

## Estimated time

6-8 hours for the full tutorial and exercises. Most of that is reading and understanding the pipeline structure. The code itself - once you have run it once - takes around 12-15 minutes to execute end to end on a single-node cluster. The walk-forward CV is the slowest component.

---

## Files

| File | Purpose |
|------|---------|
| `tutorial.md` | Main written tutorial - read this first |
| `notebook.py` | Databricks notebook - the complete pipeline in runnable form |
| `exercises.md` | Four exercises extending the pipeline |

---

## Libraries

All from Burning Cost open-source repositories:

```bash
# Local development
uv add insurance-cv shap-relativities insurance-conformal credibility rate-optimiser catboost polars optuna mlflow
```

On Databricks, the first notebook cell:

```python
%pip install \
  "insurance-cv @ git+https://github.com/burningcost/insurance-cv.git" \
  "shap-relativities[catboost] @ git+https://github.com/burningcost/shap-relativities.git" \
  "insurance-conformal[catboost] @ git+https://github.com/burningcost/insurance-conformal.git" \
  "credibility @ git+https://github.com/burningcost/credibility.git" \
  "rate-optimiser @ git+https://github.com/burningcost/rate-optimiser.git" \
  catboost polars optuna mlflow --quiet
```

---

## What you will be able to do after this module

- Build a complete motor pricing pipeline that a real pricing team could inherit and run
- Explain every decision in the pipeline: why walk-forward CV, why separate frequency and severity, why Buhlmann-Straub for credibility, why SLSQP for the optimisation
- Structure a Databricks notebook so that a colleague who did not write it can understand what each stage does and why
- Write a rate change pack to Delta tables with a full audit trail: what the model recommended, how confident we are, and what the optimiser decided
- Adapt the template for a different line of business or a different model type

---

## What makes this different from a demo

Most end-to-end pricing examples online cut corners. They use a flat DataFrame throughout with no data layer. They skip cross-validation. They have one model, not frequency and severity separately. They do not produce uncertainty intervals. They have no credibility step. And they hand-wave the rate change decision.

This pipeline does not cut those corners. Each stage corresponds to a real decision a pricing team makes. The output is something you could present to an underwriting director and defend.

---

## Pricing context

This module is the full pricing workflow, not just the modelling piece. The question it answers is: given the data we have, what rate action should we take, and how confident should we be in that recommendation? That requires modelling, uncertainty quantification, credibility judgement, and formal optimisation. All four are here.

---

## Part of the MVP bundle

Module 8 is included in the £395 full course bundle. Individual module: £99.
