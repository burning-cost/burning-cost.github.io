# Module 3: GBMs for Insurance Pricing - CatBoost

**Part of: Modern Insurance Pricing with Python and Databricks**

---

## What this module covers

Gradient boosted machines have been used for insurance pricing in the UK since the mid-2010s, mostly at the more technically ambitious direct writers. The standard critique from actuaries trained on GLMs is that GBMs are a black box: you cannot read off a table of relativities, you cannot explain a particular customer's price to a regulator, and you cannot validate the direction of effects without specialist tooling. That critique was largely fair until about 2020. It is not fair now.

This module teaches you to train a CatBoost frequency-severity model on the same motor portfolio you built a GLM for in Module 2. CatBoost is our preferred GBM library for insurance pricing for two reasons: it handles categorical rating factors natively (no dummy encoding required) and its `Poisson` and `Tweedie` loss functions implement the correct actuarial objectives without any workarounds. We cover the full pipeline from data to MLflow registration: walk-forward cross-validation with an IBNR buffer using `insurance-cv`, hyperparameter tuning with Optuna across 40 trials, and the two diagnostics that determine whether the GBM actually adds anything over the GLM - the Gini coefficient comparison and the double lift chart. Both diagnostics are computed on the same temporal test set and logged to the same MLflow experiment, so the champion-challenger comparison is reproducible and auditable.

The module ends with the model registered in MLflow as a "challenger" - not "production". The governance gate is explicit: a challenger GBM does not touch the production pricing system until a pricing committee has reviewed the SHAP relativities (Module 4) and signed off.

---

## Prerequisites

- Modules 1 and 2 completed, or equivalent: you have a Databricks workspace with `pricing.motor.claims_exposure` in Unity Catalog, and you understand what a Poisson GLM with exposure offset does
- Comfortable with the GLM output: you know what a Gini coefficient is and what "actual versus expected by factor level" tells you
- Basic Python: you can read a function, follow a for loop, and modify parameters

You do not need prior CatBoost or Optuna experience. We introduce both APIs as we use them.

---

## Estimated time

5-6 hours for the tutorial plus exercises. The notebook runs end-to-end in 30-45 minutes on a 4-core cluster (Standard_DS3_v2 on Azure). The Optuna tuning step (40 trials) takes 15-20 minutes of that.

---

## Files

| File | Purpose |
|------|---------|
| `tutorial.md` | Main written tutorial - read this first |
| `notebook.py` | Databricks notebook - CatBoost frequency/severity pipeline, Optuna tuning, GLM comparison |
| `exercises.md` | Five hands-on exercises with full solutions |

---

## What you will be able to do after this module

- Fit a CatBoost Poisson frequency model with the correct exposure offset (`baseline=np.log(exposure)`) - the single most common implementation error, and the one that silently produces wrong predictions
- Pass categorical rating factors directly to CatBoost without dummy encoding, and understand why CatBoost's ordered target statistics are preferable to one-hot encoding for factors with 5-10 levels
- Implement walk-forward cross-validation with an IBNR buffer using `insurance-cv`, and explain why random train/test splits are not valid for insurance data
- Run 40-trial Optuna hyperparameter searches targeting Poisson deviance and identify which parameters actually matter (depth and learning rate dominate; l2_leaf_reg is secondary)
- Train the Gamma-equivalent severity model using `loss_function="Tweedie:variance_power=2"` on the claims-only subset, without an exposure offset, and understand why the offset belongs in the frequency model and not the severity model
- Compute a Gini coefficient comparison between the GBM and the GLM from Module 2, and build a double lift chart that shows where the GBM finds signal the GLM misses
- Log all parameters, metrics, and model artefacts to MLflow and register the challenger model using aliases (not the deprecated stage transition API)
- Explain to a pricing committee what the double lift chart shows and what Gini lift threshold justifies deployment

---

## Why this is worth your time

The GLM you built in Module 2 is correct. It will remain the production model for most UK personal lines risks, and it should: it is interpretable, auditable, and well-understood by your pricing committee, reserving team, and the FCA. The question this module answers is not "should we replace the GLM?" but "does the GBM find genuine additional risk signal, and if so, where?"

That is the right question because the business case for a GBM is marginal or zero on most well-developed UK motor books. GLMs with interaction terms can capture most of the signal a GBM finds, especially if your actuaries have been iterating on the model for years. But the double lift chart will tell you this directly: a flat chart means go back to the GLM. A positively sloping chart means the GBM is capturing interactions the GLM is missing - typically young driver x high vehicle group combinations where the true risk is superadditive and the multiplicative GLM structure underestimates it.

The infrastructure matters as much as the result. Running a GBM as a challenger to a production GLM, with both models trained on versioned Delta data, both logged to MLflow, both evaluated on the same temporal test split, is a meaningful champion-challenger process. Running a GBM in an unversioned notebook with a random train/test split and presenting the Gini improvement to a pricing committee is not. This module builds the former.
