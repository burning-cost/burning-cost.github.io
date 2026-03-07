# Module 5: Conformal Prediction Intervals

**Part of: Modern Insurance Pricing with Python and Databricks**

---

## What this module covers

Your Tweedie GBM gives point estimates. A point estimate tells you the model's expected loss cost. It tells you nothing about how certain the model is in that figure - and certainty varies enormously across a portfolio.

A straightforward domestic combined risk sitting in a dense area of the feature space, with thousands of similar risks behind it, is not the same as an unusual commercial property in a thin cell. Both get a number from the model. Without uncertainty quantification, you cannot tell them apart at the underwriting stage.

This module covers conformal prediction as a practical solution to that problem. Conformal prediction produces intervals with a finite-sample coverage guarantee that does not depend on distributional assumptions. We implement it using the `insurance-conformal` library, which extends the standard approach with variance-weighted non-conformity scores adapted for the heteroscedastic structure of insurance data - approximately 30% narrower intervals than the naive approach, with identical coverage guarantees.

The theory is from Manna et al. (2025), arXiv:2507.06921.

---

## Prerequisites

- Completed Modules 1-4, or equivalent experience with CatBoost frequency-severity models and Python data pipelines
- Know what a prediction interval is and how it differs from a confidence interval for the mean
- Basic familiarity with Polars and CatBoost Pool syntax
- Databricks access (Free Edition is sufficient for the exercises)

---

## Estimated time

4-5 hours for the tutorial plus exercises. Calibration is fast - the conformal wrapping adds seconds to any trained model. The coverage diagnostics section takes the most thought; budget time to understand what you are looking at before accepting the intervals as fit for purpose.

---

## Files

| File | Purpose |
|------|---------|
| `tutorial.md` | Main written tutorial - read this first |
| `notebook.py` | Databricks notebook - full workflow on synthetic data |
| `exercises.md` | Five hands-on exercises with full solutions |

---

## Library

```bash
uv add "insurance-conformal[catboost]"

# On Databricks:
%pip install "insurance-conformal[catboost]" --quiet
```

Source: [github.com/burningcost/insurance-conformal](https://github.com/burningcost/insurance-conformal)

---

## What you will be able to do after this module

- Explain what split conformal prediction guarantees and what it does not - specifically the difference between marginal coverage and conditional coverage
- Understand why raw absolute residuals produce badly calibrated intervals for insurance data, and what the Pearson-weighted score fixes
- Implement the full conformal workflow: temporal split, model training, `InsuranceConformalPredictor`, calibration, interval generation
- Run and interpret `coverage_by_decile()` - the diagnostic that distinguishes genuinely calibrated intervals from ones with correct marginal coverage but poor conditional coverage
- Use prediction intervals to flag uncertain risks for manual underwriting review
- Set minimum premium floors based on the upper bound of the prediction interval
- Construct reserve range estimates from portfolio-level interval aggregation
- Know when conformal prediction breaks down: temporal drift, distribution shift, and the exchangeability assumption

---

## The insurance-conformal library

The library wraps any sklearn-compatible model with split conformal prediction. The key extension is the variance-weighted non-conformity score:

```
score(y, ŷ) = |y - ŷ| / ŷ^(p/2)
```

where p is the Tweedie variance power. Standard implementations use `|y - ŷ|`, which treats a £500 error on a £500 risk identically to a £500 error on a £50,000 risk. For Tweedie data with variance proportional to μ^p, this systematic error means intervals are too wide at the low end and too narrow where it matters - large risks.

The library reads the Tweedie power from CatBoost's `loss_function` parameter automatically.

---

## Part of the MVP bundle

This module is included in the £295 MVP bundle alongside Modules 1, 2, 4, and 6. Individual module: £79.
