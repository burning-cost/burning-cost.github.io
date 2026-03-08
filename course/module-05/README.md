# Module 5: Conformal Prediction Intervals

**Part of: Modern Insurance Pricing with Python and Databricks**

---

## What this module covers

The technical premium your model produces is a point estimate. It tells you the expected loss, but nothing about the uncertainty around that expectation. For a well-understood risk on a large book, the uncertainty is small and the point estimate is nearly sufficient. For a young driver in an unusual vehicle group with conviction points - a thin cell in the training data - the uncertainty is substantial, and using the point estimate alone as if it were certain is a modelling error with real consequences: underpriced floors, over-confident reserve estimates, and no principled basis for referring uncertain risks to human underwriters.

Conformal prediction intervals give you a lower and upper bound around the point estimate with a guaranteed coverage property. If you request 90% intervals, at least 90% of future observations will fall inside the interval. The guarantee is distribution-free in that it makes no parametric assumptions about the data - but it does require exchangeability and a well-calibrated conformity score, as the coverage-by-decile diagnostic in this module makes clear. In insurance terms: calibrate on recent business, test on more recent business, and validate that coverage is flat across risk deciles before using the intervals for any downstream purpose.

This module covers the complete conformal prediction workflow for UK motor pricing using the `insurance-conformal` library. We train a CatBoost Tweedie pure premium model, calibrate `InsuranceConformalPredictor` with the variance-weighted non-conformity score that produces correct coverage across all risk deciles, validate coverage using `CoverageDiagnostics`, and build three practical applications: flagging model-uncertain risks for underwriting referral, constructing risk-specific minimum premium floors that are defensible under Consumer Duty, and producing portfolio-level reserve range estimates that the reserving team can use. Results are written to versioned Delta tables in Unity Catalog with MLflow logging throughout.

---

## Prerequisites

- Modules 1 and 2 completed: `pricing.motor.claims_exposure` exists in Unity Catalog and you understand Poisson and Gamma GLMs
- Module 3 completed or understood: you know what a CatBoost Tweedie model is and how to train one
- Basic Python and statistics: you know what a confidence interval is and why coverage matters

You do not need prior conformal prediction experience. We introduce the theory and the `insurance-conformal` API from scratch.

---

## Estimated time

4-5 hours for the tutorial plus exercises. The notebook runs end-to-end in 15-20 minutes on a 4-core cluster.

---

## Files

| File | Purpose |
|------|---------|
| `tutorial.md` | Main written tutorial - read this first |
| `notebook.py` | Databricks notebook - full conformal prediction workflow on synthetic UK motor data |
| `exercises.md` | Five hands-on exercises with full solutions |

---

## What you will be able to do after this module

- Explain what a conformal prediction interval is, what the coverage guarantee means, and what assumptions it requires - including the specific exchangeability requirement for insurance data
- Calibrate `InsuranceConformalPredictor` from the `insurance-conformal` library using the `pearson_weighted` non-conformity score, and explain why the raw residual score fails for insurance data
- Validate intervals using `coverage_by_decile` and interpret the results: identify whether a monotone decline indicates residual heteroscedasticity or a distribution shift
- Flag the top decile of risks by relative interval width for underwriting referral, and distinguish between predictive uncertainty (wide interval) and risk level (high point estimate) - they are not the same thing
- Build risk-specific minimum premium floors from the conformal upper bound, compare them to conventional flat-multiplier floors, and articulate why the conformal approach is more consistent with FCA Consumer Duty's fair value requirement
- Produce portfolio-level reserve range estimates using both the naive (perfect correlation) and independence (CLT) aggregation methods, and explain when each is appropriate
- Write intervals and coverage diagnostics to Unity Catalog Delta tables with MLflow logging, and set up the coverage monitoring log that triggers recalibration
- Recalibrate the predictor on recent data without retraining the base model, and understand when recalibration is sufficient versus when the base model itself needs retraining

---

## Why this is worth your time

The reserving team wants a range, not a point. The underwriting director wants to know which risks to refer. The FCA wants evidence that minimum premiums are set on a principled basis. Point estimates alone cannot answer any of these questions.

The standard alternative is parametric uncertainty: assume the errors are normally distributed, compute a standard error, multiply by 1.96. This is wrong for insurance data in at least three ways: claim distributions are heavily right-skewed, model errors are heteroscedastic (larger for larger risks), and the normality assumption cannot be tested from the data you would use to compute the interval. Conformal prediction makes none of these parametric assumptions. The coverage guarantee requires no assumption about the shape of the loss distribution or the form of the model — provided calibration and test data are exchangeable and the non-conformity score is well-calibrated across risk levels, both of which this module validates explicitly.

The practical advantage is that conformal calibration is fast and separable from model training. When claim frequencies drift 8% due to weather events or premium inflation, you recalibrate the predictor on recent business in a few seconds without retraining the model. That separation - calibrate often, retrain annually - is the right operational pattern. It is also the pattern that Consumer Duty's continuous monitoring requirement suggests: you need to be able to demonstrate that your intervals are achieving stated coverage on recent business, not just on the holdout set you reported at model deployment.
