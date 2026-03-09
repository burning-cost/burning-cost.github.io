---
layout: post
title: "Your Model Validation Report Won't Survive a PRA Review"
date: 2026-03-08
author: Burning Cost
tags: [model-validation, PRA, insurance-pricing, python]
---

There is a specific kind of anxiety that visits pricing actuaries around model governance season. You have a Gini coefficient in a spreadsheet, an A/E ratio computed in a notebook, and a Word document titled something like "TPPD Frequency Model Validation 2024 FINAL v3". You know what the model does. You know it works. But you also know that if the PRA or FCA walked in tomorrow and asked for your model validation evidence, the document you would hand over is not what SS1/23 envisions.

PRA SS1/23, formally a supervisory statement for banks with internal model permissions, is now openly referenced by the PRA in its 2026 insurance supervision priorities letter. KPMG, 4most, and Clifford Chance have all been direct: insurers with material pricing models should treat SS1/23's principles as effectively binding. The FCA has its own hooks through Consumer Duty and TR24/2, which require evidence that pricing is fair across customer segments and does not systematically disadvantage long-standing customers.

What does SS1/23 Principle 4 actually require? An independent validation covering: data quality, conceptual soundness, outcome analysis, sensitivity testing, and ongoing monitoring adequacy. The validation report must name the validator, carry a date, show the test methodology and results, and form part of a traceable audit trail. It must be done by someone operationally independent of the model developers.

A Gini in a spreadsheet does not satisfy that. Neither does a notebook with no run ID, no exposure weighting, no confidence intervals, and no section on renewal customer fairness.

We built [insurance-validation](https://github.com/burning-cost/insurance-validation) to fill this gap. It is a Python library — 148 tests, ~5,850 lines — that runs the actuarial validation tests your report needs and writes the output as a structured 9-section HTML with a JSON sidecar for MRM system ingestion. MIT licence. Runs entirely locally, outputs standard HTML and JSON.

## What the library actually does

The high-level API is a single facade class:

```python
from insurance_validation import ModelValidationReport, ModelCard

card = ModelCard(
    name="Motor TPPD Frequency",
    version="3.2.0",
    purpose="Estimate claim frequency for private motor policies",
    methodology="CatBoost gradient boosting with Poisson objective",
    target="claim_count",
    features=["driver_age", "vehicle_age", "annual_mileage", "region"],
    limitations=["No telematics data", "Fleet exposure limited to <10 vehicles"],
    owner="Pricing Team",
    validator_name="Independent Validation Team",
    validation_date=date(2026, 3, 1),
    materiality_tier=2,
    approved_by=["Jane Smith - Chief Actuary", "Model Risk Committee"],
)

report = ModelValidationReport(
    model_card=card,
    y_val=y_val,
    y_pred_val=y_pred_val,
    exposure_val=exposure_val,
    y_train=y_train,
    y_pred_train=y_pred_train,
    X_train=X_train,          # Polars DataFrame, optional
    X_val=X_val,
    incumbent_pred_val=old_model_predictions,  # for double-lift
    tenure_col="years_as_customer",             # for FCA TR24/2 renewal test
    fairness_group_col="age_band",
    monitoring_owner="Head of Pricing",
    monitoring_triggers={"psi": 0.25, "ae_ratio": 1.15},
)

report.generate("motor_tppd_validation_2026.html")
report.to_json("motor_tppd_validation_2026.json")
```

The `generate()` call runs all tests, writes a self-contained HTML report (no CDN, no JavaScript, printable), and assigns a UUID `run_id` so the file can be ingested into a model risk register without ambiguity. The `to_json()` call produces a structured sidecar with the full test results for downstream ingestion.

You cannot call `generate()` without completing the `ModelCard`. This is deliberate. Incomplete documentation is the most common validation failure — the library enforces completeness at construction time, not as an afterthought.

## The tests that matter

### Gini with a confidence interval

Reporting a Gini of 0.32 is not a validation result. A 95% confidence interval on that Gini is. The library implements bootstrap Gini CIs using the percentile method with 1,000 resamples (following the approach from arXiv 2510.04556):

```python
perf = PerformanceReport(y_true, y_pred, exposure=policy_years, model_name="Motor TPPD v3.2")
result = perf.gini_with_ci(n_resamples=1000, confidence=0.95, min_acceptable=0.2)
# result.extra["ci_lower"], result.extra["ci_upper"]
```

Critically, the Gini is exposure-weighted by default. The standard ML definition — `2 * AUC - 1` computed on unweighted observations — treats a policy with 0.1 years of exposure the same as one with 1.0 years. For insurance frequency models, that is wrong. The actuarial convention is exposure-weighted Lorenz curves. The library implements this correctly, which produces a materially different (and more meaningful) result than `sklearn.metrics.roc_auc_score` on the same data.

### Poisson A/E confidence interval

An A/E ratio of 0.97 on the validation holdout is reassuring. But whether 0.97 is statistically consistent with 1.0 depends on how many claims you actually observed. The library applies the Dobson (1991) exact Poisson confidence interval via the chi-squared pivot:

```
lower = chi2(2A, 0.025) / (2E)
upper = chi2(2A+2, 0.975) / (2E)
```

where A is total observed claims and E is total expected. The test fails if 1.0 falls outside the interval (material bias detectable at 5% significance) or if the point estimate falls outside [0.90, 1.10]. The result text also carries an explicit IBNR caveat, because A/E ratios on recently-incurred claims may reflect development shortfall rather than model error.

### Hosmer-Lemeshow calibration

The overall A/E ratio hides miscalibration within risk bands. A model that over-prices young drivers and under-prices older drivers can still show an aggregate A/E of 1.0. The Hosmer-Lemeshow test groups policies into predicted-risk deciles and tests whether observed and expected counts are consistent across all deciles simultaneously. The H-L statistic is approximately chi-squared with 8 degrees of freedom for 10 groups. A p-value below 0.05 means the model is systematically miscalibrated somewhere — the report tells you where.

### Double-lift against the incumbent

When replacing an existing model, the validation question is not just "does the new model work?" but "does it work better than what we have?" The double-lift chart answers this. Policies are ranked by the ratio of new-to-incumbent predictions, grouped into deciles, and for each decile the chart shows actual, new-model, and incumbent rates. The band where the new model predicts lower risk than the incumbent — do the actual claims vindicate that?

```python
result = perf.double_lift(
    y_pred_incumbent=old_model_predictions,
    n_bands=10,
    incumbent_name="GLM v2.1",
)
# result.extra["bands"]: list of {band, actual_rate, new_model_rate, incumbent_rate, weight}
# result.extra["new_model_mae"] vs result.extra["incumbent_mae"]
```

This is a standard chart for pricing validation teams in the UK market and one that is difficult to generate correctly from scratch (the exposure-weighting of the band assignments is fiddly). The library handles it.

### Renewal cohort A/E: the FCA TR24/2 requirement

FCA TR24/2 (August 2024, Pricing Practices Review) and the earlier GIPP rules require that renewal pricing does not systematically overcharge long-standing customers. The discrimination module implements a renewal cohort A/E test segmented by tenure band:

```python
disc = DiscriminationReport(df=X_val, predictions=y_pred_val)
result = disc.renewal_cohort_ae(
    y_true=y_val,
    y_pred=y_pred_val,
    tenure_col="years_as_customer",
)
```

The test computes A/E by years-as-customer cohort and flags divergence outside [0.85, 1.15]. If your model consistently under-predicts (over-prices) for 5+ year customers relative to new business, that is a regulatory problem. This test produces traceable evidence that you checked.

### Proxy discrimination

Postcode is the canonical example. Postcodes correlate with ethnicity; using postcode relativities without analysis creates disparate impact risk under Consumer Duty. The library tests Spearman rank correlation (for numeric pairs) and Cramer's V (for categorical pairs) between model features and protected characteristic proxies, flagging pairs above 0.3. It also computes disparate impact ratios — the ratio of predicted rates between protected groups — with a 0.8 threshold (the four-fifths rule from employment discrimination case law, now applied to insurance pricing fairness).

## What the report looks like

The HTML output has nine sections mapped to SS1/23 requirements:

1. Executive Summary — RAG status (Red/Amber/Green), pass/fail counts by category, model card summary, regulatory references
2. Data Quality — missing value analysis, outlier detection, cardinality checks on training and validation sets
3. Model Development Documentation — methodology rationale, variables used, alternatives considered, known limitations
4. Performance Validation — in-sample vs out-of-sample, Gini with CI, lift chart, double-lift if incumbent provided, A/E with Poisson CI, calibration scatter
5. Fairness and Discrimination — proxy correlation matrix, disparate impact ratios, renewal cohort A/E
6. Stability — PSI on model score, feature drift per variable
7. Feature Importance — SHAP output if `insurance-validation[shap]` is installed; informational stub if not
8. Model Card and Sign-Off — full model card rendering, validator name, validation date, outstanding issues
9. Monitoring Plan — named owner, review frequency, alert triggers

Every section carries regulatory references in the margin (SS1/23 Principle 3, 4, or 5; Consumer Duty PRIN 2A.4; FCA TR24/2). This matters when your validation report needs to show auditors that your tests are not ad-hoc — they map to specific regulatory requirements.

The JSON sidecar has the same structure. Every test result includes a unique `run_id` UUID, so your model risk register can ingest the file and trace it back to a specific validation run.

## What this library is not

The README is honest about this: it is not a compliance guarantee. A Tier 1 material model at a large composite insurer needs more than this library alone. It needs independent sign-off, a formal model risk committee, and in some cases regulatory pre-notification. The library generates the technical evidence; whether that evidence is sufficient depends on your firm's model risk policy and the model's materiality classification.

The library also does not attempt to replace model development tooling. It does not build the model, select variables, or tune hyperparameters. It validates the output of whatever model you have built.

## Getting started

```bash
uv add insurance-validation
# or
pip install insurance-validation
```

Python 3.10+ required. No cloud dependency. The core install is lightweight — NumPy, SciPy, Pydantic, Jinja2. Polars is optional but strongly recommended for the data quality and feature drift modules. SHAP integration is an extras install:

```bash
uv add 'insurance-validation[shap]'
```

The library is at [github.com/burning-cost/insurance-validation](https://github.com/burning-cost/insurance-validation). It is the 21st library in the Burning Cost open source stack for UK insurance pricing.

---

**Related articles from Burning Cost:**
- [Your Model Validation Report Won't Survive a PRA Review](/2026/03/09/insurance-validation/)
- [Your Pricing Model is Drifting (and You Probably Can't Tell)](/2026/03/07/your-pricing-model-is-drifting/)
- [Your Champion/Challenger Test Has No Audit Trail](/2026/03/09/your-champion-challenger-test-has-no-audit-trail/)
