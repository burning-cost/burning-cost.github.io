---
layout: post
title: "Your Model Validation Report Won't Survive a PRA Review"
date: 2026-03-09
categories: [regulation, libraries]
tags: [model-validation, pra-ss123, fca-tr242, model-risk, gini, calibration, python, insurance-validation]
description: "Most pricing teams produce PowerPoint decks when regulators ask for model validation. PRA SS1/23 expects documented, structured, reproducible reports covering nine specific areas. insurance-validation v0.2.0 generates a compliant HTML report from a single Python call."
---

When the PRA or FCA asks for your model validation documentation, what do you actually send them?

We have seen what this looks like in practice. A pricing actuary opens a folder called something like `motor_v3_validation_2024Q3/` and finds: a PowerPoint deck with lifted Gini charts, a handful of Excel files with A/E ratios by some segments, and an email chain where sign-off was eventually obtained. The narrative exists in someone's memory. The methodology for the confidence intervals is undocumented. The person who ran the validation left last year.

This is not a PRA-compliant model validation. It is a collection of artefacts that gives the appearance of validation without the substance.

SS1/23 (May 2023, effective May 2024) is unambiguous on what independent validation must produce: documented coverage of conceptual soundness, data quality, outcome analysis, sensitivity testing, and ongoing monitoring — structured such that "a party unfamiliar with the model can understand its operation." For AI-based models, the PRA explicitly adds: testing for data transparency, bias, and fairness beyond accuracy metrics. And the FCA's TR24/2 (August 2024) thematic review found that documentation failures — not model failures — are the primary enforcement trigger.

A PowerPoint deck does not meet this bar. [`insurance-validation`](https://github.com/burning-cost/insurance-validation) does.

---

## What the library produces

`insurance-validation` v0.2.0 generates a self-contained HTML report and a JSON sidecar from a single call. The HTML is printable, has no CDN dependencies, and no JavaScript — it will render identically in Chrome, in a PDF export, and in the regulatory annexe to your submission. The JSON sidecar carries a `run_id` UUID so your Model Risk Management system can ingest and track it.

The report covers nine sections, mapped directly to PRA SS1/23 and FCA Consumer Duty obligations:

| Section | Regulatory mapping |
|---|---|
| 1. Executive Summary (RAG status) | SS1/23 Principle 4 |
| 2. Model Card and Governance | SS1/23 Principle 3 |
| 3. Data Quality | SS1/23 Principle 3, IBNR caveat |
| 4. Model Development Documentation | SS1/23 Principle 3 |
| 5. Performance Validation | SS1/23 Principle 4 |
| 6. Fairness and Discrimination | SS1/23 Principle 4, Consumer Duty / EP25/2 |
| 7. Stability | SS1/23 Principle 4 |
| 8. Feature Importance (optional) | SS1/23 Principle 3 |
| 9. Monitoring Plan and Sign-Off | SS1/23 Principle 5 |

The RAG status — red, amber, green — is computed from the test results and appears in the Executive Summary. It is not editorial. If the Hosmer-Lemeshow test rejects calibration, the report goes amber. If the renewal cohort A/E falls outside [0.85, 1.15] for a material tenure band, it goes red. The thresholds are hardcoded, transparent, and documented in the methodology section.

---

## Generating the report

Install with pip or uv:

```bash
pip install insurance-validation
```

The `ModelValidationReport` facade takes your model, training and validation data, exposure vectors, and a `ModelCard`:

```python
from insurance_validation import ModelValidationReport, ModelCard

report = ModelValidationReport(
    model=model,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    exposure_train=exposure_train,
    exposure_val=exposure_val,
    model_card=ModelCard(
        name="Motor Frequency v3.2",
        version="3.2.0",
        purpose="Predict claim frequency for UK motor portfolio",
        methodology="CatBoost gradient boosting with Poisson objective",
        target="claim_count",
        features=["age", "vehicle_age", "area", "vehicle_group"],
        limitations=["No telematics data", "Limited fleet exposure"],
        owner="Pricing Team",
    ),
)

report.generate("validation_report.html")
report.to_json("validation_report.json")
```

That is the full API for the common case. The `ModelCard` carries the information required by SS1/23 Principle 3: purpose, methodology, known limitations, and ownership. The `limitations` field goes directly into the report's Data Quality section with a built-in IBNR caveat for frequency models — you are not writing the disclaimer yourself and hoping it is sufficient.

---

## The statistical methodology

### Gini with bootstrap confidence intervals

The headline discrimination metric is the Gini coefficient, with a 95% confidence interval from 1,000 bootstrap resamples using the percentile method. We chose the percentile method over BCa: BCa is technically superior for small samples but adds meaningful complexity for marginal gain at the sample sizes typical in pricing validation (10k–100k policies). The bootstrap interval gives you something TR24/2 requires and almost no validation deck has: a quantified measure of how certain you are about your Gini, not just the point estimate.

A model with Gini 0.42 [0.38, 0.46] tells a different story from one with Gini 0.42 [0.22, 0.61]. The first is a validated result. The second is a model that needs more data before you can say anything useful.

### Poisson exact A/E confidence intervals

For frequency models, the actual-to-expected (A/E) ratio is tested with an exact Poisson confidence interval using the chi-squared pivot from Klugman, Panjer and Willmot's *Loss Models* (the standard actuarial reference). For observed claims count A and expected E, the interval is:

```
lower = χ²(2A, 0.025) / (2E)
upper = χ²(2A+2, 0.975) / (2E)
```

This is exact for Poisson data. The alternative — Wald intervals on the ratio — is approximate and performs poorly when A is small, which is exactly when your sub-segment A/E ratios are most uncertain. We use the Dobson (1991) formulation throughout, which is consistent with UK actuarial practice.

### Hosmer-Lemeshow calibration test

Calibration is assessed via the Hosmer-Lemeshow test across deciles of predicted probability. A rejection (p < 0.05) means your model is systematically over- or under-predicting in parts of the distribution — the kind of failure that produces unfair pricing outcomes even when the aggregate loss ratio looks fine. The HL test result feeds into the RAG computation.

### Double-lift chart

The performance section includes a double-lift chart: a comparison of the new model against the incumbent, sorted by the ratio of new model predictions to incumbent predictions. This is standard validation practice in UK personal lines and is required when you are retiring an existing model. A library that handles new model validation but ignores the comparative against the incumbent is missing the scenario that actually comes up in governance committees.

---

## The FCA-specific tests

### Renewal cohort A/E by tenure band (FCA TR24/2)

Section 6 implements the renewal cohort A/E test from TR24/2. The FCA's thematic review found that several firms could not demonstrate that their pricing models produced equitable outcomes across customer tenure bands — specifically, that loyal customers were not systematically over-charged relative to risk.

The library splits the validation population into tenure bands (0–1 year, 1–3 years, 3–5 years, 5+ years by default), computes A/E for each, and flags any band outside [0.85, 1.15] as a divergence. The threshold is taken from TR24/2 guidance. A divergence in the 5+ year band — your most loyal customers — moves the RAG to red.

This test does not require a separate consumer outcomes dataset. It runs on your standard validation data with a `tenure_months` column. If you do not have tenure on your validation set, the section is skipped with a documented caveat rather than silently omitted.

### Sub-segment calibration

Section 6 also runs sub-segment calibration across categorical features: by age band, by region, by vehicle group, by any categorical features you specify. Each sub-segment gets a Poisson A/E with exact CI. Segments where the CI excludes 1.0 are flagged. This is the Consumer Duty fair value evidence — documented, reproducible, attached to a run_id that your MRM system can retrieve.

---

## What the JSON sidecar enables

```json
{
  "run_id": "a3f7e2c1-...",
  "generated_at": "2026-03-09T14:23:11Z",
  "model_name": "Motor Frequency v3.2",
  "model_version": "3.2.0",
  "rag_status": "AMBER",
  "gini_train": 0.44,
  "gini_val": 0.41,
  "gini_val_ci_lower": 0.37,
  "gini_val_ci_upper": 0.45,
  "ae_val": 0.98,
  "ae_val_ci_lower": 0.94,
  "ae_val_ci_upper": 1.02,
  "hosmer_lemeshow_p": 0.12,
  "renewal_cohort_flags": ["5+ years"],
  "rag_triggers": ["renewal_cohort_5plus_outside_threshold"]
}
```

The `run_id` is a UUID generated at run time. Every time you regenerate the report — after retraining, after adding data, after fixing a data quality issue — you get a new `run_id`. Your MRM system can store the JSON, link it to the model version in your model inventory, and produce an audit trail of validation outcomes over the model lifecycle.

This is what SS1/23 Principle 1 (model inventory with change log) requires in practice: not a spreadsheet that someone remembers to update, but a machine-readable artefact that is produced automatically as part of the validation process.

---

## The documentation gap is the enforcement risk

TR24/2 is worth reading carefully, particularly its description of what went wrong at the firms it reviewed. The failures were not bad models. They were: no documented methodology for how performance metrics were chosen, no record of who approved the validation findings, no evidence that the validation function was independent of model development, no trail linking the validation report to the model version that was deployed.

These are documentation failures. They are also, under SM&CR, personal liability failures for the named Senior Manager responsible for model risk.

A validation report generated by `insurance-validation` is reproducible from the run_id, linked to a specific model version, and contains its own methodology documentation. The HTML is the report that goes to the committee. The JSON is the evidence that the MRM system holds. The two are consistent by construction — they come from the same run.

We are not claiming this eliminates model risk. A model with a Gini of 0.15 will still get a red RAG and a bad validation report. That is the point: the report tells you the truth, in a format the regulator can read.

---

## 148 tests, no external dependencies

The library ships 148 passing tests covering the statistical functions, the report generation, the RAG logic, and the JSON sidecar. The HTML template is self-contained — no CDN calls, no external fonts, no JavaScript. It will render in environments with no internet access, which matters if your validation runs in an air-gapped model development environment.

The test suite was validated on Databricks serverless (Python 3.11). The library has no hard dependency on Spark or any cloud runtime — it runs anywhere scikit-learn runs.

[`insurance-validation` on GitHub](https://github.com/burning-cost/insurance-validation) — MIT licence.
