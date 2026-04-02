---
layout: post
title: "Per-Prediction Audit Trails for Pricing Models: insurance-governance v0.3.0"
date: 2026-04-02
categories: [libraries, regulation]
tags: [model-governance, explainability, shap, consumer-duty, fca, eiopa, audit-trail, insurance-governance, prin-2a, tamper-evidence, python]
description: "insurance-governance v0.3.0 adds ExplainabilityAuditTrail: a per-prediction audit log that records SHAP values, fairness flags, and plain-language summaries for every pricing decision, with SHA-256 tamper evidence. Meets FCA PRIN 2A.3 and EIOPA-BoS-25-360 requirements."
author: burning-cost
---

The regulator's question is not "does your model perform well on a holdout set?" That is the question you were used to answering. The question coming out of PRIN 2A and EIOPA-BoS-25-360 is more specific: "Why did this customer get this price?"

Not the model in general. This customer. This quote. This date. What did the model see, what did SHAP say about each factor, what was the final premium, and if a human changed it, who and why?

Until v0.3.0, `insurance-governance` had no answer to that question. The `ModelValidationReport` could tell you the Gini, A/E, and PSI for a validation cohort. The `MRMModelCard` held static model metadata. But there was nothing that captured what actually happened at prediction time — the decision-level record that a Section 166 review or an EIOPA supervisory inspection would ask for.

v0.3.0 adds the `audit/` subpackage to fill that gap.

---

## What the audit subpackage records

The atomic unit is `ExplainabilityAuditEntry`. Every time your pricing model scores a risk, you create one:

```python
from insurance_governance.audit import ExplainabilityAuditEntry

entry = ExplainabilityAuditEntry(
    model_id="motor-freq-v3",
    model_version="3.2.0",
    input_features={
        "driver_age": 28,
        "vehicle_age": 4,
        "region": "E2",
        "ncb_years": 3,
        "annual_mileage": 12000,
    },
    feature_importances={
        "driver_age":     0.082,
        "vehicle_age":   -0.031,
        "region":         0.145,
        "ncb_years":     -0.094,
        "annual_mileage": 0.019,
    },
    prediction=0.187,          # model output — e.g. expected claim frequency
    final_premium=412.50,      # what was actually charged
    decision_basis="model_output",
)
```

The entry captures: a UUID, model ID and version, UTC timestamp, the raw input features, signed SHAP values for each factor, the model's raw output, the final charged premium (which may differ due to minimum premium rules or commercial loadings), whether a human reviewed the decision, and who.

On construction, a SHA-256 hash is computed over the canonical JSON representation of all fields. That hash is stored with the entry. Later, `entry.verify_integrity()` recomputes the hash and returns `False` if the record has been modified since it was written.

---

## The append-only log

`ExplainabilityAuditLog` writes entries to a JSONL file:

```python
from insurance_governance.audit import ExplainabilityAuditLog

log = ExplainabilityAuditLog(
    path="pricing_audit_2026_q2.jsonl",
    model_id="motor-freq-v3",
    model_version="3.2.0",
)

log.append(entry)
```

One JSON line per decision, appended immediately. No database, no schema migration, no external dependency. JSONL survives infrastructure changes — it is grep-able, pandas-importable, and readable by any tool the FCA's supervisory team might have. Regulators tend not to have a Databricks workspace.

The log has no delete or overwrite method. That is intentional. If you need to redact an entry for data protection reasons, that is a separate process involving the original file and a documented redaction log — not something the library should automate.

To verify the integrity of an entire log after retrieval:

```python
failures = log.verify_chain()
# Returns a list of dicts describing any hash mismatches
# An empty list means all entries are intact
if failures:
    for f in failures:
        print(f"Entry {f['entry_id']} at line {f['line']}: {f['reason']}")
```

For regulatory submission, `export_period()` writes a filtered copy of the log to a new file with a metadata header, preserving all entry hashes so the recipient can independently verify integrity:

```python
from datetime import datetime, timezone

log.export_period(
    start=datetime(2026, 4, 1, tzinfo=timezone.utc),
    end=datetime(2026, 6, 30, 23, 59, 59, tzinfo=timezone.utc),
    path="q2_2026_regulatory_submission.jsonl",
)
```

---

## SHAP integration

`SHAPExplainer` wraps the `shap` library with insurance-specific conventions. Feature importances are signed (positive means the factor increased the prediction). The output is a plain dict keyed by feature name, ready to drop into `ExplainabilityAuditEntry.feature_importances`.

```python
from insurance_governance.audit import SHAPExplainer

shap_explainer = SHAPExplainer(
    model=fitted_catboost,
    model_type="tree",
    feature_names=["driver_age", "vehicle_age", "region", "ncb_years", "annual_mileage"],
)

# Explain a batch of predictions
importances_batch = shap_explainer.explain(X_test)
# Returns list of dicts, one per row

# Or a single observation
importances_single = shap_explainer.explain_single(X_test[0])
```

`shap` is an optional dependency. If it is not installed, the import raises a clear `ImportError` with the installation command rather than failing silently. Many teams run `insurance-governance` without needing SHAP, and the package is large enough that forcing it on everyone would be unreasonable.

---

## Plain-language explanations

PRIN 2A.3 requires that consumers be able to understand how their premium was set. The FCA's framing is consumer understanding as an outcome, not a documentation checkbox. If your pricing model cannot generate a readable explanation of the factors that drove a specific customer's premium, that is a potential breach — not a theoretical future one.

`PlainLanguageExplainer` converts SHAP values into a paragraph a customer can actually read. The SHAP values are scaled so their sum equals the difference between the final premium and the base premium, giving additive factor impacts in pounds:

```python
from insurance_governance.audit import PlainLanguageExplainer

explainer = PlainLanguageExplainer(
    feature_labels={
        "driver_age":     "your age",
        "ncb_years":      "your no-claims discount",
        "region":         "your postcode area",
        "vehicle_age":    "the age of your vehicle",
        "annual_mileage": "your annual mileage",
    }
)

text = explainer.generate(entry, base_premium=350.00)
print(text)
```

The output looks like:

> Your premium is £412.50. The main factors affecting your premium were: Your postcode area added £95.40 to reflect local claim rates. Your no-claims discount reduced your premium by £45.20. Your age reduced your premium by £18.30. The age of your vehicle added £30.60.

That is what PRIN 2A.3 compliance looks like at the decision level.

---

## Full workflow: fit, score, log, explain

A complete example using a CatBoost frequency model:

```python
import catboost as cb
import pandas as pd
import numpy as np
from insurance_governance.audit import (
    ExplainabilityAuditEntry,
    ExplainabilityAuditLog,
    SHAPExplainer,
    PlainLanguageExplainer,
)

# --- Assume model is already fitted ---
# model = cb.CatBoostRegressor(loss_function='Poisson').fit(X_train, y_train)

FEATURES = ["driver_age", "vehicle_age", "region_code", "ncb_years", "annual_mileage"]
BASE_PREMIUM = 350.00

shap_exp = SHAPExplainer(model=model, model_type="tree", feature_names=FEATURES)
plain_exp = PlainLanguageExplainer(feature_labels={
    "driver_age":     "your age",
    "ncb_years":      "your no-claims discount",
    "region_code":    "your postcode area",
    "vehicle_age":    "the age of your vehicle",
    "annual_mileage": "your annual mileage",
})

log = ExplainabilityAuditLog(
    path="pricing_audit.jsonl",
    model_id="motor-freq-v3",
    model_version="3.2.0",
)

# Score a batch and log every decision
predictions = model.predict(X_test)
importances = shap_exp.explain(X_test)

for i, (pred, imp, row) in enumerate(zip(predictions, importances, X_test.itertuples())):
    # Apply base rate and loadings to get final premium
    final_premium = BASE_PREMIUM * np.exp(pred)

    entry = ExplainabilityAuditEntry(
        model_id="motor-freq-v3",
        model_version="3.2.0",
        input_features={f: getattr(row, f) for f in FEATURES},
        feature_importances=imp,
        prediction=float(pred),
        final_premium=round(float(final_premium), 2),
        decision_basis="model_output",
    )
    log.append(entry)

    # For web/app display: plain-language explanation
    explanation_text = plain_exp.generate(entry, base_premium=BASE_PREMIUM)

# Later: verify the log is intact
failures = log.verify_chain()
assert len(failures) == 0, f"Log integrity failures: {failures}"
```

---

## Tamper evidence: what it does and does not prove

The SHA-256 per-entry hash gives you one thing: evidence that a record has not been modified since it was written. If a record's content is changed — even a single character in a feature value or SHAP score — `verify_integrity()` returns `False`.

This is not a blockchain. We use per-entry hashes rather than chain hashes deliberately. A chain hash (where each entry's hash depends on the previous entry) makes GDPR-compliant redaction catastrophic — changing one entry invalidates every subsequent hash. With per-entry hashes, one entry can be redacted and documented without touching the rest. The regulatory requirement is tamper evidence, not cryptographic provenance of sequential ordering.

What the hash does not prove: that the log file itself was not deleted and replaced with a freshly-hashed forgery. If your threat model includes that, you need something external — object storage with write-once semantics, or an independent hash deposit. For most FCA-supervised firms, the question is whether entries were quietly edited after the fact, not whether an entire log was forged. The per-entry hash answers the former.

---

## Regulatory alignment

**FCA Consumer Duty (PRIN 2A.3):** The obligation is consumer understanding as an outcome. A firm running an automated pricing model that cannot tell a specific customer why their premium is what it is has a PRIN 2A.3 gap. `PlainLanguageExplainer` bridges that gap at the decision level.

**SM&CR:** The `reviewer_id` field on `ExplainabilityAuditEntry` is designed to reference the SM&CR Controlled Function holder responsible for the pricing decision — SMF4 (Chief Risk Officer) for model-level decisions, or the named underwriting CF holder for individual overrides. If the FCA pursues personal accountability for an AI pricing failure, the audit trail needs to name a person, not just a model version.

**EIOPA-BoS-25-360 (August 2025):** EIOPA's supervisory expectations for AI governance in insurance include "traceability and reproducibility of decisions" and "explainability to authorities and clients". The audit log satisfies traceability; `SHAPExplainer` satisfies explainability to authorities; `PlainLanguageExplainer` satisfies explainability to clients. Lloyd's Brussels subsidiaries and EU market participants are directly in scope; UK-only firms should treat it as the direction of travel for FCA guidance expected by end-2026.

**EU AI Act:** Motor and property pricing models are not high-risk AI under Annex III. The paragraph 5(c) carve-out covers life and health insurance only. GLMs are not AI systems within the Act's definition. The Articles 12-14 logging obligations do not apply. We cover the audit trail here for Consumer Duty and EIOPA reasons, not EU AI Act reasons.

---

## Integration with the existing governance stack

The audit subpackage sits alongside the existing `insurance_governance` modules without duplicating them:

- `MRMModelCard` stores static model metadata (champion/challenger status, risk tier, review dates). The `model_id` and `model_version` in `ExplainabilityAuditEntry` are foreign keys to the model card.
- `ModelValidationReport` covers portfolio-level statistical performance (Gini, A/E, PSI). The audit log covers decision-level traceability. They answer different questions.
- `Article13Document` provides EU AI Act–style static transparency documentation. The audit log provides dynamic decision-time records. `SHAPExplainer.attach_to_article13()` populates the explanation tools field in an `Article13Document` if you are building that documentation.
- `DiscriminationReport` tests for disparate impact at the portfolio level. A future release will add aggregated fairness views over the audit log — per-segment A/E analysis directly from the logged feature importances.

---

## Installation

```bash
uv add insurance-governance
# For SHAP integration:
uv add "insurance-governance[shap]"
```

Python 3.10+. The `shap` extra adds `shap>=0.44` and its native code dependencies.

Source: [github.com/burning-cost/insurance-governance](https://github.com/burning-cost/insurance-governance).

The FCA's formal AI audit trail guidance is expected by end-2026. We have designed the schema to be forwards-compatible — the field set covers every documented expectation in the existing EIOPA and Consumer Duty guidance, and the JSONL format can absorb additional fields without a breaking change. Build the trail now; extend it when the guidance arrives.

---

Related:
- [One Package, One Install: PRA SS1/23 Validation and MRM Governance Unified](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) — the existing governance stack this extends
- [Consumer Duty Outcomes Monitoring](/2026/03/25/consumer-duty-outcomes-monitoring/) — the PRIN 2A.1 fair value monitoring layer
- [EU AI Act August 2026: What Insurance Pricing Teams Need to Do Now](/2026/03/25/eu-ai-act-august-2026-insurance-pricing/) — the AI Act classification that keeps motor/property out of high-risk scope
