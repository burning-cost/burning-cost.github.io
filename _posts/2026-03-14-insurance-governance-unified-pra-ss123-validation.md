---
layout: post
title: "One Package, One Install: PRA SS1/23 Validation and MRM Governance Unified"
date: 2026-03-14
categories: [libraries, regulation]
tags: [model-validation, model-risk-management, mrm, pra-ss123, fca-consumer-duty, gini, calibration, python, insurance-governance]
description: "We shipped two packages separately — insurance-validation for statistical reports, insurance-mrm for governance workflows. Teams were installing both, pinning them separately, and occasionally running them out of sync. insurance-governance merges both into one install."
---

We made a mistake with our initial packaging.

`insurance-validation` (statistical validation reports: Gini CI, A/E, Hosmer-Lemeshow, PSI, double-lift) and `insurance-mrm` (governance workflows: model inventory, risk tier scoring, executive committee packs) were shipped as separate libraries. The reasoning seemed sound at the time: validation is a statistical concern; MRM is a process concern; keep them separate.

The problem is that they are not actually separate. The validation report produces a JSON sidecar with a `run_id`. The MRM inventory takes that `run_id` and links it to the model record. The governance pack reads the RAG status from the validation output. These are not independent tools — they are a pipeline, and packaging them separately made every team that used both manage the coupling themselves.

In practice that meant: two installs, two pinned versions, and a non-trivial chance that the `run_id` schema the validation package was writing had drifted from the schema the MRM package was reading. That is a correctness risk in the governance layer, which is precisely the thing we built these tools to make reliable.

[`insurance-governance`](https://github.com/burning-cost/insurance-governance) is the merge. One install, one import path, zero coupling to manage yourself.

```bash
pip install insurance-governance
```

---

## What is in the package

The library has two subpackages: `insurance_governance.validation` and `insurance_governance.mrm`. At the top level, everything useful is re-exported.

```python
from insurance_governance import (
    ModelValidationReport,
    ValidationModelCard,
    MRMModelCard,
    RiskTierScorer,
    ModelInventory,
    GovernanceReport,
)
```

The MRM tooling is covered in depth at [Model Risk Governance for UK Insurers: Beyond the Excel Register](/2026/03/13/your-model-risk-register-is-a-spreadsheet/). What follows covers the unified package and the integration behaviour that only works when both sides live together.

---

## The validation side

`ModelValidationReport` runs nine test categories against a model's validation set and produces a self-contained HTML report plus a JSON sidecar. The tests are mapped directly to PRA SS1/23 and FCA Consumer Duty obligations:

| Section | Regulatory anchor |
|---|---|
| Gini coefficient with 1,000-resample bootstrap 95% CI | SS1/23 Principle 4 |
| 10-band A/E lift chart with exact Poisson CI | SS1/23 Principle 4 |
| Hosmer-Lemeshow calibration test | SS1/23 Principle 4 |
| Double-lift chart (new vs incumbent) | SS1/23 Principle 4 |
| PSI on score distribution (train vs validation) | SS1/23 Principle 4 |
| Renewal cohort A/E by tenure band | FCA TR24/2 |
| Sub-segment calibration by categorical features | Consumer Duty / EP25/2 |
| Data quality (missingness, outliers, cardinality) | SS1/23 Principle 3 |
| Monitoring plan completeness | SS1/23 Principle 5 |

The overall RAG status (Green/Amber/Red) is computed from the worst-severity failure. A Hosmer-Lemeshow rejection moves the report to Amber. A renewal cohort A/E outside [0.85, 1.15] for the 5+ year tenure band moves it to Red. These thresholds are not configurable by default, because configurable thresholds mean every team sets different thresholds, and you end up with a Gini that means different things in different reports.

The minimal call:

```python
import numpy as np
from insurance_governance import ModelValidationReport, ValidationModelCard

card = ValidationModelCard(
    name="Motor Frequency v3.2",
    version="3.2.0",
    purpose="Predict claim frequency for UK motor portfolio",
    methodology="CatBoost gradient boosting with Poisson objective",
    target="claim_count",
    features=["age", "vehicle_age", "area", "vehicle_group"],
    limitations=["No telematics data"],
    owner="Pricing Team",
)

report = ModelValidationReport(model_card=card, y_val=y_val, y_pred_val=y_pred_val)
report.generate("validation_report.html")
report.to_json("validation_report.json")
```

The JSON sidecar carries the `run_id` UUID, the RAG status, and all test metrics in a machine-readable structure. That UUID is the connection point to the MRM layer.

---

## The MRM side

`RiskTierScorer` assigns a risk tier from a 0-100 composite score across six dimensions: GWP impacted, model complexity, external data use, how long since last independent validation, drift trigger history, and regulatory exposure (production status, regulatory use, customer-facing pricing).

The scoring is auditable: each dimension contributes a documented number of points, and `TierResult.rationale` returns the full breakdown as a string. When a Tier 1 assignment is questioned at the Model Risk Committee, the answer is not "judgement" — it is: GWP £125m (25.0 pts) + high complexity GBM (20.0 pts) + champion deployment + customer-facing pricing (16.3 pts) = 72.1/100. Threshold for Tier 1 is ≥ 60. That is the answer.

```python
from insurance_governance import MRMModelCard, RiskTierScorer, ModelInventory, GovernanceReport

mrm_card = MRMModelCard(
    model_id="motor-freq-v3",
    model_name="Motor TPPD Frequency",
    version="3.2.0",
    model_class="pricing",
    intended_use="Frequency pricing for private motor.",
)

scorer = RiskTierScorer()
tier = scorer.score(
    gwp_impacted=125_000_000,
    model_complexity="high",
    deployment_status="champion",
    regulatory_use=False,
    external_data=False,
    customer_facing=True,
)
```

`ModelInventory` persists the model register to a JSON file. That file should be version-controlled alongside your code. It stores model metadata, tier assignments, validation run history (linked by `run_id`), review dates, and sign-off status. The `due_for_review()` method is worth running daily from CI: a Tier 1 model approaching its annual review deadline with status `live` (not `validated`) is a governance breach before it becomes a regulatory problem.

`GovernanceReport` produces the executive committee pack — not the full 40-page technical validation report, but a two-page summary covering model purpose, risk tier with scoring rationale, last validation RAG, key metrics (Gini, A/E, calibration), assumptions register, sign-off chain, and next review date. The audience for this document is the Chief Actuary and the Model Risk Committee, not the independent validation function.

---

## Why the integration matters

The validation subpackage generates a `run_id`. The MRM subpackage consumes it. When they lived in separate packages, that contract was documented but not enforced — if you upgraded one without the other, you were on your own.

With the unified package, the `run_id` format is guaranteed to be consistent across `ModelValidationReport.to_json()` and `ModelInventory.update_validation()`. The schema version is the package version. You cannot have a mismatch.

The complete workflow:

```python
from insurance_governance import (
    ModelValidationReport, ValidationModelCard,
    MRMModelCard, RiskTierScorer, ModelInventory, GovernanceReport,
)

# 1. Statistical validation
v_card = ValidationModelCard(name="Motor Frequency v3.2", version="3.2.0", ...)
report = ModelValidationReport(model_card=v_card, y_val=y_val, y_pred_val=y_pred_val)
report.generate("validation_report.html")
validation_json = report.to_dict()

# 2. Risk tier
mrm_card = MRMModelCard(model_id="motor-freq-v3", ...)
tier = RiskTierScorer().score(gwp_impacted=125_000_000, model_complexity="high", ...)

# 3. Inventory update
inventory = ModelInventory("mrm_registry.json")
inventory.register(mrm_card, tier)
inventory.update_validation(
    model_id="motor-freq-v3",
    validation_date="2026-03-14",
    overall_rag=validation_json["rag_status"],
    next_review_date="2027-03-14",
    run_id=validation_json["run_id"],  # from the same package: guaranteed consistent
)

# 4. Executive pack
GovernanceReport(card=mrm_card, tier=tier).save_html("mrm_pack.html")
```

There is no transcription step. The Gini that appears in the executive committee pack is the same Gini that was computed by the validation report. The `run_id` that the MRM system holds is the UUID that indexes the statistical evidence. If the PRA asks for the validation evidence supporting a specific model version, you retrieve the `run_id` from the inventory and retrieve the JSON. Two queries, no manual tracing through folders.

---

## The benchmarks

On synthetic UK motor data — 50,000 policies, CatBoost Poisson frequency model, 60/20/20 temporal split:

| Task | Time |
|---|---|
| Full SS1/23 validation suite | < 5 seconds |
| HTML validation report | < 1 second |
| JSON sidecar | < 1 second |
| `RiskTierScorer.score()` | < 1 ms |
| `ModelInventory.register()` | < 10 ms |
| `GovernanceReport` HTML | < 1 second |

The bottleneck in any validation workflow is the model fitting that produces `y_pred_val`, not this library. A team running quarterly validation for 15 models completes the full governance suite — statistical tests, HTML reports, JSON sidecars, inventory updates, executive packs — in under ten minutes of wall clock time, most of which is the CatBoost fits.

---

## When to use this and when not to

**Use this if:** you are subject to PRA SS1/23 (or expect to be when it extends to insurers), you have ten or more production pricing models, and you currently manage validation evidence in analyst notebooks and governance records in Excel. The library gives you consistent tests, consistent thresholds, and a machine-readable audit trail.

**Do not use this if:** you are in reserving or capital model governance — this package is scoped entirely to pricing models. It also does not replace independent human review of validation results. It automates the tests and structures the output; it does not make the judgement about whether the results are acceptable.

---

## Our view on the regulatory timeline

PRA SS1/23 came into force for banks on 17 May 2024. Extension to insurers is widely expected in the 2026-2027 window — 4most's January 2025 analysis and PRA supervision signals both point in that direction, though no formal notice has been issued. The FCA layer is already live: TR24/2 (August 2024) identified documentation failures as the primary gap across the firms it reviewed.

The firms that will handle the PRA extension without a scramble are the ones who can already answer these questions with a query: which models are in production at what tier, when was each last validated, who approved the findings, and what is the machine-readable link between the governance record and the statistical evidence?

An Excel workbook cannot answer these questions reliably. `insurance-governance` can.

---

`insurance-governance` is MIT-licensed at [github.com/burning-cost/insurance-governance](https://github.com/burning-cost/insurance-governance). Python 3.10+. Dependencies: polars, pydantic, jinja2, scikit-learn, numpy, scipy.

---

**Related articles from Burning Cost:**
- [Model Risk Governance for UK Insurers: Beyond the Excel Register](/2026/03/13/your-model-risk-register-is-a-spreadsheet/) — the original MRM library, now `insurance_governance.mrm`
- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/) — the monitoring layer that feeds post-deployment evidence back into the inventory
