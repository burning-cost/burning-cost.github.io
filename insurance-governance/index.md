---
layout: page
title: "insurance-governance"
description: "Unified model governance for UK insurance pricing. PRA SS1/23 model validation reports and MRM governance packs in one install."
permalink: /insurance-governance/
schema: SoftwareApplication
github_repo: "https://github.com/burning-cost/insurance-governance"
pypi_package: "insurance-governance"
---

Unified model governance for UK insurance pricing teams. Combines statistical model validation and model risk management into one package, with outputs structured to align with PRA SS1/23 principles.

**[View on GitHub](https://github.com/burning-cost/insurance-governance)** &middot; **[PyPI](https://pypi.org/project/insurance-governance/)**

---

## The problem

Model validation tests and MRM governance packs were built separately, installed separately, and coupled manually. Pricing teams either managed both and maintained the coupling themselves, or skipped one. This package resolves that: one install, two coherent subpackages that share the same model card schema.

**Regulatory note:** PRA SS1/23 is directed at banks, not insurers. Insurance MRM obligations come from PS12/22, Solvency II internal model requirements, and EIOPA validation guidelines. In practice, most UK insurance MRM frameworks reference SS1/23 by analogy — it articulates sound principles regardless of firm type. This library uses SS1/23 as a reference framework in that spirit.

---

## Installation

```bash
pip install insurance-governance
```

---

## Quick start

```python
import numpy as np
from insurance_governance import (
    ModelValidationReport,
    ValidationModelCard,
    MRMModelCard,
    RiskTierScorer,
    GovernanceReport,
)

rng = np.random.default_rng(42)
n_val = 5_000
y_val        = rng.poisson(0.08, n_val).astype(float)
y_pred_val   = np.clip(rng.normal(0.08, 0.02, n_val), 0.001, None)
exposure_val = rng.uniform(0.5, 1.0, n_val)

# Statistical validation
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
report = ModelValidationReport(
    model_card=card,
    y_val=y_val,
    y_pred_val=y_pred_val,
    exposure_val=exposure_val,
)
report.generate("validation_report.html")

# MRM governance pack
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
GovernanceReport(card=mrm_card, tier=tier).save_html("mrm_pack.html")
```

---

## Key API reference

### `insurance_governance.validation` subpackage

**`ModelValidationReport`** — runs the full statistical validation suite and produces a self-contained HTML report.

```python
ModelValidationReport(
    model_card,         # ValidationModelCard
    y_val,              # observed values (numpy array)
    y_pred_val,         # predicted values (numpy array)
    exposure_val,       # exposure weights (numpy array)
)
```

- `.generate(path)` — writes HTML report
- `.results` — dictionary of `TestResult` objects keyed by test name

**`ValidationModelCard`** — metadata container for a model under validation. Stores name, version, purpose, methodology, features, limitations, owner.

**Individual report components:**

```python
from insurance_governance.validation import (
    PerformanceReport,      # Gini, lift charts, double-lift
    DiscriminationReport,   # Lorenz curve, concentration statistics
    StabilityReport,        # PSI per feature, temporal stability
    DataQualityReport,      # completeness, range checks
)
```

**Result types:**

```python
from insurance_governance import (
    TestResult,     # individual test: name, value, band (RAG), passed
    RAGStatus,      # enum: GREEN / AMBER / RED
    Severity,       # enum: INFO / WARNING / CRITICAL
    TestCategory,   # enum groups tests by type
)
```

### `insurance_governance.mrm` subpackage

**`RiskTierScorer`** — produces an objective 0–100 risk score mapping to Tier 1/2/3 from model metadata.

```python
scorer = RiskTierScorer()
tier = scorer.score(
    gwp_impacted,       # float, GWP impacted in £
    model_complexity,   # "low" | "medium" | "high"
    deployment_status,  # "champion" | "challenger" | "shadow"
    regulatory_use,     # bool
    external_data,      # bool
    customer_facing,    # bool
)
# tier.tier: 1 | 2 | 3
# tier.score: float 0-100
# tier.rationale: str
```

**`ModelInventory`** — JSON file registry for the model estate. Tracks all models, versions, owners, and risk tiers.

```python
inventory = ModelInventory("models.json")
inventory.register(mrm_card, tier_result)
inventory.list_by_tier(tier=1)
```

**`GovernanceReport`** — produces an executive committee pack in HTML, combining model card, risk tier, and validation results.

```python
GovernanceReport(card=mrm_card, tier=tier).save_html("mrm_pack.html")
```

**`MRMModelCard`** — model risk metadata container. Stores model_id, model_name, version, model_class, intended_use, assumptions, limitations, dimension scores.

---

## Benchmark

Automated validation suite catches age-band miscalibration that manual checklists miss. The HTML output includes a bootstrap Gini CI, Poisson A/E CI, double-lift chart, and renewal cohort test — the standard deliverables for a UK pricing model validation.

---

## Related blog posts

- [One Package, One Install: PRA SS1/23 Validation and MRM Governance Unified](https://burning-cost.github.io/2026/03/14/insurance-governance-unified-pra-ss123-validation/)
- [The Governance Bottleneck for EBM Adoption](https://burning-cost.github.io/2026/03/16/the-governance-bottleneck-for-ebm-adoption/)
