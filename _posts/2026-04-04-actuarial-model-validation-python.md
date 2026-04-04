---
layout: post
title: "Actuarial Model Validation in Python: Automated Reports for UK Insurance Pricing"
seo_title: "Actuarial Model Validation in Python — Automated Reports for UK Insurance"
date: 2026-04-04
categories: [model-governance, tutorials]
tags: [model-validation, insurance-governance, python, consumer-duty, solvency-ii, MRM, PRA-CP6-24, hosmer-lemeshow, gini, ae-ratio, PSI, risk-tier, validation-report, UK-insurance, pricing-actuary]
description: "How to run actuarial model validation in Python for UK insurance pricing models. Covers Solvency II Article 120 and Consumer Duty requirements, the five-test validation suite, automated HTML reports, and MRM governance packs — using the open-source insurance-governance library."
author: Burning Cost
---

Most UK pricing teams validate their models the same way: one analyst, one notebook, the same five checks they ran last time, and a Word document that took three hours to format. When a new model goes to committee, the output looks different from the one reviewed six months ago because a different analyst ran it. The Hosmer-Lemeshow test appeared in last cycle's report but not this one. The Gini confidence interval was there in Q3 but not Q4. Nobody is sure whether the PSI threshold used for the motor model is the same as the one used for home.

This matters for reasons beyond tidiness. Under Solvency II Article 120, the validation function must be independent and must produce *consistent* documentation across the model estate. Under the FCA's Consumer Duty (PS22/9), pricing models must be demonstrably fair, which requires auditable evidence — not a bespoke notebook that only one person can interpret. The PRA's CP6/24, published in 2024, set expectations for insurance model risk management that closely track the structure of SS1/23 for banks: model inventory, tiered oversight, documented assumptions, regular review cycles.

The practical problem is that none of this specifies *how* to produce consistent, auditable validation documentation. CP6/24 tells you what to have. It says nothing about the format or tooling.

This post shows how to automate actuarial model validation in Python using the [`insurance-governance`](https://github.com/burning-cost/insurance-governance) library. The result: the same five-test validation suite for every model, every release, producing self-contained HTML reports that any reviewer can open without Python installed.

```bash
pip install insurance-governance
```

---

## What model validation actually means in UK insurance

It is worth being precise about what we mean. "Model validation" is used loosely to cover several distinct activities:

**Statistical validation** — testing whether the model's predictions are accurate on held-out data. Gini coefficient (discrimination), A/E ratio (calibration), Hosmer-Lemeshow (segment-level calibration), lift chart (rank ordering), PSI (whether the score distribution has shifted since training). This is what most pricing teams mean when they say "we validated the model".

**Model risk management (MRM)** — the governance layer. Documenting what the model does, what assumptions it rests on, what its limitations are, who approved it, when it is next due for review, and what tier of oversight it requires based on its materiality and complexity. This is what the PRA means when it says "model risk management".

**Monitoring** — ongoing tracking after deployment. PSI on the in-force book, A/E ratios at quarterly intervals, Gini drift tests. This is not validation — it is surveillance. It answers whether the model that passed validation twelve months ago is still fit for purpose today. We cover monitoring separately with [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) — the [practitioner's guide to insurance model monitoring](/blog/2026/04/04/insurance-model-monitoring-python-practitioner-guide/) has the full three-scenario walkthrough.

All three are required. Most pricing teams have ad hoc versions of statistical validation. Very few have systematic MRM. Almost none have consistent documentation across both.

---

## The regulatory basis

Before the code: the rules that make this non-optional.

**Solvency II Articles 120–126** require that the internal model (where used for capital) is validated independently, with regular review cycles, and with documentation that the Board can interrogate. For pricing models outside the internal model, the obligation is less prescriptive but the logic of Article 120 — that you must understand what your model is doing, and test that it does it — flows directly into PS12/22 supervisory expectations.

**FCA Consumer Duty (PS22/9, August 2022)** requires that pricing models produce fair value outcomes. A model you cannot validate is a model you cannot demonstrate is fair. The FCA has been explicit that "black box" is not an acceptable answer to a Consumer Duty question.

**FCA EP25/2 (proxy discrimination guidance, March 2025)** adds a validation requirement specific to fairness: if a pricing model uses variables that correlate with protected characteristics, you must test for and document disparate impact. This is a validation step that most teams do not currently run.

**PRA CP6/24 (insurance model risk, 2024)** sets expectations for model inventory, validation independence, tiered review frequency, and governance sign-off. It does not mandate a specific report format, but it does expect that documentation exists and is consistent across the model estate.

The common thread: you need documentation that is reproducible, comparable across models, and auditable by people who did not produce it. This is a documentation engineering problem as much as a statistics problem.

---

## The five-test validation suite

The `insurance-governance` library runs five tests. Each is standard actuarial practice; the value of the library is running them consistently and bundling the results into a single auditable HTML.

**1. Gini coefficient with bootstrap confidence interval.** Discrimination: does the model rank risks correctly? We use 500 bootstrap resamples to produce a 95% CI. A single point estimate is not sufficient for a governance report — you need to know whether Gini of 0.31 is consistent with 0.27 to 0.35, or whether the CI is so wide the estimate is meaningless.

**2. A/E ratio with Poisson confidence interval.** Global calibration: is the model's predicted frequency consistent with observed frequency at portfolio level? The CI is under Poisson assumptions on claim count, not a normal approximation. A/E CIs that exclude 1.0 trigger an amber flag; the threshold for global recalibration defaults to 1.10.

**3. Hosmer-Lemeshow test.** Segment-level calibration. The model is stratified into deciles by predicted probability; observed vs expected is compared within each decile. This catches the case that model B in the README benchmark illustrates: a model with A/E ratio of 1.00 at portfolio level that is 18% off in the young driver segment and -15% off in the mature driver segment, with the two biases cancelling in aggregate. HL p < 0.05 triggers amber; p < 0.001 triggers red.

**4. Lift chart.** Ten-decile double-lift chart comparing the model under review against the incumbent (if provided). If no incumbent is supplied, it falls back to a single-model lift chart against a naive baseline. The chart is rendered into the HTML report.

**5. Population Stability Index.** Score distribution drift between training and validation periods. PSI > 0.10 is amber; PSI > 0.25 is red. We are explicit in the README that PSI alone is not sufficient to detect population drift of the type that degrades model performance — see the Model C benchmark result. A/E CI is the more sensitive early signal for the kind of drift that actually matters.

---

## Running validation in Python

Here is the full workflow for a motor frequency model.

```python
import numpy as np
from insurance_governance import ModelValidationReport, ValidationModelCard

rng = np.random.default_rng(42)
n = 5_000

# Held-out validation set: claim counts and model predictions
y_val      = rng.poisson(0.08, n).astype(float)
y_pred_val = np.clip(rng.normal(0.08, 0.02, n), 0.001, None)
exposure   = rng.uniform(0.5, 1.0, n)

# Model card: mandatory metadata for the report header
card = ValidationModelCard(
    name        = "Motor Frequency v3.2",
    version     = "3.2.0",
    purpose     = "Predict claim frequency for UK private motor portfolio",
    methodology = "CatBoost gradient boosting with Poisson objective",
    target      = "claim_count",
    features    = ["age", "vehicle_age", "area", "vehicle_group"],
    limitations = ["No telematics data", "Trained on 2020–2023; excludes cost-of-living exposure shift"],
    owner       = "Pricing Team",
)

report = ModelValidationReport(
    model_card         = card,
    y_val              = y_val,
    y_pred_val         = y_pred_val,
    exposure_val       = exposure,
    monitoring_owner   = "Head of Pricing",
    monitoring_triggers = {"ae_ratio": 1.10, "psi": 0.25},
)

# Green, amber, or red — machine-readable for CI/CD gates
print(report.get_rag_status())

# Self-contained HTML — no Python required to open it
report.generate("motor_freq_v3.2_validation.html")
```

The output is a self-contained HTML file. It contains the five test results, the Gini bootstrap plot, the lift chart, the A/E with CI, the HL p-value, the PSI band, and the monitoring plan. A reviewer can open it without Python. You can attach it to a committee pre-read. You can check it into git alongside the model artefact.

### Adding a double-lift chart and fairness section

If you are replacing an existing champion model, pass the incumbent predictions:

```python
report = ModelValidationReport(
    model_card        = card,
    y_val             = y_val,
    y_pred_val        = y_pred_val,
    exposure_val      = exposure,
    incumbent_pred_val = y_pred_incumbent,  # challenger vs champion
    fairness_group_col = age_band,           # numpy array of group labels
    monitoring_owner  = "Head of Pricing",
    monitoring_triggers = {"ae_ratio": 1.10, "psi": 0.25},
)
```

The `fairness_group_col` argument adds a disparate impact section to the report: A/E ratios broken out by group, with flags where any group's CI excludes the portfolio average. This is the documentation step that CP6/24 and EP25/2 require if the model uses variables that proxy protected characteristics.

---

## What the five tests actually catch

The README benchmarks three synthetic scenarios on 5,000 policies. The headline result is worth repeating:

| Scenario | Manual 4-check | Automated 5-test | Key diagnostic |
|---|---|---|---|
| Model A (well-specified) | 4/4 pass | 5/5 pass | Gini CI tight at 0.29–0.37 |
| Model B (miscalibrated) | Flags A/E | Flags A/E **+ HL** | HL p < 0.0001 — age-band bias averages out in global A/E |
| Model C (drifted) | Passes PSI | Flags A/E CI | PSI = 0.189 (below 0.25 threshold); A/E CI excludes 1.0 |

Model B is the one that matters most. A model with systematic age-band miscalibration will read A/E ≈ 1.00 at portfolio level if the age bands are balanced. A human reviewer looking only at the global A/E will pass it. Hosmer-Lemeshow flags it at p < 0.0001. This is not an edge case — it is the standard failure mode of a model that was fit on data where certain segments were underrepresented.

Model C illustrates the limits of PSI. PSI at 0.189 is below the conventional 0.25 amber threshold and would be classified as minor drift by most governance frameworks. The A/E confidence interval — which tests whether the model's aggregate prediction is consistent with observed losses in the new period — excludes 1.0. The model is miscalibrated on the drifted population despite a PSI that looks acceptable. PSI tells you the score distribution has changed. The A/E CI tells you the model is wrong. These are different questions.

---

## The MRM governance layer

Statistical validation answers "is this model accurate?" MRM governance answers "does this model have the right oversight for its materiality?"

CP6/24 expects insurers to classify models by risk tier: high-materiality models need annual review and board-level sign-off; lower-tier models can run on longer cycles with delegation to the Chief Actuary or Head of Pricing. The `RiskTierScorer` in `insurance-governance` makes this classification deterministic and auditable.

```python
from insurance_governance import MRMModelCard, RiskTierScorer, GovernanceReport, Assumption

card = MRMModelCard(
    model_id      = "motor-freq-v3",
    model_name    = "Motor TPPD Frequency",
    version       = "3.2.0",
    model_class   = "pricing",
    intended_use  = "Frequency pricing for UK private motor. Not for commercial fleet.",
    assumptions   = [
        Assumption(
            description = "Claim frequency stationarity since 2022",
            risk        = "MEDIUM",
            mitigation  = "Quarterly A/E monitoring with 1.10 trigger for review",
        ),
        Assumption(
            description = "Vehicle group relativities from 2019 ABI data",
            risk        = "LOW",
            mitigation  = "Annual review against current ABI benchmarks",
        ),
    ],
)

tier = RiskTierScorer().score(
    gwp_impacted      = 125_000_000,  # £125m of premium affected
    model_complexity  = "high",
    deployment_status = "champion",
    regulatory_use    = False,
    external_data     = False,
    customer_facing   = True,
)

GovernanceReport(card=card, tier=tier).save_html("motor_freq_mrm_pack.html")
```

The `RiskTierScorer` scores six dimensions: GWP materiality, model complexity, external data reliance, validation recency, drift history, and whether the model is customer-facing (relevant under Consumer Duty). The composite 0–100 score maps to three tiers:

- **Tier 1 (≥60):** Annual review, Model Risk Committee sign-off. This is where a £125m motor frequency model lands.
- **Tier 2 (30–59):** 18-month review, Chief Actuary sign-off.
- **Tier 3 (<30):** 24-month review, Head of Pricing sign-off.

The MRM governance pack produced by `GovernanceReport.save_html()` contains: model purpose, tier rationale with dimension-level scores, last RAG status, assumptions register with risk ratings, outstanding issues, approval conditions, and the next scheduled review date. Everything the MRC pre-read needs, in a format that is identical for every model on the estate.

---

## Tracking the model inventory

If you have ten production pricing models, you need a machine-readable inventory that tracks validation history and surfaces overdue reviews. The `ModelInventory` class manages this as a JSON file that you check into git alongside your model artefacts.

```python
from insurance_governance import ModelInventory

# Create or load inventory
inv = ModelInventory("model_inventory.json")

# Register a model after validation
inv.register(
    model_id       = "motor-freq-v3",
    model_name     = "Motor TPPD Frequency v3.2",
    tier           = 1,
    last_validated = "2026-04-04",
    rag_status     = "green",
    next_review    = "2027-04-04",
)

# Surface models past their review date
overdue = inv.list_overdue()
for m in overdue:
    print(f"{m.model_name}: due {m.next_review}, currently {m.rag_status}")
```

The inventory is a plain JSON file. It checks into your pricing model repository. Every CI/CD run that regenerates validation reports updates the entry. `list_overdue()` can gate a CI job — models past their review date block deployment of new versions until the governance cycle is complete.

---

## Connecting validation to monitoring

Validation is a point-in-time activity. Monitoring is ongoing. The connection between them needs to be explicit: the monitoring triggers specified at validation time should be the same thresholds that fire a governance review in production.

In the workflow above, `monitoring_triggers={"ae_ratio": 1.10, "psi": 0.25}` is written into the validation report HTML. When the operational monitoring (via `insurance-monitoring`) fires an alert, the on-call analyst can pull the original validation report and see that the A/E threshold was agreed at validation, documented by the validating analyst, and signed off by the committee that reviewed the report.

This is what the PRA means by "end-to-end audit trail". Not a ticket in Jira. Not a Slack message that says "PSI is looking a bit high". A chain of documented decisions from training, through validation, through deployment, through monitoring, all pointing to the same artefacts.

---

## Putting it into CI/CD

A validation report that is only produced when a human decides to run it is not a governance control. The pattern we recommend:

```yaml
# .github/workflows/validate.yml (or equivalent in your CI system)
# Run on any push to a model branch
- name: Run validation suite
  run: |
    python run_validation.py
    # Fail the job if any model is red-RAG
    python -c "
    from insurance_governance import ModelValidationReport
    import json
    with open('validation_results.json') as f:
        results = json.load(f)
    for model, status in results.items():
        assert status != 'red', f'{model} is red-RAG — review before merge'
    "
```

Red-RAG models block the merge. Amber-RAG models produce a warning but do not block — they require a documented review comment explaining why the amber is acceptable. Green models merge cleanly. The HTML reports are uploaded as artefacts. The inventory JSON is updated and committed.

This is not a novel idea. It is exactly what software engineering teams have done with test suites since the 1990s. Pricing teams are the last holdout.

---

## What this does not cover

Be clear about the limits:

**It does not validate the features.** The tests tell you whether the model's predictions are good on the validation set. They do not tell you whether the features are well-specified, whether the training sample is representative, or whether the model will hold up under regulatory challenge of its underlying variables. That requires actuarial judgement.

**It does not replace independent validation.** CP6/24 expects the validation function to be independent of model development. Running these tests yourself, on your own model, satisfies the documentation requirement but not the independence requirement. The HTML report is evidence for an independent validator to review — it is not a substitute for that review.

**It is for pricing models.** The library does not cover reserving models, capital models, or credit risk models. The statistical assumptions (Poisson frequency, GLM/GBM scoring) are calibrated to UK personal lines pricing.

**It does not assess proxy discrimination directly.** The `fairness_group_col` argument flags A/E disparities by group. It does not run the full EP25/2 proxy discrimination test — for that, use [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness), which feeds its output directly into the governance pack.

---

## The full stack

`insurance-governance` sits at the governance layer of a broader Python stack for UK insurance pricing. The libraries that feed into it:

- [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) — PSI, Gini drift, A/E ratios on deployed models. The drift signals that trigger a governance review. See the [glum workflow post](/blog/2026/04/04/glum-insurance-pricing-python-workflow/) for how monitoring integrates with the broader fitting and governance pipeline.
- [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) — EP25/2 proxy discrimination audit. The fairness evidence that goes into the governance pack.
- [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) — distribution-free prediction intervals. The uncertainty quantification that belongs in the validation report for any model used in reserving or capital.

The five-library end-to-end pipeline is documented in [Five Libraries, One Pipeline](/2026/04/04/five-libraries-one-pipeline-end-to-end-motor-pricing/).

---

The point of all of this is not to generate paperwork. It is to make the paperwork automatic so that the humans can spend their time on the decisions the paperwork is supposed to inform — not on formatting Word documents at 11pm the night before a committee meeting.
