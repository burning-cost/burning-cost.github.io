---
layout: post
title: "Your Model Risk Register Is a Spreadsheet"
date: 2026-03-19
categories: [libraries, regulation]
tags: [model-risk-management, mrm, pra-ss123, fca-consumer-duty, model-inventory, model-governance, python, insurance-mrm, insurance-validation, insurance-monitoring]
description: "PRA SS1/23 is expected to extend to insurers by 2026-2027. Most UK pricing teams will face that review with a Word template and an Excel model register. insurance-mrm is our governance layer: model inventory, risk tier scoring, and executive committee reports — built on top of insurance-validation and insurance-monitoring."
---

If the PRA asked you today for your model inventory, what would you send them?

We know what most UK insurers would send. A shared Excel workbook with a tab called "Model Register." Forty or fifty rows. Columns for model name, owner, last review date, status. The last review date column has blanks in it. The status column says "Live" for everything. There is no link to the validation report. There is no change log. The person who set it up left eighteen months ago.

That is not a model risk register. It is a list.

The distinction matters because the PRA's 2026 supervision priorities letter — issued in January 2026 — specifically flagged "gaps between assumed and realised profitability" as an area of supervisory focus. That is the PRA asking whether your pricing models are doing what you said they would do when you approved them. Answering that question requires more than a list. It requires a documented, version-linked, ongoing record of model performance, sign-off, and review status.

[`insurance-mrm`](https://github.com/burning-cost/insurance-mrm) is our 32nd open-source library. It is the governance layer for your pricing model estate: a persistent model inventory, objective risk tier scoring, and an executive committee report. It does not re-implement any statistical tests — it wraps [`insurance-validation`](https://github.com/burning-cost/insurance-validation) and [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring), which already handle those. What it adds is the governance workflow that connects statistical output to institutional accountability.

```bash
pip install insurance-mrm
```

---

## The regulatory direction of travel

PRA SS1/23 — Model Risk Management Principles for Banks — came into force on 17 May 2024. It is currently scoped to banks, building societies, and PRA-designated investment firms. Insurance firms are not formally in scope. That will change.

Deloitte (2025), 4most (January 2025), and the PRA's own supervision signals all converge on the same assessment: extension to insurers is a matter of when, not if. The most credible timeline is post-Solvency UK embedding, which points to 2026-2027. The firms that will handle that extension well are the ones who start building the infrastructure now, not the ones who scramble to retrofit documentation after the formal notice arrives.

SS1/23's five principles are: (1) model identification and risk classification, (2) governance, (3) model development, implementation, and use, (4) independent validation, and (5) model risk mitigants. The critical text in Principle 3 is that documentation must allow "a party unfamiliar with the model to understand its operation." In Principle 1, every model must be classified by complexity, business use, and materiality, with a change log.

That is a model inventory with structured metadata and an audit trail — not a spreadsheet.

The FCA layer is already live. PRIN 2A requires firms to "regularly assess, test, understand and evidence the outcomes their customers are receiving." TR24/2 (August 2024) found that documentation failures — not model failures — are the primary enforcement trigger. The firms that got caught were not running bad models. They were running models with no documented methodology for how performance metrics were chosen, no record of who approved the validation findings, and no trail linking the validation report to the model version that was deployed.

---

## Why ValidMind is not the answer

The only serious commercial MRM platform in the Python ecosystem is ValidMind. It is worth understanding why it does not work for UK insurance pricing teams.

ValidMind's open-source library ships 200+ pre-configured validation tests. It generates structured HTML reports. It references SS1/23 and SR 11-7 explicitly. On paper, it looks like the right tool.

Three problems. First, the licence. ValidMind's library is AGPL v3. Any commercial pricing team that incorporates AGPL code into its workflow must either open-source all downstream code or purchase a commercial licence. That is a hard blocker at most insurers — the legal team will not allow it, and the conversations required to get a commercial agreement signed will take longer than just building the governance layer yourself.

Second, the target market. ValidMind is built for US and EU banking compliance: credit scoring, fraud detection, market risk. It has no Gini/Lorenz support, no A/E calibration by decile, no lift charts, no FCA Consumer Duty structure, no UK proxy discrimination testing under the Equality Act 2010. It is a banking tool wearing a general MRM badge.

Third, the cloud dependency. The ValidMind library is a test runner; rendering documentation requires their SaaS. You cannot run the full workflow on-premise. For insurers with air-gapped model development environments, that is disqualifying.

`insurance-mrm` is MIT-licensed, runs anywhere Python runs, and is built specifically for UK insurance pricing governance.

---

## What the library does

The library has four components: `ModelCard`, `RiskTierScorer`, `ModelInventory`, and `GovernanceReport`. They compose into a complete governance workflow.

### ModelCard: structured model metadata

The `ModelCard` in `insurance-mrm` extends the `ModelCard` in `insurance-validation`. Where the validation card is primarily a metadata header for the technical validation report, the MRM card is designed for distribution to governance committees, model risk functions, and — eventually — regulators.

It adds: model class (`pricing`, `reserving`, `capital`, or `underwriting`), a training data period, champion/challenger status, an assumptions register with per-assumption risk levels, explicit `not_intended_for` declarations, and outstanding issues.

```python
from insurance_mrm import ModelCard, Assumption

card = ModelCard(
    model_id='motor-freq-tppd-v2.1',
    model_name='Motor TPPD Frequency',
    version='2.1.0',
    model_class='pricing',
    target_variable='claim_count',
    distribution_family='Poisson',
    rating_factors=['driver_age', 'vehicle_age', 'annual_mileage', 'region'],
    training_data_period=('2019-01-01', '2023-12-31'),
    development_date='2024-09-01',
    developer='Pricing Team',
    champion_challenger_status='champion',
    intended_use='Frequency pricing for private motor. Not commercial. Not for reserving.',
    not_intended_for=['Commercial motor', 'Fleet', 'Reserving', 'Capital'],
    assumptions=[
        Assumption(
            description='Claim frequency stationarity since 2022',
            risk='MEDIUM',
            mitigation='Quarterly A/E monitoring with PSI alert if PSI > 0.25',
        ),
        Assumption(
            description='Region as proxy for road density',
            risk='LOW',
            mitigation='None — accepted residual limitation',
        ),
    ],
    limitations=['Degrades for vehicles >10 years (thin data)', 'No telematics integration'],
    outstanding_issues=['VIF check on driver_age × annual_mileage pending'],
    approved_by=['Chief Actuary', 'Model Risk Committee'],
    approval_date='2024-10-15',
    next_review_date='2025-10-15',
    monitoring_owner='Sarah Ahmed, Head of Pricing Analytics',
    monitoring_frequency='Quarterly',
    monitoring_triggers={
        'psi_score': 0.25,
        'ae_ratio_deviation': 0.10,
        'gini_drop_pct': 0.05,
    },
)
```

The `not_intended_for` list is not decorative. SS1/23 Principle 3 requires documentation that specifies the model's scope of valid application. A model that is used outside its documented scope — because someone did not know it was not appropriate for reserving — is a governance failure that starts with absent documentation.

The `monitoring_triggers` dictionary feeds directly into `insurance-monitoring`: when a monitoring run produces a PSI above 0.25, the library can fire a review trigger automatically against the card's registered thresholds. That connection — between live monitoring output and the governance record — is the audit trail the PRA wants to see.

---

### RiskTierScorer: objective tier assignment

The standard in practice at UK insurers is for a pricing actuary to assign a risk tier by judgment. Tier 1, Tier 2, Tier 3 — someone decides, writes a justification, and that is the tier. This produces defensible documentation in good-faith firms and compliance theatre in bad-faith ones. The PRA's supervision framework cannot distinguish between the two.

`RiskTierScorer` replaces judgment with a scored rubric. Six factors, weighted and summed to a score from 0 to 100:

| Factor | Points |
|---|---|
| GWP impacted: ≥ £100m | 25 pts |
| GWP impacted: £25m–£100m | 15 pts |
| GWP impacted: £5m–£25m | 8 pts |
| GWP impacted: < £5m | 3 pts |
| Model complexity: high (GBM, ensemble, neural net) | 20 pts |
| Model complexity: medium (GLM, single decision tree) | 12 pts |
| Model complexity: low (scorecard, linear) | 5 pts |
| Production champion | 15 pts |
| Regulatory use (capital or regulatory reporting) | 15 pts |
| External data sources (postcode, credit, telematics) | 10 pts |
| Customer-facing pricing | 15 pts |

Score ≥ 60 → Tier 1. Score 30–59 → Tier 2. Score < 30 → Tier 3.

The GWP thresholds are calibrated to UK personal lines scale: the £100m threshold aligns with the PRA's materiality framing for significant insurers; the £25m lower bound roughly mirrors BIBA's mid-tier firm classification. A motor frequency model running on a £125m GWP book, priced using a GBM, deployed as champion, with customer-facing pricing, scores 75 — Tier 1. That is an auditable number with an auditable rationale.

```python
from insurance_mrm import RiskTierScorer

scorer = RiskTierScorer()
tier_result = scorer.score(
    gwp_impacted=125_000_000,
    model_complexity='high',
    deployment_status='champion',
    regulatory_use=False,
    external_data=False,
    customer_facing=True,
)

print(tier_result.tier)
# 1

print(tier_result.score)
# 75

print(tier_result.rationale)
# 'Tier 1 (Critical): materiality 25.0/25 (GWP £125.0m ≥ £100m threshold) + ...'
```

The tier determines the review schedule. Tier 1: annual revalidation, Model Risk Committee sign-off. Tier 2: 18-month revalidation, Chief Actuary sign-off. Tier 3: 24-month revalidation, Pricing Head sign-off. These thresholds are configurable, but the defaults are aligned with standard UK MRM practice for general insurance.

---

### ModelInventory: the persistent register

This is the replacement for the Excel workbook.

```python
from insurance_mrm import ModelInventory

inventory = ModelInventory(path='mrm_registry.json')
inventory.register(card, tier_result)

# What's due for review in the next 60 days?
due = inventory.due_for_review(within_days=60)

# Record a validation run
inventory.update_validation(
    model_id='motor-freq-tppd-v2.1',
    validation_date='2025-03-15',
    overall_rag='GREEN',
    next_review_date='2026-03-15',
    run_id='a3f7e2c1-...',   # UUID from insurance-validation JSON sidecar
)

# Full inventory as a list of dicts
models = inventory.list()
```

The inventory is file-backed JSON. No database, no infrastructure dependencies. It runs in a notebook, a CI pipeline, or a model development environment with no internet access. A mid-tier insurer has maybe 50-100 production models — JSON scales to thousands, which is more than enough.

The `run_id` field is the connection point between governance and statistical evidence. When `insurance-validation` generates a validation report, it produces a JSON sidecar with a UUID. `ModelInventory.update_validation()` takes that UUID and stores it against the model record. You now have a machine-readable link from the governance register to the specific validation run. If the PRA asks "what validation evidence supports this model's continued use?" — you query the inventory, retrieve the `run_id`, and retrieve the JSON that carries the Gini, A/E ratios, RAG status, and all test results.

The `due_for_review()` method is worth running as a scheduled job — daily, from a CI pipeline, with a Slack notification. A Tier 1 model where `next_review_date` is within 30 days and `status` is `live` (not `validated`) is a governance breach waiting to happen. Surfacing it automatically removes the "we didn't know" defence.

---

### GovernanceReport: the executive committee pack

`insurance-validation` produces a technical validation report: nine sections, full statistical methodology, 40-50 pages of HTML when printed. That report goes to the independent validation function and the model risk committee.

The governance committee — ExCo, board risk committee, regulator — needs something different. They need a 1-2 page summary: what does the model do, what tier is it, what did the last validation find, who approved it, when is the next review, what are the key risks.

`GovernanceReport` generates that summary.

```python
from insurance_mrm import GovernanceReport

report = GovernanceReport(
    card=card,
    tier=tier_result,
    validation_results={'overall_rag': 'GREEN', 'gini': 0.42, 'ae_ratio': 1.02},
    monitoring_results={'psi_score': 0.08, 'ae_ratio': 0.98},
)

report.save_html('governance_pack.html')
```

The HTML output covers: model identity and purpose, risk tier with scoring rationale, current RAG status drawn from the validation JSON, key validation findings (Gini, A/E, calibration — the headline numbers, not the full statistical narrative), the assumptions register with risk levels flagged, the sign-off chain with dates, monitoring plan summary, next review date, and outstanding issues.

It is readable in five minutes by someone who has not seen the technical validation report. That is the audience for this document.

The optional `validation_results` and `monitoring_results` parameters accept plain dicts — you do not need `insurance-validation` or `insurance-monitoring` installed. But when you do have them, pass the output of their `to_dict()` methods and the report renders the headline metrics automatically. If the last validation produced an AMBER RAG — triggered by a renewal cohort flag on the 5+ year tenure band — that context appears in the executive pack. The committee is looking at the same evidence, condensed rather than rewritten.

---

## The complete workflow

The three libraries compose into a pipeline.

```python
from insurance_mrm import ModelCard, RiskTierScorer, ModelInventory, GovernanceReport

# 1. Define the model
card = ModelCard(model_id='motor-freq-tppd-v2.1', ...)
tier = RiskTierScorer().score(gwp_impacted=125_000_000, ...)

# 2. Register it
inventory = ModelInventory('mrm_registry.json')
inventory.register(card, tier)

# 3. Record a validation run (insurance-validation provides the metrics)
inventory.update_validation(
    model_id='motor-freq-tppd-v2.1',
    validation_date='2025-03-15',
    overall_rag='GREEN',
    next_review_date='2026-03-15',
    run_id='a3f7e2c1-...',
)

# 4. Generate executive committee pack
GovernanceReport(
    card=card, tier=tier,
    validation_results={'overall_rag': 'GREEN', 'gini': 0.42},
).save_html('governance_pack.html')

# 5. Check what needs reviewing (run daily from CI)
print(inventory.due_for_review(within_days=30))
```

The validation statistics are computed once, by `insurance-validation`. The governance record is updated once, by `ModelInventory`. The executive report is generated automatically from both. There is no transcription, no copy-paste from an Excel tracker to a Word template, no risk of the governance record diverging from the statistical evidence.

---

## The white space this fills

It is worth being precise about what `insurance-mrm` does and does not do.

It does not compute PSI, Gini, A/E ratios, calibration tests, or any other statistical measure. `insurance-validation` and `insurance-monitoring` do that. Duplicating those tests in this library would create a maintenance burden and a divergence risk — two libraries computing the same statistic with slightly different implementations, producing different numbers depending on which one you run.

What the library adds is everything that neither validation nor monitoring provides: a persistent model register that does not require a human to remember to update it; an objective tier scoring methodology that produces an auditable rationale rather than a judgment; a governance workflow that tracks sign-off status, review dates, and escalation triggers; and an executive committee report that is distinct from the technical validation report in audience, format, and purpose.

The gap this fills is not statistical. The gap is institutional. The reason most UK pricing teams' model governance is inadequate is not that they lack statisticians — it is that there is no tooling that connects the statistical output to the governance workflow in a way that is reproducible, auditable, and appropriate for the regulatory context they operate in.

---

## Our view

The PRA SS1/23 extension to insurers will happen. The FCA Consumer Duty documentation obligations are already live. The Mills Review signals tightening accountability for AI models under SM&CR. None of this is speculative.

The firms that will handle the regulatory transition smoothly are the ones that can answer the following questions with a query, not a conversation:

- Which models are in production, at what tier, with what approval status?
- When was each model last validated, by whom, and what did the validation find?
- Which models are due for review in the next 90 days?
- For a specific model version, what is the link to the statistical evidence supporting its continued use?

An Excel workbook cannot answer these questions reliably. `insurance-mrm` can.

If your team is already using `insurance-validation` and `insurance-monitoring`, the incremental work to adopt `insurance-mrm` is low — the integration points are designed to be frictionless, and the run_id linkage between the libraries requires one additional line. If you are starting from scratch, the three libraries together give you a complete, PRA-aligned, auditable pricing governance workflow in Python, under MIT licence, with no infrastructure dependencies.

---

`insurance-mrm` is open source under the MIT licence at [github.com/burning-cost/insurance-mrm](https://github.com/burning-cost/insurance-mrm). Python 3.10+, zero mandatory dependencies beyond the standard library. `insurance-validation` and `insurance-monitoring` are optional — the library works standalone but produces richer reports when they are available.
