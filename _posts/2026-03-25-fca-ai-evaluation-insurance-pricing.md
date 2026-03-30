---
layout: post
title: "FCA AI Evaluation 2026: What the Regulator Is Looking at"
date: 2026-03-25
categories: [regulation, insurance-pricing]
tags: [fca, regulation, model-governance, fairness, consumer-duty, ai, insurance-governance, insurance-monitoring, insurance-fairness, smcr, ps21-5]
description: "The FCA has committed to evaluating how UK insurers use AI in pricing, underwriting, and claims. No new AI-specific rulebook is coming — but the regulator now expects evidence, not assertions. Here is what that means for a pricing team."
---

The FCA is not writing a new AI rulebook. They have said this clearly and repeatedly — most recently in their published approach to AI regulation, where the position is stated as: "We do not plan to introduce extra regulations for AI. Instead, we'll rely on existing frameworks." That is the good news, and it is genuine.

The bad news is what "existing frameworks" means in practice when applied by a supervisor who now understands pricing models well enough to ask pointed questions. Consumer Duty, PS21/5, SMCR, the Equality Act — together they impose an evidence standard that most pricing teams have never had to meet. The FCA's 2026 AI evaluation programme, announced in the regulator's insurance priorities letter published in February 2026, will assess how firms use AI across underwriting, pricing, and claims. The evaluation will run through 2026 and an industry report is expected.

This post covers what the FCA is looking at, why existing frameworks already reach most AI risks in pricing, and what a prepared pricing team should be able to produce.

---

## What the FCA actually said

The FCA's 2026 insurance priorities letter set out four focus areas. AI sits under the third — growth and innovation — but it reaches across all four. The specific commitment is to "evaluate the risks and opportunities of AI for insurance, assess how firms are using AI in their operations, and identify barriers to adoption."

That framing matters. The FCA is running an evaluation, not an enforcement sweep. They want to understand the landscape — who is using AI for what, how it is governed, what the consumer outcomes look like. The output will shape future supervisory expectations, which is precisely why it is worth taking seriously now.

The AI Live Testing programme, for which the second cohort opened in January 2026, gives context. FS25/5 — the feedback statement on AI Live Testing published in September 2025 — received 67 responses. The FCA's stated objective is to give firms "confidence to deploy AI systems in a way that drives growth and delivers positive outcomes for UK consumers and markets." The emphasis on positive outcomes is not accidental.

The FCA has also been consistent since at least FS23/6 (October 2023, responding to the joint DP5/22 discussion paper on AI with the PRA and Bank of England) that consumer protection is the primary area for supervisory attention. FS23/6 was unambiguous: AI could create risks through bias, discrimination, lack of explainability, and exploitation of vulnerable consumers — and firms using AI in ways that embed or amplify bias would not be acting in good faith under Consumer Duty unless differences in outcome could be objectively justified.

---

## Three things the FCA cares about most

### 1. Consumer outcomes

Consumer Duty (PRIN 2A, live July 2023) requires firms to demonstrate that products deliver good outcomes for all customers. For a pricing model, "demonstrate" means something specific: you need data showing that the model's output — the premium paid — correlates with risk borne, across all customer segments, in a way that a reasonable regulator would describe as fair value.

This is harder than it sounds. A model can be actuarially sound in aggregate and still produce outcomes that systematically disadvantage particular groups. The FCA's EP25/2 evaluation published in early 2025 — which assessed the effectiveness of the PS21/5 pricing remedies introduced in January 2022 — found that the price-walking intervention saved consumers an estimated £4.2bn over a ten-year period. The regulator has the appetite to intervene directly when pricing practices harm consumers. AI-driven pricing that replicates harm through a more sophisticated mechanism will not be treated differently.

The practical question for a pricing team is: can you show, for any customer segment the FCA might define, that the model is calibrated and the premium is proportionate to risk? If the answer involves running a one-off query against a data warehouse, that is not good enough. It needs to be a systematic, documented process.

### 2. Fairness and discrimination

The FCA reiterated in its 2024 AI update that firms using AI to deliver worse outcomes for some consumer groups "might not be acting in good faith for their consumers, unless differences in outcome can be justified objectively." That phrase — *unless differences in outcome can be justified objectively* — is doing a lot of work. Actuarial justification is a defence, but it has to be available, documented, and current.

The specific concern with ML pricing models is proxy discrimination: a model trained on postcode, vehicle type, occupation, and telematics can reconstruct protected characteristics like ethnicity or socioeconomic status with enough fidelity that the actuarial justification for individual rating factors does not cover the aggregate discriminatory effect. The Equality Act 2010, Section 19 (indirect discrimination) is not switched off by the fact that the model never sees gender or race as an input.

The FCA's reference to firms needing to "evidence the fairness of their algorithms" — language that appears in supervisory correspondence from 2025 — points to what is expected: not an assertion that you checked for bias, but documentation of what you tested, when, what you found, and what action you took.

### 3. Model governance

SMCR means that someone is personally accountable for every material AI system. A Senior Manager signing off a pricing model that the firm cannot explain, cannot monitor, and cannot correct is exposed. The FCA does not need to create new AI liability rules; SMCR already provides the mechanism.

PRA SS1/23 — the model risk management supervisory statement, published January 2023 — established the governance expectations for model risk in regulated UK firms: model inventory, tiered oversight based on materiality, independent validation, documented assumptions, monitoring plans, and audit trails. SS1/23 applies to UK banks directly. For insurers, the equivalent obligation comes from Solvency II Article 121 and PRA SS3/18; for FCA-solo firms, SS1/23 is the effective standard of practice because it is what the FCA will reach for when assessing whether model governance is adequate.

The FCA's 2026 evaluation will be asking, in effect: does your governance for AI pricing models meet the standard that SS1/23 sets out for models generally? Most pricing teams have not applied that rigour.

---

## What you should be able to produce

The FCA evaluation is not going to ask for code. It will ask for evidence — documents, reports, decisions and the reasoning behind them. If those documents do not exist, the answer is not to produce them retrospectively; it is to build the processes that generate them prospectively.

Here is the minimum set:

**A model inventory.** Every production pricing model, with its scope, GWP impact, customer-facing status, and current sign-off state. The `MRMModelCard` in [`insurance-governance`](/insurance-governance/) structures this:

```python
from insurance_governance import MRMModelCard, Assumption, RiskTierScorer, ModelInventory

card = MRMModelCard(
    model_id="motor-freq-v5",
    model_name="Motor Frequency v5",
    version="5.0.0",
    model_class="pricing",
    model_type="CatBoost",
    distribution_family="Poisson",
    target_variable="claim_count",
    intended_use="Predict claim frequency for UK private motor, used directly in premium calculation",
    not_intended_for=["commercial vehicles", "fleet pricing", "any segment outside England/Wales/Scotland"],
    gwp_impacted=92_000_000,
    customer_facing=True,
    regulatory_use=False,
    assumptions=[
        Assumption(
            description="Training data 2021-2024 is representative of current risk mix",
            risk="HIGH",
            mitigation="Monthly PSI on key rating factors; refit trigger at PSI > 0.20",
        ),
        Assumption(
            description="Postcode district used as a proxy for local risk, not socioeconomic group",
            risk="MEDIUM",
            mitigation="Annual proxy discrimination audit; correlation with IMD decile monitored",
        ),
    ],
    monitoring_owner="Lead Pricing Actuary",
    monitoring_frequency="Quarterly",
    monitoring_triggers={"psi_score": 0.20, "ae_ratio_high": 1.05, "ae_ratio_low": 0.95},
)

scorer = RiskTierScorer()
tier = scorer.score(card)

inventory = ModelInventory("mrm_registry.json")
inventory.register(card, tier)
```

The inventory is a plain JSON file that can live in version control. The audit trail is the git log.

**A current validation report.** For every Tier 1 model (high GWP, customer-facing), you need a validation report that a non-developer can read and understand. `insurance-governance` generates HTML reports aligned to PRA SS1/23:

```python
from insurance_governance import ModelValidationReport, ValidationModelCard

val_card = ValidationModelCard(
    name="Motor Frequency v5",
    version="5.0.0",
    purpose="Claim frequency prediction for UK private motor pricing",
    methodology="CatBoost gradient boosting, Poisson objective",
    target="claim_count",
    features=["driver_age", "vehicle_age", "ncd_years", "vehicle_group", "postcode_district"],
    limitations=["No telematics data", "Limited NCD history pre-2019"],
    owner="Pricing Team",
)

report = ModelValidationReport(
    model_card=val_card,
    y_val=y_val,
    y_pred_val=y_pred_val,
    exposure_val=exposure_val,
    y_train=y_train,
    y_pred_train=y_pred_train,
    fairness_group_col="gender",
    monitoring_owner="Lead Pricing Actuary",
    monitoring_triggers={"psi_score": 0.20, "ae_ratio_high": 1.05, "ae_ratio_low": 0.95},
)
report.generate("motor_freq_v5_validation.html")
report.to_json("motor_freq_v5_validation.json")
```

The JSON sidecar matters: it feeds directly into the governance pack without manual transcription.

**A fairness audit.** Run for every customer-facing model at least annually, and after any material refit. The `FairnessAudit` in `insurance-fairness` tests for disparate impact, proxy discrimination, and calibration by group:

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=catboost_model,
    data=df,
    protected_cols=["gender"],
    prediction_col="predicted_frequency",
    outcome_col="claim_count",
    exposure_col="exposure",
    factor_cols=["driver_age", "vehicle_age", "ncd_years", "vehicle_group", "postcode_district"],
    model_name="Motor Frequency v5",
)
report = audit.run()
report.to_markdown("motor_freq_v5_fairness_audit.md")
```

The output includes disparate impact ratios, proxy R-squared scores, and calibration by group. This is the evidence the FCA means when it says firms need to "evidence the fairness of their algorithms." Our post on [proxy discrimination testing](/2026/03/28/does-proxy-discrimination-testing-actually-work/) covers why Spearman correlation is not sufficient — mutual information and SHAP-based proxy scores catch the cases that matter.

**A monitoring report with a documented recommendation.** Quarterly, at a minimum. The recommendation must come from a systematic process, not an actuary's judgement applied post-hoc:

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=train_claims,
    reference_predicted=train_predicted,
    current_actual=current_claims,
    current_predicted=current_predicted,
    exposure=current_exposure,
    feature_df_reference=train_features,
    feature_df_current=current_features,
    features=["driver_age", "vehicle_age", "ncd_years", "vehicle_group", "postcode_district"],
    murphy_distribution="poisson",
    gini_bootstrap=True,
)
print(report.recommendation)   # "CONTINUE" | "RECALIBRATE" | "REFIT"
```

The three-way recommendation — built on Murphy decomposition distinguishing calibration drift from discrimination drift — is what a supervisor reviewing your monitoring record wants to see. Not "the team reviewed the A/E and decided no action was required." A documented, reproducible decision with an audit trail. Our post on [automated model monitoring](/2026/03/27/does-automated-model-monitoring-actually-work/) covers how the MonitoringReport drives this.

**A governance pack for each Tier 1 model.** This is the document that goes to your Model Risk Committee, and the document the FCA would expect to see if they asked for evidence of SMCR accountability:

```python
from insurance_governance import GovernanceReport

gov_report = GovernanceReport(
    card=card,
    tier=tier,
    validation_results={
        "overall_rag": "GREEN",
        "run_id": "2026-Q1-val",
        "run_date": "2026-01-15",
        "gini": 0.43,
        "ae_ratio": 1.01,
        "psi_score": 0.06,
    },
    monitoring_results={
        "period": "2026-Q1",
        "ae_ratio": 1.03,
        "psi_score": 0.08,
        "recommendation": "Continue",
    },
)
gov_report.save_html("motor_freq_v5_governance_pack.html")
```

The HTML output is designed for MRC consumption. It tells the committee what they are approving and what action is required, without requiring them to understand the PSI calculation.

---

## The audit trail the FCA expects

"Audit trail" is not a metaphor. The FCA expects to be able to reconstruct, for any production pricing model, the sequence of: development decision → validation → sign-off → deployment → monitoring → action. If a model has been in production for two years and the only documentation is a model card last updated at launch, that is not an audit trail.

The `ModelInventory.log_event()` method is the mechanism for keeping this current:

```python
inventory.log_event(
    model_id="motor-freq-v5",
    event_type="monitoring_trigger",
    description="PSI 0.22 on postcode_district in Q2 monitoring. A/E 1.06, within tolerance. REFIT triggered.",
    triggered_by="QuarterlyMonitoringReport 2026-Q2",
)

inventory.log_event(
    model_id="motor-freq-v5",
    event_type="status_change",
    description="Champion status granted following MRC sign-off. Previous champion motor-freq-v4 retired.",
    triggered_by="MRC minutes ref MRC-2026-031",
)
```

Every event is timestamped. Every event references the process that triggered it. When a supervisor asks "what happened between Q1 and Q3 of this year with your motor frequency model?" — you hand them the event log.

---

## How this differs from the EU AI Act

The [EU AI Act](/2026/03/25/eu-ai-act-august-2026-insurance-pricing/) imposes mandatory conformity assessments, registration in the EU AI database, and detailed Annex IV technical documentation before a high-risk AI system can be deployed. Non-compliance carries penalties of up to €35m or 7% of global revenue.

The FCA's approach is different in kind. No mandatory pre-deployment assessment, no registration requirement, no AI-specific penalty regime. Instead: existing frameworks applied with increasing supervisory sophistication, an evaluation process designed to build the evidence base for future guidance, and the implicit expectation that firms found to have inadequate governance will face scrutiny under Consumer Duty and SMCR.

Which of these is harder to manage in practice is genuinely debatable. The EU AI Act at least tells you what "compliant" looks like. The FCA's approach requires you to anticipate what a reasonable supervisor with deep knowledge of pricing models would consider adequate — and to document your way there before they ask.

The good news is that the technical answer is the same under both regimes. Model cards, validation reports, fairness audits, monitoring with documented recommendations, governance packs for MRC. A team that builds these processes is compliant with both, and positioned to answer questions from either regulator without scrambling.

---

## Where to start

If you do not have a model inventory — a complete, current register of every production pricing model with its risk tier and sign-off status — that is the first thing to build. Nothing else is possible without it. Run `RiskTierScorer` to establish which models are Tier 1 and need immediate attention.

Once the inventory exists, prioritise validation and fairness audits for Tier 1 models — those with GWP above £50m or any customer-facing model with no documented independent validation in the past twelve months. The FCA's 2026 evaluation is not going to ask about your Tier 3 reserve models.

The monitoring framework should run quarterly without manual intervention. The recommendation — Continue, Recalibrate, Refit — should be machine-generated and then reviewed by a named individual who is recorded as having reviewed it. That record is the Senior Manager accountability trail.

Start now. The FCA's evaluation is already underway.

---

*The libraries referenced in this post — [`insurance-governance`](https://github.com/burning-cost/insurance-governance), [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring), and [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) — are open source and available on PyPI.*
