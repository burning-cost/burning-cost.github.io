---
layout: post
title: "EU AI Act August 2026: What Your Pricing Team Must Do"
date: 2026-03-25
categories: [regulation]
tags: [eu-ai-act, regulation, model-governance, mrm, compliance, pra-ss123, insurance-governance, insurance-monitoring, insurance-fairness, annex-iii, high-risk-ai]
description: "Insurance pricing models are high-risk AI under Annex III of the EU AI Act. The compliance deadline is 2 August 2026 — five months away. Here is what that means in practice for a UK pricing team."
---

The EU AI Act's full enforcement regime for high-risk AI systems lands on 2 August 2026. For most UK insurers, that is five months away. We will be blunt: the majority of pricing teams are not ready, and the compliance gap is not a documentation problem — it is a structural one. The requirements in Articles 9 through 15 assume a level of systematic model governance that most pricing functions have never had to build.

This post sets out what the Act requires, which requirements map onto existing tooling, and what you actually need to do before August.

---

## Why insurance pricing is high-risk AI

Annex III of the Act classifies AI systems used "for risk assessment and pricing in relation to natural persons in the case of life and health insurance" as high-risk. The broader catch is that general insurance pricing — motor, home, commercial — falls under the "essential services" category in Annex III point 5(b): systems used to "evaluate the creditworthiness of natural persons or establish their credit score" and, critically, systems used for insurance pricing.

The FCA has confirmed that pricing AI in scope of Annex III will be treated as high-risk. The penalty for non-compliance is up to €35 million or 7% of global annual revenue — whichever is higher.

One note on timing: the European Commission's Digital Omnibus proposal from late 2025 floated a possible extension to December 2027 for some Annex III obligations. We would not plan around that. Nothing has been enacted, and prudent compliance planning means treating 2 August 2026 as binding.

---

## The eight requirements, and what each one actually means for pricing

### Article 9: Risk management system

The Act requires a continuous, documented risk management process covering identification of known and foreseeable risks, risk estimation and evaluation, and mitigation measures. "Continuous" means it cannot be a one-off document produced at model launch and never updated.

In practice, this means maintaining a live model card for every production pricing model, with documented assumptions, risk ratings, and outstanding issues. The card must be updated when the model's risk profile changes — not just annually.

`insurance-governance` implements this directly:

```python
from insurance_governance import MRMModelCard, Assumption, RiskTierScorer, ModelInventory

card = MRMModelCard(
    model_id="motor-freq-v4",
    model_name="Motor Frequency v4",
    version="4.0.0",
    model_class="pricing",
    model_type="CatBoost",
    distribution_family="Poisson",
    target_variable="claim_count",
    intended_use="Predict claim frequency for UK private motor portfolio",
    not_intended_for=["commercial vehicles", "fleet", "telematics-only segments"],
    gwp_impacted=85_000_000,
    customer_facing=True,
    regulatory_use=True,
    assumptions=[
        Assumption(
            description="Training data 2021-2024 representative of current risk mix",
            risk="HIGH",
            mitigation="Monthly PSI monitoring on key rating factors; refit trigger at PSI > 0.20",
        ),
        Assumption(
            description="Postcode districts used as proxy for local risk, not socioeconomic group",
            risk="MEDIUM",
            mitigation="Annual proxy discrimination audit using insurance-fairness",
        ),
    ],
)

scorer = RiskTierScorer()
tier = scorer.score(card)  # Returns TierResult with tier 1/2/3 and rationale

inventory = ModelInventory("mrm_registry.json")
inventory.register(card, tier)
```

Article 9 also requires documented risk mitigations. Listing assumptions without risk ratings and mitigations is compliance theatre — the Act's auditors will not accept it.

### Article 10: Data governance

Training, validation, and testing datasets must have documented governance covering collection methodology, preparation steps, and known limitations. The Act specifies that datasets should be "free of errors and complete" for the intended purpose, with documented handling of any exceptions.

For pricing models, this means documenting why your training period was chosen, what exclusions were applied (e.g. large losses above the retention, incomplete policy years), and what known biases exist in the data. Specifically, Article 10 requires examination of "possible biases" that could affect high-risk groups.

This is where the fairness audit connects to data governance. The audit evidence you produce under Article 10 is the same evidence you need for FCA Consumer Duty monitoring:

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=catboost_model,
    data=df,
    protected_cols=["gender"],
    prediction_col="predicted_frequency",
    outcome_col="claim_count",
    exposure_col="exposure",
    factor_cols=["age_band", "vehicle_group", "postcode_district", "ncd_years"],
    model_name="Motor Frequency v4",
)
report = audit.run()
report.to_markdown("article10_data_governance_audit.md")
```

Run this as part of model development, not as a sign-off checkbox. The output — disparate impact ratios, proxy R-squared scores, counterfactual premium impacts — becomes the Article 10 evidence record. Our earlier post on [proxy discrimination testing](/2026/03/28/does-proxy-discrimination-testing-actually-work/) covers why manual Spearman correlation misses the cases that matter.

### Article 11: Technical documentation

Annex IV specifies the exact contents of the technical documentation required before a high-risk AI system can be placed in operation. This includes: general description, design specifications, training methodology, validation results, intended purpose, monitoring plan, and changes made post-deployment.

The good news: a complete validation report covering Gini, A/E, PSI, Hosmer-Lemeshow, and discrimination checks already covers most of Annex IV. The `insurance-governance` validation report aligns with PRA SS1/23 and generates both HTML and JSON outputs that can feed directly into documentation systems:

```python
from insurance_governance import ModelValidationReport, ValidationModelCard

val_card = ValidationModelCard(
    name="Motor Frequency v4",
    version="4.0.0",
    purpose="Predict claim frequency for UK private motor, Annex IV documentation",
    methodology="CatBoost gradient boosting, Poisson objective, SHAP feature selection",
    target="claim_count",
    features=["driver_age", "vehicle_age", "ncd_years", "vehicle_group", "postcode_district"],
    limitations=["No telematics data in training", "Limited NCD history for new drivers"],
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
    monitoring_owner="Pricing Analytics",
    monitoring_triggers={"psi_score": 0.10, "ae_ratio_high": 1.05, "ae_ratio_low": 0.95},
)
report.generate("motor_freq_v4_annex_iv.html")
report.to_json("motor_freq_v4_annex_iv.json")
```

The JSON sidecar matters: it allows the documentation to be ingested by governance portals without manual re-entry.

### Article 12: Record-keeping and logging

High-risk AI systems must automatically log events throughout their operational lifetime, with logs retained for the period specified by national authority (the FCA has not yet published guidance, but PRA SS1/23's five-year expectation is the working assumption). Logs must be sufficient to enable tracing of the system's operation back through each prediction cycle.

For pricing, this means recording every model version change, every production score run, every monitoring alert, and every consequent action. The `ModelInventory.log_event()` method is the mechanism:

```python
inventory.log_event(
    model_id="motor-freq-v4",
    event_type="monitoring_trigger",
    description="PSI score 0.127 on vehicle_age; exceeds 0.10 threshold. A/E 1.03 (within tolerance).",
    triggered_by="MonitoringReport 2026-Q1",
)

inventory.log_event(
    model_id="motor-freq-v4",
    event_type="status_change",
    description="Model moved to champion following MRC sign-off 2026-02-14.",
    triggered_by="MRC minutes reference MRC-2026-014",
)
```

The registry writes to a plain JSON file that can live in version control. Every change is a commit. The audit trail is the git log.

### Article 13: Transparency to deployers

Insurers deploying high-risk AI must provide deployers (underwriters, brokers, internal users) with documentation enabling them to understand the system's capabilities, limitations, and conditions of use. This is distinct from the technical documentation in Article 11 — it is the operational guide for humans who use the system's outputs.

For a pricing model, this means a governance report that a non-technical committee member can read and understand:

```python
from insurance_governance import GovernanceReport

gov_report = GovernanceReport(
    card=card,
    tier=tier,
    validation_results={
        "overall_rag": "GREEN",
        "run_id": "abc-123",
        "run_date": "2026-02-01",
        "gini": 0.41,
        "ae_ratio": 1.02,
        "psi_score": 0.07,
    },
    monitoring_results={
        "period": "2026-Q1",
        "ae_ratio": 1.03,
        "recommendation": "Continue",
    },
)
gov_report.save_html("motor_freq_v4_governance_pack.html")
```

The HTML output is print-to-PDF ready and designed for Model Risk Committee consumption. The recommendations section tells the committee what action is needed, so the meeting does not become a discussion of what the numbers mean.

### Article 14: Human oversight

The Act requires that high-risk AI systems are designed to allow human oversight by natural persons. For a pricing system, this means three things in practice: (1) humans must be able to understand the output to the degree needed to verify it, (2) humans must be able to intervene and override the system, and (3) persons designated to oversee the system must have the competence, authority, and training to do so.

This is partly a process requirement, not a tooling one. Pricing teams must document who has oversight responsibility, what authority they have, and what the escalation path is when the model is flagged. The `ModelCard.monitoring_owner` and `ModelCard.approved_by` fields make this explicit in the governance record. If those fields are blank, you have an Article 14 gap.

### Article 15: Accuracy, robustness, and cybersecurity

High-risk AI systems must be designed to achieve an appropriate level of accuracy, robustness, and cybersecurity throughout their lifecycle. Accuracy must be declared in the technical documentation, and systems must be resilient to errors and inconsistencies.

For pricing, robustness testing means demonstrating that the model's performance degrades gracefully under data quality issues — missing values, outlier inputs, distribution shift. It also means having a documented monitoring regime that detects degradation before it becomes material.

The `MonitoringReport` runs this in one pass:

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
    features=["driver_age", "vehicle_age", "ncd_years", "vehicle_group"],
    murphy_distribution="poisson",
    gini_bootstrap=True,
)
print(report.recommendation)  # "CONTINUE" | "RECALIBRATE" | "REFIT"
print(report.to_polars())     # Write to Delta table for audit trail
```

The three-way recommendation — Continue / Recalibrate / Refit — maps cleanly to Article 15's requirement for documented responses to accuracy degradation. Our post on [when to recalibrate versus refit](/2026/02/28/recalibrate-or-refit/) covers the Murphy decomposition logic that drives this decision.

---

## Two requirements with no off-the-shelf answer

**Conformity assessment** (Article 43): Before deployment, high-risk AI systems require a conformity assessment confirming they meet the Act's requirements. For most Annex III systems, this is a self-assessment (an internal audit process), not a third-party certification. But it must be documented in the technical documentation and the declaration of conformity signed off by a responsible person.

**EU database registration** (Article 71): Providers of high-risk AI systems must register their systems in the EU AI database before placing them on the market or putting them into service. The database is operated by the European AI Office. The registration form asks for the information in your technical documentation — if you have that in order, registration is administrative, not technical. The registration portal opened in Q4 2025; FCA guidance on the UK equivalent is expected mid-2026.

---

## The five-month plan

If your team is starting from scratch, here is the sequence that makes sense:

**Months 1–2 (April–May)**: Build the model inventory. Every production model gets an `MRMModelCard` with real fields — no placeholders. Run `RiskTierScorer` to assign tiers. Any Tier 1 model (GWP > £50m or customer-facing with no recent validation) goes to the top of the queue.

**Month 3 (June)**: Run validation reports for all Tier 1 models. The output is your Article 11 technical documentation. Run `FairnessAudit` to generate Article 10 data governance evidence. Fill the Article 12 logging gap by instrumenting monitoring pipelines to write to the inventory.

**Month 4 (July)**: Governance report for each Tier 1 model. MRC sign-off with Article 13 transparency pack. Conformity self-assessment against the Article 9–15 checklist. Register in the EU database.

**Month 5 (first days of August)**: You are in scope from day one. Models without documentation are operating illegally under the Act.

---

## What the Act does not require

The Act does not require that your models are accurate by some absolute standard. It requires that you declare their accuracy and monitor it. A model with a Gini of 0.28 is compliant if you have documented that, validated it, and put it in the transparency pack. A model with a Gini of 0.52 is non-compliant if there is no technical documentation.

It does not require explainability at the individual prediction level. Article 13 requires transparency to deployers, not to policyholders (Article 86 creates individual rights to explanation, but that is a separate obligation from the Article 13 deployer transparency requirement).

It does not require model-free pricing. The Act specifically accommodates risk-based pricing where it is actuarially justified and non-discriminatory — that is the same standard the FCA has applied since the Gender Directive.

---

The compliance burden is real, but it is not a new kind of burden for insurers who already comply with PRA SS1/23. It is the same governance discipline — model cards, validation reports, monitoring plans, MRC sign-off — made mandatory by law. Teams that have built these practices already are finishing a project. Teams that have not are starting one.

Five months is enough time. But only just.

---

*The libraries referenced in this post — [`insurance-governance`](https://github.com/burning-cost/insurance-governance), [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring), and [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) — are open source and available on PyPI.*
