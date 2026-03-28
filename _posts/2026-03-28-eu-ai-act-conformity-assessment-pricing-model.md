---
layout: post
title: "Building an EU AI Act Conformity Assessment for Your Pricing Model"
date: 2026-03-28
categories: [regulation, model-governance]
tags: [eu-ai-act, conformity-assessment, article-11, article-14, article-15, technical-documentation, model-governance, insurance-governance, pra-ss123, consumer-duty]
description: "A step-by-step guide to building an EU AI Act conformity assessment for an insurance pricing model. Covers risk classification, Article 11 technical documentation, Article 14 human oversight, and Article 15 accuracy requirements — with Python code using insurance-governance."
---

In [the previous post](https://burning-cost.github.io/regulation/2026/03/28/eu-ai-act-insurance-pricing-what-you-need-to-know/) we established the scope: life and health individual pricing is high-risk AI under Annex III paragraph 5(c); motor and property pricing is not. Self-assessment only — no notified body. The deadline is 2 August 2026.

This post is the practical follow-up: what does a conformity assessment for a pricing model actually contain, how do you build it, and where does the insurance-governance library already do most of the work?

We are going to walk through the Article 43 Annex VI self-assessment procedure step by step. The structure follows the Act itself: risk classification, quality management system, Annex IV technical documentation, Articles 9 through 15, post-market monitoring. We will be honest about what is settled text and what is still being clarified.

---

## Why build one even if you are probably not in scope

Motor and property pricing teams should still build conformity-adjacent documentation. Three reasons.

**Regulatory direction of travel.** The EU AI Act is the first iteration. The Commission is already working on harmonised standards for the insurance sector that may extend Annex III. The EIOPA Opinion (EIOPA-BoS-25-360, August 2025) signals that EU supervisors expect equivalent governance for all AI systems in insurance, not just high-risk ones. Building documentation now is cheaper than retrofitting it under a tighter deadline.

**PRA SS1/23 overlap is substantial.** The PRA's model risk management supervisory statement requires documentation of intended purpose, limitations, validation methodology, and ongoing monitoring. That is Articles 9, 11, 13, and 15 of the EU AI Act described using different words. If you are doing SS1/23 governance properly, a conformity assessment takes a few days of formatting work, not a project.

**Consumer Duty.** The FCA expects firms to be able to demonstrate that pricing models produce fair outcomes. The Article 14 human oversight design and the Article 13 transparency documentation are evidence of that capability. A conformity assessment gives you a single artefact that satisfies multiple regulators simultaneously.

---

## Step 1: Risk classification

Before you write a word of technical documentation, establish what you are actually assessing. The classification question has three layers.

**Layer 1: Is it an AI system?** The Commission Guidelines (C/2025/3554, 6 February 2025) clarify that traditional GLMs using fixed coefficients, with no inference or learning capability, may fall outside the AI system definition entirely. If your pricing model is a Poisson GLM with fixed coefficients re-fitted quarterly, the argument that it is not an AI system is credible. We are not saying it definitively is not — that depends on your specific architecture and how you re-fit — but it is a real question worth documenting.

For GBMs, neural networks, and any model using adaptive learning, there is no credible argument that it falls outside the definition.

**Layer 2: Does the Act apply to you?** EU AI Act Article 2 applies to providers placing AI on the EU market, and to providers or deployers in third countries whose AI output is used in the EU. Pure UK domestic insurers with no EU-domiciled policyholders are not caught. Insurers with EEA subsidiaries, Lloyd's Brussels operations, or cross-border policies for EU residents are in scope for those business flows. Document this clearly: which models, applied to which customer segments, are in scope.

**Layer 3: Is it high-risk?** For insurance pricing, the operative text is Annex III paragraph 5(c): AI systems for 'risk assessment and pricing in relation to natural persons in the case of life and health insurance.' Motor, property, commercial lines, and entity-level pricing are not in this text. Life, health, PMI, critical illness, income protection, and individual annuity pricing are.

There is an Article 6(3) escape for systems that do not pose a significant risk of harm to health, safety, or fundamental rights. We would not rely on this for life pricing without a documented risk assessment to back it up. The burden of proof is on the provider.

Produce a one-page classification memo with your conclusion and the evidence. It becomes part of your technical documentation.

---

## Step 2: Quality Management System (Article 17)

The conformity assessment under Annex VI is essentially a check that your Article 17 quality management system (QMS) exists and covers the right ground. The QMS is the governance framework; the conformity assessment is the audit of it.

Article 17 requires your QMS to cover:

| QMS Element | What it means in practice |
|---|---|
| Risk management policy | Documented process for identifying, assessing, and mitigating risks in the AI system lifecycle |
| Data governance | How training data is sourced, quality-checked, and documented |
| Technical documentation | The Annex IV document pack (see Step 3) |
| Record-keeping | Automated logging of model outputs (Article 12) |
| Transparency | The Article 13 deployer-facing document |
| Human oversight measures | How underwriters interact with and override the model (Article 14) |
| Accuracy monitoring | Performance metrics, monitoring cadence, drift alerts (Article 15) |
| Post-market monitoring | Ongoing review plan per Article 72 |

Most insurers doing SS1/23 governance have most of this. The gaps are usually the Article 12 logging (automated operational logs are a specific EU AI Act requirement, not just general model monitoring) and the formal Article 13 document (which is a deployer-facing instructions document, not just internal technical documentation).

---

## Step 3: Annex IV technical documentation

This is the core document pack. Annex IV of the Act specifies what it must contain. We have mapped each requirement to what a pricing team already produces — and flagged what is missing.

### General description of the system

Required: intended purpose, version, hardware/software context, deployment form.

The ModelCard in insurance-governance covers this. Here is the minimum to populate it:

```python
from insurance_governance.mrm import ModelCard

card = ModelCard(
    model_id="life-freq-v2.3",
    model_name="Individual Life Frequency Model",
    model_class="pricing",
    version="2.3.0",
    purpose=(
        "Estimate claim frequency for individual life insurance new business. "
        "Used for premium rating only. Not for portfolio monitoring or reserving."
    ),
    methodology="CatBoost gradient boosting, Poisson objective, log offset for exposure",
    target_variable="claim_indicator",
    feature_list=[
        "age", "gender", "smoker_status", "bmi_band",
        "sum_assured_band", "product_type",
    ],
    limitations=[
        "Performance degrades for applicants over 80 (thin data: <0.3% of training set)",
        "Does not reflect impaired lives — separate manual underwriting process applies",
        "Training data covers 2015-2024; COVID-period mortality not reweighted",
    ],
    owner="Pricing Team",
    development_date=date(2025, 9, 1),
    approved_by=["Chief Actuary", "Model Risk Committee"],
)
```

The `limitations` field is doing real work here. Annex IV requires documentation of circumstances that may affect accuracy. Listing limitations without specificity — 'model performance may vary' — does not satisfy this. The limitations above are specific enough to be actionable for an underwriter.

### Data governance documentation

Annex IV requires documentation of training data: source, preparation steps, quality checks, representativeness assessment, and bias testing. Article 10 of the Act elaborates: training data must be 'relevant, sufficiently representative and, to the best extent possible, free of errors.'

The Act is silent on what 'sufficiently representative' means quantitatively. The Commission is commissioning harmonised standards that will eventually define this. Until those standards appear — which is not before August 2026 — you document what you have done and why you believe it is adequate.

What a data governance section should cover:

```python
# Document your training data in the card
card.training_data_source = (
    "Internal new business data, 2015-2024, UK individual life policies. "
    "1.2m policy-years, 47k claims. Sourced from policy admin system; "
    "extract date 2024-10-01. No external data."
)
card.data_quality_checks = [
    "Duplicate policy IDs removed (0.03% of records)",
    "Claims with reported date > 12 months post-inception excluded (potential misreport)",
    "BMI outliers (<10 or >70) set to null and imputed with median by age/gender band",
    "Population coverage: all UK regions represented; Isle of Man and Channel Islands excluded",
]
card.bias_assessment = (
    "Disparate impact tested against gender and age groups (5-year bands). "
    "Gini coefficient gap: female vs male = 0.02 (within 0.05 tolerance). "
    "Age band 75-85: Gini 0.21 vs overall 0.38 — noted in Article 13 transparency doc. "
    "No protected characteristic used directly as a rating factor."
)
```

### Technical architecture documentation

Required: description of computational elements, their interaction, and software dependencies. For a GBM pricing model, this includes: the model framework and version (CatBoost 1.2.x), hyperparameter configuration, training pipeline steps, the feature engineering logic, and how the model output is translated into a premium.

Do not write this from memory. Extract it from code:

```python
import json, catboost

model = catboost.CatBoostClassifier()
model.load_model("life_freq_v2_3.cbm")

tech_spec = {
    "framework": f"CatBoost {catboost.__version__}",
    "iterations": model.get_param("iterations"),
    "depth": model.get_param("depth"),
    "learning_rate": model.get_param("learning_rate"),
    "loss_function": model.get_param("loss_function"),
    "feature_count": model.feature_count_,
    "feature_names": model.feature_names_,
}

with open("annex_iv_tech_spec.json", "w") as f:
    json.dump(tech_spec, f, indent=2, default=str)
```

This JSON file becomes an appendix to your technical documentation. It is version-controlled alongside the model artefact. When you re-train, you re-generate it. Annex IV requires documentation of 'pre-determined changes to the system' — your training pipeline is exactly this.

---

## Step 4: Article 9 risk management

Article 9 requires a continuous risk management system throughout the model's lifecycle. For a pricing model, the risk management documentation covers:

**Known risks**: adversarial input (data manipulation by applicants), concept drift (mortality rates changing due to new treatments or epidemics), population shift (book mix changes due to distribution strategy), model misuse (applying the model outside its intended scope — e.g., to group business).

**Risk assessments for each**: likelihood, severity, and mitigation. The Act requires these to be 'reasonable in light of the generally acknowledged state of the art.' That phrase has no regulatory definition yet.

**Residual risk**: after mitigations, what risk remains and why it is acceptable.

For the audit trail, produce a risk register as a structured document:

```python
from insurance_governance.mrm import Assumption

risks = [
    Assumption(
        description="Concept drift: COVID-period mortality patterns persist in training data",
        risk="HIGH",
        mitigation=(
            "2020-2022 data downweighted by 0.4 in training. "
            "Quarterly A/E monitoring with amber alert at 1.10 and red at 1.20. "
            "Annual full retraining cycle."
        ),
        rationale="COVID excess mortality materially distorts frequency estimates for 50-70 age band",
    ),
    Assumption(
        description="Adversarial input: BMI under-reporting by applicants",
        risk="MEDIUM",
        mitigation=(
            "BMI outlier detection removes extreme values. "
            "A/E monitoring by BMI band to detect systematic misreport. "
            "No algorithmic detection — underwriting remains responsible for verification."
        ),
        rationale="Under-reporting is endemic in self-declared BMI; model cannot fully correct for it",
    ),
    Assumption(
        description="Model misuse: application to group life business",
        risk="HIGH",
        mitigation=(
            "Intended purpose documented in model card and Article 13 document. "
            "Technical access control: model API requires policy_type='individual' parameter. "
            "Annual scope audit by model owner."
        ),
        rationale="Group pricing requires different feature set and validation approach",
    ),
]
```

---

## Step 5: Article 14 human oversight

Article 14 is one of the more practically demanding requirements. The text is clear: the system must be designed so that a person can understand its capabilities and limitations, detect anomalies, interpret its outputs, and — critically — decide not to use it, or override it, in any particular situation.

It also specifically names automation bias: the documentation must show how you address the tendency to over-rely on model outputs.

For a life pricing model, the minimum requirements are:

**Underwriter training documentation.** Not just product training — specific training on what the model does not capture. The 'age 80+' and 'impaired lives' limitations in the model card are the starting point. This training should be documented and periodically refreshed.

**Override capability.** The system must technically allow an underwriter to reject the model output and substitute manual assessment. For STP (straight-through processing) systems, this means a defined referral route. Document the referral trigger conditions and what happens when they are met.

**Stop capability.** The Act requires an ability to 'interrupt the system through a stop button or similar procedure.' For a pricing model, this means the ability to suspend automated pricing (e.g., switch to manual rating tables) if the model produces anomalous outputs at scale. This should be a documented operational procedure, not just a theoretical option.

**Anomaly detection and alerting.** Implement real-time or near-real-time checks on output distributions:

```python
import numpy as np

def check_output_anomalies(
    premiums: np.ndarray,
    expected_mean: float,
    expected_std: float,
    tolerance_sigma: float = 3.0,
) -> dict:
    """
    Article 14 compliance check: flag batch outputs that fall outside
    expected range. Called in the pricing engine after each batch.
    """
    batch_mean = premiums.mean()
    batch_std = premiums.std()
    z_mean = abs(batch_mean - expected_mean) / expected_std

    anomalies = {
        "batch_size": len(premiums),
        "batch_mean": round(float(batch_mean), 2),
        "batch_std": round(float(batch_std), 2),
        "z_score_mean": round(float(z_mean), 2),
        "alert": z_mean > tolerance_sigma,
        "action_required": (
            "SUSPEND AUTOMATED PRICING — escalate to model owner"
            if z_mean > tolerance_sigma else None
        ),
    }
    return anomalies
```

The alert output goes to your model monitoring dashboard and triggers an incident record. The incident record is what you show to a regulator.

**Automation bias mitigation.** The Act names it; you need to address it. The most practical approach for a pricing model is to present model outputs alongside confidence intervals and known limitation flags, rather than just a premium figure. An underwriter who sees a premium accompanied by 'Limited data: age group 82, confidence ±18%' behaves differently from one who sees only the number.

---

## Step 6: Article 15 accuracy, robustness, and cybersecurity

Article 15 requires 'an appropriate level of accuracy, robustness and cybersecurity.' The word 'appropriate' is doing enormous work in that sentence. The Act provides no sector-specific quantitative thresholds. This is one of the significant open questions in the current regulatory landscape — the harmonised standards that would define 'appropriate' for insurance are not expected before the August 2026 deadline.

Our view: define your own thresholds, document the methodology for setting them, and demonstrate that your monitoring detects deviations. A regulator reviewing your conformity assessment is not going to reject your Gini threshold of 0.30 if you can show how you derived it from historical data. They will reject it if you cannot explain it at all.

The insurance-governance library handles the monitoring side. The `ModelValidationReport` generates the evidence pack you attach to your conformity assessment:

```python
from insurance_governance.validation import (
    ModelCard as ValidationModelCard,
    ModelValidationReport,
)
from datetime import date
import numpy as np

val_card = ValidationModelCard(
    name="Individual Life Frequency v2.3",
    version="2.3.0",
    purpose="EU AI Act Article 15 accuracy assessment — annual validation",
    methodology="CatBoost Poisson",
    model_type="GBM",
    target="claim_indicator",
    features=["age", "gender", "smoker_status", "bmi_band", "sum_assured_band"],
    limitations=[
        "Age 80+ thin data",
        "Impaired lives excluded",
    ],
    owner="Pricing Team",
    development_date=date(2025, 9, 1),
    validation_date=date(2026, 3, 1),
    validator_name="Independent Validation Team",
    monitoring_owner="Pricing Team",
)

report = ModelValidationReport(
    card=val_card,
    y_actual=y_test,
    y_predicted=y_pred,
    exposure=exposure_test,
    thresholds={
        "gini_green": 0.30,      # below overall 0.38; triggers review
        "gini_amber": 0.25,      # materially degraded; escalation required
        "ae_green": (0.92, 1.08),
        "ae_amber": (0.85, 1.15),
        "psi_green": 0.10,
        "psi_amber": 0.20,
    },
)

report.run()
report.to_html("article_15_validation_2026_03.html")
```

The HTML report output maps to Article 15 requirements: accuracy metrics with confidence intervals, stability assessment (PSI), calibration check (A/E), and discrimination assessment (Gini). Keep this report in your technical documentation pack. When you re-run at the annual validation, the new report replaces or supplements it.

**Robustness** in the Act means performance under conditions of foreseeable misuse and adversarial inputs. For a pricing model, this is stress testing: what happens to outputs if input data has systematic errors? Document the stress test results:

```python
# Stress test: all BMI values shifted +2 (underreporting scenario)
stressed_inputs = X_test.copy()
stressed_inputs["bmi_band"] = (stressed_inputs["bmi_band"] + 1).clip(upper=5)
y_stressed = model.predict(stressed_inputs)

sensitivity = {
    "scenario": "BMI systematic underreport (all bands shifted +1)",
    "mean_premium_change_pct": round(
        (y_stressed.mean() - y_pred.mean()) / y_pred.mean() * 100, 2
    ),
    "gini_change": round(gini(y_test, y_stressed) - gini(y_test, y_pred), 4),
}
```

**Cybersecurity** is mostly out of scope for a pricing actuary — this is an IT security obligation. But the technical documentation should name the cybersecurity controls that apply to the model: access control on the training pipeline, model artefact integrity verification (hash the model file and store the hash in the documentation), and audit logs on inference calls.

---

## Step 7: The Declaration of Conformity

Once you have the QMS documented, the Annex IV technical documentation completed, and the Articles 9-15 controls in place, the conformity assessment itself is an internal review: does the QMS as implemented actually satisfy the requirements?

Article 47 requires a formal Declaration of Conformity. This is a signed document stating that the system complies with the Act. For a pricing model, the signatories should be the model owner (pricing actuary or pricing director) and the SMCR-designated senior manager responsible for the model. Keep the declaration short:

> *Declaration of Conformity — [Model Name] v[Version]*
>
> *We, [Insurer Name], declare that the AI system described in technical documentation reference [doc reference], version [version], dated [date], has been assessed in accordance with the conformity assessment procedure set out in Annex VI of Regulation (EU) 2024/1689 (EU AI Act) and conforms to the requirements set out in Articles 9 to 15 of that Regulation.*
>
> *Basis: Internal control per Article 43(2). Technical documentation held at [location/reference]. Quality management system in place per Article 17.*
>
> *Signed: [Name, Title, Date]*

Then register in the EU AI database (Article 71). The database is at ai-office.ec.europa.eu. As of March 2026 the database is operational for provider registration; the user interface has been criticised for being unclear about what constitutes a 'system' for registration purposes, and the Commission has acknowledged this. Register the system as you understand it and keep a record of your registration.

---

## What is still unclear

We said at the start we would be honest about what is unsettled. Three significant gaps remain as of March 2026:

**'Sufficiently representative' training data.** Article 10 requires this. There is no quantitative standard for insurance yet. The European Commission is commissioning CEN/CENELEC to produce harmonised standards; those are expected in late 2026 or 2027. Until then, document your methodology for assessing representativeness and be prepared to defend it.

**The Article 6(3) minimal-risk escape.** If your system poses no significant risk of harm to fundamental rights, it may not be high-risk even if it sits in Annex III. The Commission has published no guidance on how to assess this for insurance pricing. We think the escape applies more clearly to motor pricing (where the financial harm is typically modest) than to life pricing (where incorrect pricing can have material consequences for coverage decisions). But we have not seen a regulator test this position.

**Substantial modification.** Article 43(4) requires a new conformity assessment for substantial modifications. The Act gives examples — change of intended purpose, change of the AI system affecting risk — but does not define 'substantial' for an insurance model. Annual retraining on new data with the same architecture is probably not a substantial modification. Changing from a GLM to a GBM almost certainly is. The grey area is: retraining with a materially different vintage of data, adding new features, or changing the target variable. Document your reasoning when you re-train.

---

## Putting it together: the timeline

If your life or health pricing model needs to be compliant by 2 August 2026, here is a realistic schedule:

| Period | Activity |
|---|---|
| April 2026 | Model inventory. For each life/health pricing model, produce a one-page classification memo. Decide which are in scope. |
| May 2026 | Annex IV technical documentation. ModelCard, data governance documentation, technical architecture spec. This is the biggest single task. |
| May–June 2026 | Articles 9-15 controls. Risk register, Article 14 oversight design, Article 15 validation report. |
| June 2026 | Article 13 transparency document (deployer-facing). Article 12 logging confirmed as operational. |
| July 2026 | Internal QMS audit: does what we documented match what we have built? Fix the gaps. |
| July 2026 | Sign the Declaration of Conformity. Register in EU AI database. |

That schedule is achievable for a single pricing model. If you are a larger insurer with eight or ten life pricing models in production, start the inventory now.

---

## The insurance-governance library

The library was designed around SS1/23 model governance. The conformity assessment mapping is not coincidental — most of what the EU AI Act requires for high-risk AI systems is what good model governance requires anyway.

The current coverage:

- `ModelCard` (MRM module): Annex IV general description, intended purpose, limitations, approval trail
- `ModelInventory`: model register across the portfolio (prerequisite for Article 71 EU database registration)
- `GovernanceReport`: executive summary suitable for inclusion in the conformity assessment pack
- `ModelValidationReport`: Article 15 accuracy and robustness evidence
- `Assumption` / risk register: Article 9 risk management documentation

What it does not yet cover: Article 12 automated logging (that requires integration with your inference infrastructure), the Article 13 deployer-facing document as a separate output, and the Declaration of Conformity template. We are working on all three.

The library is at [github.com/burning-cost/insurance-governance](https://github.com/burning-cost/insurance-governance).

---

## Related reading

- [EU AI Act and Insurance Pricing — What You Actually Need to Know](https://burning-cost.github.io/regulation/2026/03/28/eu-ai-act-insurance-pricing-what-you-need-to-know/) — scope, classification, and the four-step decision sequence
- [What Real-Data Validation Actually Looks Like: freMTPL2 and PRA SS1/23](https://burning-cost.github.io/model-governance/2026/03/28/model-validation-fremtpl2-real-data-governance/) — calibrating Article 15 thresholds to real data
- FPF Conformity Assessment Guide (April 2025): fpf.org — the most practical public guide to Annex VI procedure; not insurance-specific but the procedural framework is sound
- EU AI Act full text: EUR-Lex, Regulation (EU) 2024/1689, OJ 12.7.2024
