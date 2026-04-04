---
layout: post
title: "What PRA Auditors Will Ask to See in 2026: Mapping SS1/23 to Automated Pricing Model Monitoring"
date: 2026-04-05
categories: [regulation, monitoring]
tags: [pra-ss123, model-monitoring, model-risk, insurance-monitoring, insurance-governance, operating-boundaries, principle-4, principle-5, uk-insurance]
description: "SS1/23 is in active enforcement for banks. Its monitoring principles — operating boundaries, outcome analysis, documented escalation — are the benchmark the whole market is being measured against. Here is what auditors are specifically looking for, mapped to the artefacts you need to produce."
---

SS1/23 — PRA's supervisory statement on model risk management, published May 2023 and effective May 2024 — is no longer about implementation planning. The PRA Business Plan 2025/26 states plainly that the regulator "will continue to focus on how banks are embedding and implementing the expectations set out in SS1/23." Thematic supervisory engagement with a sample of banks started in 2024. SMFs have been put on notice that engagement will focus on SS1/23 alignment.

What supervisory engagement since May 2024 has consistently found: firms relying on spreadsheet-based monitoring, manual validation cycles, and "paper-based compliance" — policies that exist on paper but are not systematically executed. The PRA has been explicit that this is inadequate.

**One clarification before we go any further.** SS1/23 is mandatory for UK-incorporated banks, building societies, and PRA-designated investment firms with internal model approval for regulatory capital. It is not mandatory for Solvency II insurers. If you are an insurer, you are not subject to SS1/23 directly — but you are subject to the same substantive monitoring requirements via Consumer Duty, PRA's 2026 insurance supervision priorities, and SoP3/24 IMOR if your firm holds IM approval. We will come back to the insurer position explicitly. The reason to understand SS1/23 regardless is that its monitoring framework has become the de facto standard across the entire market — insurers, consultants, and auditors are all using it as the benchmark.

---

## The five principles in 60 seconds

SS1/23 is organised around five principles. The PRA's exact titles:

| Principle | Title | What it governs |
|-----------|-------|-----------------|
| 1 | Model identification and model risk classification | Comprehensive model inventory; risk tiering (High/Medium/Low); determines validation frequency |
| 2 | Governance | Board and SMF accountability; MRC structure; MRM framework self-assessment |
| 3 | Model development, implementation and use | Development standards; documentation; appropriate use; limitations communicated to users |
| 4 | Independent model validation | Validation independence; methodology review; ongoing monitoring (outcome analysis, benchmarking, sensitivity, operating boundaries) |
| 5 | Model risk mitigants | Usage restrictions; escalation procedures; auditable remediation; temporary approval process for unvalidated models |

The two principles that generate the most audit findings in 2025-2026 are 4 and 5 — specifically the ongoing monitoring and escalation requirements. That is where we will spend most of this post.

---

## Principle 4.4: what 'outcome analysis and benchmarking' actually means

Sub-principle 4.4 states: "Model monitoring should include benchmarking (e.g. by running models in parallel), sensitivity analysis and outcome analysis, with regular reports being produced."

These three terms — benchmarking, sensitivity analysis, outcome analysis — sound reasonable. The audit problem is that most pricing teams interpret "outcome analysis" as "calculate the A/E ratio quarterly and look at it." That is not outcome analysis in any sense that satisfies the principle.

Genuine outcome analysis for a pricing model requires three things.

**First, global calibration.** Is the model over- or under-predicting across the entire portfolio? An A/E ratio captures this, but only if it is accompanied by a statistical test that tells you whether the deviation is within sampling noise or represents a genuine shift. An A/E of 1.05 on a book of 40,000 policies looks concerning. On a book of 500 policies in a low-frequency class, it may be entirely consistent with the null hypothesis that the model is correctly calibrated.

**Second, local calibration.** Is the model systematically wrong in specific segments? A model that overprices a 25-year-old driver by 15% and underprices a 55-year-old driver by 12% may read 1.01 at portfolio level. The A/E ratio does not catch this. The adverse selection consequence — competitors sweep your mispriced segments — is real, material, and invisible in the headline number. Outcome analysis needs to decompose portfolio-level calibration into segment-level components and flag where the model is systematically biased.

**Third, discrimination monitoring.** Has the model's ability to rank risks degraded? A model with a Gini coefficient that has fallen from 0.42 to 0.35 over four quarters is drifting regardless of whether the A/E ratio looks fine. Benchmarking against the validation-period Gini with a proper statistical test — not just a visual comparison of point estimates — is what the principle requires.

The research paper arXiv:2510.04556 (Brauer, Menzel and Wüthrich, October 2025) provides the statistical machinery for all three of these. Murphy score decomposition separates global from local miscalibration with formal hypothesis tests; the Gini drift test establishes asymptotic normality of the Gini coefficient under the stationarity hypothesis, giving a z-statistic for whether observed Gini decline is sampling noise or genuine discrimination drift. **We note explicitly: the paper does not reference SS1/23 or PRA regulatory requirements. The mapping from the paper's methodology to SS1/23 outcome analysis and benchmarking requirements is our analytical contribution, not the paper's claim.**

In `insurance-monitoring`, both components are implemented directly:

```python
from insurance_monitoring import ModelMonitor

monitor = ModelMonitor(
    distribution="poisson",
    n_bootstrap=400,
    alpha_gini=0.32,    # one-sigma early warning
    alpha_global=0.32,
    alpha_local=0.32,
    random_state=0,
)
monitor.fit(y_ref, y_hat_ref, exposure_ref)

result = monitor.check(y_new, y_hat_new, exposure_new)
print(result.decision)          # "continue" / "recalibrate" / "refit"
print(result.global_p_value)    # calibration test p-value
print(result.local_p_value)     # segment miscalibration test p-value
print(result.gini_z_stat)       # Gini drift z-statistic
```

The `result.decision` field implements the three-way decision tree from the paper: continue (model performing within tolerance), recalibrate (global miscalibration only, adjustable via intercept), or refit (local miscalibration or Gini drift, requires new training). This decision is what you put in the monitoring report with the supporting test statistics. That is what an auditor wants to see.

---

## Principle 4.4: operating boundaries — the PRA's biggest complaint

The second major component of sub-principle 4.4 is operating boundaries. The PRA's definition, which is precise and matters:

> "The sample data range (including empirical variance-covariance relationships in the multivariate case) used to estimate the parameters of a statistical model."

The regulatory logic is straightforward. A model trained on UK motor data from 2018-2022 has operating boundaries that include the economic, regulatory, and demographic conditions of that period. When conditions move outside those boundaries, using the model without documented justification involves increased model risk — not necessarily wrong predictions, but a regime the model was not designed for. The firm must document that it is aware of the boundary exceedance and has made a considered decision to continue using the model.

The PRA's thematic finding from supervisory engagement (corroborated by 4most, ValidMind, and Deloitte analysis): "Models trained in benign environments are being used in volatile ones without clear thresholds for when they should be turned off or recalibrated."

Concrete examples from general insurance pricing:

- A motor frequency model trained on 2018-2021 data used during 2022-2023. Vehicle parts inflation peaked above 14% in that period. The model had never encountered an inflationary environment of that magnitude. No boundary check documented.
- A bodily injury severity model trained on pre-whiplash-reform claims used after the May 2021 whiplash reform came into force. The reform changed the claims distribution fundamentally. Applying the pre-reform model post-reform is a boundary breach.
- A motor frequency model trained on pre-pandemic mileage patterns applied during and after 2020-2022. Annual mileage distributions shifted materially across vehicle segments. The model's feature covariance structure — particularly the relationship between vehicle type, usage, and mileage — was outside the training distribution.
- A home insurance model used for property types or construction materials introduced after the training period (timber-frame construction volumes, new flat-roof proportions in some regions).

What an auditor expects to see is not a philosophical discussion of these risks. They want:

1. For each Tier 1 model: a documented table of training data range for key input variables — mean, min, max of key rating factors, time period, geographical coverage, claims inflation environment.
2. A quarterly check comparing current in-force portfolio conditions to those boundaries — specifically for macroeconomic inputs (claims inflation, interest rates where relevant), regulatory environment changes, and significant portfolio composition shifts.
3. When the check detects an approach or breach: a documented decision. Not silence. Not "we noted the issue." A written decision with named sign-off: "Pricing Manager confirmed on [date] that despite [condition], the model remains appropriate because [reason]. Review scheduled for [date]."
4. Evidence that the board or a relevant governance committee was informed of any material boundary breach.

The absence of documented operating boundaries for Tier 1 models is, as of 2026, one of the most common adverse findings in external model reviews.

---

## Principle 5: escalation, usage restrictions, and auditable remediation

Principle 5 is where governance meets monitoring. Sub-principle 5.2 requires: "Usage restrictions should be defined and be transparent to users, and remediation of model issues should be tracked and follow an auditable process."

This is not about having a policy document. It is about having records. Specifically:

**Usage restrictions** per model: what is this model not to be used for? The `MRMModelCard` in `insurance-governance` captures this as the `intended_use` field with explicit limitations:

```python
from insurance_governance import MRMModelCard, Assumption

mrm_card = MRMModelCard(
    model_id="motor-freq-v3",
    model_name="Motor TPPD Frequency",
    version="3.2.0",
    model_class="pricing",
    intended_use="Frequency pricing for UK private motor. Not for commercial fleet.",
    assumptions=[
        Assumption(
            description="Claim frequency stationarity since 2022",
            risk="MEDIUM",
            mitigation="Quarterly A/E monitoring; ad-hoc review if PSI > 0.25",
        ),
        Assumption(
            description="Post-whiplash reform claims distribution applies",
            risk="HIGH",
            mitigation="Bodily injury severity monitored separately; "
                       "model retrained on post-reform data from 2022",
        ),
    ],
)
```

**Escalation log**: documented instances where models approached or breached operating boundaries, with the response and sign-off. This is not a monitoring dashboard alert that someone dismisses. It is a timestamped record: alert raised, reviewed by [name, role], decision [continue/recalibrate/refit], rationale, next review date.

**Post-model adjustment (PMA) register**: every manual override of model output needs documented justification and independent review. The PRA has found firms where PMAs have accumulated over years with no audit trail for why they were introduced or when they should be retired.

**Temporary approval records** (sub-principle 5.3): if a model is used before full validation is complete, that temporary approval must be documented with an expiry and a validation timeline.

Sub-principle 5.2 also requires early warning indicators — what we would call monitoring triggers. In `insurance-governance`, these are set at validation report creation:

```python
from insurance_governance import ModelValidationReport, ValidationModelCard

val_card = ValidationModelCard(
    name="Motor Frequency v3.2",
    version="3.2.0",
    purpose="Predict claim frequency for UK private motor portfolio",
    methodology="CatBoost gradient boosting with Poisson objective",
    target="claim_count",
    features=["age", "vehicle_age", "area", "vehicle_group"],
    limitations=["No telematics data", "Not calibrated for commercial fleet"],
    owner="Pricing Team",
)

report = ModelValidationReport(
    model_card=val_card,
    y_val=y_val,
    y_pred_val=y_pred_val,
    y_train=y_train,
    y_pred_train=y_pred_train,
    exposure_val=exposure_val,
    monitoring_owner="Head of Pricing",
    monitoring_triggers={"ae_ratio": 1.10, "psi": 0.25},
)

report.generate("validation_report.html")
report.to_json("validation_report.json")
```

The `monitoring_triggers` dictionary is the documented early warning indicators: A/E breach at 1.10, PSI breach at 0.25. These thresholds become part of the validation record. When monitoring subsequently fires an alert, it references these documented thresholds rather than ad hoc judgment. That is the audit trail sub-principle 5.2 requires.

---

## The technology gap

The consistent finding from PRA supervisory engagement in 2024-2025 is that firms have monitoring processes on paper and monitoring gaps in practice. Manual quarterly spreadsheet calculations, validation reports produced by the same team that built the model, escalation processes that exist in a procedure document but are never tested.

The PRA's language is "paper-based compliance" — having the right policies and procedures documented without the systematic execution to match. This is explicitly flagged as inadequate. Full alignment "will take several years for many firms due to manual bottlenecks" — meaning the PRA expects to find ongoing gaps and is using supervisory engagement to drive progress, not to sanction firms that are visibly working on it.

The practical test for whether your monitoring is adequate is simple: could you produce, within 48 hours of being asked, for each of your Tier 1 pricing models: the model's documented operating boundaries; the last quarterly monitoring report with test statistics, not just headline ratios; the escalation log for any boundary approach or breach in the last 12 months; and the validation finding register with current open items and remediation status?

If the answer requires manual assembly across spreadsheets, email threads, and a SharePoint folder that only one person knows the structure of, you have the technology gap the PRA is flagging.

---

## For insurers: same substance, different legal hook

SS1/23 does not apply to you as an insurer. The relevant frameworks are:

**FCA Consumer Duty (PRIN 2A)**: firms must "regularly assess, test, understand and evidence the outcomes their customers are receiving." For a GI pricing model, that means documented monitoring of pricing outcomes by customer segment, not just aggregate loss ratios. The monitoring trigger structure in `insurance-governance` produces exactly this evidence. This is not aspirational — Consumer Duty has been in force since July 2023.

**PRA 2026 insurance supervision priorities** (January 2026 Dear CEO letter): the PRA requires "robust model assumptions with documented justification," "investment in data quality," and will "intensify engagement where gaps between assumed and realised profitability are most material." This is the operating boundaries requirement stated in different language. A model whose assumptions are not supported by documented monitoring output is precisely the gap the PRA will probe.

**SoP3/24 IMOR** (PRA Statement of Policy, February 2024, effective December 2024): for firms with Solvency II internal model approval, the Internal Model Ongoing Review framework requires annual attestation by a named SMF — typically the CRO — that the internal model remains fit for purpose. The attestation must be supported by documented evidence. The monitoring and governance artefacts discussed in this post are the evidence base for that attestation.

**Voluntary adoption**: many larger insurers have adopted SS1/23 principles ahead of any formal extension. KPMG and ValidMind both report this is increasingly common practice, particularly at firms that share model governance functions across banking and insurance entities.

The PRA has stated it will "provide an update on the scope for non-IM firms at a future date." As of April 2026, no such update has been issued. Whether or when SS1/23 is formally extended to insurers remains unclear. But the substance of what is being asked — systematic monitoring, documented operating boundaries, auditable escalation — is already mandatory via the routes above.

---

## The statistical foundation: arXiv:2510.04556

The paper by Brauer, Menzel and Wüthrich (arXiv:2510.04556, October 2025) does not reference SS1/23 or PRA regulatory requirements. This is not a paper about regulatory compliance. It is a paper about statistical model monitoring methodology applied to non-life insurance pricing. The mapping we describe here is our own analytical contribution — the paper does not make these claims.

What the paper provides is the statistical rigour that makes SS1/23-style monitoring defensible under challenge.

The Murphy decomposition splits the deviance loss into global miscalibration (GMCB) and local miscalibration (LMCB) components. This maps directly onto the SS1/23 distinction between overall model performance and segment-level bias. When an auditor asks "how do you know the model is not systematically mispricing a segment?" — the LMCB statistic is the quantitative answer. It is not enough to say "we looked at double lift charts." A formal test with a p-value is what "systematic, documented, proportionate monitoring" means in practice.

The Gini drift test establishes asymptotic normality of the Gini coefficient under stationarity. The z-statistic it produces — comparing new-period Gini against the distribution expected under the null hypothesis of no drift — is the right tool for the SS1/23 4.4 benchmarking requirement. A Gini comparison between quarter n and quarter n-1 is not benchmarking. Benchmarking is testing whether the current Gini is consistent with the model's performance at validation, accounting for sampling variance.

The virtual/real drift distinction addresses the operating boundaries requirement. Virtual drift is covariate shift — the portfolio composition has changed but the model's conditional predictions remain correct. Real concept drift is when the underlying relationship between features and claims has changed — the model is wrong regardless of portfolio composition. Distinguishing between the two determines the appropriate response: feature drift alone does not necessarily warrant a refit; concept drift does. This distinction is exactly what the operating boundaries analysis is trying to surface.

---

## What the artefact mapping looks like in practice

To summarise the full chain: what SS1/23 (or its insurer equivalents) requires, what statistical test satisfies it, and which library component produces the output.

| SS1/23 requirement | Statistical approach | Library component |
|-------------------|---------------------|-------------------|
| Model inventory with risk tiering | Composite score across 6 dimensions | `RiskTierScorer` → `ModelInventory.register()` |
| Documented operating boundaries | Training data range table; PSI for feature drift | `MRMModelCard` assumptions + `ModelMonitor` PSI |
| Outcome analysis: global calibration | Murphy GMCB; A/E ratio with test statistic | `ModelMonitor.check()` → `global_p_value` |
| Outcome analysis: local calibration | Murphy LMCB; isotonic regression residuals | `ModelMonitor.check()` → `local_p_value` |
| Benchmarking: discrimination monitoring | Gini drift z-statistic | `ModelMonitor.check()` → `gini_z_stat` |
| Decision logic (continue/recalibrate/refit) | Three-way decision tree | `result.decision` |
| Validation report with RAG status | Combined test results | `ModelValidationReport.generate()` |
| Usage restrictions documented | `intended_use` + `limitations` fields | `MRMModelCard` → `GovernanceReport` |
| Early warning indicators | Documented monitoring triggers | `monitoring_triggers` in `ModelValidationReport` |
| Auditable escalation log | Validation history records | `ModelInventory.update_validation()` |

Install both packages:

```bash
uv add insurance-monitoring insurance-governance
```

The validation report produces a JSON file that can be ingested into any governance system. The `ModelInventory` writes to a plain JSON file that lives in version control — every update is a git commit, so the audit trail is the commit history.

---

The PRA is not prescribing specific statistical tests. It is principles-based — it wants evidence of systematic, documented, proportionate monitoring. What "systematic" means in practice is that monitoring runs on a defined schedule, produces outputs that can be reviewed independently of the person who ran them, and feeds into a decision process with documented outcomes. What "documented" means is that the test statistics, thresholds, decisions, and sign-offs are all in writing before the fact, not assembled retrospectively when a supervisor asks for them.

The technology gap the PRA has found is not a knowledge gap. Most pricing teams understand that their quarterly A/E check is inadequate. The gap is between understanding and implementation — between knowing what good looks like and having the infrastructure to produce it systematically. That is a solvable problem.
