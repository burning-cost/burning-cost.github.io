---
layout: post
title: "Building a regulatory-grade AI pricing model for the FCA's Live Testing sandbox"
date: 2026-04-05
categories: [regulation, governance, libraries]
tags: [FCA, AI-Live-Testing, consumer-duty, sandbox, insurance-conformal, insurance-fairness, insurance-governance, uncertainty-quantification, PRIN-2A, PS22-9, FCA-Innovation-Hub, python, open-source, regulatory-submission, model-risk]
description: "FCA cohort 2 is live as of April 2026. This post maps the four things a submission actually needs to include — outcome monitoring, proxy bias detection, uncertainty quantification, explainability — to concrete Python tooling you can build today."
author: Burning Cost
---

The FCA's AI Live Testing sandbox has its second cohort running from April 2026. If you have not applied, you did not miss a trick — cohort 3 will follow. But the artefacts cohort 2 firms are building right now will become the template for what the FCA expects from the rest of the market by 2027.

This post is not about what the programme is. We covered that [in a companion piece published today](/2026/04/05/fca-ai-live-testing-cohort-2-pricing-model-requirements/). This one is about what you actually have to build and hand over, and what open-source Python tools do which parts of it.

---

## What a sandbox submission actually contains

The FCA is not reviewing a PowerPoint. Cohort 2 firms are running live AI pricing systems under FCA observation. The evidence package — what you hand to the Innovation Hub supervisors — is a combination of live monitoring output, documented test results, and escalation records. Four things need to be in that package.

### 1. Quantified uncertainty at the point of decision

A model that produces only point estimates does not meet the FCA's emerging expectations for AI pricing. Consumer Duty PRIN 2A requires firms to demonstrate that outcomes are appropriate for the consumer receiving them. For a high-uncertainty risk, a point-estimate premium is not a defensible basis for a pricing decision — you need to know how much you do not know.

Install the library:

```bash
pip install insurance-conformal
```

The library implements conformal prediction intervals with a normalisation that is correct for insurance data. The nonconformity score is \\(|y - \hat{y}| / \hat{y}^{p/2}\\), where \\(p\\) is the Tweedie power parameter. This divides the raw residual by the model's local standard deviation estimate, so a £50 error on a £200 expected loss is not treated identically to a £50 error on a £5,000 expected loss — which is exactly what happens with unnormalised conformal or parametric intervals.

```python
from insurance_conformal import TweedieConformalPredictor

predictor = TweedieConformalPredictor(tweedie_power=1.5, alpha=0.1)
predictor.fit(y_calib, y_hat_calib)

intervals = predictor.predict_interval(y_hat_new)
# intervals: DataFrame with columns [lower, upper, width, coverage_flag]
```

What the FCA submission needs: the coverage rate by decile of predicted loss, not just overall. On a heterogeneous motor book, aggregate 90% coverage can coexist with 70% coverage in the top risk decile — where the decisions are most consequential. The submission should show per-decile coverage and a defined threshold below which the pricing decision is flagged for manual review.

### 2. Proxy discrimination evidence

The FCA has not found systemic bias in pricing AI — it has said it has not yet found it. That phrasing is doing work. FCA EP25/2 (published January 2025) specifies how proxy discrimination evidence should be constructed for GI firms. Standard Spearman correlation between postcode and ethnicity proportions returns \\(|r| \approx 0.10\\) on UK motor books. A nonlinear proxy detection approach returns \\(R^2 \approx 0.62\\) for the same postcode-to-ethnicity link. The difference determines whether the test finds the problem or misses it.

```bash
pip install insurance-fairness
```

```python
from insurance_fairness.proxy_detection import proxy_r2_scores

scores = proxy_r2_scores(
    df=policy_df,
    protected_col="lsoa_prop_south_asian",
    factor_cols=["vehicle_group", "postcode_district", "ncb_years", "driver_age"],
)
# Returns dict: {factor_name: R2_score}
# R2 > 0.1 flags the factor for Equality Act Section 19 proportionality review
```

The Equality Act 2010 Section 19 test is the legal hook. For any rating factor with a proxy correlation above your defined threshold, you need a documented proportionality argument: why is this factor retained, what is the actuarial justification, and is there a less discriminatory alternative that achieves comparable risk differentiation? The FCA will ask to see this argument for postcode specifically.

The `IndirectDiscriminationAudit` class measures proxy vulnerability across the full premium model — how much a protected attribute can be inferred from the pricing output even after it has been excluded from the model inputs:

```python
from insurance_fairness.indirect import IndirectDiscriminationAudit

audit = IndirectDiscriminationAudit(
    protected_attr="ethnicity_proxy",
    proxy_features=["postcode_district"],
    exposure_col="earned_exposure",
)
result = audit.fit(X_train, y_train, X_test, y_test)
print(result.proxy_vulnerability)   # 0.0 = no leakage, 1.0 = full leakage
print(result.summary)               # per-segment breakdown
```

Note: when the protected column is an LSOA-level area proportion rather than individually observed ethnicity, the disparity ratio is attenuated downward. A measured ratio of 1.08 at 65% proxy accuracy implies a true disparity of roughly 1.12. Report the bounds, not just the point estimate.

### 3. Per-decision audit trails

The Treasury Select Committee's concern about AI transparency in insurance — raised in late 2025 — centres on whether consumers can understand adverse pricing decisions. The FCA's response is that firms need per-decision records: not a general model explanation, but a record of which features drove this specific premium for this specific consumer, retained for the regulatory look-back period.

The motor finance scheme's seventeen-year look-back is the relevant precedent for what "look-back period" might mean in practice.

```bash
pip install insurance-governance
```

```python
from insurance_governance.audit.log import ExplainabilityAuditLog
from insurance_governance.audit.entry import ExplainabilityAuditEntry

audit_log = ExplainabilityAuditLog(
    path="./audit_log.jsonl",
    model_id="motor_freq_v3.1",
    model_version="3.1.0",
)

entry = ExplainabilityAuditEntry(
    model_id="motor_freq_v3.1",
    model_version="3.1.0",
    input_features=feature_dict,           # {feature_name: value}
    feature_importances=shap_dict,         # {feature_name: shap_value}
    prediction=model_output,
    final_premium=charged_premium,
    reviewer_id="SMF7-JANE-SMITH",         # SMCR Controlled Function holder
)
audit_log.append(entry)
```

The log is append-only JSONL — every entry is content-hashed so the regulatory export proves the record has not been altered. The `export_period` method produces a filtered export for a date range, which is what you hand to the FCA supervisor.

### 4. Outcome monitoring with escalation records

This is the piece most firms have partially — they have monitoring dashboards. The FCA's ask is the bit after the dashboard: who saw it, what threshold was breached, who decided it was material, and what happened next. The monitoring output needs an escalation trail linked to a named SMCR function.

```python
from insurance_governance.outcome.framework import OutcomeTestingFramework
from insurance_governance.outcome.segments import CustomerSegment
from insurance_governance.mrm.model_card import ModelCard

card = ModelCard(
    model_id="motor_freq_v3.1",
    model_class="pricing",
    monitoring_owner="SMF7",
    ...
)

renewal_seg = CustomerSegment(name="renewal_cohort", filters={"renewal_flag": 1})
new_biz_seg = CustomerSegment(name="new_business", filters={"renewal_flag": 0})

framework = OutcomeTestingFramework(
    model_card=card,
    policy_data=policy_df,
    period="2026-Q1",
    price_col="premium",
    claim_amount_col="claim_amount",
    days_to_settlement_col="days_to_settlement",
    customer_segments=[renewal_seg, new_biz_seg],
)
results = framework.run()
framework.generate("outcome_testing_2026q1.html")
framework.to_json("outcome_testing_2026q1.json")
```

The JSON export is the machine-readable record. The HTML is for the governance committee. The `smf_owner` field on the model card links the monitoring output to a specific accountable individual — the FCA will ask to see that link.

---

## The submission package

For a firm in cohort 2, the package is not a single document. It is a pipeline that produces outputs continuously. What the FCA reviews at the end of the cohort period is:

- Coverage verification reports from `insurance-conformal`, showing per-decile interval width and coverage rates
- Proxy discrimination audit outputs from `insurance-fairness`, version-tagged to the model deployment date, with Equality Act proportionality arguments for flagged factors
- `ExplainabilityAuditLog` JSONL exports covering the cohort period, with hash-chain integrity
- `OutcomeTestingFramework` quarterly reports showing segmented A/E ratios, loss ratios by customer cohort, and any threshold breaches with escalation records

The escalation records are the critical item. The FCA is not just evaluating the measurement infrastructure. It is evaluating whether the measurement connects to decisions made by accountable people. A dashboard that fires alerts into a void fails the Consumer Duty governance standard regardless of how sophisticated the underlying statistics are.

---

## Where the Innovation Hub fits

The FCA Innovation Hub is the entry point for firms considering cohort 3 or engaging informally with the programme. It provides pre-application support — the Hub team will tell you whether your use case is in scope before you invest in the full application. For AI pricing models, the relevant scope question is whether the model is making material decisions that affect consumer outcomes, which any live motor or home pricing model does.

The Hub's published materials from cohort 1 (the digital finance cohort) are the best public signal of what the FCA's sandbox team actually looks for in a submission. They prioritise firms that have done the technical work over firms with polished slides. The four elements above are the technical work.

---

## What firms outside the sandbox should be doing

The cohort 2 evaluation report, due at year-end 2026, will set the supervisory benchmark. Firms not in the sandbox will be measured against it. The timing is tight: the Mills Review recommendations (due summer 2026) plus the evaluation report at year-end means the supervisory framework will be substantially settled by early 2027.

Building the four elements now — conformal intervals, proxy audit, audit trail, outcome framework — means you have the infrastructure in place before the framework crystallises. It also means you have the historical record showing you were running adequate monitoring during the 2025–2026 implementation period, which matters for any retrospective supervisory engagement.

None of this is novel. The methods are proven, the regulatory obligations under PRIN 2A are clear, and the libraries are free. The question is whether the pipeline is in production.

---

## References

- FCA AI Live Testing: [fca.org.uk/firms/innovation/ai-live-testing](https://www.fca.org.uk/firms/innovation/ai-live-testing)
- FCA Innovation Hub: [fca.org.uk/firms/innovation/innovation-hub](https://www.fca.org.uk/firms/innovation/innovation-hub)
- FCA EP25/2: Evaluation of proxy discrimination in general insurance pricing, January 2025
- FCA Consumer Duty (PRIN 2A), effective 31 July 2023
- FCA PS22/9: Consumer Duty policy statement, July 2022
- Equality Act 2010, Section 19 (indirect discrimination)
- Mills Review: recommendations expected summer 2026

Related on Burning Cost:

- [What the FCA's Live Testing cohort actually requires](/2026/04/05/fca-ai-live-testing-cohort-2-pricing-model-requirements/) — the four compliance pillars in detail
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the uncertainty quantification foundation
- [The Fairness Audit That Passes Without Measuring Anything](/2026/04/05/proxy-race-fairness-audit-bias-insurance-pricing/) — why LSOA proxies understate disparity
