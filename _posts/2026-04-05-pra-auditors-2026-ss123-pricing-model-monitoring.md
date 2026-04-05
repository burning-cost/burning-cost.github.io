---
layout: post
title: "What PRA auditors will ask to see in 2026: mapping SS1/23 to automated pricing model monitoring"
date: 2026-04-05
categories: [regulation, governance, monitoring]
tags: [PRA, SS1/23, model-risk, insurance-monitoring, insurance-governance, consumer-duty, TR24-2, SoP3-24, PRIN-2A, model-validation, gini-drift, calibration, CUSUM, PSI, CSI, AE-ratio, pricing-drift, audit, UK-insurance, python]
description: "SS1/23 applies to banks, not insurers — but its monitoring principles are the benchmark auditors use regardless. This post maps the five SS1/23 validation principles to what a production monitoring pipeline should be producing, and to the specific classes in insurance-monitoring and insurance-governance that cover each requirement."
author: Burning Cost
---

SS1/23 is the PRA's model risk management supervisory statement. It has been in force for UK banks since May 2023. It does not formally apply to Solvency II insurers.

That said, if you work in insurance pricing and your internal or external auditors are not using it as their de facto framework, you are in a shrinking minority. The PRA's January 2026 insurance supervision priorities letter requires firms to have "robust model assumptions with documented justification" and commits to intensifying engagement where "gaps between assumed and realised profitability are most material" — language that maps directly onto SS1/23 Principle 4 without naming the supervisory statement. The FCA's Consumer Duty (PRIN 2A) separately requires ongoing outcome monitoring with documented evidence. SoP3/24 (effective December 2024) requires annual attestation by a named SMF for firms with Solvency II internal model approval.

The regulatory hook differs. The substantive monitoring requirement is identical.

This post maps the five SS1/23 principles to what a production monitoring pipeline actually needs to produce, and to the specific tooling in `insurance-monitoring` and `insurance-governance` that addresses each one.

---

## The five principles in practice

SS1/23 is structured around five high-level principles: model identification and classification, model governance, model development and implementation, model validation, and model use and performance monitoring. For pricing teams, principles 4 and 5 are where auditors spend their time.

**Principle 4 — Model validation** requires independent challenge of model assumptions, performance benchmarking against alternatives, and documentation sufficient for a third party to reproduce the validation. The "independent" requirement means the team that built the model cannot sign off their own validation — which sounds obvious but is violated routinely in smaller firms where the same actuary builds and validates.

**Principle 5 — Model use and performance monitoring** requires that models have defined operating boundaries, that performance is monitored against those boundaries continuously (not just at annual review), and that breaches trigger documented escalation with auditable remediation.

Principle 5.2 — "usage restrictions and operating boundaries" — is where PRA supervisors for banks have found the most gaps in their first two years reviewing SS1/23 compliance. The common failure mode: the model has a validation report that notes some limitations, but nobody has translated those limitations into operational rules that actually constrain how the model is used. The limitation lives in a PDF. The model runs unconstrained in production.

Insurers are not immune to this failure mode. The PRA's 2026 priorities letter identifies firms where model outputs appear to be treated as instructions rather than inputs — "underwriting assumptions and model outputs are too optimistic, not supported by historical experience." That is a Principle 5 failure: no effective operating boundary, no mechanism to catch and escalate deteriorating performance.

---

## What automated monitoring needs to check

A production pricing model needs five things monitored and logged. Here is what each one is, why it matters, and what fires it.

### 1. Population stability

PSI (Population Stability Index) monitors whether the rating factor distribution has shifted between training and production. It answers "is this model still scoring a population it was trained on?" PSI > 0.25 is the conventional threshold for material shift requiring investigation; > 0.10 is the threshold for flagging.

```bash
pip install insurance-monitoring
```

```python
from insurance_monitoring.drift import psi, csi

# PSI on a single factor (e.g. vehicle group distribution)
stability = psi(
    reference=train_df["vehicle_group"].to_numpy(),
    current=prod_df["vehicle_group"].to_numpy(),
    exposure_weights=prod_df["earned_exposure"].to_numpy(),
)
# 0.0-0.1: no significant change
# 0.1-0.25: moderate shift — investigate
# >0.25: major shift — model may no longer be valid

# CSI across all rating factors simultaneously
factor_stability = csi(
    reference=train_df[rating_factors],
    current=prod_df[rating_factors],
    exposure_weights=prod_df["earned_exposure"].to_numpy(),
)
```

Use the exposure-weighted variant. Unweighted PSI treats a short-term policy identically to a multi-year policy. On a UK motor book with typical exposure heterogeneity, the difference is material.

### 2. Discrimination power

Gini coefficient drift is the fundamental test of whether a pricing model still does the job it was trained to do. A model can maintain calibration while losing discrimination — global A/E can look fine while the model is no longer correctly rank-ordering risks. Calibration tests alone will not catch this.

```python
from insurance_monitoring import ModelMonitor

monitor = ModelMonitor(
    distribution='poisson',
    n_bootstrap=500,
    alpha_gini=0.32,      # one-sigma, recommended for production monitoring
    alpha_global=0.32,
    alpha_local=0.32,
    random_state=0,
)
monitor.fit(y_ref, y_hat_ref, exposure_ref)

result = monitor.test(y_new, y_hat_new, exposure_new)
print(result.decision)    # 'REDEPLOY', 'RECALIBRATE', or 'REFIT'
print(result.summary())
```

The three-way decision logic matters. "RECALIBRATE" means only the global level has shifted — a balance correction fixes it. "REFIT" means the rank structure itself has degraded — recalibration will not fix it, the model needs to be rebuilt. This distinction is the substance of SS1/23 Principle 5.2 on usage restrictions: if the model's REFIT flag is firing, you have an operating boundary breach. What is the protocol?

The alpha default of 0.32 (one-sigma) is intentional and documented in arXiv:2510.04556. At alpha=0.05, the test has high Type II error for the drift magnitudes typically seen in UK motor books. A missed detection means a degraded model continues pricing. The cost of a false alarm — a model review — is low. Calibrate accordingly.

### 3. Calibration monitoring with early detection

Annual A/E review catches miscalibration after the damage is done. A motor pricing model that is 15% underpriced for twelve months before the annual review has repriced a book. The point of continuous monitoring is detection within weeks, not months.

`CalibrationCUSUM` implements the likelihood-ratio CUSUM from Franck, Driscoll, Szajnfarber, and Woodall (arXiv:2510.25573). It runs on monthly or weekly data batches and has documented ARL properties: at a 0.5% conditional false alarm rate, the in-control ARL is 200 periods. When the true rate doubles (a 100% miscalibration), ARL1 drops to around 37 periods. On monthly data, that is detection within three years of the shift. On weekly data, within nine months.

```python
from insurance_monitoring.cusum import CalibrationCUSUM

monitor = CalibrationCUSUM(
    delta_a=2.0,    # alternative: rate doubled
    cfar=0.005,     # 0.5% conditional false alarm rate per period
    n_mc=5000,
    random_state=0,
)

# Run month-by-month as data arrives
for month_data in monthly_batches:
    alarm = monitor.update(
        p=month_data["predicted_frequency"],
        y=month_data["observed_claims"],
    )
    if alarm:
        escalate(f"Calibration alarm at {alarm.time}: CUSUM={alarm.statistic:.2f}")
```

For the SS1/23 submission (for banks) or the equivalent governance documentation (for insurers), the CUSUM log is the evidence that continuous monitoring was operating during the period. The absence of alarms is a documented outcome, not a gap in the record.

### 4. Full drift attribution

The `PricingDriftMonitor` combines the Gini-based ranking test with GMCB (global miscalibration) and LMCB (local cohort-level miscalibration) tests from Brauer, Menzel and Wüthrich (arXiv:2510.04556). This is the complete two-step procedure: first determine whether rank structure has degraded, then decompose any calibration failure into level shift versus structural breakdown.

```python
from insurance_monitoring.pricing_drift import PricingDriftMonitor

monitor = PricingDriftMonitor(
    distribution='poisson',
    n_bootstrap=500,
    alpha_gini=0.32,
    random_state=0,
)
monitor.fit(y_ref, mu_ref, exposure=exposure_ref)

result = monitor.test(y_mon, mu_mon, exposure=exposure_mon)
print(result.verdict)       # 'OK', 'RECALIBRATE', or 'REFIT'
print(result.summary())
```

The LMCB test is the piece that answers the auditor's question about operating boundaries. LMCB fires when the miscalibration is cohort-level — some segments are systematically mispriced while others are not. Global A/E might look fine. The LMCB catches the structural failure that aggregate statistics miss. This is exactly the monitoring that would detect the scenario the PRA's priorities letter describes: model outputs that look reasonable in aggregate but are generating "gaps between assumed and realised profitability" in specific cohorts.

### 5. Explainability and escalation audit trail

Performance monitoring that fires alerts into a void is not governance. The SS1/23 Principle 5 requirement for "auditable remediation" means there needs to be a record of: what fired, who was notified, what decision was made, and what happened to the model as a result.

```bash
pip install insurance-governance
```

The `ExplainabilityAuditTrail` — implemented in `insurance_governance.audit` — provides the per-decision log. Combined with the `OutcomeTestingFramework`, it covers the two audit artefacts:

```python
from insurance_governance.outcome.framework import OutcomeTestingFramework

framework = OutcomeTestingFramework(
    model_card=card,
    policy_data=quarterly_data,
    period="2026-Q1",
    price_col="premium",
    claim_amount_col="claims_paid",
)
framework.run()
framework.to_json("outcome_testing_2026q1.json")
framework.generate("outcome_testing_2026q1.html")
```

The `monitoring_owner` field on the `ModelCard` links the monitoring output to the SMCR accountability chain. For banks, this is directly required by SS1/23. For insurers, PRIN 2A requires "evidence of compliance" with outcome monitoring — the same documentation substance, different legal hook.

---

## What auditors actually ask for

Based on published PRA findings from the first two years of SS1/23 supervision for banks, and from FCA Consumer Duty multi-firm review outputs (TR24/2, August 2024), the questions cluster around three areas:

**Documentation completeness.** Is there a validation report for every model in production? Is it dated, version-tagged, and signed by someone who was not the model developer? Does it document operating boundaries — not just what the model is, but the conditions under which the model should not be trusted?

**Monitoring frequency and response.** How often is performance monitored? What are the defined thresholds? When was the last threshold breach, what decision was made, and who made it? "We review models annually" does not satisfy Principle 5. "We run monthly CUSUM monitoring with a 0.5% false alarm rate, and threshold breaches are reviewed within 10 business days by [SMF designation]" does.

**Escalation evidence.** Can you show a complete record of threshold breaches and what happened after each one? Not the alert — the response. The motor finance scheme's retrospective enforcement is the reason this question now has real teeth: firms need to be able to demonstrate, years later, that they knew what their models were doing and responded appropriately when they did not.

The TR24/2 review (FCA multi-firm review of Consumer Duty implementation, August 2024) found that most insurers' monitoring approaches were "high-level summaries with little substance." The firms that produce segmented, time-stamped, version-tagged monitoring outputs with documented escalation trails are in a materially different position when a supervisor asks to see the evidence.

---

## The honest answer on SS1/23 for insurers

SS1/23 is not mandatory for Solvency II insurers. The PRA has said it will "provide an update on the scope for non-IM firms at a future date" — that was in May 2023, and no update has been issued as of April 2026. A formal extension to insurers remains possible in the 2026–2027 timeframe, but it is not confirmed and we will not pretend otherwise.

What is clear: the monitoring principles in SS1/23 — operating boundaries, continuous performance tracking, auditable escalation — are the benchmark that auditors and supervisors are applying regardless of whether the supervisory statement formally binds the firm. The PRA's 2026 insurance supervision priorities use identical language. Consumer Duty uses identical requirements with different terminology. SoP3/24 creates live attestation obligations for IM insurers that require the same underlying infrastructure.

Building the monitoring pipeline described above is the right call for any UK GI firm running a live pricing model, independent of the regulatory framing question. The question is not whether the obligation will be formally extended to insurers — it is whether your monitoring is adequate by the standard that the whole market is converging on.

---

## References

- PRA SS1/23: Model Risk Management (applicable to banks and building societies), effective May 2023
- PRA SoP3/24: Internal Model Ongoing Review (for Solvency II IM insurers), effective December 2024
- PRA Insurance Supervision Priorities, January 2026
- FCA Consumer Duty (PRIN 2A), effective July 2023
- FCA TR24/2: Multi-firm review of Consumer Duty implementation, August 2024
- Brauer, Menzel & Wüthrich (2025): arXiv:2510.04556 — model monitoring framework (Gini drift + GMCB/LMCB)
- Franck, Driscoll, Szajnfarber & Woodall (2025): arXiv:2510.25573 — calibration CUSUM

Related on Burning Cost:

- [Does model monitoring actually work?](/2026/03/27/does-automated-model-monitoring-actually-work/) — empirical results on freMTPL2
- [Does PSI actually catch pricing model drift?](/2026/03/28/does-psi-model-monitoring-actually-catch-pricing-model-drift/) — PSI vs. Gini drift on real data
- [Motor model mispricing caught by monitoring](/2026/03/23/motor-model-mispricing-caught-by-monitoring/) — worked example
- [The governance bottleneck: champion-challenger without an audit trail](/2026/03/13/your-champion-challenger-test-has-no-audit-trail/) — escalation records
