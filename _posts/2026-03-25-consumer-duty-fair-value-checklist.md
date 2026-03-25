---
layout: post
title: "Consumer Duty Fair Value Assessment: 12-Step Technical Checklist"
date: 2026-03-25
categories: [regulation, consumer-duty]
tags: [fca, fair-value, consumer-duty, insurance-governance, insurance-fairness, insurance-monitoring, ps21-5, pricing, python, polars, uk-pricing, model-risk]
description: "The FCA's first full Consumer Duty outcomes year is behind us. Here is the technical workflow a pricing actuary needs to run the annual fair value assessment — with code."
---

The FCA's first full Consumer Duty outcomes reporting year closed in January 2026. If your firm is still assembling the fair value assessment as a governance narrative rather than a reproducible technical workflow, you are already behind. Boards need a RAG-rated product-by-product summary. Compliance wants an audit trail. The FCA, if it comes knocking, wants evidence that the assessment is a live process, not a once-a-year document exercise.

What follows is the 12-step workflow we use. Each step names the data you need, the code that runs it, and the threshold between "fine" and "escalate." We have written this for UK personal lines motor and home, but the structure applies to any product line under Consumer Duty.

---

## Step 1 — Define the target market and intended use

**Data needed:** The product governance file (FG22/5 §4.3 requirements), distribution channel breakdown, and the most recent manufacturer/distributor target market agreement.

This is the starting point for everything that follows. If your target market definition is vague — "UK adults seeking motor insurance" — every downstream segmentation analysis will be inconclusive, because you have no anchor for what "the intended customer" looks like versus who is actually buying.

Be specific. Age band, vehicle type, NCD range, and channel. If you sell through PCWs and direct, those are different target markets for fair value purposes even if the product is nominally the same.

**Good:** Target market definition is documented at product level, not just at firm level, and has been reviewed since the last product launch or material change to the pricing model.

**Escalate:** Target market definition has not changed since Consumer Duty came into force in July 2023, despite material changes to underwriting appetite or distribution mix. Or: it is defined at firm level only, with no product-level specificity.

---

## Step 2 — Collect premium distribution data

**Data needed:** Policy-level data with written premium, technical premium (model output), channel, renewal year/cohort, and segment identifiers. Minimum 12 months. 24 months if you have had a model change.

```python
import polars as pl

policies = pl.read_parquet("policies_2025.parquet")

# Distribution of written premium by channel and renewal cohort
premium_dist = (
    policies
    .group_by(["channel", "cohort_year", "product"])
    .agg([
        pl.len().alias("policy_count"),
        pl.col("written_premium").mean().alias("avg_premium"),
        pl.col("written_premium").quantile(0.25).alias("p25"),
        pl.col("written_premium").median().alias("p50"),
        pl.col("written_premium").quantile(0.75).alias("p75"),
        pl.col("written_premium").quantile(0.95).alias("p95"),
    ])
    .sort(["product", "channel", "cohort_year"])
)
```

**Good:** Premium distribution is stable or moving in line with market rate changes. The spread (p75/p25) is consistent year on year within channels.

**Escalate:** P95 premium is growing faster than P50 within a channel — indicating a widening tail that may indicate differential pricing harm. Or: the distribution has a bimodal shape that cannot be explained by legitimate risk differentiation.

---

## Step 3 — Calculate fair value metrics

**Data needed:** Incurred claims by segment (paid + case reserves, minimum 24 months development), written premium by segment, exposure.

The core fair value metrics are loss ratio by segment, price dispersion, and the renewal vs new business price spread.

```python
# Loss ratio by segment
fair_value = (
    policies
    .join(claims_agg, on="policy_id", how="left")
    .group_by(["product", "channel", "segment"])
    .agg([
        pl.col("written_premium").sum().alias("gwp"),
        pl.col("incurred_claims").sum().alias("incurred"),
        pl.col("exposure").sum(),
    ])
    .with_columns([
        (pl.col("incurred") / pl.col("gwp")).alias("loss_ratio"),
    ])
)

# Price dispersion: coefficient of variation within segment
dispersion = (
    policies
    .group_by(["product", "segment"])
    .agg([
        (pl.col("written_premium").std() / pl.col("written_premium").mean())
        .alias("premium_cv"),
    ])
)

# New business vs renewal spread
nb_renewal = (
    policies
    .group_by(["product", "is_renewal"])
    .agg(pl.col("written_premium").mean().alias("avg_premium"))
    .pivot("is_renewal", index="product", values="avg_premium")
    .with_columns(
        (pl.col("true") / pl.col("false") - 1).alias("renewal_premium")
    )
)
```

**Good:** Loss ratios are within ±10 percentage points across segments for the same product. No segment has a loss ratio below 30% (implying systematic over-pricing) or above 90% (unsustainable). Renewal premium is not higher than new business for equivalent risk after PS21/5 compliance.

**Escalate:** Any segment with loss ratio below 30% or above 100%. Renewal-to-new-business spread exceeding 5% on an equivalent-risk basis. Coefficient of variation on premium above 0.35 within a segment (wide unexplained dispersion).

---

## Step 4 — Run proxy discrimination audit

**Data needed:** Policy-level features including all rating factors, plus any available protected characteristic data (age, gender if held, postcode as a proxy for ethnicity/deprivation). Predicted premium from the current champion model.

`insurance-fairness` runs the full discrimination audit: proxy detection (mutual information, partial correlation, SHAP-based proxy scores), calibration by group, disparate impact ratio, and demographic parity.

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=catboost_model,
    data=policies_with_features,
    protected_cols=["age_band", "postcode_deprivation_quintile"],
    prediction_col="predicted_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=["vehicle_group", "ncd_years", "area_code", "annual_mileage"],
)
report = audit.run()
report.summary()
report.to_markdown("fairness_audit_2026.md")
```

The proxy detection component (`proxy_detection.py`) computes mutual information and proxy R-squared between each rating factor and the protected characteristics. A proxy R-squared above 0.15 on `postcode_deprivation_quintile` for any factor warrants investigation — it means knowing that factor tells you a lot about which deprivation quintile the customer is in.

**Good:** No rating factor has proxy R-squared above 0.10 against any protected characteristic. Disparate impact ratio (adverse group mean premium / reference group mean premium) is below 1.20 after controlling for legitimate risk factors. Calibration loss ratios are within ±8 percentage points across age bands.

**Escalate:** Any factor with proxy R-squared above 0.15. Disparate impact ratio above 1.25 on an uncontrolled basis and above 1.15 after risk-adjustment. Calibration showing the model systematically under-predicts claims for a specific demographic group.

---

## Step 5 — Check for differential pricing harm — PS21/5 ENBP compliance

**Data needed:** Individual renewal quotes for existing customers, equivalent new business quotes for the same risk characteristics, and effective date of each quote.

PS21/5 (Pricing Practices) has been in force since January 2022, and we still see firms treating this as a one-time fix rather than an ongoing monitoring obligation. The requirement is clear: a renewing customer's premium must not exceed what an equivalent new customer would pay (the Equivalent New Business Price).

```python
# Calculate ENBP breach rate
enbp_check = (
    renewal_policies
    .with_columns([
        (pl.col("renewal_premium") - pl.col("enbp")).alias("enbp_gap"),
        (pl.col("renewal_premium") > pl.col("enbp")).alias("enbp_breach"),
    ])
    .group_by("product")
    .agg([
        pl.col("enbp_breach").mean().alias("breach_rate"),
        pl.col("enbp_gap").filter(pl.col("enbp_gap") > 0).mean().alias("avg_breach_amount"),
        pl.col("enbp_gap").filter(pl.col("enbp_gap") > 0).max().alias("max_breach_amount"),
        pl.len().alias("renewal_count"),
    ])
)
```

**Good:** Zero ENBP breaches. If your ENBP calculation methodology has any approximation in it (e.g. not all rating factor combinations can be directly equivalent), the breach rate should be below 0.1% with average breach amount below £5.

**Escalate:** Any breach rate above 0.5%. Any individual breach above £50. ENBP calculation methodology that has not been reviewed since the original PS21/5 implementation — the FCA's multi-firm review in 2024 found several firms using ENBP definitions that drifted from intent following model changes.

---

## Step 6 — Assess product-level outcomes data

**Data needed:** Claims acceptance rates by product and channel (last 12 months), complaints volumes and categories, mid-term adjustment (MTA) rates, and policy cancellation rates.

This is the step most pricing teams skip because the data is not in the pricing database. You need the claims and customer service data. Get it.

```python
outcomes = (
    policies
    .join(claims_outcomes, on="policy_id", how="left")
    .join(complaints, on="policy_id", how="left")
    .group_by(["product", "channel", "cohort_quarter"])
    .agg([
        pl.col("claim_accepted").mean().alias("claims_acceptance_rate"),
        pl.col("has_complaint").mean().alias("complaint_rate"),
        pl.col("had_mta").mean().alias("mta_rate"),
        pl.col("cancelled_mtd").mean().alias("cancellation_rate"),
        pl.len().alias("policy_count"),
    ])
)
```

FCA FG22/5 para 8.20 is explicit: claims acceptance rate that is systematically lower than expected is a red flag for product design that does not meet customer needs, which is a fair value problem, not just a conduct problem.

**Good:** Claims acceptance rate above 85%. Complaint rate below 0.5% of in-force policies. MTA rate stable year on year (large increases may indicate pricing that forces customers to adjust their stated risk to get a cheaper quote).

**Escalate:** Claims acceptance rate below 80%. Complaint rate above 1%. MTA rate that has increased more than 20% relative to the prior year without a clear operational explanation.

---

## Step 7 — Monitor for vulnerable customer impact

**Data needed:** Any vulnerability indicators available (payment frequency — monthly payers as a proxy for financial vulnerability, age — over-75s as a proxy for cognitive vulnerability, self-declared vulnerability flags if collected), and outcomes data from Step 6 stratified by these indicators.

Consumer Duty FG22/5 Chapter 11 specifically requires firms to consider vulnerable customers as a distinct group within fair value assessment. Monthly payers paying significantly more than annual equivalents — after adjusting for the credit cost — is a live issue in personal lines.

```python
# Segment outcome gaps for vulnerable proxies
vulnerable_outcomes = (
    policies
    .with_columns([
        (pl.col("age") >= 75).alias("older_customer"),
        (pl.col("payment_frequency") == "monthly").alias("monthly_payer"),
    ])
    .melt(
        id_vars=["policy_id", "claims_acceptance_rate", "complaint_rate",
                 "written_premium", "technical_premium"],
        value_vars=["older_customer", "monthly_payer"],
        variable_name="vulnerability_proxy",
        value_name="is_vulnerable",
    )
    .group_by(["vulnerability_proxy", "is_vulnerable"])
    .agg([
        pl.col("claims_acceptance_rate").mean(),
        pl.col("complaint_rate").mean(),
        (pl.col("written_premium") / pl.col("technical_premium")).mean()
        .alias("price_to_risk_ratio"),
    ])
)
```

**Good:** Outcome metrics for the vulnerable proxy group are within ±5 percentage points of the non-vulnerable group. Price-to-risk ratio for monthly payers is no more than 10% higher than for annual payers after accounting for the credit cost (which should be explicitly modelled and disclosed).

**Escalate:** Claims acceptance rate more than 10 percentage points lower for a vulnerable proxy group. Monthly payer premium more than 15% higher than annual equivalent after removing legitimate credit cost. Any outcome gap that has widened year on year.

---

## Step 8 — Run model drift checks

**Data needed:** Training score distribution from the last model fit, current in-force score distribution, out-of-sample actual vs expected data from the last 12 months.

A model that has drifted is, by definition, no longer pricing to risk. If it is over-predicting risk for a particular segment, that segment is being over-charged. That is a fair value problem, not just a technical one. `insurance-monitoring` runs PSI, CSI, and A/E to detect this.

```python
from insurance_monitoring.drift import psi, csi
from insurance_monitoring.calibration import ae_ratio
from insurance_monitoring import MonitoringReport

# PSI: score distribution stability
psi_result = psi(
    reference=train_scores,
    current=current_scores,
    n_bins=10,
)
print(f"PSI: {psi_result:.4f}")  # <0.10 good, 0.10-0.25 monitor, >0.25 escalate

# CSI: characteristic stability for key rating factors
csi_results = {
    feat: csi(
        reference=train_data[feat].to_numpy(),
        current=current_data[feat].to_numpy(),
    )
    for feat in ["vehicle_age", "driver_age", "ncd_years", "annual_mileage"]
}

# A/E ratio: calibration check
ae = ae_ratio(
    actual=validation_claims,
    predicted=model_predictions,
    exposure=validation_exposure,
)
print(f"A/E ratio: {ae:.3f}")  # Target 0.95-1.05

# Combined monitoring report
# MonitoringReport is a dataclass: pass reference and current period arrays.
# It runs all checks automatically in __post_init__; no .run() call needed.
report = MonitoringReport(
    reference_actual=train_claims,
    reference_predicted=train_predictions,
    current_actual=validation_claims,
    current_predicted=model_predictions,
    exposure=validation_exposure,
)
print(report.recommendation)   # 'NO_ACTION' | 'RECALIBRATE' | 'REFIT' | 'INVESTIGATE'
print(report.to_dict())        # full metrics dict with traffic-light bands
```

**Good:** PSI below 0.10 on overall score distribution. CSI below 0.10 on all material rating factors. A/E ratio between 0.95 and 1.05 at aggregate level.

**Escalate:** PSI above 0.25 (indicates substantial shift in the scored population — the model is now being applied to risks it was not trained on). Any single feature CSI above 0.25. A/E ratio outside 0.90-1.10 at aggregate, or outside 0.85-1.15 for any product line. Gini drift more than 3 percentage points below the training-set baseline.

---

## Step 9 — Validate that reserve adequacy is not cross-subsidising pricing

**Data needed:** IBNR reserve estimates by product line, ultimate loss ratios from reserving actuaries, and the technical loss ratios used in pricing model training.

This step requires a conversation between the pricing and reserving teams that does not happen often enough. If reserves are being held at a level that implies a materially different ultimate loss ratio than the pricing basis, one of them is wrong — and if pricing is wrong, customers in that product line are being mispriced.

```python
# Cross-reference pricing basis loss ratios with reserving basis
reserve_vs_pricing = (
    pl.DataFrame({
        "product": ["motor_comp", "motor_tpft", "home_buildings", "home_contents"],
        "pricing_loss_ratio": [0.68, 0.72, 0.61, 0.58],   # from pricing model training data
        "reserving_ultimate_lr": [0.71, 0.75, 0.65, 0.54],  # from last reserve review
    })
    .with_columns([
        (pl.col("reserving_ultimate_lr") - pl.col("pricing_loss_ratio"))
        .alias("basis_gap"),
        (
            (pl.col("reserving_ultimate_lr") - pl.col("pricing_loss_ratio")).abs()
            > 0.05
        ).alias("material_gap"),
    ])
)
```

**Good:** Basis gap below ±5 percentage points. The gap has been reviewed and signed off by both pricing and reserving leads within the last 12 months.

**Escalate:** Basis gap above 10 percentage points in either direction. Reserving basis that implies significantly lower ultimate loss ratios than pricing — this means customers are being charged a margin that is not visible in the pricing model and cannot be justified as a legitimate expense. This is where cross-subsidy becomes a fair value issue, not just a reserving issue.

---

## Step 10 — Document remediation actions taken since last assessment

**Data needed:** The prior year's fair value assessment, action log with owners and target dates, and evidence of completion.

The FCA does not just want to know that problems were found. It wants to know that the firm took action and that the action worked. The 2024 multi-firm review explicitly called out firms that produced fair value assessments listing the same issues year after year with no evidence of resolution.

Structure the remediation log as a structured table, not a narrative. For each action: what triggered it, what was done, when, and what the post-remediation metric looked like.

```python
remediation_log = pl.DataFrame({
    "issue": [
        "Monthly payer premium gap exceeding 15% after credit adjustment",
        "ENBP calculation not updated following model refactor in Q2 2025",
    ],
    "triggered_by": ["FVA Step 7 - Sept 2025", "FVA Step 5 - Sept 2025"],
    "action": [
        "Repriced monthly payment loading from 12% to 8% effective Nov 2025",
        "ENBP logic reviewed and patched in PS21/5 compliance run Nov 2025",
    ],
    "completed": ["2025-11-01", "2025-11-15"],
    "post_remediation_metric": [
        "Monthly payer gap: 9.2% (within threshold)",
        "ENBP breach rate: 0.04% (within threshold)",
    ],
    "status": ["CLOSED", "CLOSED"],
})
```

**Good:** All actions from the prior year assessment have a completion date and a measured outcome. No action has been open for more than 12 months.

**Escalate:** Any action from the prior year assessment that is still open with no completion date. Actions that were marked complete but where the post-remediation metric was not measured. Recurring issues that appeared in two or more consecutive assessments.

---

## Step 11 — Generate the audit trail

**Data needed:** All outputs from Steps 1-10, the current champion model and its version, training data vintage, and the sign-off chain.

`insurance-governance` generates a self-contained HTML report and JSON sidecar from your validation and monitoring outputs. For Consumer Duty fair value, wrap this with the fairness audit from Step 4 and the monitoring outputs from Step 8.

```python
from insurance_governance.validation.validation_report import ModelValidationReport
from insurance_governance.validation.model_card import ModelCard
from insurance_governance.mrm.report import GovernanceReport
from insurance_governance.mrm.model_card import ModelCard as MRMCard

# Validation report (SS1/23-aligned)
card = ModelCard(
    name="Motor Frequency v4.1",
    version="4.1.0",
    purpose="Predict claim frequency for UK motor portfolio — champion model",
    methodology="CatBoost with Poisson objective, monotonicity constraints on NCD",
    target="claim_count",
    features=["driver_age", "vehicle_group", "area_code", "ncd_years", "annual_mileage"],
    limitations=["No telematics data", "Postcode is coarsest geographic unit"],
    owner="Pricing Team",
)

report = ModelValidationReport(
    y_train=y_train,
    y_pred_train=y_pred_train,
    y_val=y_val,
    y_pred_val=y_pred_val,
    exposure_val=exposure_val,
    X_train=X_train,
    X_val=X_val,
    model_card=card,
)

report.generate("validation_report_2026.html")
report.to_json("validation_report_2026.json")

# Governance report for MRC pack
governance = GovernanceReport(
    card=card,
    validation_results=report.to_dict()["summary"],
    monitoring_results={
        "period": "2025-Q4",
        "ae_ratio": 1.02,
        "psi_score": 0.07,
        "recommendation": "No action required",
        "triggered_alerts": [],
    },
    report_date="2026-03-25",
)
governance.to_html("governance_report_2026.html")
```

The JSON sidecar is the machine-readable audit trail. Store it in version control alongside the model artefacts. If the FCA asks for evidence that the model was validated before a pricing change, a timestamped JSON with pass/fail test results is far stronger than a Word document.

**Good:** HTML report with no RED sections. JSON sidecar committed to the model registry with the same timestamp as the model artefact. Governance report signed off by Chief Actuary and Head of Pricing before submission to the board.

**Escalate:** Any RED section in the validation report that has not been resolved before board submission. Missing JSON sidecar for any model in the champion set. Governance report not reviewed by both pricing and actuarial leadership.

---

## Step 12 — Board-ready summary with RAG status per product line

**Data needed:** Outputs from all 11 preceding steps, consolidated by product line.

The board summary is not a data dump. It is a RAG table that tells the board — in one page — whether each product line is delivering fair value, and what needs to happen if not. The RAG methodology should be documented and consistent year on year so the board can see whether things are improving.

```python
board_summary = pl.DataFrame({
    "product": [
        "Motor Comprehensive",
        "Motor TPFT",
        "Home Buildings",
        "Home Contents",
    ],
    "loss_ratio_rag": ["GREEN", "GREEN", "AMBER", "GREEN"],
    "proxy_discrimination_rag": ["GREEN", "GREEN", "GREEN", "GREEN"],
    "enbp_compliance_rag": ["GREEN", "AMBER", "GREEN", "GREEN"],
    "vulnerable_customer_rag": ["GREEN", "GREEN", "AMBER", "GREEN"],
    "model_drift_rag": ["GREEN", "GREEN", "GREEN", "GREEN"],
    "outcomes_data_rag": ["GREEN", "GREEN", "AMBER", "GREEN"],
    "overall_rag": ["GREEN", "AMBER", "AMBER", "GREEN"],
    "key_action": [
        "None required",
        "Review ENBP calculation for 3-year NCD cohort — by April 2026",
        "Claims acceptance rate 79% — product design review initiated",
        "None required",
    ],
    "action_owner": [
        "-",
        "Pricing Lead",
        "Product & Pricing",
        "-",
    ],
    "action_due": ["-", "2026-04-30", "2026-06-30", "-"],
})

# Overall firm-level RAG
n_red = (board_summary["overall_rag"] == "RED").sum()
n_amber = (board_summary["overall_rag"] == "AMBER").sum()
firm_rag = "RED" if n_red > 0 else ("AMBER" if n_amber > 0 else "GREEN")
print(f"Firm-level fair value RAG: {firm_rag}")
```

**Good:** Firm-level RAG is GREEN or AMBER with a clear action plan. Every AMBER or RED item has a named owner and a due date within the next reporting cycle. The board summary includes a year-on-year comparison to show direction of travel.

**Escalate:** Any product with an overall RED rating must be reported to the board within 30 days with a remediation timeline. A firm-level RED requires escalation to the FCA under the Consumer Duty notification framework. Any AMBER that has been carried from the prior year without a completed action becomes a RED for reporting purposes.

---

## Running this as a workflow

These 12 steps are not independent. Run them in order. Steps 4, 5, and 7 (proxy discrimination, ENBP, vulnerable customer) all depend on the segment definitions from Step 1. Step 8 (model drift) must run before Step 9 (reserve cross-subsidy) because a drifted model changes the interpretation of any basis gap. Step 11 (audit trail generation) is a dependent of everything before it.

The annual cycle for a team running a quarterly monitoring cadence looks like this:

- Q1-Q3: Monthly PSI/A/E monitoring (Step 8 subset). Quarterly outcomes data update (Step 6).
- Q4: Full 12-step assessment, targeting board submission by February.
- Continuous: ENBP compliance (Step 5) on every renewal cycle, not annually.

The FCA's outcomes monitoring review in 2024 was clear that Consumer Duty is not a once-a-year exercise. The annual fair value assessment is a formalisation of what should be a continuous monitoring posture. If you are running this checklist for the first time in January each year, you have already missed several months of data that could have informed remediation.

---

All code in this post uses `insurance-fairness`, `insurance-monitoring`, and `insurance-governance` — all open source under MIT at [github.com/burning-cost](https://github.com/burning-cost). Requires Python 3.10+, polars 0.20+, and catboost 1.2+ for the fairness audit.

**Related posts:**
- [Your Model Validation Is a Checklist, Not a Test](/2025/07/13/model-validation-pra-ss123/) — PRA SS1/23 and the full technical validation suite
- [Does Proxy Discrimination Testing Actually Work?](/2026/03/28/does-proxy-discrimination-testing-actually-work/) — empirical evaluation of the insurance-fairness proxy detection methods
- [Does Automated Model Monitoring Actually Work?](/2026/03/27/does-automated-model-monitoring-actually-work/) — PSI and A/E in practice on real insurance drift scenarios
