---
layout: post
title: "Consumer Duty Fair Value Evidencing: A 12-Step Technical Checklist for Pricing Actuaries (2026)"
date: 2026-03-25
featured: true
author: Burning Cost
categories: [compliance, fairness, monitoring, tutorials]
description: "FCA PS24/1 confirms enhanced Consumer Duty requirements from April 2026. EP25/2 flags ongoing fair value supervision in motor and home. No single technical checklist exists for the pricing actuary's portion of the annual fair value assessment. Here is one."
tags: [FCA, Consumer-Duty, PS24-1, EP25-2, PRIN-2A, fair-value, fairness, monitoring, calibration, proxy-discrimination, insurance-monitoring, insurance-fairness, insurance-conformal, pricing-actuary, compliance, uk-insurance, tutorial]
---

The FCA confirmed enhanced Consumer Duty requirements in PS24/1 with April 2026 as the effective date. EP25/2 (the FCA's 2026 Insurance Supervision priorities, published February 2025) explicitly names ongoing fair value supervision across motor and home as a standing priority. TR24/2 in August 2024 described the evidence most firms were producing for their annual fair value assessments as "high-level summaries with little substance."

That gap between what firms are submitting and what the FCA expects has not closed. What is missing is not intent — it is a concrete technical process that a pricing actuary can execute, document, and defend. This post provides one.

The checklist below covers the pricing actuary's portion of the annual fair value assessment. It is not the complete Consumer Duty assessment (that includes distribution chain, board sign-off, product governance, and outcomes monitoring). It is the quantitative modelling and analysis that underpins the sections of the assessment that actuaries own: price adequacy, calibration evidence, discrimination monitoring, and model uncertainty quantification.

All code uses the `insurance-monitoring`, `insurance-fairness`, and `insurance-conformal` libraries.

---

## Before you start: what PRIN 2A actually requires

PRIN 2A.4 (the Price and Value Outcome) requires firms to charge prices that represent fair value. Critically, this is an ongoing obligation, not a point-in-time assessment. PRIN 2A.4.6 specifically requires firms to monitor outcomes and have a process for identifying when their pricing is no longer delivering fair value.

The FCA has not published a prescriptive method for how pricing actuaries should discharge this. PS24/1 adds the April 2026 enhanced requirements for annual assessment frequency and board reporting. EP25/2 signals that supervisors will be examining actual evidence packs, not just process descriptions.

The 12 steps below map to the evidence a supervisor would want to see.

---

## Step 1: Document the model scope and current production version

**Requirement:** Identify which models are in scope (frequency, severity, demand, uplift), their production version numbers, training data dates, and the last validation date. The fair value assessment must be tied to the model actually in production, not a hypothetical.

**Why it matters:** TR24/2 found fair value assessments that were disconnected from production models. If a model was updated in November and the fair value assessment used October predictions, the evidence is worthless. Version tracking is the foundation everything else rests on.

**Code:**

```python
import polars as pl
from datetime import date

# Document your production model registry — adapt to your versioning system
model_registry = pl.DataFrame({
    "model_name": ["Motor Frequency v4.2", "Motor Severity v3.1", "Demand Elasticity v1.4"],
    "model_type": ["frequency", "severity", "demand"],
    "training_data_end": [date(2025, 6, 30), date(2025, 6, 30), date(2025, 3, 31)],
    "production_since": [date(2025, 9, 1), date(2025, 8, 1), date(2025, 4, 1)],
    "last_validation_date": [date(2025, 8, 15), date(2025, 7, 28), date(2025, 3, 20)],
    "certifying_actuary": ["J. Smith", "J. Smith", "P. Kumar"],
})

print(model_registry)
# Save to evidence pack
model_registry.write_csv("evidence/01_model_registry.csv")
```

The evidence file is your anchor. Every subsequent step should reference the model version recorded here.

---

## Step 2: Verify the A/E ratio is within acceptable bounds — overall and segmented

**Requirement:** Compute the Actual/Expected ratio for both frequency and severity models. An aggregate A/E is not sufficient — segment by at least vehicle group, region, and NCD band to identify whether miscalibration is distributed or concentrated.

**Why it matters:** A portfolio-level A/E of 1.02 with a vehicle-group A/E of 1.35 for one segment means you are systematically underpricing that segment. The aggregate masks it. PRIN 2A.4.6 requires outcome monitoring, and a 35% underpricing is a fair value failure for affected customers.

A/E should use adequately developed claims only — 12-month minimum development lag for motor frequency, longer for severity. Check this explicitly before running the calculation.

**Code:**

```python
import numpy as np
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# Frequency model — use claims at adequate development only
# actual: observed claim counts; predicted: model frequency rate; exposure: earned car-years
ae_overall = ae_ratio(actual=claims_count, predicted=freq_predicted, exposure=exposure)
ae_ci = ae_ratio_ci(actual=claims_count, predicted=freq_predicted,
                    exposure=exposure, alpha=0.05, method="poisson")

print(f"Frequency A/E: {ae_overall:.4f}  95% CI: [{ae_ci['lower']:.4f}, {ae_ci['upper']:.4f}]")

# Segmented A/E by vehicle group — surface concentrated miscalibration
ae_by_vehicle = ae_ratio(
    actual=claims_count,
    predicted=freq_predicted,
    exposure=exposure,
    segments=vehicle_group,
)
print(ae_by_vehicle)

# Flag segments outside ±10% tolerance
flagged = ae_by_vehicle.filter(
    (pl.col("ae_ratio") < 0.90) | (pl.col("ae_ratio") > 1.10)
)
print(f"\n{len(flagged)} segments outside ±10% A/E tolerance")
```

Thresholds: A/E outside 0.90–1.10 is amber; outside 0.85–1.15 is red and requires documented justification or a recalibration plan.

---

## Step 3: Run the full Murphy decomposition to distinguish miscalibration from model weakness

**Requirement:** Decompose the proper scoring rule into Calibration, Resolution, and Uncertainty components (Murphy decomposition). This tells you whether poor performance reflects a calibration shift (fixable by recalibration) or a genuine model weakness (requires refit).

**Why it matters:** The FCA expects firms to understand *why* a model is underperforming, not just that it is. A recalibration applied to a model with degraded discrimination (low Resolution) will not restore fair value — it will just shift the intercept on a model that is no longer ranking risks correctly. The Murphy decomposition prevents this misdiagnosis.

**Code:**

```python
from insurance_monitoring.calibration import murphy_decomposition, CalibrationChecker

# Murphy decomposition: MCB (miscalibration), DSC (discrimination/resolution), UNC
murphy = murphy_decomposition(
    actual=claims_count,
    predicted=freq_predicted,
    distribution="poisson",
)
print(f"MCB (miscalibration): {murphy.mcb:.6f}")
print(f"DSC (resolution):     {murphy.dsc:.6f}")
print(f"UNC (uncertainty):    {murphy.unc:.6f}")
print(f"Score:                {murphy.score:.6f}")

# CalibrationChecker runs balance + autocal + Murphy in one pipeline
checker = CalibrationChecker(distribution="poisson", alpha=0.05)
checker.fit(y=claims_count, y_hat=freq_predicted, exposure=exposure)
report = checker.check(y=claims_count, y_hat=freq_predicted, exposure=exposure)
print(report.verdict())  # "OK" | "MONITOR" | "RECALIBRATE" | "REFIT"
```

If the verdict is `REFIT`, the pricing actuary's evidence pack must include a timeline for model redevelopment. A `RECALIBRATE` verdict is lower urgency but still requires documented action with a deadline.

---

## Step 4: Test for calibration stability across the monitoring period using PITMonitor

**Requirement:** Demonstrate that calibration has been stable throughout the monitoring period, not just at the point-in-time assessment date. Use anytime-valid sequential monitoring so that interim checks do not inflate false positive rates.

**Why it matters:** A/E ratios computed at year-end hide in-year deteriorations. If your motor frequency model was well-calibrated in January but drifted materially by September, a December A/E of 1.04 tells you nothing useful. PRIN 2A.4.6 requires ongoing monitoring — that means demonstrated process, not a single annual number.

`PITMonitor` uses probability integral transforms and mixture e-processes (Henzi, Murph & Ziegel, 2025) to provide a guarantee: P(ever alarm | model calibrated) ≤ α, for all time, forever. Standard monthly H-L or A/E checks do not provide this guarantee.

**Code:**

```python
from insurance_monitoring import PITMonitor
from scipy.stats import poisson as sp_poisson

# PITMonitor takes PIT values (floats in [0,1]) computed from your
# predictive distribution — not raw actuals and predicted rates.
monitor = PITMonitor(alpha=0.05, rng=42)

# Compute Poisson PITs for each policy and feed monthly batches.
# (Run this loop monthly during the year; here we simulate 12 months.)
for month_idx, (claims_month, lambda_hat_month, exp_month) in enumerate(monthly_batches):
    pits = [
        float(sp_poisson.cdf(int(y_i), mu=exp_i * lhat_i))
        for y_i, lhat_i, exp_i in zip(claims_month, lambda_hat_month, exp_month)
    ]
    alarm = monitor.update_many(pits)
    if alarm.triggered:
        print(f"Month {month_idx + 1}: ALARM — calibration shift detected")
        print(f"  Evidence: {alarm.evidence:.3f}, threshold: {alarm.threshold:.1f}")
        print(f"  Estimated changepoint: step {alarm.changepoint}")
        break

summary = monitor.summary()
print(f"Monitoring period: {summary.n_observations} observations, "
      f"evidence: {summary.evidence:.3f}, "
      f"alarm triggered: {summary.alarm_triggered}")
```

Record the `PITMonitor` summary in the evidence pack as continuous monitoring evidence. If no alarm triggered, that is positive evidence of calibration stability — not just an absence of bad news.

---

## Step 5: Check PSI and CSI for input feature drift

**Requirement:** Measure Population Stability Index (PSI) for all rating factors and Characteristic Stability Index (CSI) for model predictions. Drift in inputs that is not reflected in model output is a latent fair value risk.

**Why it matters:** EP25/2 explicitly flags book mix changes as a supervisory concern. If your motor book has shifted towards younger drivers (PSI > 0.25 on driver age) but your frequency model was trained on an older distribution, the model's implied pricing relativities are no longer appropriate for the current book. This is a fair value issue independent of the current A/E ratio.

**Code:**

```python
from insurance_monitoring.drift import psi, csi

# PSI for each rating factor — reference = training window, current = monitoring period
rating_factors = ["driver_age", "vehicle_group", "ncd_years", "postcode_district"]
psi_results = {}

for factor in rating_factors:
    psi_val = psi(
        reference=training_data[factor].to_numpy(),
        current=monitoring_data[factor].to_numpy(),
        n_bins=10,
    )
    psi_results[factor] = psi_val
    status = "GREEN" if psi_val < 0.10 else ("AMBER" if psi_val < 0.25 else "RED")
    print(f"{factor:25s}  PSI={psi_val:.4f}  [{status}]")

# CSI for model predictions (score drift)
csi_score = csi(
    reference=training_data["predicted_freq"].to_numpy(),
    current=monitoring_data["predicted_freq"].to_numpy(),
    n_bins=10,
)
print(f"\nPrediction CSI: {csi_score:.4f}")
```

Standard thresholds: PSI < 0.10 (no significant change), 0.10–0.25 (moderate drift, investigate), > 0.25 (significant drift, action required). Factors with PSI > 0.25 must be documented with an explanation and remediation plan.

---

## Step 6: Run GiniDrift test to confirm discrimination power is maintained

**Requirement:** Test whether the model's ranking power (Gini coefficient) has changed between the reference period and the current monitoring period. A model with stable A/E but declining Gini is no longer differentiating risk correctly — it is repricing to a flatter tariff structure.

**Why it matters:** Declining discrimination means some risks are being overcharged and others undercharged, even if the aggregate is correct. This is the definition of unfair pricing at the individual level. The Gini drift test (Wüthrich, Merz & Noll 2025, arXiv:2510.04556) provides a statistically grounded test rather than a visual inspection of Lorenz curves.

**Code:**

```python
from insurance_monitoring import GiniDriftTest

# Two-sample test: reference period vs current monitoring period
gini_test = GiniDriftTest(
    reference_actual=reference_actual,
    reference_predicted=reference_predicted,
    monitor_actual=monitor_actual,
    monitor_predicted=monitor_predicted,
    reference_exposure=reference_exposure,
    monitor_exposure=monitor_exposure,
    n_bootstrap=500,
    alpha=0.05,
)

result = gini_test.test()
print(f"Reference Gini:  {result.gini_reference:.4f}")
print(f"Monitor Gini:    {result.gini_monitor:.4f}")
print(f"Delta:           {result.delta:+.4f}")
print(f"z-statistic:     {result.z_statistic:.3f}")
print(f"p-value:         {result.p_value:.4f}")
print(f"Significant:     {result.significant}")

# Governance-ready paragraph for the evidence pack
print("\n" + gini_test.summary())
```

A significant result (`result.significant = True`) with negative `delta` means ranking power has declined — this requires documented investigation. A model that is increasingly flat is increasingly unfair to low-risk policyholders who are subsidising high-risk ones.

---

## Step 7: Run drift attribution to identify which factors explain performance change

**Requirement:** When A/E or Gini has deteriorated, identify which rating factors are driving the change. Generic PSI for individual factors does not capture interaction effects. Use feature-interaction-aware drift attribution.

**Why it matters:** TRIPODD (Panda et al. 2025, arXiv:2503.06606) identifies the minimal subset of features that explains model performance drift, with Bonferroni or FDR-corrected Type I error control. This is what a supervisor would want to see: not "PSI is high on driver age" but "the interaction between driver age and vehicle group accounts for 73% of the observed performance decline."

**Code:**

```python
from insurance_monitoring.interpretable_drift import InterpretableDriftDetector

# InterpretableDriftDetector: upgraded TRIPODD with exposure weighting and FDR control
detector = InterpretableDriftDetector(
    model=production_model,
    features=["driver_age", "vehicle_group", "ncd_years", "vehicle_age"],
    error_control="fdr",       # Benjamini-Hochberg for d >= 10 factors
    loss="poisson_deviance",   # canonical GLM loss for frequency models
    n_bootstrap=200,
)

detector.fit_reference(
    X=reference_X,
    y=reference_y,
    weights=reference_exposure,
)

result = detector.test(
    X=monitor_X,
    y=monitor_y,
    weights=monitor_exposure,
)

print("Significant drift factors:")
for factor in result.attributed_features:
    print(f"  {factor}")

print(f"\nDrift attribution summary:")
print(result.summary())
```

The output tells the evidence pack reader not just that drift exists but which factors to investigate and whether the drift is likely driven by distribution shift (a data problem) or genuine risk change (a model problem).

---

## Step 8: Run proxy discrimination audit — `ProxyDiscriminationAudit`

**Requirement:** Test whether the pricing model indirectly discriminates against customers sharing protected characteristics (age, gender, disability, ethnicity — Equality Act 2010, Section 19). This is not optional under PRIN 2A: the FCA has been explicit since TR24/2 that indirect discrimination through proxies is a live supervisory concern.

**Why it matters:** A model that does not use gender as a rating factor can still discriminate by gender if vehicle group, occupation, or postcode are correlated with gender. `ProxyDiscriminationAudit` computes D_proxy — a normalised L2-distance from the fitted price to the admissible (discrimination-free) price set (Lindholm, Richman, Tsanakas & Wüthrich, EJOR 2026) — and decomposes it across rating factors via Shapley effects. A D_proxy above 0.05 warrants investigation.

**Code:**

```python
import polars as pl
from insurance_fairness.diagnostics import ProxyDiscriminationAudit

# X must contain all rating factors; sensitive_col is the protected attribute
# y: observed pure premium or claim frequency
audit = ProxyDiscriminationAudit(
    model=pricing_model,
    X=policy_df,
    y=policy_df["claim_cost"],
    sensitive_col="gender",
    rating_factors=["postcode_district", "occupation_band", "vehicle_group"],
    exposure_col="exposure",
)

result = audit.fit()

print(f"D_proxy: {result.d_proxy:.4f} ({result.rag})")
print("\nShapley attribution (which factors drive proxy discrimination):")
for factor, effect in result.shapley_effects.items():
    print(f"  {factor}: phi={effect.phi:.3f} ({effect.rag})")
```

A proxy vulnerability above 0.05 (5% mean absolute gap between aware and unaware premiums) should be flagged in the fair value assessment with analysis of which factors are driving it. The `proxy_free` benchmark additionally shows how much of the vulnerability survives even when known proxies are removed — the residual is harder to explain away.

---

## Step 9: Run double fairness audit — action fairness vs outcome fairness

**Requirement:** Distinguish between pricing-time fairness (equal treatment at point of quote) and outcome fairness (equivalent value delivered over the policy lifetime). PRIN 2A Outcome 4 requires the latter. These can diverge significantly.

**Why it matters:** The empirical finding from Bian, Wang, Shi & Qi (2026, arXiv:2601.19186) is that equalising premiums across gender groups (action fairness) does not equalise loss ratios (outcome fairness). A firm auditing only action fairness may still fail Consumer Duty. `DoubleFairnessAudit` recovers the Pareto front across both dimensions so you can document the trade-off and justify where on the frontier your current pricing sits.

**Code:**

```python
from insurance_fairness import DoubleFairnessAudit

audit = DoubleFairnessAudit(n_alphas=20)
audit.fit(
    X_train=X_train_features,    # features excluding protected attribute
    y_premium=observed_premium,   # pricing outcome
    y_loss_ratio=observed_lr,     # claims / premium (value outcome)
    S_gender=gender_indicator,    # binary protected group indicator
)

result = audit.audit()
print(result.summary())

# FCA evidence section — paste directly into the fair value assessment
print(audit.report())

# Plot the Pareto front for the board paper appendix
fig = audit.plot_pareto()
fig.savefig("evidence/09_double_fairness_pareto.png", dpi=150)
```

The `audit.report()` output is formatted to slot into an evidence pack: it states the action fairness gap (Delta_1), outcome fairness gap (Delta_2), the recommended operating point, and the trade-off description. If you are operating at a Pareto-dominated point, the report says so.

---

## Step 10: Compute per-policyholder proxy vulnerability scores for the worst-affected segments

**Requirement:** Beyond the aggregate proxy vulnerability score from Step 8, identify which individual segments of your book are most exposed to proxy discrimination. This maps to the PRIN 2A requirement to monitor outcomes at the group level.

**Why it matters:** A portfolio-level proxy vulnerability of 0.03 may hide a specific segment — say, females in certain postcodes — with vulnerability of 0.25. Those policyholders are the FCA's primary concern. `ProxyVulnerabilityScore` computes per-observation proxy vulnerability and supports segmentation by any combination of rating factors.

**Code:**

```python
from insurance_fairness import (
    ProxyVulnerabilityScore,
    partition_by_proxy_vulnerability,
)

# Requires pre-computed aware and unaware premium columns
pvs = ProxyVulnerabilityScore(
    df=policy_df,
    sensitive_col="gender",
    aware_col="h_aware",
    unaware_col="h_unaware",
    exposure_col="exposure",
)

result = pvs.compute()
print(f"Portfolio vulnerability:  {result.portfolio_vulnerability:.4f}")
print(f"High-risk policyholders: {result.n_high_vulnerability} "
      f"({result.pct_high_vulnerability:.1%} of book)")

# Identify the worst-affected segments by any grouping variable
partitioned = partition_by_proxy_vulnerability(
    df=policy_df,
    aware_col="h_aware",
    unaware_col="h_unaware",
    protected_col="gender",
    segment_cols=["postcode_district", "vehicle_group"],
    exposure_col="exposure",
)
print(partitioned.head(10))   # Segments ranked by mean vulnerability
```

The output of this step is the policyholder impact analysis that a supervision team would request. "We identified X policyholders in [segment] with proxy vulnerability above 0.10, representing Y% of premium income" is the sentence the evidence pack needs.

---

## Step 11: Quantify model uncertainty with conformal prediction intervals — and bound expected shortfall

**Requirement:** Provide distribution-free uncertainty quantification for predicted premiums. The fair value assessment should state the uncertainty inherent in the pricing, not just point estimates. Use `PremiumSufficiencyController` to bound the expected shortfall from underpriced policies.

**Why it matters:** A pricing model that systematically underestimates uncertainty will underprice volatile risks. This is both a fair value issue (customers with high-variance claims are getting below-adequate cover) and a solvency concern. Conformal risk control (Angelopoulos et al. 2024, ICLR) provides a finite-sample guarantee on expected shortfall that a standard confidence interval does not.

**Code:**

```python
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.risk import PremiumSufficiencyController

# Step A: conformal prediction intervals on the calibration set
cp = InsuranceConformalPredictor(
    model=fitted_tweedie_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
)
cp.calibrate(X_cal, y_cal, exposure=exposure_cal)

intervals = cp.predict_interval(X_test, alpha=0.10)   # 90% coverage guarantee
print(f"Empirical coverage: {intervals['coverage']:.3f}")  # should be >= 0.90

# Step B: bound expected shortfall — the risk control guarantee
# B = maximum possible loss / minimum premium — set this correctly for your book
B_max = float(np.max(y_cal / premium_cal))
psc = PremiumSufficiencyController(alpha=0.05, B=B_max)
psc.calibrate(y_cal=y_cal, premium_cal=premium_cal)

print(f"Calibrated loading factor (lambda*): {psc.lambda_hat_:.3f}")
print(f"Interpretation: expected shortfall <= 5% of premium income "
      f"when premiums are multiplied by {psc.lambda_hat_:.3f}")

# Upper bounds for the monitoring period
risk_bounds = psc.predict(premium_new=premium_test)
print(risk_bounds.head())
```

The `lambda_hat_` value is the guaranteed loading factor. Document it as: "The conformal risk control loading of λ* = 1.24 provides a finite-sample guarantee that the expected claims shortfall above the loaded premium does not exceed 5% of premium income."

---

## Step 12: Generate the evidence pack and version-lock it

**Requirement:** Consolidate all outputs into a versioned, dated evidence pack that names the certifying actuary, the model versions, the monitoring period, and the regulatory mapping. This is the document the FCA would request under s166 or equivalent.

**Why it matters:** TR24/2 found evidence packs that were undated, not reproducible, and contained no regulatory mapping. An evidence pack that does not explicitly state which requirement each section addresses gives a supervisor no path through it. Include explicit mappings to PRIN 2A.4.6, PS24/1, and EP25/2.

**Code:**

```python
from datetime import date

evidence_pack = {
    "assessment_date": date.today().isoformat(),
    "monitoring_period": "2025-01-01 to 2025-12-31",
    "certifying_actuary": "J. Smith FIA",
    "models_in_scope": model_registry.to_dicts(),

    # Step 2
    "frequency_ae": {"overall": ae_overall, "ci_95": ae_ci,
                     "segments": ae_by_vehicle.to_dicts()},
    # Step 3
    "murphy_decomposition": {"mcb": murphy.mcb, "dsc": murphy.dsc,
                              "unc": murphy.unc, "verdict": report.verdict()},
    # Step 4
    "calibration_stability": {"pit_alarm_triggered": summary.alarm_triggered,
                               "evidence": summary.evidence,
                               "n_observations": summary.n_observations},
    # Step 5
    "psi_results": psi_results,
    # Step 6
    "gini_drift": {"gini_reference": result.gini_reference,
                    "gini_monitor": result.gini_monitor,
                    "delta": result.delta, "p_value": result.p_value},
    # Step 7
    "drift_attribution": {"attributed_features": result.attributed_features},
    # Step 8
    "proxy_vulnerability": {"d_proxy": result.d_proxy},
    # Step 9
    "double_fairness": {"action_gap": "see evidence/09_double_fairness_pareto.png"},
    # Step 10
    "worst_segments": partitioned.head(5).to_dicts(),
    # Step 11
    "model_uncertainty": {"lambda_star": psc.lambda_hat_,
                           "shortfall_bound_pct": 5.0},

    "regulatory_mapping": {
        "PRIN_2A_4_6": "Steps 2, 4, 8, 9, 10 — ongoing outcome monitoring",
        "PS24_1":      "Annual assessment frequency and board reporting (all steps)",
        "EP25_2":      "Motor and home fair value supervision priority (Steps 5, 6, 7)",
    },
}

import json
with open("evidence/fair_value_assessment_2026.json", "w") as f:
    json.dump(evidence_pack, f, indent=2, default=str)

print("Evidence pack written: evidence/fair_value_assessment_2026.json")
```

The JSON output is machine-readable, version-lockable with git, and reviewable by a non-technical supervisor without losing the quantitative detail. Commit it alongside your model artefacts at the point of sign-off.

---

## The steps that matter most

If you cannot run all 12 before the April 2026 deadline, prioritise in this order:

**Highest FCA visibility:** Steps 2 (A/E with CIs), 8 (indirect discrimination), and 12 (evidence pack). These are the three areas EP25/2 most directly signals.

**Highest risk of material finding:** Steps 3 (Murphy decomposition) and 9 (double fairness). These are the tests most likely to reveal that a firm believes it is compliant while actually failing Consumer Duty outcome monitoring.

**Highest regulatory novelty:** Steps 4 (PITMonitor), 6 (GiniDrift), and 11 (conformal risk control). These provide guarantees that standard actuarial monitoring cannot. The FCA's TR24/2 criticism of "high-level summaries" is most effectively countered by methods that come with finite-sample statistical guarantees — not just expert judgement applied to bar charts.

---

## Installation

```bash
uv add insurance-monitoring insurance-fairness insurance-conformal
```

All three libraries require Python 3.10+ and have no heavy dependencies beyond numpy, scipy, polars, and pandas. CatBoost and LightGBM are optional backends — the fairness and conformal libraries fall back to sklearn's GradientBoostingRegressor if neither is installed.

The libraries are at:
- [`insurance-monitoring`](/insurance-monitoring/) — PSI, CSI, A/E, GiniDrift, PITMonitor, TRIPODD
- [`insurance-fairness`](/insurance-fairness/) — ProxyDiscriminationAudit, DoubleFairnessAudit, ProxyVulnerabilityScore
- [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) — prediction intervals, PremiumSufficiencyController

---

**Related posts:**
- [FCA Consumer Duty Pricing Fairness in Python](/2026/03/20/fca-consumer-duty-pricing-fairness-python/) — the proxy discrimination audit in detail: FairnessAudit, detect_proxies, and calibration_by_group
- [The FCA Is Investigating Home and Travel Insurers](/2026/03/19/the-fca-is-investigating-home-and-travel-insurers/) — the TR24/2 findings and what a complete evidence pack looks like
- [Recalibrate or Refit?](/2026/02/28/recalibrate-or-refit/) — using Murphy decomposition to make the right call when A/E is off
- [Your Pricing Model Is Drifting](/2026/03/03/your-pricing-model-is-drifting/) — TRIPODD drift attribution with insurance-monitoring
