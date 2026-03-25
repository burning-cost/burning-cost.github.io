---
layout: post
title: "FCA Pure Protection Market Study: What Pricing Teams Should Prepare For"
date: 2026-03-25
categories: [regulation, pricing]
tags: [FCA, pure-protection, consumer-duty, fair-value, proxy-discrimination, insurance-fairness, insurance-governance, insurance-monitoring, market-study, MS24-1, life, income-protection, critical-illness, uk-personal-lines]
description: "The FCA's interim report on MS24/1 landed in January 2026 with a Q3 2026 final report expected. Here is what pure protection pricing teams need to build before that lands."
---

The FCA published the interim report for MS24/1 - its market study into the distribution of pure protection products - on 29 January 2026. The feedback window closes 31 March 2026. A final report with any agreed remedies arrives Q3 2026. So we are, right now, in the gap between interim and final: the period where firms that have thought carefully about what comes next can get ahead of it, and firms that have not are about to have a difficult summer.

This is not a general-purpose regulatory explainer. It is a list of things pricing teams should be building or documenting in the next three months, with working code for the parts where we have tooling.

---

## What the FCA actually found

The headline finding - that the pure protection market "generally offers good consumer outcomes" - is real, but it is doing a lot of work as a shield for the things the FCA is less comfortable with.

The specifics that matter for pricing teams:

**Income protection claims ratios are low.** The FCA flagged that income protection products show claims ratios around 40%, materially below the 50%+ seen across other product lines. That gap is flagged explicitly as a fair value concern requiring further examination before the final report. If you write income protection and your claims ratio looks like this, you should have a documented explanation ready.

**Premium dispersion is high and the FCA cannot explain it.** The regulator found significant price dispersion across providers for comparable risks - and acknowledged it lacks sufficient data to determine how much of that dispersion reflects legitimate underwriting differences versus something else. The phrase used in the interim report is "broad premium dispersion can indicate a certain level of price discrimination but that the FCA could not assess this based on available data." That is a regulator saying "we want better data". Expect the final report to ask for it.

**Pre-existing conditions are a live concern.** Access and affordability for consumers with pre-existing medical conditions is a stated focus. The FCA is not (yet) proposing intervention in underwriting criteria, but it is clearly watching whether pricing models systematically load or exclude this population in ways that compound the protection gap.

**Commission-driven switching is in scope.** Intermediary behaviour - specifically, encouraging customers to replace policies at the end of clawback periods to restart commission cycles - is identified as a problem even though the FCA estimates current harm is modest. This matters for pricing teams because it signals that the regulator is scrutinising the relationship between distribution arrangements and consumer outcomes in granular detail.

**Guaranteed over-50s plans are a specific concern.** These products are singled out for poor fair value: the FCA found examples where cumulative premiums paid over an average policy lifetime exceed the sum assured by more than 50%. This is a standalone problem for writers of that product, but it is also a signal about how the FCA intends to measure fair value going forward - not on price alone, but on lifetime value delivery.

---

## The Consumer Duty framing changes what evidence you need

The FCA is running this market study under the Consumer Duty lens (PRIN 2A, live July 2023). Outcome 4 - Price and Value - requires firms to demonstrate that products offer fair value, not just that premiums are competitive at point of sale.

This distinction is critical. A pricing team that can produce a competitive pricing analysis against the market has satisfied the pre-Consumer Duty test. Under Outcome 4, you also need to show that:

1. Claims ratios by segment are being monitored and are within a defensible range
2. The pricing model does not systematically disadvantage groups with protected characteristics through proxies
3. Distribution arrangements - specifically commission structures - are not eroding the value delivered to end consumers

None of this is new in principle. What changes is the evidentiary standard. The FCA will, in the final report, set expectations about what firms should be able to produce. Teams that have not built the infrastructure to answer these questions by the time the final report lands will face a compressed timeline to retrofit it.

---

## Step 1: Proxy discrimination audit for your pricing model

The protection gap focus on pre-existing medical conditions, combined with the general premium dispersion concern, creates a straightforward inference: the FCA will want to understand whether pricing models for term, CI and IP products are using proxies for protected characteristics. Postcode is the obvious one - it correlates with income, ethnicity, and disability prevalence. Occupation codes are another.

The `insurance-fairness` library has the tooling for this. The `IndirectDiscriminationAudit` is the right entry point - it computes the five benchmark premiums from Côté, Côté & Charpentier (2025) and quantifies proxy vulnerability as the mean absolute difference between the unaware and aware models:

```python
from insurance_fairness import IndirectDiscriminationAudit

audit = IndirectDiscriminationAudit(
    protected_attr="disability_indicator",   # or income_band, ethnicity_proxy
    proxy_features=["postcode_district", "occupation_code"],
    exposure_col="exposure",
)
result = audit.fit(X_train, y_train, X_test, y_test)

print(f"Proxy vulnerability: {result.proxy_vulnerability:.3f}")
print(result.segment_report)
```

A `proxy_vulnerability` score near zero means your unaware model (no protected attribute) produces similar premiums to the aware model. A large score means your postcode and occupation features are doing work that, at least in part, tracks the protected attribute. That is not automatically unlawful - indirect discrimination requires a proportionality justification - but you need to know your number before a regulator asks.

For a full audit that produces a documented evidence pack, `FairnessAudit` wraps the complete workflow:

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=catboost_model,
    data=df,
    protected_cols=["disability_indicator", "occupation_code"],
    prediction_col="predicted_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
)
report = audit.run()
report.to_markdown("protection_fairness_audit_2026Q1.md")
```

The `DoubleFairnessAudit` is worth running alongside this. Its key finding - from Bian et al. (2026) - is that equalising premiums across groups (action fairness) does not equalise loss ratios (outcome fairness). The FCA's Consumer Duty Outcome 4 requires the latter. A firm that has only run action fairness checks may still fail a Consumer Duty review:

```python
from insurance_fairness import DoubleFairnessAudit

audit = DoubleFairnessAudit(n_alphas=20)
audit.fit(
    X_train,
    y_premium,
    y_loss_ratio,   # Consumer Duty Outcome 4 is about this
    S_disability,
)
result = audit.audit()
print(audit.report())   # formatted for an FCA evidence pack
```

---

## Step 2: Claims ratio monitoring by segment

The income protection claims ratio concern is specific, but the general principle - that the FCA expects firms to track fair value outcomes not just at portfolio level but by segment - applies to all pure protection product lines.

`insurance-monitoring` has calibration and discrimination tooling that covers this. The A/E ratio by segment is the starting point:

```python
from insurance_monitoring.calibration import ae_ratio, CalibrationChecker

# Per-segment A/E monitoring
checker = CalibrationChecker(
    thresholds_amber=0.15,   # >15% deviation triggers amber
    thresholds_red=0.25,
)

for segment in ["standard_lives", "substandard_lives", "over50s"]:
    seg = df[df["segment"] == segment]
    result = ae_ratio(
        actual=seg["claims_incurred"],
        expected=seg["model_expected"],
        exposure=seg["exposure"],
    )
    print(f"{segment}: A/E = {result.ratio:.3f}, p = {result.p_value:.4f}")
```

For ongoing production monitoring without repeated-testing inflation - which is the right framework for monthly monitoring cycles - `PITMonitor` provides anytime-valid calibration change detection (Henzi, Murph & Ziegel, 2025):

```python
from insurance_monitoring import PITMonitor

monitor = PITMonitor(alpha=0.05)

# Reference period: fit on training data PITs
monitor.fit(pit_scores_reference)

# Each month, update with new PITs — alarm fires only if genuinely uncalibrated
alarm = monitor.update(pit_scores_new_cohort)
if alarm.triggered:
    print(f"Calibration shift detected: {alarm.statistic:.4f} > {alarm.threshold:.4f}")
```

The point is to build a documented trail that shows segment-level claims ratios are being tracked in something close to real time, not reconstructed after a regulatory request.

---

## Step 3: Model documentation at the level the FCA will expect

The governance library makes it straightforward to produce MRC-ready model cards that capture the assumptions, limitations, and risk tier classification in a structured format. For pure protection pricing models, the regulatory scrutiny angle requires adding explicit entries for fair value assumptions and the conditions under which the model should be reviewed:

```python
from insurance_governance import MRMModelCard, Assumption, Limitation, RiskTierScorer, GovernanceReport

card = MRMModelCard(
    model_id="ip-freq-v3",
    model_name="Income Protection Frequency",
    version="3.0.1",
    model_class="pricing",
    intended_use="Frequency pricing for individual IP policies. Retail distribution only.",
    assumptions=[
        Assumption(
            description="Claim frequency relationship with occupation is stable since 2021 SOE data",
            risk="MEDIUM",
            mitigation="Annual recalibration against insurer experience. A/E monitored quarterly.",
        ),
        Assumption(
            description="Postcode district is a legitimate risk factor, not a proxy for disability or income",
            risk="HIGH",
            mitigation=(
                "Annual IndirectDiscriminationAudit. proxy_vulnerability threshold: 0.05. "
                "FCA Consumer Duty Outcome 4 fair value assessment included in annual model review."
            ),
        ),
    ],
    limitations=[
        Limitation(
            description="Model trained on standard underwriting decisions. Substandard lives pricing extrapolated.",
            severity="HIGH",
        ),
    ],
)

scorer = RiskTierScorer()
tier = scorer.score(
    gwp_impacted=45_000_000,
    model_complexity="high",
    deployment_status="champion",
    regulatory_use=True,     # Consumer Duty evidence obligations
    external_data=False,
    customer_facing=True,
)

report = GovernanceReport(card=card, tier=tier)
report.save_html("ip_freq_v3_mrm_pack.html")
```

The `regulatory_use=True` flag on `RiskTierScorer` pushes the model into a higher risk tier, which is correct - Consumer Duty obligations mean pure protection pricing models carry regulatory use implications that were not there before July 2023.

---

## Step 4: Fair value measurement for over-50s and low-claims-ratio products

The guaranteed over-50s finding is the most concrete fair value problem in the interim report. The FCA's framework is straightforward: if cumulative premiums paid over an average policy term materially exceed the sum assured, the product does not pass a fair value test regardless of the competition landscape.

This requires pricing teams to compute expected lifetime value by entry age and sum assured band, not just at point-of-sale. The monitoring framework for this looks like:

```python
import polars as pl
from insurance_monitoring.calibration import check_balance

# Compute expected lifetime premium vs expected payout by entry age band
lifetime_df = (
    df.group_by("entry_age_band")
    .agg([
        pl.col("premium_pv").sum().alias("total_expected_premium_pv"),
        pl.col("benefit_pv").sum().alias("total_expected_benefit_pv"),
        pl.col("exposure").sum(),
    ])
    .with_columns(
        (pl.col("total_expected_benefit_pv") / pl.col("total_expected_premium_pv"))
        .alias("value_ratio")
    )
)

# Flag cohorts where value_ratio < 0.67 (FCA's implicit threshold from interim report)
at_risk = lifetime_df.filter(pl.col("value_ratio") < 0.67)
print(at_risk)
```

The 0.67 threshold is our inference from the FCA finding that at least some over-50s products pay out less than 67p per £1 of premiums collected. It is not a formal regulatory threshold - yet. But the Q3 2026 final report may well formalise something like it.

---

## What to do before Q3 2026

The practical task list:

**By end of April.** Run proxy discrimination audits on all pricing models that use postcode, occupation, or any other variable that could serve as a proxy for a protected characteristic. Get the `proxy_vulnerability` scores documented. Scores above 0.10 need a proportionality justification on file.

**By end of May.** Segment your claims ratios for all pure protection product lines. If income protection is running below 45% on any segment, have a defensible documented explanation before you need to produce one under pressure. Check over-50s lifetime value ratios by entry age band.

**By end of June.** Ensure model cards for all pure protection pricing models are at a standard where they could be presented to the FCA without editing. This means explicit documentation of fair value assumptions, proxy discrimination controls, and the conditions under which the model would be reviewed or retired.

**By Q3 2026 (final report).** Have a monitoring framework in place that generates segment-level claims ratio and calibration reports as a matter of routine - not as a one-off exercise. The FCA's final report will set expectations about what good looks like. Firms that already have the infrastructure in place will be able to demonstrate compliance quickly. Firms that need to build it after the fact face a harder conversation.

---

## The bigger picture

The interim report's overall tone - broadly positive, concerns noted, further data sought - is what a regulator sounds like before it decides what remedies to propose. The Q3 2026 final report is where teeth appear, if they are going to.

Our reading: the FCA is building a case for enhanced data collection requirements and possibly product rules for commission structures. It is not currently signalling intervention in underwriting criteria. But the attention to premium dispersion, proxy discrimination potential, and lifetime value on over-50s products tells you where the direction of travel is.

The firms that come out of the final report in a manageable position are the ones that have already answered the questions the FCA is visibly asking. The firms that do not are going to be providing that evidence in a less comfortable context.

Install the libraries:

```bash
pip install insurance-fairness insurance-governance insurance-monitoring
```

The fairness audit, governance documentation, and monitoring framework work together. Run them as a package, document the outputs, and you have the foundation of a Consumer Duty evidence file that covers the specific concerns the FCA has raised in MS24/1.

---

*The FCA's interim report (MS24/1.4) is available at [fca.org.uk](https://www.fca.org.uk/publications/market-studies/ms24-1-1-market-distribution-pure-protection). The feedback deadline is 31 March 2026. Final report expected Q3 2026.*
