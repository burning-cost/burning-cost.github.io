---
layout: post
title: "Who Owns Fair Value in a Delegated Authority Chain?"
date: 2026-04-05
categories: [regulation, fairness, governance]
tags: [FCA, consumer-duty, fair-value, delegated-authority, MGA, distribution-chain, PRIN-2A, PS22/9, insurance-fairness, insurance-monitoring, insurance-governance, multicalibration, PROD-4, UK-insurance]
description: "The FCA is consulting in Q2 2026 on exactly this question. Every layer in a DA chain — insurer, MGA, broker, PCW — is expected to evidence outcomes, not just process. Here is what that means technically and how to build the artefacts each layer needs."
author: Burning Cost
---

The FCA's consultation on insurance distribution chains opens in Q2 2026. The stated scope: clarify how Consumer Duty fair value responsibilities are allocated between manufacturers — insurers and product providers — and distributors: brokers, MGAs, PCWs, and intermediaries. At the same time, the FCA is launching an expanded oversight review of delegated authority models and remuneration arrangements, with findings expected early 2027.

Most of the commentary so far has treated this as a governance question. It is also a technical question, and the technical question is harder.

The governance question — "who is responsible for fair value at each layer?" — has a relatively clear answer: each layer is responsible for demonstrating that its own conduct does not impair value, regardless of what the layer above has certified. A manufacturer's fair value assessment does not transfer to the distributor. The distributor is not protected by pointing upstream.

The technical question is: what does "demonstrating fair value" actually require in a quantitative sense? The FCA's language under Consumer Duty (PRIN 2A) is "regularly assess, test, understand and evidence the outcomes their customers are receiving." That word — outcomes — is doing significant work. It means observed claims experience relative to expectation. It means calibration at customer segment level. It means a pricing model that was fair at deployment and is being monitored to remain fair under the distributor's book composition, which may be substantially different from the manufacturer's.

This post is about that technical question.

---

## What a delegated authority chain looks like

A typical UK motor DA arrangement has three to four layers.

**Layer 1: Manufacturer.** The insurer or Lloyd's syndicate. Sets the rating factors, maintains the actuarial model, holds the risk capital. Produces a fair value assessment under PROD 4 covering the product's design, pricing basis, and intended target market. Signs off the binding authority.

**Layer 2: MGA.** Operates under the binding authority. Has delegated underwriting authority to accept risks within agreed parameters. May apply additional underwriting criteria, adjust the manufacturer's rates within a corridor, or operate its own pricing overlay. The MGA's remuneration — typically a percentage of GWP plus, critically, profit commission — is the FCA's primary concern. Profit commission arrangements create an incentive for the MGA to reject borderline risks, suppress claims, or select the lower end of a pricing corridor. Each of these impairs fair value in a way that is invisible in the manufacturer's FVA.

**Layer 3: Broker or intermediary.** Places business through the MGA. May have a preferred MGA panel from which it earns additional remuneration. The broker's incentive structure, if it steers customers towards products that generate higher remuneration rather than better value, is also in scope. Contingent commissions — commissions contingent on volume or loss ratio targets — are specifically flagged in the FCA's 2026 priorities for intermediaries.

**Layer 4: PCW.** Price comparison websites influence which products are presented first and at what prominence. Default product selection — the product shown at the top of a comparison page before the customer applies any filters — is explicitly in scope. A PCW that defaults to a product based on remuneration rather than value is impairing fair value at the top of the distribution funnel.

The FCA's position is that each of these layers must evidence fair value outcomes independently. The manufacturer cannot certify for the MGA. The MGA cannot certify for the broker. The broker cannot certify for the PCW.

---

## The problem with the current approach

The standard approach to fair value certification in a DA chain is sequential sign-off: the manufacturer produces a PROD 4 FVA for the product, the MGA and broker acknowledge it as part of their binding authority or distribution agreement, and the combined document is filed as evidence. This is process documentation. It is not outcome evidence.

The FCA's TR24/2 (August 2024) multi-firm review covered 28 product manufacturers and 39 distributors. The finding, plainly stated: most fair value assessments "lacked granularity to identify whether specific groups were receiving poor outcomes." The report found "limited evidence that firms were monitoring outcomes at the customer group level." The documents existed; the evidence did not.

The specific gap TR24/2 identified is the gap between what the manufacturer modelled and what the distributor's customers actually received. A manufacturer may model a target market with a 62% loss ratio. If an MGA's profit commission structure creates an incentive to underwrite lower-risk customers within the agreed parameters — cherry-picking within corridor — the manufacturer's 62% loss ratio assumption applies to a broader population than the MGA is writing. The MGA's customers may be receiving a worse price-to-risk ratio than the manufacturer intended. The manufacturer's FVA does not capture this. The MGA has not run its own outcome analysis.

This is the specific problem the Q2 2026 consultation is designed to address. The FCA wants each layer to run its own outcome monitoring, at customer group level, and to show the evidence.

---

## What outcome evidence looks like at MGA level

The PRIN 2A.9 requirement for outcomes monitoring translates, in a DA context, into the following minimum. Per quarter, an MGA should be able to produce:

1. **Calibration split by customer segment.** Not just a portfolio A/E ratio. A/E by age band, by distribution channel, by geography, and by any other dimension where the MGA's book composition differs materially from the manufacturer's assumed target market. The question is not whether the book is profitable in aggregate. It is whether specific customer groups are paying more than their risk warrants — or less — relative to what the manufacturer modelled.

2. **Proxy discrimination check.** Where the MGA's distribution channel creates a demographic composition that differs from the manufacturer's target market, the MGA needs to verify that the manufacturer's rating factors are not functioning as proxies for protected characteristics in the MGA's specific book. A factor that is a clean risk signal across the manufacturer's full market may be correlated with ethnicity in a specific geographical concentration served by a particular MGA.

3. **Remuneration impact documentation.** Specifically for profit commission arrangements: a documented record of whether the profit commission structure is creating a detectable pattern in underwriting decisions. Claims acceptance rates, premium-to-risk ratios at the low and high end of the pricing corridor, and referral patterns all leave measurable traces. If the profit commission incentive is distorting underwriting, the data will show it.

4. **Linked to manufacturer assumptions.** The MGA's outcome monitoring should be linkable to the manufacturer's FVA assumptions. Not identical — the MGA's book will differ from the manufacturer's modelled target market — but the comparison should be explicit: here is what the manufacturer assumed, here is what our book shows, here is our explanation of the difference.

None of this is new regulation. It is what PRIN 2A.9 already requires. What is new is that the Q2 2026 consultation will formalise which layer must produce which evidence and what happens when it does not exist.

---

## The technical framework: decomposing fair value at each layer

Fair value, in the FCA's Consumer Duty sense, is not a single number. It is a statement about outcomes across customer groups. A product can have a 62% loss ratio in aggregate and systematically overprice 60-year-old customers in low-income postcodes while underpricing 25-year-old customers in affluent urban areas. The aggregate looks fine. The cross-section is not fair value.

The technical decomposition we recommend for DA chains has three components.

**First component: multicalibration monitoring.** Standard portfolio-level calibration monitoring — the A/E ratio check — answers the question "is the model well-calibrated overall?" Multicalibration monitoring answers "is the model well-calibrated for every subgroup at every premium band?" These are different questions with different answers.

The distinction matters in a DA context because the MGA's book composition may differ from the manufacturer's in ways that expose specific (subgroup, premium band) combinations that were adequately covered in the manufacturer's aggregate calibration but are mispriced for the MGA's particular customer mix.

`MulticalibrationMonitor` in `insurance-monitoring` runs this check:

```python
from insurance_monitoring import MulticalibrationMonitor

# Fit on manufacturer reference data (or MGA reference period)
monitor = MulticalibrationMonitor(
    n_bins=10,
    min_z_abs=1.96,
    min_relative_bias=0.05,
    min_exposure=50.0,
)
monitor.fit(
    y_true=y_ref,
    y_pred=y_pred_ref,
    groups=age_bands_ref,
    exposure=exposure_ref,
)

# Update each quarter with MGA's live data
result = monitor.update(
    y_true=y_actual,
    y_pred=y_predicted,
    groups=age_bands,
    exposure=exposure,
)

summary = result.summary()
print(summary["n_alerts"])        # cells with significant miscalibration
print(summary["overall_pass"])   # True if no (subgroup, premium band) cells alert
if summary["worst_cell"]:
    wc = summary["worst_cell"]
    print(f"Worst cell: group={wc['group']}, bin={wc['bin_idx']}, "
          f"bias={wc['relative_bias']:+.1%}, z={wc['z_stat']:.2f}")
```

An alert fires only when both conditions hold: the relative bias (observed minus expected, divided by expected) exceeds 5%, and the z-statistic exceeds 1.96. This dual gate prevents small-cell noise from generating false positives while also filtering out statistically significant but economically trivial bias. A cell with a 4.9% bias and a z-statistic of 3.1 does not alert. A cell with a 7% bias and a z-statistic of 2.3 does. This is the right structure for a DA context where some cells will be thin.

The `groups` parameter takes any hashable labels — age band, distribution channel, geography. In a DA context, the obvious grouping is by the MGA's specific distribution channel: business introduced by Broker A versus business introduced by PCW B. If the calibration is materially different between channels, that is the fair value signal the FCA is looking for.

**Second component: proxy discrimination at distributor level.** The manufacturer may have clean proxy detection at their market level. The MGA's specific book may exhibit correlations that do not appear in the manufacturer's aggregate data. This is not a theoretical concern. An MGA with a geographic concentration in areas with high ethnic minority populations may find that certain postcode bands behave as proxies for ethnicity at the MGA level in ways that were diluted in the manufacturer's national book.

The `FairnessAudit` class in `insurance-fairness` runs proxy detection alongside the full demographic parity and calibration check:

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=None,                             # pass None to use existing predictions
    data=mga_book_df,
    protected_cols=["age_band_proxy"],      # or deprivation decile as proxy
    prediction_col="model_premium",
    outcome_col="incurred_loss",
    exposure_col="exposure",
    factor_cols=["postcode_district", "vehicle_group", "ncd_years", "channel"],
    model_name="Motor Freq v3.2 — MGA Book",
    run_proxy_detection=True,
    run_counterfactual=False,
    n_bootstrap=200,
)
report = audit.run()

# Overall RAG — red means a protected characteristic has a significant finding
print(report.overall_rag)

# Proxy factors flagged at amber or red across any protected characteristic
print(report.flagged_factors)

# Demographic parity log-ratio for age band proxy
dp = report.results["age_band_proxy"].demographic_parity
print(f"DP log-ratio: {dp.log_ratio:+.4f} [{dp.rag.upper()}]")

# Export the evidence record
report.to_markdown(f"fairness_audit_mga_{quarter}.md")
```

The evidence record produced by `to_markdown()` is what PRIN 2A.9 requires: a documented audit of whether the MGA's book, using the manufacturer's pricing model, is producing fair outcomes for the MGA's specific customer mix. This is distinct from the manufacturer's FVA. It is the MGA's own evidence.

**Third component: the governance chain.** The two monitoring outputs — multicalibration alerts and fairness audit results — need to be connected to a governance record that documents the accountability allocation across the DA chain. This is where `insurance-governance` comes in.

The `MRMModelCard` captures the intended-use scope and the limitations relevant to each distribution layer:

```python
from insurance_governance import MRMModelCard, Assumption, Limitation

mga_card = MRMModelCard(
    model_id="motor-freq-v3-mga-book",
    model_name="Motor Frequency v3.2 — MGA Distribution",
    version="3.2.1",
    model_class="pricing",
    intended_use=(
        "Frequency pricing for UK private motor policies introduced through "
        "the [MGA name] binding authority. Applies manufacturer base model "
        "within agreed pricing corridor."
    ),
    not_intended_for=[
        "Commercial fleet",
        "Policies outside agreed binding authority parameters",
        "New business outside manufacturer target market",
    ],
    portfolio_scope="[MGA name] binding authority — UK private motor",
    assumptions=[
        Assumption(
            description=(
                "MGA book composition consistent with manufacturer target "
                "market assumptions within ±10% on key demographic bands"
            ),
            risk="HIGH",
            mitigation=(
                "Quarterly multicalibration monitoring by age band and channel; "
                "FairnessAudit quarterly against manufacturer reference dataset"
            ),
        ),
        Assumption(
            description="Profit commission structure creates no detectable underwriting selection bias",
            risk="HIGH",
            mitigation=(
                "Quarterly calibration split by profit commission status; "
                "referral pattern monitoring; underwriting acceptance rate by segment"
            ),
        ),
    ],
    monitoring_owner="MGA Head of Pricing",
    monitoring_frequency="Quarterly",
    monitoring_triggers={"ae_ratio_segment": 1.10, "dp_log_ratio": 0.15},
)
```

The `monitoring_triggers` dictionary is the documented threshold at which the MGA escalates to the manufacturer. The key regulatory point: the trigger level and the escalation process need to be written into the binding authority agreement, not established informally after an issue arises. The FCA's oversight review will check whether accountability allocation is explicit, not assumed.

---

## The PCW problem

PCWs occupy an unusual position in this framework. They do not underwrite, do not set premiums, and do not have direct access to claims outcomes. But they influence which products customers see first and therefore which products they buy. If default product selection is driven by remuneration rather than value, the PCW is impairing fair value at the top of the funnel — without ever touching the underlying product.

The FCA's problem with PCW incentives is structural. A PCW that earns a higher fee for presenting Product A at the top of a comparison page is creating a selection effect on the manufacturer's and MGA's book. The customers who buy Product A via the PCW because it appeared first are not necessarily the customers for whom Product A offers the best value. The PCW's remuneration structure creates book composition differences between manufacturers — differences that neither the manufacturer nor the MGA directly controls.

For PCWs, the relevant fair value evidence is not claims-based calibration — they do not have the claims data. It is outcome evidence about whether the products given prominent placement genuinely represent better value for the customers who select them. This requires linked data between the PCW's click-through and placement decisions and the subsequent outcomes reported by manufacturers and MGAs. That data linkage is precisely what the Q2 2026 consultation is expected to require formalising.

For manufacturers and MGAs, the PCW problem manifests as a data quality issue: the PCW channel's book composition is driven partly by the PCW's commercial structure, which the manufacturer did not model. The multicalibration and fairness checks above are how you detect whether the PCW channel is creating calibration differences. Running those checks with `groups=channel` and separating PCW-sourced business from direct and broker-sourced business is the diagnostic.

---

## What the Q2 2026 consultation is likely to require

The consultation has not opened yet. What we know from the FCA's February 2026 insurance priorities communication, the MGAA analysis, and the TLT intermediaries briefing is the following:

The FCA intends to create clearer rules about which entity in the chain is responsible for which piece of fair value evidence. At present, responsibility is often ambiguous: the binding authority agreement acknowledges the manufacturer's FVA without specifying what the MGA must independently produce. The consultation will address this ambiguity.

The FCA will specifically address remuneration. Profit commission and contingent commission arrangements are named. The question being asked is whether these arrangements can be structured in ways that are consistent with fair value obligations — and if so, how that must be documented.

The FCA will address the data sharing requirements between chain layers. A manufacturer cannot assess whether its FVA assumptions hold for the MGA's book if the MGA does not share claims outcome data. The current structure often involves minimal data sharing. The consultation is expected to require more.

**The firms that will be best positioned when the consultation closes are those that are already running the monitoring now.** Not because the consultation will require exactly the methodology described above — we do not know the precise requirements yet. Because the consultation will require outcomes evidence, and outcomes evidence takes time to accumulate. A firm that starts quarterly multicalibration monitoring and fairness auditing in Q2 2026 will have 12 months of documented outcomes by the time the consultation translates into rule changes. A firm that waits for the rules to finalise will start with no history.

---

## The documentation standard that regulators actually check

One practical note on what "documented monitoring" means, based on what the FCA's TR24/2 found inadequate and what PRA supervisory engagement has found inadequate in analogous contexts (model risk reviews, Consumer Duty implementation assessments).

Regulators are not looking for a policy document. They are looking for evidence that the monitoring ran, when it ran, what it found, and what was done about it. The specific artefacts:

A **monitoring log** with dated entries. Not a dashboard that shows the current state. A log that shows the state at each monitoring cycle: what the multicalibration result was, whether any cells alerted, what the fairness audit found, what decision was made. The decision is mandatory: if the monitoring ran clean, document that. If it flagged something, document the finding, the review, and the decision to act or not to act.

A **version-tagged record** of which manufacturer model version the MGA was operating against at each date. If the manufacturer updates the model, the MGA's monitoring baseline needs to be updated too, and the update needs to be documented.

A **named sign-off** at each cycle. Under SM&CR, the individual responsible for the MGA's pricing and fair value compliance is a named Senior Manager. The monitoring log should show that person's sign-off. An undated Excel file with no named owner is not evidence.

An **escalation record** for any cycle where monitoring flagged an issue. What was flagged, who was notified (including notification to the manufacturer if the issue implicates the manufacturer's FVA assumptions), what was decided, when the next review was scheduled.

The `ModelInventory` in `insurance-governance` handles version tracking and event logs. The monitoring and audit outputs from `insurance-monitoring` and `insurance-fairness` produce JSON records that can be logged against the inventory entry, creating the audit trail directly from the monitoring run rather than assembling it retrospectively.

---

## All four regulatory posts

This is the fourth and final post in the series covering Q1 2026 UK insurance regulatory priorities. The full set:

- **FCA AI Live Testing Cohort 2** — what firms entering the sandbox need to demonstrate on bias detection, uncertainty quantification, and audit trails
- **Motor Insurance Pricing at the Floor** — repricing headroom in a deteriorating NCR environment without triggering Consumer Duty scrutiny
- **SS1/23 Monitoring: What PRA Auditors Expect in 2026** — mapping the SS1/23 model monitoring principles to automated pricing model governance artefacts
- **This post**: who owns fair value in a delegated authority chain, and what the evidence base for each layer looks like technically

The through-line across all four: the regulatory direction of travel in UK general insurance is from process documentation to outcomes evidence. Having a policy is not enough. Having a monitoring trigger threshold is not enough. The question regulators are now asking is: do you have the dated, named, signed evidence that the monitoring ran, the outcomes were assessed, and decisions were made?

---

## References

- FCA Insurance Regulatory Priorities 2026, February 2026 — fca.org.uk
- FCA TR24/2: Multi-firm review of product governance under Consumer Duty, August 2024
- FCA PRIN 2A: Consumer Duty, effective 31 July 2023
- FCA PROD 4: Product governance (manufacturers and distributors)
- Consumer Duty Outcome 4: Price and value (PRIN 2A.4)
- MGAA: FCA Regulatory Priorities 2026 — what they mean for MGAs and delegated underwriting
- TLT: FCA Regulatory Priorities: Insurance — what intermediaries and distributors need to know
- Clifford Chance: FCA Insurance Outlook for 2026
- Denuit, Michaelides & Trufin (2026): Multicalibration in Insurance Pricing. arXiv:2603.16317

Related on Burning Cost:

- [What PRA Auditors Will Ask to See in 2026: Mapping SS1/23 to Automated Pricing Model Monitoring](/2026/04/05/ss123-monitoring-what-pra-auditors-expect-2026/)
- [Motor Finance Redress: What PS26/3 Tells Insurance Pricing Teams About Their Own Exposure](/2026/04/04/motor-finance-redress-ps263-insurance-pricing-governance/)
- [Mean Parity Is Not Tail Parity](/2026/04/04/demographic-parity-tails-regression-insurance-fairness/)
- [Multicalibration Monitoring in Python: Subgroup-Level Calibration for Insurance Pricing](/2026/04/04/insurance-model-monitoring-python-practitioner-guide/)
