---
layout: post
title: "The FCA's First Consolidated Insurance Priorities: What Pricing Teams Need to Do Now"
date: 2026-03-25
categories: [regulation]
tags: [fca, consumer-duty, fairness, governance, monitoring, ai, pet-insurance, pmi, insurance-fairness, insurance-governance, insurance-monitoring, pricing-regulation, uk]
description: "The FCA replaced 40+ portfolio letters with a single annual priorities document on 24 February 2026. Here is what it says, what it means for pricing teams, and which library tools map directly to the four pillars."
---

On 24 February 2026 the FCA published its first *Regulatory Priorities: Insurance* report — a 17-page document that replaces more than 40 portfolio letters. Sarah Pritchard (Deputy CEO) and Graeme Reynolds (Interim Director of Insurance) signed the foreword. The stated aim is a "clear, succinct one-stop shop" that "should act as a guide for firms' boards and chief executives."

We have read the document in full. Below is what it actually says, stripped of the usual regulatory ambiguity, with a direct read-through for pricing teams.

---

## What changed and why it matters

The old portfolio letter system was fragmented. A Head of Pricing at a composite insurer received letters for general insurance, motor, home, premium finance, and possibly life — each written by different supervisory teams, with inconsistent priorities and overlapping timetables. The new single report covers retail insurers, wholesale insurers, insurance intermediaries, PCWs, life insurers, and funeral plan providers under one cover.

The format change is not cosmetic. It signals that the FCA wants firms to read this as a board-level document, not a supervisor's checklist. Reynolds said at the ABI Conference in February 2026: "We want to raise standards and improve access to insurance." The framing is growth-permissive in tone but operationally specific in the timeline.

---

## The four pillars

### 1. Consumer understanding, claims handling and service quality

The report is blunt: "too many people have poor claims experiences." The three firm-level obligations are clear communication of cover, prompt and fair claims handling, and outcome monitoring.

Two pricing-relevant points sit inside this pillar.

**Sales process analysis.** The FCA will analyse "how different sales processes affect consumer outcomes" with work starting Q1 2026 and findings published later in the year. PCW presentation of comparable information is called out specifically. If your premium reflects the sales channel — and almost every personal lines tariff does — expect scrutiny of whether the channel differential is justifiable under Consumer Duty.

**Value measures rules review.** A review of value measures rules starts Q1 2026 and concludes Q4 2026. The FCA will assess "the impact of our rules and how firms have implemented them." The Motor Total Loss work is the reference point: around 270,000 drivers will receive compensation as a direct result. That is what inadequate value measurement looks like in enforcement terms.

### 2. Increasing access to insurance

This is the pillar that contains the two most watched sub-sectors for 2026.

**Pet insurance and private medical insurance** appear by name in the monitoring section: "Areas we continue to monitor for potential future action include pet insurance and private medical insurance, which have seen price rises and issues with consumer understanding." The FCA is not launching a formal study yet, but the language "continue to monitor for potential future action" is the standard precursor. Pet and PMI pricing teams should treat this as a Q3 2026 review preparation window.

**Premium finance.** The market study concluded with an estimated £157m annual saving to consumers since 2022, driven by average APR reductions of 4.1% (up to 8% where the FCA intervened directly). 23 million policies use premium finance. The FCA will continue monitoring APRs and act where fair value concerns persist. If your pricing process does not explicitly model the APR burden on monthly-pay customers as part of Fair Value Assessment, that is the gap.

**Social renters and mental health conditions.** The FCA will work with Fair4All Finance on home contents insurance for social renters (Q1 2026, findings by end of 2027), and with the ABI and Money and Mental Health Policy Institute on travel insurance underwriting for consumers with pre-existing mental health conditions (Q1 2026). Neither of these is a Pricing team lead item, but underwriting rating decisions that systematically exclude these groups will be in scope for Consumer Duty Outcome 4 (Price and Value).

**Pure protection.** The final report from the pure protection market study lands Q3 2026. The FCA wants to "reduce the protection gap, and improve consumer awareness and claims experiences." Closing the protection gap is an access story, not just a product design one.

### 3. Supporting growth and innovation

The AI section is explicit about the tools available. The FCA "encourage[s] firms to experiment with AI and use our sandbox services, and look at our Innovation Pathways." The document names three specific pathways:

- **AI Lab** — for testing AI applications before deployment
- **FCA Sandbox** — for new product or business model innovations
- **Supercharged Sandbox** — "specific services open to smaller market participants"

An evaluation report from AI Live Testing will be published by end of 2026. The Q1 2026 timeline shows an "Artificial Intelligence review" to "engage industry on the uses, risks and opportunities of AI in insurance and how we can remove barriers to encourage safe use."

This is not a threat — the FCA is asking to be involved, not to block. If your firm is building AI-driven underwriting or pricing models, the Innovation Pathways reduce regulatory risk at the development stage, not just at deployment. We think using the AI Lab is materially better than relying on post-deployment supervision.

The **captive insurance** framework is on the timeline: consultation in Q3 2026, framework introduced 2027 (joint with PRA). **Cyber insurance** gets a dedicated product review concluding later in 2026.

### 4. Simplifying regulation

The FCA is removing the pricing practices data returns (CP25/35 — three specific returns), rationalising conflicts of interest rules (CP25/36), and removing some product-specific rules (CP25/37). For pricing teams, the deletion of the General Insurance Pricing data returns matters. The FCA is simultaneously consulting on new GAP insurance rules and information disclosure requirements, so the net burden change will depend on the final PS.

The SMCR review with Treasury and PRA aims to "halve its regulatory burden." Whether that materialises is a question for H2 2026.

The Consumer Duty to non-UK consumers question gets a consultation in H1 2026 (international scope of ICOBS and PROD 4). Relevant for London Market writers with cross-border retail books.

---

## What this means for pricing teams specifically

Three concrete actions follow from this document.

**First: outcome monitoring is not optional.** The FCA uses "test outcomes" as a first-order firm obligation alongside "be clear" and "handle claims fairly." For pricing, this means documented monitoring of whether the price relativities you charge are producing good outcomes — not just whether the technical model is accurate. `insurance-monitoring` gives you the right tools here: the `MonitoringReport` class runs calibration, discrimination, and PSI checks with traffic-light output. The `PITMonitor` is specifically suited to ongoing consumer outcome monitoring because it gives anytime-valid type I error control — you do not need to pre-specify monitoring intervals.

```python
from insurance_monitoring import MonitoringReport, PITMonitor
from insurance_monitoring.calibration import ae_ratio_ci

# Outcome monitoring: is the model still fair and calibrated
# across the book, including vulnerable segments?
report = MonitoringReport(
    y_true=claims,
    y_pred=predicted_premiums,
    X=rating_factors,
    exposure=exposure,
    gini_bootstrap=True,   # bootstrap CIs for Gini drift governance
)
report.run()
report.print_summary()   # RAG status per metric

# Anytime-valid calibration check — no scheduled review required,
# no look-schedule inflation of type I error
monitor = PITMonitor(alpha=0.05)
monitor.update(y_true=new_claims, y_pred=new_predictions)
if monitor.alarm:
    print(f"Calibration alarm at t={monitor.t}: recalibrate or investigate")
```

**Second: proxy discrimination auditing for pet and PMI.** The FCA's pet and PMI watch-list entry is not yet enforcement, but the December 2025 Research Note on Motor Insurance Pricing and Local Area Ethnicity shows the direction. If your pet or PMI tariff contains geographic or demographic features, you need documented proxy discrimination monitoring before the FCA arrives. `insurance-fairness` v0.6.0's `DoubleFairnessAudit` is designed for exactly the Consumer Duty Outcome 4 question — it tests both action fairness (are you charging different amounts?) and outcome fairness (are loss ratios equivalent across groups?). Equalising premiums across protected groups does not automatically satisfy Outcome 4.

```python
from insurance_fairness import DoubleFairnessAudit
from insurance_fairness.diagnostics import ProxyDiscriminationAudit

# Step 1: detect proxies before the FCA finds them for you
proxy_audit = ProxyDiscriminationAudit(
    protected_col="local_area_ethnicity_index",   # or any proxy candidate
    factor_cols=["postcode_sector", "breed_group", "age_band"],
)
proxy_result = proxy_audit.run(df)
print(proxy_result.summary())   # D_proxy scalar + Shapley attribution

# Step 2: double fairness — action AND outcome
double_audit = DoubleFairnessAudit(n_alphas=20)
double_audit.fit(
    X_train,
    y_premium=pure_premium,
    y_loss_ratio=claims / premium,
    S_group=protected_indicator,
)
result = double_audit.audit()
print(result.summary())
print(double_audit.report())    # FCA evidence pack section, ready to attach
```

**Third: governance documentation for AI models.** The FCA's AI review starts Q1 2026. "Firms should consider testing ideas in our AI Lab" is not enforceable, but "they must monitor outcomes for consumers closely" is Consumer Duty. For any ML pricing model, an `MRMModelCard` from `insurance-governance` creates the governance artefact the FCA will want to see: risk tier, assumptions, limitations, validation status, and owner.

```python
from insurance_governance import MRMModelCard, RiskTierScorer, ModelInventory

# Document the model
card = MRMModelCard(
    model_name="PetInsurancePricingGBM_v2",
    model_type="CatBoost frequency-severity",
    use_case="Pure premium rating for UK pet insurance",
    owner="Pricing Actuary",
    assumptions=[
        "Breed group is a legitimate risk factor under Equality Act s.22 exception",
        "Geographic factors are not proxies for protected characteristics",
    ],
    limitations=[
        "Trained on 2022-2025 experience; extreme vet cost inflation not in training",
        "Breed groups mapped at product-line level, not individual animal",
    ],
)

# Score risk tier — higher tiers require more governance
scorer = RiskTierScorer()
tier = scorer.score(card)
print(f"Risk tier: {tier.tier} — {tier.rationale}")

# Add to inventory for model risk committee reporting
inventory = ModelInventory.load("model_inventory.json")
inventory.add(card)
inventory.save("model_inventory.json")
```

---

## The monitoring watch-list in practice

The document lists "other areas of focus" that do not yet have formal programmes. Besides pet and PMI, the FCA is reviewing financial crime systems and controls "across a sample of our larger firms" and will publish good practice findings in 2026. Child Trust Funds get a focused product review (Q1 2026). Closed-book life insurance products get a review starting Q2 2026.

For pricing and reserving teams at life insurers, the closed-book review is the one to watch: the FCA will "consider how firms are ensuring good outcomes for consumers" following Consumer Duty. That will include whether guaranteed premium terms are still fair value and whether dormant policyholders are receiving equivalent treatment to active ones.

---

## What we think

The consolidation into a single annual document is genuinely useful. The old portfolio letter system meant that pricing priorities were often buried in supervisor letters that the Head of Pricing never saw. This document is written for boards. That is an improvement.

The pet and PMI watch-list entry is the most actionable signal in the document for personal lines pricing teams. The FCA does not need to launch a market study to request information; it can do so through supervisory engagement. Firms that already have documented proxy audits and outcome monitoring for these lines are in a materially better position than those that do not.

The AI stance is pragmatic. The FCA is not trying to restrict AI adoption; it is trying to ensure outcome monitoring keeps pace with deployment speed. That is a reasonable ask, and the Innovation Pathways reduce the cost of early engagement. We think firms that treat the AI Lab as a compliance cost are misreading the signal: early engagement on a novel model reduces the risk of a retrospective supervisory intervention, which is the expensive outcome.

The simplification agenda is real but incremental. Deleting three pricing data returns and rationalising conflicts rules is meaningful operational relief, but the Consumer Duty remains the primary analytical burden. Firms hoping for a step-change reduction in regulatory load will be disappointed.

The full document is available at [fca.org.uk/publication/regulatory-priorities/insurance-report.pdf](https://www.fca.org.uk/publication/regulatory-priorities/insurance-report.pdf).

---

**Libraries referenced in this post:**
- [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) — outcome monitoring, calibration, Gini drift, PITMonitor
- [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) — proxy discrimination auditing, double fairness, optimal transport correction
- [`insurance-governance`](https://github.com/burning-cost/insurance-governance) — MRM model cards, risk tier scoring, governance reports
