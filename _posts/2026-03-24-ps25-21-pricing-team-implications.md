---
layout: post
title: "What FCA PS25/21 Actually Means for Your Pricing Team"
date: 2026-03-24
categories: [regulation, governance, pricing]
tags: [FCA, PS25/21, PROD4, product-governance, fair-value, pricing, insurance-governance, model-risk, Consumer-Duty]
description: "PS25/21 abolished the mandatory 12-month product review cycle in December 2025. Harm-proportionate review cadence is now the requirement. Here is what that means for actuarial governance calendars and how to document it."
---

PS25/21 came into force on 9 December 2025. Most of the coverage focused on the CPD change (the 15-hour minimum is gone) and the employer's liability reporting simplification. Both of those matter to compliance teams. Neither of them is what should be keeping your pricing actuaries busy in Q1 2026.

The change that actually affects pricing governance is this: the mandatory annual product review cycle has been abolished. PROD 4 no longer requires manufacturers to review non-investment insurance products at least once every 12 months. Instead, you must determine review frequency based on the product's potential for customer harm, arising from its specific risk factors. Call it harm-proportionate review cadence — that is what it is.

That sounds like a simplification. In the short term, for a mature product with a stable book, it might be. But the flip side is that "we review annually because the rules say so" is no longer a sufficient governance rationale. You now have to have a documented, evidenced position on *why* a given product warrants a given review frequency — and you need to update that position when data suggests harm risk is changing.

For pricing teams, this creates two immediate practical problems.

---

## The calendar problem

Most UK pricing teams run actuarial governance off an annual calendar: rate review in Q3, committee sign-off in Q4, refresh deployed by January. That calendar was partly driven by the PROD 4 12-month floor. Remove the floor and the calendar still makes sense for some products — but now you have to argue why, not just point to the rule.

The argument has to be risk-based. The FCA's guidance in PS25/21 is that firms should consider factors including product complexity, distribution channel, complaint and claims data trends, whether the product is newly launched, and whether pricing is stable. A simple motor product on a well-monitored book with a low complaint rate and stable claims development might warrant an 18-month cycle. A newly-launched telematics product where scoring algorithms are still being calibrated, or a liability product where large loss patterns are volatile, should be reviewed more often, not less.

The practical implication: your pricing governance calendar needs to be rebuilt around a written rationale per product, not a shared 12-month default. If you have 15 product lines, that is 15 documented positions. The documentation has to hold up if the FCA asks why product X was reviewed in November 2025 but not again until March 2027.

---

## The fair value problem

The other shift in PS25/21 that matters for pricing teams is the clarification of manufacturer versus distributor responsibilities in fair value assessment.

Under the new rules, product reviews are explicitly the manufacturer's responsibility. Distributors must review their product distribution arrangements — a related but distinct obligation. The FCA was explicit in its response to consultation feedback: even in lead-firm arrangements (where one insurer takes responsibility for product design under the new PS25/21 framework), all co-manufacturers remain jointly responsible for ensuring products deliver fair value. The lead firm arrangement changes who leads the review process; it does not change who is on the hook for the outcome.

For pricing teams at insurers who distribute through intermediaries, this means the fair value assessment cannot be delegated to the intermediary. The insurer's pricing team has to have its own evidence that the product delivers fair value — which in practice means having data on actual customer outcomes, not just a model output or a market comparison.

This is not new territory under Consumer Duty. But PS25/21 sharpened the wording at the same time as removing the fixed review schedule. A firm that reduces review frequency because "PROD 4 allows it" and simultaneously allows its fair value evidence to go stale is taking a position the FCA will not view sympathetically.

---

## What your governance documentation needs to say

The documentation requirement is now more demanding than before, not less, even though the rule has been simplified.

Under the old annual cycle, a compliant firm could produce a review pack once a year and tick the box. Under the harm-proportionate model, a compliant firm needs:

1. A written policy that defines what makes a product high, medium, or low harm-risk — covering at minimum complexity, distribution channel, complaints trend, claims volatility, and time since last material change
2. A documented classification for each product under that policy, with the assigned review frequency
3. Monitoring data sufficient to trigger an out-of-cycle review if the harm indicators change — if complaint rates spike, if claims development deteriorates, if a distribution channel is mis-selling, the review frequency must be updated
4. Evidence of fair value assessment done by the manufacturer, not delegated

Point 3 is where most teams will find this hardest. The 12-month default was convenient precisely because it did not require ongoing monitoring as a trigger. Harm-proportionate cadence requires you to have a live view of whether the product is delivering what it is supposed to deliver. That is a monitoring problem as much as a governance problem.

---

## How insurance-governance helps

The [`insurance-governance`](/insurance-governance/) library was built around PRA SS1/23 principles, but the same framework maps directly to what PS25/21 now requires. Specifically, the `ModelInventory` and `GovernanceReport` components address points 1–3 above.

`ModelInventory` is a JSON-backed registry of your production models. Each entry carries a `next_review_date` and a validation run history. The important addition under the PS25/21 framework is that next review date should now be set by your risk classification policy, not by a calendar default.

```bash
pip install insurance-governance
```

```python
from insurance_governance.mrm import ModelInventory, MRMModelCard, RiskTierScorer

inventory = ModelInventory("governance_registry.json")

# Score risk tier based on objective criteria
scorer = RiskTierScorer()
tier = scorer.score(
    gwp_impacted=85_000_000,   # GWP exposed to this model
    model_complexity="medium",
    deployment_status="champion",
    regulatory_use=False,
    external_data=True,        # telematics feed — external data flags higher risk
    customer_facing=True,
)

card = MRMModelCard(
    model_id="home-freq-v2",
    model_name="Home Contents Frequency",
    version="2.1.0",
    model_class="pricing",
    intended_use="Frequency pricing for home contents — standard policies only.",
)

# Register with review date driven by risk tier, not calendar
review_months = {1: 6, 2: 12, 3: 18}  # Tier 1 = highest risk, shortest cycle
import datetime
next_review = datetime.date.today() + datetime.timedelta(days=30 * review_months[tier.tier])
card.next_review_date = str(next_review)

inventory.register(card, tier=tier)
print(inventory.overdue())
```

The `tier.tier` value comes from the six-dimension composite score — GWP, model complexity, deployment status, regulatory use, external data use, customer-facing. A model touching £85m of GWP, using external telematics data, and directly pricing customers will score Tier 1 regardless of how benign you think it is. A legacy book-pricing model in run-off, limited customer exposure, no external data, might legitimately score Tier 3 — and under the harm-proportionate framework, you now have a documented, auditable basis for an 18-month review cycle rather than an annual one.

The `GovernanceReport` generates an executive pack (HTML + JSON) covering model purpose, risk tier rationale, last validation RAG status, assumptions register, and the next review date. That is your PROD 4 documentation artefact. It does not replace actuarial judgement, but it makes the paper trail consistent across 15 product lines rather than bespoke across each one.

```python
from insurance_governance.mrm import GovernanceReport

GovernanceReport(card=card, tier=tier).save_html("home_contents_governance_pack.html")
```

The output is print-to-PDF ready. It includes the risk tier rationale in plain English — the six dimension scores and the composite — which is exactly the kind of structured evidence that justifies a non-standard review cycle under the new rules.

---

## The honest caveat

PS25/21 is described by the FCA as a simplification, and in the narrow sense that the 12-month floor is gone, it is one. But simplification here means shifting from a bright-line rule to a principles-based obligation that requires evidence. That is harder to comply with, not easier — it just gives well-organised teams more flexibility.

Teams that have done the work to classify their products by harm risk, build monitoring triggers, and maintain consistent governance documentation will genuinely benefit from the new framework. They can argue for longer cycles on stable products and shorter cycles where the risk warrants it, and they will have the documentation to back both positions.

Teams that assumed "simpler rules" meant less governance work are in a worse position now than they were in November 2025.

If you are implementing PS25/21 changes in Q1 2026 and need to get the documentation framework in place quickly, the [`insurance-governance`](/insurance-governance/) library is the fastest route to consistent, audit-ready governance packs across your pricing model inventory.

**Related:**
- [One Package, One Install: PRA SS1/23 Validation and MRM Governance Unified](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) — the full insurance-governance walkthrough
- [The EBM Is Sitting in Your Notebook Because You Can't Show It to the Committee](/2026/03/16/the-governance-bottleneck-for-ebm-adoption/) — governance translation between model outputs and committee packs
- [FCA Proxy Discrimination Testing in Python](/2026/03/22/fca-proxy-discrimination-python-testing-guide/) — the EP25/2 audit requirement that sits alongside PS25/21 product governance
