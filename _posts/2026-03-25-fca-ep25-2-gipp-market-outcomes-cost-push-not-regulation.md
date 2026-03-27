---
layout: post
title: "What FCA EP25/2 Actually Shows: Cost-Push, Not Regulation, Drove UK Insurance Inflation"
date: 2026-03-25
categories: [regulation]
tags: [EP25-2, GIPP, PS21-5, pricing-practices, claims-inflation, market-outcomes, FCA, motor, home, DiD, insurance-monitoring, insurance-trend]
description: "FCA EP25/2 published July 2025. Expected claim costs per home policy up 49% from £92 to £138. Average inception premium up only 5%. The data says insurers absorbed the shock — not passed it — and GIPP got the credit it deserved in motor, not home."
---

There has been a lot of noise in the UK market about insurance premium inflation since 2022. Much of that noise has been politically charged: the FCA imposed pricing controls, and now customers are paying more. The implication being that GIPP caused the problem.

FCA EP25/2, published in July 2025, is a 50-page evaluation of the GIPP remedies drawing on policy-level data from 16 home insurers and 13 motor insurers, covering roughly 80% of the home market by gross written premium. It runs from Q1 2019 to Q1 2024. We think it deserves more careful reading than it has received, because the headline story is almost the opposite of the political narrative.

This is not a post about the PS21/5 rules themselves — we have [a separate post on that](/2026/03/25/ps21-5-renewal-pricing-end-to-end-python/). This post is about what the outcomes data actually shows, and what pricing teams should be tracking as a result.

---

## The number that matters most for home insurance

Table 2 of EP25/2 sets out descriptive statistics for home policies in the pre-intervention period (2019–2021) and post-intervention period (2022–2024 Q1). The figure that jumps out is the expected cost of claims (ECC) for core home policies:

| Period | Average ECC per policy |
|---|---|
| Pre-GIPP (2019–2021) | £92.26 |
| Post-GIPP (2022–2024 Q1) | £137.51 |

That is a **49% rise in expected claim costs** on the core home product. Over the same period, total price at inception rose from £248.52 to £260.92 — a **5% increase**.

If you want to tell a story about insurer profiteering, that table does not support it. The ECC is what the insurer's pricing model expects to pay out. It is not claims inflation speculation — it is the technical price basis. A 49% cost increase with a 5% premium increase implies severe margin compression in home. The FCA notes this plainly: "the expected cost of claims for core policies rose sharply from £92.26 to £137.51 — a 49% increase... likely influenced by rising inflation, particularly in the increased cost of repair materials."

---

## Motor: costs up 12%, premiums up 11%

Motor looks different. ECC for motor rose from £312.26 to £349.38 — a 12% rise — while total motor inception prices rose from £445.46 to £497.90, an increase of 11.5%.

Two things are happening here simultaneously. First, motor inflation is real but more contained than home (at least in ECC terms). Second, the distribution of that cost has shifted: new business customers went from paying £458.24 to £569.38 (up £111), while existing customers went from £437.48 to £460.19 (up only £22.71). That is the GIPP intervention working as intended: new customers absorbing more of the true cost of risk, rather than being used as loss-leaders funded by overcharging loyal customers.

The FCA's causal analysis found GIPP to be statistically significantly associated with a £6.63 per policy reduction in motor premiums, with a ten-year central estimate of £1.6 billion in consumer savings (range: £163 million to £3.0 billion). That wide range reflects genuine uncertainty about the counterfactual — what would have happened without the intervention — not methodological weakness.

---

## Home: why the causal result is insignificant

For home, the FCA found no statistically significant causal relationship between GIPP and prices. Across buildings only, contents only, and combined policies, the CDiD (continuous difference-in-differences) estimates were inconclusive.

This is not evidence that consumers got worse outcomes. The FCA is explicit: "we cannot establish a statistically significant causal link between the reduction in prices and GIPP... this should not be interpreted as evidence that GIPP has made consumers in the home market worse off overall."

Their interpretation is that market dynamics swamped the intervention effect. The 49% ECC increase is the reason. When underlying cost structure doubles, a pricing control designed to stop companies charging loyal customers 35% more than new customers becomes second-order. The mechanism of GIPP (closing the new/renewal price gap) still worked — the home renewal differential halved, from £95.38 to £49.17 — but the overall level effect was overwhelmed by macroeconomic cost push.

---

## Product quality: the compulsory excess story

The FCA found mixed signals on product quality. Claims payouts were broadly stable. Cover limits in home increased for most tenures. But motor compulsory excess rose post-GIPP.

We think the compulsory excess trend matters more than it is being given credit for in coverage of this paper. When the new/renewal pricing gap closes, one mechanism firms can use to maintain margin on new business is product adjustment rather than price adjustment. A higher compulsory excess reduces insurer exposure without triggering the ENBP cap comparison. It is not hollowing out (the FCA found no reduction in peril coverage), but it is a quality adjustment that should be in your monitoring framework.

The FCA also flags the rise of 'essentials' products — basic motor cover at lower price — as a potential avoidance mechanism. Brand cycling and tiered pricing both allow firms to maintain new business price competitiveness while the existing book pays more, without technically breaching the renewal pricing rule. The FCA acknowledges it "cannot fully reflect this wider industry shift" from its sample.

---

## The inflation decomposition problem

Here is what all of this means for a pricing actuary trying to explain 2022–2024 to a board or a regulator.

The EP25/2 data shows that nominal premium increases were broadly matched by cost increases, with home actually lagging cost by a substantial margin. But most boards are not asking whether home pricing is adequate — they are asking why prices are higher than they were in 2020. The answer has three components:

1. **Claims frequency and severity inflation** — repair costs, labour, materials, vehicle write-off values. For motor: paint +16%, spare parts +11% (Q3 2022 to Q3 2023, per EP25/2).
2. **GIPP structural effect** — new business premiums rising because the cross-subsidy from loyal customers is no longer available to fund them. One-time repricing event.
3. **Consumer Duty fair value obligation** — insurers actively reducing the renewal penalty means the back book repriced upward, some of which overlaps with (2).

What the FCA explicitly cannot say — and what EP25/2 does not claim — is that GIPP *caused* the overall rise in average premiums. In motor the causal estimate points the other direction: GIPP was associated with lower prices by £6.63 per policy. The premium inflation visible to consumers is a cost story, not a regulatory story.

You can quantify each of these components using [`insurance-trend`](https://github.com/burning-cost/insurance-trend), which decomposes loss cost into frequency and severity trends and can ingest external ONS indices for deflation:

```python
from insurance_trend import LossCostTrendFitter, ExternalIndex

# Fetch ONS motor repair cost index (SPPI G4520 — Maintenance & repair of motor vehicles)
# ONS series HPTH: the right deflator for motor severity
repair_idx = ExternalIndex.from_ons("HPTH")

fitter = LossCostTrendFitter(
    periods=["2019Q1", "2019Q2", "2019Q3", "2019Q4",
             "2020Q1", "2020Q2", "2020Q3", "2020Q4",
             "2021Q1", "2021Q2", "2021Q3", "2021Q4",
             "2022Q1", "2022Q2", "2022Q3", "2022Q4",
             "2023Q1", "2023Q2", "2023Q3", "2023Q4"],
    claim_counts=claim_counts_by_quarter,
    earned_exposure=exposure_by_quarter,
    total_paid=paid_by_quarter,
    external_index=repair_idx,
)

result = fitter.fit()
print(result.summary())
# Returns: frequency trend (ann.), severity trend (ann.),
# superimposed inflation above repair cost index, projected loss cost
print(result.decompose())
```

If you can show your board that claims severity is running at +8% per annum above the ONS motor repair index — using publicly verifiable external data rather than internal assumptions — the pricing increase becomes defensible without recourse to vague references to "inflation."

---

## What pricing teams should be monitoring

EP25/2 is a backward-looking evaluation covering 2019 to Q1 2024. The FCA used it to answer four questions: did price walking stop, did prices fall, did product quality rise, did switching costs fall. They got mixed answers but were broadly satisfied with motor outcomes.

The ongoing monitoring question is different. The FCA is now watching for:

1. **Residual price walking** — the annual attestation obligation remains. Any positive slope in margin vs. tenure gets investigated.
2. **Avoidance patterns** — brand cycling and tiered product strategies are flagged explicitly. Supervision is watching.
3. **Compulsory excess creep** — not formally prohibited, but the FCA noted it. It is on their radar.
4. **Claims ratio trends** — the FCA cited a fall from 64% to 56% in motor claims cost as a proportion of premium (2022 to 2023). A continued decline would attract attention under Consumer Duty fair value obligations.

All four of these are measurable with [`insurance-monitoring`](/insurance-monitoring/). The calibration module gives you A/E ratio tracking; the drift attribution module (TRIPODD) tells you which rating factors are driving margin movements; the discrimination module gives you the Gini drift test if your ranking is shifting.

For tenure-based monitoring specifically, the relevant check is whether expected margin as a function of tenure is flat:

```python
import numpy as np
from sklearn.linear_model import Ridge
from insurance_monitoring.drift_attribution import DriftAttributor

# Fit a margin model on your reference period (e.g. 2022 data)
# Features: tenure, region, vehicle_age, distribution channel (encoded)
# Target: expected_margin = (core_price - ECC) / core_price
margin_model = Ridge().fit(X_ref, y_margin_ref)

attributor = DriftAttributor(
    model=margin_model,
    features=["tenure", "region", "vehicle_age",
              "channel_pcw", "channel_direct", "channel_intermediary"],
    alpha=0.05,
    loss="mse",
    n_bootstrap=200,
)
attributor.fit_reference(X_ref, y_margin_ref, train_on_ref=False)

# Run against latest quarterly data
result = attributor.test(X_monitor, y_margin_monitor)

if result.drift_detected:
    print(f"Top drift attributors: {result.attributed_features}")
    # tenure appearing here is a compliance signal to investigate
    # before your annual GIPP attestation is due
```

The FCA's supervisory work on GIPP breaches found that most anomalies were technical pricing errors rather than deliberate design. That is both reassuring and a prompt: your monitoring needs to catch the technical errors before the FCA's annual review does.

---

## The board narrative

If you are writing the pricing narrative for your board or your compliance committee, the EP25/2 data supports something like this:

*The FCA's own evaluation shows that home insurance expected claim costs rose 49% from £92 to £138 per policy between 2019–2021 and 2022–2024, while average inception premiums rose only 5%, from £249 to £261. In motor, the causal analysis found that GIPP was associated with a £6.63 per policy reduction in prices, not an increase. Premium inflation visible to consumers in both lines reflects cost push — repair materials, labour, parts — not regulatory overhead or margin expansion.*

That is a defensible position because it is sourced directly from the FCA's own policy-level dataset, covering 80% of the home market. It is not a trade association estimate or an insurer's internal analysis. It is what the regulator found when it looked.

---

## What EP25/2 leaves unanswered

The paper has honest limitations. The CDiD methodology requires stronger functional form assumptions than binary DiD, and the causal estimates are sensitive to how price-walking intensity is defined. The compliance cost data was voluntary — nine of 16 home firms responded — so the burden estimates may overstate due to self-selection. And the evaluation ends in Q1 2024; it says nothing about the period since.

The FCA also explicitly flags that it cannot determine distributional effects: which consumers benefited from the loyalty pricing correction, and which lost the benefit of low new-business prices. That question — who gained and who lost from GIPP — is not answered here and requires consumer-level data the FCA does not have.

Finally, EP25/2 does not address proxy discrimination or demographic disparity in pricing. That is covered by Consumer Duty (PRIN 2A) and TR24/2, not this paper. Anyone citing EP25/2 as authority for fair value monitoring across protected characteristics is misattributing the source.

---

**Source:** FCA Evaluation Paper 25/2, "An evaluation of our General Insurance Pricing Practices (GIPP) remedies," July 2025. [Available at fca.org.uk](https://www.fca.org.uk/publications/corporate-documents/evaluation-paper-25-2-general-insurance-pricing-practices-remedies). All figures from Tables 2–7 and descriptive text of the published PDF.
