---
layout: post
title: "Premium Finance APR as a Pricing Problem"
date: 2026-03-25
categories: [pricing, regulation, consumer-duty]
tags: [premium-finance, apr, consumer-duty, fca, ms24-2, fair-value, insurance-fairness, insurance-monitoring, segment-monitoring, credit-broker, uk-motor, uk-home]
description: "FCA MS24/2 (February 2026) means pricing teams now own the APR question. Here is how to treat it as a pricing problem — with the same tools used for the insurance itself."
---

The FCA published MS24/2 — its final report on the premium finance market study — in February 2026. The headline finding is broadly positive: average APRs fell from 23.3% in 2022 to 19.2% by 2026, saving customers roughly £157 million annually. The FCA decided against market-wide interventions such as APR caps or commission bans. It concluded the market is broadly working.

That framing invites complacency. Read past the summary and MS24/2 sets up a significant and ongoing compliance obligation — one that lands squarely on pricing teams, not just the credit function.

The reason is Consumer Duty. And the reason it is a pricing problem, not just a compliance problem, is that the APR a customer pays is not an independent credit decision. It is a function of the customer's risk segment, the insurer's commercial arrangements with the premium finance provider, and the competitive position at the point of sale. Pricing teams already have all the data and most of the methodology to audit this. They are just not yet being asked to.

---

## Why APR varies by segment

In the majority of the UK market, premium finance is offered as an add-on to the insurance product. The insurer or broker refers the customer to a premium finance provider — often a wholly-owned subsidiary or a contracted third party — and receives a commission in exchange. That commission is typically a percentage of the premium financed, which means the absolute commission is larger on higher-premium policies.

Here is the mechanism that creates segment-level APR disparity:

1. The premium finance provider sets an APR that recovers cost of funds, credit losses, and a margin. Cost of funds is roughly flat across the portfolio (the provider has a single funding cost). Credit losses are not flat — they vary with the creditworthiness of the insured.

2. The commission paid to the insurer is applied as a deduction from the rate the provider needs to charge to hit its return. If the commission is flat (say, 25% of the premium financed), the provider's net margin varies across segments. The provider compensates by charging higher APRs to lower-credit-quality segments where credit losses are higher.

3. The insurer is simultaneously a Consumer Duty principal — responsible for the insurance product's fair value — and a credit broker under the Consumer Credit Act. Under PRIN 2A.4, the fair value obligation applies to the whole package a customer receives. If the insurer's commission structure is causing high APRs for specific risk segments, that is a Consumer Duty issue even if the insurance premium itself is technically correctly priced.

MS24/2 confirmed this dual role is not theoretical. The FCA found no evidence of systematic double-counting (credit risk priced into both the premium and the APR), but it explicitly flagged that fair value assessments must cover the total cost to the customer, not just the insurance element. Where 20% of motor and home premium finance customers are paying above 30% APR, the distribution of those customers matters enormously for Consumer Duty compliance.

---

## What Consumer Duty actually requires

Consumer Duty (PRIN 2A, which came into force July 2023) requires firms to demonstrate that their products deliver fair value at segment level — not just in aggregate. Outcome 4 (Price and Value) is not satisfied by showing the average APR fell by 4.1 percentage points. It requires firms to demonstrate that no identifiable group of customers is systematically receiving poor value.

The FCA's multi-firm review under Consumer Duty (published 2024) was explicit that firms cannot hide behind portfolio averages. If a specific segment — say, lower-income policyholders who cannot pay the full annual premium and therefore must use instalments — is consistently offered APRs above 30%, that is a fair value failure regardless of what the average looks like.

The problem for pricing teams is that the required analysis is not currently baked into the product sign-off process for most insurers. The insurance pricing model is owned by the pricing team. The premium finance proposition — including the APR, the commission rate, and the terms — is often owned separately, by a treasury or credit function that may have limited visibility of the segment-level distribution of who actually uses it.

MS24/2 does not mandate a specific remedy. But supervisory engagement with "outlier firms" — which the FCA committed to — will focus on whether firms have segment-level evidence that the APR offered represents fair value. Firms that cannot produce that evidence are exposed.

---

## The APR fair value model

This is where pricing methodology comes in. The question of whether APR represents fair value for a given segment is structurally identical to asking whether a premium represents fair value: does the cost charged to this segment reflect the actual risk and cost of serving that segment?

For the insurance premium, pricing actuaries answer this question through calibration-by-group analysis. For premium finance, the equivalent question is:

> Is the APR offered to segment S commensurate with the credit risk, administrative cost, and reasonable margin for serving segment S?

The inputs to this model are:

- **Credit loss rate by segment.** This should ideally come from actual premium finance arrears data. If that data sits in a separate system, getting access to it is the first prerequisite.
- **Administrative cost.** Broadly flat — origination cost per policy does not vary dramatically by risk segment.
- **Funding cost.** Flat at the portfolio level.
- **Commission paid to the insurer.** This is the loaded element. If the commission is a flat percentage of the premium, and the insurance premium varies by risk segment, the effective cost that the provider needs to recover from the APR is heterogeneous across segments.

A segment with high insurance premium and low credit risk generates a large commission for the insurer and requires low credit loss reserves for the provider. The APR for that segment should be low. A segment with low premium (or a high-frequency, cheaper-premium cohort) and higher credit risk generates a smaller commission and higher credit losses. The APR for that segment will be higher, and that is not inherently unfair — but it needs to be justified with data, not assumed.

---

## Applying `insurance-fairness`

The `calibration_by_group` function in [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) is designed for exactly this kind of segment-level outcome audit. The standard use case is checking whether an insurance pricing model is equally well-calibrated across protected-characteristic groups. The premium finance APR audit is the same calculation, applied to a different outcome.

```python
import polars as pl
from insurance_fairness import calibration_by_group

# df has one row per policy that used premium finance
# apr_offered: the APR offered to the customer
# apr_justified: the APR that our cost model says is justified for this segment

result = calibration_by_group(
    df=df,
    protected_col="credit_score_decile",
    prediction_col="apr_justified",   # model prediction
    outcome_col="apr_offered",        # what the customer actually got
    exposure_col="n_policies",
)

print(result.max_disparity)
# Values above 0.15 in log-space indicate material segment-level divergence
# between what the cost model justifies and what customers are charged

for decile, group_ae in result.actual_to_expected.items():
    for group, ae in group_ae.items():
        if ae > 1.15:
            print(f"Decile {decile}, group {group}: A/E = {ae:.2f} — overcharge flag")
```

The `CalibrationResult.actual_to_expected` dict gives the A/E ratio for each segment-by-decile cell. An A/E above 1.0 means customers in that segment are charged more than the cost model justifies. `result.max_disparity` is the headline figure: the worst overcharge across all cells. This is the audit output a supervisory engagement will ask for.

For the full Consumer Duty evidence pack, `DoubleFairnessAudit` (introduced in v0.6.0) is more appropriate. It distinguishes between action fairness — are prices the same at the point of quoting? — and outcome fairness — does the product deliver equivalent value after the policy is live? Under Consumer Duty Outcome 4, outcome fairness is what the FCA is actually measuring.

```python
from insurance_fairness import DoubleFairnessAudit

audit = DoubleFairnessAudit(n_alphas=20)
audit.fit(
    X_train,          # customer features (segment, channel, product type)
    y_premium,        # primary outcome: total cost of credit (premium + finance charge)
    y_value,          # fairness outcome: value received (claims + service) / total cost
    S_income_band,    # protected group: income band or credit score decile
)
result = audit.audit()
print(result.summary())
print(audit.report())  # FCA evidence pack section
```

The key result from Bian et al. (2026, arXiv:2601.19186) — which this implements — is that equalising action outcomes does not equalise value outcomes. A firm that charges the same APR to all segments regardless of credit risk may satisfy action fairness (Delta_1 = 0) while still failing outcome fairness (Delta_2 remains large) because low-risk segments are effectively subsidising high-risk ones. The FCA's fair value framework does not require equal APRs. It requires that each segment receives value commensurate with what they pay.

---

## Ongoing monitoring with `insurance-monitoring`

The fair value obligation is not a one-off exercise. Consumer Duty requires ongoing monitoring. This means the APR distribution across segments needs to be tracked over time, with the same rigour applied to insurance pricing model calibration.

`CalibrationChecker` from [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) supports this directly. The fit/check pattern is designed for the monitoring use case: establish a reference at product launch or last review, then check each new period against it.

```python
from insurance_monitoring.calibration import CalibrationChecker

# apr_offered: observed APR (y in calibration terms)
# apr_justified: cost-model APR (y_hat)

checker = CalibrationChecker(distribution="gamma", alpha=0.05)

# Reference: last sign-off period
checker.fit(y=apr_offered_ref, y_hat=apr_justified_ref, exposure=n_policies_ref)

# Run at each monitoring cycle (quarterly, or after any commercial renegotiation
# with the premium finance provider)
report = checker.check(y=apr_offered_q, y_hat=apr_justified_q, exposure=n_policies_q)
print(report.verdict())
```

The `CalibrationChecker` runs three diagnostics: the balance property test (is the portfolio-level A/E still within tolerance?), the auto-calibration test (is the relationship between justified and offered APR still monotonic and well-specified?), and the Murphy decomposition (has the source of any miscalibration shifted from bias to resolution?).

The Murphy decomposition is particularly useful here. When a commission renegotiation shifts APR, it typically shifts the bias component of miscalibration — the provider raises APRs uniformly, or uniformly in a tier — rather than the resolution component. A drift that shows up only in bias is a simpler remediation story than one that shows up in resolution (where the rank ordering of segments has changed, which suggests something structural in the commission structure has shifted).

For segment-level monitoring, run a separate `CalibrationChecker` per credit score decile rather than pooling. Miscalibration in the top-risk decile diluted by a well-calibrated low-risk population will pass a portfolio-level test while the high-APR segment remains undetected:

```python
segments = df["credit_score_decile"].unique().sort()
checkers = {}

for seg in segments:
    seg_df = df.filter(pl.col("credit_score_decile") == seg)
    c = CalibrationChecker(distribution="gamma", alpha=0.05)
    c.fit(
        y=seg_df["apr_offered_ref"].to_numpy(),
        y_hat=seg_df["apr_justified_ref"].to_numpy(),
        exposure=seg_df["n_policies_ref"].to_numpy(),
    )
    checkers[seg] = c

# Each quarter, update and check
for seg, checker in checkers.items():
    seg_q = df_current.filter(pl.col("credit_score_decile") == seg)
    report = checker.check(
        y=seg_q["apr_offered"].to_numpy(),
        y_hat=seg_q["apr_justified"].to_numpy(),
        exposure=seg_q["n_policies"].to_numpy(),
    )
    if "RECALIBRATE" in report.verdict() or "REFIT" in report.verdict():
        print(f"Segment {seg}: {report.verdict()}")
```

Running `CalibrationChecker` at the segment level without any multiplicity correction inflates the false positive rate — with ten deciles and quarterly checks over three years, you will see spurious alarms even from a well-functioning product. Apply Bonferroni correction to `alpha` (`alpha=0.05/10` with ten segments) or, if you have more than around fifteen segments, switch to Benjamini-Hochberg FDR control before acting on any individual segment flag.

---

## The practical gap to close

Most pricing teams do not currently have visibility of the premium finance arrears data that sits with the provider or in a credit function. The first step is not technical — it is organisational. Getting the credit loss data joined to the pricing dataset is prerequisite to any of the analysis above.

The second gap is governance. The APR is typically set in a commercial negotiation with the premium finance provider, not in the pricing sign-off process. If the pricing committee signs off on the insurance premium without also reviewing the segment-level APR distribution, the Consumer Duty obligation is not being discharged. MS24/2 does not specify that the same committee must review both. But it is hard to see how a fair value assessment of the total product can be valid if the pricing team has not seen the APR distribution and approved it.

The third gap is the cost model itself. The justified APR calculation requires a segment-level credit loss model. Most insurers have this implicitly — they know the claims frequency and severity by rating cell, which correlates strongly with propensity to default on premium finance. Building a simple credit loss model on that foundation, even a GLM with two or three features, gives enough structure to produce a defensible justified APR by segment.

None of this is analytically novel. It is a direct application of tools and methods the pricing team already uses for the insurance product. The novelty is that MS24/2 and Consumer Duty have made it a regulatory expectation, not a nice-to-have.

---

## What good looks like

A pricing team with a complete response to MS24/2 can demonstrate four things:

1. **A segment-level cost model** for premium finance, showing justified APR by risk and credit profile, updated at each rate review.

2. **A calibration audit** comparing offered APR to justified APR, with A/E ratios below the firm's defined tolerance at each monitoring cell.

3. **A double-fairness assessment** covering both action fairness (price treatment at point of sale) and outcome fairness (value delivered relative to cost), producing the evidence pack that supervisory engagement will request.

4. **An ongoing monitoring process** with per-segment calibration checks, multiplicity-corrected thresholds, and automatic escalation when commission structures are renegotiated or when segment-level APR drifts outside tolerance.

The FCA is not going to mandate a specific methodology. It will ask whether firms have evidence. The tools to produce that evidence are the same tools pricing teams already use. The gap is applying them to a product that pricing teams have not traditionally owned.

That gap is closing.

```bash
uv add insurance-fairness
uv add insurance-monitoring
```

Sources: [insurance-fairness](https://github.com/burning-cost/insurance-fairness) · [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) · [FCA MS24/2 Final Report (February 2026)](https://www.fca.org.uk/publications/market-studies/ms24-2-premium-finance)

---

- [Your Book Has Shifted and Your Model Doesn't Know](/2026/03/02/your-book-has-shifted-and-your-model-doesnt-know/) — the drift monitoring toolkit that applies equally to APR calibration monitoring
- [Model Value in Pounds: Translating Gini Improvement to Loss Ratio](/2027/01/14/model-value-in-pounds-translating-gini-improvement-to-loss-ratio/) — quantifying what better segment calibration is worth commercially
