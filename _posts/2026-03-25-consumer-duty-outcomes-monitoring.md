---
layout: post
title: "Consumer Duty Outcomes Monitoring: What Data to Collect and What to Report"
date: 2026-03-25
categories: [regulatory, governance, libraries]
tags: [consumer-duty, fca, ps22-9, outcomes-monitoring, fair-value, vulnerable-customers, insurance-governance, insurance-fairness, python, uk-insurance, prin-2a, double-fairness, proxy-vulnerability]
description: "The FCA's first full outcomes monitoring year under Consumer Duty is under way. Here is what data to collect, what the regulator actually wants to see in MI, and how insurance-governance and insurance-fairness wire it together."
---

The FCA published its findings from a multi-firm review of insurance outcomes monitoring in early 2026, having assessed submissions from 20 larger firms against PRIN 2A.9 and Chapter 11 of FG22/5. The verdict was blunt: most firms were reporting process completion rates, not customer outcomes. A firm that tells its board "100% of products have had a fair value assessment completed" has told it nothing about whether customers are receiving fair value. The FCA said so explicitly.

This is the first full outcomes monitoring year — the rules for open products came into force 31 July 2023, for closed books 31 July 2024, and 2025/26 is the first cycle where the FCA is examining monitoring frameworks with real teeth. If your programme looks like a repurposed TCF dashboard, it needs rebuilding.

This post covers what data to collect, how to structure the MI, and where [`insurance-governance`](https://github.com/burning-cost/insurance-governance) and [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) fit into the plumbing.

---

## The four outcomes and what they actually require

Consumer Duty is structured around four outcomes under PRIN 2A: Products and Services, Price and Value, Consumer Understanding, and Consumer Support. Most firms have decent coverage of the first and last two — products are approved through a governance process, and support metrics like complaint rates and claims handling times are usually already flowing into board MI.

The one that trips firms up is Outcome 2: Price and Value. Under PRIN 2A.4, firms must assess whether the price a retail customer pays is reasonable relative to the overall benefits. That requires evidence at the product level, but also across customer cohorts — specifically, the FCA wants to see whether any group of customers is systematically receiving worse value than another group on the same product.

That is not a fair value framework question. It is a fairness question. The FCA's multi-firm review found that few firms had made this connection.

---

## What data to collect

### Premium and exposure distribution by outcome cohort

The starting dataset is your policy extract with earned premium, exposure, and claims — segmented by renewal year and cohort. "Cohort" here has two meanings:

**Rating cohort:** the set of customers who bought or renewed in a given quarter, used to track whether premium adequacy has held. Loss ratios for Q3 2023 cohort customers are now fully developed; you can see whether the premium charged was adequate.

**Customer segment:** a grouping relevant to fair value — tenure band, distribution channel, product variant, geographic region. The FCA's concern is cross-segment variation: if your direct customers have a loss ratio of 62% and your aggregator customers have 91%, you have a value transfer problem that needs explaining or addressing.

```python
import polars as pl

policies = pl.read_parquet("policies.parquet")

# Premium and loss ratio by renewal cohort and channel
cohort_summary = (
    policies
    .group_by(["renewal_year", "channel", "product_variant"])
    .agg([
        pl.sum("earned_premium").alias("earned_premium"),
        pl.sum("incurred_claims").alias("incurred_claims"),
        pl.sum("exposure_years").alias("exposure_years"),
        pl.len().alias("policy_count"),
    ])
    .with_columns([
        (pl.col("incurred_claims") / pl.col("earned_premium")).alias("loss_ratio"),
        (pl.col("earned_premium") / pl.col("exposure_years")).alias("avg_premium"),
    ])
)
```

This is the base table your fair value metrics sit on top of. You need at least three years of rolling cohorts to identify trends rather than noise.

### Claims outcomes by segment

The FCA found that claims monitoring was the most consistent gap. Firms were monitoring claim frequency and average cost but almost none were monitoring claims settlement outcomes — whether customers received a fair settlement relative to the value they were due.

What to add:

- **Claims acceptance rate by segment** — not just overall. A 94% acceptance rate on average can mask an 85% rate for one distribution channel or tenure band.
- **Time to settlement by segment** — the FCA is increasingly treating slow settlement as a consumer harm indicator.
- **Settlement value relative to sum insured** — for home contents in particular, under-settlement is a known issue for customers with characteristics of vulnerability who do not challenge initial offers.
- **Declined claims rate** — tracked quarterly, with an amber trigger at >2pp movement.

```python
claims = pl.read_parquet("claims.parquet")

claims_outcomes = (
    claims
    .group_by(["segment", "year_quarter"])
    .agg([
        pl.len().alias("claims_registered"),
        (pl.col("outcome") == "accepted").sum().alias("claims_accepted"),
        pl.mean("days_to_settlement").alias("avg_days_settlement"),
        pl.median("settlement_pct_sum_insured").alias("median_settlement_pct"),
    ])
    .with_columns(
        (pl.col("claims_accepted") / pl.col("claims_registered")).alias("acceptance_rate")
    )
)
```

### Vulnerable customer metrics

Under PRIN 2A.9.7, firms must monitor whether customers with characteristics of vulnerability receive outcomes at least as good as the overall customer population. This requires a proxy — most firms do not hold a self-declared vulnerability flag at policy level, and the ones that do have significant under-identification.

The practical approach is to use `ProxyVulnerabilityScore` from `insurance-fairness`. It computes the per-policyholder proxy discrimination penalty: how much does the model's premium differ from the discrimination-free aware premium? High proxy vulnerability scores identify policyholders whose premium is being materially inflated (or deflated) because their risk factors are acting as proxies for protected characteristics correlated with vulnerability.

```python
from insurance_fairness import (
    ProxyVulnerabilityScore,
    PremiumSpectrum,
    partition_by_proxy_vulnerability,
)
import numpy as np

# unaware: your model's premium (no protected attribute)
# aware: discrimination-free premium from DiscriminationFreePrice
spectrum = PremiumSpectrum(
    best_estimate=None,        # optional; provide if you have a model fitted with D
    unaware=policy_df["model_premium"].to_numpy(),
    aware=policy_df["aware_premium"].to_numpy(),
    hyperaware=None,
    corrective=None,
    sensitive_values=policy_df["proxy_protected_indicator"].to_numpy(),
)

pvs = ProxyVulnerabilityScore()
result = pvs.compute(spectrum)

# result.proxy_vulnerability: per-policyholder Δ_proxy = µ_U - µ_A
# Positive = model premium exceeds discrimination-free premium
# Default RAG thresholds: amber >5% of aware premium, red >15%

# Partition by vulnerability exposure for segment-level reporting
partitions = partition_by_proxy_vulnerability(
    result,
    spectrum,
    n_bins=5,
)
# partitions gives quintile bands from lowest to highest proxy discrimination exposure
```

The `partition_by_proxy_vulnerability` output is your vulnerable customer monitoring metric: what share of your book is in the amber and red proxy vulnerability bands, tracked quarterly. A shift upward in the red band percentage is an early warning that your pricing model is creating differential outcomes for proxied protected groups.

---

## What to report

### The board MI structure

The FCA's review found two failure modes in board reporting: firms that produced walls of data with no narrative, and firms that produced glossy summaries with no data. Good MI has a defined structure. We recommend four sections, each on one slide:

**1. Outcomes scorecard (RAG by product)**

One row per product. Columns: Price and Value RAG, Consumer Support RAG, Complaints trend, Claims acceptance trend, Vulnerable customer flag. The RAG ratings are driven by the metrics below with defined tolerances, not by editorial judgment. If a metric has moved outside tolerance, the cell is amber or red and the narrative explains why and what remediation is planned.

**2. Fair value deep-dive (for any product with non-green Price and Value)**

Loss ratio by cohort vintage and distribution channel. Premium adequacy trajectory. Any cross-segment value transfer identified and quantified (e.g., "aggregator channel subsidising renewal book by approximately £X per policy"). Recommended action with owner and deadline.

**3. Claims outcomes**

Acceptance rate trend (24 months), time to settlement trend, median settlement percentage trend. Any significant segment-level divergence flagged with root cause hypothesis.

**4. Vulnerable customer outcomes**

Proxy vulnerability band distribution (quarterly), compared to prior quarter and prior year. Claims acceptance rate for high-PV quintile versus overall. Complaint rate for high-PV quintile versus overall. If the high-PV quintile is receiving materially worse outcomes, that is a Consumer Duty breach in the making.

---

## Wiring governance into the monitoring cycle

The `ModelCard` in `insurance-governance` carries the monitoring trigger structure that connects your pricing model to the outcomes monitoring MI. The key fields are `monitoring_triggers` and `trigger_actions`:

```python
from insurance_governance import MRMModelCard, Assumption

card = MRMModelCard(
    model_id="motor-freq-tppd-v3",
    model_name="Motor TPPD Frequency Model",
    version="3.0.2",
    model_class="pricing",
    intended_use="Rate pure premium for UK private motor TPPD, direct and aggregator channels",
    customer_facing=True,
    gwp_impacted=180_000_000,
    monitoring_frequency="Monthly",
    monitoring_triggers={
        "ae_ratio":              0.05,   # A/E deviation > 5% triggers review
        "loss_ratio_channel_gap": 0.15,  # channel loss ratio gap > 15pp
        "acceptance_rate_drop":  0.03,   # acceptance rate fall > 3pp vs prior year
        "pv_red_band_pct":       0.10,   # >10% of book in red PV band
    },
    trigger_actions={
        "ae_ratio":              "Notify Chief Actuary; commence ad-hoc validation",
        "loss_ratio_channel_gap": "Fair value assessment review; board notification within 30 days",
        "acceptance_rate_drop":  "Claims audit; notify Conduct Risk",
        "pv_red_band_pct":       "Fairness audit via DoubleFairnessAudit; escalate to Consumer Duty Champion",
    },
    assumptions=[
        Assumption(
            description="Model trained on 2021-2024 data; performance stable across cohorts",
            risk="MEDIUM",
            mitigation="Monthly A/E monitoring; annual refit if AE ratio >5%",
        ),
        Assumption(
            description="No material proxy discrimination in rating factors",
            risk="HIGH",
            mitigation="Quarterly ProxyVulnerabilityScore run; annual FairnessAudit",
        ),
    ],
)
```

The `monitoring_triggers` dict is machine-readable. Your monitoring pipeline reads it, runs the metrics, and compares each metric to its threshold. If a threshold is breached, the corresponding `trigger_actions` entry dictates the escalation path. This is not a sophisticated system — it is a standardised contract between the model card and the monitoring run, which means the compliance team can audit it without reading Python code.

### DoubleFairnessAudit as the annual Outcome 4 evidence pack

The conceptual insight that Consumer Duty forces on pricing is the distinction between action fairness and outcome fairness. Your pricing model might be entirely gender-blind at the point of quoting — action fairness — yet still produce loss ratios that differ significantly between men and women because the risk factors you use are correlated with gender in ways that translate into differential product value.

`DoubleFairnessAudit` in `insurance-fairness` v0.6.0 runs both measurements and recovers the Pareto front between them:

```python
from insurance_fairness import DoubleFairnessAudit

audit = DoubleFairnessAudit(n_alphas=20)
audit.fit(
    X_train,       # features excluding protected attribute
    y_premium,     # primary outcome: pure premium
    y_loss_ratio,  # fairness outcome: actual claims / premium paid
    S_gender,      # binary protected group indicator (0/1)
)
result = audit.audit()
print(result.summary())
# → prints Delta_1 (action fairness), Delta_2 (outcome fairness),
#   recommended Pareto-optimal policy, and deviation from current policy

print(audit.report())   # FCA evidence pack section — paste into board appendix
fig = audit.plot_pareto()
fig.savefig("pareto_front_motor_tppd_2025.png")
```

The `audit.report()` output is written to be dropped into a board appendix as the Price and Value evidence section. It quantifies, for your specific book, how much of your current Delta_1 (any residual action-level bias) is translating into Delta_2 (outcome-level differential value). If Delta_2 is large and your current policy is far from the Pareto front, you have a documented Consumer Duty gap with a quantified scale.

The empirical result from Bian et al. (2026) that motivated the library is worth restating here: on a real-world dataset, setting gender premiums equal (Delta_1 = 0) left outcome fairness (Delta_2) essentially unchanged. The Pareto-optimal policy was not the equality-constrained policy. Firms running only action-fairness checks are not auditing for what Consumer Duty actually requires.

---

## The MI structure the FCA wants to see

Pulling this together into the format the FCA was looking for in its multi-firm review:

**At product level, annually (for the board Consumer Duty report):**
- Fair value assessment with loss ratio by cohort, channel, and segment
- DoubleFairnessAudit Delta_1 and Delta_2 with Pareto front
- Vulnerable customer outcomes table (proxy vulnerability band distribution, claims and complaint rates by PV quintile)
- Any identified cross-segment value transfers, quantified

**At portfolio level, quarterly (for the Consumer Duty Champion MI):**
- Outcomes scorecard (RAG by product, all four outcomes)
- Claims acceptance rate and settlement time trends
- Proxy vulnerability band distribution trend
- Model card monitoring trigger status (which triggers fired, what actions are in train)

**At model level, monthly (for the pricing governance committee):**
- A/E ratio versus prior year and versus trigger threshold
- PSI for key rating factors
- Channel loss ratio gap versus trigger threshold
- `GovernanceReport` output from `insurance-governance` for any Tier 1 model that has had a trigger fire

The FCA found that firms generating the most data were not producing the best MI. The board pack that is most defensible under scrutiny is one where each metric has a defined tolerance, each tolerance breach has a documented escalation path, and the narrative explains what has been done — not just what has been observed.

---

## One thing to do before April

If your outcomes monitoring programme cannot answer "which quintile of customers is receiving the worst value on each product, and why?", it is not meeting the spirit of PRIN 2A.9. The FCA has signalled it will use multi-firm reviews to identify firms that need supervisory engagement. Being in the top half of the distribution on monitoring quality is not difficult — most firms are not there yet.

Running `ProxyVulnerabilityScore` against your current motor book takes under a minute. Running `DoubleFairnessAudit` on your frequency model's predictions takes a few minutes. If the results are clean, you have evidence. If they are not, better to find out internally than when the FCA asks.

```bash
uv add insurance-fairness insurance-governance
```

---

**Related posts:**
- Model Governance for UK Insurance Pricing: Building the MRC Pack — the full `insurance-governance` walkthrough: ModelCard, RiskTierScorer, GovernanceReport
- Proxy Discrimination in Insurance Pricing: The Côté Toolbox — per-policyholder proxy vulnerability metrics, PremiumSpectrum, and the parity cost decomposition
- [Your Book Has Shifted and Your Model Doesn't Know](/2026/03/02/your-book-has-shifted-and-your-model-doesnt-know/) — PSI-based drift detection that integrates with the monitoring trigger framework
