---
layout: post
title: "FCA Consumer Duty Outcome Testing: Automating the Evidence Pack"
date: 2026-04-01
categories: [regulation, governance]
tags: [FCA, consumer-duty, PRIN-2A, outcome-testing, insurance-governance, FCA-2026, fair-value, claims-outcomes, price-walking, GIPP, PS21-5, FG22-5, AI-governance, pricing-AI, UK-insurance, insurance-pricing, regulatory-compliance]
description: "The FCA's February 2026 insurance priorities report signals active supervisory review of AI pricing in Q1-Q2 2026. Under existing PRIN 2A, firms must already demonstrate Consumer Duty outcome monitoring — but most pricing teams don't have automated tooling for it. insurance-governance v0.2.0 ships OutcomeTestingFramework to close that gap."
author: Burning Cost
---

The FCA published its first dedicated insurance regulatory priorities report in February 2026 (Pub ref: 1-009074). Signed by Sarah Pritchard and Graeme Reynolds. Forty-plus portfolio letters replaced with one annual document. Four priorities. The one that matters most for pricing teams using AI is buried in Priority 3, under "Supporting Growth and Innovation":

*"We support firms' use of AI, but they must monitor outcomes for consumers closely."*

That sentence is not new guidance. It is a restatement of obligations that have existed since Consumer Duty (PRIN 2A) came into force in July 2023. What is new is the signal that follows it: the FCA will evaluate how firms use AI in underwriting, claims, and consumer services in Q1-Q2 2026, and will publish an AI Live Testing evaluation report by end of 2026.

The enforcement risk is not an AI-specific rule that you might be breaching. It is the FCA's supervisory review finding that a firm running AI pricing has no documented outcome monitoring. That triggers Consumer Duty enforcement under existing PRIN 2A. The FCA does not need to write new rules to act — the obligation to test outcomes is already there.

`insurance-governance` v0.2.0 adds an `OutcomeTestingFramework` subpackage to automate the evidence pack.

---

## What "outcome testing" actually means under Consumer Duty

The phrase "test outcomes" in the February report is shorthand for the monitoring framework in PRIN 2A.9 and Chapter 11 of FG22/5. The FCA's multi-firm review on insurance outcomes monitoring makes the operational expectation concrete. The requirement is a causal chain:

1. **Defined intended outcomes** — what your AI pricing model is supposed to deliver for customers
2. **Monitoring metrics** — data that tests whether those outcomes are delivered
3. **Identification of poor outcomes** — what triggers investigation
4. **Actions and escalation** — documented corrective steps when metrics show failure

The FCA's multi-firm review found that most insurers tracked timeliness and decline rates but not settlement value adequacy. It found "overly focused on processes being completed rather than on outcomes delivered." It found insufficient causal links between monitoring data and corrective action.

For AI pricing specifically, the relevant Consumer Duty outcome is the Price and Value outcome (PRIN 2A.3): price must be proportionate to benefits; firms must demonstrate fair value annually. A pricing model that systematically disadvantages a customer segment is an outcome failure under PRIN 2A — not just a model risk management issue.

The four Consumer Duty outcomes for pricing teams:
1. **Products and Services** — product meets target market needs
2. **Price and Value** — price proportionate to benefits; annual fair value assessment required
3. **Consumer Understanding** — communications tested for effectiveness
4. **Consumer Support** — claims and queries handled promptly and fairly

Claims outcomes (3 and 4) are in scope because your pricing model affects which customers claim and how. A model that prices segments differently will produce different claim populations — their claims experience is part of the outcome chain.

---

## What was built

New subpackage: `insurance_governance.outcome`, with six modules and 104 tests.

The design follows the existing `ModelValidationReport` pattern in the library: a high-level facade accepts your data and column names, runs all applicable tests, and produces HTML and JSON reports. The facade is `OutcomeTestingFramework`.

### The framework facade

```python
from insurance_governance import MRMModelCard
from insurance_governance.outcome import OutcomeTestingFramework, CustomerSegment
import polars as pl

card = MRMModelCard(
    model_id="motor-freq-v3",
    model_name="Motor Frequency v3",
    version="3.0.0",
)
df = pl.read_parquet("policies_2025q4.parquet")

# Define segments you care about
renewal_seg = CustomerSegment(
    name="Renewal",
    filter_fn=lambda df: df["policy_type"] == "renewal",
)
aggregator_seg = CustomerSegment(
    name="Aggregator",
    filter_fn=lambda df: df["channel"] == "aggregator",
    is_vulnerable=False,
)

framework = OutcomeTestingFramework(
    model_card=card,
    policy_data=df,
    period="2025-Q4",
    price_col="gross_premium",
    claim_amount_col="claims_paid",
    claim_outcome_col="claim_declined",       # binary: 1=declined, 0=paid
    days_to_settlement_col="days_to_settle",
    renewal_indicator_col="is_renewal",
    customer_segments=[renewal_seg, aggregator_seg],
)

results = framework.run()
print(framework.get_rag_status())   # RAG: GREEN / AMBER / RED

framework.generate("outcome_report_2025q4.html")
framework.to_json("outcome_report_2025q4.json")
```

The column arguments are all optional. Passing `None` for any column skips the corresponding test suite — so you can run only the tests relevant to your data. Claims teams that do not have settlement adequacy data can still run fair value ratio and timeliness tests.

### PriceValueMetrics

Three tests, each returning an `OutcomeResult`:

**Fair value ratio** (PRIN 2A.3): total claims paid divided by total premiums. The FCA does not mandate a specific threshold, but 0.70 is the common personal lines floor for triggering a fair value review. Below 0.70 generates a CRITICAL severity result with four documented corrective actions:

```python
from insurance_governance.outcome.metrics import PriceValueMetrics
import numpy as np

result = PriceValueMetrics.fair_value_ratio(
    premiums=premiums,
    claims_paid=claims_paid,
    expenses=expenses,
    period="2025-Q4",
    segment="Aggregator",
    threshold=0.70,
)
# result.passed, result.metric_value (claims ratio), result.corrective_actions
```

**Price dispersion by segment**: computes median premium per segment, flags if the max/min median ratio exceeds 1.50. This is a portfolio-level dispersion test, not a protected-characteristic fairness test — use `insurance_fairness` for Equality Act groups. A ratio above 1.50 across business segments warrants documented justification. The framework produces one informational result per segment and one summary pass/fail result.

**Renewal vs new business gap** (PS21/5 GIPP check): exposure-weighted mean renewal premium versus mean new business premium. The FCA's General Insurance Pricing Practices rules (effective January 2022) prohibit price walking — renewals must not be priced above the equivalent new business price. The test flags any gap above 5% (our de minimis tolerance) as a CRITICAL failure with explicit reference to PS21/5 audit obligations.

### ClaimsMetrics

Three further tests:

**Settlement value adequacy**: compares agreed settlements to independent reference valuations. Ratio below 0.95 (mean settlement / mean reference) triggers review. The FCA's multi-firm review specifically cited failure to monitor settlement value adequacy — tracking timeliness without this is incomplete. Reference valuations might be engineer assessments, third-party market values, or actuarial reserve estimates.

**Decline rate by segment**: compares claim decline rates across customer segments. A max/min ratio above 1.50 triggers a WARNING. This is not a protected-characteristic test — it is a Consumer Duty segment outcome check. Large disparities between, say, direct-written and delegated-authority claims populations warrant investigation regardless of protected characteristics. The test produces per-segment informational results plus a summary disparity result.

**Timeliness SLA**: percentage of claims settled within `sla_days` (default 5 for fast-track motor). Pass threshold is 80% compliance. Mean, median, and 90th percentile settlement times are included in the details for root-cause analysis:

```python
from insurance_governance.outcome.metrics import ClaimsMetrics

result = ClaimsMetrics.timeliness_sla(
    days_to_settlement=days_arr,
    period="2025-Q4",
    sla_days=10,   # complex claims — use appropriate SLA
)
print(result.extra["p90_days"])   # 90th percentile, key diagnostic
```

### OutcomeResult and RAG status

Every test produces an `OutcomeResult` dataclass:

```python
from insurance_governance.outcome.results import OutcomeResult

# OutcomeResult fields:
# outcome: str           — 'price_value' | 'claims'
# test_name: str
# passed: bool
# metric_value: float    — the computed metric
# threshold: float       — the threshold it was tested against
# period: str
# segment: str           — None for portfolio-wide
# details: str           — human-readable summary
# severity: Severity     — INFO / WARNING / CRITICAL
# corrective_actions: list[str]
# extra: dict            — test-specific additional data
```

The `corrective_actions` field is the critical piece for the FCA causal chain. Every failed test has pre-populated corrective actions referencing the specific regulatory obligation (PRIN 2A.3, PS21/5, etc.). These are starting points, not legal advice — teams should review and extend them. But they give the board champion something substantive to sign off rather than a bare metric.

RAG status rolls up across all results: any CRITICAL failure is RED, any WARNING is AMBER, all INFO passes are GREEN. `framework.get_rag_status()` returns the aggregate.

### Custom results via extra_results

The framework accepts custom `OutcomeResult` objects via `extra_results`. This is how you inject metrics that are not in the standard test suites — complaints rate, NPS score, manual qualitative assessments:

```python
from insurance_governance.outcome.results import OutcomeResult
from insurance_governance.validation.results import Severity

complaints_result = OutcomeResult(
    outcome="support",
    test_name="complaints_rate",
    passed=complaints_rate_per_1000 < 5.0,
    metric_value=complaints_rate_per_1000,
    threshold=5.0,
    period="2025-Q4",
    details=f"Complaints per 1,000 policies: {complaints_rate_per_1000:.2f}",
    severity=Severity.WARNING if complaints_rate_per_1000 >= 5.0 else Severity.INFO,
)

framework = OutcomeTestingFramework(
    ...
    extra_results=[complaints_result],
)
```

The HTML report includes these alongside the computed results with no distinction. The JSON sidecar captures everything for audit trail ingestion.

---

## What this does and does not do

The `OutcomeTestingFramework` is the live-data Consumer Duty evidence layer. It is not model validation. The existing `ModelValidationReport` in `insurance-governance` covers statistical model testing — performance metrics on held-out data, stability checks, discrimination analysis. That runs at model development time.

`OutcomeTestingFramework` runs against live policy data for a reporting period. It answers: "For the customers who actually received prices from this model in 2025-Q4, were the Consumer Duty outcomes delivered?" These are different questions. Both are required under Consumer Duty — the FCA explicitly distinguishes development-phase validation from ongoing outcome monitoring.

The framework also does not replace the annual fair value assessment that PRIN 2A.3 requires. The fair value ratio test is a quantitative signal, not a complete fair value assessment. A 0.72 claims ratio does not automatically mean fair value is demonstrated — there may be additional benefits, costs, or customer group considerations that require qualitative assessment. What it does is flag the cases that require investigation and provide a documented starting point.

---

## The FCA's actual enforcement risk

We want to be precise about what the February 2026 report does and does not change.

It does not create a new AI governance framework for insurers. PRA SS1/23 (Model Risk Management Principles) applies to banks, not insurers — any reference to it applying to GI or life firms is incorrect. The FCA has not published quantitative outcome testing thresholds, prescribed testing frequencies, or explainability standards. Technical detail is deferred to the AI Live Testing evaluation report expected by end of 2026.

What the report does is elevate "test outcomes" from an implied Consumer Duty obligation to a named FCA expectation, and then announces that the FCA will be looking at this in Q1-Q2 2026. The supervisory review mechanism under PRIN 2A does not require new rules. If the FCA's review finds a firm running AI pricing with no documented outcome monitoring, they have the tools to act under existing Consumer Duty obligations.

The practical minimum for being supervisory-ready is the causal chain: defined outcomes, monitoring metrics, evidence that you investigate failures, evidence that investigations produce actions. An HTML report with a RAG status and pre-populated corrective actions is not a full programme. But it is the foundation. An actuarial function that can walk into an FCA supervisory conversation with quarterly outcome testing reports for each AI pricing model is in a materially better position than one that cannot.

For firms already doing some form of outcome monitoring — most are, at least for claims timeliness — the value of the framework is standardisation and the JSON audit trail. Having all outcomes in one report, with consistent severity levels and corrective actions, is what the FCA's multi-firm review said good practice looked like.

---

## Integration with the existing library

`OutcomeTestingFramework` integrates with the existing `insurance-governance` library:

- Takes a `MRMModelCard` as its first argument — the same card used in `ModelValidationReport`, so the model is identified consistently across both development-phase validation and ongoing outcome monitoring
- Reuses `Severity` and `RAGStatus` from `validation.results` — no new enums
- The JSON output can be attached to `GovernanceReport` for a complete model governance evidence pack

```python
from insurance_governance import GovernanceReport, MRMModelCard
from insurance_governance.mrm.scorer import RiskTierScorer
from insurance_governance.outcome import OutcomeTestingFramework

# Development-phase: risk tier, model card, validation report
# (existing library, unchanged)

# Ongoing: outcome testing per quarter
framework = OutcomeTestingFramework(model_card=card, ...)
outcome_data = framework.to_dict()

report = GovernanceReport(
    model_card=card,
    risk_tier=tier,
    outcome_testing_results=outcome_data,  # attaches to the governance pack
)
```

---

## Installation

```bash
uv add "insurance-governance>=0.2.0"
```

The new subpackage is at `insurance_governance.outcome`. Clean re-exports:

```python
from insurance_governance.outcome import (
    OutcomeTestingFramework,
    CustomerSegment,
    OutcomeResult,
    PriceValueMetrics,
    ClaimsMetrics,
)
```

Source: [github.com/burning-cost/insurance-governance](https://github.com/burning-cost/insurance-governance). 104 tests across five test files.

The FCA's February 2026 report is at fca.org.uk, Pub ref: 1-009074. The Consumer Duty outcome monitoring multi-firm review is separately available. PRIN 2A.9 and FG22/5 Chapter 11 are the primary regulatory references.

---

## Related posts

- [insurance-governance: MRM Model Cards and Validation Reports](/governance/2026/03/25/insurance-governance-mrm-model-cards/) — the development-phase validation layer that OutcomeTestingFramework complements
- [The EU AI Act and Insurance Pricing Fairness](/regulation/fairness/2026/04/01/eu-ai-act-algorithmic-fairness-insurance-pricing-actuaries/) — the broader AI regulation context
- [FCA PS25/21 and Proxy Discrimination in Motor Pricing](/regulation/2026/03/28/fca-ps2521-proxy-discrimination-motor-pricing/) — the pricing practices regulatory background
