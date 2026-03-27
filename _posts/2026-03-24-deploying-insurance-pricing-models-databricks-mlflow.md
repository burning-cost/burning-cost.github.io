---
layout: post
title: "Safe Model Deployment for Insurance Pricing: Champion/Challenger with a Full Audit Trail"
date: 2026-03-24
categories: [techniques]
tags: [insurance-deploy, mlops, champion-challenger, icobs, enbp, audit, databricks, uk-personal-lines]
description: "insurance-deploy provides the champion/challenger infrastructure, audit trail, and ICOBS 6B compliance tooling that MLflow does not. Here is how to use it."
author: Burning Cost
---

The model handoff from pricing actuary to production is one of the most error-prone steps in insurance. The actuary fits a CatBoost model, validates it on held-out data, and declares it ready. Then it gets handed to whoever manages the rating engine - which in most UK personal lines teams means emailing a Python script and a set of CSV factor tables to someone in IT, who loads them into Radar or whatever system runs pricing, and sends back a confirmation that it is live.

There is no version control on which model is currently in production. There is no record of who approved the transition. If a regulator asks "which model was pricing renewals in Q3 2025?", the answer is probably a dig through email chains or a Confluence page that was last updated eighteen months ago. And there is certainly no automated check that the new model is not breaching ICOBS 6B.2.51R on renewal quotes.

This is the problem [`insurance-deploy`](/insurance-governance/) solves. It is not a cloud deployment platform. It does not push containers to Kubernetes or manage APIs. It is the infrastructure layer between "model fits well in validation" and "model is in production with a defensible paper trail" - the part that almost every pricing team either skips or builds badly.

## What MLflow gives you and what it does not

MLflow's model registry is the most common tool pricing teams reach for when they want to version control their models. It works well for the core use case: logging model artefacts, tracking experiments, and promoting model versions through staging and production stages. If you are running on Databricks, MLflow Unity Catalog makes model governance straightforward.

What MLflow does not provide is insurance-specific:

- It has no concept of a champion/challenger experiment with deterministic routing.
- It logs nothing about individual pricing decisions - which model priced which quote, at what price, with what ENBP.
- It has no ICOBS 6B compliance tracking. It cannot generate the audit report an SMF holder needs to sign for the FCA's annual ENBP attestation.
- It has no power analysis for how long your challenger needs to run before you have credible evidence to promote it.

`insurance-deploy` fills those gaps. It does not replace MLflow; they address different layers of the problem. MLflow handles the model experiment tracking and notebook-level artefact storage. `insurance-deploy` handles the operational pricing lifecycle that starts when a model is declared production-ready.

## The registry

The registry is the starting point. You register a fitted model object with a name and a version string. The library serialises it to disk using joblib and records a SHA-256 hash. Load-time verification checks that hash on every use.

```python
from catboost import CatBoostRegressor
from insurance_deploy import ModelRegistry

registry = ModelRegistry("./model_registry")

# Register the current champion
motor_v2 = registry.register(
    fitted_catboost_v2,
    name="motor_freq",
    version="2.0",
    metadata={
        "training_date": "2025-11-01",
        "training_rows": 487_000,
        "out_of_time_gini": 0.3241,
        "features": ["driver_age", "vehicle_group", "area", "ncd_years", "occupation"],
        "approved_by": "A.Smith",
        "approval_date": "2025-11-14",
    },
)

# Register the challenger
motor_v3 = registry.register(
    fitted_catboost_v3,
    name="motor_freq",
    version="3.0",
    metadata={
        "training_date": "2026-01-15",
        "training_rows": 512_000,
        "out_of_time_gini": 0.3389,
        "features": ["driver_age", "vehicle_group", "area", "ncd_years", "occupation", "annual_mileage"],
        "approved_by": "A.Smith",
        "approval_date": "2026-02-03",
    },
)
```

The `metadata` dict takes anything you want to store. We put training date, row count, out-of-time Gini, feature list, and approver details there. This metadata is written to `registry.json` alongside the model files and persists across Python sessions.

The registry is append-only. You cannot overwrite or delete a registered version. That is deliberate: the audit trail requires knowing what was deployed, not just what is deployed now.

```python
# Retrieve later
registry.get("motor_freq", "2.0")
# ModelVersion('motor_freq:2.0' [champion], registered=2025-11-14)

registry.list("motor_freq")
# [ModelVersion('motor_freq:2.0' ...), ModelVersion('motor_freq:3.0' ...)]
```

## Setting up the experiment

An `Experiment` pairs a champion with a challenger and defines the routing logic. The routing uses SHA-256(policy_id + experiment_name), taking the last 8 hex characters as an integer modulo 100. This is deterministic: given a policy ID and experiment name, the routing decision is always the same, and any auditor can recompute it from first principles.

```python
from insurance_deploy import Experiment

exp = Experiment(
    name="motor_freq_v3_vs_v2",
    champion=motor_v2,
    challenger=motor_v3,
    challenger_pct=0.10,
    mode="shadow",  # default and recommended
)
```

**Shadow mode is the default and the right choice for most teams.** In shadow mode, every quote is priced by the champion. The challenger scores in parallel, and its output is logged but never shown to the customer. You get a full-population comparison of champion vs challenger predictions with zero regulatory risk.

Live mode routes `challenger_pct` of policies to receive challenger prices. Before enabling live mode, read the FCA Consumer Duty (PRIN 2A) guidance on fair value. Two customers with identical risk profiles paying different prices simultaneously raises legitimate questions. The library will warn you:

```
UserWarning: Live mode routes real quotes to challenger model. This may raise
FCA Consumer Duty (PRIN 2A) fair value concerns... Obtain legal sign-off
before enabling live mode in production.
```

That warning is not boilerplate. We mean it.

## Logging quotes

Every pricing decision goes into an append-only SQLite log via `QuoteLogger`. The schema records the policy ID, experiment name, which arm (champion or challenger) the policy was routed to, which model version priced the quote, the quoted price, the ENBP, whether this was a renewal, and a UTC timestamp.

```python
from insurance_deploy import QuoteLogger

logger = QuoteLogger("./quotes.db")

# At quote time, for each policy:
arm = exp.route(policy_id)
live_model = exp.live_model(policy_id)    # champion in shadow mode
shadow_model = exp.shadow_model(policy_id)  # challenger in shadow mode

# Score both
live_price = live_model.predict(X_policy)[0] * base_rate
shadow_price = shadow_model.predict(X_policy)[0] * base_rate

# Log the live (customer-facing) quote
logger.log_quote(
    policy_id=policy_id,
    experiment_name=exp.name,
    arm=arm,
    model_version=live_model.version_id,
    quoted_price=live_price,
    enbp=enbp_for_this_policy,   # provide for renewals
    renewal_flag=is_renewal,
    exposure=0.5,                # 6-month policy
)

# Log the shadow prediction separately
logger.log_quote(
    policy_id=policy_id,
    experiment_name=exp.name,
    arm="challenger",
    model_version=shadow_model.version_id,
    quoted_price=shadow_price,
    renewal_flag=is_renewal,
    exposure=0.5,
)
```

The ENBP field is where `insurance-deploy` diverges most clearly from a generic MLops tool. ICOBS 6B.2.51R requires that renewal prices do not exceed the Equivalent New Business Price for an identical risk profile. The library does not calculate ENBP - that calculation is your pricing team's responsibility, per ICOBS 6B methodology - but it records the value you provide and flags breaches in real time.

When you log a renewal quote with an ENBP, the library sets `enbp_flag = 1` if `quoted_price <= enbp`, and `enbp_flag = 0` (a breach) otherwise. A breach triggers an immediate `UserWarning`. The FCA's 2023 multi-firm review found 83% of firms non-compliant with ICOBS 6B. That rate is almost certainly partly a record-keeping failure - firms that were compliant in practice but could not demonstrate it. `QuoteLogger` makes the demonstration straightforward.

## Tracking KPIs

After a few weeks of data, `KPITracker` gives you the operational view.

```python
from insurance_deploy import KPITracker

tracker = KPITracker(logger)

# Tier 1: immediately available
vol = tracker.quote_volume("motor_freq_v3_vs_v2")
# {
#   'champion': {'n': 9847, 'mean_price': 438.20, 'median_price': 412.00, ...},
#   'challenger': {'n': 9921, 'mean_price': 441.80, 'median_price': 414.50, ...}
# }

# Tier 2: at bind (a few days after quote)
hr = tracker.hit_rate("motor_freq_v3_vs_v2")
# {
#   'champion': {'quoted': 9847, 'bound': 3119, 'hit_rate': 0.317},
#   'challenger': {'quoted': 9921, 'bound': 3104, 'hit_rate': 0.313}
# }
```

The KPI tracker has four tiers. Tier 1 (quote volume, price distribution, ENBP compliance rate) is available immediately. Tier 2 (hit rate, GWP) is available a few days after quotes are issued. Tier 3 (claim frequency) is available at 6-9 months but carries an IBNR caveat - the library will warn you that at six months development, 30-40% of ultimate motor claims may not yet be reported. Tier 4 (developed loss ratio) requires 12+ months.

The power analysis is worth running before you start the experiment, not six months in when you have already collected data:

```python
pa = tracker.power_analysis(
    "motor_freq_v3_vs_v2",
    target_delta_lr=0.03,   # detect a 3pp loss ratio difference
    target_delta_hr=0.02,   # detect a 2pp hit rate difference
)
print(pa["lr_total_months_with_development"])
# 26.4
```

UK motor at a 10% challenger split, targeting a 3 percentage point loss ratio difference, needs roughly 26 months of data (14 months to collect enough bound policies, plus 12 months of claims development). Most teams do not know this when they start. Running the power analysis upfront sets realistic expectations with the pricing committee.

## Statistical comparison

When the data is mature, `ModelComparison` runs the formal tests.

```python
from insurance_deploy import ModelComparison

comp = ModelComparison(tracker)

# Block bootstrap on policy-level loss ratios (12m+ development)
result = comp.bootstrap_lr_test(
    "motor_freq_v3_vs_v2",
    n_bootstrap=10_000,
    development_months=12,
    seed=42,
)
print(result.summary())
# Test: bootstrap_lr_test | Experiment: motor_freq_v3_vs_v2
# Champion estimate:  0.6821 (n=3119)
# Challenger estimate: 0.6594 (n=3104)
# Difference (challenger - champion): -0.0227
# 95% CI: [-0.0411, -0.0044]
# p-value: 0.0082
#
# Conclusion: CHALLENGER_BETTER
# Recommendation: Challenger shows significantly better loss_ratio
# (p=0.008, 95% CI [-0.0411, -0.0044]). Human review recommended
# before promotion. Document the promotion decision with reviewer
# name and date.
```

The bootstrap resamples at policy level, preserving within-policy correlation. The conclusion is never automatic promotion. The library surfaces evidence; the pricing actuary and their sign-off authority decide. If you want an early signal before loss ratio data matures, `frequency_test()` runs a Poisson GLM comparison at 6-9 months.

## The ENBP audit report

This is the output that compliance teams and SMF holders actually need. `ENBPAuditReport.generate()` produces a markdown report covering: total quotes by arm, ENBP compliance rate by arm, any breaches with policy-level detail, routing audit (with the hash methodology explained), and an attestation section for the SMF holder to sign.

```python
from insurance_deploy import ENBPAuditReport

reporter = ENBPAuditReport(logger)
md = reporter.generate(
    experiment_name="motor_freq_v3_vs_v2",
    period_start="2026-01-01",
    period_end="2026-03-31",
    firm_name="Acme Insurance Ltd",
    smf_holder="J. Brown (SMF7)",
)

# In a Databricks notebook:
displayHTML(md)

# Or write to file for inclusion in governance packs:
with open("q1_2026_enbp_audit.md", "w") as f:
    f.write(md)
```

The report includes a section on routing methodology that reads: "Routing is deterministic: SHA-256(policy_id + experiment_name), last 8 hex characters modulo 100. Any routing decision can be recomputed independently from policy_id and experiment_name." That sentence matters in a regulatory context. A routing methodology that an auditor cannot independently verify is a compliance liability.

## Querying the log in Polars

For ad hoc analysis, `QuoteLogger.to_polars()` returns any of the three tables as a Polars DataFrame:

```python
quotes_df = logger.to_polars("quotes")
# shape: (19768, 11)
# columns: id, policy_id, experiment_name, arm, model_version,
#          quoted_price, enbp, renewal_flag, enbp_flag, exposure, timestamp

# Mean price by arm and renewal status
(
    quotes_df
    .group_by(["arm", "renewal_flag"])
    .agg(
        pl.col("quoted_price").mean().alias("mean_price"),
        pl.col("enbp_flag").mean().alias("enbp_compliance_rate"),
        pl.len().alias("n"),
    )
    .sort(["arm", "renewal_flag"])
)
```

## What this does not replace

`insurance-deploy` is not a serving infrastructure. It does not run an HTTP endpoint. It does not manage Databricks job clusters or handle real-time scoring at scale. For those problems, the right tools are Databricks Model Serving, MLflow's serving layer, or a FastAPI wrapper around your model.

It also does not calculate ENBP. The ICOBS 6B methodology for determining the Equivalent New Business Price for a specific customer on renewal is your pricing team's responsibility. The library records and audits whatever value you pass in.

What it does is make the paper trail automatic. Every quote, every routing decision, every ENBP comparison is logged without any manual intervention. The audit report writes itself. The bootstrap test runs in ten seconds. The power analysis runs before the experiment starts. These are the parts that UK pricing teams typically either skip or build badly as ad hoc scripts that someone has to remember to run.

## Installation

```bash
uv add insurance-deploy
```

Python 3.10 or later. No cloud dependencies. The SQLite database runs locally or on any shared filesystem that Databricks notebooks can write to - a mounted object storage path works fine for multi-notebook environments.

The source is at [github.com/burning-cost/insurance-deploy](https://github.com/burning-cost/insurance-deploy).

- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/)
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/)
