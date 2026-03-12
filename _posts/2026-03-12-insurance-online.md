---
layout: post
title: "Bandit Algorithms for GIPP-Compliant Price Experimentation"
date: 2026-03-12
categories: [libraries, pricing, experimentation]
tags: [bandits, UCB, Thompson-sampling, LinUCB, GIPP, FCA, price-experimentation, ENBP, audit-trail, insurance-online, python]
description: "The FCA found 28 firms non-compliant with PS21/5 GIPP price differentiation rules in 2023-24. The problem isn't intent — it's the absence of a principled, documented framework for the rate adjustments that happen at quotation time. insurance-online brings bandit algorithms to insurance with GIPP constraints, ENBP ceilings, and tamper-evident audit logs."
post_number: 78
---

The FCA's review of General Insurance Pricing Practices (PS21/5) found 28 firms non-compliant in 2023-24. The violations were not primarily about charging renewers more than new business customers — that was explicitly banned and most firms fixed it. The violations were subtler: A/B tests that discriminated by protected characteristic, rate adjustments that lacked documented methodology, and experiments that accumulated long records of who was charged what without any statistical framework justifying the allocation.

Price experimentation is necessary. You cannot calibrate a demand model without observational or experimental data on how customers respond to prices. The question is how to do it in a way that is both statistically sound and FCA-evidenced.

[`insurance-online`](https://github.com/burning-cost/insurance-online) implements bandit algorithms for insurance price experimentation with GIPP PS21/5 compliance built in from the first line.

```bash
uv add insurance-online
```

## Why bandits, not A/B tests

A fixed A/B test allocates customers uniformly across arms for a predetermined period. Bandits adapt the allocation to exploit arms that perform better while continuing to explore — the explore-exploit trade-off. For insurance pricing:

- You don't want to hold 50% of customers on an unprofitable rate for six weeks waiting for statistical power
- Claim frequency outcomes take months to develop; conversion and retention outcomes are available immediately
- Customer segments have heterogeneous responses; a single best arm is a simplification

The trade-off is interpretability. Bandits require more careful documentation to satisfy FCA requirements on equal treatment. `insurance-online` builds that documentation into the algorithm.

## Thompson Sampling with conjugate priors

For conversion experiments (binary outcome: did the customer buy?), Thompson Sampling with a Beta-Bernoulli conjugate prior is the natural choice.

```python
from insurance_online import ThompsonPolicy, BetaBernoulliPrior

policy = ThompsonPolicy(
    arms=["price_A", "price_B", "price_C"],
    prior=BetaBernoulliPrior(alpha=1.0, beta=1.0)
)

# At quotation time
arm = policy.select_arm(context=customer_features)

# After outcome is observed
policy.update(arm=arm, reward=converted)
```

For claim frequency (Poisson outcome), the Gamma-Poisson conjugate prior:

```python
from insurance_online import ThompsonPolicy, GammaPoissonPrior

policy = ThompsonPolicy(
    arms=["rate_group_1", "rate_group_2"],
    prior=GammaPoissonPrior(alpha=1.0, beta=1.0),
    exposure_weighted=True
)
```

## UCB1 for deterministic exploration

UCB1 maintains an upper confidence bound on each arm's expected reward and selects the arm with the highest UCB. It has stronger theoretical guarantees (Auer et al. 2002 logarithmic regret bound) and produces more interpretable decisions — each allocation can be justified by reference to the arm's confidence interval.

```python
from insurance_online import UCB1Policy

policy = UCB1Policy(
    arms=["rate_A", "rate_B", "rate_C"],
    exploration_param=2.0  # standard sqrt(2) UCB
)
```

## LinUCB for contextual experimentation

When customers have covariates, LinUCB (Li et al. 2010) learns a linear reward model per arm and maintains per-arm confidence ellipsoids. Sherman-Morrison rank-1 updates make this efficient for high-volume quotation systems.

```python
from insurance_online import LinUCBPolicy

policy = LinUCBPolicy(
    arms=["price_low", "price_mid", "price_high"],
    context_dim=len(customer_features),
    alpha=1.0  # exploration parameter
)

arm = policy.select_arm(context=customer_features)
policy.update(arm=arm, context=customer_features, reward=outcome)
```

## GIPP compliance constraints

The GIPP PS21/5 constraints are not soft preferences — they are hard constraints on every arm selection.

```python
from insurance_online import GIPPConstraint, FairnessConstraint, SafetyConstraint

# ENBP ceiling: no arm may expect to harm the customer
enbp_constraint = GIPPConstraint(
    enbp_model=fitted_enbp_model,
    max_expected_harm=0.0  # strict non-harm
)

# Fairness: allocation must not discriminate by protected characteristic
fairness_constraint = FairnessConstraint(
    protected_col="postcode_deprivation_decile",
    max_chi2_divergence=3.84  # 5% significance
)

# Safety: no arm with mean reward below LCB floor
safety_constraint = SafetyConstraint(
    min_expected_conversion=0.05
)

policy = ThompsonPolicy(
    arms=["A", "B", "C"],
    constraints=[enbp_constraint, fairness_constraint, safety_constraint]
)
```

If an arm violates a constraint, it is excluded from selection for that customer. The audit log records the constraint that was triggered.

## Tamper-evident audit log

Every arm selection, outcome update, and constraint trigger is written to a SQLite database with WAL mode and SHA-256 arm fingerprints. The fingerprint hashes the arm configuration so you can detect post-hoc changes to rate definitions.

```python
from insurance_online import AuditLog

audit = AuditLog(path="experiment_audit.db")
policy = ThompsonPolicy(arms=[...], audit_log=audit)

# Generate FCA evidence report
from insurance_online import ComplianceReport
report = ComplianceReport(audit_log=audit)
report.save("gipp_compliance_evidence.html")
```

The compliance report includes: arm allocation statistics by customer segment, protected characteristic balance tests, ENBP distribution by arm, exploration-exploitation breakdown, and period-by-period audit trail.

## What this covers and what it doesn't

`insurance-online` handles the within-session quotation decision: given this customer arriving now, which price to show. It does not handle:

- **PCW rank effects**: your price is one of several displayed simultaneously; the rank effect means arm rewards are not independent of competitor arms. This requires a competitive bandit model. That's out of scope for v1.
- **Non-stationary rewards**: seasonal claim frequency shifts make the stationary reward assumption approximate. Use Thompson Sampling with discounted updates if you expect non-stationarity.
- **Multi-objective optimisation**: conversion and long-run CLV have different time horizons. Scalarising them requires a utility function. The ENBP constraint proxies for CLV protection but it's not a full multi-objective solution.

---

`insurance-online` is available on [PyPI](https://pypi.org/project/insurance-online/) and [GitHub](https://github.com/burning-cost/insurance-online). 168 tests. The Databricks notebook runs a synthetic conversion experiment with GIPP constraints enabled.

Related: [rate-optimiser](https://burning-cost.github.io/2026/02/21/constrained-rate-optimisation-efficient-frontier/) for offline rate optimisation, [insurance-dro](https://burning-cost.github.io/2026/03/25/insurance-dro/) for robust optimisation, [insurance-deploy](https://burning-cost.github.io/2026/03/15/your-champion-challenger-test-has-no-audit-trail/) for champion/challenger deployment with audit trail.
