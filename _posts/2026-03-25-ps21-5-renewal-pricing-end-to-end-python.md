---
layout: post
title: "PS21/5 Renewal Pricing End to End in Python"
date: 2026-03-25
categories: [regulation, pricing, tutorial]
tags: [FCA, PS21/5, ENBP, renewal-pricing, causal-inference, portfolio-optimisation, fairness, governance, python, insurance-causal, insurance-optimise, insurance-fairness, insurance-governance, tutorial]
description: "A complete Python workflow for PS21/5 compliance: DML elasticity estimation, CATE heterogeneity, ENBP-constrained optimisation, proxy discrimination audit, and governance sign-off. Uses real library APIs throughout."
---

PS21/5 came into force on 1 January 2022. More than four years later, we still see pricing teams handling the equivalent new business price (ENBP) constraint as a post-hoc cap bolted onto the end of a renewal rating engine — a lookup table check that clips anything above the ENBP ceiling. This works, in the same way that braking at the last possible moment works. It is not a pricing strategy.

A proper PS21/5 implementation builds the ENBP constraint into the optimisation objective from the start, uses causal methods to estimate renewal price sensitivity (not the naive regression estimates that contaminate most retention models), checks the resulting pricing for proxy discrimination before it goes live, and produces a governance record that will survive an FCA review. This post walks through that workflow in Python.

We use four libraries: [insurance-causal](https://burning-cost.github.io/insurance-causal), [insurance-optimise](https://burning-cost.github.io/insurance-optimise), [insurance-fairness](https://burning-cost.github.io/insurance-fairness), and [insurance-governance](https://burning-cost.github.io/insurance-governance). All code uses their real APIs; nothing is invented.

---

## The problem with naive renewal elasticity

Before writing any code, it is worth being clear about what we are estimating and why naive approaches fail.

A retention model trained on observed renewals does not give you the causal price elasticity. It gives you the correlation between the price you charged and whether the customer renewed — confounded by every factor that simultaneously drove your pricing decision and the customer's renewal choice. High-risk customers get high premiums and also lapse more. NCD-5 customers get low premiums and also renew more. The naive estimate will understate elasticity for high-risk segments and overstate it for low-risk ones, which means any optimiser fed those estimates will systematically overprice the high-risk book and underprice the loyal book. You end up somewhere near PS21/5 non-compliance by accident.

The correct answer is double machine learning (DML): partial out the confounders from both the treatment (price) and the outcome (renewal), then estimate the effect in the residualised space. `insurance-causal`'s `PremiumElasticity` implements the Riesz representer variant (Chernozhukov et al. 2022), which avoids estimating the generalised propensity score — important for renewal portfolios where high-premium policies have renewal rates of 20–30% and the GPS is numerically unstable.

---

## Step 1: Elasticity estimation with DML

```python
import pandas as pd
import numpy as np
from insurance_causal.autodml import PremiumElasticity, SelectionCorrectedElasticity

# df has one row per renewing policy with:
#   X: confounders (age, ncd_years, vehicle_group, area, channel, prior_claims)
#   D: log_premium (the treatment — log of renewal premium offered)
#   Y: renewed (binary, 0/1)
#   exposure: years (all 1.0 for annual renewals, fractional for mid-term)

X = df[["age", "ncd_years", "vehicle_group", "area", "channel", "prior_claims"]]
D = df["log_premium"].values
Y = df["renewed"].values
exposure = df["exposure"].values

model = PremiumElasticity(
    outcome_family="poisson",   # Poisson for binary count with exposure
    n_folds=5,
    nuisance_backend="catboost",
    riesz_type="forest",
)
model.fit(X, D, Y, exposure=exposure)
result = model.estimate()

print(result.summary())
# AME: -0.41  (95% CI: -0.48, -0.34)
# Interpretation: a 1% premium increase reduces renewal probability by ~0.41pp
# on average across the portfolio
```

The AME is the portfolio-level average. For PS21/5 purposes we also need the segment-level effects — which customer types are most sensitive to renewal loading — because a uniform loading that is acceptable on average may still create a discriminatory outcome in specific segments.

```python
# Segment results: elasticity by channel × NCD band
segments = pd.cut(df["ncd_years"], bins=[0, 1, 3, 5], labels=["NCD0-1", "NCD2-3", "NCD4-5"])
segment_results = model.effect_by_segment(segments)

for seg in segment_results:
    print(f"{seg.label}: AME={seg.ate:.3f} (CI: {seg.ci_lower:.3f}, {seg.ci_upper:.3f})")
# NCD0-1:  AME=-0.68  (most sensitive — the segment walking their renewal)
# NCD2-3:  AME=-0.38
# NCD4-5:  AME=-0.19  (least sensitive — loyal customers who do not shop around)
```

This segment spread matters. A -0.41 portfolio average is hiding a 3.5x heterogeneity in sensitivity between new-to-NCD customers and loyal NCD-5 customers. The PS21/5 spirit (not just its letter) is that you should not be loading inertia from the NCD-5 segment into their renewal price.

---

## Step 2: CATE heterogeneity with Causal Forest

The `effect_by_segment()` call above uses pre-defined segments. If you want to discover which segments have the highest elasticity without imposing a partition, `HeterogeneousElasticityEstimator` fits a causal forest and returns a per-policy CATE estimate.

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
)

confounders = ["age", "ncd_years", "vehicle_group", "area", "channel", "prior_claims"]

est = HeterogeneousElasticityEstimator(
    n_estimators=500,
    catboost_iterations=300,
)
est.fit(
    df,
    outcome="renewed",
    treatment="log_premium",
    confounders=confounders,
)
cates = est.cate(df)   # per-policy CATE vector, shape (n,)

# Formal test of heterogeneity: best linear predictor (BLP)
# H0: no heterogeneity (beta_1 = 0 in the BLP of tau(X) on CATE proxy)
inf = HeterogeneousInference(n_splits=100, k_groups=5)
result = inf.run(df, estimator=est, cate_proxy=cates)
print(result.summary())
# BLP beta_1 = 1.24 (p < 0.001) — heterogeneity confirmed
# GATES by quintile:
#   Q1 (least sensitive): -0.12
#   Q5 (most sensitive):  -0.74
```

The BLP test (Chernozhukov, Demirer, Duflo & Fernandez-Val 2020) provides the formal evidence that heterogeneity exists — not just that the CATE estimates vary, but that they contain signal above noise. You want this in your governance documentation.

The practical output here is the per-policy `cates` vector. Feed it to the optimiser as the elasticity input, and the optimiser will price each policy according to its actual price sensitivity rather than a segment average.

---

## Step 3: Portfolio optimisation with ENBP constraints

`PortfolioOptimiser` in `insurance-optimise` takes the ENBP constraint as a first-class input. The `enbp` vector (new business equivalent price per policy) becomes the upper bound on renewal premiums; `enbp_buffer` lets you set a safety margin below that ceiling.

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

# ENBP vector: what a new customer with identical risk characteristics would pay
# Typically this comes from your new business rating engine run against the renewal book
enbp = df["enbp_estimate"].values
technical_price = df["technical_premium"].values
expected_loss = df["expected_loss_cost"].values
p_renewal_base = df["base_retention"].values  # from retention model at current price

config = ConstraintConfig(
    lr_max=0.72,            # aggregate loss ratio cap
    retention_min=0.82,     # floor on renewal retention rate
    max_rate_change=0.15,   # ±15% per policy year-on-year
    enbp_buffer=0.01,       # stay 1% inside the ENBP ceiling
    technical_floor=True,   # never price below technical
)

opt = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss,
    p_demand=p_renewal_base,
    elasticity=cates,           # per-policy CATE from Step 2
    renewal_flag=np.ones(len(df), dtype=bool),
    enbp=enbp,
    constraints=config,
)

result = opt.optimise()
print(result)
# OptimisationResult:
#   Status: optimal
#   Expected profit: £4.2M
#   Aggregate LR: 0.703
#   Retention rate: 0.836
#   Policies at ENBP ceiling: 2,847 (11.4%)
#   Policies at technical floor: 411 (1.6%)
```

The "policies at ENBP ceiling" figure is the key PS21/5 diagnostic. 11.4% of the book is constrained by the ENBP cap — these are the customers who, if you ignored the regulation, you would price above their ENBP. The optimiser has not just capped them; it has set them to the ENBP ceiling and redistributed the margin to compliant policies.

Note the `elasticity=cates` line: we are using the per-policy causal estimates from Step 2. If you use the naive OLS elasticity instead, the optimiser will target the wrong customers for retention discounts and overcharge the wrong ones. The ENBP constraint will catch the worst outcomes, but you will leave margin on the table throughout the distribution.

---

## Step 4: Proxy discrimination audit

Renewal pricing under PS21/5 has a discrimination risk that new business pricing does not: because you are deliberately differentiating between renewing and new customers by channel, inertia, and shopping behaviour, you can inadvertently proxy-discriminate against protected groups. PCW customers are disproportionately young. Loyal NCD-5 customers are disproportionately older. A retention loading applied uniformly by NCD band will have a disparate impact.

`detect_proxies` from `insurance-fairness` runs mutual information, partial correlation, and proxy R-squared against each rating factor:

```python
from insurance_fairness import detect_proxies, FairnessAudit

# First, check which rating factors proxy for age (a protected characteristic
# under Equality Act 2010, s.19 indirect discrimination)
proxy_result = detect_proxies(
    df=df_renewals,
    protected_col="age_band",      # binned age: "under_25", "25-50", "51-65", "over_65"
    factor_cols=["ncd_years", "channel", "vehicle_group", "area_code", "prior_claims"],
)
proxy_result.summary()
# Factor           MI score    Proxy R²    RAG
# channel          0.18        0.14        AMBER
# ncd_years        0.31        0.27        RED
# vehicle_group    0.09        0.06        GREEN
# area_code        0.06        0.04        GREEN
# prior_claims     0.12        0.09        AMBER
```

`ncd_years` flagging RED is expected — NCD is strongly correlated with age in any UK motor book, because older drivers have had more time to accumulate no-claims years. The question is not whether the correlation exists (it does), but whether the way you are using NCD in renewal pricing results in a systematically worse outcome for older customers as a group.

Run the full `FairnessAudit` to get calibration-by-group and demographic parity diagnostics on the optimised renewal premiums:

```python
# Build a minimal sklearn-compatible model wrapper for the optimised premiums
# (FairnessAudit expects a CatBoost model or a predict() callable)
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=renewal_pricing_model,        # your CatBoost or GLM
    data=df_renewals,
    protected_cols=["age_band", "gender"],
    prediction_col="optimised_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
)
report = audit.run()
report.summary()
report.to_markdown("renewal_fairness_audit_2026Q1.md")
```

If the proxy audit surfaces a RED flag you cannot eliminate (NCD is a legitimate risk factor; removing it would damage pricing accuracy), document the justification. The FCA's position in TR24/2 is that proxy discrimination requires proportionate justification, not that it is absolutely prohibited. The documentation has to show (a) you ran the audit, (b) you assessed the impact, and (c) you concluded the use of NCD is justified on actuarial grounds. The `FairnessAudit` report gives you all three.

---

## Step 5: Governance sign-off with GovernanceReport

The whole workflow is worthless if it is not documented in a form that survives an FCA review. `insurance-governance` provides `GovernanceReport` — the two-page executive summary for Model Risk Committee sign-off — and `MRMModelCard`, which is the structured metadata record.

```python
from insurance_governance import MRMModelCard, GovernanceReport
from insurance_governance import Assumption, Limitation

card = MRMModelCard(
    model_id="motor-renewal-optimiser-2026",
    model_name="UK Motor Renewal Portfolio Optimiser",
    version="1.2.0",
    model_class="pricing",
    intended_use=(
        "Set renewal premiums for UK private motor portfolio "
        "in compliance with FCA PS21/5 ENBP constraint. "
        "Not to be used for new business or commercial lines."
    ),
    not_intended_for=[
        "New business pricing",
        "Commercial motor",
        "Setting ENBP benchmarks (separate system)",
    ],
    target_variable="renewal_premium_multiplier",
    distribution_family="Gaussian",
    model_type="SLSQP constrained optimiser, CausalForest elasticity inputs",
    rating_factors=[
        "ncd_years", "age", "vehicle_group", "area_code",
        "channel", "prior_claims", "enbp_estimate",
    ],
    training_data_period=("2022-01-01", "2025-12-31"),
    development_date="2026-03-01",
    developer="Pricing Analytics",
    champion_challenger_status="champion",
    customer_facing=True,
    gwp_impacted=85_000_000,   # £85M GWP
    assumptions=[
        Assumption(
            description=(
                "ENBP estimates from new business engine are accurate "
                "to within ±3% for equivalent risk profiles."
            ),
            risk="HIGH",
            mitigation=(
                "Quarterly back-test of ENBP estimates vs actual new "
                "business quotes for equivalent risk profiles. "
                "Tolerance: mean absolute error < 5%."
            ),
        ),
        Assumption(
            description="DML elasticity estimates are stable quarter-on-quarter.",
            risk="MEDIUM",
            mitigation="Re-estimate elasticity model on rolling 12-month window each quarter.",
        ),
    ],
    limitations=[
        Limitation(
            description="Elasticity estimates unreliable for policies with < 3 years of renewal history.",
            impact="May overprice new policyholders at first renewal.",
            population_at_risk="Policies with policy_age < 3 years (approx. 18% of book).",
            monitoring_flag=True,
        ),
    ],
    approved_by=["Head of Pricing", "Chief Actuary"],
    approval_date="2026-03-20",
    regulatory_use=True,
)

gov_report = GovernanceReport(
    card=card,
    validation_results={
        "overall_rag": "GREEN",
        "run_id": "val-2026-Q1-001",
        "run_date": "2026-03-15",
        "gini": 0.38,
        "ae_ratio": 1.01,
        "psi_score": 0.07,
    },
)

gov_report.to_html("renewal_optimiser_governance_2026Q1.html")
gov_report.to_dict()   # for Confluence or MRC portal ingestion
```

The `to_html()` output is self-contained (no CDN dependencies), print-to-PDF ready, and covers model identity, risk tier, validation RAG, assumptions register with risk ratings, and outstanding issues. It is what the FCA would expect to see if they asked for your pricing model governance documentation.

---

## What this does not solve

Three problems sit outside this workflow.

**ENBP estimation quality.** Everything in Step 3 depends on how accurate your ENBP estimates are. If your new business engine does not price equivalent risks equivalently across channels, your ENBP benchmarks are wrong, and the ENBP ceiling is a fiction. Quarterly back-testing of ENBP estimates against actual new business quotes for equivalent risk profiles is not optional — it is the thing that makes the constraint mean something.

**Causal identification for renewals.** The DML approach in Step 1 handles confounding from observed covariates. It does not handle unobserved confounders — particularly, the fact that customers who are most price-sensitive have probably already left or negotiated. If your renewal book is systematically selected (the most elastic customers churned in prior years), your elasticity estimates will be attenuated. `SelectionCorrectedElasticity` in `insurance-causal` addresses this using the Riesz sample-selection approach (arXiv:2601.08643), but it requires instruments, which are rarely available in motor portfolios.

**The ENBP benchmark is not fair value.** PS21/5 says renewal premiums cannot exceed ENBP. It does not say ENBP is a fair price. If your new business pricing is itself unfair (proxying for protected characteristics, charging materially different prices for identical risks by channel), the ENBP cap inherits that unfairness. The fairness audit in Step 4 needs to run on the ENBP benchmarks as well as the renewal premiums.

---

## Further reading

- [What FCA PS25/21 Actually Means for Your Pricing Team](/2026/03/24/ps25-21-pricing-team-implications/) — governance calendar implications of the PROD 4 changes
- [Measuring Rate Change Impact with DiD and ITS in Python](/2026/03/25/measure-rate-change-impact-python-did-its/) — evaluating whether your PS21/5-compliant renewal prices actually changed retention
- [insurance-causal documentation](https://burning-cost.github.io/insurance-causal) — `PremiumElasticity`, `HeterogeneousElasticityEstimator`, `SelectionCorrectedElasticity`
- [insurance-optimise documentation](https://burning-cost.github.io/insurance-optimise) — `PortfolioOptimiser`, `ConstraintConfig`, ENBP constraint implementation
- [insurance-fairness documentation](https://burning-cost.github.io/insurance-fairness) — `FairnessAudit`, `detect_proxies`, proxy discrimination diagnostics
- [insurance-governance documentation](https://burning-cost.github.io/insurance-governance) — `GovernanceReport`, `MRMModelCard`, MRC pack generation
- FCA PS21/5 policy statement: [fca.org.uk/publication/policy/ps21-5.pdf](https://www.fca.org.uk/publication/policy/ps21-5.pdf)
