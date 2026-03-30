---
layout: post
title: "PS21/5 End-to-End: Renewal Pricing Optimisation That Actually Satisfies the FCA"
date: 2026-03-25
categories: [pricing, regulation, libraries]
tags: [PS21-5, ENBP, renewal-pricing, causal-inference, causal-forest, CATE, portfolio-optimisation, fairness, governance, fca, motor, home, python, insurance-causal, insurance-optimise, insurance-fairness, insurance-governance]
description: "The complete PS21/5 compliance workflow: CATE estimation with insurance-causal, ENBP-constrained optimisation with insurance-optimise, fairness audit with insurance-fairness, and sign-off pack with insurance-governance."
---

Most UK pricing teams implemented PS21/5 the same way: build the renewal price as usual, then clip anything above ENBP at the end. That approach is legally sufficient. It is not optimal, and it misses what the FCA was actually asking for.

The ENBP cap, bolted on as a post-hoc check, does not redistribute the margin that the constraint removes. It just cuts it. A renewal book where 15% of policies are sitting exactly at the ENBP ceiling has a pricing strategy that is running close to the regulatory edge, and it is leaving profit on the table because the optimisation was not done jointly. The commercially correct way to comply with PS21/5 is to feed ENBP as a hard constraint into the optimiser — the solver then actively redistributes margin to compliant policies rather than blindly discarding it.

This post shows the complete workflow: causal CATE estimation, customer segmentation by price sensitivity, constrained optimisation with ENBP enforcement, fairness checks, and a governance sign-off pack.

Four libraries: [`insurance-causal`](/insurance-causal/), [`insurance-optimise`](/insurance-optimise/), [`insurance-fairness`](/insurance-fairness/), [`insurance-governance`](/insurance-governance/).

```bash
pip install insurance-causal insurance-optimise insurance-fairness insurance-governance
```

---

## Step 1: Estimate per-customer CATE

The input to any renewal optimiser is a price elasticity — the rate at which renewal probability changes with price. The question is where that elasticity comes from.

An OLS regression of renewal indicator on log price change is wrong. The same risk factors that drive your technical price also drive whether a customer renews independently of price. High-risk customers are harder to retain for reasons that have nothing to do with what you charge them. OLS conflates the causal price effect with this risk-lapse correlation, and the resulting elasticity is biased: typically underestimated for high-risk segments and overestimated for loyal, low-risk customers.

The correct approach is Double Machine Learning (DML) with heterogeneous treatment effects via a causal forest. DML partials out the influence of observed confounders from both the outcome and the treatment before estimating the causal effect. The causal forest gives per-customer estimates rather than a population average.

```python
from insurance_causal.causal_forest import (
    HeterogeneousElasticityEstimator,
    HeterogeneousInference,
)

confounders = ["age_band", "ncd_years", "vehicle_group", "channel", "postcode_band"]

est = HeterogeneousElasticityEstimator(n_estimators=200, catboost_iterations=300)
est.fit(
    df,
    outcome="renewed",
    treatment="log_price_change",
    confounders=confounders,
)

# Per-customer CATE: d(P(renew))/d(log_price_change)
cates = est.cate(df)

# Formal test that heterogeneity is real (BLP), and GATES by quintile
inf = HeterogeneousInference(n_splits=100, k_groups=5)
hte_result = inf.run(df, estimator=est, cate_proxy=cates)
print(hte_result.summary())
hte_result.plot_gates()
```

The `HeterogeneousInference` step matters. Without the BLP test, the CATE estimates are point estimates without statistical guarantees — useful for ranking, but not for formal evidence. BLP (Best Linear Predictor) tests whether the CATE proxy is a significant predictor of heterogeneity. GATES reports the average treatment effect for each quintile of the CATE distribution. In a typical UK motor book, you will see a 3–4x spread between the most and least price-sensitive quintiles: NCD 0–1 customers on PCW tend to be much more elastic than NCD 4–5 direct customers.

For a detailed walkthrough of the causal forest workflow, see [OLS Elasticity in a Formula-Rated Book Measures the Wrong Thing](/2026/03/14/causal-price-elasticity-for-uk-renewal-pricing/).

---

## Step 2: Segment customers by price sensitivity

The GATES output from `HeterogeneousInference` tells you whether and where heterogeneity exists. Before feeding CATEs to the optimiser, it is worth checking the CLAN (Characteristics of the Least-Affected and Most-Affected) to understand the business logic behind the segments:

```python
# CLAN: which observable characteristics describe the high/low elasticity tails?
hte_result.plot_clan()
```

In practice, CLAN output for UK motor typically identifies channel (PCW vs direct) and NCD level as the dominant differentiators. This matters for the FCA. The Consumer Duty (PRIN 2A) requires that pricing does not produce systematically worse outcomes for groups defined by protected characteristics. If price sensitivity happens to correlate strongly with deprivation, age, or gender, then optimising against unmodified CATE estimates could create an outcome fairness problem even where ENBP compliance is maintained. We address this in Step 4.

For now, add the CATE estimates to the policy frame:

```python
df = df.with_columns(
    pl.Series("elasticity", cates)
)
```

---

## Step 3: Optimise renewal prices under PS21/5 constraints

With per-customer elasticity estimates in hand, we solve the joint optimisation problem. The `PortfolioOptimiser` takes technical price, expected loss cost, baseline renewal probability, elasticity, and the ENBP for each renewal policy. The `ConstraintConfig` expresses all the constraints simultaneously; the solver (SLSQP with analytical Jacobians) finds the multiplier vector that maximises expected profit subject to all of them.

```python
import numpy as np
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

config = ConstraintConfig(
    lr_max=0.72,           # loss ratio ceiling
    retention_min=0.82,    # minimum expected renewal retention
    max_rate_change=0.20,  # ±20% per-policy guardrail
    enbp_buffer=0.02,      # 2% headroom below ENBP ceiling
    technical_floor=True,  # no policy below cost
)

opt = PortfolioOptimiser(
    technical_price=df["technical_price"].to_numpy(),
    expected_loss_cost=df["expected_loss_cost"].to_numpy(),
    p_demand=df["p_renew_current"].to_numpy(),
    elasticity=df["elasticity"].to_numpy(),
    renewal_flag=df["is_renewal"].to_numpy(),
    enbp=df["enbp"].to_numpy(),
    prior_multiplier=df["prior_multiplier"].to_numpy(),
    constraints=config,
)

result = opt.optimise()
print(result)
```

The `enbp_buffer=0.02` parameter sets the effective ENBP upper bound to `ENBP × 0.98` rather than `ENBP × 1.0`. We recommend maintaining a small buffer. The FCA's rule is a hard ceiling, and ENBP estimates carry their own uncertainty — if your new business quote engine is periodically re-rated, there is a risk that a policy priced exactly at ENBP today will breach on re-rate tomorrow.

The result exposes the key PS21/5 diagnostic you should be reporting to your Board:

```python
# What fraction of the book would breach ENBP without the constraint?
n_at_ceiling = result.summary_df["enbp_binding"].sum()
pct_at_ceiling = 100 * n_at_ceiling / len(result.summary_df)
print(f"Policies at ENBP ceiling: {pct_at_ceiling:.1f}%")

# Portfolio metrics
print(f"Expected loss ratio:  {result.expected_loss_ratio:.3f}")
print(f"Expected retention:   {result.expected_retention:.3f}")
print(f"Expected profit:      £{result.expected_profit:,.0f}")
```

If `pct_at_ceiling` is above 15%, the book is running close to the regulatory edge. That is not necessarily a compliance problem — it may reflect that your technical prices are well-calibrated and the market is competitive — but it warrants explicit sign-off. The audit trail is JSON:

```python
import json
print(json.dumps(result.audit_trail, indent=2, default=str))
```

For the full treatment of the optimisation problem, see [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/).

---

## Step 4: Run fairness checks

There are two distinct fairness questions here, and they are not the same.

The first is **action fairness**: does the pricing process treat customers with similar risk identically, regardless of protected characteristics? This is the traditional proxy discrimination test. The second is **outcome fairness**: does the pricing product deliver equivalent value across customer groups after it is live? Consumer Duty Outcome 4 (Price and Value) requires the latter, not merely the former. Equalising premiums across, say, gender (action fairness = 0) does not equalise loss ratios across groups (outcome fairness may still be violated).

We run both. For the action fairness check, `detect_proxies` tests whether your rating factors are acting as proxies for protected characteristics:

```python
import polars as pl
from insurance_fairness import detect_proxies, FairnessAudit

# Check whether rating factors proxy for deprivation and age
proxy_result_deprivation = detect_proxies(
    df=df,
    protected_col="deprivation_quintile",
    factor_cols=["vehicle_group", "postcode_band", "occupation", "channel"],
)
proxy_result_deprivation.summary()

proxy_result_age = detect_proxies(
    df=df,
    protected_col="age_band",
    factor_cols=["vehicle_group", "ncd_years", "occupation", "channel"],
)
proxy_result_age.summary()
```

A `ProxyDetectionResult` with `flagged_factors` containing `channel` is particularly significant in a PS21/5 context: if PCW customers are systematically younger or from higher-deprivation postcodes (both plausible in UK motor), then an optimiser that takes more margin from inelastic segments may inadvertently disadvantage a protected group.

For outcome fairness, we use `FairnessAudit` against the optimised premiums:

```python
audit = FairnessAudit(
    model=pricing_model,
    data=df.with_columns(pl.Series("optimised_premium", result.new_premiums)),
    protected_cols=["age_band", "deprivation_quintile"],
    prediction_col="optimised_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
)
fairness_report = audit.run()
fairness_report.summary()
fairness_report.to_markdown("fairness_audit.md")
```

The `FairnessAudit` runs calibration-by-group, demographic parity, Gini by group, and Theil index across all protected characteristics simultaneously. For the Consumer Duty narrative, the Gini-by-group numbers are the ones the FCA will scrutinise: is the pricing model's predictive accuracy materially worse for any protected group?

For a detailed treatment of the proxy discrimination workflow, see [FCA Proxy Discrimination Testing in Python](/2026/03/22/fca-proxy-discrimination-python-testing-guide/).

---

## Step 5: Generate a sign-off pack

The code is not enough. A PS21/5 compliance programme needs a governance trail: who reviewed the optimisation, what constraints were active, what the outcome of the fairness checks was, and when the next review is scheduled. The `insurance-governance` package structures this as an MRM model card and a committee-ready report.

```python
from insurance_governance import (
    MRMModelCard,
    Assumption,
    Limitation,
    RiskTierScorer,
    ModelInventory,
    GovernanceReport,
)

card = MRMModelCard(
    model_id="renewal-optimiser-v1",
    model_name="PS21/5-Compliant Renewal Pricing Optimiser",
    version="1.0.0",
    model_class="pricing",
    intended_use=(
        "Per-policy renewal price optimisation for private motor. "
        "ENBP constraint enforced per FCA PS21/5 / ICOBS 6B.2. "
        "Not for commercial lines."
    ),
    assumptions=[
        Assumption(
            description="ENBP estimates derived from Q4 2025 new business quote engine",
            risk="HIGH",
            mitigation="Quarterly back-test of ENBP estimates vs live NB quotes; re-run if delta > 3%",
        ),
        Assumption(
            description="Price elasticity estimated from 2023-2024 renewal data; pre-inflation cycle",
            risk="MEDIUM",
            mitigation="Re-estimate DML elasticity annually; monitor via A/E on retention post-implementation",
        ),
        Assumption(
            description="Log-linear demand model (constant elasticity)",
            risk="LOW",
            mitigation="Logistic demand model available as fallback; tested in scenario analysis",
        ),
    ],
    limitations=[
        Limitation(
            description="ENBP quality: constraint effectiveness is bounded by ENBP estimation accuracy",
        ),
        Limitation(
            description="Selection bias: elasticity estimates attenuated by churned customers absent from renewal data",
        ),
    ],
)

scorer = RiskTierScorer()
tier = scorer.score(
    gwp_impacted=150_000_000,
    model_complexity="high",
    deployment_status="champion",
    regulatory_use=True,
    external_data=False,
    customer_facing=True,
)

inventory = ModelInventory("mrm_registry.json")
inventory.register(card, tier)

report = GovernanceReport(
    card=card,
    tier=tier,
    validation_results={
        "overall_rag": "GREEN",
        "run_date": "2026-03-25",
        "gini": 0.43,
        "ae_ratio": 1.02,
    },
)
report.save_html("renewal_optimiser_governance_pack.html")
```

The HTML report is print-to-PDF ready. The two assumptions marked `HIGH` and `MEDIUM` risk are the ones the Model Risk Committee should be asking about. In practice, ENBP quality is the weak point of most PS21/5 implementations: the regulatory constraint is only as tight as your ability to accurately estimate what you would charge the same customer as new business through the same channel. If that estimate is noisy, the constraint is noisy.

For the full governance workflow, see [One Package, One Install: Solvency II Article 121 / PRA SS3/18 Model Governance Unified](/2026/03/14/insurance-governance-unified-pra-ss123-validation/). Note: insurer model governance is governed by Solvency II Article 121 and PRA SS3/18, not SS1/23 (which applies to banks).

---

## What you actually have at the end

Run through, the five steps produce:

1. Per-customer CATE estimates with formal BLP heterogeneity tests and GATES by quintile
2. An SLSQP-optimised multiplier vector that maximises expected profit subject to ENBP, LR, retention, rate change, and technical floor constraints simultaneously
3. A `result.audit_trail` JSON with constraint values, shadow prices, solver convergence, and the fraction of policies binding at ENBP
4. Proxy detection results for deprivation and age; `FairnessAudit` covering calibration, demographic parity, Gini, and Theil across protected groups
5. An MRM model card with risk-rated assumptions and an HTML governance pack

That is a defensible PS21/5 compliance programme. It is not a legal opinion. Whether any specific set of renewal prices satisfies ICOBS 6B.2 is a judgement call that requires review by your legal and actuarial leads — code cannot substitute for that. What the code can do is make the inputs to that review precise, repeatable, and auditable.

The constraint we are most often asked about is ENBP quality. Quarterly back-tests comparing ENBP estimates to live new business quotes are non-negotiable. A 5% systematic error in ENBP estimates means you cannot be confident the constraint is binding where you think it is. The `enbp_buffer` parameter gives you a safety margin, but it should not substitute for measuring the estimation error directly.

---

*All four libraries used in this post are open source and installable via pip. The PS21/5 / ICOBS 6B.2 rules apply to home and motor renewals in the UK. FCA EP25/2 (2025) estimated that the GIPP remedies saved UK motor consumers approximately £1.6 billion over the first three years of implementation.*
