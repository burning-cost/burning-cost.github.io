---
layout: post
title: "Rate Change Heterogeneity with Panel Causal Forests and causalfe"
date: 2026-03-12
categories: [tutorials, pricing, causal-inference]
tags: [panel-data, causal-forest, fixed-effects, causalfe, heterogeneous-treatment-effects, rate-change, CFFE, FCA, DML, insurance]
description: "causalfe (arXiv:2601.10555, Jan 2026) implements Causal Forest Fixed Effects for panel data — the method for estimating heterogeneous treatment effects controlling for unobserved policyholder effects. This post shows how to use it for insurance rate change evaluation, where the unobserved factor is individual risk appetite."
post_number: 81
canonical_url: "https://burning-cost.github.io/2026/03/12/insurance-causal-panel/"
---

Rate change evaluation in insurance has a problem that neither standard A/B tests nor synthetic difference-in-differences fully solves: heterogeneous response.

When you raise rates by 8% on a cohort, you get an average response. Some customers churn. Some stay regardless. Some would have churned anyway because their NCD changed. Standard causal inference methods give you the average treatment effect — the average across all of these customers. They do not tell you which customers are rate-elastic, which are rate-inelastic, and which would have churned regardless of the rate movement.

Causal Forest Fixed Effects (CFFE), implemented in [`causalfe`](https://pypi.org/project/causalfe/) (Pu & Laber 2026, arXiv:2601.10555), combines panel data fixed effects — which absorb unobserved individual heterogeneity — with causal forest heterogeneous treatment effect estimation. We don't need to build a separate library for this: causalfe is already on PyPI.

```bash
pip install causalfe
```

## The unobserved heterogeneity problem in insurance panels

A three-year panel of your renewal book has a key feature: you observe the same policyholder in multiple years. Policyholders differ in their unobserved risk appetite — some are highly price-sensitive, some are not. That unobserved factor correlates with both the treatment (they receive larger rate increases because they're in a segment with worse loss experience) and the outcome (they churn because they're price-sensitive).

Standard causal forest, without fixed effects, confounds individual risk appetite with the treatment effect. Fixed effects partial out the unobserved individual-level variation before estimating treatment effect heterogeneity.

## Basic CFFE usage for rate change analysis

```python
import causalfe
import pandas as pd
import numpy as np

# Panel structure: one row per policy-year
# panel_id: unique customer identifier
# year: policy year
# W: treatment (rate change %, centred at 0)
# Y: outcome (1=churned, 0=retained)
# X: covariates (vehicle_age, ncb_years, years_on_book, region, ...)

df = pd.read_parquet("renewal_panel.parquet")

cffe = causalfe.CausalForestFE(
    n_estimators=500,
    min_samples_leaf=10,
    panel_id_col="policy_id",
    time_col="policy_year",
    n_jobs=-1
)
cffe.fit(
    X=df[covariates],
    W=df["rate_change_pct"],
    Y=df["churned"],
    panel_id=df["policy_id"],
    time=df["policy_year"]
)
```

## Estimating conditional average treatment effects

```python
# CATE: E[Y(W=w) - Y(W=0) | X=x]
cate = cffe.effect(df[covariates])
df["estimated_elasticity"] = cate

# Segment by CATE quartile
df["elasticity_quartile"] = pd.qcut(cate, q=4, labels=["Q1", "Q2", "Q3", "Q4"])

print(df.groupby("elasticity_quartile").agg({
    "rate_change_pct": "mean",
    "churned": "mean",
    "estimated_elasticity": "mean"
}))
```

Policyholders in Q4 (most elastic) require different rate treatment than Q1 (least elastic). The CFFE estimate separates genuine price elasticity from observable risk characteristics.

## Confidence intervals via bootstrap

```python
# CFFE does not yet implement honest inference; use bootstrap for CIs
from causalfe import bootstrap_effect_ci

ci_lower, ci_upper = bootstrap_effect_ci(
    model=cffe,
    X=df[covariates],
    n_bootstrap=200,
    ci_level=0.95
)
```

## Connecting CFFE to FCA evidence requirements

Under GIPP PS21/5, pricing evidence packs require you to demonstrate that rate differentials reflect genuine risk differences, not behavioural exploitation. CFFE provides:

1. **Individual treatment effect estimates**: FCA Consumer Duty requires you to show customer-level outcomes. CFFE produces per-customer elasticity scores.
2. **Fixed effects**: The within-customer panel variation is cleanly separated from between-customer variation. You can argue to the FCA that your rate changes are responding to risk, not to individual behavioural patterns.
3. **Heterogeneity decomposition**: You can show which observed covariates predict rate sensitivity, and demonstrate that protection-characteristic-correlated segments are not systematically more elastic.

```python
# Check that rate elasticity is not correlated with protected characteristics
from scipy.stats import pearsonr

correlation, p_value = pearsonr(cate, df["deprivation_decile"])
print(f"Elasticity-deprivation correlation: {correlation:.3f} (p={p_value:.4f})")
# Target: |correlation| < 0.05 for FCA EP25/2 compliance
```

## When to use CFFE vs insurance-causal-policy

[`insurance-causal-policy`](https://github.com/burning-cost/insurance-causal-policy) implements Synthetic Difference-in-Differences for evaluating whether a rate change worked — the average effect at the cohort level. It answers: "Did the 8% rate increase on the 30-39 age band achieve the targeted loss ratio improvement?"

CFFE answers a different question: "Which customers were rate-elastic, and by how much?" Use CFFE when you need heterogeneous treatment effects for:
- Identifying which segments can absorb rate increases without churning
- Building customer-level retention predictions that account for individual rate sensitivity
- Demonstrating to the FCA that retention interventions are not exploiting specific customer segments

## Limitations

causalfe v0.1.x has some rough edges:
- The fixed effects estimator assumes balanced panels or handles missing periods via mean imputation (check documentation for your version)
- Bootstrap confidence intervals are not honest in the Wager-Athey (2018) sense; they do not account for model selection uncertainty
- Continuous treatments (rate change percentage) require careful normalisation; extreme rate changes may violate overlap

For the average treatment effect on the full renewal book, the ATE from CFFE with fixed effects should be close to the SDID estimate from insurance-causal-policy. If they differ substantially, investigate whether the panel is balanced and whether the fixed effects are absorbing too much signal.

---

`causalfe` is available on [PyPI](https://pypi.org/project/causalfe/). No wrapper library needed — the original package is well-designed for direct use.

Related: [insurance-causal-policy](https://burning-cost.github.io/2026/03/13/your-rate-change-didnt-prove-anything/) for SDID rate change evaluation, [insurance-causal](https://burning-cost.github.io/2026/03/05/your-rating-factor-might-be-confounded/) for causal price elasticity, [insurance-online](https://burning-cost.github.io/2026/03/12/insurance-online/) for GIPP-compliant price experimentation.
